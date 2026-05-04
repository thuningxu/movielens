#!/usr/bin/env python3
"""
HSTU for generative recommendation on MovieLens.

Step 2 (this file): replace the placeholder vanilla pre-norm transformer
block with the real HSTU block per Meta 2024 ("Actions Speak Louder than
Words"). HSTU = pointwise attention (SiLU, not softmax) + GLU-style gating
+ relative-position bias from log-bucketed time deltas.

Step 1 (5bf86c6): sequence-level training with per-position causal loss
(SASRec/HSTU framing). Each train example = one user's full event sequence;
loss applied at every valid position predicting engagement of the next event.
Eval still emits per-(user, candidate, ts) rows for AUC continuity vs simple_v2.

Reference:
    Zhai et al., "Actions Speak Louder than Words: Trillion-Parameter
    Sequential Transducers for Generative Recommendations." Meta, 2024.
    https://arxiv.org/abs/2402.17152

Task is unchanged from the prior attempts (legacy/, simple_v2/):
    label = 1 if rating >= 4 else 0  (with random unrated as easy negs)
    metric = val_auc on ml-25m at SEED=42
    bar    = simple_v2 locked baseline val 0.8594 / test 0.8455
"""

import logging
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

SEED = int(os.environ.get("SEED", "42"))
np.random.seed(SEED)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from prepare import load_data, evaluate

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger(__name__)

# ─── Config ─────────────────────────────────────────────────────────
DATASET = os.environ.get("DATASET", "ml-25m")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# HSTU hyperparameters (placeholders — tune once the real model lands)
# apr30 defaults are the operational-best config (val 0.8567 single-seed,
# 0.8563 2-seed mean on ml-25m at SEED=42). To reproduce earlier byte-equivalent
# baselines, override the relevant flags to OFF — see program.md for cycle history.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "64"))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "3"))     # apr30 best: 3L matches 4L AUC, 27% faster
NUM_HEADS = int(os.environ.get("NUM_HEADS", "4"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "100"))         # apr30 best: 100 events (×2 = 200 tokens with INTERLEAVE=1)
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
NUM_TIME_BUCKETS = int(os.environ.get("NUM_TIME_BUCKETS", "32"))  # log-spaced time-delta buckets
NUM_RATING_BUCKETS = 10                                 # 0.5★ → bucket 0, 5★ → bucket 9
ENGAGED_BUCKET_THRESHOLD = 7                            # bucket >= 7 ⇔ rating >= 4 (label=1)

LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "20"))    # apr30 best — earlier cycles used 5 or 15

# Cold-start content metadata flags (apr30, Idea 1). All default OFF for
# byte-equivalence with the prior baseline. When enabled, each adds a
# projection (or embedding) summed into the per-position item-side input
# AND used symmetrically when scoring a candidate, so the metadata appears
# in BOTH the sequence and the dot product. Zero-init keeps the OFF→ON
# transition smooth (initial logits unchanged at step 0; signal grows as
# the projections train).
USE_GENOME = int(os.environ.get("USE_GENOME", "1"))   # apr30 best: ON
USE_GENRE = int(os.environ.get("USE_GENRE", "1"))     # apr30 best: ON
USE_YEAR = int(os.environ.get("USE_YEAR", "1"))       # apr30 best: ON

# Popularity prior (apr30, cold_user intervention plan, option D). Default OFF
# for byte-equivalence with commit 6886442. When USE_POP_PRIOR=1 a static
# per-item normalized log-rating-count feature (computed ONCE at startup from
# train_df only — never val/test, so train-derived counts are inherently
# strictly-prior to all val/test rows by virtue of prepare.load_data's
# time-based split) is projected by a learnable Linear(1, EMBED_DIM, bias=False)
# and SUMMED into item_full_embed at every call site (sequence content tokens,
# per-position training targets, eval candidate). The projection weight is
# zero-init so step-0 ON state == OFF state byte-equivalent at the
# candidate-scoring level — same pattern as the genome/genre/year zero-init
# path. Module construction guarded so OFF-state RNG is preserved exactly.
# Motivation: cold_user (80% of val) has no item-side history, so HSTU's
# per-candidate scoring lacks a useful "popular items are more likely engaged"
# prior; an explicit popularity scalar could fill that gap.
USE_POP_PRIOR = int(os.environ.get("USE_POP_PRIOR", "0"))

# Bucketed absolute rating-timestamp embedding (variant A, may03 cold_user
# follow-up). Default OFF for byte-equivalence with commit c109f3c. When
# USE_RATING_TS=1, every CONTENT token (interleaved) or fused position gets a
# learnable embedding indexed by the absolute rating timestamp bucketed into
# NUM_TS_BUCKETS=32 monthly buckets over ml-25m's train timestamp range. The
# bucket boundaries are computed ONCE at startup from train_df only (so val/test
# are never used for binning, preserving the strict-prior guarantee). The
# rating_ts_embed module is constructed only when USE_RATING_TS=1, so OFF-state
# RNG state is byte-identical to commit c109f3c. Zero-init weight makes step-0
# ON state == OFF state at the candidate-scoring level.
#
# Motivation: cold_user (80% of val) gap to simple_v2 (~0.003) is structural —
# pop_prior was null and item_stats regressed. Adding absolute rating-ts to the
# content token alongside year_embed lets the model learn ts × year interactions
# (movie-age-at-rating, era-conditional taste, calendar-phase effects).
#
# Bucketing: clip((ts - TRAIN_TS_MIN) / SECONDS_PER_MONTH, 0, NUM_TS_BUCKETS-1).
# PAD positions have ts=0 (epoch 1970), well before TRAIN_TS_MIN, so the clip
# folds them onto bucket 0 (along with the earliest real-ts events). Pad
# positions are masked out of attention regardless, so bucket-0 collision is
# benign. Application is per-position (interleaved: content tokens only;
# fused: every position) and SUMMED into the per-position input — same
# additive pattern as rating_embed and year_embed.
USE_RATING_TS = int(os.environ.get("USE_RATING_TS", "0"))
NUM_TS_BUCKETS = 32
SECONDS_PER_MONTH = 30.44 * 86400

# Per-item rating statistics (apr30, cold_user intervention plan, variant B).
# Default OFF for byte-equivalence with commit fd96d2a. When USE_ITEM_STATS=1
# a static per-item 3-vector of normalized rating statistics (mean, std,
# frac_engaged) is computed ONCE at startup from train_df only and projected by
# a learnable Linear(3, EMBED_DIM, bias=False) into item_full_embed at every
# call site. Same time-leak guarantee as USE_POP_PRIOR: stats are derived
# entirely from train_df rows, and prepare.load_data's time-based split places
# all train timestamps strictly before val/test, so the aggregation is
# inherently strictly-prior to every val/test sample. The 3 scalars are
#   - mean_rating / 5.0       (in [0.1, 1.0]; 0.0 imputed if no train ratings)
#   - std_rating  / 2.5       (in [0, 1];   0.0 imputed if <2 train ratings)
#   - frac_engaged            (in [0, 1];   0.0 imputed if no train ratings)
# Count is intentionally EXCLUDED — USE_POP_PRIOR already projects log1p(count)
# and stacking both would be redundant with the popularity signal. Motivation:
# pop_prior alone gave +0.0001 overall / +0.0003 cold_user (sub-noise) because
# item_embed already encodes count via co-occurrence; the missing signal is
# the rating DISTRIBUTION (mean / std / frac_engaged), which simple_v2's
# i_hist_rat_mean captured directly. Projection weight is zero-init so step-0
# ON == OFF byte-equivalent at the candidate-scoring level; module guarded so
# OFF-state RNG is preserved exactly.
USE_ITEM_STATS = int(os.environ.get("USE_ITEM_STATS", "0"))

# Stabilization mechanisms (apr30, post-spike). Both default OFF (off-state
# byte-equivalent to 200bc86). Motivation: with USE_GENOME/GENRE/YEAR=1
# (M1), training spikes at epoch 5 — train_loss 0.515 → 7.21, val_auc drops
# 0.05. Reproducible across M1 and M2 (genome-only); M0 (no metadata) is
# stable. Hypothesis: zero-init projections + Adam moment cascade at the
# activation transition; gradient clipping is a direct counter, and Xavier
# init is an alternative path that avoids the cascade entirely.
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))   # apr30 best: 1.0 (catches mild spikes)
PROJ_INIT_MODE = os.environ.get("PROJ_INIT_MODE", "xavier")  # apr30 best: xavier (slightly better than zero)
if PROJ_INIT_MODE not in {"zero", "xavier"}:
    raise ValueError(f"PROJ_INIT_MODE must be 'zero' or 'xavier', got {PROJ_INIT_MODE!r}")

# MLP head on h_t (apr30, Critic Round 1 pick). Default OFF (off-state
# byte-equivalent to commit 8368ebb). When MLP_HEAD=1, h_t is projected
# through a 2-layer MLP (D → 2D → D) before the dot product against the
# candidate item embedding, allowing non-linear interactions between the
# contextualized user state and content features. The head_mlp module is
# constructed only when MLP_HEAD=1 so OFF-state RNG is preserved exactly.
# Init: standard PyTorch defaults (Kaiming-uniform on Linear weights). We
# do NOT zero-init the final layer because the MLP REPLACES h_t (no
# residual), so a zero output would collapse all logits to zero and val_auc
# to 0.5 at step 0. With Kaiming init the MLP starts as a non-trivial
# function of h_t — the model loses byte-equivalence on ON path but begins
# learning immediately. Param cost at D=64: 2*D*D + 2D + D*2D + D = 16,576.
MLP_HEAD = int(os.environ.get("MLP_HEAD", "1"))       # apr30 best: ON
MLP_HEAD_DROPOUT = float(os.environ.get("MLP_HEAD_DROPOUT", "0.1"))

# Auxiliary rating-regression head (apr30, simple_v2 port). Default OFF so the
# off-state is byte-identical to commit 2bf0713 (the aux_head Linear is NOT
# constructed when AUX_RATING_WEIGHT==0, preserving the RNG draw sequence
# consumed by every subsequent module init). When AUX_RATING_WEIGHT > 0, a
# parallel Linear(EMBED_DIM, 1) head predicts the per-position normalized
# rating bucket (rating_bucket / 9.0 ∈ [0, 1]) from the same hidden state h_t
# used for the main BCE scoring path. Combined loss in train_one_epoch is
#     total_loss = bce_loss + AUX_RATING_WEIGHT * masked_mse
# where the mask is the same as for BCE: only valid + content positions
# (interleaved) or only valid (current, next) pairs (fused). Inputs to the aux
# head are the raw encoder outputs h_t (NOT the MLP-projected variant), so the
# auxiliary regression task is a parallel structural pull on the encoder, not
# a diagnostic of the projected scoring head — this mirrors simple_v2 where
# both the main and aux heads consumed the same raw concat.
AUX_RATING_WEIGHT = float(os.environ.get("AUX_RATING_WEIGHT", "0.0"))

# Interleaving flag (apr30, post-1edb678). When 0 (default) sequence is the
# fused (item+rating) representation: each event = one position, x_t =
# item_full_embed(m_t) + rating_embed(rating_bucket_t). Byte-identical to
# the prior baseline.
#
# When 1, switch to the paper-canonical interleaved representation per Meta
# 2024 §3: each event becomes a (content, action) PAIR of tokens, so a
# user's sequence of N events materializes as 2N positions:
#     [c_0, a_0, c_1, a_1, ..., c_{N-1}, a_{N-1}]
# where c_i is item-side (movieId-derived; metadata applied via
# item_full_embed) and a_i is action-side (rating_bucket-derived; pure
# learned embedding, no metadata since metadata is per-item not
# per-rating-bucket).
#
# At training, BCE loss is applied at every CONTENT position 2i: the model
# scores h_{2i} against item_full_embed(c_i) and is supervised by
# is_engaged(rating_bucket_i). At eval, a candidate is appended as a
# content token at position 2N, the model produces h_{2N}, and we score
# dot(h_{2N}, item_full_embed(candidate)).
#
# OFF→ON is NOT byte-equivalent: a new action_embed table is constructed
# only when INTERLEAVE=1 (so OFF-state RNG state is preserved exactly),
# and the sequence layout differs entirely. Documented as such; this is
# an architecture-level switch, not a regularization knob.
#
# SEQ_LEN keeps EVENT semantics in both regimes. INTERLEAVE=1 internally
# allocates 2*SEQ_LEN tokens per sequence so a sweep at SEQ_LEN=100
# matches the compute footprint of the SEQ_LEN=200 fused baseline.
INTERLEAVE = int(os.environ.get("INTERLEAVE", "1"))   # apr30 best: ON (paper-canonical Meta 2024 §3)

# LR schedule + optimizer flags (apr30). Both default to OFF state matching
# the prior baseline byte-for-byte: LR_SCHEDULE="constant" and OPTIMIZER="adam"
# build the same plain `Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)`
# call and skip scheduler.step() entirely (scheduler is None — no extra method
# calls, no extra RNG draws).
#
# LR_SCHEDULE="cosine_warmup": linear warmup from LR/100 to LR over WARMUP_STEPS,
# then cosine decay from LR to LR*0.05 over the remaining steps. Stepped once
# per training batch.
#
# OPTIMIZER="adamw": switch to AdamW with decoupled weight decay applied ONLY
# to nn.Linear weights (matrices). Embedding weights, biases, and LayerNorm
# parameters get weight_decay=0 via param-group split — the standard "no decay
# on embeddings or norms" recipe used in transformer training.
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "constant")
assert LR_SCHEDULE in {"constant", "cosine_warmup", "cawr"}, f"unknown LR_SCHEDULE={LR_SCHEDULE}"
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "500"))
# CAWR (cosine annealing with warm restarts) HPs (may03). Only consulted when
# LR_SCHEDULE="cawr"; defaults give 5 equal cycles over 20 epochs with a 20%
# trough (eta_min = 0.2 * LR). With LR=1e-3 the trough is 2e-4 — well above
# the failure point of cosine_warmup-alone's deep decay tail.
CAWR_T_0_EPOCHS = int(os.environ.get("CAWR_T_0_EPOCHS", "4"))
CAWR_T_MULT = int(os.environ.get("CAWR_T_MULT", "1"))
CAWR_ETA_MIN_FRAC = float(os.environ.get("CAWR_ETA_MIN_FRAC", "0.2"))
OPTIMIZER = os.environ.get("OPTIMIZER", "adam")
assert OPTIMIZER in {"adam", "adamw"}, f"unknown OPTIMIZER={OPTIMIZER}"

# Mixed-precision flag (apr30). Default OFF for byte-equivalence with the prior
# baseline. When USE_BF16=1 the forward pass and loss computation in both
# train_one_epoch and evaluate_model run inside torch.amp.autocast with
# dtype=bfloat16 on CUDA; optimizer keeps fp32 master weights (PyTorch AMP
# standard — no GradScaler needed for bf16, whose dynamic range matches fp32).
# OFF state uses contextlib.nullcontext, a no-op CM that doesn't touch dtype
# or RNG, so val_auc is byte-identical to commit 765596c. Bf16 path may differ
# by <0.005 due to reduced mantissa precision in matmul accumulators; that's
# the standard accuracy/throughput tradeoff and should not affect AUC at our
# scale. Only meaningful on CUDA — on CPU/MPS the autocast is a no-op for the
# bf16 dtype, so the flag silently degrades to fp32.
USE_BF16 = int(os.environ.get("USE_BF16", "1"))       # apr30 best: ON (27% faster, no AUC cost on CUDA)

# Year embedding bucket scheme. ml-25m titles span 1874..2019; ml-100k spans
# 1922..1998. Coverage 1850..2049 = 200 buckets handles all observed datasets
# with safety margin and costs ~12 KB at D=64. year_id = clip(year - YEAR_MIN,
# 0, NUM_YEAR_BUCKETS - 1); missing-year sentinel uses bucket 0 (1850, well
# outside any real movie's release year).
YEAR_MIN = 1850
NUM_YEAR_BUCKETS = 200

# Per-row prediction dump flag (apr30, post-hoc strata analysis). Default OFF
# so the off-state is byte-equivalent — when SAVE_PREDS=0 evaluate_model only
# accumulates the `all_scores`/`all_labels` arrays it needs for AUC, no extra
# tensors built and no CSV I/O. When SAVE_PREDS=1 the final eval call also
# materializes per-row uid / mid / label / score / prefix_len arrays and
# writes them to SAVE_PREDS_PATH at the END of training. The pred buffer
# only persists for the final eval (cleared each call) so memory is bounded.
# Stratum analysis is computed offline by scripts/eval_strata.py.
SAVE_PREDS = int(os.environ.get("SAVE_PREDS", "0"))
SAVE_PREDS_PATH = os.environ.get("SAVE_PREDS_PATH", "/tmp/eval_strata.csv")


# ─── Movie metadata (cold-start content features) ───────────────────
def load_movie_metadata(movies_df: pd.DataFrame, dataset: str, num_items: int):
    """Load per-movie content features aligned to the +1-shifted movieId convention.

    Returns three numpy arrays sized (num_items + 1, ...) where row 0 is the
    PAD slot (all zeros) and rows 1..num_items are the real movies' metadata.
    This matches the embedding-table convention used by HSTU.item_embed (PAD=0,
    real movies = 1..num_items) so a single +1-shifted movieId indexes both
    item_embed and any metadata table.

    Returns:
        genome:  (num_items + 1, GENOME_DIM) float32. Per-movie tag-genome
                 relevance scores from genome-scores.csv. Movies without genome
                 data (e.g. all of ml-100k, or new movies in ml-25m) get a
                 zero row. GENOME_DIM = 1128 for ml-25m, 0 for datasets with
                 no genome file (graceful fallback: ml-100k → empty matrix).
        genre:   (num_items + 1, num_genres) float32. Multi-hot genre vector
                 per movie. num_genres is the number of unique genre tokens
                 observed in movies_df (~20 for MovieLens).
        year_id: (num_items + 1,) int64. Bucket index = clip(year - YEAR_MIN,
                 0, NUM_YEAR_BUCKETS - 1). Movies without a parseable year, and
                 the PAD slot, get bucket 0 (1850). The HSTU.year_embed table
                 is zero-initialized so year_id=0 contributes the same zero
                 vector as a missing entry.

    All datasets share the same genre/year code path. Genome falls back to an
    empty (0-width) matrix when genome-scores.csv is absent, and the model is
    careful to skip the genome path entirely when GENOME_DIM == 0 — that way
    USE_GENOME=1 on ml-100k is a benign no-op rather than a crash.
    """
    data_root = Path(__file__).resolve().parent / "data"

    # Genre multi-hot from movies_df["genres"] (pipe-separated, e.g. "Action|Comedy").
    # movies_df is the prepare.load_data() output: movieIds already mapped to
    # contiguous 0..num_items-1, and genres preserved as the original string.
    all_genres_set = set()
    for g in movies_df["genres"].dropna():
        all_genres_set.update(g.split("|"))
    all_genres = sorted(all_genres_set - {""})
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    num_genres = len(all_genres)
    genre = np.zeros((num_items + 1, max(num_genres, 1)), dtype=np.float32)
    for _, row in movies_df.iterrows():
        mid = int(row["movieId"])
        if 0 <= mid < num_items and isinstance(row["genres"], str):
            for g in row["genres"].split("|"):
                if g in genre_to_idx:
                    genre[mid + 1, genre_to_idx[g]] = 1.0

    # Year bucket from regex on title. Missing-year movies and the PAD row
    # both use bucket 0 (1850), which is outside any real release year.
    year_id = np.zeros(num_items + 1, dtype=np.int64)
    for _, row in movies_df.iterrows():
        mid = int(row["movieId"])
        if 0 <= mid < num_items:
            m = re.search(r"\((\d{4})\)", str(row.get("title", "")))
            if m:
                y = int(m.group(1)) - YEAR_MIN
                year_id[mid + 1] = max(0, min(NUM_YEAR_BUCKETS - 1, y))

    # Tag genome (1128-d per movie, ml-25m only). The genome CSV uses RAW
    # movieIds, so we re-derive prepare.py's movie_map by reading ratings.csv
    # and applying the same `unique() in time-sorted order` rule. Datasets
    # without genome data fall through to a (num_items + 1, 0) array; the
    # model checks GENOME_DIM > 0 before invoking genome_proj.
    genome_path = data_root / dataset / "genome-scores.csv"
    if genome_path.exists():
        # Recompute the prepare.py movie_map. For ml-25m the raw ratings are
        # in ratings.csv with the same column scheme as load_data uses.
        if dataset == "ml-25m":
            raw_path = data_root / dataset / "ratings.csv"
            raw = pd.read_csv(raw_path)
        elif dataset == "ml-1m":
            raw_path = data_root / dataset / "ratings.dat"
            raw = pd.read_csv(raw_path, sep="::", engine="python",
                              names=["userId", "movieId", "rating", "timestamp"])
        elif dataset == "ml-10m":
            raw_path = data_root / "ml-10M100K" / "ratings.dat"
            raw = pd.read_csv(raw_path, sep="::", engine="python",
                              names=["userId", "movieId", "rating", "timestamp"])
        else:
            raw = None
        if raw is not None:
            raw = raw.sort_values("timestamp").reset_index(drop=True)
            movie_map = {int(mid): i for i, mid in enumerate(raw["movieId"].unique())}
            gdf = pd.read_csv(genome_path)
            num_tags = int(gdf["tagId"].max())
            genome = np.zeros((num_items + 1, num_tags), dtype=np.float32)
            gdf["mapped_mid"] = gdf["movieId"].map(movie_map)
            gdf = gdf.dropna(subset=["mapped_mid"])
            gdf["mapped_mid"] = gdf["mapped_mid"].astype(int)
            have = 0
            for mid, group in gdf.groupby("mapped_mid"):
                if 0 <= mid < num_items:
                    tag_ids = group["tagId"].values.astype(int) - 1
                    genome[mid + 1, tag_ids] = group["relevance"].values.astype(np.float32)
                    have += 1
            log.info(f"  tag genome: {have}/{num_items} movies "
                     f"({100 * have / num_items:.1f}%), dim={num_tags}")
        else:
            genome = np.zeros((num_items + 1, 0), dtype=np.float32)
    else:
        # ml-100k and any future dataset without a genome file. USE_GENOME=1
        # is treated as a no-op in this case (HSTU checks genome_dim > 0).
        genome = np.zeros((num_items + 1, 0), dtype=np.float32)
        log.info(f"  no tag genome at {genome_path} — USE_GENOME will no-op")

    log.info(f"  metadata: genome_dim={genome.shape[1]}  "
             f"num_genres={num_genres}  year_buckets=[{YEAR_MIN}..{YEAR_MIN + NUM_YEAR_BUCKETS - 1}]")
    return genome, genre, year_id


# ─── Data ───────────────────────────────────────────────────────────
def build_user_sequences(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Group ratings by user, sort by timestamp, return per-user event arrays.

    Each event row is (movieId+1, rating_bucket, timestamp). Easy negatives
    (rating == 0 sentinel) are excluded from the history; only real ratings
    contribute to the sequence representation.

    movieId shift: prepare.py remaps movieIds to [0, num_items-1], so movieId
    0 is a real movie. We shift by +1 here so PAD slot = 0 in the embedding
    table and real movies occupy 1..num_items. Without this shift,
    nn.Embedding(..., padding_idx=0) silently zeroes gradients for movieId-0
    events and conflates left-pad slots with real movieId-0 occurrences.

    Note: load_data() never injects easy negatives, so the rating>0 filter is
    a defensive no-op here. It matters for any future code path that consumes
    a hybrid frame.
    """
    real = df[df["rating"] > 0].copy()
    real = real.sort_values(["userId", "timestamp"])
    real["rating_bucket"] = ((real["rating"].clip(0.5, 5.0) - 0.5) * 2).astype(np.int64)
    sequences = {}
    for uid, group in real.groupby("userId", sort=False):
        sequences[int(uid)] = np.stack(
            [group["movieId"].to_numpy(np.int64) + 1,
             group["rating_bucket"].to_numpy(np.int64),
             group["timestamp"].to_numpy(np.int64)],
            axis=1,
        )
    return sequences


def _pad_left(items: np.ndarray, ratings: np.ndarray, timestamps: np.ndarray, seq_len: int):
    """Left-pad item/rating/timestamp arrays to seq_len with zeros (most recent at end).

    Pad ts=0 means "epoch 1970" — well before any real ML rating (1995+),
    so any pad-vs-real time delta lands in the largest log bucket. That
    bucket's bias is irrelevant because the pad position is masked out
    of attention regardless; we just need the bucket index to be valid
    and not produce NaN/overflow.
    """
    n = items.shape[0]
    if n >= seq_len:
        items = items[-seq_len:]
        ratings = ratings[-seq_len:]
        timestamps = timestamps[-seq_len:]
        mask = np.ones(seq_len, dtype=np.float32)
        return (items.astype(np.int64), ratings.astype(np.int64),
                timestamps.astype(np.int64), mask)
    out_items = np.zeros(seq_len, dtype=np.int64)
    out_ratings = np.zeros(seq_len, dtype=np.int64)
    out_timestamps = np.zeros(seq_len, dtype=np.int64)
    out_mask = np.zeros(seq_len, dtype=np.float32)
    if n > 0:
        out_items[-n:] = items
        out_ratings[-n:] = ratings
        out_timestamps[-n:] = timestamps
        out_mask[-n:] = 1.0
    return out_items, out_ratings, out_timestamps, out_mask


def _build_interleaved(items: np.ndarray, ratings: np.ndarray, timestamps: np.ndarray,
                       mask: np.ndarray, seq_len_events: int):
    """Convert per-event arrays of length seq_len_events into the interleaved
    2N-token layout [c_0, a_0, c_1, a_1, ..., c_{N-1}, a_{N-1}].

    Returns six (2N,) numpy arrays:
      content_ids:  +1-shifted movieId at content positions (2i), 0 elsewhere.
      action_ids:   +1-shifted rating_bucket at action positions (2i+1), 0 elsewhere.
                    Index 0 is PAD; ratings 0.5..5.0 → buckets 0..9 → action_ids 1..10.
      ts_2n:        per-position timestamp, with c_i and a_i sharing event i's ts.
      mask_2n:      per-position validity (1.0 at real positions, 0.0 at pad).
                    A pair (c_i, a_i) is either both valid or both pad — pairs
                    are padded as units from the left.
      is_content:   bool, True at even positions (content), False at odd (action).
      content_rating_bucket: per-position UN-SHIFTED rating bucket aligned to
                             content positions (so [2i] holds rating_bucket_i).
                             Used by training to compute the engaged target.
                             Zeros at action positions and pad (harmless — those
                             positions are masked out of loss).

    The +1 shift on action_ids parallels the movieId shift: PAD index = 0,
    real rating buckets occupy 1..NUM_RATING_BUCKETS so action_embed's
    padding_idx=0 row is never gradient-updated by real events.
    """
    N = seq_len_events
    L = 2 * N
    content_ids = np.zeros(L, dtype=np.int64)
    action_ids = np.zeros(L, dtype=np.int64)
    ts_2n = np.zeros(L, dtype=np.int64)
    mask_2n = np.zeros(L, dtype=np.float32)
    content_rating_bucket = np.zeros(L, dtype=np.int64)
    # Even indices = content, odd = action.
    is_content = np.zeros(L, dtype=bool)
    is_content[0::2] = True
    # Place per-event values into the (c_i, a_i) pair.
    content_ids[0::2] = items                              # already +1-shifted
    action_ids[1::2] = ratings + 1                          # +1 shift; PAD slot = 0
    # Action_ids at content positions stay 0 (PAD), and content_ids at action
    # positions stay 0 (PAD); the embedding lookup never overlaps.
    # If the original event was pad (mask=0), force action_ids/content_ids to 0
    # so neither real-row gradients flow through PAD slots. Easy: action_ids
    # already gets +1, but if mask[i] == 0 we want action_ids[2i+1] == 0.
    pad_event = mask == 0
    if pad_event.any():
        action_ids[1::2] = np.where(pad_event, 0, action_ids[1::2])
        content_ids[0::2] = np.where(pad_event, 0, content_ids[0::2])
    ts_2n[0::2] = timestamps
    ts_2n[1::2] = timestamps                                # pair shares one ts
    mask_2n[0::2] = mask
    mask_2n[1::2] = mask                                    # pair valid/pad together
    content_rating_bucket[0::2] = ratings
    return content_ids, action_ids, ts_2n, mask_2n, is_content, content_rating_bucket


class SequenceTrainDataset(Dataset):
    """One sample = one user's full event sequence (truncated to last SEQ_LEN
    if longer). Per-position causal training: at every valid position t, the
    model predicts engagement of event[t+1]. No injected easy negatives —
    sequence training is dense per-position; that was a sample-level artifact.
    """

    def __init__(self, history: dict[int, np.ndarray], seq_len: int, min_events: int = 2):
        # Need ≥2 events per sequence: at least one (t, t+1) prediction pair.
        self.uids = sorted(uid for uid, ev in history.items() if ev.shape[0] >= min_events)
        self.history = history
        self.seq_len = seq_len

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        events = self.history[uid]
        items = events[:, 0]
        ratings = events[:, 1]
        timestamps = events[:, 2]
        items, ratings, timestamps, mask = _pad_left(items, ratings, timestamps, self.seq_len)
        out = {
            "uid": uid,
            "hist_items": items,
            "hist_ratings": ratings,
            "hist_ts": timestamps,
            "hist_mask": mask,
        }
        if INTERLEAVE:
            content_ids, action_ids, ts_2n, mask_2n, is_content, content_rb = \
                _build_interleaved(items, ratings, timestamps, mask, self.seq_len)
            out["content_ids"] = content_ids
            out["action_ids"] = action_ids
            out["ts_2n"] = ts_2n
            out["mask_2n"] = mask_2n
            out["is_content"] = is_content
            out["content_rating_bucket"] = content_rb
        return out


class EvalDataset(Dataset):
    """One sample = one (user, candidate_movie, target_ts) eval row, paired
    with the user's prefix sequence (events strictly before target_ts) sliced
    from a precomputed dynamic-history dict. Mirrors simple_v2's
    EVAL_DYNAMIC_HIST=1 mechanism.
    """

    def __init__(self, df: pd.DataFrame, history: dict[int, np.ndarray], seq_len: int):
        self.uid = df["userId"].to_numpy(np.int64)
        # +1 shift to match the item-embedding convention: PAD=0,
        # real movies = 1..num_items. See build_user_sequences() docstring.
        self.mid = df["movieId"].to_numpy(np.int64) + 1
        self.lbl = df["label"].to_numpy(np.float32)
        self.ts = df["timestamp"].to_numpy(np.int64)
        self.history = history
        self.seq_len = seq_len

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        uid = int(self.uid[idx])
        ts = int(self.ts[idx])
        events = self.history.get(uid)
        if events is None or events.shape[0] == 0:
            items = np.zeros(self.seq_len, dtype=np.int64)
            ratings = np.zeros(self.seq_len, dtype=np.int64)
            timestamps = np.zeros(self.seq_len, dtype=np.int64)
            mask = np.zeros(self.seq_len, dtype=np.float32)
        else:
            cut = np.searchsorted(events[:, 2], ts, side="left")
            window = events[max(0, cut - self.seq_len):cut]
            items, ratings, timestamps, mask = _pad_left(
                window[:, 0], window[:, 1], window[:, 2], self.seq_len,
            )
        out = {
            "uid": uid,
            "mid": int(self.mid[idx]),
            "label": float(self.lbl[idx]),
            "hist_items": items,
            "hist_ratings": ratings,
            "hist_ts": timestamps,
            "hist_mask": mask,
        }
        if INTERLEAVE:
            content_ids, action_ids, ts_2n, mask_2n, is_content, content_rb = \
                _build_interleaved(items, ratings, timestamps, mask, self.seq_len)
            # Eval-time: also pass the candidate ts so the model can append the
            # candidate as a content token at position 2N with its own ts (used
            # for the time-delta bias against the prefix).
            out["content_ids"] = content_ids
            out["action_ids"] = action_ids
            out["ts_2n"] = ts_2n
            out["mask_2n"] = mask_2n
            out["is_content"] = is_content
            out["content_rating_bucket"] = content_rb
            out["target_ts"] = ts
        return out


def collate_train(batch):
    out = {
        "uid": [b["uid"] for b in batch],
        "hist_items": torch.tensor(np.stack([b["hist_items"] for b in batch])),
        "hist_ratings": torch.tensor(np.stack([b["hist_ratings"] for b in batch])),
        "hist_ts": torch.tensor(np.stack([b["hist_ts"] for b in batch])),
        "hist_mask": torch.tensor(np.stack([b["hist_mask"] for b in batch])),
    }
    if INTERLEAVE:
        out["content_ids"] = torch.tensor(np.stack([b["content_ids"] for b in batch]))
        out["action_ids"] = torch.tensor(np.stack([b["action_ids"] for b in batch]))
        out["ts_2n"] = torch.tensor(np.stack([b["ts_2n"] for b in batch]))
        out["mask_2n"] = torch.tensor(np.stack([b["mask_2n"] for b in batch]))
        out["is_content"] = torch.tensor(np.stack([b["is_content"] for b in batch]))
        out["content_rating_bucket"] = torch.tensor(
            np.stack([b["content_rating_bucket"] for b in batch]),
        )
    return out


def collate_eval(batch):
    out = {
        "uid": [b["uid"] for b in batch],
        "mid": torch.tensor([b["mid"] for b in batch], dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "hist_items": torch.tensor(np.stack([b["hist_items"] for b in batch])),
        "hist_ratings": torch.tensor(np.stack([b["hist_ratings"] for b in batch])),
        "hist_ts": torch.tensor(np.stack([b["hist_ts"] for b in batch])),
        "hist_mask": torch.tensor(np.stack([b["hist_mask"] for b in batch])),
    }
    if INTERLEAVE:
        out["content_ids"] = torch.tensor(np.stack([b["content_ids"] for b in batch]))
        out["action_ids"] = torch.tensor(np.stack([b["action_ids"] for b in batch]))
        out["ts_2n"] = torch.tensor(np.stack([b["ts_2n"] for b in batch]))
        out["mask_2n"] = torch.tensor(np.stack([b["mask_2n"] for b in batch]))
        out["is_content"] = torch.tensor(np.stack([b["is_content"] for b in batch]))
        out["target_ts"] = torch.tensor([b["target_ts"] for b in batch], dtype=torch.long)
    return out


# ─── Model ──────────────────────────────────────────────────────────
def time_delta_buckets(timestamps: torch.Tensor, num_buckets: int = NUM_TIME_BUCKETS) -> torch.Tensor:
    """Convert pairwise time deltas to log-spaced bucket indices.

    Args:
        timestamps: (B, L) Unix-second timestamps (int64).
        num_buckets: total buckets. Bucket 0 is "Δt = 0" (tied/same instant).
            Buckets 1..(num_buckets-2) are log-spaced from 1s to ~1 year.
            Bucket (num_buckets-1) is the ">1 year" overflow bucket.

    Returns:
        (B, L, L) int64 bucket indices, where entry [b, i, j] is the bucket
        for Δt = ts[b, i] - ts[b, j]. We bucket |Δt|; the causal mask
        guarantees only entries with i >= j contribute to attention, but
        keeping it symmetric simplifies the bias lookup at minimal cost.

    The scheme is chosen for ml-25m (timestamps in [1995, 2019], so the
    largest real Δt is ~24 years ≈ 7.6e8 s):
        bucket 0  → Δt = 0                     (tied/same instant)
        bucket k  → Δt in [2^(k-1), 2^k) s     for k in 1..num_buckets-2
        bucket n-1→ Δt >= 2^(num_buckets-2) s  (overflow)
    For num_buckets=32 the max-but-one bucket covers Δt in [2^29, 2^30)
    seconds (~17 to ~34 years), which exceeds any real ml-25m delta — so
    only pad-vs-real entries (pad ts=0, real ts ~1.5e9, log2≈30.5) reach
    the overflow bucket, and those positions are masked out of attention.
    Implementation: floor(log2(Δt)) clamped, then +1 to leave bucket 0
    for Δt == 0.
    """
    delta = (timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)).abs()  # (B, L, L)
    # Δt == 0 → bucket 0; Δt > 0 → 1 + floor(log2(Δt)) clamped to [1, n-1].
    log_floor = torch.zeros_like(delta)
    nonzero = delta > 0
    if nonzero.any():
        log_floor[nonzero] = torch.floor(torch.log2(delta[nonzero].float())).long()
    bucket = torch.where(nonzero, log_floor + 1, torch.zeros_like(log_floor))
    bucket = bucket.clamp(0, num_buckets - 1)
    return bucket


class HSTUBlock(nn.Module):
    """One HSTU block per Meta 2024 §3.

    Forward:
      1. h = LayerNorm(x); split SiLU(h W_uvqk) into U, V, Q, K of shape (B, L, D).
      2. Pointwise attention (NOT softmax):
           A = SiLU(Q · K^T / sqrt(D_h) + rel_bias)            # (B, H, L, L)
         where rel_bias is from log-bucketed pairwise time deltas. The
         causal+pad mask is applied as a multiplicative zero AFTER SiLU
         (so pad and future positions contribute exactly 0, no NaN risk
         from passing -inf through SiLU).
      3. AV = A @ V; gated = LayerNorm(AV) ⊙ U.
      4. out = gated W_o; return x + out.

    No separate FFN — the GLU + pointwise attention plays both roles.

    Deviations from the paper noted for the Validator:
      - Bias table: per-block, learnable, shape (num_buckets, num_heads).
        Init zeros so the block reduces to no-bias attention at step 0.
        Paper does per-block; per-head adds expressivity at trivial cost
        (32 × 4 = 128 scalars per block).
      - Mask handling: post-SiLU multiplicative zero rather than pre-SiLU
        -inf addition, to avoid NaN from SiLU(-inf). Mathematically the
        same outcome (masked positions contribute 0) but numerically safer.
      - LayerNorm on AV before gating: paper uses a norm here; we use
        nn.LayerNorm. (Some HSTU codebases use RMSNorm; we stick with LN
        for parity with the rest of the stack and PyTorch primitives.)
    """

    def __init__(self, dim: int, num_heads: int, num_time_buckets: int, dropout: float):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_in = nn.LayerNorm(dim)
        self.uvqk = nn.Linear(dim, 4 * dim)
        self.norm_out = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        # Per-head learnable bias indexed by log-bucketed pairwise Δt.
        # Init zeros so the block starts with no positional bias and learns
        # to differentiate buckets from data.
        self.rel_bias_embed = nn.Embedding(num_time_buckets, num_heads)
        nn.init.zeros_(self.rel_bias_embed.weight)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor,
                causal_bool: torch.Tensor, time_buckets: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:            (B, L, D)
          valid_mask:   (B, L) float, 1 for real positions, 0 for pad.
          causal_bool:  (L, L) bool, True at allowed (i >= j) positions.
          time_buckets: (B, L, L) int64 bucket indices for the relative bias.

        Returns:
          (B, L, D)
        """
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        h = self.norm_in(x)
        u, v, q, k = torch.chunk(torch.nn.functional.silu(self.uvqk(h)), 4, dim=-1)
        # Reshape Q, K, V to per-head: (B, L, D) → (B, H, L, Dh)
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        # Raw scores: (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)

        # Add relative-position bias. Embedding lookup gives (B, L, L, H);
        # permute to (B, H, L, L).
        bias = self.rel_bias_embed(time_buckets).permute(0, 3, 1, 2)
        scores = scores + bias

        # Pointwise activation (NOT softmax — HSTU's signature).
        attn = torch.nn.functional.silu(scores)

        # Apply causal + pad mask as a multiplicative zero. Pad-key positions
        # zeroed via valid_mask broadcast over query dim; future positions
        # zeroed via causal_bool. SiLU(0) = 0, so no NaN risk.
        # combined: (B, 1, L, L) — 1 where (j real) AND (i >= j), else 0.
        key_valid = valid_mask.view(B, 1, 1, L)               # (B, 1, 1, L)
        causal = causal_bool.view(1, 1, L, L)                 # (1, 1, L, L)
        keep = key_valid * causal.to(valid_mask.dtype)        # (B, 1, L, L)
        attn = attn * keep

        attn = self.dropout(attn)

        # Pointwise output: A @ V → (B, H, L, Dh) → (B, L, D)
        av = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        gated = self.norm_out(av) * u                          # GLU-style gate
        out = self.proj(gated)
        return x + out


class HSTU(nn.Module):
    """HSTU model per Meta 2024: causal stack of HSTU blocks over (item +
    rating) embeddings with SASRec-style dot-product scoring against a
    candidate item embedding.

    Dataset additions (vs. step 1 placeholder): hist_ts feeds the relative
    position bias via log-bucketed pairwise time deltas. The bucketing is
    computed once per batch in `encode()` (B, L, L int64) and shared across
    all blocks — each block has its own per-head bias table indexed by these
    buckets.

    Cold-start content metadata (apr30, Idea 1, opt-in via USE_GENOME /
    USE_GENRE / USE_YEAR / USE_POP_PRIOR / USE_ITEM_STATS flags). When
    enabled, each per-position item-side embedding becomes
        item_full_embed(m) = item_embed(m)
                           + (USE_GENOME     ? genome_proj(genome[m])             : 0)
                           + (USE_GENRE      ? genre_proj(genre[m])               : 0)
                           + (USE_YEAR       ? year_embed(year_id[m])             : 0)
                           + (USE_POP_PRIOR  ? pop_proj(item_pop_feature[m])      : 0)
                           + (USE_ITEM_STATS ? item_stats_proj(item_stats[m])     : 0)
    and this *same* construction is used both at sequence input AND at
    candidate scoring (symmetric path). Projection weights are zero-init
    (xavier-init when PROJ_INIT_MODE=xavier for the genome/genre/year three;
    pop_proj and item_stats_proj are always zero-init), so OFF→ON keeps step-0
    logits identical to OFF and the metadata signal grows monotonically as the
    projections train.

    Bucketed absolute rating-timestamp (may03, variant A, opt-in via
    USE_RATING_TS=1). When enabled, an additional rating_ts_embed(ts_bucket)
    is summed into the per-position input — at content tokens only in
    INTERLEAVE=1, at every position in INTERLEAVE=0. Application happens in
    encode / encode_interleaved / encode_interleaved_with_candidate, NOT in
    item_full_embed: ts is per-event, not per-item, so the colocation has to
    be at the sequence-construction level. Zero-init weight makes step-0 ON
    state == OFF state at the candidate-scoring level.

    OFF-state byte-equivalence: when all six flags are 0, item_full_embed and
    the encode paths skip every conditional branch and behave identically to
    the prior baseline. The genome/genre/year tensors are allocated regardless
    (for codepath simplicity); the popularity, item-stats, and rating-ts
    modules are constructed only when their respective flag is 1 so OFF-state
    memory and RNG draws are unchanged.
    """

    def __init__(self, num_items: int, num_rating_buckets: int,
                 genome: torch.Tensor, genre: torch.Tensor, year_id: torch.Tensor,
                 item_pop: torch.Tensor, item_stats: torch.Tensor,
                 train_ts_min: int):
        super().__init__()
        # Index 0 is PAD; real movies occupy 1..num_items. Callers
        # (build_user_sequences, EvalDataset) shift movieIds by +1 so the
        # left-pad slot (also 0) and real movieId-0 do not collide.
        self.item_embed = nn.Embedding(num_items + 1, EMBED_DIM, padding_idx=0)
        self.rating_embed = nn.Embedding(num_rating_buckets, EMBED_DIM)
        self.blocks = nn.ModuleList([
            HSTUBlock(EMBED_DIM, NUM_HEADS, NUM_TIME_BUCKETS, DROPOUT)
            for _ in range(NUM_LAYERS)
        ])
        # No concat-head here: scoring is dot(h_t, item_embed(candidate)).
        # SASRec-style sharing of item_embed between input and output is
        # standard for sequence recommenders and halves head params.
        # TODO: action-type embedding (binary engaged vs implicit) once we add easy negs to history

        # INTERLEAVE=1: separate action_embed table for the action tokens
        # interleaved between content tokens. Constructed ONLY when
        # INTERLEAVE=1 so the OFF-state RNG state is preserved exactly
        # (PyTorch consumes RNG draws on Embedding.__init__ for the default
        # normal init; building this unconditionally would shift every
        # subsequent random tensor — including any module built after it).
        # Index 0 is PAD; rating buckets 0..NUM_RATING_BUCKETS-1 occupy
        # indices 1..NUM_RATING_BUCKETS, parallel to the movieId +1 shift.
        self.use_interleave = bool(INTERLEAVE)
        if self.use_interleave:
            self.action_embed = nn.Embedding(num_rating_buckets + 1, EMBED_DIM,
                                             padding_idx=0)

        # Cold-start content metadata buffers and projections. Buffers are
        # registered (not parameters) — only the projections train.
        self.register_buffer("genome_table", genome, persistent=False)
        self.register_buffer("genre_table", genre, persistent=False)
        self.register_buffer("year_id_table", year_id, persistent=False)
        self.genome_dim = int(genome.shape[1])
        self.genre_dim = int(genre.shape[1])
        # USE_GENOME on a dataset without genome data (e.g. ml-100k) is a
        # silent no-op: genome_dim==0 → genome_proj has no input → skip.
        self.use_genome = bool(USE_GENOME) and self.genome_dim > 0
        self.use_genre = bool(USE_GENRE) and self.genre_dim > 0
        self.use_year = bool(USE_YEAR)
        # PROJ_INIT_MODE controls whether the metadata projections start at
        # zero (OFF→ON byte-equivalent at step 0; default) or Xavier-uniform
        # (non-trivial step-0 metadata signal; tests the "zero-init causes
        # Adam moment cascade at activation transition" spike hypothesis).
        def _init_proj(weight: torch.Tensor) -> None:
            if PROJ_INIT_MODE == "xavier":
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.zeros_(weight)
        if self.use_genome:
            self.genome_proj = nn.Linear(self.genome_dim, EMBED_DIM, bias=False)
            _init_proj(self.genome_proj.weight)
        if self.use_genre:
            self.genre_proj = nn.Linear(self.genre_dim, EMBED_DIM, bias=False)
            _init_proj(self.genre_proj.weight)
        if self.use_year:
            self.year_embed = nn.Embedding(NUM_YEAR_BUCKETS, EMBED_DIM)
            _init_proj(self.year_embed.weight)

        # MLP head on h_t. Constructed ONLY when MLP_HEAD=1 so OFF-state RNG
        # state is byte-identical to commit 8368ebb (the dropout, dense, and
        # init RNG draws would all shift if we built this module unconditionally).
        # Default PyTorch init (Kaiming-uniform) is intentional: see MLP_HEAD
        # config-comment for why zero-init would be wrong here.
        self.use_mlp_head = bool(MLP_HEAD)
        if self.use_mlp_head:
            self.head_mlp = nn.Sequential(
                nn.Linear(EMBED_DIM, 2 * EMBED_DIM),
                nn.GELU(),
                nn.Dropout(MLP_HEAD_DROPOUT),
                nn.Linear(2 * EMBED_DIM, EMBED_DIM),
            )

        # Auxiliary rating-regression head. Constructed LAST so OFF-state RNG
        # state is preserved across every prior module init (no extra RNG draws
        # when AUX_RATING_WEIGHT==0). Default PyTorch init (Kaiming-uniform on
        # the weight, zero bias) — the aux head is a parallel branch and does
        # not feed back into the main scoring path, so non-zero init is fine
        # (it converges to predict the rating from h_t without affecting the
        # main BCE logit at step 0).
        self.use_aux_rating = AUX_RATING_WEIGHT > 0
        if self.use_aux_rating:
            self.aux_head = nn.Linear(EMBED_DIM, 1)

        # Popularity prior (USE_POP_PRIOR=1, apr30 cold_user intervention plan
        # option D). Constructed AFTER all prior conditional modules so the
        # OFF-state RNG state is byte-identical to commit 6886442 — when the
        # flag is 0, neither the buffer nor the projection exists, no RNG is
        # consumed for this feature, and item_full_embed never references it.
        # Buffer is not persistent (rebuilt each run from train_df by main()).
        # Zero-init projection mirrors the genome/genre/year zero-init pattern
        # so step-0 ON state == OFF state byte-equivalent at the
        # candidate-scoring level (the popularity contribution is exactly 0
        # before the projection trains).
        self.use_pop_prior = bool(USE_POP_PRIOR)
        if self.use_pop_prior:
            self.register_buffer("item_pop_feature", item_pop, persistent=False)
            self.pop_proj = nn.Linear(1, EMBED_DIM, bias=False)
            nn.init.zeros_(self.pop_proj.weight)

        # Per-item rating statistics (USE_ITEM_STATS=1, apr30 cold_user
        # intervention plan variant B). Constructed AFTER pop_proj so the
        # OFF-state RNG state is byte-identical to commit fd96d2a — when the
        # flag is 0, neither the buffer nor the projection exists, no RNG is
        # consumed for this feature, and item_full_embed never references it.
        # Buffer is non-persistent (rebuilt each run from train_df by main()).
        # Zero-init mirrors pop_proj: step-0 ON == OFF byte-equivalent at the
        # candidate-scoring level. The 3 input dims are the (mean_norm,
        # std_norm, frac_engaged) per-item stats computed in main() from
        # train_df only — see USE_ITEM_STATS config-comment for the time-leak
        # guarantee and normalization rationale.
        self.use_item_stats = bool(USE_ITEM_STATS)
        if self.use_item_stats:
            self.register_buffer("item_stats_table", item_stats, persistent=False)
            self.item_stats_proj = nn.Linear(3, EMBED_DIM, bias=False)
            nn.init.zeros_(self.item_stats_proj.weight)

        # Bucketed absolute rating-timestamp embedding (USE_RATING_TS=1, may03
        # cold_user variant A). Constructed AFTER item_stats so OFF-state RNG
        # state is byte-identical to commit c109f3c — when the flag is 0,
        # neither the embedding nor the bucketing constants exist as module
        # state, no RNG is consumed for this feature, and the encode paths
        # never branch into the rating-ts code. Zero-init weight makes step-0
        # ON state == OFF state at the candidate-scoring level (the rating-ts
        # contribution is exactly 0 before the embedding trains). The
        # train_ts_min / seconds_per_month buffers are registered (not parameters)
        # so they ride along to the model device on .to(DEVICE) and don't get
        # wd-decayed by AdamW. Applies to CONTENT tokens only in INTERLEAVE=1
        # (action tokens get pure action_embed) and to every position in
        # INTERLEAVE=0 (fused mode, where each position is one event).
        self.use_rating_ts = bool(USE_RATING_TS)
        if self.use_rating_ts:
            self.rating_ts_embed = nn.Embedding(NUM_TS_BUCKETS, EMBED_DIM)
            nn.init.zeros_(self.rating_ts_embed.weight)
            self.register_buffer(
                "train_ts_min",
                torch.tensor(int(train_ts_min), dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "seconds_per_month",
                torch.tensor(float(SECONDS_PER_MONTH), dtype=torch.float32),
                persistent=False,
            )

    def _ts_bucket(self, ts: torch.Tensor) -> torch.Tensor:
        """Bucket absolute Unix-second timestamps into NUM_TS_BUCKETS monthly
        buckets over ml-25m's train range.

        Args:
            ts: (..., ) int64 Unix-second timestamps. PAD positions have ts=0
                (epoch 1970), well before TRAIN_TS_MIN, so they clip to bucket 0.

        Returns:
            (..., ) int64 bucket indices in [0, NUM_TS_BUCKETS-1].
        """
        # Subtract train_ts_min in int64, then divide by seconds_per_month in
        # fp32 (range fits well within fp32 precision: ~24 years ≈ 7.6e8 s,
        # / SECONDS_PER_MONTH ≈ 290 — far from fp32 mantissa limits).
        delta = (ts - self.train_ts_min).float()
        bucket = (delta / self.seconds_per_month).long()
        return bucket.clamp(0, NUM_TS_BUCKETS - 1)

    def item_full_embed(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Symmetric per-position item-side embedding used for both sequence
        input AND candidate scoring. Same construction in both spots is the
        whole point: the candidate at score time gets the same metadata-fused
        embedding that its observations contributed at training time.

        item_ids is the +1-shifted movieId convention (PAD=0, real=1..num_items).
        Output shape matches item_embed(item_ids), i.e. (..., EMBED_DIM).
        """
        e = self.item_embed(item_ids)
        if self.use_genome:
            e = e + self.genome_proj(self.genome_table[item_ids])
        if self.use_genre:
            e = e + self.genre_proj(self.genre_table[item_ids])
        if self.use_year:
            e = e + self.year_embed(self.year_id_table[item_ids])
        if self.use_pop_prior:
            # item_pop_feature: (num_items+1,) ∈ [0,1]. Gather to (..., 1) and
            # project to (..., EMBED_DIM). Symmetric: applies at sequence
            # content tokens, training-time per-position scoring targets, and
            # eval candidate — all three call sites pass through this method.
            e = e + self.pop_proj(self.item_pop_feature[item_ids].unsqueeze(-1))
        if self.use_item_stats:
            # item_stats_table: (num_items+1, 3) ∈ [0,1]. Gather to (..., 3)
            # and project to (..., EMBED_DIM). Same symmetric pattern as
            # pop_prior — applies at every call site that produces an
            # item-side embedding.
            e = e + self.item_stats_proj(self.item_stats_table[item_ids])
        return e

    def encode(self, hist_items: torch.Tensor, hist_ratings: torch.Tensor,
               hist_ts: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        """Run the causal stack and return per-position hidden states (B, T, D)."""
        B, T = hist_items.shape
        h = self.item_full_embed(hist_items) + self.rating_embed(hist_ratings)
        if self.use_rating_ts:
            # Fused mode: each position is one event with item+rating+ts. Add
            # the bucketed-ts embedding to every position (parallel to rating_embed
            # — same additive role at the per-event level). PAD positions have
            # ts=0 → clipped to bucket 0; pad-position contribution is masked
            # out of attention regardless via hist_mask.
            h = h + self.rating_ts_embed(self._ts_bucket(hist_ts))
        # Causal mask as a bool over (T, T): True where i >= j (allowed).
        causal_bool = torch.tril(torch.ones(T, T, dtype=torch.bool, device=h.device))
        # Pairwise log-bucketed time deltas, computed once and reused across blocks.
        # (B, L, L) int64 — ~ B·L·L·8 bytes; for B=256, L=200 that's ~80 MB.
        # Acceptable for ml-25m at our default batch.
        time_buckets = time_delta_buckets(hist_ts, NUM_TIME_BUCKETS)
        for block in self.blocks:
            h = block(h, hist_mask, causal_bool, time_buckets)
        return h

    def encode_interleaved(self, content_ids: torch.Tensor, action_ids: torch.Tensor,
                           ts_2n: torch.Tensor, mask_2n: torch.Tensor,
                           is_content: torch.Tensor) -> torch.Tensor:
        """Run the causal stack on the interleaved 2N-token sequence.

        Each event materializes as (c_i, a_i): c_i is the content token (item)
        at even position 2i, a_i is the action token (rating bucket) at odd
        position 2i+1. Per Meta 2024 §3, this lets the model produce a hidden
        state at every CONTENT position whose causal context is exactly the
        prior pairs — i.e. h_{2i} sees [c_0, a_0, ..., c_{i-1}, a_{i-1}, c_i]
        and predicts a_i.

        Args:
          content_ids: (B, 2N) int64. Even slots: +1-shifted movieId. Odd slots: 0.
          action_ids:  (B, 2N) int64. Even slots: 0. Odd slots: +1-shifted bucket.
          ts_2n:       (B, 2N) int64 timestamps; c_i and a_i share event i's ts.
          mask_2n:     (B, 2N) float, 1.0 at valid positions, 0.0 at PAD.
                       Pairs (c_i, a_i) are valid/pad together.
          is_content:  (B, 2N) bool, True at even positions, False at odd.

        Returns:
          h: (B, 2N, D) per-position hidden states.

        Embedding selection: at every position emit exactly one type of
        embedding via torch.where on the is_content mask. Both lookups
        operate on the full (B, 2N) tensor (with PAD=0 at the
        non-applicable positions, which the padding_idx=0 rows zero out)
        but only one is selected per position. Cleaner and more vectorized
        than scatter; the cost is one extra (B, 2N, D) tensor materialized
        and discarded — negligible at our scale.

        Metadata only applies to CONTENT tokens (item_full_embed): metadata
        is per-movie, not per-rating-bucket. Action tokens get pure
        action_embed (no metadata path).
        """
        B, L = content_ids.shape
        # Content embedding (with metadata) at every position; selected at
        # even positions only. PAD (id=0) → padding_idx zeros — no
        # contribution at action positions where content_ids = 0.
        content_emb = self.item_full_embed(content_ids)        # (B, 2N, D)
        if self.use_rating_ts:
            # Interleaved mode: rating-ts embedding lives in the CONTENT token
            # (colocated with year_embed via item_full_embed), so the model
            # can learn ts × year interactions. Action tokens stay pure
            # action_embed — adding to content_emb pre-where ensures no
            # contribution leaks into action positions. ts_2n shares the same
            # value at c_i and a_i (per _build_interleaved); we use ts_2n
            # directly because the where-select picks content_emb only at
            # even positions where this contribution is wanted.
            content_emb = content_emb + self.rating_ts_embed(self._ts_bucket(ts_2n))
        action_emb = self.action_embed(action_ids)             # (B, 2N, D)
        # torch.where picks content_emb at even (is_content=True) positions
        # and action_emb at odd (is_content=False) positions. Broadcast the
        # mask over the embedding-dim axis.
        x = torch.where(is_content.unsqueeze(-1), content_emb, action_emb)

        # Causal triangular mask over 2N positions. h_{2i} (content c_i) sees
        # positions 0..2i = [c_0, a_0, ..., c_{i-1}, a_{i-1}, c_i] but NOT
        # a_i — exactly the paper-canonical setup. Standard `i >= j` lower
        # triangle achieves this because position 2i is index 2i and a_i
        # lives at 2i+1 > 2i, so it's always above the diagonal.
        causal_bool = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        time_buckets = time_delta_buckets(ts_2n, NUM_TIME_BUCKETS)
        for block in self.blocks:
            x = block(x, mask_2n, causal_bool, time_buckets)
        return x

    def encode_interleaved_with_candidate(self, content_ids: torch.Tensor,
                                          action_ids: torch.Tensor,
                                          ts_2n: torch.Tensor, mask_2n: torch.Tensor,
                                          is_content: torch.Tensor,
                                          candidate: torch.Tensor,
                                          target_ts: torch.Tensor) -> torch.Tensor:
        """Eval-time interleaved encode: append the candidate as a content token
        at position 2N (the new last position, after all prior 2N events) and
        return the hidden state h at that position.

        The total sequence becomes 2N+1 tokens. The candidate is placed at
        position 2N as a content token (no paired action) so the model
        produces h_{2N} after attending over the user's full prior history —
        analogous to evaluating "given the user's history of paired events,
        what's the score for engaging with this candidate?"

        Returns:
          h_last: (B, D) — the hidden state at the appended candidate position.
        """
        B, L = content_ids.shape
        L1 = L + 1
        device = content_ids.device
        # Extend each tensor by one position. The new slot is content (is_content=True),
        # mask=1, content_id=candidate, action_id=0, ts=target_ts.
        new_content = torch.cat([content_ids, candidate.unsqueeze(1)], dim=1)            # (B, 2N+1)
        new_action = torch.cat([action_ids,
                                torch.zeros(B, 1, dtype=action_ids.dtype, device=device)],
                               dim=1)
        new_ts = torch.cat([ts_2n, target_ts.unsqueeze(1)], dim=1)
        new_mask = torch.cat([mask_2n,
                              torch.ones(B, 1, dtype=mask_2n.dtype, device=device)], dim=1)
        new_is_content = torch.cat([is_content,
                                    torch.ones(B, 1, dtype=torch.bool, device=device)],
                                   dim=1)

        content_emb = self.item_full_embed(new_content)        # (B, 2N+1, D)
        if self.use_rating_ts:
            # Eval-time symmetric application: the appended candidate at
            # position 2N is a CONTENT token whose ts is the eval row's
            # target_ts (already concatenated into new_ts above). Bucketing
            # uses the same train-derived constants as training, so the
            # candidate sees the same ts vocabulary. The candidate's
            # rating_ts_embed contribution lands in content_emb[:, -1, :]
            # before torch.where selects it (is_content=True at -1).
            content_emb = content_emb + self.rating_ts_embed(self._ts_bucket(new_ts))
        action_emb = self.action_embed(new_action)             # (B, 2N+1, D)
        x = torch.where(new_is_content.unsqueeze(-1), content_emb, action_emb)

        causal_bool = torch.tril(torch.ones(L1, L1, dtype=torch.bool, device=device))
        time_buckets = time_delta_buckets(new_ts, NUM_TIME_BUCKETS)
        for block in self.blocks:
            x = block(x, new_mask, causal_bool, time_buckets)
        return x[:, -1, :]                                    # (B, D)

    def _project_head(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the MLP head to h, if enabled. OFF-state: pass-through.

        When MLP_HEAD=1, replaces h with head_mlp(h) before the dot product
        against the candidate item embedding. The MLP shape (D → 2D → D)
        preserves the contract that scoring is dot(h_proj, item_full_embed),
        so the rest of the scoring path is untouched. Works on any shape
        ending in EMBED_DIM (handles both per-position (B,T,D) and last-token
        (B,D) tensors via Sequential's broadcasting over leading dims).
        """
        if self.use_mlp_head:
            return self.head_mlp(h)
        return h

    def score_per_position(self, h: torch.Tensor, target_items: torch.Tensor) -> torch.Tensor:
        """Dot-product scoring at every position. h:(B,T,D), target_items:(B,T)."""
        h = self._project_head(h)                             # (B, T, D)
        target_e = self.item_full_embed(target_items)         # (B, T, D)
        return (h * target_e).sum(dim=-1)                     # (B, T)

    def score_eval(self, h: torch.Tensor, hist_mask: torch.Tensor,
                   candidate: torch.Tensor) -> torch.Tensor:
        """Score the LAST valid position's hidden state against a candidate.

        With LEFT-padding, real events always occupy the rightmost positions
        [seq_len-n, seq_len-1], so the last valid position is seq_len-1 for
        any warm user. Empty-history rows are also indexed at seq_len-1 — the
        hidden state there is computed from all-PAD inputs and the dot product
        with the candidate embedding gives the model's cold-user prior.
        """
        last_h = h[:, -1, :]                                  # (B, D)
        last_h = self._project_head(last_h)                   # (B, D)
        cand_e = self.item_full_embed(candidate)              # (B, D)
        return (last_h * cand_e).sum(dim=-1)                  # (B,)


# ─── Train + eval ───────────────────────────────────────────────────
def build_optimizer(model, lr, weight_decay, optimizer_name):
    """Build the optimizer per OPTIMIZER flag.

    OFF state ("adam"): plain `torch.optim.Adam(model.parameters(), ...)` with
    the existing call signature — byte-identical to the prior baseline.

    ON state ("adamw"): AdamW with param-group split. Weight decay applies
    ONLY to weight matrices of nn.Linear (and similar 2-D+ tensors). Embedding
    weights, biases, and LayerNorm parameters get weight_decay=0 via the
    "no decay" group. Partition rule: a parameter goes to no_decay if any of
        - p.ndim < 2 (covers bias vectors, LayerNorm gain/bias which are 1-D),
        - "embed" in name.lower() (catches item_embed, rating_embed, year_embed,
          rel_bias which is technically an Embedding too),
        - "norm" in name.lower() (catches LayerNorm modules named norm_in/norm_out),
        - name endswith ".bias".
    Everything else (Linear weights, the projection matrices uvqk/proj/genome_proj/
    genre_proj, and the head_mlp Linears) goes to the decay group.
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        decay_params = []
        no_decay_params = []
        decay_names = []
        no_decay_names = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (p.ndim < 2
                    or "embed" in name.lower()
                    or "norm" in name.lower()
                    or name.endswith(".bias")):
                no_decay_params.append(p)
                no_decay_names.append(name)
            else:
                decay_params.append(p)
                decay_names.append(name)
        n_decay = len(decay_params)
        n_no_decay = len(no_decay_params)
        n_decay_params = sum(p.numel() for p in decay_params)
        n_no_decay_params = sum(p.numel() for p in no_decay_params)
        log.info(f"  AdamW param groups: decay={n_decay} ({n_decay_params:,} params), "
                 f"no_decay={n_no_decay} ({n_no_decay_params:,} params)")
        return torch.optim.AdamW([
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=lr)
    else:
        raise ValueError(f"unknown OPTIMIZER={optimizer_name}")


def build_scheduler(optimizer, lr, schedule_name, warmup_steps, total_steps):
    """Build the LR scheduler per LR_SCHEDULE flag.

    OFF state ("constant"): returns None — no scheduler.step() calls in the
    training loop, no extra method calls, byte-identical to the prior baseline.

    ON state ("cosine_warmup"): SequentialLR composed of
      - LinearLR warmup from LR*0.01 to LR over the first WARMUP_STEPS steps
        (start_factor=0.01 maps to "LR/100" peak ratio per spec),
      - CosineAnnealingLR decay from LR to LR*0.05 over the remaining steps
        (eta_min=LR*0.05 — 5% floor, not 0, to keep some learning rate at the
        tail rather than freezing the model at the very end).
    """
    if schedule_name == "constant":
        return None
    elif schedule_name == "cosine_warmup":
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        cosine_steps = max(1, total_steps - warmup_steps)
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=lr * 0.05),
            ],
            milestones=[warmup_steps],
        )
    elif schedule_name == "cawr":
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        steps_per_epoch = total_steps // MAX_EPOCHS
        T_0 = CAWR_T_0_EPOCHS * steps_per_epoch
        T_mult = CAWR_T_MULT
        eta_min = lr * CAWR_ETA_MIN_FRAC
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    else:
        raise ValueError(f"unknown LR_SCHEDULE={schedule_name}")


def train_one_epoch(model, loader, optimizer, scheduler=None):
    """Sequence-level training: per-position causal BCE on next-event engagement.

    When GRAD_CLIP > 0, clip global grad norm between backward and step. The
    pre-clip norm is averaged across batches and returned for diagnostic
    logging — useful for confirming the clip threshold is actually firing
    (if avg pre-clip norm << GRAD_CLIP, the clip is a no-op; if avg >> GRAD_CLIP,
    most steps are clipped). When GRAD_CLIP == 0.0, no clip call is made and
    grad-norm is not measured (preserving byte-equivalence with the prior
    baseline; avg_grad_norm returns as None in that case).

    INTERLEAVE=1 path: per-position BCE at CONTENT positions only. At
    position 2i, the model scores h_{2i} against item_full_embed(c_i) (the
    same item; this is the paper-canonical "content token's hidden state
    encodes the user's response to seeing this item"), and the target is
    is_engaged(rating_bucket_i) computed from content_rating_bucket. Action
    positions get loss_mask = 0.

    AUX_RATING_WEIGHT > 0: per-position MSE on (rating_bucket / 9.0) ∈ [0, 1]
    from the same h_t feeding the main scoring path. Mask is identical to the
    BCE mask (loss_mask interleaved / pair_mask fused), so each position that
    contributes to BCE also contributes to the auxiliary regression target.
    For interleaved, the aux head runs at every (B, 2N, D) position but the
    mask isolates the content slots; for fused, it runs at h_in = h[:, :-1]
    and predicts the NEXT event's rating bucket (parallel to the BCE target,
    which is engagement of the next event). The aux Linear is constructed
    only when AUX_RATING_WEIGHT > 0; OFF-state code path skips the aux
    branch entirely and is byte-identical to the prior baseline.
    """
    model.train()
    total_loss = 0.0
    total_positions = 0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    aux_loss_sum = 0.0
    aux_loss_count = 0
    bce = nn.BCEWithLogitsLoss(reduction="none")
    use_aux = model.use_aux_rating
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (USE_BF16 and DEVICE == "cuda") else nullcontext()
    )
    for batch in loader:
        if INTERLEAVE:
            content_ids = batch["content_ids"].to(DEVICE)        # (B, 2N)
            action_ids = batch["action_ids"].to(DEVICE)          # (B, 2N)
            ts_2n = batch["ts_2n"].to(DEVICE)                    # (B, 2N)
            mask_2n = batch["mask_2n"].to(DEVICE)                # (B, 2N)
            is_content = batch["is_content"].to(DEVICE)          # (B, 2N) bool
            content_rb = batch["content_rating_bucket"].to(DEVICE)  # (B, 2N)

            with autocast_ctx:
                h = model.encode_interleaved(content_ids, action_ids,
                                             ts_2n, mask_2n, is_content)  # (B, 2N, D)

                # Per-content-position scoring:
                #   At even position 2i (content c_i), score h_{2i} against
                #   item_full_embed(c_i). Target = engaged(rating_bucket_i).
                # The dot product is "the user's hidden state at the content
                # slot agrees with this item's embedding when they engaged."
                logits = model.score_per_position(h, content_ids)    # (B, 2N)
                targets = (content_rb >= ENGAGED_BUCKET_THRESHOLD).float()
                # Loss mask: only at valid CONTENT positions. Action positions
                # contribute 0; pad pairs (mask_2n=0) contribute 0.
                loss_mask = mask_2n * is_content.float()
                loss_per = bce(logits, targets)                      # (B, 2N)
                masked = loss_per * loss_mask
                n_valid = loss_mask.sum().clamp(min=1.0)
                loss = masked.sum() / n_valid
                # Auxiliary rating-regression head: predict normalized rating
                # bucket (content_rb / 9.0) from h_t (raw encoder output, NOT
                # the MLP-projected variant). Same mask as BCE so only valid
                # content positions contribute.
                if use_aux:
                    aux_pred = model.aux_head(h).squeeze(-1)         # (B, 2N)
                    aux_target = content_rb.float() / 9.0            # (B, 2N) ∈ [0, 1]
                    aux_sq = (aux_pred - aux_target) ** 2 * loss_mask
                    aux_mse = aux_sq.sum() / n_valid
                    loss = loss + AUX_RATING_WEIGHT * aux_mse
            n_positions = int(loss_mask.sum().item())
        else:
            hist_items = batch["hist_items"].to(DEVICE)         # (B, T)
            hist_ratings = batch["hist_ratings"].to(DEVICE)     # (B, T)
            hist_ts = batch["hist_ts"].to(DEVICE)               # (B, T)
            hist_mask = batch["hist_mask"].to(DEVICE)           # (B, T)

            with autocast_ctx:
                h = model.encode(hist_items, hist_ratings, hist_ts, hist_mask)  # (B, T, D)

                # Per-position next-event prediction:
                #   At position t (0..T-2), score h_t against item_embed(events[t+1])
                #   target = engagement bucket of events[t+1] (rating >= 4 → 1)
                # h, items, ratings, mask are all aligned over T positions.
                h_in = h[:, :-1, :]                                 # (B, T-1, D)
                next_items = hist_items[:, 1:]                      # (B, T-1)
                next_ratings = hist_ratings[:, 1:]                  # (B, T-1)
                # valid position = both current AND next event are real (not pad)
                pair_mask = hist_mask[:, :-1] * hist_mask[:, 1:]    # (B, T-1)

                logits = model.score_per_position(h_in, next_items)  # (B, T-1)
                targets = (next_ratings >= ENGAGED_BUCKET_THRESHOLD).float()
                loss_per = bce(logits, targets)                     # (B, T-1)
                masked = loss_per * pair_mask
                n_valid = pair_mask.sum().clamp(min=1.0)
                loss = masked.sum() / n_valid
                # Auxiliary rating-regression head: predict normalized rating
                # bucket of the NEXT event (next_ratings / 9.0) from h_in. Same
                # pair_mask as BCE. h_in (pre-MLP) feeds the aux head — parallel
                # branch off the encoder, not the projected scoring head.
                if use_aux:
                    aux_pred = model.aux_head(h_in).squeeze(-1)      # (B, T-1)
                    aux_target = next_ratings.float() / 9.0          # (B, T-1) ∈ [0, 1]
                    aux_sq = (aux_pred - aux_target) ** 2 * pair_mask
                    aux_mse = aux_sq.sum() / n_valid
                    loss = loss + AUX_RATING_WEIGHT * aux_mse
            n_positions = int(pair_mask.sum().item())

        optimizer.zero_grad()
        loss.backward()
        if GRAD_CLIP > 0.0:
            # clip_grad_norm_ returns the pre-clip global norm (a tensor).
            pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            grad_norm_sum += float(pre_clip)
            grad_norm_count += 1
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += masked.sum().item()
        total_positions += n_positions
        if use_aux:
            aux_loss_sum += float(aux_mse.detach())
            aux_loss_count += 1
    avg_loss = total_loss / max(1, total_positions)
    avg_grad_norm = (grad_norm_sum / grad_norm_count) if grad_norm_count > 0 else None
    avg_aux_loss = (aux_loss_sum / aux_loss_count) if aux_loss_count > 0 else None
    return avg_loss, avg_grad_norm, avg_aux_loss


@torch.no_grad()
def evaluate_model(model, loader, save_preds_path: str | None = None):
    """Per-(user, candidate, ts) eval: dot(h_T, item_embed(candidate)) → sigmoid → AUC.

    INTERLEAVE=1 path: append the candidate as a content token at position 2N
    (after all 2N prefix tokens), forward the model, and score
    dot(h_{2N}, item_full_embed(candidate)) at the appended position. The
    candidate's ts is the eval row's target_ts (so the time-delta bias
    against the prefix is computed from the right cutoff).

    When `save_preds_path` is not None, also accumulate per-row uid / mid /
    label / score / prefix_len and write a CSV at the given path before
    returning. Off-state byte-equivalent to the prior baseline (no extra
    tensors built or appended when save_preds_path is None).
    """
    model.eval()
    all_scores, all_labels = [], []
    pred_rows = [] if save_preds_path else None
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (USE_BF16 and DEVICE == "cuda") else nullcontext()
    )
    for batch in loader:
        cand = batch["mid"].to(DEVICE)
        label = batch["label"]
        if INTERLEAVE:
            content_ids = batch["content_ids"].to(DEVICE)
            action_ids = batch["action_ids"].to(DEVICE)
            ts_2n = batch["ts_2n"].to(DEVICE)
            mask_2n = batch["mask_2n"].to(DEVICE)
            is_content = batch["is_content"].to(DEVICE)
            target_ts = batch["target_ts"].to(DEVICE)
            with autocast_ctx:
                last_h = model.encode_interleaved_with_candidate(
                    content_ids, action_ids, ts_2n, mask_2n, is_content,
                    cand, target_ts,
                )                                                    # (B, D)
                last_h = model._project_head(last_h)
                cand_e = model.item_full_embed(cand)                 # (B, D)
                logit = (last_h * cand_e).sum(dim=-1)                # (B,)
        else:
            hist_items = batch["hist_items"].to(DEVICE)
            hist_ratings = batch["hist_ratings"].to(DEVICE)
            hist_ts = batch["hist_ts"].to(DEVICE)
            hist_mask = batch["hist_mask"].to(DEVICE)
            with autocast_ctx:
                h = model.encode(hist_items, hist_ratings, hist_ts, hist_mask)
                logit = model.score_eval(h, hist_mask, cand)
        # Cast to fp32 for sigmoid + AUC accumulation. Outside autocast scope
        # bf16 tensors are returned as-is; .float() makes the numpy conversion
        # match the fp32 path bit-equivalent. (No-op when autocast is null.)
        scores_np = torch.sigmoid(logit.float()).detach().cpu().numpy()
        labels_np = label.numpy()
        all_scores.append(scores_np)
        all_labels.append(labels_np)
        if pred_rows is not None:
            # prefix_len = number of valid PRIOR EVENTS the model saw on this row.
            # Fused path: hist_mask is (B, N) per-event, so .sum() over dim=1 == events.
            # Interleaved path: mask_2n is (B, 2N) and pads/keeps content+action as a
            # PAIR (both masked together by _build_interleaved), so events = mask/2.
            if INTERLEAVE:
                prefix_len = (mask_2n.sum(dim=1) // 2).long().detach().cpu().numpy()
            else:
                prefix_len = hist_mask.sum(dim=1).long().detach().cpu().numpy()
            uids = np.asarray(batch["uid"], dtype=np.int64)
            mids = cand.detach().cpu().numpy().astype(np.int64)
            pred_rows.append(np.stack([uids, mids, labels_np.astype(np.float64),
                                       scores_np.astype(np.float64),
                                       prefix_len.astype(np.int64)], axis=1))
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    if pred_rows is not None:
        arr = np.concatenate(pred_rows, axis=0)
        df = pd.DataFrame({
            "uid": arr[:, 0].astype(np.int64),
            "mid": arr[:, 1].astype(np.int64),
            "label": arr[:, 2].astype(np.float32),
            "score": arr[:, 3].astype(np.float32),
            "prefix_len": arr[:, 4].astype(np.int64),
        })
        df.to_csv(save_preds_path, index=False)
        log.info(f"  saved {len(df)} eval predictions to {save_preds_path}")
    return evaluate(labels, scores)["auc"]


def main():
    t0 = time.time()
    log.info(f"Loading {DATASET} (raw rating events; no feature engineering)")
    data = load_data(DATASET)
    train_df, val_df, test_df = data["train"], data["val"], data["test"]
    movies_df = data["movies"]
    stats = data["stats"]
    log.info(f"  num_users={stats['num_users']}  num_items={stats['num_items']}  "
             f"num_train={stats['num_train']}  num_val={stats['num_val']}  num_test={stats['num_test']}")

    # Cold-start content metadata. Loaded once; allocated regardless of flags
    # so the model construction is uniform. Tables sized (num_items + 1, ...)
    # to align with the +1-shifted movieId convention used throughout the model.
    log.info(f"Loading movie metadata (USE_GENOME={USE_GENOME} USE_GENRE={USE_GENRE} USE_YEAR={USE_YEAR})")
    genome_np, genre_np, year_id_np = load_movie_metadata(movies_df, DATASET, stats["num_items"])
    genome_t = torch.from_numpy(genome_np).to(DEVICE)
    genre_t = torch.from_numpy(genre_np).to(DEVICE)
    year_id_t = torch.from_numpy(year_id_np).to(DEVICE)

    # Popularity prior table (USE_POP_PRIOR=1): per-item normalized log-rating-count
    # built ONCE from train_df only (val/test rows never enter this aggregation,
    # and prepare.load_data's time-based split guarantees train timestamps strictly
    # precede val/test, so train counts are inherently strictly-prior to every
    # val/test row). Sized (num_items + 1,) to match the +1-shifted movieId
    # convention: row 0 is PAD (count=0 → log1p=0 → normalized 0). Allocated
    # regardless of the flag so the constructor signature stays uniform; the
    # buffer is registered (and pop_proj built) only when use_pop_prior is True.
    train_real = train_df[train_df["rating"] > 0]
    item_counts = (
        train_real.groupby("movieId").size()
        .reindex(range(stats["num_items"]), fill_value=0)
        .values
    )
    item_log_count_real = np.log1p(item_counts.astype(np.float64)).astype(np.float32)
    max_log = float(item_log_count_real.max())
    item_pop_real = item_log_count_real / max(max_log, 1.0)            # (num_items,) ∈ [0, 1]
    item_pop_np = np.zeros(stats["num_items"] + 1, dtype=np.float32)   # (num_items+1,)
    item_pop_np[1:] = item_pop_real                                    # row 0 = PAD = 0
    item_pop_t = torch.from_numpy(item_pop_np).to(DEVICE)
    log.info(f"  pop prior: max_log_count={max_log:.3f}  "
             f"items_with_train_ratings={int((item_counts > 0).sum())}/{stats['num_items']}")

    # Per-item rating statistics table (USE_ITEM_STATS=1): 3 scalars per item
    # — (mean_rating/5.0, std_rating/2.5, frac_engaged) — computed ONCE from
    # train_df ONLY. Time-leak guarantee: the source frame is `train_real`
    # from above (`train_df[train_df["rating"] > 0]`), which contains zero
    # val/test rows by construction (prepare.load_data's split is purely
    # time-based and disjoint by row index). So every aggregation here is
    # strictly-prior to all val/test samples — the same guarantee that
    # backs USE_POP_PRIOR. Allocated regardless of the flag so the
    # constructor signature stays uniform; the buffer/projection are
    # registered only when use_item_stats is True.
    #
    # NaN imputation:
    #   - Items with ZERO train ratings: mean=NaN, std=NaN, frac=NaN → impute
    #     0 for all three scalars. The signal is "we have no info", which
    #     differs from "we know mean=0" since the smallest valid mean is
    #     0.5/5.0 = 0.1 — so 0 is unambiguously a sentinel.
    #   - Items with EXACTLY 1 train rating: mean and frac are well-defined,
    #     but pandas .std() returns NaN (default ddof=1 needs n>=2). Impute
    #     std=0 only — leave mean and frac at their real single-sample values.
    item_groupby = train_real.groupby("movieId")
    mean_rating = item_groupby["rating"].mean().reindex(range(stats["num_items"]))
    std_rating = item_groupby["rating"].std().reindex(range(stats["num_items"]))
    frac_engaged = (
        item_groupby["rating"].apply(lambda r: (r >= 4).mean())
        .reindex(range(stats["num_items"]))
    )
    mean_norm = (mean_rating / 5.0).fillna(0.0).to_numpy(dtype=np.float32)
    std_norm = (std_rating / 2.5).fillna(0.0).to_numpy(dtype=np.float32)
    frac_norm = frac_engaged.fillna(0.0).to_numpy(dtype=np.float32)
    item_stats_real = np.stack([mean_norm, std_norm, frac_norm], axis=1)   # (num_items, 3)
    item_stats_np = np.zeros((stats["num_items"] + 1, 3), dtype=np.float32)
    item_stats_np[1:] = item_stats_real                                    # row 0 = PAD = zeros
    item_stats_t = torch.from_numpy(item_stats_np).to(DEVICE)
    log.info(f"  item stats: mean_norm avg={float(mean_norm.mean()):.3f}  "
             f"std_norm avg={float(std_norm.mean()):.3f}  "
             f"frac_engaged avg={float(frac_norm.mean()):.3f}")

    # Bucketed absolute rating-timestamp constants (USE_RATING_TS=1, may03
    # variant A). TRAIN_TS_MIN is the minimum timestamp in train_df only — val
    # and test never enter this aggregation (per prepare.load_data's time-based
    # split, train timestamps strictly precede val/test, so this is a safe
    # train-only statistic). The HSTU constructor receives train_ts_min and
    # registers it as a buffer along with SECONDS_PER_MONTH; bucketing happens
    # in HSTU._ts_bucket on the fly during forward. Computed regardless of the
    # flag so the constructor signature stays uniform; the embedding is built
    # only when USE_RATING_TS=1.
    train_ts_min = int(train_df["timestamp"].min())
    train_ts_max = int(train_df["timestamp"].max())
    span_seconds = train_ts_max - train_ts_min
    span_months = span_seconds / SECONDS_PER_MONTH
    log.info(f"  rating-ts bucketing: train_ts_min={train_ts_min}  "
             f"train_ts_max={train_ts_max}  span={span_months:.1f} months  "
             f"num_buckets={NUM_TS_BUCKETS}  "
             f"~{span_months / NUM_TS_BUCKETS:.1f} months/bucket")

    # Eval history uses train+val with per-sample strict-prior cutoff (mirrors
    # simple_v2's EVAL_DYNAMIC_HIST=1 — the +0.022 win at apr28ad came from val
    # rows seeing their OWN earlier val rows in history, while side="left"
    # excludes the sample itself and any tied events). This is sequential-eval
    # framing, not i.i.d. leakage. See simple_v2/CLAUDE.md apr28ad description.
    log.info("Building per-user event sequences")
    train_history = build_user_sequences(train_df)
    eval_history = build_user_sequences(pd.concat([train_df, val_df], ignore_index=True))

    train_ds = SequenceTrainDataset(train_history, SEQ_LEN)
    val_ds = EvalDataset(val_df, eval_history, SEQ_LEN)
    log.info(f"  train_sequences={len(train_ds)}  val_rows={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_train)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_eval)

    model = HSTU(stats["num_items"], NUM_RATING_BUCKETS,
                 genome=genome_t, genre=genre_t, year_id=year_id_t,
                 item_pop=item_pop_t, item_stats=item_stats_t,
                 train_ts_min=train_ts_min).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"HSTU: {n_params/1e6:.2f}M params on {DEVICE}  "
             f"(layers={NUM_LAYERS}, heads={NUM_HEADS}, dim={EMBED_DIM}, "
             f"time_buckets={NUM_TIME_BUCKETS}) "
             f"use_genome={model.use_genome} use_genre={model.use_genre} use_year={model.use_year} "
             f"use_pop_prior={model.use_pop_prior} use_item_stats={model.use_item_stats} "
             f"use_rating_ts={model.use_rating_ts} "
             f"use_mlp_head={model.use_mlp_head} use_interleave={model.use_interleave} "
             f"use_aux_rating={model.use_aux_rating}"
             + (f" aux_w={AUX_RATING_WEIGHT}" if model.use_aux_rating else ""))

    optimizer = build_optimizer(model, LR, WEIGHT_DECAY, OPTIMIZER)
    # Total scheduler steps = batches/epoch * MAX_EPOCHS. LR_SCHEDULE="constant"
    # short-circuits to scheduler=None — no scheduler.step() calls below, off-state
    # byte-equivalent to the prior plain-Adam baseline.
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = build_scheduler(optimizer, LR, LR_SCHEDULE, WARMUP_STEPS, total_steps)
    log.info(f"  optimizer={OPTIMIZER}  lr_schedule={LR_SCHEDULE}  "
             f"warmup_steps={WARMUP_STEPS}  total_steps={total_steps}  "
             f"start_lr={optimizer.param_groups[0]['lr']:.2e}  "
             f"use_bf16={bool(USE_BF16)}")
    if LR_SCHEDULE == "cawr":
        steps_per_epoch = total_steps // MAX_EPOCHS
        cawr_T_0 = CAWR_T_0_EPOCHS * steps_per_epoch
        log.info(f"  cawr: T_0={cawr_T_0} steps ({CAWR_T_0_EPOCHS} epochs)  "
                 f"T_mult={CAWR_T_MULT}  eta_min={LR * CAWR_ETA_MIN_FRAC:.2e} "
                 f"(eta_min_frac={CAWR_ETA_MIN_FRAC})")

    best_val_auc = 0.0
    for epoch in range(MAX_EPOCHS):
        train_loss, avg_grad_norm, avg_aux_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
        )
        # Only request the per-row prediction dump on the FINAL epoch's eval,
        # both to keep prior epochs byte-equivalent to the off path AND to
        # reflect the deployed model state. SAVE_PREDS=0 → save_path stays None
        # for every call, so off-state is byte-equivalent.
        save_path = (
            SAVE_PREDS_PATH
            if (SAVE_PREDS and epoch == MAX_EPOCHS - 1)
            else None
        )
        val_auc = evaluate_model(model, val_loader, save_preds_path=save_path)
        # Only append grad_norm when GRAD_CLIP fired (avg_grad_norm not None);
        # only append aux_loss when AUX_RATING_WEIGHT > 0 (avg_aux_loss not None).
        # OFF-state log line is unchanged from the prior baseline.
        gn_part = f" grad_norm={avg_grad_norm:.2f}" if avg_grad_norm is not None else ""
        aux_part = f" aux_loss={avg_aux_loss:.4f}" if avg_aux_loss is not None else ""
        cur_lr = optimizer.param_groups[0]["lr"]
        log.info(f"epoch {epoch}: train_loss={train_loss:.4f} val_auc={val_auc:.4f}{aux_part}{gn_part} lr={cur_lr:.2e}")
        best_val_auc = max(best_val_auc, val_auc)

    total = time.time() - t0
    print(f"\nval_auc:          {best_val_auc:.6f}")
    print(f"total_seconds:    {total:.1f}")
    print(f"dataset:          {DATASET}")
    print(f"num_params_M:     {n_params/1e6:.2f}")
    print(f"# bar to clear (simple_v2 locked): val 0.8594 / test 0.8455")


if __name__ == "__main__":
    main()
