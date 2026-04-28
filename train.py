#!/usr/bin/env python3
"""
DLRM for hybrid engagement prediction on MovieLens.

Label scheme:
  - 1: user rated >= 4 (watched and liked)
  - 0: user rated < 4 (hard negative) OR random unrated movie (easy negative)

Loss: BCE (calibrated probabilities for threshold-based serving).
"""

import copy
import logging
import os
import sys
from pathlib import Path
import time

import numpy as np

# Fix all random seeds for reproducibility
SEED = int(os.environ.get("SEED", "42"))
np.random.seed(SEED)
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from prepare import load_data_hybrid, evaluate, print_summary

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
log = logging.getLogger("train")

# ─── Helpers ────────────────────────────────────────────────────────

def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _masked_softmax(scores, valid_mask, dim=-1):
    masked_scores = scores.masked_fill(~valid_mask, -1e4)
    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights * valid_mask.to(weights.dtype)
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return weights / denom


# ─── Configuration ──────────────────────────────────────────────────
DATASET = os.environ.get("DATASET", "ml-25m")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16384"))
LR = float(os.environ.get("LR", "7e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "5e-5"))
EMBED_DIM = int(os.environ.get("EMBED_DIM", "28"))
HISTORY_LEN = int(os.environ.get("HISTORY_LEN", "100"))
ITEM_HIST_LEN = int(os.environ.get("ITEM_HIST_LEN", "30"))
NUM_DENSE = 17  # 1 timestamp + 5 user hist bins + 1 user count + 5 item hist bins + 1 item count + 1 ug_dot + 1 year + 1 genre_count + 1 movie_age
NEG_RATIO = int(os.environ.get("NEG_RATIO", "1"))  # random unrated negatives per positive in training data
EVAL_EVERY = 1
PATIENCE = int(os.environ.get("PATIENCE", "3"))
ACCUM_STEPS = int(os.environ.get("ACCUM_STEPS", "2"))
RECENCY_FRAC = float(os.environ.get("RECENCY_FRAC", "0.7"))
TRAIN_NEG_MODE = os.environ.get("TRAIN_NEG_MODE", "anchor_pos_catalog")  # global | anchor_pos | anchor_pos_catalog
POST_RECENCY_NEG_RESAMPLE = _env_flag("POST_RECENCY_NEG_RESAMPLE", True)
POST_RECENCY_EASY_NEG_PER_POS = float(os.environ.get("POST_RECENCY_EASY_NEG_PER_POS", "0.4"))
USER_HIST_MODE = os.environ.get("USER_HIST_MODE", "rating")  # din | mean | rating
ITEM_HIST_MODE = os.environ.get("ITEM_HIST_MODE", "din")  # din | mean | off
USER_HIST_CONTEXT = os.environ.get("USER_HIST_CONTEXT", "causal_masked")  # static | causal_masked
ITEM_HIST_CONTEXT = os.environ.get("ITEM_HIST_CONTEXT", "causal_masked")  # static | causal_masked
GENOME_FUSION_MODE = os.environ.get("GENOME_FUSION_MODE", "legacy")  # legacy | mask_only
USER_GENOME = os.environ.get("USER_GENOME", "scalar_dot")  # off | scalar_dot
USER_GENOME_TARGET = os.environ.get("USER_GENOME_TARGET", "genome_field")  # dense_e | genome_field
USE_CAUSAL_SA = _env_flag("USE_CAUSAL_SA", True)
USE_TORCH_COMPILE = _env_flag("USE_TORCH_COMPILE", True)
EMBED_DROPOUT = float(os.environ.get("EMBED_DROPOUT", "0.1"))
MLP_DROPOUT = float(os.environ.get("MLP_DROPOUT", "0.3"))
LABEL_SMOOTH = float(os.environ.get("LABEL_SMOOTH", "0.05"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "0.0"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "0"))
GENOME_BOTTLENECK_HIDDEN = os.environ.get("GENOME_BOTTLENECK_HIDDEN", "256,64")
GENOME_BOTTLENECK_DROPOUT = float(os.environ.get("GENOME_BOTTLENECK_DROPOUT", "0.0"))
TX_ENCODER_LAYERS = int(os.environ.get("TX_ENCODER_LAYERS", "0"))
TX_ENCODER_HEADS = int(os.environ.get("TX_ENCODER_HEADS", "2"))
TX_FFN_RATIO = int(os.environ.get("TX_FFN_RATIO", "2"))
TX_DROPOUT = float(os.environ.get("TX_DROPOUT", "0.1"))
TX_POOL = os.environ.get("TX_POOL", "target")  # target | rating | cls
TX_POS = os.environ.get("TX_POS", "learned")   # learned | sinusoidal
TX_BYPASS_SE = int(os.environ.get("TX_BYPASS_SE", "0"))
TX_GATE_INIT = float(os.environ.get("TX_GATE_INIT", "0.0"))  # for ε-test only

if TX_POOL not in {"target", "rating", "cls"}:
    raise ValueError(f"Unknown TX_POOL: {TX_POOL}")
if TX_POS not in {"learned", "sinusoidal"}:
    raise ValueError(f"Unknown TX_POS: {TX_POS}")

if TRAIN_NEG_MODE not in {"global", "anchor_pos", "anchor_pos_catalog"}:
    raise ValueError(f"Unknown TRAIN_NEG_MODE: {TRAIN_NEG_MODE}")
if USER_HIST_CONTEXT not in {"static", "causal_masked"}:
    raise ValueError(f"Unknown USER_HIST_CONTEXT: {USER_HIST_CONTEXT}")
if ITEM_HIST_CONTEXT not in {"static", "causal_masked"}:
    raise ValueError(f"Unknown ITEM_HIST_CONTEXT: {ITEM_HIST_CONTEXT}")
if GENOME_FUSION_MODE not in {"legacy", "mask_only"}:
    raise ValueError(f"Unknown GENOME_FUSION_MODE: {GENOME_FUSION_MODE}")
if USER_GENOME not in {"off", "scalar_dot"}:
    raise ValueError(f"Unknown USER_GENOME: {USER_GENOME}")
if USER_GENOME_TARGET not in {"dense_e", "genome_field"}:
    raise ValueError(f"Unknown USER_GENOME_TARGET: {USER_GENOME_TARGET}")

# ─── Device ─────────────────────────────────────────────────────────
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')  # enable TF32 tensor cores
log.info(
    f"Device: {DEVICE} | "
    f"user_hist={USER_HIST_MODE} | item_hist={ITEM_HIST_MODE} | "
    f"user_ctx={USER_HIST_CONTEXT} | item_ctx={ITEM_HIST_CONTEXT} | "
    f"causal_sa={USE_CAUSAL_SA} | dim={EMBED_DIM} | item_hist_len={ITEM_HIST_LEN} | "
    f"train_neg={TRAIN_NEG_MODE} | post_recency_resample={POST_RECENCY_NEG_RESAMPLE} "
    f"| easy_neg_per_pos={POST_RECENCY_EASY_NEG_PER_POS:.2f} "
    f"| genome_fusion={GENOME_FUSION_MODE} | user_genome={USER_GENOME}"
)

# ─── Load Data ──────────────────────────────────────────────────────
total_start = time.time()
data = load_data_hybrid(DATASET, neg_ratio=NEG_RATIO, train_neg_mode=TRAIN_NEG_MODE)
train_df, val_df = data["train"], data["val"]
movies_df, stats = data["movies"], data["stats"]
user_all_items = data["user_all_items"]
num_users, num_items = stats["num_users"], stats["num_items"]
log.info(f"Dataset: {DATASET} | Users: {num_users} | Items: {num_items} | "
         f"Train: {stats['num_train']} | Val: {stats['num_val']} | "
         f"Pos rate: {stats['pos_rate']:.2%}")


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (with disk caching)
# ═══════════════════════════════════════════════════════════════════

import hashlib, json
_cache_key = hashlib.md5(json.dumps({
    "feature_version": 5,
    "dataset": DATASET, "neg_ratio": NEG_RATIO,
    "history_len": HISTORY_LEN, "item_hist_len": ITEM_HIST_LEN,
    "num_users": num_users, "num_items": num_items,
}).encode()).hexdigest()[:12]
_cache_path = Path(__file__).parent / "data" / f"features_{_cache_key}.npz"

if _cache_path.exists():
    log.info(f"Loading cached features from {_cache_path.name}...")
    _cache = np.load(_cache_path, allow_pickle=True)
    movie_genres = _cache["movie_genres"]
    num_genres = movie_genres.shape[1]
    user_hist_bins = _cache["user_hist_bins"]
    item_hist_bins = _cache["item_hist_bins"]
    user_count_norm = _cache["user_count_norm"]
    item_count_norm = _cache["item_count_norm"]
    user_histories = _cache["user_histories"]
    user_hist_ratings = _cache["user_hist_ratings"]
    user_hist_timestamps = _cache["user_hist_timestamps"]
    item_histories = _cache["item_histories"]
    item_hist_ratings = _cache["item_hist_ratings"]
    item_hist_timestamps = _cache["item_hist_timestamps"]
    user_genre_affinity = _cache["user_genre_affinity"]
    ts_min = float(_cache["ts_min"])
    ts_range = float(_cache["ts_range"])
    movie_year = _cache["movie_year"]
    movie_genre_count = _cache["movie_genre_count"]
    genome_matrix = _cache["genome_matrix"]
    has_genome = _cache["has_genome"]
    user_genome = _cache["user_genome"]
    has_user_genome = _cache["has_user_genome"]
    GENOME_DIM = genome_matrix.shape[1]
    PAD_IDX = num_items
    USER_PAD_IDX = num_users
    ITEM_HIST_LEN = item_histories.shape[1]
    del _cache
    log.info("Cached features loaded.")
else:
    log.info("Computing features (will cache for next run)...")
    real_train = train_df[train_df["rating"] > 0]

    # 1. Genre multi-hot encoding
    all_genres_set = set()
    for g in movies_df["genres"].dropna():
        all_genres_set.update(g.split("|"))
    all_genres = sorted(all_genres_set - {""})
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    num_genres = len(all_genres)
    movie_genres = np.zeros((num_items, num_genres), dtype=np.float32)
    for _, row in movies_df.iterrows():
        mid = int(row["movieId"])
        if mid < num_items and isinstance(row["genres"], str):
            for g in row["genres"].split("|"):
                if g in genre_to_idx:
                    movie_genres[mid, genre_to_idx[g]] = 1.0

    # 2. Rating histograms (vectorized)
    _rt_uids = real_train["userId"].values
    _rt_mids = real_train["movieId"].values
    _rt_ratings = real_train["rating"].values
    _rt_bins = np.clip((_rt_ratings - 0.01) // 1, 0, 4).astype(np.intp)
    user_hist_bins = np.zeros((num_users, 5), dtype=np.float32)
    item_hist_bins = np.zeros((num_items, 5), dtype=np.float32)
    np.add.at(user_hist_bins, (_rt_uids, _rt_bins), 1)
    np.add.at(item_hist_bins, (_rt_mids, _rt_bins), 1)
    user_count = user_hist_bins.sum(axis=1)
    item_count = item_hist_bins.sum(axis=1)
    user_hist_bins = user_hist_bins / (user_count[:, None] + 1e-8)
    item_hist_bins = item_hist_bins / (item_count[:, None] + 1e-8)
    user_count_norm = (user_count - user_count.mean()) / (user_count.std() + 1e-8)
    item_count_norm = (item_count - item_count.mean()) / (item_count.std() + 1e-8)
    del _rt_bins

    # 3. User/item history sequences (vectorized)
    PAD_IDX = num_items
    USER_PAD_IDX = num_users

    def _build_history_arrays(ids, targets, ratings, timestamps, num_entities, pad_idx, max_len):
        sort_idx = np.lexsort((timestamps, ids))
        s_ids, s_targets = ids[sort_idx], targets[sort_idx]
        s_ratings = ratings[sort_idx].astype(np.float32) / 5.0
        s_timestamps = timestamps[sort_idx].astype(np.int32)
        hist = np.full((num_entities, max_len), pad_idx, dtype=np.int64)
        hist_rat = np.zeros((num_entities, max_len), dtype=np.float32)
        hist_ts = np.zeros((num_entities, max_len), dtype=np.int32)
        boundaries = np.where(np.diff(s_ids) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(s_ids)]])
        entity_ids = s_ids[starts]
        for i in range(len(entity_ids)):
            eid, s, e = entity_ids[i], starts[i], ends[i]
            length = min(e - s, max_len)
            hist[eid, -length:] = s_targets[e - length:e]
            hist_rat[eid, -length:] = s_ratings[e - length:e]
            hist_ts[eid, -length:] = s_timestamps[e - length:e]
        return hist, hist_rat, hist_ts

    _rt_ts = real_train["timestamp"].values
    user_histories, user_hist_ratings, user_hist_timestamps = _build_history_arrays(
        _rt_uids, _rt_mids, _rt_ratings, _rt_ts, num_users, PAD_IDX, HISTORY_LEN)
    item_histories, item_hist_ratings, item_hist_timestamps = _build_history_arrays(
        _rt_mids, _rt_uids, _rt_ratings, _rt_ts, num_items, USER_PAD_IDX, ITEM_HIST_LEN)

    # 4. User-genre affinity
    user_genre_affinity = np.zeros((num_users, num_genres), dtype=np.float32)
    user_genre_count = np.zeros((num_users, num_genres), dtype=np.float32)
    _genres_of_items = movie_genres[_rt_mids]
    np.add.at(user_genre_affinity, _rt_uids, _rt_ratings[:, None].astype(np.float32) * _genres_of_items)
    np.add.at(user_genre_count, _rt_uids, _genres_of_items)
    del _rt_uids, _rt_mids, _rt_ratings, _rt_ts, _genres_of_items
    mask = user_genre_count > 0
    user_genre_affinity[mask] /= user_genre_count[mask]
    mu, sigma = user_genre_affinity[mask].mean(), user_genre_affinity[mask].std() + 1e-8
    user_genre_affinity[mask] = (user_genre_affinity[mask] - mu) / sigma

    # 5. Timestamp normalization
    ts_min = float(real_train["timestamp"].min())
    ts_range = float(real_train["timestamp"].max() - ts_min) + 1.0

    # 6. Movie release year
    import re
    movie_year = np.zeros(num_items, dtype=np.float32)
    for _, row in movies_df.iterrows():
        mid = int(row["movieId"])
        if mid < num_items:
            m = re.search(r'\((\d{4})\)', str(row.get("title", "")))
            if m:
                movie_year[mid] = float(m.group(1))
    valid_years = movie_year[movie_year > 0]
    if len(valid_years) > 0:
        median_year = np.median(valid_years)
        movie_year[movie_year == 0] = median_year
        movie_year = (movie_year - movie_year.mean()) / (movie_year.std() + 1e-8)

    # 7. Genre count per movie
    movie_genre_count = movie_genres.sum(axis=1).astype(np.float32)
    movie_genre_count = (movie_genre_count - movie_genre_count.mean()) / (movie_genre_count.std() + 1e-8)

    # 8. Tag genome features (1128 relevance scores per movie)
    _genome_path = Path(__file__).parent / "data" / DATASET / "genome-scores.csv"
    if _genome_path.exists():
        log.info("Loading tag genome scores...")
        # Replicate prepare.py's ID remapping: sort ratings by timestamp, unique movieIds in order
        _raw_ratings_path = Path(__file__).parent / "data" / DATASET
        if DATASET == "ml-25m":
            _raw_ratings = pd.read_csv(_raw_ratings_path / "ratings.csv")
        elif DATASET == "ml-10m":
            _raw_ratings = pd.read_csv(
                Path(__file__).parent / "data" / "ml-10M100K" / "ratings.dat",
                sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"])
        elif DATASET == "ml-1m":
            _raw_ratings = pd.read_csv(
                _raw_ratings_path / "ratings.dat", sep="::", engine="python",
                names=["userId", "movieId", "rating", "timestamp"])
        else:
            _raw_ratings = None
        if _raw_ratings is not None:
            _raw_ratings = _raw_ratings.sort_values("timestamp").reset_index(drop=True)
            _all_movies_ordered = _raw_ratings["movieId"].unique()
            _movie_map = {mid: i for i, mid in enumerate(_all_movies_ordered)}
            _genome_df = pd.read_csv(_genome_path)
            _num_tags = int(_genome_df["tagId"].max())
            genome_matrix = np.zeros((num_items, _num_tags), dtype=np.float32)
            has_genome = np.zeros(num_items, dtype=np.float32)
            # Map genome movieIds to remapped IDs
            _genome_df["mapped_mid"] = _genome_df["movieId"].map(_movie_map)
            _genome_df = _genome_df.dropna(subset=["mapped_mid"])
            _genome_df["mapped_mid"] = _genome_df["mapped_mid"].astype(int)
            # Pivot: fill genome_matrix
            for mid, group in _genome_df.groupby("mapped_mid"):
                if mid < num_items:
                    tag_ids = group["tagId"].values.astype(int) - 1  # 1-indexed to 0-indexed
                    genome_matrix[mid, tag_ids] = group["relevance"].values.astype(np.float32)
                    has_genome[mid] = 1.0
            log.info(f"Tag genome: {int(has_genome.sum())} movies with data out of {num_items} ({has_genome.mean()*100:.1f}%)")
            del _genome_df, _raw_ratings, _all_movies_ordered, _movie_map
        else:
            genome_matrix = np.zeros((num_items, 1128), dtype=np.float32)
            has_genome = np.zeros(num_items, dtype=np.float32)
    else:
        log.info("No genome-scores.csv found, using zeros.")
        genome_matrix = np.zeros((num_items, 1128), dtype=np.float32)
        has_genome = np.zeros(num_items, dtype=np.float32)
    GENOME_DIM = genome_matrix.shape[1]

    # 9. Per-user genome profile: mean of genome relevance vectors over user's high-rated genome-having items.
    log.info("Computing user genome profiles...")
    user_genome = np.zeros((num_users, GENOME_DIM), dtype=np.float32)
    _ug_count = np.zeros(num_users, dtype=np.float32)
    _hr_mask = (real_train["rating"].values >= 4.0)
    _hr_uids = real_train["userId"].values[_hr_mask].astype(np.int64)
    _hr_mids = real_train["movieId"].values[_hr_mask].astype(np.int64)
    _has_g = has_genome[_hr_mids] > 0
    _hr_uids = _hr_uids[_has_g]
    _hr_mids = _hr_mids[_has_g]
    _chunk = 500_000
    for i in range(0, len(_hr_uids), _chunk):
        u_chunk = _hr_uids[i:i + _chunk]
        m_chunk = _hr_mids[i:i + _chunk]
        np.add.at(user_genome, u_chunk, genome_matrix[m_chunk])
        np.add.at(_ug_count, u_chunk, 1.0)
    _has_ug = _ug_count > 0
    user_genome[_has_ug] /= _ug_count[_has_ug, None]
    has_user_genome = _has_ug.astype(np.float32)
    log.info(f"User genome: {int(_has_ug.sum())}/{num_users} users with profile ({_has_ug.mean()*100:.1f}%)")
    del _hr_mask, _hr_uids, _hr_mids, _has_g, _has_ug, _ug_count

    # Save cache
    np.savez_compressed(_cache_path,
        movie_genres=movie_genres, user_hist_bins=user_hist_bins,
        item_hist_bins=item_hist_bins, user_count_norm=user_count_norm,
        item_count_norm=item_count_norm, user_histories=user_histories,
        user_hist_ratings=user_hist_ratings, user_hist_timestamps=user_hist_timestamps,
        item_histories=item_histories, item_hist_ratings=item_hist_ratings,
        item_hist_timestamps=item_hist_timestamps, user_genre_affinity=user_genre_affinity,
        ts_min=np.array(ts_min), ts_range=np.array(ts_range),
        movie_year=movie_year, movie_genre_count=movie_genre_count,
        genome_matrix=genome_matrix, has_genome=has_genome,
        user_genome=user_genome, has_user_genome=has_user_genome,
    )
    log.info(f"Features cached to {_cache_path.name}")


# ═══════════════════════════════════════════════════════════════════
# RECENCY FILTER — drop oldest 20% of real ratings (features already computed from full data)
# ═══════════════════════════════════════════════════════════════════

full_real_train = train_df[train_df["rating"] > 0].copy()
real_mask = train_df["rating"] > 0
real_ratings = train_df[real_mask].sort_values("timestamp")
cutoff = int(len(real_ratings) * (1 - RECENCY_FRAC))
keep_real = real_ratings.iloc[cutoff:].index
keep_neg = train_df[~real_mask].index
train_df = train_df.loc[keep_real.union(keep_neg)].reset_index(drop=True)
n_dropped = int(real_mask.sum()) - len(keep_real)
log.info(f"Recency filter: kept {RECENCY_FRAC:.0%} of real ratings, dropped {n_dropped} oldest")

if POST_RECENCY_NEG_RESAMPLE:
    kept_real_train = train_df[train_df["rating"] > 0].reset_index(drop=True)
    kept_pos = kept_real_train[kept_real_train["label"] == 1].reset_index(drop=True)
    old_easy_neg = int((train_df["rating"] == 0).sum())
    num_easy_neg = int(round(len(kept_pos) * POST_RECENCY_EASY_NEG_PER_POS))
    if len(kept_pos) == 0 or num_easy_neg == 0:
        train_df = kept_real_train.copy()
        log.info("Post-recency neg resample: no kept positives; dropped existing easy negatives")
    else:
        from scipy.sparse import csr_matrix

        rng = np.random.RandomState(SEED)
        full_rows = full_real_train["userId"].values.astype(np.int64)
        full_cols = full_real_train["movieId"].values.astype(np.int64)
        rated_matrix = csr_matrix(
            (np.ones(len(full_rows), dtype=bool), (full_rows, full_cols)),
            shape=(num_users, num_items),
        )
        item_first_seen = np.full(num_items, int(full_real_train["timestamp"].max()), dtype=np.int64)
        first_seen_series = full_real_train.groupby("movieId")["timestamp"].min()
        item_first_seen[first_seen_series.index.values.astype(np.int64)] = first_seen_series.values.astype(np.int64)

        anchor_idx = rng.randint(0, len(kept_pos), size=num_easy_neg)
        anchor_rows = kept_pos.iloc[anchor_idx]
        neg_users = anchor_rows["userId"].values.astype(np.int64)
        neg_timestamps = anchor_rows["timestamp"].values.astype(np.int64)
        neg_items = rng.randint(0, num_items, size=num_easy_neg)
        for _ in range(10):
            is_rated = np.array(rated_matrix[neg_users, neg_items]).flatten().astype(bool)
            is_rated |= item_first_seen[neg_items] > neg_timestamps
            n_bad = int(is_rated.sum())
            if n_bad == 0:
                break
            neg_items[is_rated] = rng.randint(0, num_items, size=n_bad)

        new_easy_neg = pd.DataFrame({
            "userId": neg_users,
            "movieId": neg_items,
            "rating": 0.0,
            "timestamp": neg_timestamps,
            "label": 0,
        })
        train_df = pd.concat([kept_real_train, new_easy_neg], ignore_index=True)
        log.info(
            f"Post-recency neg resample: replaced {old_easy_neg} easy neg with {num_easy_neg} "
            f"({POST_RECENCY_EASY_NEG_PER_POS:.2f} per kept positive)"
        )


# ═══════════════════════════════════════════════════════════════════
# DATASET — precompute features on GPU (lookup tables stay compact)
# ═══════════════════════════════════════════════════════════════════

# Compact lookup tables on GPU (indexed by user/item ID, not per-sample)
_user_histories_gpu = torch.from_numpy(user_histories).to(DEVICE)       # (num_users, L)
_user_hist_ratings_gpu = torch.from_numpy(user_hist_ratings).to(DEVICE) # (num_users, L)
_user_hist_ts_gpu = torch.from_numpy(user_hist_timestamps).to(DEVICE)   # (num_users, L)
_item_histories_gpu = torch.from_numpy(item_histories).to(DEVICE)       # (num_items, IL)
_item_hist_ratings_gpu = torch.from_numpy(item_hist_ratings).to(DEVICE) # (num_items, IL)
_item_hist_ts_gpu = torch.from_numpy(item_hist_timestamps).to(DEVICE)   # (num_items, IL)
_movie_genres_gpu = torch.from_numpy(movie_genres).to(DEVICE)           # (num_items, G)
_genome_gpu = torch.from_numpy(genome_matrix).to(DEVICE)               # (num_items, GENOME_DIM)
_has_genome_gpu = torch.from_numpy(has_genome).to(DEVICE)              # (num_items,)
_user_genome_gpu = torch.from_numpy(user_genome).to(DEVICE).half()     # (num_users, GENOME_DIM) fp16
_has_user_genome_gpu = torch.from_numpy(has_user_genome).to(DEVICE)    # (num_users,)

def _build_gpu_tensors(df):
    """Precompute per-sample features on GPU. Histories/genres looked up at training time."""
    uids = df["userId"].values.astype(np.int64)
    mids = df["movieId"].values.astype(np.int64)
    labels = df["label"].values.astype(np.float32)
    sample_ts = df["timestamp"].values.astype(np.int32)
    ts_norm = ((df["timestamp"].values - ts_min) / ts_range).astype(np.float32)
    ug_dot = np.sum(user_genre_affinity[uids] * movie_genres[mids], axis=1).astype(np.float32)
    movie_age = ts_norm - movie_year[mids]
    dense = np.column_stack([
        ts_norm,
        user_hist_bins[uids], user_count_norm[uids],
        item_hist_bins[mids], item_count_norm[mids],
        ug_dot,
        movie_year[mids], movie_genre_count[mids], movie_age,
    ]).astype(np.float32)
    return (
        torch.from_numpy(uids).to(DEVICE),
        torch.from_numpy(mids).to(DEVICE),
        torch.from_numpy(dense).to(DEVICE),
        torch.from_numpy(labels).to(DEVICE),
        torch.from_numpy(sample_ts).to(DEVICE),
    )

log.info("Precomputing training tensors on GPU...")
train_uids, train_mids, train_dense, train_labels, train_ts = _build_gpu_tensors(train_df)
n_train = len(train_labels)
n_batches_per_epoch = n_train // BATCH_SIZE


# ═══════════════════════════════════════════════════════════════════
# EVAL SET — val rated items + sampled unrated negatives
# ═══════════════════════════════════════════════════════════════════

# Val positives: rated >= 4 in val set (label=1)
# Val hard negatives: rated < 4 in val set (label=0)
# Val easy negatives: random unrated movies (label=0), 1 per val positive
_val_pos_mask = val_df["label"] == 1
_val_pos = val_df[_val_pos_mask]
_val_hard_neg = val_df[~_val_pos_mask]
_n_val_pos = len(_val_pos)

# Build user -> all items (train + val) for eval neg sampling exclusion
_val_user_all = {}
for uid, items in user_all_items.items():
    _val_user_all[uid] = set(items)
for uid, group in val_df.groupby("userId"):
    if uid not in _val_user_all:
        _val_user_all[uid] = set()
    _val_user_all[uid].update(group["movieId"].values)

# Sample fixed easy negatives for val positives
_eval_rng = np.random.RandomState(42)
_easy_neg_users = _val_pos["userId"].values.astype(np.int64)
_easy_neg_items = np.empty(_n_val_pos, dtype=np.int64)
for i in range(_n_val_pos):
    uid = _easy_neg_users[i]
    rated = _val_user_all.get(uid, set())
    mid = _eval_rng.randint(0, num_items)
    while mid in rated:
        mid = _eval_rng.randint(0, num_items)
    _easy_neg_items[i] = mid

# Combine: val positives + val hard negatives + easy negatives
_eval_users = np.concatenate([
    _val_pos["userId"].values, _val_hard_neg["userId"].values, _easy_neg_users
]).astype(np.int64)
_eval_items = np.concatenate([
    _val_pos["movieId"].values, _val_hard_neg["movieId"].values, _easy_neg_items
]).astype(np.int64)
_eval_ts = np.concatenate([
    _val_pos["timestamp"].values, _val_hard_neg["timestamp"].values, _val_pos["timestamp"].values
])
_eval_ts_norm = ((_eval_ts - ts_min) / ts_range).astype(np.float32)
_eval_labels = np.concatenate([
    np.ones(_n_val_pos), np.zeros(len(_val_hard_neg)), np.zeros(_n_val_pos)
])

log.info("Precomputing eval tensors on GPU...")
_eval_uids_t, _eval_mids_t, _eval_dense_t, _, _eval_ts_t = _build_gpu_tensors(
    pd.DataFrame({
        "userId": _eval_users, "movieId": _eval_items,
        "timestamp": _eval_ts, "label": _eval_labels,
    })
)
n_eval = len(_eval_users)
log.info(f"Eval set: {_n_val_pos} pos + {len(_val_hard_neg)} hard neg + {_n_val_pos} easy neg = {n_eval} total")


def run_eval():
    """Evaluate AUC on validation set."""
    model.eval()
    eval_batch = BATCH_SIZE * 2
    all_scores = []
    with torch.no_grad():
        for start in range(0, n_eval, eval_batch):
            end = min(start + eval_batch, n_eval)
            u = _eval_uids_t[start:end]
            m = _eval_mids_t[start:end]
            logits = model(
                u, m, _eval_dense_t[start:end],
                _user_histories_gpu[u], _user_hist_ratings_gpu[u], _user_hist_ts_gpu[u], _eval_ts_t[start:end],
                _movie_genres_gpu[m],
                _item_histories_gpu[m], _item_hist_ratings_gpu[m], _item_hist_ts_gpu[m],
                _genome_gpu[m], _has_genome_gpu[m],
                _user_genome_gpu[u], _has_user_genome_gpu[u],
            )
            all_scores.append(torch.sigmoid(logits).cpu().numpy())

    scores = np.concatenate(all_scores)
    return evaluate(_eval_labels, scores)


# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════

class TxBlock(nn.Module):
    """Pre-LN transformer encoder block: MHA + FFN with GELU."""
    def __init__(self, d_model, num_heads, ffn_ratio, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # x: (B, L, D), attn_mask: (B, 1, L, L) bool, True = keep
        B, L, D = x.shape
        h = self.ln1(x)
        Q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D_h)
        K = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # SDPA expects attn_mask as additive float OR bool (True = keep). Use bool form.
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o_proj(out)
        x = x + self.attn_drop(out)
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.ffn_drop(h)
        return x


def _sinusoidal_pos_embed(seq_len, d_model, device):
    """Standard sinusoidal positional embedding (Vaswani 2017)."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d_model, device=device, dtype=torch.float32).unsqueeze(0)
    div = torch.exp(-(i // 2) * 2 * np.log(10000.0) / d_model)
    angles = pos * div
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class DLRM(nn.Module):
    def __init__(self):
        super().__init__()
        D = EMBED_DIM

        self.user_embed = nn.Embedding(num_users, D)
        self.item_embed = nn.Embedding(num_items, D)
        self.hist_embed = nn.Embedding(num_items + 1, D, padding_idx=PAD_IDX)
        self.rater_embed = nn.Embedding(num_users + 1, D, padding_idx=USER_PAD_IDX)
        self.zero_item_hist = ITEM_HIST_MODE == "off"
        self.use_causal_sa = USE_CAUSAL_SA
        self.user_hist_mode = USER_HIST_MODE
        self.item_hist_mode = ITEM_HIST_MODE

        # Rating projection: scalar rating → D-dim vector
        self.rating_proj = nn.Linear(1, D)

        # Lightweight causal self-attention on history (single head, no FFN)
        self.hist_q = nn.Linear(D, D, bias=False)
        self.hist_k = nn.Linear(D, D, bias=False)
        self.hist_v = nn.Linear(D, D, bias=False)

        # DIN: target-item-aware attention over history
        self.din_attn = nn.Sequential(
            nn.Linear(3 * 2 * D, 64),  # 2*D per position (item+rating), 3 groups
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Item-side DIN: target-user-aware attention over recent raters
        self.item_din_attn = nn.Sequential(
            nn.Linear(3 * 2 * D, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.bottom_mlp = nn.Sequential(
            nn.Linear(NUM_DENSE, 128),
            nn.ReLU(),
            nn.Linear(128, D),
            nn.ReLU(),
        )

        self.genre_proj = nn.Sequential(
            nn.Linear(num_genres, D),
            nn.ReLU(),
        )

        # Tag genome: learned bottleneck compression 1128 → D with gating
        _genome_hidden_dims = [int(h) for h in GENOME_BOTTLENECK_HIDDEN.split(",") if h.strip()]
        _genome_layers = []
        _prev_dim = GENOME_DIM
        for _h in _genome_hidden_dims:
            _genome_layers.append(nn.Linear(_prev_dim, _h))
            _genome_layers.append(nn.ReLU())
            _prev_dim = _h
        _genome_layers.append(nn.Dropout(GENOME_BOTTLENECK_DROPOUT))
        _genome_layers.append(nn.Linear(_prev_dim, D))
        self.genome_proj = nn.Sequential(*_genome_layers)
        # sigmoid gate to blend genome with item_embed fallback
        self.genome_fusion_mode = GENOME_FUSION_MODE
        if GENOME_FUSION_MODE == "legacy":
            self.genome_gate = nn.Linear(D, D)
        elif GENOME_FUSION_MODE == "mask_only":
            self.genome_gate = nn.Linear(1, D)

        # User × item content alignment scalar → small projection.
        self.user_genome_mode = USER_GENOME
        self.user_genome_target = USER_GENOME_TARGET
        if USER_GENOME == "scalar_dot":
            self.ug_dot_proj = nn.Linear(1, D)

        # Squeeze-and-Excitation: learn per-field importance weights
        # SE field count and cross_dim depend on TX flags:
        #   - TX_ENCODER_LAYERS=0: 7 fields, cross_dim = 7*D (unchanged baseline)
        #   - TX_ENCODER_LAYERS>0, USER_HIST_MODE="off": tx replaces user_hist → still 7 fields
        #   - TX_ENCODER_LAYERS>0, USER_HIST_MODE!="off", TX_BYPASS_SE=0: 8 fields through SE → cross_dim = 8*D
        #   - TX_ENCODER_LAYERS>0, USER_HIST_MODE!="off", TX_BYPASS_SE=1: 7 fields SE + tx concat → cross_dim = 8*D
        if TX_ENCODER_LAYERS > 0 and USER_HIST_MODE != "off" and TX_BYPASS_SE == 0:
            n_se_fields = 8
        else:
            n_se_fields = 7
        if TX_ENCODER_LAYERS > 0 and USER_HIST_MODE != "off":
            cross_dim = 8 * D  # 7 base fields + tx_field (either inside SE or concatenated)
        else:
            cross_dim = 7 * D  # baseline OR tx replaces user_hist
        self.n_se_fields = n_se_fields
        self.se = nn.Sequential(
            nn.Linear(n_se_fields, n_se_fields * 4),
            nn.ReLU(),
            nn.Linear(n_se_fields * 4, n_se_fields),
            nn.Sigmoid(),
        )

        # Top MLP: SE-reweighted fields → prediction
        self.top_mlp = nn.Sequential(
            nn.Linear(cross_dim, 256),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

        # ─── Transformer encoder (off-state byte-equivalent guard) ────
        # All TX modules constructed here AFTER _init_weights() per learning #11.
        # Off-state (TX_ENCODER_LAYERS=0): no parameters added, no RNG drawn.
        if TX_ENCODER_LAYERS > 0:
            self.tx_layers = TX_ENCODER_LAYERS
            self.tx_pool = TX_POOL
            self.tx_pos_mode = TX_POS
            self.tx_bypass_se = TX_BYPASS_SE
            self.tx_replaces_user_hist = (USER_HIST_MODE == "off")
            if TX_POS == "learned":
                self.tx_pos_e = nn.Parameter(torch.zeros(HISTORY_LEN, D))
                nn.init.normal_(self.tx_pos_e, 0.0, 0.02)
            else:
                # Sinusoidal: register as buffer (deterministic, not learned)
                self.register_buffer("tx_pos_e", _sinusoidal_pos_embed(HISTORY_LEN, D, torch.device("cpu")))
            self.tx_in_drop = nn.Dropout(TX_DROPOUT)
            self.tx_blocks = nn.ModuleList([
                TxBlock(D, TX_ENCODER_HEADS, TX_FFN_RATIO, TX_DROPOUT)
                for _ in range(TX_ENCODER_LAYERS)
            ])
            # Re-init TX block linears with xavier (consistent with rest of model)
            for blk in self.tx_blocks:
                for m in blk.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            self.tx_out_proj = nn.Linear(D, D)
            nn.init.kaiming_uniform_(self.tx_out_proj.weight, a=5 ** 0.5)
            if self.tx_out_proj.bias is not None:
                nn.init.zeros_(self.tx_out_proj.bias)
            self.tx_out_ln = nn.LayerNorm(D)
            # Gate as bare Parameter (NOT nn.Linear) per learning #11.
            if TX_GATE_INIT == 0.0:
                self.tx_gate = nn.Parameter(torch.zeros(D))
            else:
                self.tx_gate = nn.Parameter(torch.full((D,), TX_GATE_INIT))
            if TX_POOL == "cls":
                self.tx_cls = nn.Parameter(torch.zeros(1, 1, D))
                nn.init.normal_(self.tx_cls, 0.0, 0.02)
            elif TX_POOL == "target":
                # Cross-attention pool: target item_e is the 1-token query.
                # Use a learned linear projection for query alignment.
                self.tx_target_q = nn.Linear(D, D)
                nn.init.xavier_uniform_(self.tx_target_q.weight)
                if self.tx_target_q.bias is not None:
                    nn.init.zeros_(self.tx_target_q.bias)
                self.tx_target_k = nn.Linear(D, D)
                nn.init.xavier_uniform_(self.tx_target_k.weight)
                if self.tx_target_k.bias is not None:
                    nn.init.zeros_(self.tx_target_k.bias)
                self.tx_target_v = nn.Linear(D, D)
                nn.init.xavier_uniform_(self.tx_target_v.weight)
                if self.tx_target_v.bias is not None:
                    nn.init.zeros_(self.tx_target_v.bias)
        else:
            self.tx_layers = 0
            self.tx_replaces_user_hist = False
            self.tx_bypass_se = 0

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def forward(self, user_id, movie_id, dense, history, hist_ratings, hist_timestamps, sample_timestamps,
                genres, item_hist, item_hist_ratings, item_hist_timestamps, genome_raw, has_genome_mask,
                user_genome_raw=None, has_user_genome_mask=None):
        user_e = nn.functional.dropout(self.user_embed(user_id), EMBED_DROPOUT, self.training)
        item_e = nn.functional.dropout(self.item_embed(movie_id), EMBED_DROPOUT, self.training)
        hist_valid = history != PAD_IDX
        if USER_HIST_CONTEXT == "causal_masked":
            hist_valid = hist_valid & (hist_timestamps < sample_timestamps.unsqueeze(1))
        item_hist_valid = item_hist != USER_PAD_IDX
        if ITEM_HIST_CONTEXT == "causal_masked":
            item_hist_valid = item_hist_valid & (item_hist_timestamps < sample_timestamps.unsqueeze(1))
        hist_mask = hist_valid.unsqueeze(-1)
        item_hist_mask = item_hist_valid.unsqueeze(-1)

        # --- User-side: causal self-attention + DIN ---
        raw_hist_item_e = self.hist_embed(history)                     # (B, L, D) — save raw
        hist_item_e = raw_hist_item_e
        hist_rat_e = self.rating_proj(hist_ratings.unsqueeze(-1))      # (B, L, D)
        if self.use_causal_sa:
            Q = self.hist_q(hist_item_e)                                   # (B, L, D)
            K = self.hist_k(hist_item_e)                                   # (B, L, D)
            V = self.hist_v(hist_item_e)                                   # (B, L, D)
            scale = Q.size(-1) ** 0.5
            causal_scores = torch.bmm(Q, K.transpose(1, 2)) / scale       # (B, L, L)
            L = history.size(1)
            causal_valid = torch.tril(torch.ones((L, L), device=history.device, dtype=torch.bool))
            key_valid = hist_valid.unsqueeze(1).expand(-1, L, -1)
            causal_weights = _masked_softmax(causal_scores, causal_valid.unsqueeze(0) & key_valid, dim=-1)
            hist_item_e = torch.bmm(causal_weights, V) + raw_hist_item_e

        if self.user_hist_mode == "din":
            hist_e_raw = torch.cat([hist_item_e, hist_rat_e], dim=-1)     # (B, L, 2D)
            target_e = torch.cat([item_e, item_e], dim=-1)                # (B, 2D) — no rating for target
            target_exp = target_e.unsqueeze(1).expand_as(hist_e_raw)      # (B, L, 2D)
            attn_in = torch.cat([hist_e_raw, target_exp, hist_e_raw * target_exp], dim=-1)
            attn_w = self.din_attn(attn_in).squeeze(-1)                   # (B, L)
            attn_w = _masked_softmax(attn_w, hist_valid, dim=-1).unsqueeze(-1)
            user_hist_e = (hist_item_e * attn_w).sum(dim=1)               # (B, D)
        elif self.user_hist_mode == "mean":
            denom = hist_mask.sum(dim=1).clamp_min(1)
            user_hist_e = (hist_item_e * hist_mask).sum(dim=1) / denom
        elif self.user_hist_mode == "rating":
            rating_weights = hist_ratings.unsqueeze(-1) * hist_mask
            denom = rating_weights.sum(dim=1).clamp_min(1e-6)
            user_hist_e = (hist_item_e * rating_weights).sum(dim=1) / denom
        elif self.user_hist_mode == "off":
            user_hist_e = torch.zeros_like(item_e)
        else:
            raise ValueError(f"Unknown USER_HIST_MODE: {self.user_hist_mode}")

        # --- Item-side DIN: attention over users who rated this item ---
        rater_e = nn.functional.dropout(self.rater_embed(item_hist), EMBED_DROPOUT, self.training)  # (B, IL, D)
        if self.item_hist_mode == "din":
            rater_rat_e = self.rating_proj(item_hist_ratings.unsqueeze(-1)) # (B, IL, D)
            rater_e_raw = torch.cat([rater_e, rater_rat_e], dim=-1)       # (B, IL, 2D)
            query_e = torch.cat([user_e, user_e], dim=-1)                 # (B, 2D)
            query_exp = query_e.unsqueeze(1).expand_as(rater_e_raw)       # (B, IL, 2D)
            item_attn_in = torch.cat([rater_e_raw, query_exp, rater_e_raw * query_exp], dim=-1)
            item_attn_w = self.item_din_attn(item_attn_in).squeeze(-1)    # (B, IL)
            item_attn_w = _masked_softmax(item_attn_w, item_hist_valid, dim=-1).unsqueeze(-1)
            item_hist_e = (rater_e * item_attn_w).sum(dim=1)              # (B, D)
        elif self.item_hist_mode == "mean":
            denom = item_hist_mask.sum(dim=1).clamp_min(1)
            item_hist_e = (rater_e * item_hist_mask).sum(dim=1) / denom
        elif self.item_hist_mode == "off":
            item_hist_e = torch.zeros_like(item_e)
        else:
            raise ValueError(f"Unknown ITEM_HIST_MODE: {self.item_hist_mode}")

        dense_e = self.bottom_mlp(dense)
        genre_e = self.genre_proj(genres)

        # Tag genome: learned compression with gating
        genome_e = self.genome_proj(genome_raw)                           # (B, D)
        if self.genome_fusion_mode == "legacy":
            gate_input = genome_e
        elif self.genome_fusion_mode == "mask_only":
            gate_input = has_genome_mask.unsqueeze(-1)                    # (B, 1)
        gate = torch.sigmoid(self.genome_gate(gate_input))                # (B, D)
        genome_field = gate * genome_e + (1 - gate) * item_e.detach()     # blend genome with item fallback

        # User × item content alignment: dot(user_genome, item_genome) / GENOME_DIM
        # → Linear(1, D) → added to either dense_e or genome_field (per USER_GENOME_TARGET).
        if self.user_genome_mode == "scalar_dot":
            ug_raw = user_genome_raw.float()
            dot = (ug_raw * genome_raw).sum(dim=-1, keepdim=True) / GENOME_DIM  # (B, 1)
            valid = has_user_genome_mask.unsqueeze(-1) * has_genome_mask.unsqueeze(-1)
            ug_field = self.ug_dot_proj(dot * valid)                      # (B, D)
            if self.user_genome_target == "dense_e":
                dense_e = dense_e + ug_field
            else:  # genome_field
                genome_field = genome_field + ug_field

        # Transformer encoder over user history (optional, off by default)
        tx_field = None
        if self.tx_layers > 0:
            B, L = history.shape
            D = raw_hist_item_e.size(-1)
            # Input: raw token + rating + positional, scaled by sqrt(D)
            pos = self.tx_pos_e[:L].unsqueeze(0)                              # (1, L, D)
            tx_x = (raw_hist_item_e + hist_rat_e + pos) * (D ** 0.5)
            if self.tx_pool == "cls":
                cls_tok = self.tx_cls.expand(B, -1, -1)                       # (B, 1, D)
                tx_x = torch.cat([cls_tok, tx_x], dim=1)                      # (B, L+1, D)
                # CLS attends to everything; everything else only to valid history
                cls_valid = torch.ones(B, 1, device=history.device, dtype=torch.bool)
                tx_valid = torch.cat([cls_valid, hist_valid], dim=1)          # (B, L+1)
            else:
                tx_valid = hist_valid                                         # (B, L)
            tx_x = self.tx_in_drop(tx_x)
            # Build attention mask: causal + key-padding (True = keep).
            tx_L = tx_x.size(1)
            causal = torch.tril(torch.ones((tx_L, tx_L), device=history.device, dtype=torch.bool))
            key_keep = tx_valid.unsqueeze(1).expand(-1, tx_L, -1)             # (B, L, L)
            attn_mask = (causal.unsqueeze(0) & key_keep).unsqueeze(1)         # (B, 1, L, L)
            for blk in self.tx_blocks:
                tx_x = blk(tx_x, attn_mask)
            # Pool
            if self.tx_pool == "rating":
                # Rating-weighted mean over history positions (CLS not present here).
                rating_weights = hist_ratings.unsqueeze(-1) * hist_valid.unsqueeze(-1)
                denom = rating_weights.sum(dim=1).clamp_min(1e-6)
                tx_pooled = (tx_x * rating_weights).sum(dim=1) / denom        # (B, D)
            elif self.tx_pool == "cls":
                tx_pooled = tx_x[:, 0]                                        # (B, D)
            else:  # target — cross-attention with target item_e as 1-token query
                Q = self.tx_target_q(item_e).unsqueeze(1)                     # (B, 1, D)
                K = self.tx_target_k(tx_x)                                    # (B, L, D)
                V = self.tx_target_v(tx_x)                                    # (B, L, D)
                scale = D ** 0.5
                scores = torch.bmm(Q, K.transpose(1, 2)).squeeze(1) / scale   # (B, L)
                pool_valid = tx_valid                                         # (B, L) — same support as tx_x
                w = _masked_softmax(scores, pool_valid, dim=-1).unsqueeze(-1) # (B, L, 1)
                tx_pooled = (V * w).sum(dim=1)                                # (B, D)
            tx_pooled = self.tx_out_proj(tx_pooled)
            tx_pooled = self.tx_out_ln(tx_pooled)
            tx_field = tx_pooled * self.tx_gate                               # zero at init → off-state byte-eq

        # Field assembly: handle TX integration modes
        if self.tx_layers > 0 and self.tx_replaces_user_hist:
            user_hist_e = tx_field
            tx_field_for_se = None  # already merged
        else:
            tx_field_for_se = tx_field

        if tx_field_for_se is not None and self.tx_bypass_se == 0:
            # TX as 8th field inside SE
            fields = torch.stack([user_e, item_e, user_hist_e, item_hist_e, dense_e, genre_e, genome_field, tx_field_for_se], dim=1)  # (B, 8, D)
            se_input = fields.mean(dim=2)
            se_weights = self.se(se_input).unsqueeze(-1)
            fields = fields * se_weights
            x0 = fields.reshape(fields.size(0), -1)                            # (B, 8*D)
        elif tx_field_for_se is not None and self.tx_bypass_se == 1:
            # 7 fields through SE, tx_field concatenated AFTER SE flatten
            fields = torch.stack([user_e, item_e, user_hist_e, item_hist_e, dense_e, genre_e, genome_field], dim=1)  # (B, 7, D)
            se_input = fields.mean(dim=2)
            se_weights = self.se(se_input).unsqueeze(-1)
            fields = fields * se_weights
            x0 = torch.cat([fields.reshape(fields.size(0), -1), tx_field_for_se], dim=-1)  # (B, 7*D + D)
        else:
            # Baseline 7-field path (TX_ENCODER_LAYERS=0, or TX replaces user_hist)
            fields = torch.stack([user_e, item_e, user_hist_e, item_hist_e, dense_e, genre_e, genome_field], dim=1)  # (B, 7, D)
            se_input = fields.mean(dim=2)
            se_weights = self.se(se_input).unsqueeze(-1)
            fields = fields * se_weights
            x0 = fields.reshape(fields.size(0), -1)                            # (B, 7*D) — flatten

        return self.top_mlp(x0).squeeze(-1)


model = DLRM().to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
log.info(f"Parameters: {num_params / 1e6:.1f}M | Genres: {num_genres}")

# Compile model for CUDA graphs / kernel fusion
if DEVICE.type == "cuda" and USE_TORCH_COMPILE:
    model = torch.compile(model)


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP — BCE loss
# ═══════════════════════════════════════════════════════════════════

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss()
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

training_start = time.time()
peak_memory_mb = 0.0
best_auc = 0.0
best_state = None
evals_without_improvement = 0
global_step = 0
eval_every_steps = max(n_batches_per_epoch // 3, 1)  # eval ~3x per epoch
epoch = 0

while True:
    model.train()
    epoch_loss = 0.0

    # Shuffle training data each epoch via random permutation
    perm = torch.randperm(n_train, device=DEVICE)

    for i in range(n_batches_per_epoch):
        idx = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        u_batch = train_uids[idx]
        m_batch = train_mids[idx]
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                u_batch, m_batch, train_dense[idx],
                _user_histories_gpu[u_batch], _user_hist_ratings_gpu[u_batch],
                _user_hist_ts_gpu[u_batch], train_ts[idx],
                _movie_genres_gpu[m_batch],
                _item_histories_gpu[m_batch], _item_hist_ratings_gpu[m_batch],
                _item_hist_ts_gpu[m_batch],
                _genome_gpu[m_batch], _has_genome_gpu[m_batch],
                _user_genome_gpu[u_batch], _has_user_genome_gpu[u_batch],
            )
            # Label smoothing: soft targets
            smooth_labels = train_labels[idx] * (1.0 - 2 * LABEL_SMOOTH) + LABEL_SMOOTH
            loss = criterion(logits, smooth_labels)

        scaler.scale(loss / ACCUM_STEPS).backward()
        if (i + 1) % ACCUM_STEPS == 0:
            if GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if WARMUP_STEPS > 0:
                _opt_step = (i + 1) // ACCUM_STEPS + epoch * (n_batches_per_epoch // ACCUM_STEPS)
                if _opt_step < WARMUP_STEPS:
                    _scale = (_opt_step + 1) / WARMUP_STEPS
                    for _pg in optimizer.param_groups:
                        _pg["lr"] = LR * _scale
                else:
                    for _pg in optimizer.param_groups:
                        _pg["lr"] = LR
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item()
        global_step += 1

        # Sub-epoch evaluation
        if global_step % eval_every_steps == 0:
            avg_loss = epoch_loss / (i + 1)
            elapsed = time.time() - training_start
            val_metrics = run_eval()
            val_auc = val_metrics["auc"]
            improved = "***" if val_auc > best_auc else ""
            log.info(f"Step {global_step:6d} | Loss {avg_loss:.4f} | Val AUC {val_auc:.4f} {improved} | {elapsed:.0f}s")

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            if evals_without_improvement >= PATIENCE:
                log.info(f"Early stopping: no improvement for {PATIENCE} evals (best AUC: {best_auc:.4f})")
                break
            model.train()

    epoch += 1
    if evals_without_improvement >= PATIENCE:
        break

# Record peak CUDA memory after training
if DEVICE.type == "cuda":
    peak_memory_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)

training_seconds = time.time() - training_start


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

if best_state is not None:
    model.load_state_dict(best_state)
    log.info(f"Restored best model (AUC: {best_auc:.4f})")

metrics = run_eval()
total_seconds = time.time() - total_start

print_summary(metrics, training_seconds, total_seconds, peak_memory_mb, num_params, stats)
