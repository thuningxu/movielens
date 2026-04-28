#!/usr/bin/env python3
"""
Linear baseline for hybrid engagement prediction on MovieLens.

Restart from scratch (apr28). Same prediction goals and metric as the
prior project at legacy/, but the model itself is a single Linear head:

    concat(features) -> Linear(in, 1) -> sigmoid

No hidden layers, no attention, no MLP. The features are also stripped
to the bones: only raw IDs, raw history sequences, and pure content
metadata (genres, tag genome, movie year, timestamp). All pre-computed
user/item statistics (rating histograms, counts, user genome profiles,
user-genre affinity) are out — those are aggregations the model can learn
from raw data if they actually help.

Label scheme (unchanged):
  - 1: user rated >= 4 (watched and liked)
  - 0: user rated < 4 (hard negative) OR random unrated movie (easy negative)
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

SEED = int(os.environ.get("SEED", "42"))
np.random.seed(SEED)

import pandas as pd
import torch
import torch.nn as nn

from prepare import load_data_hybrid, evaluate, print_summary

# ─── Logging ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
log = logging.getLogger("train")

# ─── Config (env-overridable) ──────────────────────────────────────
DATASET = os.environ.get("DATASET", "ml-25m")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16384"))
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-5"))
EMBED_DIM = int(os.environ.get("EMBED_DIM", "28"))
HISTORY_LEN = int(os.environ.get("HISTORY_LEN", "100"))
ITEM_HIST_LEN = int(os.environ.get("ITEM_HIST_LEN", "30"))
NEG_RATIO = int(os.environ.get("NEG_RATIO", "1"))
PATIENCE = int(os.environ.get("PATIENCE", "3"))
EVAL_PER_EPOCH = int(os.environ.get("EVAL_PER_EPOCH", "3"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "20"))

# History pooling modes (mean is the byte-equivalent default).
USER_HIST_POOL = os.environ.get("USER_HIST_POOL", "rating_centered")
ITEM_HIST_POOL = os.environ.get("ITEM_HIST_POOL", "rating_centered")
assert USER_HIST_POOL in {"mean", "rating", "rating_centered"}, USER_HIST_POOL
assert ITEM_HIST_POOL in {"mean", "rating", "rating_centered"}, ITEM_HIST_POOL

# Optional add-on fields (each appends one D-dim field to the concat).
USER_HIST_DISLIKE_POOL = int(os.environ.get("USER_HIST_DISLIKE_POOL", "0"))
USER_HIST_LAST_POSITION = int(os.environ.get("USER_HIST_LAST_POSITION", "0"))
ITEM_HIST_LAST_POSITION = int(os.environ.get("ITEM_HIST_LAST_POSITION", "0"))

# Pivot used by rating_centered pool mode. Default 0.6 = 3 stars / 5 (current behavior).
POOL_PIVOT = float(os.environ.get("POOL_PIVOT", "0.6"))

# Optional per-side learnable timestamp decay multiplier on the rating-centered weight.
# Off-state (default 0): no Parameter constructed, no decay logic in forward — byte-equivalent.
# When on, theta is initialized to USER_HIST_DECAY_INIT / ITEM_HIST_DECAY_INIT (default -10
# = near-no-decay; large gradient-starved zone — pick init in [-2, +5] range to actually exercise).
USER_HIST_DECAY = int(os.environ.get("USER_HIST_DECAY", "0"))
ITEM_HIST_DECAY = int(os.environ.get("ITEM_HIST_DECAY", "0"))
USER_HIST_DECAY_INIT = float(os.environ.get("USER_HIST_DECAY_INIT", "-10.0"))
ITEM_HIST_DECAY_INIT = float(os.environ.get("ITEM_HIST_DECAY_INIT", "-10.0"))

# ─── Device ────────────────────────────────────────────────────────
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
log.info(f"Device: {DEVICE} | dim={EMBED_DIM} | hist={HISTORY_LEN}/{ITEM_HIST_LEN} | "
         f"neg_ratio={NEG_RATIO} | lr={LR} | wd={WEIGHT_DECAY}")

t_total_start = time.time()

# ─── Data ──────────────────────────────────────────────────────────
data = load_data_hybrid(DATASET, neg_ratio=NEG_RATIO,
                        train_neg_mode="anchor_pos_catalog")
train_df, val_df = data["train"], data["val"]
movies_df, stats = data["movies"], data["stats"]
user_all_items = data["user_all_items"]
num_users, num_items = stats["num_users"], stats["num_items"]
log.info(f"Dataset: {DATASET} | Users: {num_users} | Items: {num_items} | "
         f"Train: {stats['num_train']} | Val: {stats['num_val']} | "
         f"Pos rate: {stats['pos_rate']:.2%}")

# ─── Feature engineering (with disk cache) ─────────────────────────
# Bones-only feature set:
#   - per movie: genre multi-hot, tag genome (1128), release year
#   - per user: history sequence (last K items + ratings)
#   - per item: history sequence (last K raters + ratings)
#   - per sample: timestamp normalized

_cache_key = hashlib.md5(json.dumps({
    "feature_version": "restart-2",       # restart-2 adds per-position timestamps
    "dataset": DATASET, "neg_ratio": NEG_RATIO,
    "history_len": HISTORY_LEN, "item_hist_len": ITEM_HIST_LEN,
    "num_users": num_users, "num_items": num_items,
}).encode()).hexdigest()[:12]
_cache_path = Path(__file__).parent / "data" / f"features_{_cache_key}.npz"

if _cache_path.exists():
    log.info(f"Loading cached features from {_cache_path.name}...")
    _c = np.load(_cache_path, allow_pickle=True)
    movie_genres = _c["movie_genres"]
    num_genres = movie_genres.shape[1]
    user_histories = _c["user_histories"]
    user_hist_ratings = _c["user_hist_ratings"]
    user_hist_timestamps = _c["user_hist_timestamps"]
    item_histories = _c["item_histories"]
    item_hist_ratings = _c["item_hist_ratings"]
    item_hist_timestamps = _c["item_hist_timestamps"]
    ts_min = float(_c["ts_min"])
    ts_range = float(_c["ts_range"])
    movie_year = _c["movie_year"]
    genome_matrix = _c["genome_matrix"]
    GENOME_DIM = genome_matrix.shape[1]
    del _c
else:
    log.info("Computing features (will cache for next run)...")
    real_train = train_df[train_df["rating"] > 0]

    # Genre multi-hot
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

    # User/item history sequences (raw, last K)
    _uids = real_train["userId"].values
    _mids = real_train["movieId"].values
    _ratings = real_train["rating"].values
    _ts = real_train["timestamp"].values
    PAD_IDX = num_items
    USER_PAD_IDX = num_users
    def _build_history(ids, targets, ratings, timestamps, n_entities, pad_idx, max_len):
        sort_idx = np.lexsort((timestamps, ids))
        s_ids, s_targets = ids[sort_idx], targets[sort_idx]
        s_ratings = ratings[sort_idx].astype(np.float32) / 5.0
        s_ts = timestamps[sort_idx].astype(np.int64)
        hist = np.full((n_entities, max_len), pad_idx, dtype=np.int64)
        hist_rat = np.zeros((n_entities, max_len), dtype=np.float32)
        hist_ts = np.zeros((n_entities, max_len), dtype=np.int32)
        boundaries = np.where(np.diff(s_ids) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(s_ids)]])
        for s, e in zip(starts, ends):
            eid = s_ids[s]
            length = min(e - s, max_len)
            hist[eid, -length:] = s_targets[e - length:e]
            hist_rat[eid, -length:] = s_ratings[e - length:e]
            hist_ts[eid, -length:] = s_ts[e - length:e].astype(np.int32)
        return hist, hist_rat, hist_ts

    user_histories, user_hist_ratings, user_hist_timestamps = _build_history(
        _uids, _mids, _ratings, _ts, num_users, PAD_IDX, HISTORY_LEN)
    item_histories, item_hist_ratings, item_hist_timestamps = _build_history(
        _mids, _uids, _ratings, _ts, num_items, USER_PAD_IDX, ITEM_HIST_LEN)
    del _uids, _mids, _ratings, _ts

    # Timestamp normalization
    ts_min = float(real_train["timestamp"].min())
    ts_range = float(real_train["timestamp"].max() - ts_min) + 1.0

    # Movie release year (parsed from title)
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

    # Tag genome (1128 relevance scores per movie, ml-25m only — zeros otherwise)
    _genome_path = Path(__file__).parent / "data" / DATASET / "genome-scores.csv"
    if _genome_path.exists():
        log.info("Loading tag genome scores...")
        if DATASET == "ml-25m":
            _raw = pd.read_csv(Path(__file__).parent / "data" / DATASET / "ratings.csv")
        elif DATASET == "ml-10m":
            _raw = pd.read_csv(Path(__file__).parent / "data" / "ml-10M100K" / "ratings.dat",
                               sep="::", engine="python",
                               names=["userId", "movieId", "rating", "timestamp"])
        elif DATASET == "ml-1m":
            _raw = pd.read_csv(Path(__file__).parent / "data" / DATASET / "ratings.dat",
                               sep="::", engine="python",
                               names=["userId", "movieId", "rating", "timestamp"])
        else:
            _raw = None
        if _raw is not None:
            _raw = _raw.sort_values("timestamp").reset_index(drop=True)
            _movie_map = {mid: i for i, mid in enumerate(_raw["movieId"].unique())}
            _gdf = pd.read_csv(_genome_path)
            _num_tags = int(_gdf["tagId"].max())
            genome_matrix = np.zeros((num_items, _num_tags), dtype=np.float32)
            _gdf["mapped_mid"] = _gdf["movieId"].map(_movie_map)
            _gdf = _gdf.dropna(subset=["mapped_mid"])
            _gdf["mapped_mid"] = _gdf["mapped_mid"].astype(int)
            _have = 0
            for mid, group in _gdf.groupby("mapped_mid"):
                if mid < num_items:
                    tag_ids = group["tagId"].values.astype(int) - 1
                    genome_matrix[mid, tag_ids] = group["relevance"].values.astype(np.float32)
                    _have += 1
            log.info(f"Tag genome: {_have}/{num_items} movies ({100*_have/num_items:.1f}%)")
        else:
            genome_matrix = np.zeros((num_items, 1128), dtype=np.float32)
    else:
        genome_matrix = np.zeros((num_items, 1128), dtype=np.float32)
    GENOME_DIM = genome_matrix.shape[1]

    np.savez_compressed(_cache_path,
        movie_genres=movie_genres,
        user_histories=user_histories, user_hist_ratings=user_hist_ratings,
        user_hist_timestamps=user_hist_timestamps,
        item_histories=item_histories, item_hist_ratings=item_hist_ratings,
        item_hist_timestamps=item_hist_timestamps,
        ts_min=np.array(ts_min), ts_range=np.array(ts_range),
        movie_year=movie_year,
        genome_matrix=genome_matrix,
    )
    log.info(f"Features cached to {_cache_path.name}")

PAD_IDX = num_items
USER_PAD_IDX = num_users

# Move lookup tables to GPU
_user_hist_t = torch.from_numpy(user_histories).to(DEVICE)
_user_hist_rat_t = torch.from_numpy(user_hist_ratings).to(DEVICE)
_user_hist_ts_t = torch.from_numpy(user_hist_timestamps).to(DEVICE)
_item_hist_t = torch.from_numpy(item_histories).to(DEVICE)
_item_hist_rat_t = torch.from_numpy(item_hist_ratings).to(DEVICE)
_item_hist_ts_t = torch.from_numpy(item_hist_timestamps).to(DEVICE)
_movie_genres_t = torch.from_numpy(movie_genres).to(DEVICE)
_genome_t = torch.from_numpy(genome_matrix).to(DEVICE)
_movie_year_t = torch.from_numpy(movie_year).to(DEVICE)


# Per-sample dense features (just timestamp; year is per-movie, looked up at forward time)
def _build_sample_tensors(df):
    uids = df["userId"].values.astype(np.int64)
    mids = df["movieId"].values.astype(np.int64)
    labels = df["label"].values.astype(np.float32)
    ts_raw = df["timestamp"].values.astype(np.int64)
    ts_norm = ((ts_raw - ts_min) / ts_range).astype(np.float32).reshape(-1, 1)
    return (torch.from_numpy(uids).to(DEVICE),
            torch.from_numpy(mids).to(DEVICE),
            torch.from_numpy(ts_norm).to(DEVICE),
            torch.from_numpy(ts_raw).to(DEVICE),
            torch.from_numpy(labels).to(DEVICE))

log.info("Precomputing training tensors...")
train_uids, train_mids, train_ts, train_ts_raw, train_labels = _build_sample_tensors(train_df)
n_train = len(train_labels)

# ─── Eval set: val pos + val hard neg + sampled easy neg ──────────
_val_pos_mask = val_df["label"] == 1
_val_pos = val_df[_val_pos_mask]
_val_hard_neg = val_df[~_val_pos_mask]
_n_val_pos = len(_val_pos)

_val_user_all = {uid: set(items) for uid, items in user_all_items.items()}
for uid, group in val_df.groupby("userId"):
    _val_user_all.setdefault(uid, set()).update(group["movieId"].values)

_eval_rng = np.random.RandomState(42)
_easy_neg_users = _val_pos["userId"].values.astype(np.int64)
_easy_neg_items = np.empty(_n_val_pos, dtype=np.int64)
for i in range(_n_val_pos):
    rated = _val_user_all.get(_easy_neg_users[i], set())
    mid = _eval_rng.randint(0, num_items)
    while mid in rated:
        mid = _eval_rng.randint(0, num_items)
    _easy_neg_items[i] = mid

_eval_df = pd.DataFrame({
    "userId": np.concatenate([_val_pos["userId"].values, _val_hard_neg["userId"].values, _easy_neg_users]),
    "movieId": np.concatenate([_val_pos["movieId"].values, _val_hard_neg["movieId"].values, _easy_neg_items]),
    "timestamp": np.concatenate([_val_pos["timestamp"].values, _val_hard_neg["timestamp"].values, _val_pos["timestamp"].values]),
    "label": np.concatenate([np.ones(_n_val_pos), np.zeros(len(_val_hard_neg)), np.zeros(_n_val_pos)]),
})
log.info("Precomputing eval tensors...")
eval_uids, eval_mids, eval_ts, eval_ts_raw, eval_labels_t = _build_sample_tensors(_eval_df)
n_eval = len(eval_uids)
log.info(f"Eval set: {_n_val_pos} pos + {len(_val_hard_neg)} hard neg + {_n_val_pos} easy neg = {n_eval}")


# ═══════════════════════════════════════════════════════════════════
# MODEL — single Linear head on concatenated features
# ═══════════════════════════════════════════════════════════════════

def _pool_history(embed, ratings, valid, mode,
                  decay_theta=None, sample_ts=None, hist_ts=None, ts_range=None):
    """Pool a (B, L, D) embedding sequence over valid positions.

    embed:   (B, L, D)
    ratings: (B, L)       — already normalized to [0, 1] (rating / 5.0)
    valid:   (B, L)       — float, 1.0 for non-PAD, 0.0 for PAD
    mode: 'mean' | 'rating' | 'rating_centered'

    Optional decay (only valid with mode='rating_centered'):
      decay_theta: scalar nn.Parameter; rate = softplus(theta) >= 0.
      sample_ts:   (B,)   raw per-sample timestamp (int64 or float).
      hist_ts:     (B, L) raw per-position history timestamp (int32/float).
      ts_range:    scalar, denominator for normalizing time gaps.

    Returns (B, D).
    """
    valid_e = valid.unsqueeze(-1)                                     # (B, L, 1)
    if mode == "mean":
        count = valid.sum(dim=1, keepdim=True).clamp(min=1.0)         # (B, 1)
        return (embed * valid_e).sum(dim=1) / count
    if mode == "rating":
        w = ratings * valid                                           # (B, L)
        w_e = w.unsqueeze(-1)                                         # (B, L, 1)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)            # (B, 1)
        return (embed * w_e).sum(dim=1) / denom
    if mode == "rating_centered":
        # Allow negative weights; normalize by sum of absolute weights.
        w = (ratings - POOL_PIVOT) * valid                            # (B, L)
        if decay_theta is not None:
            # softplus-parameterized non-negative rate; init theta=-10 -> rate≈4.5e-5
            rate = nn.functional.softplus(decay_theta)
            time_gap = (sample_ts.float().unsqueeze(-1) - hist_ts.float()) / ts_range
            time_gap = time_gap.clamp(min=0.0)                        # safety
            recency = torch.exp(-rate * time_gap)                     # (B, L), in (0, 1]
            w = w * recency
        w_e = w.unsqueeze(-1)
        denom = w.abs().sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (embed * w_e).sum(dim=1) / denom
    raise ValueError(f"Unknown pool mode: {mode}")


class LinearBaseline(nn.Module):
    """concat(features) -> Linear(in, 1) -> sigmoid."""
    def __init__(self, num_users, num_items, num_genres, genome_dim, embed_dim):
        super().__init__()
        D = embed_dim
        # Embeddings (+1 row for PAD)
        self.user_embed = nn.Embedding(num_users + 1, D, padding_idx=num_users)
        self.item_embed = nn.Embedding(num_items + 1, D, padding_idx=num_items)
        # Concat dim:
        #   user_e (D) + item_e (D)
        #   user_hist pool (D) + user_hist mean rating (1)
        #   item_hist pool (D) + item_hist mean rating (1)
        #   genre multi-hot (num_genres, raw — no projection)
        #   ts_norm (1) + movie_year (1)
        #   genome (genome_dim)
        # Optional add-on fields (each adds D when its flag is on):
        #   USER_HIST_DISLIKE_POOL: parallel "dislike" pool over user history
        #   USER_HIST_LAST_POSITION: most recent valid item embedding
        #   ITEM_HIST_LAST_POSITION: most recent valid rater embedding
        addon_fields = (USER_HIST_DISLIKE_POOL
                        + USER_HIST_LAST_POSITION
                        + ITEM_HIST_LAST_POSITION)
        self.in_dim = 4 * D + 2 + num_genres + 2 + genome_dim + addon_fields * D
        self.head = nn.Linear(self.in_dim, 1)

        # Optional learnable per-side timestamp-decay rate.
        # Constructed AFTER all default-init layers so the off-state RNG sequence
        # is byte-identical to before. softplus(-10) ≈ 4.5e-5 → near-no-decay at init.
        if USER_HIST_DECAY:
            self.user_decay_theta = nn.Parameter(torch.tensor(USER_HIST_DECAY_INIT))
        if ITEM_HIST_DECAY:
            self.item_decay_theta = nn.Parameter(torch.tensor(ITEM_HIST_DECAY_INIT))

    def forward(self, uids, mids, ts, ts_raw=None):
        u_e = self.user_embed(uids)
        i_e = self.item_embed(mids)

        # User history: pool item_embed over valid (non-PAD) positions
        u_hist = _user_hist_t[uids]                       # (B, L)
        u_hist_rat = _user_hist_rat_t[uids]               # (B, L)
        u_hist_e = self.item_embed(u_hist)                # (B, L, D)
        u_valid = (u_hist != PAD_IDX).float()             # (B, L)
        u_count = u_valid.sum(dim=1).clamp(min=1.0)       # (B,)
        if USER_HIST_DECAY:
            u_hist_ts = _user_hist_ts_t[uids]             # (B, L) int32
            u_hist_pool = _pool_history(
                u_hist_e, u_hist_rat, u_valid, USER_HIST_POOL,
                decay_theta=self.user_decay_theta,
                sample_ts=ts_raw, hist_ts=u_hist_ts, ts_range=ts_range)
        else:
            u_hist_pool = _pool_history(u_hist_e, u_hist_rat, u_valid, USER_HIST_POOL)
        u_hist_rat_mean = (u_hist_rat * u_valid).sum(dim=1) / u_count
        u_hist_rat_mean = u_hist_rat_mean.unsqueeze(-1)                        # (B, 1)

        # Item history: pool user_embed over valid raters
        i_hist = _item_hist_t[mids]                       # (B, IL)
        i_hist_rat = _item_hist_rat_t[mids]               # (B, IL)
        i_hist_e = self.user_embed(i_hist)                # (B, IL, D)
        i_valid = (i_hist != USER_PAD_IDX).float()        # (B, IL)
        i_count = i_valid.sum(dim=1).clamp(min=1.0)
        if ITEM_HIST_DECAY:
            i_hist_ts = _item_hist_ts_t[mids]             # (B, IL) int32
            i_hist_pool = _pool_history(
                i_hist_e, i_hist_rat, i_valid, ITEM_HIST_POOL,
                decay_theta=self.item_decay_theta,
                sample_ts=ts_raw, hist_ts=i_hist_ts, ts_range=ts_range)
        else:
            i_hist_pool = _pool_history(i_hist_e, i_hist_rat, i_valid, ITEM_HIST_POOL)
        i_hist_rat_mean = (i_hist_rat * i_valid).sum(dim=1) / i_count
        i_hist_rat_mean = i_hist_rat_mean.unsqueeze(-1)

        # Item content: raw genre multi-hot + raw genome + year
        genre_raw = _movie_genres_t[mids]                 # (B, num_genres)
        genome_e = _genome_t[mids]                        # (B, GENOME_DIM)
        year = _movie_year_t[mids].unsqueeze(-1)          # (B, 1)

        parts = [
            u_e, i_e,
            u_hist_pool, u_hist_rat_mean,
            i_hist_pool, i_hist_rat_mean,
            genre_raw,
            ts, year,
            genome_e,
        ]

        # Optional add-on fields (only appended if flag is on; off-state is byte-equivalent)
        if USER_HIST_DISLIKE_POOL:
            # Dislike pool: weight by (1 - rating) over valid positions.
            w = (1.0 - u_hist_rat) * u_valid                          # (B, L)
            denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)        # (B, 1)
            dislike_pool = (u_hist_e * w.unsqueeze(-1)).sum(dim=1) / denom
            parts.append(dislike_pool)
        if USER_HIST_LAST_POSITION:
            # Most recent valid user-history slot (PAD on the left, recent on the right).
            last_e = u_hist_e[:, -1, :]                               # (B, D)
            last_valid = u_valid[:, -1].unsqueeze(-1)                 # (B, 1)
            parts.append(last_e * last_valid)
        if ITEM_HIST_LAST_POSITION:
            last_e = i_hist_e[:, -1, :]                               # (B, D)
            last_valid = i_valid[:, -1].unsqueeze(-1)                 # (B, 1)
            parts.append(last_e * last_valid)

        x = torch.cat(parts, dim=-1)
        return self.head(x).squeeze(-1)


model = LinearBaseline(num_users, num_items, num_genres, GENOME_DIM, EMBED_DIM).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
log.info(f"Parameters: {n_params/1e6:.1f}M | genome_dim={GENOME_DIM} | "
         f"in_dim={model.in_dim} | head_params={(model.in_dim + 1)}")

opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.BCEWithLogitsLoss()


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP — patience-based early stopping, sub-epoch eval
# ═══════════════════════════════════════════════════════════════════

def run_eval():
    model.eval()
    eval_batch = BATCH_SIZE * 2
    scores = []
    with torch.no_grad():
        for s in range(0, n_eval, eval_batch):
            e = min(s + eval_batch, n_eval)
            logits = model(eval_uids[s:e], eval_mids[s:e], eval_ts[s:e],
                           ts_raw=eval_ts_raw[s:e])
            scores.append(torch.sigmoid(logits).cpu().numpy())
    model.train()
    scores = np.concatenate(scores)
    labels = eval_labels_t.cpu().numpy()
    return evaluate(labels, scores)


t_train_start = time.time()
best_auc = -1.0
best_state = None
no_improve = 0
n_batches_per_epoch = n_train // BATCH_SIZE
eval_interval = max(1, n_batches_per_epoch // EVAL_PER_EPOCH)
step = 0
loss_sum = 0.0
loss_n = 0
done = False

for epoch in range(MAX_EPOCHS):
    perm = torch.randperm(n_train, device=DEVICE)
    for b in range(n_batches_per_epoch):
        idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        logits = model(train_uids[idx], train_mids[idx], train_ts[idx],
                       ts_raw=train_ts_raw[idx])
        loss = loss_fn(logits, train_labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += float(loss.item())
        loss_n += 1
        step += 1

        if step % eval_interval == 0:
            metrics = run_eval()
            avg_loss = loss_sum / max(1, loss_n)
            elapsed = time.time() - t_train_start
            star = ""
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                star = " ***"
            else:
                no_improve += 1
            log.info(f"Step {step:6d} | Loss {avg_loss:.4f} | Val AUC {metrics['auc']:.4f}{star} | {elapsed:.0f}s")
            loss_sum = 0.0
            loss_n = 0
            if no_improve >= PATIENCE:
                log.info(f"Early stopping: no improvement for {PATIENCE} evals (best AUC: {best_auc:.4f})")
                done = True
                break
    if done:
        break

if best_state is not None:
    model.load_state_dict(best_state)
    log.info(f"Restored best model (AUC: {best_auc:.4f})")

t_train_end = time.time()
final_metrics = run_eval()

if torch.cuda.is_available():
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
else:
    peak_mem_mb = 0.0

print_summary(
    metrics=final_metrics,
    training_seconds=t_train_end - t_train_start,
    total_seconds=time.time() - t_total_start,
    peak_memory_mb=peak_mem_mb,
    num_params=n_params,
    stats=stats,
)
