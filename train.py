#!/usr/bin/env python3
"""
Linear baseline for hybrid engagement prediction on MovieLens.

Restart from scratch (apr28). Same input features and prediction goals as the
legacy DLRM at legacy/train.py, but the model itself is a single Linear head:

    concat(all features) -> Linear(in, 1) -> sigmoid

No hidden layers, no attention, no MLP. The point is a clean low ceiling
to build up from. See README.md and program.md for the broader plan.

Label scheme (unchanged from legacy):
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
# Same feature set as the legacy DLRM:
#   - genre multi-hot, rating histograms, user/item count
#   - user/item history sequences (last K + ratings)
#   - user-genre affinity, timestamp, year, genre count
#   - tag genome (1128) + per-user genome profile (1128)

_cache_key = hashlib.md5(json.dumps({
    "feature_version": 5,                 # match legacy cache version
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
    user_hist_bins = _c["user_hist_bins"]
    item_hist_bins = _c["item_hist_bins"]
    user_count_norm = _c["user_count_norm"]
    item_count_norm = _c["item_count_norm"]
    user_histories = _c["user_histories"]
    user_hist_ratings = _c["user_hist_ratings"]
    item_histories = _c["item_histories"]
    item_hist_ratings = _c["item_hist_ratings"]
    user_genre_affinity = _c["user_genre_affinity"]
    ts_min = float(_c["ts_min"])
    ts_range = float(_c["ts_range"])
    movie_year = _c["movie_year"]
    movie_genre_count = _c["movie_genre_count"]
    genome_matrix = _c["genome_matrix"]
    has_genome = _c["has_genome"]
    user_genome = _c["user_genome"]
    has_user_genome = _c["has_user_genome"]
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

    # Rating histograms (5-bin per user/item)
    _uids = real_train["userId"].values
    _mids = real_train["movieId"].values
    _ratings = real_train["rating"].values
    _bins = np.clip((_ratings - 0.01) // 1, 0, 4).astype(np.intp)
    user_hist_bins = np.zeros((num_users, 5), dtype=np.float32)
    item_hist_bins = np.zeros((num_items, 5), dtype=np.float32)
    np.add.at(user_hist_bins, (_uids, _bins), 1)
    np.add.at(item_hist_bins, (_mids, _bins), 1)
    user_count = user_hist_bins.sum(axis=1)
    item_count = item_hist_bins.sum(axis=1)
    user_hist_bins /= (user_count[:, None] + 1e-8)
    item_hist_bins /= (item_count[:, None] + 1e-8)
    user_count_norm = (user_count - user_count.mean()) / (user_count.std() + 1e-8)
    item_count_norm = (item_count - item_count.mean()) / (item_count.std() + 1e-8)
    del _bins

    # User/item history sequences
    PAD_IDX = num_items
    USER_PAD_IDX = num_users
    def _build_history(ids, targets, ratings, timestamps, n_entities, pad_idx, max_len):
        sort_idx = np.lexsort((timestamps, ids))
        s_ids, s_targets = ids[sort_idx], targets[sort_idx]
        s_ratings = ratings[sort_idx].astype(np.float32) / 5.0
        hist = np.full((n_entities, max_len), pad_idx, dtype=np.int64)
        hist_rat = np.zeros((n_entities, max_len), dtype=np.float32)
        boundaries = np.where(np.diff(s_ids) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [len(s_ids)]])
        for s, e in zip(starts, ends):
            eid = s_ids[s]
            length = min(e - s, max_len)
            hist[eid, -length:] = s_targets[e - length:e]
            hist_rat[eid, -length:] = s_ratings[e - length:e]
        return hist, hist_rat

    _ts = real_train["timestamp"].values
    user_histories, user_hist_ratings = _build_history(_uids, _mids, _ratings, _ts,
                                                       num_users, PAD_IDX, HISTORY_LEN)
    item_histories, item_hist_ratings = _build_history(_mids, _uids, _ratings, _ts,
                                                       num_items, USER_PAD_IDX, ITEM_HIST_LEN)

    # User-genre affinity
    user_genre_affinity = np.zeros((num_users, num_genres), dtype=np.float32)
    user_genre_count = np.zeros((num_users, num_genres), dtype=np.float32)
    _gi = movie_genres[_mids]
    np.add.at(user_genre_affinity, _uids, _ratings[:, None].astype(np.float32) * _gi)
    np.add.at(user_genre_count, _uids, _gi)
    mask = user_genre_count > 0
    user_genre_affinity[mask] /= user_genre_count[mask]
    mu = user_genre_affinity[mask].mean()
    sigma = user_genre_affinity[mask].std() + 1e-8
    user_genre_affinity[mask] = (user_genre_affinity[mask] - mu) / sigma
    del _uids, _mids, _ratings, _ts, _gi

    # Timestamp normalization
    ts_min = float(real_train["timestamp"].min())
    ts_range = float(real_train["timestamp"].max() - ts_min) + 1.0

    # Movie release year
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

    # Genre count per movie
    movie_genre_count = movie_genres.sum(axis=1).astype(np.float32)
    movie_genre_count = (movie_genre_count - movie_genre_count.mean()) / (movie_genre_count.std() + 1e-8)

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
            has_genome = np.zeros(num_items, dtype=np.float32)
            _gdf["mapped_mid"] = _gdf["movieId"].map(_movie_map)
            _gdf = _gdf.dropna(subset=["mapped_mid"])
            _gdf["mapped_mid"] = _gdf["mapped_mid"].astype(int)
            for mid, group in _gdf.groupby("mapped_mid"):
                if mid < num_items:
                    tag_ids = group["tagId"].values.astype(int) - 1
                    genome_matrix[mid, tag_ids] = group["relevance"].values.astype(np.float32)
                    has_genome[mid] = 1.0
            log.info(f"Tag genome: {int(has_genome.sum())}/{num_items} movies "
                     f"({has_genome.mean()*100:.1f}%)")
        else:
            genome_matrix = np.zeros((num_items, 1128), dtype=np.float32)
            has_genome = np.zeros(num_items, dtype=np.float32)
    else:
        genome_matrix = np.zeros((num_items, 1128), dtype=np.float32)
        has_genome = np.zeros(num_items, dtype=np.float32)
    GENOME_DIM = genome_matrix.shape[1]

    # Per-user genome profile (mean of genome over user's high-rated genome-having items)
    log.info("Computing user genome profiles...")
    user_genome = np.zeros((num_users, GENOME_DIM), dtype=np.float32)
    _ug_count = np.zeros(num_users, dtype=np.float32)
    _hr = real_train["rating"].values >= 4.0
    _hu = real_train["userId"].values[_hr].astype(np.int64)
    _hm = real_train["movieId"].values[_hr].astype(np.int64)
    _hg = has_genome[_hm] > 0
    _hu = _hu[_hg]; _hm = _hm[_hg]
    for i in range(0, len(_hu), 500_000):
        u_chunk = _hu[i:i + 500_000]
        m_chunk = _hm[i:i + 500_000]
        np.add.at(user_genome, u_chunk, genome_matrix[m_chunk])
        np.add.at(_ug_count, u_chunk, 1.0)
    _has_ug = _ug_count > 0
    user_genome[_has_ug] /= _ug_count[_has_ug, None]
    has_user_genome = _has_ug.astype(np.float32)
    del _hr, _hu, _hm, _hg, _has_ug, _ug_count

    np.savez_compressed(_cache_path,
        movie_genres=movie_genres, user_hist_bins=user_hist_bins,
        item_hist_bins=item_hist_bins, user_count_norm=user_count_norm,
        item_count_norm=item_count_norm, user_histories=user_histories,
        user_hist_ratings=user_hist_ratings,
        user_hist_timestamps=np.zeros((num_users, HISTORY_LEN), dtype=np.int32),
        item_histories=item_histories, item_hist_ratings=item_hist_ratings,
        item_hist_timestamps=np.zeros((num_items, ITEM_HIST_LEN), dtype=np.int32),
        user_genre_affinity=user_genre_affinity,
        ts_min=np.array(ts_min), ts_range=np.array(ts_range),
        movie_year=movie_year, movie_genre_count=movie_genre_count,
        genome_matrix=genome_matrix, has_genome=has_genome,
        user_genome=user_genome, has_user_genome=has_user_genome,
    )
    log.info(f"Features cached to {_cache_path.name}")

PAD_IDX = num_items
USER_PAD_IDX = num_users

# Move lookup tables to GPU
_user_hist_t = torch.from_numpy(user_histories).to(DEVICE)
_user_hist_rat_t = torch.from_numpy(user_hist_ratings).to(DEVICE)
_item_hist_t = torch.from_numpy(item_histories).to(DEVICE)
_item_hist_rat_t = torch.from_numpy(item_hist_ratings).to(DEVICE)
_movie_genres_t = torch.from_numpy(movie_genres).to(DEVICE)
_genome_t = torch.from_numpy(genome_matrix).to(DEVICE)
_user_genome_t = torch.from_numpy(user_genome).to(DEVICE)

# Per-sample dense features (computed once for train + eval)
def _build_sample_tensors(df):
    uids = df["userId"].values.astype(np.int64)
    mids = df["movieId"].values.astype(np.int64)
    labels = df["label"].values.astype(np.float32)
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
    return (torch.from_numpy(uids).to(DEVICE),
            torch.from_numpy(mids).to(DEVICE),
            torch.from_numpy(dense).to(DEVICE),
            torch.from_numpy(labels).to(DEVICE))

log.info("Precomputing training tensors...")
train_uids, train_mids, train_dense, train_labels = _build_sample_tensors(train_df)
n_train = len(train_labels)
DENSE_DIM = train_dense.shape[1]

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
eval_uids, eval_mids, eval_dense, eval_labels_t = _build_sample_tensors(_eval_df)
n_eval = len(eval_uids)
log.info(f"Eval set: {_n_val_pos} pos + {len(_val_hard_neg)} hard neg + {_n_val_pos} easy neg = {n_eval}")


# ═══════════════════════════════════════════════════════════════════
# MODEL — single Linear head on concatenated features
# ═══════════════════════════════════════════════════════════════════

class LinearBaseline(nn.Module):
    """concat(all features) -> Linear(in, 1) -> sigmoid."""
    def __init__(self, num_users, num_items, num_genres, dense_dim, genome_dim, embed_dim):
        super().__init__()
        D = embed_dim
        # Embeddings (+1 row for PAD)
        self.user_embed = nn.Embedding(num_users + 1, D, padding_idx=num_users)
        self.item_embed = nn.Embedding(num_items + 1, D, padding_idx=num_items)
        # Single Linear projection of genre multi-hot (no hidden layer; this is not an MLP)
        self.genre_proj = nn.Linear(num_genres, D, bias=False)
        # Concat dim:
        #   user_e (D) + item_e (D)
        #   user_hist mean (D) + user_hist mean rating (1)
        #   item_hist mean (D) + item_hist mean rating (1)
        #   genre_proj (D)
        #   dense (dense_dim)
        #   genome (genome_dim) + user_genome (genome_dim)
        self.in_dim = 5 * D + 2 + dense_dim + 2 * genome_dim
        self.head = nn.Linear(self.in_dim, 1)

    def forward(self, uids, mids, dense):
        u_e = self.user_embed(uids)
        i_e = self.item_embed(mids)

        # User history: mean-pool item_embed over valid (non-PAD) positions
        u_hist = _user_hist_t[uids]                       # (B, L)
        u_hist_rat = _user_hist_rat_t[uids]               # (B, L)
        u_hist_e = self.item_embed(u_hist)                # (B, L, D)
        u_valid = (u_hist != PAD_IDX).float().unsqueeze(-1)
        u_count = u_valid.sum(dim=1).clamp(min=1.0)
        u_hist_pool = (u_hist_e * u_valid).sum(dim=1) / u_count                # (B, D)
        u_hist_rat_mean = (u_hist_rat * u_valid.squeeze(-1)).sum(dim=1) / u_count.squeeze(-1)
        u_hist_rat_mean = u_hist_rat_mean.unsqueeze(-1)                        # (B, 1)

        # Item history: mean-pool user_embed over valid raters
        i_hist = _item_hist_t[mids]                       # (B, IL)
        i_hist_rat = _item_hist_rat_t[mids]               # (B, IL)
        i_hist_e = self.user_embed(i_hist)                # (B, IL, D)
        i_valid = (i_hist != USER_PAD_IDX).float().unsqueeze(-1)
        i_count = i_valid.sum(dim=1).clamp(min=1.0)
        i_hist_pool = (i_hist_e * i_valid).sum(dim=1) / i_count
        i_hist_rat_mean = (i_hist_rat * i_valid.squeeze(-1)).sum(dim=1) / i_count.squeeze(-1)
        i_hist_rat_mean = i_hist_rat_mean.unsqueeze(-1)

        # Genres + genome lookups
        genre_e = self.genre_proj(_movie_genres_t[mids])  # (B, D)
        genome_e = _genome_t[mids]                        # (B, GENOME_DIM)
        u_genome = _user_genome_t[uids]                   # (B, GENOME_DIM)

        x = torch.cat([
            u_e, i_e,
            u_hist_pool, u_hist_rat_mean,
            i_hist_pool, i_hist_rat_mean,
            genre_e,
            dense,
            genome_e, u_genome,
        ], dim=-1)
        return self.head(x).squeeze(-1)


model = LinearBaseline(num_users, num_items, num_genres, DENSE_DIM, GENOME_DIM, EMBED_DIM).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
log.info(f"Parameters: {n_params/1e6:.1f}M | dense_dim={DENSE_DIM} | genome_dim={GENOME_DIM} | "
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
            logits = model(eval_uids[s:e], eval_mids[s:e], eval_dense[s:e])
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
    # Shuffle training indices
    perm = torch.randperm(n_train, device=DEVICE)
    for b in range(n_batches_per_epoch):
        idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        logits = model(train_uids[idx], train_mids[idx], train_dense[idx])
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

# Final eval (uses best weights)
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
