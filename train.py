#!/usr/bin/env python3
"""
DLRM with implicit feedback + BPR (Bayesian Personalised Ranking) loss.

Changes from the explicit-rating baseline:
  - All ratings treated as positive implicit feedback
  - BPR pairwise loss: -log(sigmoid(score_pos - score_neg))
  - Online negative sampling: random unrated items as negatives
  - Evaluation: AUC over (val positives vs unrated items)
"""

import copy
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from prepare import load_data_implicit, evaluate, print_summary, TIME_BUDGET

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

# ─── Configuration ──────────────────────────────────────────────────
DATASET = os.environ.get("DATASET", "ml-1m")
BATCH_SIZE = 8192
LR = 1e-4
WEIGHT_DECAY = 1e-4
EMBED_DIM = 16
HISTORY_LEN = 50
NUM_DENSE = 7  # 1 timestamp + 3 user stats + 3 item stats
EVAL_EVERY = 1
PATIENCE = 10

# ─── Device ─────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
log.info(f"Device: {DEVICE}")

# ─── Load Data ──────────────────────────────────────────────────────
total_start = time.time()
data = load_data_implicit(DATASET)
train_df, val_df = data["train"], data["val"]
movies_df, stats = data["movies"], data["stats"]
all_item_ids = data["all_item_ids"]
user_pos_items = data["user_pos_items"]
num_users, num_items = stats["num_users"], stats["num_items"]
log.info(f"Dataset: {DATASET} | Users: {num_users} | Items: {num_items} | "
         f"Train: {stats['num_train']} | Val: {stats['num_val']} | Implicit feedback mode")


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

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

# 2. User and item statistics (computed on train only)
user_stats = np.zeros((num_users, 3), dtype=np.float32)
item_stats = np.zeros((num_items, 3), dtype=np.float32)

for uid, group in train_df.groupby("userId"):
    r = group["rating"].values.astype(np.float32)
    user_stats[uid] = [len(r), r.mean(), r.std() if len(r) > 1 else 0.0]

for mid, group in train_df.groupby("movieId"):
    r = group["rating"].values.astype(np.float32)
    item_stats[mid] = [len(r), r.mean(), r.std() if len(r) > 1 else 0.0]

for arr in [user_stats, item_stats]:
    for col in range(arr.shape[1]):
        mu = arr[:, col].mean()
        sigma = arr[:, col].std() + 1e-8
        arr[:, col] = (arr[:, col] - mu) / sigma

# 3. User history sequences (last HISTORY_LEN items from training data)
PAD_IDX = num_items
user_histories = np.full((num_users, HISTORY_LEN), PAD_IDX, dtype=np.int64)
for uid, group in train_df.groupby("userId"):
    items = group["movieId"].values
    seq = items[-HISTORY_LEN:]
    user_histories[uid, -len(seq):] = seq

# 4. Timestamp normalization parameters (from training data)
ts_min = float(train_df["timestamp"].min())
ts_range = float(train_df["timestamp"].max() - ts_min) + 1.0

# 5. Pre-compute user positive item sets as sorted arrays for fast negative sampling
user_pos_arrays = {}
for uid, pos_set in user_pos_items.items():
    user_pos_arrays[uid] = np.array(sorted(pos_set), dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════
# DATASET — BPR triplet sampling
# ═══════════════════════════════════════════════════════════════════

class BPRDataset(Dataset):
    """Returns (user, pos_item, neg_item) triples with online negative sampling."""
    def __init__(self, df):
        self.user_ids = df["userId"].values.astype(np.int64)
        self.movie_ids = df["movieId"].values.astype(np.int64)
        self.ts_norm = ((df["timestamp"].values - ts_min) / ts_range).astype(np.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        pos_mid = self.movie_ids[idx]

        # Sample negative: random item user hasn't interacted with
        neg_mid = np.random.randint(0, num_items)
        pos_set = user_pos_items.get(uid, set())
        # Rejection sampling (fast when num_items >> num_user_interactions)
        while neg_mid in pos_set:
            neg_mid = np.random.randint(0, num_items)

        ts = self.ts_norm[idx]

        # Dense features for positive item
        pos_dense = np.concatenate([
            [ts],
            user_stats[uid],
            item_stats[pos_mid],
        ])

        # Dense features for negative item
        neg_dense = np.concatenate([
            [ts],
            user_stats[uid],
            item_stats[neg_mid],
        ])

        return (
            uid,                          # int64
            pos_mid,                      # int64
            neg_mid,                      # int64
            pos_dense,                    # float32[NUM_DENSE]
            neg_dense,                    # float32[NUM_DENSE]
            user_histories[uid],          # int64[HISTORY_LEN]
            movie_genres[pos_mid],        # float32[num_genres]
            movie_genres[neg_mid],        # float32[num_genres]
        )


train_loader = DataLoader(
    BPRDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=False, drop_last=True,
)


# ═══════════════════════════════════════════════════════════════════
# MODEL — same DLRM architecture, scores individual (user, item) pairs
# ═══════════════════════════════════════════════════════════════════

class DLRM(nn.Module):
    """
    Deep Learning Recommendation Model.

    Architecture:
      - Bottom MLP transforms dense features -> embed_dim
      - Sparse features (user, item, genre, history) each -> embed_dim
      - Feature interaction: pairwise dot products of all embedding vectors
      - Top MLP: concat of dot products + embedding vectors -> logit
    """

    def __init__(self):
        super().__init__()
        D = EMBED_DIM

        self.user_embed = nn.Embedding(num_users, D)
        self.item_embed = nn.Embedding(num_items, D)
        self.hist_embed = nn.Embedding(num_items + 1, D, padding_idx=PAD_IDX)

        self.bottom_mlp = nn.Sequential(
            nn.Linear(NUM_DENSE, 64),
            nn.ReLU(),
            nn.Linear(64, D),
            nn.ReLU(),
        )

        self.genre_proj = nn.Sequential(
            nn.Linear(num_genres, D),
            nn.ReLU(),
        )

        # Top MLP: 5 vectors -> C(5,2)=10 dots + 5*D features
        self.top_mlp = nn.Sequential(
            nn.Linear(10 + 5 * D, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self._init_weights()

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

    def forward(self, user_id, movie_id, dense, history, genres):
        user_e = self.user_embed(user_id)
        item_e = self.item_embed(movie_id)

        hist_e = self.hist_embed(history)
        mask = (history != PAD_IDX).unsqueeze(-1).float()
        hist_e = (hist_e * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        dense_e = self.bottom_mlp(dense)
        genre_e = self.genre_proj(genres)

        vecs = [user_e, item_e, hist_e, dense_e, genre_e]
        dots = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dots.append((vecs[i] * vecs[j]).sum(dim=-1, keepdim=True))

        out = torch.cat(dots + vecs, dim=-1)
        return self.top_mlp(out).squeeze(-1)


model = DLRM().to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
log.info(f"Parameters: {num_params / 1e6:.1f}M | Genres: {num_genres}")


# ═══════════════════════════════════════════════════════════════════
# EVALUATION — AUC on val set (positives = val interactions, negatives = unrated items)
# Pre-compute fixed eval negatives for stable AUC across epochs
# ═══════════════════════════════════════════════════════════════════

# Build user -> all positive items (train + val) for eval negative sampling
_val_user_all_pos = {}
for uid, pos_set in user_pos_items.items():
    _val_user_all_pos[uid] = set(pos_set)
for uid, group in val_df.groupby("userId"):
    if uid not in _val_user_all_pos:
        _val_user_all_pos[uid] = set()
    _val_user_all_pos[uid].update(group["movieId"].values)

# Collect val positives
_eval_val_users = val_df["userId"].values.astype(np.int64)
_eval_val_items = val_df["movieId"].values.astype(np.int64)
_eval_val_ts = ((val_df["timestamp"].values - ts_min) / ts_range).astype(np.float32)
_n_val = len(_eval_val_users)

# Pre-sample fixed negatives (deterministic seed for reproducibility)
_eval_rng = np.random.RandomState(42)
_eval_neg_items = np.empty(_n_val, dtype=np.int64)
for i in range(_n_val):
    uid = _eval_val_users[i]
    pos_set = _val_user_all_pos.get(uid, set())
    neg = _eval_rng.randint(0, num_items)
    while neg in pos_set:
        neg = _eval_rng.randint(0, num_items)
    _eval_neg_items[i] = neg

# Pre-build all eval arrays
_eval_all_users = np.concatenate([_eval_val_users, _eval_val_users])
_eval_all_items = np.concatenate([_eval_val_items, _eval_neg_items])
_eval_all_ts = np.concatenate([_eval_val_ts, _eval_val_ts])
_eval_all_labels = np.concatenate([np.ones(_n_val), np.zeros(_n_val)])

_eval_all_dense = np.stack([
    np.concatenate([[_eval_all_ts[i]], user_stats[_eval_all_users[i]], item_stats[_eval_all_items[i]]])
    for i in range(len(_eval_all_users))
]).astype(np.float32)

_eval_all_histories = user_histories[_eval_all_users]
_eval_all_genres = movie_genres[_eval_all_items]

log.info(f"Eval set prepared: {_n_val} pos + {_n_val} neg = {2*_n_val} samples")


def run_eval():
    """Evaluate AUC on validation set with pre-computed fixed negatives."""
    model.eval()
    eval_batch = BATCH_SIZE * 2
    all_scores = []
    with torch.no_grad():
        for start in range(0, len(_eval_all_users), eval_batch):
            end = min(start + eval_batch, len(_eval_all_users))
            u = torch.from_numpy(_eval_all_users[start:end]).to(DEVICE)
            m = torch.from_numpy(_eval_all_items[start:end]).to(DEVICE)
            d = torch.from_numpy(_eval_all_dense[start:end]).to(DEVICE)
            h = torch.from_numpy(_eval_all_histories[start:end]).to(DEVICE)
            g = torch.from_numpy(_eval_all_genres[start:end]).to(DEVICE)
            logits = model(u, m, d, h, g)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())

    scores = np.concatenate(all_scores)
    return evaluate(_eval_all_labels, scores)


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP — BPR pairwise loss
# ═══════════════════════════════════════════════════════════════════

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

training_start = time.time()
peak_memory_mb = 0.0
epoch = 0
best_auc = 0.0
best_state = None
evals_without_improvement = 0

while True:
    elapsed = time.time() - training_start
    if elapsed >= TIME_BUDGET:
        break

    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for uid, pos_mid, neg_mid, pos_dense, neg_dense, history, pos_genres, neg_genres in train_loader:
        if time.time() - training_start >= TIME_BUDGET:
            break

        uid = uid.to(DEVICE)
        history = history.to(DEVICE)

        # Score positive items
        pos_score = model(
            uid, pos_mid.to(DEVICE), pos_dense.to(DEVICE),
            history, pos_genres.to(DEVICE),
        )

        # Score negative items
        neg_score = model(
            uid, neg_mid.to(DEVICE), neg_dense.to(DEVICE),
            history, neg_genres.to(DEVICE),
        )

        # BPR loss: -log(sigmoid(pos_score - neg_score))
        loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

        if DEVICE.type == "mps":
            try:
                mem = torch.mps.current_allocated_memory() / (1024 * 1024)
                peak_memory_mb = max(peak_memory_mb, mem)
            except Exception:
                pass

    epoch += 1
    avg_loss = epoch_loss / max(n_batches, 1)
    elapsed = time.time() - training_start

    if epoch % EVAL_EVERY == 0:
        val_metrics = run_eval()
        val_auc = val_metrics["auc"]
        improved = "***" if val_auc > best_auc else ""
        log.info(f"Epoch {epoch:4d} | Loss {avg_loss:.4f} | Val AUC {val_auc:.4f} {improved} | {elapsed:.0f}s / {TIME_BUDGET}s")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            evals_without_improvement = 0
        else:
            evals_without_improvement += 1

        if evals_without_improvement >= PATIENCE:
            log.info(f"Early stopping: no improvement for {PATIENCE} evals (best AUC: {best_auc:.4f})")
            break
    else:
        log.info(f"Epoch {epoch:4d} | Loss {avg_loss:.4f} | {elapsed:.0f}s / {TIME_BUDGET}s")

training_seconds = time.time() - training_start


# ═══════════════════════════════════════════════════════════════════
# EVALUATION (uses prepare.evaluate)
# ═══════════════════════════════════════════════════════════════════

if best_state is not None:
    model.load_state_dict(best_state)
    log.info(f"Restored best model (AUC: {best_auc:.4f})")

metrics = run_eval()
total_seconds = time.time() - total_start

print_summary(metrics, training_seconds, total_seconds, peak_memory_mb, num_params, stats)
