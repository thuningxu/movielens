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
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from prepare import load_data_hybrid, evaluate, print_summary, TIME_BUDGET

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
WEIGHT_DECAY = 1e-5
EMBED_DIM = 16
HISTORY_LEN = 50
NUM_DENSE = 8  # 1 timestamp + 3 user stats + 3 item stats + 1 user-genre affinity dot
NEG_RATIO = 4  # random unrated negatives per positive in training data
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
data = load_data_hybrid(DATASET, neg_ratio=NEG_RATIO)
train_df, val_df = data["train"], data["val"]
movies_df, stats = data["movies"], data["stats"]
user_all_items = data["user_all_items"]
num_users, num_items = stats["num_users"], stats["num_items"]
log.info(f"Dataset: {DATASET} | Users: {num_users} | Items: {num_items} | "
         f"Train: {stats['num_train']} | Val: {stats['num_val']} | "
         f"Pos rate: {stats['pos_rate']:.2%}")


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

# 2. User and item statistics (computed on rated items in train only, before neg augmentation)
# Filter to real ratings (rating > 0) for stats computation
real_train = train_df[train_df["rating"] > 0]
user_stats = np.zeros((num_users, 3), dtype=np.float32)
item_stats = np.zeros((num_items, 3), dtype=np.float32)

for uid, group in real_train.groupby("userId"):
    r = group["rating"].values.astype(np.float32)
    user_stats[uid] = [len(r), r.mean(), r.std() if len(r) > 1 else 0.0]

for mid, group in real_train.groupby("movieId"):
    r = group["rating"].values.astype(np.float32)
    item_stats[mid] = [len(r), r.mean(), r.std() if len(r) > 1 else 0.0]

for arr in [user_stats, item_stats]:
    for col in range(arr.shape[1]):
        mu = arr[:, col].mean()
        sigma = arr[:, col].std() + 1e-8
        arr[:, col] = (arr[:, col] - mu) / sigma

# 3. User history sequences (last HISTORY_LEN rated items from training data)
PAD_IDX = num_items
user_histories = np.full((num_users, HISTORY_LEN), PAD_IDX, dtype=np.int64)
for uid, group in real_train.groupby("userId"):
    items = group.sort_values("timestamp")["movieId"].values
    seq = items[-HISTORY_LEN:]
    user_histories[uid, -len(seq):] = seq

# 4. User-genre affinity: average rating per genre for each user
user_genre_affinity = np.zeros((num_users, num_genres), dtype=np.float32)
user_genre_count = np.zeros((num_users, num_genres), dtype=np.float32)
for _, row in real_train.iterrows():
    uid = int(row["userId"])
    mid = int(row["movieId"])
    rating = row["rating"]
    for g_idx in range(num_genres):
        if movie_genres[mid, g_idx] > 0:
            user_genre_affinity[uid, g_idx] += rating
            user_genre_count[uid, g_idx] += 1
mask = user_genre_count > 0
user_genre_affinity[mask] /= user_genre_count[mask]
# Normalize
mu = user_genre_affinity[mask].mean()
sigma = user_genre_affinity[mask].std() + 1e-8
user_genre_affinity[mask] = (user_genre_affinity[mask] - mu) / sigma

# 5. Timestamp normalization (from real ratings)
ts_min = float(real_train["timestamp"].min())
ts_range = float(real_train["timestamp"].max() - ts_min) + 1.0


# ═══════════════════════════════════════════════════════════════════
# DATASET — pointwise BCE
# ═══════════════════════════════════════════════════════════════════

class RecDataset(Dataset):
    def __init__(self, df):
        self.user_ids = df["userId"].values.astype(np.int64)
        self.movie_ids = df["movieId"].values.astype(np.int64)
        self.labels = df["label"].values.astype(np.float32)
        self.ts_norm = ((df["timestamp"].values - ts_min) / ts_range).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        mid = self.movie_ids[idx]
        # User-genre affinity dot product with movie's genre vector
        ug_dot = np.float32(np.dot(user_genre_affinity[uid], movie_genres[mid]))
        dense = np.concatenate([
            [self.ts_norm[idx]],
            user_stats[uid],
            item_stats[mid],
            [ug_dot],
        ])
        return (
            uid,
            mid,
            dense,
            user_histories[uid],
            movie_genres[mid],
            self.labels[idx],
        )


train_loader = DataLoader(
    RecDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=False, drop_last=True,
)


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

_eval_dense = np.stack([
    np.concatenate([
        [_eval_ts_norm[i]],
        user_stats[_eval_users[i]],
        item_stats[_eval_items[i]],
        [np.dot(user_genre_affinity[_eval_users[i]], movie_genres[_eval_items[i]])],
    ])
    for i in range(len(_eval_users))
]).astype(np.float32)
_eval_histories = user_histories[_eval_users]
_eval_genres = movie_genres[_eval_items]

log.info(f"Eval set: {_n_val_pos} pos + {len(_val_hard_neg)} hard neg + {_n_val_pos} easy neg = {len(_eval_users)} total")


def run_eval():
    """Evaluate AUC on validation set."""
    model.eval()
    eval_batch = BATCH_SIZE * 2
    all_scores = []
    with torch.no_grad():
        for start in range(0, len(_eval_users), eval_batch):
            end = min(start + eval_batch, len(_eval_users))
            u = torch.from_numpy(_eval_users[start:end]).to(DEVICE)
            m = torch.from_numpy(_eval_items[start:end]).to(DEVICE)
            d = torch.from_numpy(_eval_dense[start:end]).to(DEVICE)
            h = torch.from_numpy(_eval_histories[start:end]).to(DEVICE)
            g = torch.from_numpy(_eval_genres[start:end]).to(DEVICE)
            logits = model(u, m, d, h, g)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())

    scores = np.concatenate(all_scores)
    return evaluate(_eval_labels, scores)


# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════

class DLRM(nn.Module):
    def __init__(self):
        super().__init__()
        D = EMBED_DIM

        self.user_embed = nn.Embedding(num_users, D)
        self.item_embed = nn.Embedding(num_items, D)
        self.hist_embed = nn.Embedding(num_items + 1, D, padding_idx=PAD_IDX)

        # DIN: target-item-aware attention over history
        self.din_attn = nn.Sequential(
            nn.Linear(3 * D, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

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

        hist_e_raw = self.hist_embed(history)                          # (B, L, D)
        target_e = item_e.unsqueeze(1).expand_as(hist_e_raw)          # (B, L, D)
        attn_in = torch.cat([hist_e_raw, target_e, hist_e_raw * target_e], dim=-1)
        attn_w = self.din_attn(attn_in).squeeze(-1)                   # (B, L)
        attn_w = attn_w.masked_fill(history == PAD_IDX, -1e9)
        attn_w = torch.softmax(attn_w, dim=-1).unsqueeze(-1)          # (B, L, 1)
        hist_e = (hist_e_raw * attn_w).sum(dim=1)                     # (B, D)

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
# TRAINING LOOP — BCE loss
# ═══════════════════════════════════════════════════════════════════

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss()

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

    for uid, mid, dense, history, genres, label in train_loader:
        if time.time() - training_start >= TIME_BUDGET:
            break

        uid = uid.to(DEVICE)
        mid = mid.to(DEVICE)
        dense = dense.to(DEVICE)
        history = history.to(DEVICE)
        genres = genres.to(DEVICE)
        label = label.to(DEVICE)

        logits = model(uid, mid, dense, history, genres)
        loss = criterion(logits, label)

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
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

if best_state is not None:
    model.load_state_dict(best_state)
    log.info(f"Restored best model (AUC: {best_auc:.4f})")

metrics = run_eval()
total_seconds = time.time() - total_start

print_summary(metrics, training_seconds, total_seconds, peak_memory_mb, num_params, stats)
