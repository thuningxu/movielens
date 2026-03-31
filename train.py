#!/usr/bin/env python3
"""
DLRM baseline for MovieLens pointwise recommendation.
This is the file you modify for experimentation.

Features:
  - Sparse: userId, movieId, genre (multi-hot), user history sequence
  - Dense: normalized timestamp, user stats (count, mean, std), item stats (count, mean, std)

Model: DLRM with bottom MLP, embedding interaction (pairwise dots), top MLP.
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
from prepare import load_data, evaluate, print_summary, TIME_BUDGET

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Force unbuffered stdout so logs appear in real time when redirected
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
log = logging.getLogger("train")

# ─── Configuration ──────────────────────────────────────────────────
DATASET = os.environ.get("DATASET", "ml-1m")
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-5
EMBED_DIM = 16
HISTORY_LEN = 50
NUM_DENSE = 7  # 1 timestamp + 3 user stats + 3 item stats
EVAL_EVERY = 5  # evaluate on val set every N epochs
PATIENCE = 20  # early stop after N evals with no improvement

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
data = load_data(DATASET)
train_df, val_df = data["train"], data["val"]
movies_df, stats = data["movies"], data["stats"]
num_users, num_items = stats["num_users"], stats["num_items"]
log.info(f"Dataset: {DATASET} | Users: {num_users} | Items: {num_items} | "
         f"Train: {stats['num_train']} | Val: {stats['num_val']} | Pos rate: {stats['pos_rate']:.2%}")


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (modify freely)
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
user_stats = np.zeros((num_users, 3), dtype=np.float32)  # count, mean_rating, std_rating
item_stats = np.zeros((num_items, 3), dtype=np.float32)

for uid, group in train_df.groupby("userId"):
    r = group["rating"].values.astype(np.float32)
    user_stats[uid] = [len(r), r.mean(), r.std() if len(r) > 1 else 0.0]

for mid, group in train_df.groupby("movieId"):
    r = group["rating"].values.astype(np.float32)
    item_stats[mid] = [len(r), r.mean(), r.std() if len(r) > 1 else 0.0]

# Normalize to zero mean, unit variance
for arr in [user_stats, item_stats]:
    for col in range(arr.shape[1]):
        mu = arr[:, col].mean()
        sigma = arr[:, col].std() + 1e-8
        arr[:, col] = (arr[:, col] - mu) / sigma

# 3. User history sequences (last HISTORY_LEN items from training data)
PAD_IDX = num_items  # padding token for history embedding
user_histories = np.full((num_users, HISTORY_LEN), PAD_IDX, dtype=np.int64)
for uid, group in train_df.groupby("userId"):
    items = group["movieId"].values
    seq = items[-HISTORY_LEN:]
    user_histories[uid, -len(seq):] = seq

# 4. Timestamp normalization parameters (from training data)
ts_min = float(train_df["timestamp"].min())
ts_range = float(train_df["timestamp"].max() - ts_min) + 1.0


# ═══════════════════════════════════════════════════════════════════
# DATASET (modify freely)
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
        dense = np.concatenate([
            [self.ts_norm[idx]],
            user_stats[uid],
            item_stats[mid],
        ])
        return (
            uid,                        # int64
            mid,                        # int64
            dense,                      # float32[NUM_DENSE]
            user_histories[uid],        # int64[HISTORY_LEN]
            movie_genres[mid],          # float32[num_genres]
            self.labels[idx],           # float32
        )


train_loader = DataLoader(
    RecDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=False, drop_last=True,
)
val_loader = DataLoader(
    RecDataset(val_df), batch_size=BATCH_SIZE * 2, shuffle=False,
    num_workers=0,
)


# ═══════════════════════════════════════════════════════════════════
# MODEL (modify freely)
# ═══════════════════════════════════════════════════════════════════

class DLRM(nn.Module):
    """
    Deep Learning Recommendation Model.

    Architecture:
      - Bottom MLP transforms dense features → embed_dim
      - Sparse features (user, item, genre, history) each → embed_dim
      - Feature interaction: pairwise dot products of all embedding vectors
      - Top MLP: concat of dot products + embedding vectors → logit
    """

    def __init__(self):
        super().__init__()
        D = EMBED_DIM

        # Sparse embeddings
        self.user_embed = nn.Embedding(num_users, D)
        self.item_embed = nn.Embedding(num_items, D)
        self.hist_embed = nn.Embedding(num_items + 1, D, padding_idx=PAD_IDX)

        # Bottom MLP for dense features
        self.bottom_mlp = nn.Sequential(
            nn.Linear(NUM_DENSE, 64),
            nn.ReLU(),
            nn.Linear(64, D),
            nn.ReLU(),
        )

        # Genre projection (multi-hot → embed_dim)
        self.genre_proj = nn.Sequential(
            nn.Linear(num_genres, D),
            nn.ReLU(),
        )

        # Top MLP: 5 vectors → C(5,2)=10 dots + 5*D features
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
        # Embed sparse features
        user_e = self.user_embed(user_id)           # (B, D)
        item_e = self.item_embed(movie_id)           # (B, D)

        # History: embed → masked mean pooling
        hist_e = self.hist_embed(history)             # (B, L, D)
        mask = (history != PAD_IDX).unsqueeze(-1).float()  # (B, L, 1)
        hist_e = (hist_e * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B, D)

        # Dense features through bottom MLP
        dense_e = self.bottom_mlp(dense)              # (B, D)

        # Genre multi-hot through projection
        genre_e = self.genre_proj(genres)             # (B, D)

        # Feature interaction: pairwise dot products
        vecs = [user_e, item_e, hist_e, dense_e, genre_e]
        dots = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dots.append((vecs[i] * vecs[j]).sum(dim=-1, keepdim=True))

        # Concatenate dots + original vectors → top MLP
        out = torch.cat(dots + vecs, dim=-1)
        return self.top_mlp(out).squeeze(-1)


model = DLRM().to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
log.info(f"Parameters: {num_params / 1e6:.1f}M | Genres: {num_genres}")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def run_eval():
    """Evaluate current model on validation set. Returns metrics dict."""
    model.eval()
    all_labels, all_scores = [], []
    with torch.no_grad():
        for user_id, movie_id, dense, history, genres, label in val_loader:
            logits = model(
                user_id.to(DEVICE), movie_id.to(DEVICE), dense.to(DEVICE),
                history.to(DEVICE), genres.to(DEVICE),
            )
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(label.numpy())
    return evaluate(np.concatenate(all_labels), np.concatenate(all_scores))


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP (modify freely)
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

    for user_id, movie_id, dense, history, genres, label in train_loader:
        if time.time() - training_start >= TIME_BUDGET:
            break

        user_id = user_id.to(DEVICE)
        movie_id = movie_id.to(DEVICE)
        dense = dense.to(DEVICE)
        history = history.to(DEVICE)
        genres = genres.to(DEVICE)
        label = label.to(DEVICE)

        logits = model(user_id, movie_id, dense, history, genres)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

        # Track MPS memory
        if DEVICE.type == "mps":
            try:
                mem = torch.mps.current_allocated_memory() / (1024 * 1024)
                peak_memory_mb = max(peak_memory_mb, mem)
            except Exception:
                pass

    epoch += 1
    avg_loss = epoch_loss / max(n_batches, 1)
    elapsed = time.time() - training_start

    # Periodic validation with early stopping
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
# EVALUATION (do not modify — uses prepare.evaluate)
# ═══════════════════════════════════════════════════════════════════

# Restore best model if we have one
if best_state is not None:
    model.load_state_dict(best_state)
    log.info(f"Restored best model (AUC: {best_auc:.4f})")

metrics = run_eval()
total_seconds = time.time() - total_start

print_summary(metrics, training_seconds, total_seconds, peak_memory_mb, num_params, stats)
