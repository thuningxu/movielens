#!/usr/bin/env python3
"""
HSTU stub for generative recommendation on MovieLens.

Status: scaffolding only. End-to-end pipeline runs on ml-100k but the
model is a placeholder (embedding sum + linear head). Replace the
PlaceholderModel with the actual HSTU stack to make this attempt meaningful.

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
import sys
import time
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
EMBED_DIM = int(os.environ.get("EMBED_DIM", "64"))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "4"))
NUM_HEADS = int(os.environ.get("NUM_HEADS", "4"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "200"))         # per-user context length
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
NUM_RATING_BUCKETS = 10                                 # 0.5★ → bucket 0, 5★ → bucket 9

LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "5"))


# ─── Data ───────────────────────────────────────────────────────────
def build_user_sequences(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Group ratings by user, sort by timestamp, return per-user event arrays.

    Each event row is (movieId, rating_bucket, timestamp). Easy negatives
    (rating == 0 sentinel) are excluded from the history; only real ratings
    contribute to the sequence representation.
    """
    real = df[df["rating"] > 0].copy()
    real = real.sort_values(["userId", "timestamp"])
    real["rating_bucket"] = ((real["rating"].clip(0.5, 5.0) - 0.5) * 2).astype(np.int64)
    sequences = {}
    for uid, group in real.groupby("userId", sort=False):
        sequences[int(uid)] = np.stack(
            [group["movieId"].to_numpy(np.int64),
             group["rating_bucket"].to_numpy(np.int64),
             group["timestamp"].to_numpy(np.int64)],
            axis=1,
        )
    return sequences


class HSTUDataset(Dataset):
    """One sample = one (user, target_movie, target_label) triple, paired
    with the user's prior history strictly before target_ts.

    TODO: replace the per-sample history slice with sequence-level training
    once the HSTU model accepts a packed batch of sequences.
    """

    def __init__(self, df: pd.DataFrame, history: dict[int, np.ndarray], seq_len: int):
        self.uid = df["userId"].to_numpy(np.int64)
        self.mid = df["movieId"].to_numpy(np.int64)
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
        if events is None:
            hist_items = np.zeros(self.seq_len, dtype=np.int64)
            hist_ratings = np.zeros(self.seq_len, dtype=np.int64)
            hist_mask = np.zeros(self.seq_len, dtype=np.float32)
        else:
            cut = np.searchsorted(events[:, 2], ts, side="left")
            window = events[max(0, cut - self.seq_len):cut]
            n = window.shape[0]
            hist_items = np.zeros(self.seq_len, dtype=np.int64)
            hist_ratings = np.zeros(self.seq_len, dtype=np.int64)
            hist_mask = np.zeros(self.seq_len, dtype=np.float32)
            if n > 0:
                hist_items[-n:] = window[:, 0]
                hist_ratings[-n:] = window[:, 1]
                hist_mask[-n:] = 1.0
        return {
            "uid": uid,
            "mid": int(self.mid[idx]),
            "label": float(self.lbl[idx]),
            "hist_items": hist_items,
            "hist_ratings": hist_ratings,
            "hist_mask": hist_mask,
        }


# ─── Model ──────────────────────────────────────────────────────────
class HSTUBlock(nn.Module):
    """One HSTU block: pre-norm + gated linear unit + relative-position-bias
    attention + residual. Causal.

    TODO: implement per Meta 2024 §3.
        - PreNorm(x) → SiLU(W_q x) ⊙ (W_k x), with relative position bias
        - PreNorm(x) → SiLU(W_u x) ⊙ (W_v x) for the GLU FFN
        - Residual around each
    """

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        # placeholder
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # placeholder: vanilla pre-norm transformer block, no relative bias
        h = self.norm(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=(mask < 0.5), need_weights=False)
        x = x + attn_out
        x = x + self.ffn(x)
        return x


class PlaceholderHSTU(nn.Module):
    """Stub model: embedding sum + tiny transformer + scoring head.

    The real HSTU implementation (per Meta 2024) replaces the body with the
    HSTU block stack and adds relative-position bias from time deltas.
    """

    def __init__(self, num_items: int, num_rating_buckets: int):
        super().__init__()
        self.item_embed = nn.Embedding(num_items + 1, EMBED_DIM, padding_idx=0)
        self.rating_embed = nn.Embedding(num_rating_buckets, EMBED_DIM)
        self.blocks = nn.ModuleList([
            HSTUBlock(EMBED_DIM, NUM_HEADS, DROPOUT) for _ in range(NUM_LAYERS)
        ])
        self.head = nn.Linear(EMBED_DIM * 2, 1)
        # TODO: relative-position bias from log-bucketed time deltas
        # TODO: action-type embedding (binary engaged vs implicit) once we add easy negs to history

    def forward(self, hist_items, hist_ratings, hist_mask, candidate_items):
        h = self.item_embed(hist_items) + self.rating_embed(hist_ratings)
        for block in self.blocks:
            h = block(h, hist_mask)
        # take the last valid position as the user representation
        # TODO: replace with proper next-position decoding
        last_pos = hist_mask.sum(dim=1).clamp(min=1).long() - 1
        batch_idx = torch.arange(h.size(0), device=h.device)
        user_rep = h[batch_idx, last_pos]
        cand_rep = self.item_embed(candidate_items)
        logit = self.head(torch.cat([user_rep, cand_rep], dim=-1)).squeeze(-1)
        return logit


# ─── Train + eval ───────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, *, train: bool):
    model.train(train)
    total_loss = 0.0
    total_n = 0
    all_scores, all_labels = [], []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    for batch in loader:
        hist_items = batch["hist_items"].to(DEVICE)
        hist_ratings = batch["hist_ratings"].to(DEVICE)
        hist_mask = batch["hist_mask"].to(DEVICE)
        cand = torch.tensor(batch["mid"], device=DEVICE)
        label = torch.tensor(batch["label"], device=DEVICE, dtype=torch.float32)
        logit = model(hist_items, hist_ratings, hist_mask, cand)
        loss = bce(logit, label)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        total_n += label.numel()
        all_scores.append(torch.sigmoid(logit).detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return total_loss / max(1, total_n), evaluate(labels, scores)["auc"]


def main():
    t0 = time.time()
    log.info(f"Loading {DATASET} (raw rating events; no feature engineering)")
    data = load_data(DATASET)
    train_df, val_df, test_df = data["train"], data["val"], data["test"]
    stats = data["stats"]
    log.info(f"  num_users={stats['num_users']}  num_items={stats['num_items']}  "
             f"num_train={stats['num_train']}  num_val={stats['num_val']}  num_test={stats['num_test']}")

    # Build per-user sequences from train + val (for val we use train-only history;
    # for test we use train+val history — same semantics as simple_v2's
    # EVAL_DYNAMIC_HIST=1 mechanism, which is the bar to clear).
    log.info("Building per-user event sequences")
    train_history = build_user_sequences(train_df)
    eval_history = build_user_sequences(pd.concat([train_df, val_df], ignore_index=True))

    # TODO: collator that packs sequences for the real HSTU; for now per-sample slice.
    def collate(batch):
        return {
            "uid": [b["uid"] for b in batch],
            "mid": [b["mid"] for b in batch],
            "label": [b["label"] for b in batch],
            "hist_items": torch.tensor(np.stack([b["hist_items"] for b in batch])),
            "hist_ratings": torch.tensor(np.stack([b["hist_ratings"] for b in batch])),
            "hist_mask": torch.tensor(np.stack([b["hist_mask"] for b in batch])),
        }

    train_ds = HSTUDataset(train_df, train_history, SEQ_LEN)
    val_ds = HSTUDataset(val_df, eval_history, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = PlaceholderHSTU(stats["num_items"], NUM_RATING_BUCKETS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"PlaceholderHSTU: {n_params/1e6:.2f}M params on {DEVICE}")
    log.info("⚠️  Placeholder model — replace with real HSTU block stack to make this meaningful.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auc = 0.0
    for epoch in range(MAX_EPOCHS):
        train_loss, train_auc = run_epoch(model, train_loader, optimizer, train=True)
        val_loss, val_auc = run_epoch(model, val_loader, optimizer=None, train=False)
        log.info(f"epoch {epoch}: train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
                 f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")
        best_val_auc = max(best_val_auc, val_auc)

    total = time.time() - t0
    print(f"\nval_auc:          {best_val_auc:.6f}")
    print(f"total_seconds:    {total:.1f}")
    print(f"dataset:          {DATASET}")
    print(f"num_params_M:     {n_params/1e6:.2f}")
    print(f"# bar to clear (simple_v2 locked): val 0.8594 / test 0.8455")


if __name__ == "__main__":
    main()
