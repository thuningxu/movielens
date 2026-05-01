#!/usr/bin/env python3
"""
HSTU stub for generative recommendation on MovieLens.

Status: scaffolding only. End-to-end pipeline runs on ml-100k but the
model is a placeholder (vanilla pre-norm transformer + dot-product head).
Replace the HSTUBlock body with the actual HSTU stack to make this
attempt meaningful.

Step 1 (this file): sequence-level training with per-position causal loss
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
ENGAGED_BUCKET_THRESHOLD = 7                            # bucket >= 7 ⇔ rating >= 4 (label=1)

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
            [group["movieId"].to_numpy(np.int64),
             group["rating_bucket"].to_numpy(np.int64),
             group["timestamp"].to_numpy(np.int64)],
            axis=1,
        )
    return sequences


def _pad_left(items: np.ndarray, ratings: np.ndarray, seq_len: int):
    """Left-pad item/rating arrays to seq_len with zeros (most recent at end)."""
    n = items.shape[0]
    if n >= seq_len:
        items = items[-seq_len:]
        ratings = ratings[-seq_len:]
        mask = np.ones(seq_len, dtype=np.float32)
        return items.astype(np.int64), ratings.astype(np.int64), mask
    out_items = np.zeros(seq_len, dtype=np.int64)
    out_ratings = np.zeros(seq_len, dtype=np.int64)
    out_mask = np.zeros(seq_len, dtype=np.float32)
    if n > 0:
        out_items[-n:] = items
        out_ratings[-n:] = ratings
        out_mask[-n:] = 1.0
    return out_items, out_ratings, out_mask


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
        items, ratings, mask = _pad_left(items, ratings, self.seq_len)
        return {
            "uid": uid,
            "hist_items": items,
            "hist_ratings": ratings,
            "hist_mask": mask,
        }


class EvalDataset(Dataset):
    """One sample = one (user, candidate_movie, target_ts) eval row, paired
    with the user's prefix sequence (events strictly before target_ts) sliced
    from a precomputed dynamic-history dict. Mirrors simple_v2's
    EVAL_DYNAMIC_HIST=1 mechanism.
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
        if events is None or events.shape[0] == 0:
            items = np.zeros(self.seq_len, dtype=np.int64)
            ratings = np.zeros(self.seq_len, dtype=np.int64)
            mask = np.zeros(self.seq_len, dtype=np.float32)
        else:
            cut = np.searchsorted(events[:, 2], ts, side="left")
            window = events[max(0, cut - self.seq_len):cut]
            items, ratings, mask = _pad_left(window[:, 0], window[:, 1], self.seq_len)
        return {
            "uid": uid,
            "mid": int(self.mid[idx]),
            "label": float(self.lbl[idx]),
            "hist_items": items,
            "hist_ratings": ratings,
            "hist_mask": mask,
        }


def collate_train(batch):
    return {
        "uid": [b["uid"] for b in batch],
        "hist_items": torch.tensor(np.stack([b["hist_items"] for b in batch])),
        "hist_ratings": torch.tensor(np.stack([b["hist_ratings"] for b in batch])),
        "hist_mask": torch.tensor(np.stack([b["hist_mask"] for b in batch])),
    }


def collate_eval(batch):
    return {
        "uid": [b["uid"] for b in batch],
        "mid": torch.tensor([b["mid"] for b in batch], dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "hist_items": torch.tensor(np.stack([b["hist_items"] for b in batch])),
        "hist_ratings": torch.tensor(np.stack([b["hist_ratings"] for b in batch])),
        "hist_mask": torch.tensor(np.stack([b["hist_mask"] for b in batch])),
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # placeholder: vanilla pre-norm transformer block, no relative bias.
        # key_padding_mask and attn_mask must share dtype (both float here) to
        # avoid the PyTorch mismatched-mask deprecation path.
        h = self.norm(x)
        kp_mask = torch.zeros_like(mask)
        kp_mask = kp_mask.masked_fill(mask < 0.5, float("-inf"))
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=kp_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(x)
        return x


class PlaceholderHSTU(nn.Module):
    """Stub model: causal transformer over (item + rating) embeddings with
    SASRec-style dot-product scoring against a candidate item embedding.

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
        # No concat-head here: scoring is dot(h_t, item_embed(candidate)).
        # SASRec-style sharing of item_embed between input and output is
        # standard for sequence recommenders and halves head params.
        # TODO: relative-position bias from log-bucketed time deltas
        # TODO: action-type embedding (binary engaged vs implicit) once we add easy negs to history

    def encode(self, hist_items: torch.Tensor, hist_ratings: torch.Tensor,
               hist_mask: torch.Tensor) -> torch.Tensor:
        """Run the causal stack and return per-position hidden states (B, T, D)."""
        B, T = hist_items.shape
        h = self.item_embed(hist_items) + self.rating_embed(hist_ratings)
        # Causal mask: position t can only attend to ≤t. nn.MultiheadAttention
        # expects a float mask added to attention logits (so -inf blocks).
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=h.device), diagonal=1,
        )
        for block in self.blocks:
            h = block(h, hist_mask, causal)
        return h

    def score_per_position(self, h: torch.Tensor, target_items: torch.Tensor) -> torch.Tensor:
        """Dot-product scoring at every position. h:(B,T,D), target_items:(B,T)."""
        target_e = self.item_embed(target_items)              # (B, T, D)
        return (h * target_e).sum(dim=-1)                     # (B, T)

    def score_eval(self, h: torch.Tensor, hist_mask: torch.Tensor,
                   candidate: torch.Tensor) -> torch.Tensor:
        """Score the LAST valid position's hidden state against a candidate."""
        # mask_sum == 0 means empty history; we still index 0 and let the
        # learned-bias dot product handle it (loss/AUC harness will treat this
        # as a generic prior). hist_mask>0 yields ≥1 for any user with ≥1 event.
        last_pos = hist_mask.sum(dim=1).clamp(min=1).long() - 1
        batch_idx = torch.arange(h.size(0), device=h.device)
        h_last = h[batch_idx, last_pos]                       # (B, D)
        cand_e = self.item_embed(candidate)                   # (B, D)
        return (h_last * cand_e).sum(dim=-1)                  # (B,)


# ─── Train + eval ───────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer):
    """Sequence-level training: per-position causal BCE on next-event engagement."""
    model.train()
    total_loss = 0.0
    total_positions = 0
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for batch in loader:
        hist_items = batch["hist_items"].to(DEVICE)         # (B, T)
        hist_ratings = batch["hist_ratings"].to(DEVICE)     # (B, T)
        hist_mask = batch["hist_mask"].to(DEVICE)           # (B, T)

        h = model.encode(hist_items, hist_ratings, hist_mask)  # (B, T, D)

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += masked.sum().item()
        total_positions += int(pair_mask.sum().item())
    avg_loss = total_loss / max(1, total_positions)
    return avg_loss


@torch.no_grad()
def evaluate_model(model, loader):
    """Per-(user, candidate, ts) eval: dot(h_T, item_embed(candidate)) → sigmoid → AUC."""
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        hist_items = batch["hist_items"].to(DEVICE)
        hist_ratings = batch["hist_ratings"].to(DEVICE)
        hist_mask = batch["hist_mask"].to(DEVICE)
        cand = batch["mid"].to(DEVICE)
        label = batch["label"]
        h = model.encode(hist_items, hist_ratings, hist_mask)
        logit = model.score_eval(h, hist_mask, cand)
        all_scores.append(torch.sigmoid(logit).detach().cpu().numpy())
        all_labels.append(label.numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    return evaluate(labels, scores)["auc"]


def main():
    t0 = time.time()
    log.info(f"Loading {DATASET} (raw rating events; no feature engineering)")
    data = load_data(DATASET)
    train_df, val_df, test_df = data["train"], data["val"], data["test"]
    stats = data["stats"]
    log.info(f"  num_users={stats['num_users']}  num_items={stats['num_items']}  "
             f"num_train={stats['num_train']}  num_val={stats['num_val']}  num_test={stats['num_test']}")

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

    model = PlaceholderHSTU(stats["num_items"], NUM_RATING_BUCKETS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"PlaceholderHSTU: {n_params/1e6:.2f}M params on {DEVICE}")
    log.info("⚠️  Placeholder model — replace with real HSTU block stack to make this meaningful.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auc = 0.0
    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_auc = evaluate_model(model, val_loader)
        log.info(f"epoch {epoch}: train_loss={train_loss:.4f} val_auc={val_auc:.4f}")
        best_val_auc = max(best_val_auc, val_auc)

    total = time.time() - t0
    print(f"\nval_auc:          {best_val_auc:.6f}")
    print(f"total_seconds:    {total:.1f}")
    print(f"dataset:          {DATASET}")
    print(f"num_params_M:     {n_params/1e6:.2f}")
    print(f"# bar to clear (simple_v2 locked): val 0.8594 / test 0.8455")


if __name__ == "__main__":
    main()
