# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repo for movie recommendation on MovieLens. Uses a hybrid engagement prediction task: predict whether a user will rate a movie >= 4 stars, with both "watched but didn't like" (hard negatives) and "random unrated" (easy negatives) as label=0. Output is a calibrated probability via BCE loss, suitable for front-page recommendation with a threshold.

Current model: DLRM with rating-aware DIN + causal self-attention, item-side DIN, 4 GDCN cross layers, FinalMLP two-stream with bilinear.
Current AUC: **0.806 on ml-25m** (deterministic, SEED=42). See `program.md` for full experiment history (~120 experiments).

## Commands

```bash
# Quick smoke test (ml-100k, ~seconds) — only for crash detection, NOT for AUC comparison
DATASET=ml-100k python3 train.py

# Standard experiment (ml-25m on NVIDIA L4, ~5-10 minutes)
DATASET=ml-25m python3 train.py

# Full experiment run (redirected, for autoresearch loop)
DATASET=ml-25m python3 train.py > run.log 2>&1

# Check results
grep "^val_auc:\|^peak_memory_mb:" run.log
```

## Architecture

- **`prepare.py`** — Data download/loading (all MovieLens sizes), `load_data_hybrid()` for the current formulation, time-based train/val/test splits, AUC evaluation, `print_summary()`. May be modified for data setup changes. Keep `evaluate()` and `print_summary()` stable.
- **`train.py`** — The experimentation file. Feature engineering, model architecture (DLRM + DIN + GDCN + FinalMLP), training loop. Primary file to modify.
- **`program.md`** — The autoresearch protocol: setup, experiment loop, logging, full experiment history and learnings from ~80 experiments.
- **`results.tsv`** — Experiment log (untracked). Tab-separated: commit, val_auc, memory_mb, status, description.

## Key Details

- **Metric**: val_auc (higher is better).
- **Label**: rating >= 4 → positive (1), rating < 4 or random unrated → negative (0).
- **Device**: NVIDIA L4 GPU (CUDA). Auto-detects CUDA/CPU.
- **Training termination**: Early stopping (patience=2 evals, sub-epoch eval 2x/epoch), no fixed time budget.
- **Datasets**: `ml-100k` (smoke test only), `ml-1m` (fast iteration), `ml-10m` (medium), `ml-25m` (default, full scale). Selected via `DATASET` env var.
- **Reproducibility**: Deterministic training (SEED=42). Run-to-run variance <0.001 AUC.
- Data is auto-downloaded to `data/` on first use. Not checked into git.

## Current model architecture (train.py)

```
Features:
  - Sparse: userId, movieId (embeddings, dim=28, with dropout 0.1)
  - Genre: multi-hot → linear projection → dim=28
  - User history: last 100 items + ratings → causal self-attention → DIN (target-aware) → dim=28
  - Item history: last 30 raters + ratings → item-side DIN (target-user-aware) → dim=28
  - Dense: timestamp, user rating histogram (5-bin), user count, item rating histogram (5-bin), item count, ug_dot (1), year (1), genre_count (1), movie_age (1) → bottom MLP → dim=28

Interaction: 4 GDCN gated cross layers over concatenated 6×28=168 dim vector

Two-stream: user stream (256→64) + item stream (256→64) + bilinear interaction
Top MLP: (168 + 64 + 64 + 64) → 256 → 128 → 64 → 1 (with dropout 0.2)

Loss: BCEWithLogitsLoss with label smoothing 0.1
Optimizer: Adam, LR=1e-4, weight_decay=1e-5
AMP: fp16, torch.compile, TF32 tensor cores
Training: batch=16384, grad accum 8x (effective 131K), sub-epoch eval 2x, patience=2
Params: ~13M | VRAM: ~9.7 GB on L4
```

## Critical learnings from experiments

1. **New information > more capacity.** Item-side DIN (+0.029), rating-aware history, movie metadata, rating histograms (+0.003) all helped. Bigger MLPs, more heads, deeper layers all hurt.
2. **Richer features unlock more capacity.** 4 GDCN layers failed at 0.742 without histogram bins but succeeded at 0.804 with them. embed_dim=28 works with histograms+4GDCN but 32 still overfits. More informative features shift the overfitting threshold.
3. **Only embed dropout works for regularization.** 0.1 is the sweet spot. All other regularization (MLP dropout changes, weight decay, label smoothing >0.1, contrastive losses, EMA) hurt or had no effect.
4. **Fixed seeds are essential.** Variance was ~0.05 AUC before seeding. After SEED=42, <0.001.
5. **Training procedure changes don't work.** LR schedules, warmup, multi-task, contrastive losses — all tried, all failed across 3 datasets.
6. **Rating histograms > summary statistics.** 5-bin rating distributions capture more signal than mean+std. The distribution shape matters.
