# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repo for movie recommendation on MovieLens. Uses a hybrid engagement prediction task: predict whether a user will rate a movie >= 4 stars, with both "watched but didn't like" (hard negatives) and "random unrated" (easy negatives) as label=0. Output is a calibrated probability via BCE loss, suitable for front-page recommendation with a threshold.

Current model: DLRM with rating-aware DIN + causal self-attention, item-side DIN, 3 GDCN cross layers, FinalMLP two-stream with bilinear.
Current AUC: **0.799 on ml-25m** (deterministic, SEED=42). See `program.md` for full experiment history (~80 experiments).

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
  - Sparse: userId, movieId (embeddings, dim=24, with dropout 0.1)
  - Genre: multi-hot → linear projection → dim=24
  - User history: last 100 items + ratings → causal self-attention → DIN (target-aware) → dim=24
  - Item history: last 30 raters + ratings → item-side DIN (target-user-aware) → dim=24
  - Dense: timestamp, user stats (3), item stats (3), ug_dot (1), year (1), genre_count (1), movie_age (1) → bottom MLP → dim=24

Interaction: 3 GDCN gated cross layers over concatenated 6×24=144 dim vector

Two-stream: user stream (256→64) + item stream (256→64) + bilinear interaction
Top MLP: (144 + 64 + 64 + 64) → 256 → 128 → 64 → 1 (with dropout 0.2)

Loss: BCEWithLogitsLoss with label smoothing 0.1
Optimizer: Adam, LR=1e-4, weight_decay=1e-5
AMP: fp16, torch.compile, TF32 tensor cores
Training: batch=16384, grad accum 8x (effective 131K), sub-epoch eval 2x, patience=2
Params: ~11M | VRAM: ~10.5 GB on L4
```

## Critical learnings from experiments

1. **New information > more capacity.** Item-side DIN (+0.029), rating-aware history, movie metadata all helped. Bigger MLPs, more heads, deeper layers all hurt.
2. **Scale unlocks capacity.** 3 GDCN layers and embed_dim=24 hurt on ml-10m but helped on ml-25m. More data shifts the overfitting threshold.
3. **Only embed dropout works for regularization.** 0.1 is the sweet spot. All other regularization (MLP dropout changes, weight decay, label smoothing >0.1, contrastive losses, EMA) hurt or had no effect.
4. **Fixed seeds are essential.** Variance was ~0.05 AUC before seeding. After SEED=42, <0.001.
5. **Training procedure changes don't work.** LR schedules, warmup, multi-task, contrastive losses — all tried, all failed across 3 datasets.
