# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repo for movie recommendation on MovieLens. Uses a hybrid engagement prediction task: predict whether a user will rate a movie >= 4 stars, with both "watched but didn't like" (hard negatives) and "random unrated" (easy negatives) as label=0. Output is a calibrated probability via BCE loss, suitable for front-page recommendation with a threshold.

Current model: DLRM with DIN attention, DCN-V2 cross layers, user-genre affinity features.
Current AUC: ~0.77 on ml-1m (+/- 0.016 run-to-run variance). See `program.md` for full experiment history.

## Commands

```bash
# Quick smoke test (ml-100k, ~seconds) — only for crash detection, NOT for AUC comparison
DATASET=ml-100k python3 train.py

# Standard experiment (ml-10m on NVIDIA L4, ~5-15 minutes)
DATASET=ml-10m python3 train.py

# Full experiment run (redirected, for autoresearch loop)
DATASET=ml-10m python3 train.py > run.log 2>&1

# Check results
grep "^val_auc:\|^peak_memory_mb:" run.log
```

## Architecture

- **`prepare.py`** — Data download/loading (all MovieLens sizes), `load_data_hybrid()` for the current formulation, time-based train/val/test splits, AUC evaluation, `print_summary()`. May be modified for data setup changes. Keep `evaluate()` and `print_summary()` stable.
- **`train.py`** — The experimentation file. Feature engineering, model architecture (DLRM + DIN + DCN-V2), training loop. Primary file to modify.
- **`program.md`** — The autoresearch protocol: setup, experiment loop, logging, full experiment history and learnings from 22 experiments.
- **`results.tsv`** — Experiment log (untracked). Tab-separated: commit, val_auc, memory_mb, status, description.

## Key Details

- **Metric**: val_auc (higher is better).
- **Label**: rating >= 4 → positive (1), rating < 4 or random unrated → negative (0).
- **Device**: NVIDIA L4 GPU (CUDA). Auto-detects CUDA/CPU.
- **Training termination**: Early stopping (patience=10 evals), no fixed time budget.
- **Datasets**: `ml-100k` (smoke test only), `ml-1m` (fast iteration), `ml-10m` (default for experiments), `ml-25m` (full scale). Selected via `DATASET` env var.
- **Variance**: Run-to-run variance is ~0.03 AUC on ml-1m. Improvements < 0.01 are noise. Consider fixing all random seeds or averaging 3+ runs.
- Data is auto-downloaded to `data/` on first use. Not checked into git.

## Current model architecture (train.py)

```
Features:
  - Sparse: userId, movieId (embeddings, dim=16)
  - Genre: multi-hot → linear projection → dim=16
  - User history: last 50 items → DIN attention (target-item-aware) → dim=16
  - Dense: timestamp, user stats (3), item stats (3), user-genre affinity dot (1) → bottom MLP → dim=16

Interaction: DCN-V2 with 2 cross layers over concatenated 5×16=80 dim vector

Top MLP: 80 → 256 → 128 → 64 → 1 (with dropout 0.2)

Loss: BCEWithLogitsLoss
Optimizer: Adam, LR=1e-4, weight_decay=1e-5
Early stopping: patience=10 evals
```

## Critical learnings from experiments

1. **Architecture changes work, training procedure changes don't.** DIN, DCN-V2, features helped. LR schedules, regularization, focal loss all hurt.
2. **Model overfits after 1-3 epochs consistently.** This is the main bottleneck. More data (ml-10m+) may help.
3. **ml-100k is smoke test only.** Never compare AUC on ml-100k.
4. **DIN attention is fast on L4.** Single-head: ~18s/epoch on CUDA (was ~30s on MPS). Multi-head DIN may now be feasible.
