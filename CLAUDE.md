# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Restart (apr28) of the MovieLens hybrid engagement prediction project. Same task as legacy: predict whether a user will rate a movie >= 4 stars, with both "rated < 4" (hard negatives) and "random unrated" (easy negatives) as label=0. BCE loss over calibrated probabilities. Metric: val_auc on ml-25m (deterministic, SEED=42).

The legacy project at `legacy/` reached **val_auc = 0.8284** but two separate ceiling tests (apr27, apr27c) confirmed the architecture family is saturated. This restart begins from the **simplest possible model — a single Linear head on concatenated features — with the same input features**, so future architectural decisions can be motivated by clean ablations rather than 540 experiments of inherited assumptions.

The starting baseline AUC will be substantially below 0.8284. Build it back up deliberately.

## Commands

```bash
# Sync the repo-local environment
uv sync

# Quick smoke test (ml-100k, ~seconds) — crash detection only, NOT for AUC comparison
DATASET=ml-100k uv run python train.py

# Standard experiment (ml-25m on the current CUDA GPU)
DATASET=ml-25m uv run python train.py

# Full experiment run (redirected, for autoresearch loop)
DATASET=ml-25m uv run python train.py > run.log 2>&1

# Check results
grep "^val_auc:\|^peak_memory_mb:" run.log
```

## Architecture

- **`prepare.py`** — Shared with legacy. Data download/loading (all MovieLens sizes), `load_data_hybrid()`, time-based train/val/test splits, AUC evaluation, `print_summary()`. **Do not modify the evaluation harness.**
- **`train.py`** — The current baseline + experimentation file. Same input features as legacy; model is `concat → Linear(in, 1) → sigmoid`. No hidden layer.
- **`program.md`** — Fresh experiment log starting at the apr28 baseline.
- **`legacy/`** — Archive of the prior project. Available for reference if useful, but don't feel obligated to inherit its conclusions.
- **`results.tsv`** — Experiment log (untracked). Tab-separated: commit, val_auc, memory_mb, status, description.

## Key Details

- **Metric**: val_auc on ml-25m (higher is better).
- **Label**: rating >= 4 → positive (1); rating < 4 OR random unrated → negative (0).
- **Device**: Single CUDA GPU. Auto-detects CUDA / MPS / CPU.
- **Environment**: Use the repo-local `uv` env (`uv sync`, then `uv run ...`).
- **Datasets**: `ml-100k` (smoke test only, no genome data), `ml-1m` (fast iteration), `ml-10m` (medium), `ml-25m` (default, has genome data).
- **Reproducibility**: Deterministic at SEED=42. Run-to-run variance at the same seed is <0.001 AUC; seed-to-seed variance is larger and should be estimated empirically before declaring any win.
- **Data**: auto-downloaded to `data/` on first use; not checked into git.
- **Feature cache**: `data/features_<hash>.npz` is built on first run per (dataset, history-len) and reused afterward.

## Current checked-in baseline (train.py)

`concat(features) → Linear(in, 1) → sigmoid`. The features:

```
- userId  → Embedding(num_users, 28)            → user_e (28)
- movieId → Embedding(num_items, 28)            → item_e (28)
- User history (last 100 items + ratings):
    mean-pool of item_embed over valid positions → user_hist_pool (28)
    mean rating in user history                  → user_hist_rat_mean (1)
- Item history (last 30 raters + ratings):
    mean-pool of user_embed over valid raters    → item_hist_pool (28)
    mean rating in item history                  → item_hist_rat_mean (1)
- Genre multi-hot (20) → Linear(20, 28, bias=False) → genre_e (28)
- Dense (17): timestamp, user-rating-histogram (5), user-count, item-rating-histogram (5), item-count, ug_dot, year, genre-count, movie_age
- Tag genome (1128, raw)                          → genome (1128)
- Per-user genome profile (1128, raw)             → user_genome (1128)

concat → Linear(in_dim, 1) → sigmoid

Loss: BCEWithLogitsLoss
Optimizer: Adam, lr=1e-3, weight_decay=1e-5
Training: batch=16384, sub-epoch eval 3×, patience=3 evals, max 20 epochs
```

The "linear" naming refers to the prediction head — embeddings are still trainable (~6M params for ml-25m). The `genre_proj` is a single bias-free Linear that exists purely to project a high-dimensional one-hot into the same dim as the embeddings; it has no nonlinearity.

## Discipline

- **Multi-seed verification is mandatory for any keep claim.** Estimate the seed-noise floor (e.g., 3-4 baseline seeds) before testing candidates; declare a win only when the lift is statistically distinguishable from that floor.
- **Smoke-test on ml-100k for crashes only**, not for AUC. ml-100k has no genome data and is too small for the linear baseline to be informative.
- **`prepare.py:evaluate()` is the ground truth.** Do not modify it.
- **Keep `train.py` simple while it's small.** When the model grows past ~500 lines, split into `model.py` / `data.py` / `train.py`.
