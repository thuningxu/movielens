# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Restart (apr28) of the MovieLens hybrid engagement prediction project. Same task as legacy: predict whether a user will rate a movie >= 4 stars, with both "rated < 4" (hard negatives) and "random unrated" (easy negatives) as label=0. BCE loss over calibrated probabilities. Metric: val_auc on ml-25m (deterministic, SEED=42).

The legacy project at `legacy/` reached **val_auc = 0.8284** but two separate ceiling tests (apr27, apr27c) confirmed the architecture family is saturated. This restart begins from the **simplest possible model — a single Linear head on concatenated features — with the same input features**, so future architectural decisions can be motivated by clean ablations rather than 540 experiments of inherited assumptions.

Current baseline: **0.8282 on ml-25m at SEED=42** (5-seed mean +0.00175 over the prior 0.8263 LR/WD-retuned baseline; σ ≈ 0.00008 across SEEDs 42-46). Reached by stacking three individually sub-threshold mechanisms — rating-centered pool, multiplicative cross fields including `ts ⊙ i_e`, and an auxiliary rating-residual regression head — that compound super-additively. The progression is documented in `program.md`'s cumulative table.

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
- **Reproducibility**: Deterministic at SEED=42. Run-to-run variance at the same seed is <1e-5 AUC; seed-to-seed variance for the current linear baseline is σ ≈ 0.00008 across SEED ∈ {42,43,44,45,46} — about 10× tighter than the legacy DLRM's σ ≈ 0.00078. Estimate empirically again whenever the model architecture changes meaningfully.
- **Data**: auto-downloaded to `data/` on first use; not checked into git.
- **Feature cache**: `data/features_<hash>.npz` is built on first run per (dataset, history-len) and reused afterward.

## Current checked-in baseline (train.py)

`concat(features) → Linear(in, 1) → sigmoid`. Stripped to the bones — only raw IDs, raw history sequences, and pure content metadata. All pre-computed user/item statistics (rating histograms, counts, user-genre affinity, user genome profile) are removed on the principle that aggregations are relationships the model should learn from raw data, not inputs hand-specified before training.

```
- userId  → Embedding(num_users, 28)            → user_e (28)
- movieId → Embedding(num_items, 28)            → item_e (28)
- User history (last 100 items + ratings):
    rating-centered pool of item_embed           → user_hist_pool (28)
      weight = (rating - 0.6) * is_valid
      normalize by sum(|weight|).clamp(1e-6)
    mean rating in user history                  → user_hist_rat_mean (1)
- Item history (last 30 raters + ratings):
    rating-centered pool of user_embed           → item_hist_pool (28)
      weight = (rating - 0.6) * is_valid
    mean rating in item history                  → item_hist_rat_mean (1)
- Genre multi-hot (raw, no projection)            → genre (num_genres, e.g. 20)
- timestamp_norm                                  → ts (1)
- movie_year                                      → year (1)
- Tag genome (1128, raw)                          → genome (1128)
- Cross fields (CROSS_FIELDS=1, default on):
    u_e ⊙ i_e                                      → cross_ui (28)
    u_hist_pool ⊙ i_e                              → cross_uhist_item (28)
    i_hist_pool ⊙ u_e                              → cross_ihist_user (28)
- Cross field (CROSS_TS_ITEM=1, default on):
    ts_norm ⊙ i_e                                  → cross_ts_item (28)

concat → Linear(in_dim, 1) → sigmoid    # in_dim = 4*28 + 2 + 20 + 2 + 1128 + 4*28 = 1376 (ml-25m, with default cross fields and ts-item cross)

Loss: BCEWithLogitsLoss + AUX_RATING_WEIGHT (=25) × masked_mse on rating regression head
Optimizer: Adam, lr=3e-4, weight_decay=5e-5
Item-embed regularization: Adam WD + FREQ_WD_LAMBDA (=1e-4) × per-item L2 weighted 1/sqrt(count+5)
Cross fields: 4 Hadamard products (u_e⊙i_e, u_hist⊙i_e, i_hist⊙u_e, ts⊙i_e)
Training: batch=16384, sub-epoch eval 3×, patience=3 evals, max 20 epochs
```

The "linear" naming refers to the prediction head — embeddings are still trainable (~6M params for ml-25m). Genre multi-hot, timestamp, year, and tag genome go straight into the concat with no intermediate projection (a `Linear(20, 28) → Linear(in, 1)` chain is expressively equivalent to a direct slice in the head).

## Discipline

- **Multi-seed verification is mandatory for any keep claim.** Estimate the seed-noise floor (e.g., 3-4 baseline seeds) before testing candidates; declare a win only when the lift is statistically distinguishable from that floor.
- **Smoke-test on ml-100k for crashes only**, not for AUC. ml-100k has no genome data and is too small for the linear baseline to be informative.
- **`prepare.py:evaluate()` is the ground truth.** Do not modify it.
- **Keep `train.py` simple while it's small.** When the model grows past ~500 lines, split into `model.py` / `data.py` / `train.py`.
