# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Restart (apr28) of the MovieLens hybrid engagement prediction project. Same task as legacy: predict whether a user will rate a movie >= 4 stars, with both "rated < 4" (hard negatives) and "random unrated" (easy negatives) as label=0. BCE loss over calibrated probabilities. Metric: val_auc on ml-25m (deterministic, SEED=42).

The legacy project at `legacy/` reached **val_auc = 0.8284** but two separate ceiling tests (apr27, apr27c) confirmed the architecture family is saturated. This restart begins from the **simplest possible model — a single Linear head on concatenated features — with the same input features**, so future architectural decisions can be motivated by clean ablations rather than 540 experiments of inherited assumptions.

Current baseline: **0.8594 on ml-25m at SEED=42** (5-seed mean **0.859289**) with `EVAL_DYNAMIC_HIST=1 FREQ_WD_LAMBDA=0 LR=1e-3`. Reached by stacking three post-apr28o mechanisms:
1. **Eval-time dynamic user history** (`EVAL_DYNAMIC_HIST=1`, apr28ad): at evaluation each sample's u_hist_pool is rebuilt from train+val ratings strictly prior to sample's ts. Cold_user stratum 0.787 → 0.815 (+0.028), drives overall +0.022 5-seed mean.
2. **Drop tail-item regularizer** (`FREQ_WD_LAMBDA=0`, apr28ag): at the dynamic regime tail items need larger embeddings to feed useful dynamic-history signal; +0.0015 5-seed mean on top of apr28ad.
3. **HP retune** (`LR=1e-3`, apr28ah): the static-regime LR=3e-4 was over-conservative at the new regime; with FREQ_WD off + dynamic eval signal the model wants more aggressive updates. +0.008 5-seed mean on top of apr28ag.

This is **not the same evaluation setup** as the legacy 0.8284 ceiling — it requires inference-time access to the user's prior ratings (how production recommender systems actually work). The static-history baseline (apr28o stack) was 0.828188, which the legacy DLRM matched within 0.0002. Apr28ad's mechanism is structurally orthogonal: a feature-engineering / inference-time change rather than an architecture change.

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
- **Reproducibility**: Deterministic at SEED=42. Run-to-run variance at the same seed is <1e-5 AUC. Seed-to-seed variance is regime-dependent: at the **static-history regime** (EVAL_DYNAMIC_HIST=0, apr28o stack at 0.8282) σ ≈ 0.00008 across SEED ∈ {42,43,44,45,46} — about 10× tighter than the legacy DLRM's σ ≈ 0.00078. At the **dynamic-history regime** (EVAL_DYNAMIC_HIST=1, apr28ad at 0.8498 5-seed mean) lift-σ widens to ~0.003 because dynamic histories amplify per-seed variance through the cold_user stratum. **The multi-seed bar (mean ≥ +0.0007, 5/5 positive, min ≥ -0.0003) was set at the static regime; at the dynamic regime use lift-σ ~0.003 to set new bars** (e.g., apr28af verified at +0.001 mean was correctly judged sub-noise). Re-estimate empirically whenever the baseline regime changes.
- **Data**: auto-downloaded to `data/` on first use; not checked into git.
- **Feature cache**: `data/features_<hash>.npz` is built on first run per (dataset, history-len, recency-frac) and reused afterward. Current `feature_version=restart-6` (per-movie tag-text embedding from MiniLM, apr28ac); a checkout pre-restart-6 will trigger one-time rebuild on first run.

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

Eval-time mechanism (apr28ad — opt-in, default OFF):
- EVAL_DYNAMIC_HIST=1: at evaluation each sample's u_hist is rebuilt
  per-sample from train+val ratings strictly prior to the sample's
  timestamp (vs the static per-user history built once from train).
  Off-state byte-equivalent. Lifts cold_user stratum AUC 0.787 → 0.815
  and overall val_auc 0.828 → 0.846 SEED=42 (5-seed mean +0.022). The
  headline 0.8463 / 0.8498 baseline numbers REQUIRE this flag set.
```

The "linear" naming refers to the prediction head — embeddings are still trainable (~6M params for ml-25m). Genre multi-hot, timestamp, year, and tag genome go straight into the concat with no intermediate projection (a `Linear(20, 28) → Linear(in, 1)` chain is expressively equivalent to a direct slice in the head).

**To reproduce the headline baseline**: `EVAL_DYNAMIC_HIST=1 DATASET=ml-25m uv run python train.py`. Without `EVAL_DYNAMIC_HIST=1`, the model trains identically and reproduces the static-history baseline of 0.8282 (apr28o stack), which still matches the legacy DLRM ceiling within 0.0002.

## Discipline

- **Multi-seed verification is mandatory for any keep claim.** Estimate the seed-noise floor (e.g., 3-4 baseline seeds) before testing candidates; declare a win only when the lift is statistically distinguishable from that floor.
- **Smoke-test on ml-100k for crashes only**, not for AUC. ml-100k has no genome data and is too small for the linear baseline to be informative.
- **`prepare.py:evaluate()` is the ground truth.** Do not modify it.
- **Keep `train.py` simple while it's small.** When the model grows past ~500 lines, split into `model.py` / `data.py` / `train.py`.
