# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repo for movie recommendation on MovieLens. Uses a hybrid engagement prediction task: predict whether a user will rate a movie >= 4 stars, with both "watched but didn't like" (hard negatives) and "random unrated" (easy negatives) as label=0. Output is a calibrated probability via BCE loss, suitable for front-page recommendation with a threshold.

Best historical single-model AUC: **0.821 on ml-25m** (deterministic, SEED=42).
Best historical ensemble AUC: **0.854 on ml-25m** (HistGBM stacking of 59 diverse model variants, 3-fold CV validated).
The checked-in `train.py` is the restart baseline and is not the exact historical field-attention/two-stream model. Treat `train.py` as the source of truth for the current baseline and `program.md` as the experiment history.
See `program.md` for full experiment history (~500 experiments). See `README.md` for AUC progress chart.

## Commands

```bash
# Sync the repo-local environment
uv sync

# Quick smoke test (ml-100k, ~seconds) — only for crash detection, NOT for AUC comparison
DATASET=ml-100k uv run python train.py

# Standard experiment (ml-25m on the current CUDA GPU; runtime is hardware-dependent)
DATASET=ml-25m uv run python train.py

# Full experiment run (redirected, for autoresearch loop)
DATASET=ml-25m uv run python train.py > run.log 2>&1

# Check results
grep "^val_auc:\|^peak_memory_mb:" run.log
```

## Architecture

- **`prepare.py`** — Data download/loading (all MovieLens sizes), `load_data_hybrid()` for the current formulation, time-based train/val/test splits, AUC evaluation, `print_summary()`. May be modified for data setup changes. Keep `evaluate()` and `print_summary()` stable.
- **`train.py`** — The experimentation file. Feature engineering, model architecture, and training loop. Primary file to modify. Treat the code as the source of truth for the checked-in baseline.
- **`program.md`** — The autoresearch protocol: setup, experiment loop, logging, full experiment history (~500 experiments), and historical learnings.
- **`results.tsv`** — Experiment log (untracked). Tab-separated: commit, val_auc, memory_mb, status, description.

## Key Details

- **Metric**: val_auc (higher is better).
- **Label**: rating >= 4 → positive (1), rating < 4 or random unrated → negative (0).
- **Device**: Single CUDA GPU machine. Auto-detects CUDA/CPU.
- **Environment**: Use the repo-local `uv` environment (`uv sync`, then `uv run ...`).
- **Training termination**: Early stopping (patience=3 evals, sub-epoch eval 3x/epoch), no fixed time budget.
- **Datasets**: `ml-100k` (smoke test only), `ml-1m` (fast iteration), `ml-10m` (medium), `ml-25m` (default, full scale). Selected via `DATASET` env var.
- **Reproducibility**: Deterministic training (SEED=42). Run-to-run variance <0.001 AUC.
- Data is auto-downloaded to `data/` on first use. Not checked into git.

## Current checked-in baseline (train.py)

```
Features:
  - Sparse: userId, movieId (embeddings, dim=28, with dropout 0.1)
  - Genre: multi-hot → linear projection → dim=28
  - User history: last 100 items + ratings → causal self-attention + residual → rating-weighted pooling over causal prefix → dim=28
  - Item history: last 30 raters + ratings → item-side DIN (target-user-aware) over causal prefix → dim=28
  - Tag genome: 1128-dim relevance scores → bottleneck MLP (1128→256→64→28) → sigmoid gate with detached item_e fallback for missing data → dim=28
  - Dense: timestamp, user rating histogram (5-bin), user count, item rating histogram (5-bin), item count, ug_dot (1), year (1), genre_count (1), movie_age (1) → bottom MLP → dim=28

Interaction: squeeze-and-excitation field reweighting across 7 fields, then flatten to 7×28 = 196

Top MLP: 196 → 256 → 128 → 64 → 1 (with dropout 0.2)

Loss: BCEWithLogitsLoss with label smoothing 0.1
Optimizer: Adam, LR=7e-5, weight_decay=1e-4
AMP: fp16, torch.compile, TF32 tensor cores
Training: batch=16384, grad accum 2× (effective 33K), NEG_RATIO=1, TRAIN_NEG_MODE=anchor_pos_catalog, POST_RECENCY_NEG_RESAMPLE=1, POST_RECENCY_EASY_NEG_PER_POS=0.75, USER_HIST_MODE=rating, USER_HIST_CONTEXT=causal_masked, ITEM_HIST_CONTEXT=causal_masked, RECENCY_FRAC=0.8, sub-epoch eval ~3×, patience=3
Params/VRAM: printed at runtime; historical runs fit comfortably on a 24 GB L4
```

Best historical single-model variant differed from this checked-in baseline: 1-head field attention across fields plus FinalMLP two-stream user/item heads with a bilinear interaction. See `README.md` and `program.md` for that architecture in context.

## Critical learnings from ~460 experiments

See `program.md` for the full list. The most important:

1. **New information > more capacity.** Features that add genuinely new signal help (item-side DIN +0.029, tag genome +0.008, rating histograms +0.003). Bigger MLPs, more heads, deeper layers all hurt.
2. **Richer features unlock more capacity.** 4 GDCN layers and embed_dim=28 only work because histogram bins provide richer input. Feature quality shifts the overfitting threshold.
3. **Training procedure changes rarely work.** LR schedules, warmup, multi-task, contrastive, BPR, focal — all tried across 3 datasets, almost all failed.
4. **Tag genome works with learned compression, not PCA.** PCA-32 failed (0.798). Learned 3-layer bottleneck MLP succeeds (0.814). The sigmoid gate gracefully handles 78% missing data.
5. **NEG_RATIO is the hidden lever.** Reducing from 4→1 gave +0.005 AUC — the biggest HP-only gain. Fewer random negatives = cleaner signal focused on hard negatives.
6. **HP combinations stack.** NEG_RATIO + WD + ACCUM_STEPS + LR each contributed incrementally for +0.006 total.
7. **Time-valid easy negatives matter.** Replacing synthetic median-timestamp train negatives with anchored positive-event timestamps and catalog-valid sampled items improved the checked-in baseline to ~0.8238 on `ml-25m`.
8. **Field attention > GDCN.** 1-head MHA across 7 fields with residual slightly beats 4 gated cross layers (0.8207 vs 0.8201). Simpler and fewer parameters.
9. **Diverse ensemble is the breakthrough path.** 59 architecturally diverse models ensembled via HistGBM stacking: 0.854. Key: low prediction correlation between partners, not high individual AUC. GBM captures non-linear model complementarity that LogReg misses.
10. **10-trial HP sweeps per idea.** Never test an architecture idea once and discard. The NEG_RATIO breakthrough came from systematic HP sweep after the "ceiling" was declared.
11. **HistGBM >> LogReg for stacking.** LogReg: 0.836, MLP: 0.850, HistGBM: 0.854. Non-linear stacking is critical when models have diverse error patterns.
