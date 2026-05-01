# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Generative-recommendation attempt on MovieLens using **HSTU** (Hierarchical Sequential Transduction Units, Meta 2024 — *Actions Speak Louder than Words*). Same task and metric as the prior `legacy/` and `simple_v2/` attempts: predict whether a user will rate a movie ≥ 4 stars. BCE loss, val_auc on ml-25m at SEED=42 (deterministic).

The two prior attempts are archived as subdirectories:
- `legacy/` — DLRM-style architecture, **val 0.8284** ceiling.
- `simple_v2/` — Linear head over engineered features, **val 0.8594 / test 0.8455** locked. See `simple_v2/CLAUDE.md` for the apr28 stack details.

This attempt drops the engineered-feature framing entirely. HSTU treats the user as a sequence of (item, action, time) tokens and uses pointwise causal attention to predict next-event engagement. No hand-specified pools, no concat features, no cross fields.

**Status**: stub. `train.py` has the layout and TODOs; no working model yet.

## Commands

```bash
# Sync the repo-local environment
uv sync

# Smoke test (ml-100k, ~seconds — crash detection only, NOT for AUC comparison)
DATASET=ml-100k uv run python train.py

# Standard experiment (ml-25m on the current CUDA GPU)
DATASET=ml-25m uv run python train.py

# Reproduce the simple_v2 locked baseline (for cross-attempt comparison)
EVAL_DYNAMIC_HIST=1 FREQ_WD_LAMBDA=0 LR=1e-3 DATASET=ml-25m uv run python simple_v2/train.py
```

## Layout

- **`prepare.py`** — Shared. `load_data()` returns raw `train`/`val`/`test` DataFrames with columns `userId, movieId, rating, timestamp, label`. `evaluate(labels, scores)` is the AUC ground-truth. **Do not modify the evaluation harness.**
- **`train.py`** — HSTU model + sequence data pipeline + training loop. Currently a stub.
- **`program.md`** — Experiment log for this attempt.
- **`legacy/`**, **`simple_v2/`** — Frozen archives. Each has its own `CLAUDE.md` and `program.md`.
- **`data/`** — Auto-downloaded; gitignored.

## Key details

- **Metric**: val_auc on ml-25m (higher is better).
- **Label**: rating ≥ 4 → positive (1); rating < 4 OR random unrated → negative (0).
- **Comparison points**: `simple_v2` locked baseline is **val 0.8594** / **test 0.8455** at SEED=42. legacy DLRM ceiling is **val 0.8284**.
- **Device**: Single CUDA GPU. Auto-detects CUDA / MPS / CPU.
- **Datasets**: `ml-100k` (smoke test only), `ml-1m` (fast iteration), `ml-10m` (medium), `ml-25m` (default).
- **Reproducibility**: Deterministic at SEED=42. Re-estimate seed-noise floor when the architecture stabilizes (simple_v2's lift-σ ≈ 0.003 at the dynamic-history regime is not transferable — HSTU has different variance characteristics).

## HSTU architecture (target)

Per Meta 2024:

- **Token stream**: each user becomes a sequence `[(item_1, action_1, time_1), …, (item_N, action_N, time_N)]` sorted by timestamp. Action = rating bucket (or binary engaged/not). Item = movieId. Time = relative time delta (or absolute, bucketed).
- **Embedding**: each token component embedded; sum or concat; layernorm.
- **HSTU block** (×L): pre-norm + gated linear unit (`SiLU(W1 x) ⊙ (W2 x)`) + relative-position-bias attention + residual. Causal mask.
- **Output head**: per-position MLP → P(engage at next step | history).
- **Loss**: BCE on next-event engagement label.
- **Eval**: at each (user, movie, ts) val sample, score the candidate movie by feeding the user's prior train+val events ending just before `ts` and reading out the head at the appended candidate position.

## Discipline (carried over from the prior attempts)

- **Multi-seed verification is mandatory for any keep claim.** Estimate the seed-noise floor before testing candidates; declare a win only when the lift is statistically distinguishable.
- **Smoke-test on ml-100k for crashes only**, not for AUC.
- **`prepare.py:evaluate()` is the ground truth.** Do not modify it.
- **Keep `train.py` simple while it's small.** Split into `model.py` / `data.py` / `train.py` only after the file grows past ~500 lines.
- **The simple_v2 baseline is the bar.** Any HSTU result must be measured against `val 0.8594` / `test 0.8455` to be a "win." Same eval harness, same split.
