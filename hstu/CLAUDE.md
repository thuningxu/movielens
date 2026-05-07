# CLAUDE.md (hstu/)

This file provides guidance to Claude Code (claude.ai/code) when working in the **hstu/** subdirectory.

## Project Overview

Generative-recommendation attempt on MovieLens using **HSTU** (Hierarchical Sequential Transduction Units, Meta 2024 — *Actions Speak Louder than Words*). Same task and metric as the prior `legacy/` and `simple_v2/` attempts: predict whether a user will rate a movie ≥ 4 stars. BCE loss, val_auc on ml-25m at SEED=42 (deterministic).

The two prior attempts are archived as sibling subdirectories:
- `../legacy/` — DLRM-style architecture, **val 0.8284** ceiling.
- `../simple_v2/` — Linear head over engineered features, **val 0.8594 / test 0.8455** locked.

This attempt drops the engineered-feature framing entirely. HSTU treats the user as a sequence of (item, action, time) tokens and uses pointwise causal attention to predict next-event engagement. No hand-specified pools, no concat features, no cross fields.

**Final status (concluded)**: L=3 D=128 sliding is the operational best — **val 0.8626 (2-seed mean SEED=42-43, σ ≈ 0.0001)** beats simple_v2 0.8594 by **+0.0032**. **test 0.8652 (single-shot SEED=42)** beats simple_v2 0.8455 by **+0.0197**. The sliding-window mechanism (ceil(N/SEQ_LEN) non-overlapping windows per user) recovers the 54% of train events that the SEQ_LEN=100 truncation drops for heavy users on ml-25m.

After sliding, ten subsequent probes — extend-30, stride=50, InfoNCE, D=256, L=4, sinusoidal time bias, L=2, NUM_HEADS=8, MLP_HEAD off, kitchen-sink features — were all null or sub-noise. The may6 4-probe convergence sweep formally declared this configuration the HSTU ceiling on ml-25m. Future lifts would require structural changes (pre-trained text item embeddings, cross-dataset pre-training, BPR ranking loss, etc.), not incremental hyperparameter or feature probes.

## Commands (run from project root)

```bash
# Sync the repo-local environment
uv sync

# Smoke test (ml-100k, ~seconds — crash detection only, NOT for AUC comparison)
DATASET=ml-100k uv run python hstu/train.py

# PRODUCTION experiment (ml-25m, every-epoch eval, ~140 min for 20 epochs)
# Defaults are the may05 best config (val 0.8626 2-seed, test 0.8652):
# EMBED_DIM=128, NUM_LAYERS=3, INTERLEAVE=1, USE_BF16=1, GRAD_CLIP=1.0,
# PROJ_INIT_MODE=xavier, MLP_HEAD=1, USE_GENOME=USE_GENRE=USE_YEAR=1,
# SEQ_LEN=100, SLIDING_WINDOW=1, MAX_EPOCHS=20, LR=1e-3 constant.
DATASET=ml-25m uv run python hstu/train.py

# Add RUN_TEST=1 to additionally evaluate on the held-out test set
# (single-shot at the best-val checkpoint).
RUN_TEST=1 DATASET=ml-25m uv run python hstu/train.py

# FAST-ITERATION preset (~46 min for 20 epochs, ~3× over production):
# - USE_COMPILE=1: torch.compile per-block fusion (60s upfront, ~1.5× train+eval)
# - EVAL_EVERY_N_EPOCHS=5: eval at ep 4,9,14,19 + final (saves ~80% of eval time)
# - EVAL_BATCH_SIZE=2048: marginal but doesn't hurt
# Reported val_auc tracks final-epoch (peak across captured eval points), so it
# may be 0.0001-0.0003 below every-epoch peak.
USE_COMPILE=1 EVAL_EVERY_N_EPOCHS=5 EVAL_BATCH_SIZE=2048 DATASET=ml-25m uv run python hstu/train.py

# COMPILE-ONLY preset (~100 min for 20 epochs, ~1.5× over production):
# Keeps per-epoch eval (production-correct) but speeds up via torch.compile.
USE_COMPILE=1 DATASET=ml-25m uv run python hstu/train.py

# Reproduce the simple_v2 locked baseline (for cross-attempt comparison)
EVAL_DYNAMIC_HIST=1 FREQ_WD_LAMBDA=0 LR=1e-3 DATASET=ml-25m uv run python simple_v2/train.py

# CHECKPOINTING: save/resume/test-only flags. All default empty (byte-equivalent OFF).
# - CHECKPOINT_DIR=path: save model+optimizer+scheduler+best+RNG+config to path/last.pt
#   at every eval epoch. Use ./checkpoints/<run_name>/ — NOT /tmp (gets wiped on reboot).
# - RESUME=path/last.pt: load checkpoint and continue training from saved epoch+1.
# - TEST_FROM=path/last.pt: load checkpoint and run RUN_TEST eval ONLY (skip training).
CHECKPOINT_DIR=./checkpoints/myrun DATASET=ml-25m uv run python hstu/train.py
RESUME=./checkpoints/myrun/last.pt MAX_EPOCHS=30 DATASET=ml-25m uv run python hstu/train.py
TEST_FROM=./checkpoints/myrun/last.pt RUN_TEST=1 DATASET=ml-25m uv run python hstu/train.py
```

## Layout

- **`../prepare.py`** — Shared across all three attempts. `load_data()` returns raw `train`/`val`/`test` DataFrames with columns `userId, movieId, rating, timestamp, label`. `evaluate(labels, scores)` is the AUC ground-truth. **Do not modify the evaluation harness.**
- **`train.py`** — HSTU model + sequence data pipeline + training loop (~2400 lines). Includes content metadata (`USE_GENOME`/`USE_GENRE`/`USE_YEAR`), training stabilization (`GRAD_CLIP`, `PROJ_INIT_MODE`), MLP head (`MLP_HEAD`, `MLP_HEAD_DROPOUT`), interleaved tokens (`INTERLEAVE`), bf16 mixed precision (`USE_BF16`), LR schedule (`LR_SCHEDULE`, `WARMUP_STEPS`, `OPTIMIZER`), aux rating head (`AUX_RATING_WEIGHT`), held-out test eval (`RUN_TEST`), checkpointing (`CHECKPOINT_DIR`/`RESUME`/`TEST_FROM`) — flags default to ON for the operational-best stack and OFF for experimental probes; all OFF-states byte-equivalent.
- **`program.md`** — Full experiment log for this attempt (apr30 → may6).
- **`scripts/eval_strata.py`** — Post-hoc per-stratum AUC analysis on saved eval prediction CSVs.
- **`../data/`** — Auto-downloaded; gitignored.

## Key details

- **Metric**: val_auc on ml-25m (higher is better).
- **Label**: rating ≥ 4 → positive (1); rating < 4 OR random unrated → negative (0).
- **Final comparison**: HSTU **val 0.8626 / test 0.8652** vs simple_v2 **val 0.8594 / test 0.8455** = **+0.0032 val / +0.0197 test**. legacy DLRM ceiling was **val 0.8284**.
- **Device**: Single CUDA GPU. Auto-detects CUDA / MPS / CPU.
- **Datasets**: `ml-100k` (smoke test only), `ml-1m` (fast iteration), `ml-10m` (medium), `ml-25m` (default).
- **Reproducibility**: Deterministic at SEED=42. HSTU seed-noise floor σ ≈ 0.0001 from 2-seed verification at sliding regime.

## HSTU architecture

Per Meta 2024:

- **Token stream**: each user becomes a sequence `[(item_1, action_1, time_1), …, (item_N, action_N, time_N)]` sorted by timestamp. Interleaved variant emits 2N tokens with content/action alternating.
- **Embedding**: item + rating (+ optional content metadata: genome, genre, year).
- **HSTU block** (×3): pre-norm + Linear(D, 4D) split into U/V/Q/K → SiLU pointwise attention with log-bucketed time-delta bias → AV gated by U → residual.
- **Output head**: 2-layer MLP on h_t (D → 2D → D) → dot product against `item_full_embed(candidate)`.
- **Loss**: BCE on per-position next-event engagement (rating ≥ 4 binary label) at content positions only.

## Discipline (lessons carried into the project's conclusion)

- **Multi-seed verification is mandatory for any keep claim.** Single-seed lifts at this magnitude (≤ 0.001) routinely turn null on second seed.
- **`../prepare.py:evaluate()` is the ground truth.** Do not modify it.
- **More information beats more compute.** The only structural lift in this attempt (sliding window, +0.0033 val) recovered dropped events; every "more capacity" probe (D, L, heads, layers) was null. Future probes should target new information, not more model.
- **Cheap probes with null priors compound.** When 4 consecutive single-seed probes are null/regress, the operational best is the architectural ceiling on this dataset.
