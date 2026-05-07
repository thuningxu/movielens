# MovieLens engagement prediction — three attempts

Predict whether a user will rate a movie ≥ 4 stars (positive engagement) on MovieLens. Same task and metric across three sequential attempts; each subdirectory holds one attempt as a frozen archive plus its own experiment log.

**Final result**: HSTU sliding-window — **val 0.8626, test 0.8652** on ml-25m at SEED=42.

## Headline comparison (ml-25m, SEED=42)

| Attempt | val_auc | test_auc | params | mechanism |
|---|---|---|---|---|
| `legacy/` (DLRM-style) | 0.8284 | — | — | Causal SA + DIN + tag-genome bottleneck + 4-layer MLP. ~540 experiments; ceiling confirmed by two ablations (apr27, apr27c). |
| `simple_v2/` (linear head) | 0.8594 | 0.8455 | 6.0M | Single Linear over engineered features (rating-centered history pools, multi-hot genres, raw tag genome, manual cross fields, eval-time dynamic-history). Locked at apr28ah after 5 consecutive nulls. |
| **`hstu/` (sequence model)** | **0.8626** (2-seed mean) | **0.8652** (single-shot) | 8.05M | HSTU (Meta 2024) over (item, action, time) tokens with sliding-window training. Beats simple_v2 by **+0.0032 val / +0.0197 test**. |

Each attempt's bar was the prior attempt's locked number. legacy was the first standalone result; simple_v2 cleared legacy on val by +0.031; HSTU cleared simple_v2 on val by +0.0032 and on test by +0.0197 — a much larger test-side gap because HSTU's sequence-summary representation generalizes through the test period (2018-01 to 2019-11) better than simple_v2's engineered concat (val→test gap +0.0026 vs −0.0139).

## The arc

**`legacy/` — DLRM-style architecture.** First serious attempt: causal self-attention over engineered features, DIN-style attention pool, tag-genome bottleneck, 4-layer MLP head. ~540 experiments tuning every hyperparameter and feature combination, but the architecture family saturated at val 0.8284. Two independent ceiling tests confirmed no further lift available within this design space.

**`simple_v2/` — linear-head restart.** Started over with the simplest possible model — single Linear over the same engineered features — to motivate future decisions with clean ablations rather than 540 experiments of inherited assumptions. Three structural changes from the static baseline (0.8221) lifted to 0.8594: **eval-time dynamic user history** (rebuild u_hist_pool at inference from train+val strictly prior to each row), **drop tail-item regularizer** (FREQ_WD_LAMBDA=0 — at the dynamic regime, tail items need larger embeddings to feed useful history signal), and **HP retune** (LR=1e-3 instead of static-era 3e-4). Locked at apr28ah after 5 consecutive nulls.

**`hstu/` — sequence model.** Dropped the engineered-feature framing entirely. Treated each user as a token stream of (item, action, time) events; HSTU's pointwise causal attention with log-bucketed time-delta bias predicts next-event engagement. Started below simple_v2 and climbed through capacity scaling (D=64→128, the diagnostic capacity flip closing the cold_user gap), content metadata (genome+genre+year zero-init projections), training stabilization (gradient clip + Xavier projection init), interleaved tokens (paper-canonical Meta 2024 §3), bf16 mixed precision, MLP head, then **sliding-window training** — the architectural unlock that recovered the 54% of train events the SEQ_LEN=100 truncation was dropping for heavy users. After sliding, ten subsequent probes (extend-30, stride=50, InfoNCE, D=256, L=4, sinusoidal time bias, L=2, NUM_HEADS=8, MLP_HEAD off, kitchen-sink features) were all null or sub-noise; the may6 4-probe convergence sweep formally declared the operational best the HSTU ceiling on ml-25m.

## Layout

```
movielens/
├── README.md       ← this file (project overview)
├── prepare.py      ← SHARED: load_data, evaluate. The metric is the ground truth.
├── pyproject.toml  ← shared environment
├── legacy/         ← attempt 1: DLRM, val 0.8284
├── simple_v2/      ← attempt 2: linear head + engineered features, val 0.8594 / test 0.8455
└── hstu/           ← attempt 3: sequence model (HSTU), val 0.8626 / test 0.8652
```

Each subdirectory has its own `train.py`, `program.md` (full experiment log), and `CLAUDE.md` (per-attempt working-context notes). Read the per-attempt files for details.

## Quickstart

```bash
uv sync

# HSTU (the final attempt — current best)
DATASET=ml-25m uv run python hstu/train.py
RUN_TEST=1 DATASET=ml-25m uv run python hstu/train.py   # also evaluate held-out test

# simple_v2 (linear head locked baseline)
EVAL_DYNAMIC_HIST=1 FREQ_WD_LAMBDA=0 LR=1e-3 DATASET=ml-25m uv run python simple_v2/train.py

# legacy (DLRM)
DATASET=ml-25m uv run python legacy/train.py
```

Smoke-test any of them on ml-100k (`DATASET=ml-100k`) for crash detection — runs in seconds, but the AUC is not comparable to ml-25m.

## Cross-cutting lessons

- **Multi-seed verification is mandatory for any keep claim.** Single-seed lifts at this magnitude (≤ 0.001) routinely turn null on second seed. Estimate the seed-noise floor before testing candidates.
- **More information beats more compute.** The biggest lifts in each attempt came from *what the model sees* (eval-time dynamic history in simple_v2; sliding-window event recovery in HSTU), not from more capacity, more epochs, or more aggressive optimization.
- **`prepare.py:evaluate()` is the ground truth.** Never modify it. Cross-attempt comparisons require identical metric and label semantics.
- **Architecture saturates faster than people think.** legacy hit its ceiling at 540 experiments; HSTU declared its ceiling at the may6 4-probe sweep. Keep score with explicit ceiling tests rather than perpetual hyperparameter sweeps.
- **Test-set generalization is the final arbiter.** simple_v2's engineered-concat representation overfits val (val→test gap −0.0139); HSTU's sequence-summary representation generalizes (val→test gap +0.0026). The same val_auc rank order as test_auc is not guaranteed for different model classes.

## Reproducibility

All three attempts are deterministic at SEED=42 on a single CUDA GPU. The reported val/test numbers are reproducible from a clean checkout (`uv sync` then the relevant command above). Per-attempt `program.md` contains the full cycle history including failed probes, so future re-attempts can avoid re-running the same dead-ends.
