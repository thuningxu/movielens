# autoresearch — MovieLens Recommendation (restart, apr28)

Fresh experimentation loop on top of a deliberately simple linear baseline. Same data + features + metric as the legacy project at `legacy/`; new starting architecture and new experiment log.

## Why restart?

Legacy (`legacy/program.md`) reached val_auc = 0.8284 on ml-25m via ~540 experiments converging on a DLRM-style architecture. Two ceiling tests (apr27 and apr27c) confirmed the architecture family is saturated for new architectural surface — every addition is sub-noise or negative. Apr27b extracted +0.0017 from HP retuning, but that's the only direction that still moves.

Rather than spend more time fighting that local optimum, we're starting from the simplest possible model (single Linear head on concatenated features), keeping the input features and prediction task unchanged, and rebuilding the architecture deliberately from below.

## Setup

- **Branch convention**: each experiment cycle on `restart/<tag>` (e.g., `restart/apr28` for the initial scaffold).
- **`prepare.py`**: shared with legacy, do not modify (`evaluate()` is the ground-truth metric).
- **`train.py`**: the experimentation file. Currently the linear baseline; will grow.
- **Multi-seed discipline**: σ ≈ 0.00078 across SEED ∈ {42–46} (legacy learning #14). Any keep requires 5-seed verification: mean lift ≥ +0.0007, 5/5 positive, min lift ≥ -0.0003.
- **Feature cache**: `data/features_<hash>.npz` is built on first ml-25m run and reused. The new `train.py` uses the same cache key as legacy, so it reads the existing cache (features are byte-identical).

## Current operating mode (autonomous loop)

Same 4-role pattern as legacy apr27+:

1. **Researcher** proposes one concrete idea family.
2. **Critic** attacks it against repo history (especially `legacy/program.md`) and current code; tightens or vetoes.
3. **MLE** implements behind a feature flag with byte-equivalent off-state (per legacy learning #11).
4. **Validator** reviews the MLE diff against the surviving spec before any GPU sweep.

Operating rules:

- One idea family at a time, one GPU job at a time.
- Apr27b methodology: 18-22 trial sweep + 4-seed multi-seed verify of any qualifier. Single-seed bar: lift ≥ max(2σ, +0.001) ≈ +0.0016. Multi-seed bar: 5-seed mean ≥ +0.0007, 5/5 positive.
- `ml-100k` is for crash detection only.
- All env-flag additions to `train.py` must preserve byte-equivalent default behavior (legacy learning #11).
- Architectural-graveyard items (apr27 cycles 1-10 dead axes, apr27c full transformer) are off-limits unless the restart has changed the calculus.

## Experiment log (restart/apr28) — ml-25m, linear baseline scaffold

**Goal**: establish the simplest possible baseline AUC as a known-good floor for future ablations.

**Setup**:
- Model: `concat(features) → Linear(in, 1) → sigmoid`. No hidden layer, no attention, no MLP head.
- Features: same as legacy (userId, movieId, user/item history mean-pool, genre, dense, raw genome 1128, raw user_genome 1128).
- Optimizer: Adam, LR=1e-3, WD=1e-5 (typical defaults; not tuned).
- Training: batch=16384, sub-epoch eval 3×, patience=3, max 20 epochs.
- HP defaults inherited from legacy where multi-seed-verified (NEG_RATIO=1, anchor_pos_catalog).

**Pending**: first ml-25m run.

## Research backlog

The next experiments after the baseline lands. Roughly in priority order; revisit after seeing the baseline AUC.

### High-prior families (substantively different from legacy)

1. **Two-tower retrieval-then-rerank** — separate user-tower and item-tower, dot-product score with a learnable bias. Different scoring shape from the legacy concat-MLP. Prior literature: well-known win on the same benchmarks where DLRM-style models hit a ceiling.
2. **Pure sequential model on user history** — a transformer encoder that takes `(item_id, rating, time-delta)` triples and predicts the next item; train as next-item prediction with the BCE label as a parallel head. Different objective, may unlock new gradient signal that DLRM-style models can't extract.
3. **Generative retrieval (TIGER-style)** — codebook-quantized item IDs, autoregressive prediction. Speculative; not on the critical path.
4. **External data via IMDB plot summaries** — the only clear path to genuinely new content signal. Legacy never tried this. Per legacy learning #1: new information > more capacity.

### HP / regularization ablations on the baseline

Once a single Linear baseline AUC is established, these tell us which legacy decisions actually matter for the simple model:

- LR / WD sweep around the linear-baseline defaults (LR=1e-3 / WD=1e-5 is unmotivated)
- `RECENCY_FRAC` sweep (legacy used 0.7; was it the linear baseline's optimum or just the apr27b joint optimum?)
- `NEG_RATIO` re-confirmation (legacy NEG_RATIO=1 was for the DLRM model)
- Embedding dim sweep (legacy converged on 28 for the DLRM; linear baseline may want different)
- Whether genome should be raw (1128) or compressed (per legacy learning #4, only learned bottleneck works)

### Things to NOT redo (graveyard from legacy)

See `legacy/program.md` for the full list. Highlights:

- PCA-compressed genome (failed at 0.798)
- NEG_RATIO=4 with global negatives (apr04 era; +0.005 worse than NEG_RATIO=1 + anchor-pos-catalog)
- LR schedules, warmup (consistently neutral or hurt across many experiments)
- Multi-task auxiliary losses (apr27 cycle 8: null)
- HSTU paper-faithful pointwise normalization (apr27 cycle 10: dead at all depths)
- Per-position genome similarity in rating-pool (apr27 cycle 9: dead, plus 1.7× wall-clock)

## Useful references

- `legacy/CLAUDE.md` — 16 distilled learnings from ~540 experiments
- `legacy/program.md` — full experiment history
- `legacy/results.tsv` — per-trial log (apr01 → apr27c)
- BARS benchmark (Zhu et al., SIGIR 2022) — different task (tag CTR) but good training practices
- HSTU paper: "Actions Speak Louder than Words" — Zhai et al., Meta, ICML 2024
- TIGER paper: "Recommender Systems with Generative Retrieval" — Rajput et al., Google, NeurIPS 2023
