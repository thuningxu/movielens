# autoresearch — MovieLens Recommendation (restart, apr28)

Fresh experimentation loop on top of a deliberately simple linear baseline. Same data + features + metric as the prior project at `legacy/`; new starting architecture and new experiment log.

## Why restart?

The prior project (see `legacy/`) reached val_auc = 0.8284 on ml-25m via several hundred experiments converging on a DLRM-style architecture, then plateaued. Rather than continue iterating in that local optimum, this restart begins from the simplest possible model (single Linear head on concatenated features), keeping the input features and prediction task unchanged, and rebuilds the architecture deliberately from below.

## Setup

- **Branch convention**: each experiment cycle on `restart/<tag>` (e.g., `restart/apr28` for the initial scaffold).
- **`prepare.py`**: do not modify (`evaluate()` is the ground-truth metric).
- **`train.py`**: the experimentation file. Currently the linear baseline; will grow.
- **Multi-seed discipline**: estimate the seed-noise floor empirically before declaring any win. A few baseline seeds give you σ; require multi-seed verification with mean lift comfortably above that floor and a positive sign at every seed.
- **Feature cache**: `data/features_<hash>.npz` is built on first ml-25m run and reused. Cache key is `restart-2` (includes per-position `hist_ts` for optional decay experiments).

## Current operating mode (autonomous loop)

Four roles per cycle:

1. **Researcher** proposes one concrete idea family.
2. **Critic** attacks it against the current code; tightens or vetoes.
3. **MLE** implements behind a feature flag with a byte-equivalent off-state (so the prior baseline still reproduces exactly).
4. **Validator** reviews the MLE diff against the surviving spec before any GPU sweep.

Operating rules:

- One idea family at a time, one GPU job at a time.
- `ml-100k` is for crash detection only; AUC comparisons use `ml-25m`.
- Every env-flag addition must preserve byte-equivalent default behavior — be especially careful with module construction order (PyTorch's RNG state advances on every `nn.Module.__init__` that draws random weights).
- Log every trial to `results.tsv` with `commit / val_auc / memory_mb / status / description`.

## Experiment log

Brief notes on cycles run on the restart. Detailed per-trial data lives in `results.tsv` and `sweep_*.log`.

### `autoresearch/apr28` — history-aggregation cycle

Starting baseline (post-scaffold, post-stripping, post-genre-projection drop): **val_auc = 0.821875** at SEED=42 with mean-pool over user/item history.

**Win — rating-centered pool stack** (`0e53f29`). Both `USER_HIST_POOL` and `ITEM_HIST_POOL` switched from `mean` to `rating_centered`: weight = `(rating − pivot) * is_valid`, normalize by sum of absolute weights. Items rated above the pivot push the pool *toward* their embedding; items below push *away*. Pivot = 0.6 (3 stars / 5).

5-seed multi-seed verification (SEED 42–46):
- USER alone: mean lift +0.001241, 5/5 positive, min +0.001097
- Stack (USER + ITEM): **mean lift +0.002579, 5/5 positive, min +0.002397** — current main baseline.

Why it works: signed weights encode preference direction, which a single Linear head over the resulting pool can directly exploit. Mean-pool throws away the sign.

Linear-baseline noise floor: **σ ≈ 0.00008** across SEED ∈ {42,43,44,45,46} — about 10× tighter than the legacy DLRM's σ ≈ 0.00078. Multi-seed bar is still ≥ +0.0007 mean lift with 5/5 positive (the historical convention), but smaller true effects are now resolvable.

**Cycle 1 — fixed-pivot sweep** (`7cf3adf`, sweep `cycle1_pivot.json`). Tested `POOL_PIVOT ∈ {0.5, 0.6, 0.7, 0.8}` × 5 seeds = 20 trials. Pivot=0.6 (current) is the optimum across all seeds; every other pivot regresses. Verdict: **no need to make the pivot learnable**, the median rating happens to be the right pivot.

**Cycle 2 — timestamp decay** (`2cc2e66` + `c9c551d`, sweep `cycle2_decay.json`). Added per-side learnable softplus-decay of position weights: `w *= exp(-softplus(theta) * (sample_ts − hist_ts) / ts_range)`. Cache extended to `restart-2` to store per-position timestamps. Init sweep at SEED=42 over θ ∈ {−2, −1, 0, +1, +2} (both sides, plus user-only/item-only at θ=0): all 7 trials within +1e-5 of the no-decay baseline — **null result**. Decay is correctly wired (path verified, AUC moves monotonically with init), but the lift is below the σ ≈ 0.00008 floor. Likely explanation: the right-aligned history window already enforces "recent" — within the last K positions, additional time-decay carries no extra signal beyond what the rating + recency-of-window already capture.

The decay infrastructure (env flags, cache fields, parameter construction) is left in place — defaults are byte-equivalent, infrastructure is available if a future cycle wants to combine decay with a different pool operator.

## Useful references

- `legacy/` — full history of the prior project, including its `program.md`, `results.tsv`, and `CLAUDE.md`. Available if useful; not authoritative for the restart.
