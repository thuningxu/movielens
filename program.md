# autoresearch — MovieLens Recommendation (restart, apr28)

Fresh experimentation loop on top of a deliberately simple linear baseline. Same data + features + metric as the prior project at `legacy/`; new starting architecture and new experiment log.

## Why restart?

The prior project (see `legacy/`) reached val_auc = 0.8284 on ml-25m via several hundred experiments converging on a DLRM-style architecture, then plateaued. Rather than continue iterating in that local optimum, this restart begins from the simplest possible model (single Linear head on concatenated features), keeping the input features and prediction task unchanged, and rebuilds the architecture deliberately from below.

## Setup

- **Branch convention**: each experiment cycle on `restart/<tag>` (e.g., `restart/apr28` for the initial scaffold).
- **`prepare.py`**: do not modify (`evaluate()` is the ground-truth metric).
- **`train.py`**: the experimentation file. Currently the linear baseline; will grow.
- **Multi-seed discipline**: estimate the seed-noise floor empirically before declaring any win. A few baseline seeds give you σ; require multi-seed verification with mean lift comfortably above that floor and a positive sign at every seed.
- **Feature cache**: `data/features_<hash>.npz` is built on first ml-25m run and reused. The cache key currently matches the legacy format, so the existing cache is read directly.

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

## Useful references

- `legacy/` — full history of the prior project, including its `program.md`, `results.tsv`, and `CLAUDE.md`. Available if useful; not authoritative for the restart.
