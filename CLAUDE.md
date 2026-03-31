# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous research repo for improving pointwise movie recommendation (AUC) using DLRM-style models on MovieLens data. Follows the autoresearch loop defined in `program.md`.

## Commands

```bash
# Quick smoke test (ml-100k, ~seconds)
DATASET=ml-100k uv run train.py

# Standard experiment (ml-1m, default, ~minutes)
DATASET=ml-1m uv run train.py

# Full experiment run (redirected, for autoresearch loop)
DATASET=ml-1m uv run train.py > run.log 2>&1

# Check results
grep "^val_auc:\|^peak_memory_mb:" run.log
```

## Architecture

- **`prepare.py`** — Data download/loading (all MovieLens sizes), time-based train/val/test splits, AUC evaluation, `print_summary()`. May be modified when the model demands a different data setup (e.g. implicit feedback, negative sampling). Keep `evaluate()` and `print_summary()` stable.
- **`train.py`** — The experimentation file. Feature engineering, DLRM model, training loop. All modifications go here.
- **`program.md`** — The autoresearch protocol: setup, experiment loop, logging, research directions.
- **`results.tsv`** — Experiment log (untracked). Tab-separated: commit, val_auc, memory_mb, status, description.

## Key Details

- **Metric**: val_auc (higher is better). Binary label: rating >= 4 is positive.
- **Device**: MPS (Apple Silicon). Falls back to CUDA/CPU.
- **Time budget**: 10 minutes training wall clock per experiment (constant in `prepare.py`).
- **Datasets**: `ml-100k` (unit test), `ml-1m` (iteration), `ml-10m`, `ml-25m` (scale). Selected via `DATASET` env var.
- Data is auto-downloaded to `data/` on first use. Not checked into git.
