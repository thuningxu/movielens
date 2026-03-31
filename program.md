# autoresearch — MovieLens Recommendation

Autonomous experimentation loop for improving pointwise recommendation (AUC) on MovieLens.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar31`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed: data download/loading, train/val/test splits, AUC evaluation, constants. Do not modify.
   - `train.py` — the file you modify. Feature engineering, model architecture, optimizer, training loop.
4. **Verify dependencies**: Run `uv run python -c "import torch; print(torch.backends.mps.is_available())"` to confirm MPS is available.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on MPS (Apple Silicon GPU). The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup and evaluation overhead). Launch it as:

```bash
DATASET=ml-1m uv run train.py > run.log 2>&1
```

**Dataset selection** via the `DATASET` env var:
- `ml-100k` — 100K ratings, for quick unit testing of code changes (~seconds)
- `ml-1m` — 1M ratings, default for experimentation (~minutes)
- `ml-10m` — 10M ratings, for larger-scale validation
- `ml-25m` — 25M ratings, full scale (if 10-min budget is insufficient, we move to NVIDIA GPU)

Use `ml-100k` to quickly validate that code changes don't crash, then `ml-1m` for real metric comparison.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: feature engineering, feature transformations, model architecture, optimizer, hyperparameters, training loop, batch size, model size, sequence modeling, negative sampling, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, splits, and constants.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_auc.** Since the time budget is fixed, you don't need to worry about training time — it's always 10 minutes. Everything is fair game: change the feature engineering, the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. M1 Max has 32–64 GB unified memory. Some increase is acceptable for meaningful AUC gains, but it should not OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_auc:          0.823456
val_logloss:      0.543210
training_seconds: 600.1
total_seconds:    615.3
peak_memory_mb:   2048.0
dataset:          ml-1m
num_users:        6040
num_items:        3706
num_train:        800168
num_params_M:     1.2
```

Extract the key metrics:

```bash
grep "^val_auc:\|^peak_memory_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_auc	memory_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_auc achieved (e.g. 0.823456) — use 0.000000 for crashes
3. peak memory in MB, round to .0f (e.g. 2048) — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_auc	memory_mb	status	description
a1b2c3d	0.823456	2048	keep	baseline DLRM
b2c3d4e	0.831200	2100	keep	increase embed_dim to 32
c3d4e5f	0.820100	2048	discard	remove history features
d4e5f6g	0.000000	0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar31`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: first do a quick smoke test with `DATASET=ml-100k uv run train.py > run.log 2>&1`, check it doesn't crash, then do the real run with `DATASET=ml-1m uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_auc:\|^peak_memory_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_auc improved (higher), you "advance" the branch, keeping the git commit
9. If val_auc is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~10 minutes of training (+ overhead for data loading and eval). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, MPS error, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes, try different feature engineering. The loop runs until the human interrupts you, period.

## Research directions to explore

Starting ideas (not exhaustive — use your judgment and creativity):

**Feature engineering:**
- Temporal features: hour of day, day of week, time since user's first/last rating
- User-genre affinity: user's average rating per genre
- Item popularity features: rating count, recency-weighted popularity
- Negative sampling: add unrated items as negatives alongside explicit ratings

**Model architecture:**
- Vary embedding dimensions, MLP widths/depths
- Attention-based history aggregation instead of mean pooling
- Multi-head self-attention over user history (transformer-style)
- DCN (Deep & Cross Network) style feature crossing
- Residual connections, layer normalization

**Training:**
- Learning rate schedules (warmup + cosine decay)
- Different optimizers (AdamW, SGD with momentum)
- Gradient clipping
- Label smoothing
- Focal loss for hard-example mining
