# autoresearch — MovieLens Recommendation

Autonomous experimentation loop for improving pointwise recommendation (AUC) on MovieLens.

## Current operating mode (2026-04-25)

- This repository now runs on a single-GPU machine. Run at most one training job at a time.
- Older references below to `2x NVIDIA L4`, parallel agents, or simultaneous worktrees are historical logs, not the current operating protocol.
- Treat `train.py` as the source of truth for the checked-in baseline. Historical best remains **0.821 single-model** until re-run on this machine.

## Current autonomous loop (single GPU, 3 roles)

Use three roles every cycle, but serialize GPU usage:

1. **Researcher** proposes one concrete idea family, not a grab bag of unrelated tweaks.
2. **Critic** attacks the idea against repo history and current code, then either tightens it or vetoes it in favor of a stronger underexplored family.
3. **MLE** implements the surviving idea behind clean config flags, smoke tests it, then runs the real `ml-25m` evaluation sweep on the single GPU.

Operating rules:

- One idea family at a time.
- One GPU job at a time.
- Every serious idea gets up to **10 trials** (initial implementation plus targeted HP sweeps) before being declared dead, unless it fails fast and clearly.
- `ml-100k` is for crash detection only. Keep/discard decisions come from `ml-25m`.
- Judge families, not single lucky runs. Keep only if there is either a clear `>= +0.001` win over the true current baseline, or a credible ridge of nearby configs at `>= +0.0007`.
- Default first-class underexplored families are: missingness-aware genome fusion, causal-consistent history construction, and explicit recency/data-conditioning. Do not recycle historically dead families under new names.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr25`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed: data download/loading, train/val/test splits, AUC evaluation, constants. Do not modify.
   - `train.py` — the file you modify. Feature engineering, model architecture, optimizer, training loop.
4. **Sync the environment**: Run `uv sync` to create/update the repo-local environment from `pyproject.toml` and `uv.lock`.
5. **Verify dependencies**: Run `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"` to confirm CUDA is available and only one GPU is visible.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on the single available CUDA GPU. Do not launch concurrent training jobs or smoke tests on this machine. The current checked-in `train.py` uses early stopping with patience=3 evals and sub-epoch evaluation roughly 3x/epoch, but the code is the source of truth. Launch it as:

```bash
DATASET=ml-25m uv run python train.py 2>&1 | tee run.log
```

**Dataset selection** via the `DATASET` env var:
- `ml-100k` — 100K ratings, for quick smoke testing of code changes (~seconds)
- `ml-1m` — 1M ratings, fast iteration (~minutes)
- `ml-10m` — 10M ratings, medium scale (runtime depends on the current GPU)
- `ml-25m` — 25M ratings, **default for experimentation** (runtime depends on the current GPU)

Use `ml-100k` to quickly validate that code changes don't crash, then `ml-25m` for real metric comparison.

**What you CAN do:**
- Modify `train.py` — this is the primary file you edit. Everything is fair game: feature engineering, feature transformations, model architecture, optimizer, hyperparameters, training loop, batch size, model size, sequence modeling, negative sampling, etc.
- Modify `prepare.py` when the model demands a different training data setup (e.g. implicit feedback, negative sampling, different label definitions, new data splits). The evaluation function and summary printer should remain stable.

**What you CANNOT do:**
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_auc.** Training terminates via early stopping, so you don't need to worry about time budgets. Everything is fair game: change the feature engineering, the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing.

**Memory** is a soft constraint. Stay within the VRAM budget of the single available GPU. Historical runs fit on a 24 GB L4, but re-check peak memory on the current machine.

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

1. Pick one idea family from the backlog (or invent a new one).
2. Researcher proposes the family and the likely high-value sweep surface.
3. Critic attacks it using repo history and current code; narrow or replace the idea before touching the GPU.
4. MLE modifies `train.py` with the experimental idea, keeping the baseline path intact behind config flags when practical.
5. git commit.
6. Smoke test: `DATASET=ml-100k uv run python train.py` — check it doesn't crash.
7. Real run(s): `DATASET=ml-25m uv run python train.py 2>&1 | tee run.log`. Preserve every run log on disk while the full family sweep runs serially on the single GPU.
8. Read results: `grep "^val_auc:\|^peak_memory_mb:" run.log`
9. If grep is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix it after a few attempts, give up on this idea.
10. Record results in `results.tsv` (NOTE: do not commit results.tsv, leave it untracked by git).
11. Keep or discard the **family**, not just the first run. If the family shows a real improvement signal, keep the best commit and `git push` to upstream.
12. If the family is a dead end, discard only the experiment commit(s) you just made. Use `git reset --hard HEAD~1` only when the last commit is your own throwaway experiment and the worktree is otherwise clean; otherwise revert the specific edit safely.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Crashes**: If a run crashes (OOM, CUDA error, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes, try different feature engineering. The loop runs until the human interrupts you, period.

## Research backlog

Prioritized by expected impact and implementation difficulty. Reference: BARS/FuxiCTR MovielensLatest_x1 CTR leaderboard (AUC).

### Problem formulation history

Three formulations were tried. AUC is NOT comparable across them — different tasks.

| Formulation | Positive | Negative | Loss | AUC range |
|-------------|----------|----------|------|-----------|
| Explicit | rating >= 4 | rating < 4 | BCE | ~0.69 (ml-100k) |
| Implicit | any rating | random unrated | BPR | ~0.84 (ml-1m) |
| **Hybrid (current)** | rating >= 4 | rated < 4 (hard) + random unrated (easy) | BCE | ~0.79 (ml-1m), ~0.74 (ml-10m), **up to 0.821 historical single-model on ml-25m** |

The hybrid formulation is the current setup. It predicts "will user engage positively?" — suitable for front-page recommendation where you need calibrated probabilities and a threshold.

Reference: LightFM achieves ~0.86 (BPR) / ~0.90 (WARP) on ml-100k implicit feedback.

### Experiment log (autoresearch/mar31) — ml-1m

> **⚠ All AUC values below are on ml-1m (1M ratings, 6K users, 3.7K items). NOT comparable to ml-10m results.**
> Run-to-run variance was **~0.03 AUC** (no seed fixing at this point).

**Kept improvements** (cumulative, all architecture/feature changes):

| # | Experiment | AUC | Delta | Commit |
|---|-----------|-----|-------|--------|
| 0 | Baseline: hybrid BCE, DLRM | 0.756 | — | e8e0f07 |
| 4 | DIN attention for user history | 0.767 | +0.011 | ae4ccd8 |
| 6 | User-genre affinity dot product feature | 0.777 | +0.010 | e8130cd |
| 7 | DCN-V2: 2 cross layers | 0.788 | +0.011 | 997ee5b |
| 15 | Wider bottom MLP (128) + deeper top MLP (3 layers) | 0.790 | +0.002 | 592170e |

**Discarded experiments** (all hurt or were within noise):

| # | Experiment | AUC | Why it failed |
|---|-----------|-----|---------------|
| 1 | Online negative sampling | 0.757 | No improvement over fixed |
| 2 | LR=5e-4 + AdamW + cosine + wider MLP | 0.642 | LR way too high |
| 3 | LR=2e-4 + cosine decay | 0.719 | Collapsed at epoch 3 |
| 5 | Hour-of-day temporal features | 0.696 | Hurt significantly |
| 8 | DCN-V2 3 cross layers | 0.772 | Too much capacity |
| 9 | Dropout 0.4 | 0.787 | No change |
| 10 | Embed_dim=32 | 0.779 | More overfitting |
| 11 | Batch_size=2048 | 0.771 | Worse and slower |
| 12 | NEG_RATIO=8 | 0.785 | No improvement |
| 13 | FinalNet field gate | 0.743 | Overfits badly |
| 14 | Popularity-biased neg sampling | 0.718 | Task too hard |
| 16 | HISTORY_LEN=100 | 0.784 | Dilutes attention |
| 17 | Movie release year feature | 0.785 | No improvement |
| 18 | Residual connections in top MLP | 0.784 | No improvement |
| 19 | Weight_decay=1e-3 | 0.773 | Too strong |
| 20 | Multi-head DIN (2 heads) | 0.791 | Only 1 epoch ran (too slow), likely noise |
| 21 | Deeper DIN attention (3-layer) | 0.751 | 16min/epoch, unusable |
| 22 | NEG_RATIO=2 | 0.775 | Only 1 epoch (multi-head was still loaded) |

**Tier 1 ideas also tested via parallel agents (all hurt on ml-1m, not re-tested on ml-10m):**

| Experiment | AUC | Why |
|-----------|-----|-----|
| ReduceLROnPlateau (factor=0.5, patience=3) | 0.758 | LR decayed too aggressively before convergence |
| BatchNorm in MLPs | 0.676 | Hurt significantly, changed loss landscape |
| Embedding L2 reg (lambda=0.001) | 0.753 | Full-table norm penalty too strong |
| Gradient clipping (max_norm=1.0) | 0.741 | No benefit |
| Focal loss (gamma=2) | 0.726 | Over-suppresses easy negatives |

### Experiment log (autoresearch/apr01) — ml-10m

> **All AUC values below are on ml-10m (10M ratings, 70K users, 10.7K items). Deterministic (SEED=42).**

**Kept improvements** (cumulative):

| # | Experiment | AUC | Delta | Commit |
|---|-----------|-----|-------|--------|
| 0 | Baseline (sub-epoch eval, embed dropout 0.1, patience=5) | 0.692 | — | 7a93a21 |
| 1 | Movie year + genre count + movie age dense features | 0.705 | +0.013 | 6c4cfc0 |
| 2 | GDCN gated cross layers | 0.710 | +0.005 | af84445 |
| 3 | User hist ratings + item-side DIN over recent raters | 0.739 | +0.029 | 870042b |

**Discarded on ml-10m (46 experiments, all at 0.689-0.747):**
- Embed_dim=32 + wider MLP + dropout 0.3 — 0.696 (overfits)
- Multi-task aux rating MSE — 0.708 (no improvement)
- LR=5e-5 — 0.717 (too slow) | LR=2e-4 — 0.744 (slightly worse)
- Separate embed LR 3x — 0.723 (worse)
- DIN + GRU dual history — 0.725 (slower and worse)
- NEG_RATIO=2 — 0.733 | NEG_RATIO=6 — 0.745 (no improvement)
- Embed dropout 0.15 — 0.738 | 0.2 — 0.718 (0.1 is better)
- Batch_size=16384 — 0.731 (too few steps)
- MaskNet instance masking — 0.741 (no improvement)
- History embed dropout 0.1 — 0.718 (hurts DIN)
- MLP dropout 0.3 — 0.723 (too strong)
- Eval 4x/epoch + patience=8 — 0.725 (worse)
- BST 1-layer transformer — 0.740 (slower, no gain)
- HISTORY_LEN=100 — 0.722 (dilutes attention)
- AdamW wd=0.01 — 0.712 (too strong)
- EMA decay=0.999 — 0.738 (no improvement)
- Feature-level dropout 0.1 — 0.735 (too aggressive)
- bf16 AMP — 0.731 (worse than fp16)
- Embed_dim=32 with dropout — 0.708 (still overfits)
- 3 GDCN cross layers — 0.719 (too much capacity)
- DuoRec contrastive — 0.734 | In-batch negatives — 0.734 (no improvement)
- MoE 4 experts — 0.689 (overfits badly)
- FM pairwise field interactions — 0.699 (hurts)
- ITEM_HIST_LEN=50 — 0.742 | 20 — 0.747 (30 is sweet spot)
- Wider top MLP 512-256-64 — 0.720 (overfits)
- Residual skip x0 — 0.736 | Deeper DIN 3-layer — 0.725 (overfits)
- Multi-head DIN 2 heads — 0.737 (no improvement)
- LR warmup 500 steps — 0.744 | Cross layer dropout — 0.745 (no improvement)
- User-item MF dot product — 0.723 (overfits)
- LightGCN random init — 0.734 (no useful signal)
- Label smoothing 0.2 — 0.747 (0.1 is better)
- SiLU/Swish — 0.741 (worse than ReLU)
- Deeper bottom MLP — 0.717 (overfits)
- ACCUM_STEPS=4 — 0.746 (too few updates)
- History masking augmentation — 0.747 (no change)
- Cross-attention user/item hist — 0.729 (hurts)
- Genre-enriched DIN — 0.730 (redundant)
- User-genre affinity vector field — 0.720 (overfits)

### Experiment log (autoresearch/apr01) — ml-25m

> **All AUC values below are on ml-25m (25M ratings, 162K users, 59K items). Deterministic (SEED=42).**

**Kept improvements** (cumulative, building on ml-10m model):

| # | Experiment | AUC | Delta | Commit |
|---|-----------|-----|-------|--------|
| 0 | ml-25m baseline (ml-10m model as-is) | 0.770 | — | d731df0 |
| 1 | HISTORY_LEN=100 | 0.771 | +0.001 | a4d2399 |
| 2 | 3 GDCN cross layers | 0.781 | +0.010 | e6b7bf6 |
| 3 | patience=2, TF32 tensor cores | 0.781 | speed | 306ae72 |
| 4 | embed_dim=24 | 0.793 | +0.012 | 90732b8 |
| 5 | batch_size=16384 | 0.796 | +0.003 | fde1deb |
| 6 | ACCUM_STEPS=4→8 | 0.798 | +0.002 | 70ebdeb |
| 7 | wider stream MLPs (256-64) | 0.798 | +0.001 | c2d8a81 |
| 8 | Lightweight causal self-attention before DIN | 0.799 | +0.001 | 15cf247 |

**Discarded on ml-25m (22 experiments, all at 0.770-0.799):**
- embed_dim=32 — 0.778 (slight overfit even with 25m data)
- embed_dim=32 + dropout 0.2 — 0.792 (still worse than dim=24)
- embed_dim=20 — 0.789 (24 is better)
- embed dropout 0.05 — 0.770 (0.1 is better)
- 4 GDCN cross layers — 0.742 (too much capacity)
- ITEM_HIST_LEN=50 — 0.793 (no change from 30)
- BST 1-layer Transformer — 0.778 (4x slower, no gain)
- 2-head causal attention — 0.799 (same AUC, 2.5x slower)
- Item-side causal attention — 0.791 (slower and worse)
- LR=2e-4 — 0.771 (worse)
- MLP dropout 0.3 — 0.796 (no improvement over 0.2)
- Tag genome features (1128-dim) — 0.798 (no improvement, 23% coverage)
- Multi-task rating MSE — 0.797 (no improvement)
- NEG_RATIO=6 — 0.798 (no improvement)
- weight_decay=1e-4 — 0.792 (too strong)
- LightGCN BPR (training failed to converge) — 0.797
- Wider DIN attention (128 hidden) — 0.797 (worse)
- User + item bias terms — 0.792 (overfits)
- Residual top MLP — 0.793 (worse and slower)
- LayerNorm after cross-network — 0.795 (hurts)
- Learnable DIN temperature — 0.799 (no change)
- Cosine annealing LR — 0.799 (no change)

### Experiment log (autoresearch/apr02) — ml-25m

> **All AUC values below are on ml-25m. Deterministic (SEED=42). 2x NVIDIA L4, parallel experiments.**

**Kept improvements** (cumulative, building on apr01 model at 0.799):

| # | Experiment | AUC | Delta | Commit |
|---|-----------|-----|-------|--------|
| 0 | Baseline (reproduce 0.799) | 0.799 | — | 1fadb15 |
| 1 | Rating histogram 5-bin features replacing mean+std | 0.802 | +0.003 | c6b5576 |
| 2 | 4 GDCN gated cross layers | 0.804 | +0.002 | 2f05af1 |
| 3 | embed_dim=28 | 0.806 | +0.002 | cc40190 |

**Discarded on ml-25m (37 architecture experiments, all at 0.786-0.806):**
- DIEN AUGRU replacing self-attn+DIN — 0.786 (slower and worse)
- Gated fusion replacing concat+top_mlp — 0.779 (information bottleneck)
- Position embeddings in DIN — 0.794 (causal attn already encodes order)
- Time-gap features in DIN — 0.799 (neutral, ts already a dense feature)
- Popularity trajectory features — 0.799 (redundant with existing)
- Separate embed LR 3x — 0.800 (marginal, unstable later epochs)
- EulerNet complex-valued interactions — 0.799 (same space as GDCN)
- Contrastive loss on stream outputs — 0.799 (training procedure)
- Multi-scale DIN short(10)+long(100) — 0.787 (redundant capacity)
- RankMixer MLP-Mixer replacing GDCN — 0.799 (worse interaction modeling)
- Histogram dot+cosine similarity — 0.804 (not reproducible at 0.804)
- Deeper bottom MLP 3-layer — 0.791 (overfits)
- Histogram bins + dot + cosine (19 dense) — 0.796 (too many features)
- Co-rating density feature — 0.799 (redundant with counts)
- NEG_RATIO=3 — 0.800 (fewer negatives hurts discrimination)
- Wider bottom MLP 256 — 0.789 (overfits)
- ACCUM_STEPS=16 — 0.799 (too few updates)
- User-genre cross-attention — 0.799 (redundant with ug_dot)
- Decade one-hot replacing scalar year — 0.799 (lost granularity)
- Eval 3x/epoch + patience=3 — 0.799 (same peak, just later stop)
- Label smoothing 0.05 — 0.800 (0.1 is better)
- Freeze embeds after epoch 1 — 0.799 (early stop before epoch 1)
- LR=5e-5 — 0.802 (same AUC, 2x slower)
- LR=7e-5 + patience=3 — 0.802 (same AUC, slower)
- All dropout 0.15 — 0.802 (within noise)
- Item histogram in item-DIN query — 0.799 (redundant with dense)
- Bilinear histogram interaction — 0.800 (overfits)
- NEG_RATIO=5 — 0.801 (marginal worse)
- 5 GDCN cross layers — 0.800 (too deep)
- embed_dim=32 + embed dropout 0.15 — 0.802 (still overfits)
- ITEM_HIST_LEN=50 — 0.806 (no change, +4GB VRAM)
- HISTORY_LEN=150 — 0.806 (no change, +4GB VRAM)
- 2-head causal attention — 0.799 (worse)
- embed_dim=26 — 0.798 (28 is sweet spot)
- embed_dim=30 — 0.803 (slight overfit)
- ACCUM_STEPS=4 — 0.805 (slightly worse)
- Wider stream MLP 384→64 — 0.800 (overfits)

**Feature-focused experiments (autoresearch/apr02b, 12 experiments, all ≤0.806):**
- SA on combined item+rating — 0.797 (2D→D projection bottleneck)
- Residual SA raw+contextual — 0.806 (neutral)
- SVD-32 collaborative factors as 7th field — 0.802 (redundant with ID embeddings)
- Kaiming He init — 0.801 (NaN instability, worse than Xavier)
- Tag genome PCA-32 as 7th field — 0.798 (22% coverage, genre imputation insufficient)
- Title char-trigram hash 64d (dense) — OOM (81 dense features too large)
- Title char-trigram hash 64d (field) — 0.802 (no useful text signal from titles)
- JS divergence + user entropy — 0.795 (derived interaction features hurt)
- User velocity + drift + recency — 0.787 (temporal dynamics hurt significantly)
- SVD dot product scalar — 0.779 (MF prediction interferes with learned representations)
- Item polarization — 0.804 (derived from histogram, slight hurt)
- Genre-aware DIN target (item_e+genre_e) — 0.806 (neutral)

### Key learnings

1. **More data helps significantly.** ml-10m→ml-25m gave +0.023 AUC for free (same model). Ideas that failed on ml-10m (3 GDCN layers, HISTORY_LEN=100, batch_size=16384, embed_dim=24) all worked on ml-25m.

2. **Scale unlocks capacity.** 3 GDCN layers (+0.010) and embed_dim=24 (+0.012) were the biggest wins on ml-25m, both of which hurt on ml-10m. The overfitting threshold shifts with data size.

3. **Item-side DIN is the single biggest architectural win.** Attention over recent raters (+0.029 on ml-10m). Captures collaborative signal that static features can't.

4. **Feature enrichment works.** Movie year, genre count, movie age (+0.013), user history ratings in DIN, item history ratings. Enriching both user and item towers was critical.

5. **Architecture changes work, training procedure changes don't.** GDCN gates, DIN variants, FinalMLP streams, bilinear all helped. LR schedules, contrastive losses, EMA, multi-task, warmup all hurt.

6. **Fixed random seeds are essential.** Variance was ~0.05 AUC before seeding. After SEED=42, variance <0.001.

7. **fp16 > bf16 for attention-heavy models.** bf16 hurts by ~0.006-0.008 AUC.

8. **Transformers don't help (yet).** BST was 4x slower and no better than DIN on both ml-10m and ml-25m. The DIN attention mechanism is already very effective for this task.

9. **Batch size matters on large datasets.** batch_size=16384 + ACCUM_STEPS=8 (effective 131K) helped on ml-25m but hurt on ml-10m.

10. **Rating histograms > summary statistics.** 5-bin distributions (+0.003) capture more than mean+std. The full distribution shape matters for engagement prediction.

11. **Richer features unlock deeper/wider models.** 4 GDCN layers failed at 0.742 before histogram bins, succeeded at 0.804 with them. embed_dim=28 works with histograms+4GDCN (0.806) but 30/32 still overfit. Feature quality shifts the capacity threshold.

12. **Most experiments are neutral, not harmful.** At 0.799-0.806, ~90% of changes return within ±0.002 of baseline. The model is extremely well-optimized and resistant to perturbation.

13. **MovieLens metadata is exhausted.** Tag genome (22% coverage), movie titles (char trigrams), SVD factors — none provided useful signal beyond existing features. The dataset lacks rich content metadata.

14. **Derived features from existing data hurt.** JS divergence, entropy, polarization, temporal dynamics — all derived from existing histograms/timestamps — either added noise or were redundant. The model already extracts these patterns from raw features.

15. **SVD/MF predictions interfere with end-to-end learning.** SVD dot product (0.779) and full SVD factors (0.802) both hurt. The pre-computed collaborative signal conflicts with the jointly trained embeddings rather than complementing them.

### Historical apr03 backlog (on GPU, ml-25m, baseline 0.806 at the time)

> After ~130 experiments across 4 sessions, the model appears near-saturated on MovieLens data.
> Most remaining ideas are high-risk/low-probability. External data would likely be needed for significant gains.

#### Tried in apr03 (18 experiments, all neutral or worse at 0.806)
- SWA (averaged 4 checkpoints) — 0.805 (worse than best single checkpoint)
- DropPath 0.1 on GDCN deltas — 0.806 (neutral)
- xDeepFM CIN 2-layer — 0.790 (explicit crosses overfit badly)
- Trained LightGCN BPR embeddings — crash (sparse matmul too slow on 25m)
- Poly-1 loss — 0.806 (neutral)
- Snapshot averaging (avg top-2 weights) — 0.806 (2nd best too weak)
- Asymmetric loss pos_weight=2.0 — 0.805 (neutral)
- R-Drop KL regularization — 0.806 (neutral; 2x slower)
- Embedding mixup alpha=0.2 — 0.806 (neutral; 2x slower)
- Wide & Deep (raw embeds in top MLP) — 0.804 (wide path adds noise)
- LR=2e-4 — 0.803 (too aggressive; collapses epoch 2)
- weight_decay=1e-4 — 0.800 (too strong)
- batch_size=8192 — 0.805 (faster divergence)
- LR=5e-5 + patience=3 — 0.806 (same peak; 2x slower)
- ACCUM_STEPS=12 eff. batch 197K — 0.806 (neutral)
- NEG_RATIO=8 — 0.796 (too many negatives dilutes signal)
- 3-stream FinalMLP (interaction stream) — 0.802 (extra capacity overfits)

#### Untried ideas (as of apr03, now mostly tried — see apr04/apr05 logs)
- ~~PinSage~~ — LightGCN failed; PinSage untried but GNN signal is redundant with DIN history
- ~~HSTU~~ — SASRec transformer tried and failed (0.80-0.81); HSTU might be different but risky
- ~~External data~~ — Still untried; only clear path to genuinely new content signal

---

The next two sections document an older 2-GPU workflow. Keep them as historical experiment logs only; do not follow them on the current single-GPU machine.

### Experiment log (autoresearch/apr03c) — ml-25m, multi-agent setup

> **Multi-agent parallel experimentation with deeper tuning.**
> Target: 0.810 AUC (from 0.806 baseline).
> Hardware: 2x NVIDIA L4 (CUDA:0, CUDA:1), parallel experiments.

#### Multi-agent protocol

**Agents:**
- **Research Scientist** (coordinator) — proposes ideas, reviews results, directs engineers, decides keep/discard after full tuning budget
- **MLE-1** (GPU 0, `CUDA_VISIBLE_DEVICES=0`) — executes experiments in git worktree
- **MLE-2** (GPU 1, `CUDA_VISIBLE_DEVICES=1`) — executes experiments in git worktree

**Workflow (kanban-style):**
1. Researcher proposes 2 experiment ideas with initial hyperparameters
2. MLE-1 and MLE-2 each take one idea, work in isolated git worktrees
3. Each MLE: modify train.py → smoke test (ml-100k) → full run (ml-25m) → report results
4. Researcher reviews results and decides:
   - If promising (within 0.005 of baseline or better): assign HP tuning variations
   - If clearly broken (>0.015 below baseline): discard and assign new idea
   - If improved: keep, merge to branch
5. **Tuning budget: up to 10 trials per architecture idea** (initial + 9 HP variations) before declaring it invalid
6. After tuning budget exhausted, researcher decides keep best or discard all

**Key philosophy change from previous sessions:**
- Previously: try one config, immediately discard if no AUC gain
- Now: give each architecture idea a fair chance with proper HP tuning
- Challenge previous "exhausted" conclusions — revisit with better strategies
- The 0.806 plateau may be an artifact of under-tuning, not a true ceiling

**Git workflow:**
- Each MLE works in a temporary git worktree (isolated copy)
- Successful experiments get merged back to `autoresearch/apr03c`
- Failed experiments: worktree discarded, no trace on branch
- Results logged to results.tsv (untracked)

#### Research agenda (apr03c)

**Round 1: New data sources (challenging "metadata exhausted")**
1. **User-generated tags (tags.csv)** — 1.09M tags, 72.5% movie coverage (vs 22% genome). Untried data source. Derive: tag hash features, tag popularity, user-tag behavior.
2. **Tag genome with learned attention compression** — Full 1128-dim relevance vectors. Previous PCA-32 attempt (0.798) used crude dimensionality reduction. Try: attention pooling over genome dimensions, proper missing-data embedding.

**Round 2+ (depending on Round 1 results):**
- Combine tag sources if both help
- Recency-weighted DIN with temporal decay
- Two-phase training (freeze embeddings → fine-tune)
- embed_dim=32 with spectral normalization
- HSTU-style sequential architecture

#### Experiment results

**Kept improvements** (cumulative, building on 0.806 baseline):

| # | Experiment | AUC | Delta | Commit |
|---|-----------|-----|-------|--------|
| R1-2 | Tag genome learned MLP compression (1128→128→D) + sigmoid gate | 0.8108 | +0.005 | 81f9e60 |
| R2-1 | Deeper genome bottleneck (1128→256→64→D) 3-layer | 0.8138 | +0.003 | de28ad8 |

**Discarded (10 experiments):**
- R1-1: User tags top-200 multi-hot + tag_count + user_tag_dot as 7th field — 0.8071 (marginal)
- R2-1a: Wider genome (1128→256→D) — 0.8114 (marginal over 0.8108)
- R2-2: User tags stacked on genome as 8th field — 0.8008 (overfits badly)
- R3-1: 4-layer genome bottleneck (512→128→64→D) — 0.7985 (overfits)
- R3-2: Genome dropout 0.2 — 0.8059 (too strong)
- R3-3: Genome no detach on gate — 0.8059 (no change)
- R3-4: User tags as 3 dense features — 0.8105 (slight regression)

**Key findings:**
- Tag genome with learned compression is the breakthrough — PCA failed (0.798) but learned MLP succeeds (0.811-0.814)
- 3-layer bottleneck (256→64→D) beats 2-layer (128→D) — compression forces better feature extraction
- 4-layer is too deep (overfits). The sweet spot is 3 layers.
- User-generated tags provide no useful signal in any form (field, dense, or combined with genome)
- The sigmoid gate for missing-data fallback is critical (22% coverage handled gracefully)

---

### Experiment log (autoresearch/apr03d) — ml-25m, multi-agent v2

> **4-agent setup with research critic. Target: 0.820 AUC (from 0.8138 baseline).**
> Hardware: 2x NVIDIA L4 (CUDA:0, CUDA:1), parallel experiments.

#### Multi-agent protocol v2

**Agents:**
- **Research Scientist** (coordinator) — proposes experiment ideas with rationale
- **Research Critic** (reviewer) — challenges proposals before execution, reviews results for sanity, suggests refinements
- **MLE-1** (GPU 0) — executes experiments in git worktree
- **MLE-2** (GPU 1) — executes experiments in git worktree

**Workflow:**
1. Researcher proposes 2 experiment ideas with rationale
2. **Critic reviews proposals** — challenges assumptions, identifies risks, suggests alternatives or modifications
3. Researcher revises plan based on critique
4. MLE-1 and MLE-2 execute (parallel worktrees)
5. Researcher + Critic review results together
6. Keep/tune/discard decision with up to 10 trials per idea

**Why the critic role:**
- Pre-screens ideas to avoid wasting GPU time on doomed experiments
- Catches inconsistencies in results (e.g., wrong baseline, worktree issues)
- Provides adversarial pressure on idea quality
- Lightweight: analysis only, no code execution

#### Experiment results

**apr03d (6 experiments, all ≤0.8139):**
- Cross-network projection, genome-gated DIN, attention pooling — all neutral/worse

---

### Experiment log (autoresearch/apr03e) — ml-25m, HP sweep breakthrough

> **0.8138 → 0.8201 AUC via systematic HP tuning. 80 experiments.**
> Key discovery: NEG_RATIO=1 is the hidden lever (+0.005 from NEG_RATIO=4).

**Kept improvements** (cumulative):

| # | Experiment | AUC | Delta |
|---|-----------|-----|-------|
| 1 | NEG_RATIO=2, PATIENCE=3, eval 3x/epoch | 0.8171 | +0.003 |
| 2 | NEG_RATIO=1 | 0.8183 | +0.001 |
| 3 | WD=5e-5, ACCUM_STEPS=4 | 0.8197 | +0.001 |
| 4 | LR=8e-5 | 0.8201 | +0.000 |

**Learnings:**
- NEG_RATIO 4→2→1 monotonically improves AUC. Fewer random negatives = cleaner training signal.
- HP combinations stack: NEG_RATIO + WD + ACCUM + LR = +0.006 total.
- Architecture is saturated at 0.8201: 80 experiments (attention, cross-net, features, loss) all neutral.
- The 0.8138 "ceiling" was a tuning problem, not information-theoretic.
- R2-2: Attention-based genome pooling (item-aware softmax over 1128 dims) — 0.8008 (worse)

**Critic-driven findings:**
- The Critic pre-screened and killed 2 proposals (embed_dim=32 and temporal decay) that would have wasted GPU time based on prior failures
- Attention pooling is strictly worse than MLP bottleneck for genome compression
- Honest assessment: 0.820 is not achievable without external data. Realistic ceiling is ~0.815.

**Key learnings (apr03d):**
16. **The Critic role saves GPU time.** Pre-screening killed embed_dim=32 (failed 4x) and temporal decay (failed 2x) before wasting runs. Validated prior experiment conclusions hold.
17. **MLP bottleneck > attention for genome compression.** Softmax over 1128 dims spreads too thin. Fixed bottleneck (256→64→D) forces useful compression.
18. **0.8138 appears to be near the ceiling for MovieLens-25m with this task formulation.** After 175+ experiments, all architectural/training changes are neutral (±0.002). Only genuinely new data (tag genome) moved the needle.

---

### Experiment log (autoresearch/apr04) — ml-25m, architecture sweep

> **0.8210 single model. ~200 experiments.**

**Kept single-model improvements** (building on 0.8201):

| # | Experiment | AUC | Delta |
|---|-----------|-----|-------|
| 1 | Replace GDCN with 1-head field attention, WD=3e-5 | 0.8207 | +0.001 |
| 2 | History residual (raw + contextual), WD=5e-5 | 0.8210 | +0.000 |

**Discarded architecture ideas (~200 experiments, all ≤0.8210 single model):**
- SASRec transformer (10 trials): 0.803-0.813 (strictly worse than causal SA + DIN)
- GRU/DIEN (5 trials): 0.811-0.814 (worse and 4x slower)
- BPR ranking loss (5 trials): 0.817-0.820 (worse than BCE)
- LightGCN pre-trained embeddings (4 trials): 0.811-0.817 (redundant with DIN history)
- Soft labels from ratings (5 trials): 0.815-0.819 (worse)
- Multi-task rating regression (5 trials): 0.815-0.816 (worse)
- Genre-enriched DIN (10 trials): 0.815-0.821 (neutral, genre already captured)
- User-genre dense features (10 trials): 0.815-0.818 (worse, ug_dot already optimal)
- Cross-attention user/item fields: 0.816 (worse)
- Separate user/item field attention: 0.817 (worse)
- Wider/deeper streams, wider top MLP: all worse
- Label threshold 3.5/4.5: 0.821/0.754 (4.0 is optimal)
- Focal loss, sample weighting, popularity negatives: all worse
- Cosine annealing, warmup, SWA: all neutral
- Separate LR for embeddings/MLPs: all worse
- Data augmentation (masking, noise, dropout): all neutral
- Different seeds: negligible variance
- 90/5/5 split: not comparable (different eval set)

**Key learnings (apr04):**
24. **Field attention is a slight improvement over GDCN.** 1-head MHA across 7 fields beats 4 gated cross layers (0.8207 vs 0.8201). Simpler, fewer params, slightly better.
25. **SASRec/transformers are worse than DIN.** 10 trials of transformer-based sequence models all regressed -0.01 to -0.02. The lightweight causal SA + DIN combination is better suited to this task.
26. **GRU/DIEN is worse than causal SA.** Sequential processing loses to parallel attention on this data.
27. **The single-model architecture is saturated.** After 200 architecture experiments at 0.8210, no modification improves. The model topology is well-matched to MovieLens features.

---

### Experiment log (autoresearch/apr05) — ml-25m, single-model exploration

> **No further single-model improvement over 0.8210. ~20 experiments.**

**Single-model results (no improvement over 0.8210):**
- Temporal features (hour/day): hurt (0.816)
- User activity rate: neutral (0.819)
- Item trend, time gap, rating variance: all hurt
- Asymmetric DIN: 0.820 (closest, not significant)
- 2-layer field attention, deeper bottom MLP, higher LR: all worse

### What to try next (backlog for future sessions)

> Historical checked-in best: single model 0.821.
> Restart update (2026-04-27): the checked-in baseline on this branch now reproduces at **0.82722** on `ml-25m` with `RECENCY_FRAC=0.8`, `LR=7e-5`, `WEIGHT_DECAY=5e-5`, `ACCUM_STEPS=2`, `TRAIN_NEG_MODE=anchor_pos_catalog`, `POST_RECENCY_NEG_RESAMPLE=1`, `POST_RECENCY_EASY_NEG_PER_POS=0.75`, `USER_HIST_MODE=rating`, `USER_HIST_CONTEXT=causal_masked`, `ITEM_HIST_CONTEXT=causal_masked`, `USER_GENOME=scalar_dot`, and `USER_GENOME_TARGET=genome_field`.
> Cycle 8 win (apr27): adding a per-user content profile (mean of genome vectors over user's high-rated genome-having items) and feeding `dot(user_genome, item_genome) / GENOME_DIM → Linear(1, 28)` into the genome_field gave +0.000944 over the prior 0.82628 baseline at WD=5e-5. Stacking with mask-aware genome gate hurt (-0.0003 at the same WD), so kept the legacy gate. 9-trial sweep around the surviving config produced 7 configs in the +0.0007 to +0.0009 ridge.
> Curriculum neg sampling TRIED (10 trials, +0.0006 max) — not the breakthrough hoped for.

#### NEW: Simplification hypothesis (apr06 discussion)
The model may be stuck in a local minimum due to over-parameterization. Evidence:
- Mean pool (no attention) gets 0.819 (only -0.003 from full model)
- Removing causal SA is neutral (0.820 vs 0.821)
- 500 architecture experiments all converge to same ~0.821-0.822
- A simpler model might find a sharper, higher optimum

**Simplification experiment plan (10 trials):**
1. Strip to MINIMAL model: user/item embed + DIN (no causal SA) + bottom MLP + genre + genome → simple concat → top MLP. No field attention, no two-stream, no bilinear.
2. Tune the minimal model aggressively (LR, WD, dropout, dim sweep)
3. Add components back ONE AT A TIME, measure each contribution
4. If minimal model matches ~0.820, it's a better base for exploration (fewer params, cleaner gradients)

#### NEW: Inter-rating timing features (low priority, 2-3 trials)
- Burstiness: std/mean of inter-rating time gaps per user
- Recent activity acceleration: gap trend (are gaps getting shorter or longer?)
- NOT raw activity rate (already tried, neutral)

#### Single model ideas (not yet tried or undertried)

**High priority (Critic-recommended, never tried):**
1. **Curriculum negative sampling** — Start training with NEG_RATIO=4 (easy negatives), decay to NEG_RATIO=1 over epochs. HSTU paper showed +0.015-0.025 AUC from curriculum sampling. This is the single most promising untried idea.
2. **Temporal negative sampling** — Sample negatives from items popular BEFORE the target interaction timestamp, not random. More realistic "what the user could have seen" negatives.
3. **Stratified negative sampling by popularity** — Different NEG_RATIO for popular vs long-tail items (e.g., 3/2/1 by decile). Popular items need more negatives to distinguish.
4. **Recency as feature, not data split** — Add recency_decay = exp(-lambda * time_gap) as a dense feature instead of filtering training data. Lets the model learn to weight recent data more.

**Medium priority (discussed, partially explored):**
5. **Single-model HP ridge around the checked-in SE baseline** — `RECENCY_FRAC=0.8`, `LR=7e-5`, `WEIGHT_DECAY=1e-4`, `ACCUM_STEPS=2`, `TRAIN_NEG_MODE=anchor_pos_catalog`, `POST_RECENCY_NEG_RESAMPLE=1`, `POST_RECENCY_EASY_NEG_PER_POS=0.75`, `USER_HIST_MODE=rating`, `USER_HIST_CONTEXT=causal_masked`, `ITEM_HIST_CONTEXT=causal_masked` now reproduces at 0.82628 on `ml-25m`. Keep sweeping nearby `LR`/`WD`/`PATIENCE`/`ACCUM` settings, but treat this causal-masked setting as the current anchor.
6. **Collaborative filtering features** — Pre-compute ALS/SVD on implicit feedback (all ratings = positive), add top-K latent factors as dense features. Different from the SVD that was tried before (which was on explicit ratings).
7. **User cluster embeddings** — Soft-cluster users by history similarity (K-means on user_genre_affinity), use cluster ID as a sparse feature. Adds collaborative signal without the overhead of full GNN.
8. **Replace genre projection with small transformer** (user-suggested) — Model genre interactions (Action+Sci-Fi means different things than either alone). Pushed back due to only 20 genres, but worth 1 trial.

**Speculative (high-risk, high-reward):**
9. **HSTU architecture** — Hierarchical sequential transduction (Meta, ICML 2024). Complete model rewrite. Prior SASRec attempt failed, but HSTU is fundamentally different (hierarchical, not flat).
10. **External data** — IMDB plot summaries via links.csv → IMDB API. Poster images. Not in MovieLens but the only clear path to genuinely new content signal.
11. **PinSage** — GraphSAGE with random walk sampling on user-item bipartite graph. LightGCN failed (sparse matmul too slow), but PinSage uses sampling which scales better.

### Experiment log (autoresearch/apr27) — ml-25m, post-cycle-8 follow-up

> **No keep this branch. ~63 trials across 8 cycles. Baseline held at 0.827224.**
> Goal: find further single-model improvements on top of cycle-8's user-genome scalar_dot win.

**Cycles 1-5: Variations on the user-genome content-alignment channel** (each 2-3 trials, all dead/sub-threshold):
- C1 rating-and-recency-weighted user_genome aggregation: dead. The 1-d scalar dot in `ug_dot_proj` is an information bottleneck — different aggregation methods (mean, rating-weighted, recency-weighted) all collapse to similar dot products in the GENOME_DIM-d space.
- C2 elemwise / shared-weight `genome_proj`: dead. `Linear(1128, D)` overfit catastrophically (-0.0175); shared-weight gave gradient conflict on `genome_proj` (-0.0009).
- C3 implicit-feedback CF SVD as scalar dot dense feature: dead. Self-leak fixed via dropped-recency source, but signal still redundant with end-to-end embeddings (same finding as apr02b's SVD-32 attempt).
- C4 user_dislike_genome (mean over rating<4 items): dead. Standalone gives ~+0.001 same as like-channel; combined with like channel actively HURTS (-0.0015) due to redundancy + path contention.
- C5 item_rater_genome (per-item mean of `user_genome` over high-rating raters): dead. Same pattern as C4 — standalone equivalent to like-only, combined hurts.

**Pattern across C1-C5**: any second scalar-dot signal added to `genome_field` actively hurts. The genome-alignment channel is hard-saturated at one scalar of capacity. Different aggregations are equivalent; combinations conflict.

**Cycle 6: Field-pair bilinear interactions (architectural redesign)** — 12 trials.
- 21 explicit pair scalars from upper triangle of 7×7 fields, gated, concatenated to top_mlp input.
- All trials in [-0.001, -0.0002] band. Best -0.000156 at WD=8e-5.
- Verdict: dead. The top_mlp's first Linear (196→256) already learns implicit pair interactions from the flatten path; explicit pair scalars don't add complementary signal.

**Cycle 7: HSTU-style gated unified block on user history** — 20 trials.
- Replaces the 1-head causal SA + DIN with a `Linear(D, 4D) → split [U,V,Q,K] → gate*attn(QK)V → LN_post + residual` block.
- Key insight: `tokens=item_only` works (-0.0001), `tokens=rating_aug` crashes (-0.0014) because the rating signal already enters DIN's input concat — adding it to attention tokens creates path contention.
- Best config: WD=1.5e-4, dropout=0, item_only tokens, sigmoid gate, silu U → +0.000599. **Ridge of 8 configs in [+0.0005, +0.0006] band.**
- Wall-clock: 1.5× baseline (Critic kill threshold was 1.3×).
- Verdict: sub-threshold. Doesn't clear +0.0007 ridge tripwire; wall-clock disqualifying.

**Cycle 8: Multi-task auxiliary user-genome reconstruction loss** — 20 trials (15 valid post RNG-fix).
- Aux head: `user_embed (clean) → Linear(D, 1128) → predict user_genome[u]`. Aux loss MSE or BCE-per-tag on the 78%-coverage targets.
- **Critical bug discovered mid-cycle**: `nn.Linear.__init__` calls `reset_parameters()` (kaiming) at construction, drawing global RNG. Even with the aux head registered LAST in the model, those construction-time RNG draws shifted state for `_init_weights()`'s xavier inits on **other** modules — making AUX_W=1e-10 (effectively zero contribution) regress AUC by -0.0016. Fix: use bare `nn.Parameter(torch.zeros(...))` instead of `nn.Linear`.
- Post-fix: aux signal is fully NULL across all 14 valid trials (weights 0.05 to 10.0, MSE and BCE, with/without detach, with/without USER_GENOME path).
- Verdict: dead. The aux gradient signal genuinely doesn't shape user_embed in a way that helps BCE.

**Cycle 9: Per-position genome content alignment in user-history rating-pool** — 6 trials.
- Mechanism: `sim_w = 1 + GENOME_SIM_SCALE * dot(genome[hist_item], genome[target]) / GENOME_DIM`, multiplied onto rating_weights before normalization. No new params; pad-row zero-extension on `_genome_gpu` to handle PAD_IDX safely; fp16 gather + einsum to bound memory.
- Scales swept ∈ {1, 5, 10, 20, 50}, plus one variant with USER_GENOME=off. All within ±0.000035 of baseline.
- Wall-clock 1.7× baseline (above Critic 1.3× kill threshold) — disqualifying even if signal existed.
- Verdict: dead. The per-position content gate is redundant with what `hist_embed` (end-to-end content learning) + the existing rating-pool aggregation already capture.

**Cycle 10: Full HSTU rewrite (multi-layer, paper-faithful pointwise normalization)** — 6 trials.
- Per block: `Linear(D, 4D) → split [U,V,Q,K] → silu(QKᵀ/√D)/N_valid_per_row + sigmoid(U)*attn@V → residual`. Stacked N layers with shared weights. Optional relative-position bias.
- Sweep: N ∈ {1,2,3,4} with `l_div` norm + 1× clamp_sum + 1× no-rel-pos.
- Results: ALL within ±0.000136 of baseline (clamp_sum was -0.000701, slightly worse).
- Wall-clock: N=1 → 1.16× baseline, N=2 → 1.87×, N=3 → 2.47×, N=4 → 3.10×.
- Verdict: dead. **The HSTU paper's pointwise normalization (`silu/N_valid` count-divide) genuinely doesn't transfer to ml-25m scale** — softmax in cycle 7 was actually doing useful contrast that pointwise-norm loses. Cycle 7's softmax-lite (+0.000599) was the strict best of the HSTU family.

### Key learnings (apr27)

35. **The genome-alignment channel is information-bottlenecked at one scalar.** Multiple aggregation methods (mean, weighted, dislike, item-rater) all collapse to the same scalar dot signal. Two scalars conflict; element-wise vector overflows.
36. **Field-pair interactions are implicit in `top_mlp` first Linear.** Explicit pair scalars don't add value because the 196→256 Linear already mixes all 196 cross-features.
37. **HSTU-style gated attention on user history works at +0.000599 ridge but is wall-clock-limited.** The signal is real but the 1.5× compute cost disqualifies it from being a single-model keep. (Could revisit if wall-clock becomes secondary.)
38. **`nn.Linear.__init__` draws RNG at construction**, polluting downstream xavier inits even when modules are registered last. Use `nn.Parameter(torch.zeros(...))` for new heads when off-state byte-equivalence matters.
39. **Validators should test off-state byte-equivalence with `flag=ε` (tiny but >0)**, not just `flag=0`. The `flag=0` skips construction entirely; `flag=ε` exercises the construction path.
40. **The 0.8272 baseline is near-saturated** for content-alignment additions, field-interaction redesigns, and label-prediction aux losses. Future progress likely requires either bigger architectural shifts (full HSTU rewrite, transformer encoder) or external data.
41. **HSTU paper's pointwise-normalization claim does NOT transfer to ml-25m scale.** Cycle 10 swept N=1..4 layers of the paper-faithful `silu(QKᵀ/√D) / N_valid` count-divide form — all trials within noise of baseline, strictly worse than cycle 7's softmax-lite at the same depth (which gave +0.000599 ridge). Softmax's strong contrast is doing useful work at this dataset scale; HSTU paper's gains likely require billion-scale data the paper actually used.

### Experiment log (autoresearch/apr27b) — ml-25m, 100-trial HP saturation-confirmation sweep

> **KEEP. Baseline lifted 0.827224 → 0.828432 (SEED=42) via stacked sub-noise HP knobs. 5-seed mean lift +0.00170 (5/5 positive, min lift +0.0009).**
> Goal: confirm 0.8272 saturation OR find any remaining ridge gains via systematic HP-only sweep on the existing apr26 cycle-8 architecture.

**Setup**: MLE plumbed 8 new env flags (SEED, EMBED_DROPOUT, MLP_DROPOUT, LABEL_SMOOTH, GRAD_CLIP, WARMUP_STEPS, GENOME_BOTTLENECK_HIDDEN, GENOME_BOTTLENECK_DROPOUT). Validator confirmed byte-equivalence at all defaults (ml-100k & ml-1m smoke tests + ε-flag tests catching learning-#11 RNG-construction drift).

**Phases (100 trials total, ~5 GPU hours wall-clock)**:

- **P1 LR×WD (18 LHS)**: All 18 trials at or below baseline. Top-3 anchors (LR=7e-5 row, WD=3e-5/5e-5/8e-5) tied within ±0.0002. The cycle-8 ridge is *flat*, not under-resolved. NO LIFT.
- **P2 Negatives (12)**: Strong monotonic trend — lower POST_RECENCY_EASY_NEG_PER_POS and lower RECENCY_FRAC both lift. Top single-knob: `RECENCY_FRAC=0.7` +0.000421, `POST_RECENCY_EASY_NEG_PER_POS=0.4` +0.000350. Joint (0.6, 0.75) +0.000302. All sub-+0.001 single-seed.
- **P3 Effective batch (5)**: Baseline (eff 33K) is the optimum. ACCUM=4/BATCH=8192 (eff 32K) +0.0001. NO LIFT.
- **P4 Dropout (9)**: `MLP_DROPOUT=0.3` consistently +0.0002 lift across EMBED values. `EMBED=0.05/MLP=0.3` best single trial at +0.000172. Sub-+0.001.
- **P6 Patience (1)**: PATIENCE=5 → no effect (early stop pattern unchanged). NO LIFT.
- **P7 Grad clip (4)**: All values {1.0, 5.0, 10.0} produce identical baseline AUC — gradients never exceed even 1.0 in normal training. NO LIFT.
- **P8 Warmup (4)**: All warmup values {200, 500, 1000} hurt monotonically (-0.0002 to -0.0005). Confirms learning #3. NO LIFT.
- **P9 Genome bottleneck (4)**: Smaller dims `128,32` regress -0.0025 (capacity-starved). Bigger dims `512,128` regress -0.0005 (overfit). `256,32` regress -0.0016. **`GENOME_BOTTLENECK_DROPOUT=0.0` lifts +0.000363** at SEED=42 (single-seed) — dropout in the genome MLP was hurting the new scalar_dot user-genome path.
- **P10 Noise floor (4)**: Baselines at SEED ∈ {43, 44, 45, 46} = {0.8262, 0.8250, 0.8261, 0.8263}. **σ ≈ 0.00078, mean ≈ 0.82589**. SEED=42 (0.8272) is ~+0.0013 fortunate. Critical finding: the prior CLAUDE.md "<0.001 variance" was *deterministic re-runs at the same seed*, not seed variance.
- **P11 Multi-seed verify (40 + 3 backfill)**: 5 single-knob candidates × 4 new seeds (43-46) = 20 trials, then 4 joint configs × 4 new seeds = 16, plus 4 SEED=42 backfill for joint configs to complete 5-seed verification.

**Multi-seed-verified winners (5-seed mean lift, all 5/5 positive, all min ≥ -0.0003)**:

| Config | 5-seed mean lift | min lift | SEED=42 val_auc |
|---|---|---|---|
| **all-in (RECENCY=0.7, EASY_NEG=0.4, GENOME_DROP=0.0, MLP_DROP=0.3)** | **+0.00170** | +0.0009 | **0.828432** |
| top-3 (RECENCY=0.7, EASY_NEG=0.4, GENOME_DROP=0.0) | +0.00152 | +0.0008 | 0.828306 |
| RECENCY=0.7 + EASY_NEG=0.4 | +0.00108 | +0.0005 | 0.827971 |
| RECENCY=0.7 + GENOME_DROP=0.0 | +0.00091 | +0.0008 | (4-seed only) |

**Single-knob 5-seed results (all 5/5 positive but sub-+0.0007 mean — sub-threshold individually)**:

- RECENCY_FRAC=0.7 alone: mean +0.0005, min +0.0002
- POST_RECENCY_EASY_NEG_PER_POS=0.4 alone: mean +0.00064, min +0.0001
- GENOME_BOTTLENECK_DROPOUT=0.0 alone: mean +0.00032, min +0.0001
- EMBED=0.05 + MLP=0.3 alone: mean +0.0002, min +0.0001
- EASY_NEG=0.6 + RECENCY=0.75 (P2 joint): mean +0.00056, min +0.0003

**Why stacking works (heuristic)**: All 4 winning knobs reduce regularization/noise in the negative-easy-bias direction:
- `RECENCY_FRAC=0.7` (was 0.8): drop 30% of oldest ratings (was 20%) → cleaner training distribution closer to validation
- `EASY_NEG=0.4` (was 0.75): fewer random easy negatives per positive → focus on hard negatives
- `GENOME_BOTTLENECK_DROPOUT=0.0` (was 0.1): genome MLP already bottlenecked at 256→64→28 — dropout was redundant after the cycle-8 scalar_dot user-genome head was added
- `MLP_DROPOUT=0.3` (was 0.2): top MLP needs more regularization now that it processes a richer genome_field signal

The four signals are independent enough that they sum to ~+0.0017 from individual ~+0.0003-0.0006 lifts.

**Action**: Promoted all-in defaults in `train.py`. New baseline at SEED=42 = 0.828432.

### Key learnings (apr27b)

42. **Seed variance dwarfs deterministic noise.** σ ≈ 0.00078 across SEED ∈ {42-46} with the cycle-8 architecture. Single-seed lifts of +0.0003-0.0005 are deeply in noise. Multi-seed verification (≥4 new seeds + SEED=42 reference) is mandatory for any HP "win".
43. **Sub-noise single-knob lifts STACK.** Apr27b's four winning knobs each produce sub-+0.0007 single-seed lift but together produce +0.00170 5-seed mean. Past sweeps that dismissed individual sub-+0.001 lifts as "noise" were leaving real signal on the table — when multiple knobs trend in the same direction (lower-is-better, more-data, less-redundant-regularization), they often stack.
44. **PATIENCE/GRAD_CLIP/WARMUP all dead** at the cycle-8 baseline. Gradients never exceed magnitude 1.0; warmup hurts monotonically (confirms learning #3); patience just defers overfitting without changing val_auc peak.
45. **Genome bottleneck dims are well-tuned at 256→64→28.** Smaller (128,32 or 256,32) loses capacity. Bigger (512,128) overfits. The dimension surface is sharp — but genome MLP *dropout* at 0.1 was over-regularization once the cycle-8 scalar_dot path was added.
46. **The cycle-8 LR×WD ridge is flat, not under-resolved.** Apr27b 18-point LHS produced 18 trials clustered ±0.0011 around baseline with NO clear winner. Past sweeps suggesting "ridge has more room" were sampling-noise artifacts, not genuine under-resolution.

### Useful references

- BARS benchmark (Zhu et al., SIGIR 2022) — different task (tag CTR) but good training practices and model zoo
- FuxiCTR library (github.com/reczoo/FuxiCTR) — well-tuned CTR model implementations
- LightFM — implicit feedback baseline (BPR/WARP loss), ~0.86-0.90 AUC on ml-100k
- HSTU paper: "Actions Speak Louder than Words" — Zhai et al., Meta, ICML 2024
- TIGER paper: "Recommender Systems with Generative Retrieval" — Rajput et al., Google, NeurIPS 2023
- Wukong: "Towards a Scaling Law for Large-Scale Recommendation" — Zhang et al., Meta, 2024
