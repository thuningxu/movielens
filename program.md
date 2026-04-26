# autoresearch — MovieLens Recommendation

Autonomous experimentation loop for improving pointwise recommendation (AUC) on MovieLens.

## Current operating mode (2026-04-25)

- This repository now runs on a single-GPU machine. Run at most one training job at a time.
- Older references below to `2x NVIDIA L4`, parallel agents, or simultaneous worktrees are historical logs, not the current operating protocol.
- Treat `train.py` as the source of truth for the checked-in baseline. Historical bests remain **0.821 single-model** and **0.854 ensemble** until they are re-run on this machine.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr25`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed: data download/loading, train/val/test splits, AUC evaluation, constants. Do not modify.
   - `train.py` — the file you modify. Feature engineering, model architecture, optimizer, training loop.
4. **Verify dependencies**: Run `python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"` to confirm CUDA is available and only one GPU is visible.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on the single available CUDA GPU. Do not launch concurrent training jobs, smoke tests, or stacker runs on this machine. The current checked-in `train.py` uses early stopping with patience=3 evals and sub-epoch evaluation roughly 3x/epoch, but the code is the source of truth. Launch it as:

```bash
DATASET=ml-25m python3 train.py > run.log 2>&1
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

1. Pick an experiment idea from the backlog (or invent a new one).
2. Modify `train.py` with the experimental idea.
3. git commit.
4. Smoke test: `DATASET=ml-100k python3 train.py` — check it doesn't crash.
5. Real run: `DATASET=ml-25m python3 train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
6. Read results: `grep "^val_auc:\|^peak_memory_mb:" run.log`
7. If grep is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix it after a few attempts, give up on this idea.
8. Record results in `results.tsv` (NOTE: do not commit results.tsv, leave it untracked by git).
9. If val_auc improved: keep the commit, `git push` to upstream.
10. If val_auc is equal or worse: `git reset --hard HEAD~1` to discard.

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
- Snapshot ensemble (avg top-2 weights) — 0.806 (2nd best too weak)
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
- Cross-network projection, genome-gated DIN, ensemble, attention pooling — all neutral/worse

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
- Ensemble test proved genome model has no complementary error pattern with non-genome variant
- Attention pooling is strictly worse than MLP bottleneck for genome compression
- Honest assessment: 0.820 is not achievable without external data. Realistic ceiling is ~0.815.

**Key learnings (apr03d):**
16. **The Critic role saves GPU time.** Pre-screening killed embed_dim=32 (failed 4x) and temporal decay (failed 2x) before wasting runs. Validated prior experiment conclusions hold.
17. **Ensembling a model with its ablation provides no complementarity.** The genome model strictly subsumes the non-genome model — errors are correlated, not diverse.
18. **MLP bottleneck > attention for genome compression.** Softmax over 1128 dims spreads too thin. Fixed bottleneck (256→64→D) forces useful compression.
19. **0.8138 appears to be near the ceiling for MovieLens-25m with this task formulation.** After 175+ experiments, all architectural/training changes are neutral (±0.002). Only genuinely new data (tag genome) moved the needle.

---

### Experiment log (autoresearch/apr04) — ml-25m, architecture + ensemble

> **0.8210 single model → 0.8242 ensemble. ~200 experiments.**
> Target: 0.84. Key breakthrough: diverse architecture ensemble.

**Kept single-model improvements** (building on 0.8201):

| # | Experiment | AUC | Delta |
|---|-----------|-----|-------|
| 1 | Replace GDCN with 1-head field attention, WD=3e-5 | 0.8207 | +0.001 |
| 2 | History residual (raw + contextual), WD=5e-5 | 0.8210 | +0.000 |

**Ensemble results:**

| Ensemble | Method | AUC |
|----------|--------|-----|
| Best 3-model (fieldattn+meanpool+gdcn) | Simple average | 0.8228 |
| Best 5-model (fieldattn+ratingpool+nostream+noitemdin+dim16) | Simple average | 0.8234 |
| Best 22-model | LogReg stacking (5-fold CV, C=0.0005) | **0.8242** |
| Best 5-model rank-based | Rank averaging | 0.8236 |

**Ensemble members trained (22 variants):**
- fieldattn (0.821), gdcn (0.819), meanpool (0.819), nogenome (0.807)
- ratingpool (0.819), dinonly (0.818), neg3 (0.814), dim16 (0.818), nostream (0.818)
- hist30 (0.821), noitemdin (0.821), minimal (0.806), gdcn_slow (0.817), film (0.817)
- revhist (0.821), hist10 (0.820), dim56 (0.818)
- mf (0.780), widedeep (0.816), itemonly (0.818), neg0 (0.759), regression (0.812)

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
- Different seeds, ensemble of same architecture: negligible variance
- 90/5/5 split: not comparable (different eval set)

**Key learnings (apr04):**
24. **Field attention is a slight improvement over GDCN.** 1-head MHA across 7 fields beats 4 gated cross layers (0.8207 vs 0.8201). Simpler, fewer params, slightly better.
25. **Diverse ensemble is the breakthrough.** +0.003 from single model. The key is architectural diversity (low prediction correlation), not number of models. Mean pool (0.944 corr) >> GDCN (0.979 corr) as ensemble partner.
26. **SASRec/transformers are worse than DIN.** 10 trials of transformer-based sequence models all regressed -0.01 to -0.02. The lightweight causal SA + DIN combination is better suited to this task.
27. **GRU/DIEN is worse than causal SA.** Sequential processing loses to parallel attention on this data.
28. **The single-model architecture is saturated.** After 200 architecture experiments at 0.8210, no modification improves. The model topology is well-matched to MovieLens features.
29. **Ensemble beats single model when diversity is high.** Models with different inductive biases (mean pool, no DIN, small dims) contribute more than similar-architecture variants.

---

### Experiment log (autoresearch/apr05) — ml-25m, recency + ensemble expansion

> **Ensemble expanded from 0.8242 to 0.8339 via recency-diverse models. ~20 experiments.**
> Key discovery: models trained on different time slices provide massive ensemble diversity.

**Single-model results (no improvement over 0.8210):**
- Temporal features (hour/day): hurt (0.816)
- User activity rate: neutral (0.819)
- Item trend, time gap, rating variance: all hurt
- Asymmetric DIN: 0.820 (closest, not significant)
- 2-layer field attention, deeper bottom MLP, higher LR: all worse

**Recency-diverse models for ensemble:**
- recent50 (train on recent 50% only): 0.815-0.823 (varies by implementation)
- recent40/60/70/80: 0.814-0.817 (less data = lower individual AUC)
- old50: 0.793 (old data is very noisy for recent prediction)
- recent50_meanpool: 0.815 (mean pool variant)
- recent50_noitemdin: 0.815

**Ensemble expansion:**

| Ensemble | Method | AUC | Models |
|----------|--------|-----|--------|
| apr04 baseline | LogReg 22 models | 0.8242 | Architecture-diverse |
| + recency variants | LogReg 40 models | 0.8339 | + 18 recency/diversity |
| + more variants | LogReg 60 models | 0.8364 | + 20 extreme variants |
| **HistGBM stacking** | **HistGBM 59 models** | **0.8534** | Non-linear stacking |
| MLP stacking | MLP 59 models | 0.8495 | 2-layer neural stacking |

**Key learnings (apr05):**
30. **Recency-diverse models are the best ensemble partners.** Models trained on different time slices make different errors because user preferences drift. Even weak individual models (0.81) contribute to ensemble via low correlation.
31. **The ensemble path has more headroom than single-model.** Going from 22→40 models gave +0.010 AUC. Each maximally-diverse variant adds ~0.001-0.002 to the ensemble.
32. **Old ratings are noisy for recent prediction.** Training on only recent data sometimes gives higher single-model AUC, but the effect depends on feature engineering (which uses full history).
33. **HistGBM >> LogReg for stacking.** Non-linear stacking (HistGBM: 0.854) massively outperforms linear (LogReg: 0.836) on 59 diverse models. GBM learns which model to trust for which samples. 3-fold CV validated.
34. **59 diverse models span the full architecture space.** Variants include: field attention, GDCN, mean pool, DIN-only, no-DIN, small/large embeddings, different NEG_RATIO (0-8), recency filters (25-90%), label noise, pure MF, wide&deep, regression loss, shuffled history, etc.

### What to try next (backlog for future sessions)

> Historical checked-in bests: single model 0.821, ensemble 0.854.
> A later exploratory note below mentions 0.8223 from recency tuning, but it was not committed. Reproduce it before treating it as the current baseline.
> Single model improvements feed into ensemble — a better base model lifts ALL variants.
> Curriculum neg sampling TRIED (10 trials, +0.0006 max) — not the breakthrough hoped for.
> Stacker feature engineering WORKED (+0.003, metadata features help HistGBM).

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
5. Generate ensemble variants from simplified model (different inductive bias = more diversity)

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
5. **Single-model recency tuning** — Exploratory note: `RECENCY_FRAC=0.8 + LR=1e-4` reportedly gave 0.8223, but it was not committed. Needs a clean rerun plus a 10-trial sweep with WD/ACCUM/PATIENCE variations.
6. **Collaborative filtering features** — Pre-compute ALS/SVD on implicit feedback (all ratings = positive), add top-K latent factors as dense features. Different from the SVD that was tried before (which was on explicit ratings).
7. **User cluster embeddings** — Soft-cluster users by history similarity (K-means on user_genre_affinity), use cluster ID as a sparse feature. Adds collaborative signal without the overhead of full GNN.
8. **Replace genre projection with small transformer** (user-suggested) — Model genre interactions (Action+Sci-Fi means different things than either alone). Pushed back due to only 20 genres, but worth 1 trial.

**Speculative (high-risk, high-reward):**
9. **HSTU architecture** — Hierarchical sequential transduction (Meta, ICML 2024). Complete model rewrite. Prior SASRec attempt failed, but HSTU is fundamentally different (hierarchical, not flat).
10. **External data** — IMDB plot summaries via links.csv → IMDB API. Poster images. Not in MovieLens but the only clear path to genuinely new content signal.
11. **PinSage** — GraphSAGE with random walk sampling on user-item bipartite graph. LightGCN failed (sparse matmul too slow), but PinSage uses sampling which scales better.

#### Ensemble ideas (not yet tried)

**High priority:**
1. **Rebuild ensemble on improved single model** — If single model goes from 0.821→0.823, retrain all 59 variants from the new base and re-stack. Could push ensemble from 0.854→0.86+.
2. **HistGBM hyperparameter optimization** — Current best was iter=300, depth=6, lr=0.2. Full GridSearchCV might find better. Also try XGBoost, LightGBM, CatBoost.
3. **Train more recency variants** — KEEP_FRAC=0.15, 0.35, 0.45, 0.55, 0.65, 0.85, 0.95. Each adds ensemble diversity.
4. **Curriculum-trained models for ensemble** — If curriculum neg sampling works (#1 above), train 5 variants with different curriculum schedules.

**Medium priority:**
5. **Factorization machine on ensemble predictions** — FM on (pred_i, pred_j) pairs captures interaction effects that LogReg/GBM may miss.
6. **Two-stage stacking** — First stage: 5 LogReg meta-learners on subsets of models. Second stage: GBM on the 5 meta-learner outputs.
7. **SHAP-driven model pruning** — Use SHAP to identify which of the 59 models actually contribute. Remove dead weight to reduce overfitting risk.
8. **Dropout-based ensemble** — Train 1 model with high dropout (0.4), generate 10 stochastic predictions from the same checkpoint. Cheap diversity.

### Useful references

- BARS benchmark (Zhu et al., SIGIR 2022) — different task (tag CTR) but good training practices and model zoo
- FuxiCTR library (github.com/reczoo/FuxiCTR) — well-tuned CTR model implementations
- LightFM — implicit feedback baseline (BPR/WARP loss), ~0.86-0.90 AUC on ml-100k
- HSTU paper: "Actions Speak Louder than Words" — Zhai et al., Meta, ICML 2024
- TIGER paper: "Recommender Systems with Generative Retrieval" — Rajput et al., Google, NeurIPS 2023
- Wukong: "Towards a Scaling Law for Large-Scale Recommendation" — Zhang et al., Meta, 2024
