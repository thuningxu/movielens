# autoresearch — MovieLens Recommendation

Autonomous experimentation loop for improving pointwise recommendation (AUC) on MovieLens.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar31`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed: data download/loading, train/val/test splits, AUC evaluation, constants. Do not modify.
   - `train.py` — the file you modify. Feature engineering, model architecture, optimizer, training loop.
4. **Verify dependencies**: Run `python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` to confirm CUDA is available.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on NVIDIA L4 GPU (CUDA). Training terminates via early stopping (patience=2 evals, sub-epoch eval 2x/epoch), not a fixed time budget. Launch it as:

```bash
DATASET=ml-25m python3 train.py > run.log 2>&1
```

**Dataset selection** via the `DATASET` env var:
- `ml-100k` — 100K ratings, for quick smoke testing of code changes (~seconds)
- `ml-1m` — 1M ratings, fast iteration (~minutes)
- `ml-10m` — 10M ratings, medium scale (~5-15 minutes on L4)
- `ml-25m` — 25M ratings, **default for experimentation** (~5-10 minutes on L4)

Use `ml-100k` to quickly validate that code changes don't crash, then `ml-25m` for real metric comparison.

**What you CAN do:**
- Modify `train.py` — this is the primary file you edit. Everything is fair game: feature engineering, feature transformations, model architecture, optimizer, hyperparameters, training loop, batch size, model size, sequence modeling, negative sampling, etc.
- Modify `prepare.py` when the model demands a different training data setup (e.g. implicit feedback, negative sampling, different label definitions, new data splits). The evaluation function and summary printer should remain stable.

**What you CANNOT do:**
- Install new packages or add dependencies beyond what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_auc.** Training terminates via early stopping, so you don't need to worry about time budgets. Everything is fair game: change the feature engineering, the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing.

**Memory** is a soft constraint. The NVIDIA L4 has 24 GB VRAM. Some increase is acceptable for meaningful AUC gains, but it should not OOM.

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
| **Hybrid (current)** | rating >= 4 | rated < 4 (hard) + random unrated (easy) | BCE | ~0.79 (ml-1m), ~0.74 (ml-10m), **0.806 (ml-25m)** |

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

### What to try next (on GPU, ml-25m, baseline 0.806)

> After ~130 experiments across 4 sessions, the model appears near-saturated on MovieLens data.
> Most remaining ideas are high-risk/low-probability. External data would likely be needed for significant gains.

#### Untried ideas (speculative)
- **Trained LightGCN embeddings** — train LightGCN with BPR loss offline, freeze embeddings, use as features. The random-init version failed (0.734 on ml-10m) but properly trained GCN embeddings capture multi-hop collaborative signal.
- **PinSage** — GraphSAGE with random walk sampling. Scalable graph embeddings.
- **xDeepFM CIN** — Compressed Interaction Network for explicit high-order crosses. Different interaction pattern from GDCN.
- **HSTU (Meta, ICML 2024)** — hierarchical sequential transduction. Would require significant architectural rewrite.
- **External data** — IMDB plot summaries, poster images, user reviews. Not available in MovieLens but would provide genuinely new content signal.
- **SWA (Stochastic Weight Averaging)** — average weights from multiple training points to reduce overfitting. Built into PyTorch, no new deps.
- **Stochastic depth on GDCN** — randomly skip cross layers during training (like DropPath). Regularize the 4-layer cross network.

### Useful references

- BARS benchmark (Zhu et al., SIGIR 2022) — different task (tag CTR) but good training practices and model zoo
- FuxiCTR library (github.com/reczoo/FuxiCTR) — well-tuned CTR model implementations
- LightFM — implicit feedback baseline (BPR/WARP loss), ~0.86-0.90 AUC on ml-100k
- HSTU paper: "Actions Speak Louder than Words" — Zhai et al., Meta, ICML 2024
- TIGER paper: "Recommender Systems with Generative Retrieval" — Rajput et al., Google, NeurIPS 2023
- Wukong: "Towards a Scaling Law for Large-Scale Recommendation" — Zhang et al., Meta, 2024
