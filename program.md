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

Each experiment runs on NVIDIA L4 GPU (CUDA). Training terminates via early stopping (patience=10 evals), not a fixed time budget. Launch it as:

```bash
DATASET=ml-10m python3 train.py > run.log 2>&1
```

**Dataset selection** via the `DATASET` env var:
- `ml-100k` — 100K ratings, for quick smoke testing of code changes (~seconds)
- `ml-1m` — 1M ratings, fast iteration (~minutes)
- `ml-10m` — 10M ratings, **default for experimentation** (~5-15 minutes on L4)
- `ml-25m` — 25M ratings, full scale

Use `ml-100k` to quickly validate that code changes don't crash, then `ml-10m` for real metric comparison.

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
5. Real run: `DATASET=ml-10m python3 train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
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
| **Hybrid (current)** | rating >= 4 | rated < 4 (hard) + random unrated (easy) | BCE | **~0.79** (ml-1m), **~0.74** (ml-10m) |

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

### Key learnings

1. **Overfitting is the dominant problem.** The model peaks at 1-2.5 epochs on ml-10m then degrades. Embed dropout 0.1 is the only regularization that helps — it slows memorization without hurting peak AUC.

2. **Feature enrichment works, especially on the item side.** Movie year, genre count, movie age as dense features (+0.013). User history ratings in DIN (+0.029 combined with item-side DIN). The item tower was severely underspecified.

3. **Item-side DIN is a major win.** Attention over recent raters ("users like you rated this") gives strong collaborative signal. This was the single biggest improvement (+0.029).

4. **Architecture changes work, training procedure changes don't.** GDCN gates (+0.005), item-side DIN (+0.029), features (+0.013) all helped. LR schedules, regularization, loss changes, EMA, multi-task all hurt or had no effect.

5. **Fixed random seeds are essential.** Run-to-run variance on ml-10m was ~0.05 AUC before seeding. After fixing seeds (SEED=42), variance is <0.001. This was blocking our ability to detect real improvements.

6. **fp16 > bf16 for attention-heavy models.** bf16's reduced mantissa precision hurts DIN attention weights by ~0.006-0.008 AUC. fp16 with `-1e4` masking is the right choice.

7. **VRAM optimization via ID-based lookup.** Storing lookup tables by user/item ID instead of per-sample saves ~12 GB VRAM on ml-10m (14.4 → 2.7 GB), enabling larger models.

8. **ml-100k is only useful as a smoke test.** AUC on ml-100k does not predict ml-10m performance.

### What to try next (on GPU)

#### Tier 0 — Infrastructure
- ~~**Fix random seeds**~~ — DONE: deterministic training (SEED=42). Variance eliminated.
- ~~**Scale to ml-10m**~~ — DONE: ml-10m is now the default dataset, feature engineering vectorized for scale.
- ~~**Adjust TIME_BUDGET**~~ — DONE: removed fixed time budget, training now terminates via early stopping only.
- ~~**Move to NVIDIA GPU**~~ — DONE: running on NVIDIA L4 (CUDA). Precomputed GPU tensors, AMP fp16, torch.compile.
- ~~**VRAM optimization**~~ — DONE: ID-based lookup instead of per-sample precompute (14.4 GB → 2.7 GB).

#### Tier 1 — Quick wins (easy, try first)
- **DuoRec contrastive loss** — two forward passes with different dropout masks, InfoNCE auxiliary loss. Addresses representation collapse.
- **In-batch negatives** — use other items in the batch as contrastive negatives. Free diverse negatives, no sampling needed.
- **FinalMLP two-stream** — split features into user stream + item stream, process separately, bilinear fusion.
- **Tag features** — ml-10m has 95K user tags. Build tag multi-hot per movie → learned projection. 5371/10677 movies have tags.
- **Embed_dim=32 with embed dropout** — embed_dim=32 hurt before without dropout. Now that embed dropout 0.1 is in place, larger embeddings might work.
- **3 cross layers with GDCN gates** — 3 plain cross layers hurt before (too much capacity), but GDCN gates may prevent that.

#### Tier 2 — Sequential modeling (medium, high potential on GPU)
- **SASRec** — causal Transformer over user history. 2-3 layers, 2-4 heads. Replace or augment DIN attention. Well-proven on MovieLens.
- **DIEN (Deep Interest Evolution Network)** — GRU with target-aware attentional update gate (AUGRU). Captures interest drift over time.
- **CL4SRec** — contrastive learning with sequence augmentations (crop, mask, reorder). Auxiliary InfoNCE loss.
- **BERT4Rec** — bidirectional masked item prediction. Auxiliary masked-item loss + primary BCE.

#### Tier 3 — Advanced architectures (medium-hard)
- **RankMixer (ByteDance)** — MLP-Mixer adapted for CTR: field-mixing + channel-mixing MLPs. Replaces cross-network.
- **EulerNet** — complex-valued feature interactions via Euler's formula. Novel interaction modeling.
- **xDeepFM** — Compressed Interaction Network for explicit high-order feature crosses.
- **MoE (Mixture of Experts)** — replace MLP layers with N expert networks + gating. Specializes for user/item clusters.
- **Knowledge distillation** — train large teacher, distill to smaller student. Regularization benefit.

#### Tier 4 — Research frontier (hard, potentially transformative)
- **HSTU (Meta, ICML 2024)** — hierarchical sequential transduction. Generative recommendation at scale. Processes history at session + item level.
- **TIGER (Google, NeurIPS 2023)** — generative retrieval with semantic IDs via RQ-VAE + autoregressive Transformer.
- **LLM-enhanced features** — encode movie titles/descriptions with sentence transformer, add as dense features. Helps cold start.
- **RecFormer / P5** — text-to-text recommendation via LM fine-tuning. Different paradigm entirely.
- **Wukong scaling laws (Meta, 2024)** — scaling laws for recommendation: wider embeddings + deeper interaction > deeper MLPs.

#### Already tried, didn't work (don't retry as-is)

**From ml-1m experiments (autoresearch/mar31) — may behave differently on ml-10m:**
- LR schedules (ReduceLROnPlateau, cosine decay) — all hurt
- BatchNorm in MLPs — hurt significantly
- Embedding L2 regularization (full-table norm) — too strong
- Focal loss (gamma=2) — over-suppresses easy negatives
- Label smoothing — hurt
- FinalNet field gate — overfits
- Popularity-biased negative sampling — task too hard
- Larger embed_dim (32) without dropout — more overfitting
- 3 cross layers (plain DCN-V2) — too much capacity
- HISTORY_LEN=100 with DIN only — dilutes attention
- Stronger weight_decay (1e-3) — too aggressive

**From ml-10m experiments (autoresearch/apr01):**
- Embed_dim=32 + wider MLP + dropout 0.3 — 0.696 (worse, overfits faster)
- Multi-task aux rating MSE (weight=0.1) — 0.708 (no improvement)
- LR=5e-5 — 0.717 (too slow to converge)
- Separate embed LR (3x higher for embeddings) — 0.723 (worse)
- DIN + GRU dual history — 0.725 (slower and worse)
- NEG_RATIO=2 — 0.733 (no improvement)
- Embed dropout 0.15 — 0.738 (0.1 is better)
- Embed dropout 0.2 — 0.718 (too strong)
- Batch_size=16384 — 0.731 (too few steps per epoch)
- MaskNet instance masking — 0.741 (no improvement)
- History embed dropout 0.1 — 0.718 (hurts DIN attention)
- MLP dropout 0.3 — 0.723 (too strong, 0.2 is better)
- Eval 4x/epoch + patience=8 — 0.725 (worse than 2x/epoch + patience=5)
- BST 1-layer transformer + target attn — 0.740 (slower, no gain)
- HISTORY_LEN=100 with ID-lookup — 0.722 (dilutes attention)
- AdamW wd=0.01 — 0.712 (too strong)
- EMA decay=0.999 — 0.738 (no improvement)
- Feature-level dropout 0.1 — 0.735 (too aggressive)
- bf16 AMP — 0.731 (slightly worse than fp16 for attention-heavy model)

### Useful references

- BARS benchmark (Zhu et al., SIGIR 2022) — different task (tag CTR) but good training practices and model zoo
- FuxiCTR library (github.com/reczoo/FuxiCTR) — well-tuned CTR model implementations
- LightFM — implicit feedback baseline (BPR/WARP loss), ~0.86-0.90 AUC on ml-100k
- HSTU paper: "Actions Speak Louder than Words" — Zhai et al., Meta, ICML 2024
- TIGER paper: "Recommender Systems with Generative Retrieval" — Rajput et al., Google, NeurIPS 2023
- Wukong: "Towards a Scaling Law for Large-Scale Recommendation" — Zhang et al., Meta, 2024
