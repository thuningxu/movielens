# autoresearch — MovieLens Recommendation (restart, apr28)

Fresh experimentation loop on top of a deliberately simple linear baseline. Same data + features + metric as the prior project at `legacy/`; new starting architecture and new experiment log.

## Why restart?

The prior project (see `legacy/`) reached val_auc = 0.8284 on ml-25m via several hundred experiments converging on a DLRM-style architecture, then plateaued. Rather than continue iterating in that local optimum, this restart begins from the simplest possible model (single Linear head on concatenated features), keeping the input features and prediction task unchanged, and rebuilds the architecture deliberately from below.

## Setup

- **Branch convention**: each experiment cycle on `restart/<tag>` (e.g., `restart/apr28` for the initial scaffold).
- **`prepare.py`**: do not modify (`evaluate()` is the ground-truth metric).
- **`train.py`**: the experimentation file. Currently the linear baseline; will grow.
- **Multi-seed discipline**: estimate the seed-noise floor empirically before declaring any win. A few baseline seeds give you σ; require multi-seed verification with mean lift comfortably above that floor and a positive sign at every seed.
- **Feature cache**: `data/features_<hash>.npz` is built on first ml-25m run and reused. Cache key is `restart-3` (includes per-position `hist_ts` plus per-item training-positive count for `FREQ_WD_LAMBDA`).

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

### Cumulative progress

| Stage | val_auc (SEED=42) | Cum. lift |
|---|---|---|
| Linear baseline scaffold | 0.7848 | 0 |
| Stripped pre-computed user/item stats | 0.8217 | +0.037 |
| Centered pool stack (apr28b) | 0.8246 | +0.040 |
| Cross fields (apr28b) | 0.8251 | +0.040 |
| LR=3e-4, WD=5e-5 (apr28g) | 0.8263 | +0.042 |
| **Sub-noise stack of 3 (apr28o)** | **0.8282** | **+0.043** |

Legacy DLRM ceiling: 0.8284. Restart linear-head model now within 0.0002 of legacy with much simpler architecture.

### `autoresearch/apr28r` — two-tower α-residual scoring — null

**Null** (`4694396`, `041557b`). Researcher proposed adding a parallel scoring path: separate user/item MLPs whose outputs are dotted and added to the main logit via a learnable scalar α. User tower input = u_e + u_hist_pool + u_hist_rat_mean (57d); item tower input = i_e + i_hist_pool + i_hist_rat_mean + genre + year + genome (1206d). Each tower: `Linear(D_in, H) → ReLU → Linear(H, T, bias=False)`.

**Init bug caught in smoke** (apr28p replay): first attempt zero-initialized BOTH towers' second linear → `u_vec=i_vec=0 → dot=0` forever, killing all tower gradients (verified: ml-100k smoke produced identical val_auc to off-state). Fix: ASYMMETRIC zero-init — only user_tower's W2 zeroed; item_tower's W2 keeps default kaiming. Then at step 0: `u_vec=0, i_vec≠0 → dot=0` (logit unchanged), but `∂dot/∂(user_W2) = i_vec · ReLU ≠ 0` → user trains step 1+; once `user_W2` drifts, item starts training step 2+. Validator confirmed: ml-100k off=0.5881 unchanged, TWO_TOWER=1=0.7256 (gradients now flow).

Pre-screen 3-cell sweep on ml-25m at SEED=42 (vs 0.828188 baseline):

| Cell | val_auc | Δ |
|---|---|---|
| H=32, T=32 | 0.814878 | -0.0133 |
| H=64, T=32 | 0.812463 | -0.0157 |
| H=64, T=64 | 0.810243 | -0.0179 |

All cells regress; bigger tower → worse (monotonic). All trials finished in 73-74s vs ~290s baseline — early stopping fired immediately because val_auc plateaus or regresses with the towers active.

**Lesson:** Adding a parallel parametric scoring path (dot of MLP outputs) on top of the saturated linear head + 4 manual crosses + aux head is not orthogonal — it's redundant capacity that the optimizer cannot integrate without dropping out the existing concat-Linear signal. Same shape of failure as apr28p (DCN cross): more parametric capacity → more harm. The `α=0.1` residual gate doesn't help because the tower output's variance scales with `√(T · D_item)`; α · tower_score has comparable magnitude to the baseline logit, which forces the head to compensate.

**Combined with apr28p (DCN cross), apr28q (multi-pool), apr28r (two-tower):** three different parametric capacity additions — all NULL or worse. Strong signal that the bottleneck is not at the scoring layer or the field-stack interaction layer; it's at the representation / new-information level.

### `autoresearch/apr28q` — multi-pool concat (mean / last-position) — null

**Null** (`feb954d`). Researcher proposed adding plain mean-pool of item_embed (USER side) / user_embed (ITEM side) over valid history positions, alongside the existing rating-centered pool — testing whether mean-pool encodes signal that's orthogonal to the rating-centered pool. Critic raised the cosine-correlation concern (centered ≈ 0.85-0.95 cosine to mean on right-skewed ratings) and tightened the budget to ≥+0.0006 single-seed gate.

Pre-screen at SEED=42 vs 0.828188 baseline:

| Cell | val_auc | Δ |
|---|---|---|
| `USER_HIST_MEAN_POOL=1` | 0.828839 | **+0.000651** |
| `USER + ITEM mean` | 0.828265 | +0.000077 |
| `USER_HIST_LAST_POSITION=1` | 0.828109 | -0.000079 |
| `ITEM_HIST_MEAN_POOL=1` | 0.827798 | -0.000390 |

Only USER mean alone signed; ITEM mean regresses; combining the two dilutes USER mean's lift down to noise — item-side mean-pool is actively harmful, not just neutral. Last-position is null (consistent with apr28l's recency-saturation finding).

5-seed verify of `USER_HIST_MEAN_POOL=1`:

| SEED | baseline | umean | lift |
|---|---|---|---|
| 42 | 0.828188 | 0.828839 | +0.000651 |
| 43 | 0.828073 | 0.827828 | **-0.000245** |
| 44 | 0.828024 | 0.828105 | +0.000081 |
| 45 | 0.828098 | 0.828454 | +0.000356 |
| 46 | 0.828266 | 0.828361 | +0.000095 |

Mean lift **+0.000188**, 4/5 positive, min -0.000245, lift-σ 0.000336, t ≈ 0.56. Below all three multi-seed bars; the SEED=42 single-seed +0.000651 was an RNG artifact. The Critic's cosine-correlation prediction was effectively right.

**Lesson:** Mean-pool and rating-centered pool are not as orthogonal as the math suggests; on a right-skewed rating distribution they alias each other to the point where the head can't extract additional signal beyond the centered pool's. Last-position is bracketed by recency saturation. Item-side multi-pool is actively harmful, suggesting the item-side rating-centered pool already has the right inductive bias and adding mean-pool there just adds noise.

### `autoresearch/apr28p` — DCN/DCNv2/DCNv3 cross network — null

**Null** (`73d4262`). The Researcher noticed our concat already includes 4 manual Hadamard crosses (`u_e⊙i_e`, `u_hist_pool⊙i_e`, `i_hist_pool⊙u_e`, `ts_norm⊙i_e`) and proposed a DCN cross network on the 4-field 112-d stack to learn higher-order interactions automatically. Critic vetted; pre-screen 5-cell sweep on ml-25m at SEED=42 (vs 0.828188 baseline):

| Cell | val_auc | Δ |
|---|---|---|
| dcnv2 r=4 add | 0.821182 | -0.0070 |
| dcnv2 r=8 replace | 0.819523 | -0.0087 |
| dcnv3 r=8 add | 0.819242 | -0.0089 |
| dcnv2 r=8 add | 0.816820 | -0.0114 |
| dcnv2 r=16 add | 0.816032 | -0.0122 |

All cells regress; higher rank → worse; replace mode (strips manual crosses) is no better than smaller-rank add — the manual crosses are net positive on top of even the best DCN cell. dcnv3's gate dampens but doesn't reverse harm.

**Implementation note** — initial sweep at `5edbff5` returned exact 0.827536 across all three low-rank cells (gradient-flow bug). Root cause: low-rank `W = U V^T` had **both** U.weight and V.weight zero-initialized. Then `∂loss/∂U = (∂loss/∂y)·V(x) = 0` (because V(x)=0 with V=0) **and** `∂loss/∂V = U^T·(∂loss/∂y) = 0` (because U=0). Adam never moved either; cross stayed at the passthrough identity forever.

Fix at `73d4262`: keep V at default kaiming-uniform; zero only U.weight and U.bias. Then `y = U(V(x)) = 0` at step 0 (passthrough preserved), but `∂loss/∂U = V(x) ≠ 0` (V is random non-zero), so Adam moves U. After step 1, U≠0 makes `∂loss/∂V` non-zero too. Validator confirmed: ml-100k rank-distinctness test produces 4 distinct val_auc across r∈{4,8,16} and dcnv3 r=8, all > off-state — gradients flow.

**Lesson:** when the baseline already includes hand-tuned Hadamard crosses on the natural pairs (user × item, user × item-history, item × user-history, ts × item), a parametric DCN cross over the same 4-field stack is redundant capacity that the linear head can't exploit. The linear-head + manual-cross combination saturates this 4-field interaction space.

### `autoresearch/apr28o` — sub-noise stacking win

**Win** (`b983125`). Three individually sub-threshold mechanisms compound super-additively:

- `AUX_RATING_WEIGHT=25.0` — auxiliary Linear head predicting normalized rating; combined with main BCE head via `bce + 25 × masked_mse`. The aux MSE is masked to skip random unrated easy negatives. Standalone aux sweep peaked at weight=20-25 with single-seed +0.000750 (5-seed mean +0.000634, sub-threshold alone).
- `FREQ_WD_LAMBDA=1e-4` — per-item L2 weighted by `1/sqrt(item_count + 5)`; tail items get more regularization. Smooth unimodal sweep, peak at lambda=1e-4 with single-seed +0.000200 (sub-threshold alone).
- `CROSS_TS_ITEM=1` — 4th cross field `ts_norm ⊙ i_e` for temporal drift in item preference. Standalone single-seed +0.000358 (5-seed mean +0.000528, sub-threshold alone).

Stack 5-seed verify (vs LR/WD baseline 0.826251):
- SEED=42: +0.001937
- SEED=43: +0.001764
- SEED=44: +0.001569
- SEED=45: +0.001730
- SEED=46: +0.001737

Mean lift **+0.001747**, 5/5 positive, min +0.001569 — ~17.5σ above the σ ≈ 0.0001 noise floor. New baseline at SEED=42: **0.828188**.

This validates the **legacy "sub-noise stacking" lesson** (apr27b's 4-knob HP stack). Mechanisms that individually sit just below the noise bar can still represent real, orthogonal contributions; compounding 3-4 of them clears the bar by a wide margin.

### `autoresearch/apr28h-n` — null cycles between LR/WD and aux

After the apr28g LR×WD win, ran 7 cheap-HP and code-change cycles. All sub-threshold individually but the apr28o stack uses the surviving signal from apr28h (CROSS_TS_ITEM) and apr28n (FREQ_WD_LAMBDA):

- **apr28h** — `CROSS_TS_ITEM=1` redo at new defaults: 5-seed mean +0.000528, sub-threshold (same as apr28c at old baseline; HP shift didn't move it across alone — but it's a real signal that the stack uses).
- **apr28i** — EMBED_DIM ∈ {16, 28, 40, 56, 84}: all within ±0.00015 of baseline. Flat.
- **apr28j** — NEG_RATIO ∈ {2, 4}: monotonic regression from 1; 1 confirmed optimal.
- **apr28k** — BATCH_SIZE × PATIENCE: all within ±0.00003. PATIENCE=4/5 produce *identical* val_auc to default (model converges before extended patience matters).
- **apr28l** — HISTORY_LEN ∈ {50, 150, 200}: 50 ≈ default; longer slightly regresses. 100 is fine.
- **apr28m** — ITEM_HIST_LEN ∈ {15, 60, 100}: all within ±0.00005. 30 fine.
- **apr28n** — `FREQ_WD_LAMBDA` sweep: smooth unimodal peak at 1e-4 with single-seed +0.000200 (sub-threshold alone but useful in the apr28o stack).

### `autoresearch/apr28g` — LR × WD retune

After 4 consecutive null cycles (apr28c-f explored ts × i_e cross, MLP head HP/LR, DIN attention, WD-only sweep — all sub-threshold or regressing), a 7-cell LR × WD grid found that the new richer-feature baseline benefits from a softer optimizer.

Top 2 configs at SEED=42:
- LR=3e-4, WD=5e-5: 0.826251 (+0.00113)
- LR=1e-3, WD=5e-5: 0.826036 (+0.00092)

5-seed multi-seed verify (SEED 42-46):
- LR=3e-4, WD=5e-5: **mean lift +0.001152, 5/5 positive, min +0.000850**
- LR=1e-3, WD=5e-5: mean lift +0.000861, 5/5 positive, min +0.000650

Both clean keeps. Promoted (LR=3e-4, WD=5e-5) as new defaults — the stronger of the two and the dominant factor (WD=5e-5) is captured by both.

### `autoresearch/apr28c-f` — null cycles

Brief log of cycles between cross-fields and LR×WD wins:

- **apr28c — ts × i_e cross**: mean lift +0.000486, 5/5 positive but below +0.0007 bar. Sub-threshold.
- **apr28d — MLP head HP + LR rescue**: 10 trials across (hidden ∈ {32, 64, 128}) × (dropout ∈ {0.3, 0.5}) × (LR ∈ {1e-3, 3e-4, 1e-4}). All regress -0.003 to -0.013. Default LR causes overfit-in-67s; lower LR doesn't rescue. MLP head as currently designed is dead.
- **apr28e — DIN target-aware attention**: 5 configs across (attn_hidden ∈ {32, 64, 128}) × LR variants. All regress -0.002 to -0.025; best -0.0021. Trial wall-clock 4-10× baseline due to (B, L, 3*D) attn features tensor. DIN as currently implemented can't beat the rating-centered + cross-fields baseline.
- **apr28f — WD sweep alone**: tested WD ∈ {1e-6, 1e-4, 5e-4, 1e-3} at LR=1e-3. WD=1e-4 within noise of default (1e-5); extremes hurt. Combined with LR variation in apr28g revealed the (3e-4, 5e-5) optimum.

### `autoresearch/apr28b` — multiplicative cross fields cycle

Started from the apr28 baseline (0.824570). Researcher + Critic debated the backlog over 4 rounds and converged on a joint 2×2 sweep: cross fields × MLP head.

**Win — cross fields with Linear head** (commit on this branch). Three Hadamard cross-feature vectors appended to the concat:
- `u_e ⊙ i_e` (28-d): user × item embedding
- `u_hist_pool ⊙ i_e` (28-d): user-history pool × item
- `i_hist_pool ⊙ u_e` (28-d): item-history pool × user

The linear head literally cannot represent multiplicative interactions; these crosses give it that capability for free (no new learnable params, just +84 dims to the head's input).

5-seed multi-seed verification (SEED 42-46):
- Cell A baseline: 0.824510 mean
- Cell B (Linear + cross): 0.825234 mean → **mean lift +0.000724, 5/5 positive, min +0.000334** (just clears the +0.0007 bar at ~7σ above noise floor σ ≈ 0.0001)
- Cell C (MLP head, no cross): -0.0098 mean — **catastrophic overfit at default HP** (HIDDEN=128, dropout=0.2). Training stops at ~67s vs 200s baseline.
- Cell D (MLP + cross): -0.0136 mean — worse than C; cross fields compound the MLP's overfit.

MLP head is a viable direction but requires HP tuning (smaller hidden width, more dropout, or head-only LR) — deferred to a follow-up cycle.

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

## Research backlog

**Status as of apr28o**: Most Tier 1-2 ideas have been explored. The remaining backlog is increasingly skewed toward Tier 3-5 (bigger architectural shifts, new data sources). The cheap HP/aggregator space is largely saturated at the current architecture.

**Surviving ideas worth running** (after apr28o):
- **Tier 4-5**: IMDB plot summaries (genuinely new signal, deferred), transformer encoder + target attention (deferred), generative retrieval / TIGER (speculative).
- **Re-explore**: User-level tag genome with learned aggregation (originally removed in the strip; mechanism may now fit).
- **Speculative**: MLP head with smarter optimization (zero-init last layer, head-only LR via param groups, gradient warmup) — apr28d's null was at default LR; may be rescuable.

Ideas organized by approximate cost. Each entry specifies a sweep budget — **don't declare an idea dead until at least the listed number of trials are run, with multi-seed verification of any single-seed qualifier**.

Conventions:
- "Pre-screen" trials are single-seed (SEED=42) to pick a candidate.
- "Verify" trials run the best pre-screen config at 4 new seeds (43, 44, 45, 46) for a 5-seed total. Bar: 5-seed mean lift ≥ +0.0007, 5/5 positive, min lift ≥ -0.0003.
- Budget noted as `pre-screen + verify = total` trials at ~3 min each on the current 16 GB GPU.

### Tier 1 — HP / capacity on current architecture (cheap; null-result-tolerant)

1. **Embedding dim sweep** — currently `EMBED_DIM=28` (inherited from legacy). The restart never re-tuned it for the linear head + centered pool.
   - Pre-screen: dim ∈ {16, 28, 40, 56, 84} at SEED=42 (5 trials)
   - Verify: top-1 vs current at 4 new seeds (4 trials)
   - **Budget: 5 + 4 = 9 trials**. Prior: 0.25.

2. **Single hidden layer (MLP head)** — replace `Linear(1264, 1)` with `Linear(1264, H) → ReLU → Linear(H, 1)`. The aggregation work has saturated what a *linear* head can extract; capacity here is the next obvious test.
   - Pre-screen: hidden_width H ∈ {64, 128, 256, 512} at SEED=42 (4 trials)
   - Verify: top-1 at 4 seeds (4 trials)
   - **Budget: 4 + 4 = 8 trials**. Prior: 0.50 — strongest cheap candidate.

3. **MLP depth + dropout joint sweep** — only run if (2) lifts. Adds depth + regularization.
   - Pre-screen: (depth, dropout) ∈ {1, 2, 3} × {0.0, 0.1, 0.2, 0.3} at SEED=42 = 12 trials
   - Verify: top-2 × 4 seeds = 8 trials
   - **Budget: 12 + 8 = 20 trials**. Prior: 0.30 conditional on (2) lifting.

4. **LR + WD joint re-tune** — current `LR=1e-3, WD=1e-5` is the scaffold default; never tuned.
   - Pre-screen: LR × WD = {3e-4, 1e-3, 3e-3} × {1e-6, 1e-5, 1e-4} = 9 trials at SEED=42
   - Verify: top-2 × 4 seeds = 8 trials
   - **Budget: 9 + 8 = 17 trials**. Prior: 0.30.

5. **NEG_RATIO sweep for the linear head** — legacy `NEG_RATIO=1` was tuned for the DLRM. Restart inherited it; may not be the same optimum.
   - Pre-screen: NEG_RATIO ∈ {0, 0.5, 1, 2, 4} at SEED=42 (5 trials)
   - Verify: top-1 at 4 seeds (4 trials)
   - **Budget: 5 + 4 = 9 trials**. Prior: 0.25.

### Tier 2 — Aggregation improvements (continuation of apr28 work)

6. **DIN target-aware attention (user history)** — replace mean/centered pool with attention where the *target item embedding* is the query and history items are keys/values. Adds real capacity in the aggregator.
   - Pre-screen: (attn_hidden, dropout) ∈ {32, 64, 128} × {0.0, 0.1} = 6 trials at SEED=42
   - Plus an alternate: dot-product attention (no MLP) as a no-param variant
   - Verify: top-1 × 4 seeds = 4 trials
   - **Budget: 7 + 4 = 11 trials**. Prior: 0.40.

7. **DIN target-aware attention (item history)** — symmetric to (6); query = target user embedding, keys/values = item's historical raters. Independent of (6); can run after.
   - Same sweep shape as (6).
   - **Budget: 7 + 4 = 11 trials**. Prior: 0.30.

8. **DIN on both sides combined** — only after (6) and (7) individually verified.
   - Pre-screen: 1 trial (best (6) + best (7))
   - Verify: 4 seeds (4 trials)
   - **Budget: 5 trials**. Conditional prior: 0.40 of additive lift if both (6) and (7) win.

9. **Multi-pool concat** — concatenate mean-pool + centered-pool + last-position embedding into one (3D)-dim field per side. Tests whether multiple pooling perspectives stack.
   - Pre-screen: 3 variants (user only, item only, both sides) at SEED=42
   - Verify: top-1 × 4 seeds
   - **Budget: 3 + 4 = 7 trials**. Prior: 0.15 (centered already captures dominant signal).

10. **Element-wise user×target / item×target cross fields** — the linear head can't synthesize crosses; pre-compute `user_e * item_e` and concat as a 28-d field. (FM-style trick.)
    - Pre-screen: variants {user×item only, user×item + user×genome_field, etc.} = ~3 trials
    - Verify: top-1 × 4 seeds
    - **Budget: 3 + 4 = 7 trials**. Prior: 0.30.

### Tier 3 — Architectural additions (medium cost)

11. **Two-tower architecture** — separate user-tower (concat user-side features → MLP → user_vec) and item-tower (concat item-side features → MLP → item_vec). Score = `dot(user_vec, item_vec)`. Different scoring shape from concat-MLP.
    - Pre-screen: tower_dim ∈ {32, 64, 128}, MLP depth ∈ {1, 2} = 6 trials
    - Verify: top-1 × 4 seeds = 4 trials
    - **Budget: 6 + 4 = 10 trials**. Prior: 0.30.

12. **Causal self-attention encoder before pool** — enrich each history position's representation via 1 transformer block (single layer, 1-2 heads), then pool.
    - Pre-screen: heads ∈ {1, 2}, ffn_ratio ∈ {2, 4}, pool ∈ {centered, target-attention} = 8 trials
    - Verify: top-1 × 4 seeds = 4 trials
    - **Budget: 8 + 4 = 12 trials**. Prior: 0.20 (legacy SA was sub-bar; restart context is different but not a slam dunk).

13. **Bilinear / pair-cross layer** — for each pair of fields in the concat, compute a bilinear `f_i^T W_ij f_j` scalar; sum these as the score. Generalizes FM.
    - Pre-screen: pair_subset ∈ {all, user×item only, target × history pools} = 3 trials
    - Verify: top-1 × 4 seeds = 4 trials
    - **Budget: 3 + 4 = 7 trials**. Prior: 0.20.

### Tier 4 — Bigger architectural changes (higher cost; do after tiers 1-3)

14. **Transformer encoder + target attention (full DIN-tower)** — combines (6) and (12). 2-3 layers of self-attention over history, then target-aware attention pool.
    - Pre-screen: layers ∈ {1, 2}, heads ∈ {1, 2, 4}, pool variants ≈ 12 trials
    - Verify: top-2 × 4 seeds = 8 trials
    - **Budget: 12 + 8 = 20 trials**. Prior: 0.30 (hopefully synergistic).

15. **Pure sequential next-item objective as auxiliary loss** — predict the held-out next-item from history with a parallel head; main BCE head unchanged. May provide auxiliary gradient signal that improves embeddings.
    - Pre-screen: aux_weight ∈ {0.0, 0.1, 0.5, 1.0}, head_arch ∈ {2 variants} = 8 trials
    - Verify: top-1 × 4 seeds = 4 trials
    - **Budget: 8 + 4 = 12 trials**. Prior: 0.15 (legacy aux losses were null; sequential is structurally different).

16. **TIGER-style generative retrieval** — codebook-quantize item IDs (RQ-VAE), autoregressively predict next-item codes. Substantial rebuild.
    - Out of "single sweep cycle" scope. Estimated: 2-3 weeks of implementation, then 30+ trials.
    - Prior: 0.20 — speculative but novel.

### Tier 5 — New information sources

17. **IMDB plot summaries** — via `links.csv` → IMDB API → small text encoder (e.g., MiniLM, distilBERT) → embed each movie's summary. Concat alongside genre + genome.
    - Implementation cost: substantial (data fetch, API key, text encoder).
    - Sweep: encoder choice (3-4 candidates) × pooling, then HP retune.
    - **Budget: 20-30 trials post-implementation**. Prior: 0.40 — genuinely new signal that the restart explicitly hasn't tried.

18. **Movie poster images** — visual encoder → embedding. Niche but a different modality.
    - Implementation cost: highest (image fetch, vision model).
    - **Budget: 20+ trials**. Prior: 0.20 (likely redundant with text + genome).

19. **User-level tag genome (re-add)** — was removed in the strip; could be re-added via a *learned* aggregation rather than the fixed mean-of-rated≥4. E.g., attention pool of `user_history * genome` weighted by rating-centered weights.
    - Pre-screen: 3-4 variants
    - **Budget: 3 + 4 = 7 trials**. Prior: 0.20 — tests whether the strip removed real signal or just bad aggregation.

### Operating notes

- **Run ideas in priority order** within each tier; don't skip ahead unless the cheaper ones lift surprisingly fast (rare per legacy history).
- **Always multi-seed verify any single-seed qualifier** (single-seed lift ≥ +0.0007 → 4 verify seeds). The pivot sweep showed how much per-seed values can shift; don't promote based on SEED=42 alone.
- **Budget audit**: tier-1 totals ~63 trials (~3 hours), tier-2 ~36 (~2 hours), tier-3 ~29 (~1.5 hours). Tiers 4-5 are larger, multi-cycle commitments.
- **Cumulative prior on at least one tier-1+2 win**: roughly 0.85 (if all 9 ideas were independently tried). The marginal lift from the *best* of these is the actual question.

## Useful references

- `legacy/` — full history of the prior project, including its `program.md`, `results.tsv`, and `CLAUDE.md`. Available if useful; not authoritative for the restart.
