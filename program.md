# autoresearch — MovieLens Recommendation (restart, apr28)

Fresh experimentation loop on top of a deliberately simple linear baseline. Same data + features + metric as the prior project at `legacy/`; new starting architecture and new experiment log.

## Why restart?

The prior project (see `legacy/`) reached val_auc = 0.8284 on ml-25m via several hundred experiments converging on a DLRM-style architecture, then plateaued. Rather than continue iterating in that local optimum, this restart begins from the simplest possible model (single Linear head on concatenated features), keeping the input features and prediction task unchanged, and rebuilds the architecture deliberately from below.

## Setup

- **Branch convention**: each experiment cycle on `restart/<tag>` (e.g., `restart/apr28` for the initial scaffold).
- **`prepare.py`**: do not modify (`evaluate()` is the ground-truth metric).
- **`train.py`**: the experimentation file. Currently the linear baseline; will grow.
- **Multi-seed discipline**: estimate the seed-noise floor empirically before declaring any win. A few baseline seeds give you σ; require multi-seed verification with mean lift comfortably above that floor and a positive sign at every seed.
- **Feature cache**: `data/features_<hash>.npz` is built on first ml-25m run and reused. Cache key is `restart-5` (`restart-3` added per-item training-positive count; `restart-4` added per-user genome aggregate; `restart-5` added per-user count + per-user genre affinity).

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

### `autoresearch/apr28aa` — legacy recency port + anon-user fallback + OOV decomp — null

**Null** (`cba61f2`). User-prompted exploration sequence after noticing two underexplored ideas: legacy's "fewer + more recent" trick (program.md learning #15: +0.00108 5-seed mean), and the cold-start problem (70.59% of ml-25m val users are OOV; 82.89% of val ratings touch an OOV user). Researcher/Critic 3-round debate converged on a 5-cell concurrent batch testing both mechanisms.

Pre-screen at SEED=42 (vs 0.828188):

| Cell | val_auc | Δ |
|---|---|---|
| P0 baseline | 0.828188 | 0 (sanity ✓) |
| P1 RECENCY=0.7 only | 0.812769 | -0.0154 |
| P2 RECENCY=0.7 + RESAMPLE | 0.814493 | -0.0137 |
| P3 ANON_FALLBACK=1 only | **0.828188** | **0 (bit-equal!)** |
| P4 full stack | 0.826251 | -0.0019 |

OOV-stratified eval (P3 vs P4 — 4 strata: warm / cold_user / cold_item / cold_both):

| Stratum | n | P3 (ANON) | P4 (recency+resample+ANON) |
|---|---|---|---|
| warm | 471K (12.6%) | 0.8029 | 0.8156 (+0.013) |
| cold_user | 2.6M (70.3%) | 0.7870 | 0.7888 (+0.002) |
| cold_item | 125K (3.4%) | 0.8702 | 0.8701 (~0) |
| cold_both | 511K (13.7%) | 0.9056 | 0.9078 (+0.002) |

**Two diagnostic findings:**

1. **ANON_FALLBACK is broken by design.** anon_user_embed never receives gradient during training because no training row is cold — every training userId has ≥1 real rating in train (easy-negs are anchored to positive samples' users → easy-neg users are warm). anon stays at init zeros forever. P3 vs P0: bit-equal val_auc to 6 decimals (random u_e for OOV val users contributes label-uncorrelated noise below AUC resolution). Fix would require **stochastic warm-row dropout**: with probability p, replace a warm user's u_e with anon during training to force gradient into anon.

2. **Recency port doesn't transfer to apr28o stack.** Legacy got +0.00108 5-seed mean on the field-attention DLRM; apr28aa gets -0.015 on the linear head. Per-stratum P4 is better than P3 on EVERY stratum (warm +0.013, cold_user +0.002, cold_both +0.002), but the OVERALL val_auc regresses by -0.002 — Simpson's paradox. The recency filter rebalances score magnitudes across strata; cold_user dominates (70% of rows) and its sub-noise lift can't compensate for the cross-stratum ranking redistribution. The apr28o stack's AUX_RATING head likely needs the full real-rating distribution to train its MSE head; dropping 30% breaks that.

3. **Cold-start IS the bottleneck.** Cold_user AUC 0.787 vs warm 0.815 (+0.028 headroom). Cold_user is 70% of val rows. But neither mechanism in apr28aa moves it.

**Lessons**:
- The OOV-decomposed eval is now a permanent diagnostic (15 LOC, eval-only, ~3s overhead).
- ANON_FALLBACK as designed is dead. Future cold-start fixes need training-time gradient into the anonymous embedding (e.g., stochastic warm-row dropout, or inductive embedding via item-history aggregation).
- Recency mechanism is structurally incompatible with the apr28o stack via the AUX head dependency on full rating distribution. Could try without AUX_RATING but that's giving up an established win.
- Sub-noise stack candidates from apr28o-style stacking are exhausted.

**12 nulls now** (apr28p-aa). Final apr28-restart baseline: 0.828188. The cold-start headroom is real but requires structurally different mechanisms (val-time history rebuild, IMDB plot text, or an inductive user representation that can be trained without cold training rows).

### `autoresearch/apr28z` — all DIN/attention variants — sub-noise null

**Null** (`36896ae`). Per user's "try all DINs" directive, ran 8 cells covering every untested attention/DIN variant: user-side dot-product attention (mirror of apr28y), user-side DIN-MLP-parallel at hidden ∈ {16, 32}, item-side DIN-MLP-parallel, user-side DIN-replace at hidden ∈ {8, 16}, plus rating-modulation and cross variants on user-side dot-product attention.

Pre-screen at SEED=42 (vs 0.828188):

| Cell | val_auc | Δ |
|---|---|---|
| **uattn_a** (USER_ATTN=1) | 0.829011 | **+0.000823** |
| uattn_c (USER_ATTN + RATING_MOD) | 0.828945 | +0.000757 |
| uattn_b (USER_ATTN + CROSS) | 0.828892 | +0.000704 |
| idin_par_h16 (item DIN-MLP) | 0.826308 | -0.001880 |
| udin_rep_h16 (DIN-replace h=16) | 0.825905 | -0.002283 |
| udin_par_h32 (user DIN-MLP h=32) | 0.823900 | -0.004288 |
| udin_rep_h8 (DIN-replace h=8) | 0.822800 | -0.005388 |
| udin_par_h16 (user DIN-MLP h=16) | 0.791600 | **-0.036588** (collapse) |

Three USER_ATTN variants signed single-seed; all DIN-MLP and DIN-replace variants regressed. Pattern confirms apr28r/v/y lessons: parametric capacity adds at the apr28o stack hurt; only zero-param mechanisms can sign.

5-seed verify of cell A (USER_ATTN=1):

| SEED | baseline | uattn_a | lift |
|---|---|---|---|
| 42 | 0.828188 | 0.829011 | +0.000823 |
| 43 | 0.828073 | 0.827671 | -0.000402 |
| 44 | 0.828024 | 0.828563 | +0.000539 |
| 45 | 0.828098 | 0.828462 | +0.000364 |
| 46 | 0.828266 | 0.828509 | +0.000243 |

Mean lift **+0.000313**, 4/5 positive, min **-0.000402**, lift-σ ≈ 0.000455. Below all three multi-seed bars (mean ≥ +0.0007, 5/5 positive, min ≥ -0.0003). Same RNG-collapse shape as apr28q (+0.000188 mean) and apr28w (+0.000123 mean), but largest sub-noise mean of any single mechanism in 11 cycles.

**Stacking attempt** (apr28o-style super-additive): tested USER_ATTN with USER_HIST_MEAN_POOL (+0.000188 sub-noise from apr28q) and b65k_lr12e4 (+0.000123 sub-noise from apr28w):

| Stack | single-seed Δ | vs USER_ATTN alone |
|---|---|---|
| USER_ATTN + UHMP | +0.000715 | -0.000108 (UHMP aliases) |
| USER_ATTN + b65k_lr12e4 | +0.000245 | -0.000578 (b65k hurts) |
| triple (all three) | +0.000020 | -0.000803 (collapse) |

**No super-additivity.** UHMP aliases USER_ATTN (both pool over u_hist_e); large-batch + large-LR actively interferes with USER_ATTN's gradient flow. The apr28o stacking trick does NOT generalize — the apr28o components were structurally orthogonal (CROSS_TS_ITEM = new field, FREQ_WD_LAMBDA = regularization, AUX_RATING_WEIGHT = parallel objective); the apr28z candidates are all in similar mechanism families and interfere.

**Lessons**:
- USER_ATTN extracts a real but small signal (+0.000313 mean) at L=100 user history, where the existing diagonal cross misses non-linear similarity reweighting.
- Item-side dot-product attention (apr28y) is null — IL=30 history is short enough that the existing cross saturates the available signal.
- DIN with MLP scoring catastrophically fails at the apr28o stack (parametric capacity replay of apr28r/v).
- The apr28o sub-noise stacking trick works only for orthogonal mechanism families; it does NOT generalize to multiple aggregator-side adds.

**Final apr28-restart baseline: 0.828188.** No cycle has cleared the multi-seed bar since apr28o.

### `autoresearch/apr28y` — item-side target-aware attention — null

**Null** (`95d0eb1`). User raised the structural concern that the item-history pool throws away an obvious signal: rater weights are unconditioned by user-rater similarity. Existing `i_hist_pool ⊙ u_e` cross gives the linear head `Σ_k w_k · ⟨a ⊙ u_e, rater_k⟩` (linear functional of per-rater similarities, weighted by rating-centered weight). Target attention would add user-conditional **non-linear** softmax reweighting on similarity.

3-round Researcher/Critic debate converged on a 3-cell pre-screen of dot-product target attention as a parallel pooling field on item history. Zero learnable params (cells A, B); 1 scalar α for rating-modulation (cell C, init=0).

```python
scores = (u_e.unsqueeze(1) * i_hist_e).sum(-1) / sqrt(D)   # (B, IL=30)
scores = scores.masked_fill(i_valid < 0.5, -inf)
attn = softmax(scores, dim=1)
attn_out = (i_hist_e * attn.unsqueeze(-1)).sum(1)          # (B, 28)
```

Pre-screen at SEED=42 (vs 0.828188):

| Cell | val_auc | Δ |
|---|---|---|
| A — attn-only | 0.827898 | -0.000290 |
| B — attn + cross | 0.827785 | -0.000403 |
| C — rating-modulated (learnable α) | 0.827901 | -0.000287 |

All 3 regress; below kill threshold (+0.0003). NULL, no verify.

**Lesson**: Critic's orthogonality concern — that the existing diagonal-scaled rating-weighted cross already extracts the linear-functional version of the per-rater similarity signal — was correct. Non-linear softmax reweighting on similarity adds no detectable signal at the apr28o stack; it slightly hurts, likely from optimizer-time interference with the existing rating-centered pool path. Closes the "non-linear similarity reweighting on item history" door.

### `autoresearch/apr28t-x` — final cheap-cycle batch — all null

Per user direction "finish all cheap cycles," ran 5 sub-cycles (~2 hours of GPU time) covering the remaining cheap candidates from the backlog. All NULL. Eight consecutive null cycles total (apr28p-x). 0.828188 is established as the final linear-head baseline.

**Cumulative summary at SEED=42 vs 0.828188:**

| Sub-cycle | Mechanism | Best cell | Δ | Verdict |
|---|---|---|---|---|
| apr28t | POOL_PIVOT re-sweep at apr28o stack | pivot=0.65 | -0.0005 | pivot=0.6 still optimal |
| apr28u | USER_FREQ_WD_LAMBDA | 1e-5 | -0.0001 | Monotonic regression |
| apr28v | USER_GENRE_AFFINITY_CROSS (20-d Hadamard) | on | -0.0020 | apr28p/r-pattern regression |
| apr28w | BATCH × LR cotune | b65k_lr12e4 | +0.000353 single → +0.000123 5-seed | RNG artifact at SEED=42 |
| apr28x | CROSS_GENRE_GENOME (per-movie content scalar) | on | -0.0001 | Sub-noise |

apr28w was the only signing single-seed cell (+0.000353); 5-seed verify collapsed to +0.000123 (3/5 positive). Same shape as apr28q's RNG artifact: lift σ across seeds (~0.00018) is large relative to the mean lift, so SEED=42 routinely produces a false-positive single-seed signal.

**Lesson set across apr28p-x (8 nulls):**
- Parametric capacity additions on existing fields (apr28p DCN cross, apr28r two-tower, apr28v genre-affinity cross): all regress monotonically with capacity. The linear head + 4 manual crosses + aux head is at saturation.
- Aggregator-side parallel pools (apr28q multi-pool): alias rating-centered pool on right-skewed rating distribution.
- Single-scalar new-information signals (apr28s user-genome, apr28x genre-genome): sub-noise; the existing fields already capture the marginal information at this representation.
- HP retunes (apr28t pivot, apr28u user-freq-WD, apr28w batch-LR): all sub-noise; the apr28o-tuned HPs are at the right optimum for this representation.
- The +0.0007 multi-seed bar is probably never going to be cleared by any single mechanism on this representation. Even apr28o's win came from STACKING three sub-noise mechanisms super-additively.

**Takeaway:** the 8-null streak doesn't mean apr28 is wrong; it means the representation has been mined exhaustively. Future lift requires NEW input information (IMDB plot summaries, Tier 5 #17, prior 0.40, ~2-week build) or a structurally different prediction objective. Both are out of single-cycle scope and warrant their own roadmaps.

**Cheap-cycle phase: closed at 0.828188.**

### `autoresearch/apr28s` — user × candidate genome scalar dot — null

**Null** (`9a42a86`). After apr28p/q/r ruled out parametric capacity additions, Researcher pivoted to NEW INFORMATION: precompute per-user genome aggregate (rating-centered weighted average of historical movies' genomes, 1128-d per user) and expose `dot(user_genome_agg, candidate_genome) / GENOME_DIM` as a single scalar field appended to the concat.

Critic rejected the original 2256-dim Hadamard cross variant on direct legacy evidence: legacy ran ~63 trials of this family, learning #10 says "user-genome content alignment is information-bottlenecked at ONE scalar; vector forms overfit catastrophically (apr27 cycle 8: -0.0175)." Counter-proposal: scalar-only.

Cache version bumped `restart-3 → restart-4` to add `user_genome_agg` precompute. Off-state byte-equivalent (flag default 0). Implementation chunks the per-user aggregate to avoid the (162541, 100, 1128) ≈ 73 GB intermediate.

Pre-screen single cell at SEED=42 (vs 0.828188 baseline):

| Cell | val_auc | Δ |
|---|---|---|
| `USER_GENOME_AGG_DOT=1` | 0.828298 | +0.000110 |

Below Critic's +0.0003 kill threshold. NULL, no verify.

**Lesson:** Legacy's +0.000944 win on the DLRM does not transfer to the linear head. Two plausible reasons: (a) the linear head cannot exploit the user-genome compatibility scalar as efficiently as the DLRM's interaction layers did; (b) apr28o's stack already extracts most of the genome signal via the existing `genome` (raw 1128-d) field + `i_hist_pool ⊙ u_e` cross — the user-genome compatibility scalar is largely redundant once those are in the concat.

**Combined signal across apr28p/q/r/s (four consecutive nulls):** the linear-head + 4-cross + aux-rating baseline at 0.828188 is genuinely close to the representation's ceiling for ml-25m. Future cycles need either (a) genuinely new input features (IMDB plot text — Tier 5 expensive), (b) a different prediction objective (sequential next-item — Tier 4 #15), or (c) accept the ~0.0002 gap to legacy DLRM as inherent to the simpler architecture.

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
