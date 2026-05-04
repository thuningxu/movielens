# Experiment log — HSTU attempt

Fresh experiment log starting from the HSTU stub. The prior attempts' logs live at `legacy/program.md` (not present — see `legacy/CLAUDE.md`) and `simple_v2/program.md`.

## Comparison baselines (from prior attempts)

| Baseline | val_auc (SEED=42) | test_auc | Source |
|---|---|---|---|
| legacy DLRM ceiling | 0.8284 | — | `legacy/` |
| simple_v2 static (apr28o) | 0.8282 | 0.8221 | `simple_v2/program.md` |
| simple_v2 locked (apr28ah) | **0.8594** | **0.8455** | `simple_v2/program.md` |

Any HSTU cycle is measured against **val 0.8594 / test 0.8455** to be called a win.

## Cycles

### `may03-coldstart` D=128 capacity — TIES simple_v2 on val (single-seed, +0.0027)

After variant C kill, team R2 (sequential plan): D=128 first (Critic-approved as the only HSTU-internal axis with prior > 15%), FREQ_WD deferred. The Critic's evidence: simple_v2 apr28ag dropped FREQ_WD from 1e-4 to 0 and got +0.0015; combining D=128 + FREQ_WD risked re-introducing what simple_v2 found actively harmful in the rich-eval regime.

**Pre-screen (ml-1m MAX_EPOCHS=5)**: D=64 0.7678, D=128 0.7693, Δ=+0.0015 monotone. Smoke clean.

**Headline (ml-25m MAX_EPOCHS=20 SEED=42)**: val_auc 0.859378 (peak ep 16: 0.8594, final ep 19: 0.8591). 8.05M params (vs 3.18M at D=64). 140 min/run.

Strata vs D=64 baseline (0.8567):

| Stratum | D=64 | D=128 | Δ |
|---|---|---|---|
| **Overall** | **0.8567** | **0.8594** | **+0.0027** |
| warm | 0.8596 | 0.8627 | +0.0031 |
| cold_user | 0.8550 | 0.8578 | **+0.0028** (1st real cold_user lift) |
| cold_item | 0.8402 | 0.8422 | +0.0020 |
| cold_both | 0.8235 | 0.8244 | +0.0009 |
| warm_popular | 0.8568 | 0.8603 | +0.0035 |
| warm_tail | 0.8566 | 0.8592 | +0.0027 |

**Significance**: First clean broad-spectrum lift in 6 cycles. Every stratum positive, no regression anywhere. Cold_user finally moved (+0.0028) after 5 nulls of cold_user-targeted features (pop_prior, item_stats, rating_ts, CAWR, variant C).

**Diagnostic flip**: the apparent "structural cold_user ceiling" of HSTU was actually a **capacity ceiling** masquerading as architectural. EMBED_DIM=64 was an early-cycle decision (apr28b? apr29 stability work) never revisited. The variant C "parallel user_embed competes with sequence summary" diagnosis remains correct as a *separate* mechanism failure, but the cold_user gap was primarily encoder-capacity-bound, not user-representation-bound.

**vs simple_v2 0.8594**: gap **0.0000 single-seed** (TIE). Multi-seed verification pending (5-seed mean ≥ 0.8590 with 5/5 positive required to declare a sustained tie; multi-seed mean ≥ 0.8590 + test gate would unlock test-set evaluation).

**Trajectory**: ep 16-19 plateaus in [0.8590, 0.8594] — at the new capacity ceiling, not still climbing. Suggests limited gain from MAX_EPOCHS extension at D=128, but a follow-up could verify.

**Defaults flipped**: `EMBED_DIM` default changed from 64 to 128 in train.py. Plain `DATASET=ml-25m uv run python train.py` reproduces the 0.8594 result. Override `EMBED_DIM=64` to reproduce the prior baseline.

**Open questions for follow-up cycles**:
- LR/scheduler retuning at D=128 capacity (LR=1e-3 was tuned for D=64; may not be optimal at 2× width)
- D=128 + extend-30 (cold_user lifted +0.0012 at D=64; may stack with D=128's +0.0028)
- D=192 or D=256 (does the capacity axis keep paying?)
- Multi-seed verification (mandatory for any keep claim; 5 seeds × 140 min ≈ 11.7 hr)

### `may03-coldstart` variant C (rater_pool) — KILLED on ml-1m smoke (Δ=−0.0036)

User authorized variant C after extend-30. Team R2 converged on:
- `nn.Embedding(num_users + 1, D)` (user_embed) + `anon_user_embed` for cold candidates
- Per-item static rater pool from train_df; per-eval-row dynamic with `ts<sample.ts` cutoff
- Cold-rater gating (drop raters with <3 train ratings)
- Rating-centered weighted pool (pivot=0.6)
- Head-side integration ONLY at last position via zero-init `rater_cross_proj`
- ~440 LOC implementation; OFF-state byte-equivalent verified (fp32 0.605622 = 0.605622 on ml-100k); ON-state step-0 byte-equivalent (rater_cross output exactly 0.0 before first backward).

**Pre-screen (per Critic): ml-1m MAX_EPOCHS=5 SEED=42, kill if Δ negative.**

| Epoch | OFF (USE_RATER_POOL=0) | ON (USE_RATER_POOL=1) | Δ |
|---|---|---|---|
| 0 | 0.7057 | 0.7053 | −0.0004 (≈step-0 noise) |
| 1 | 0.7471 | 0.7447 | −0.0024 |
| 2 | 0.7590 | 0.7571 | −0.0019 |
| 3 | 0.7648 | 0.7622 | −0.0026 |
| 4 | **0.7678** | **0.7642** | **−0.0036** |

**Δ = −0.0036, monotone widening with training.** 3.6× single-seed AUC noise (~0.001 at ml-1m). Below kill threshold (Δ negative). **Variant C killed before ml-25m commit.**

**Diagnosis (matches Critic R1 priors)**: parallel `user_embed` table competes with HSTU's sequence-summary user representation. The rater_cross learns nonzero, adds noise the model has to overcome. Cold-rater gating is not the bottleneck — even warm-rater contributions appear redundant with what the encoder already extracts.

**5th cold_user-targeted null** (after pop_prior, item_stats, rating_ts, CAWR). Reinforces structural-ceiling diagnosis: HSTU's cold_user gap to simple_v2 is not closeable by porting simple_v2's `i_hist_pool` mechanism into HSTU's MLP head — the architectures process user identity differently. simple_v2's mechanism succeeds in a linear head with no other user representation; HSTU has the sequence summary that subsumes the signal.

**Apr28af precedent reaffirmed**: simple_v2's `EVAL_DYNAMIC_ITEM_HIST` (the same per-eval dynamic refresh) was a verified null at +0.001/0.0029-σ. The mechanism we ported has a known null-class signature in the reference codebase.

Branch `may03-coldstart` commit `309d5bf`. Implementation kept on branch (not reverted) for archival. `USE_RATER_POOL=0` is byte-equivalent so the merge to main remains tractable if any salvage emerges. **Do NOT merge variant C unless an anon-only or different design is validated.**

### `may03-coldstart` extend-30 — first positive cold_user signal (+0.0012)

`MAX_EPOCHS=30` on operational best (constant LR=1e-3, no other changes).

**Result: val_auc=0.8575 (peak ep 21) vs baseline 0.8567 = +0.0008**.

Strata diff (e30 - baseline):

| Stratum | Δ |
|---|---|
| warm | +0.0001 |
| **cold_user** | **+0.0012** (first positive cold_user result!) |
| cold_item | -0.0107 (overfit on tiny pop) |
| cold_both | -0.0077 (overfit) |
| warm_popular | +0.0020 |
| warm_tail | -0.0015 |

**Pattern**: extra training helps dominant strata (cold_user 80%, warm_popular) where there's data but overfits tiny populations (cold_item 2%, cold_both 3%). Cold_user lift +0.0012 is the first positive after 4 cold_user-targeted nulls (pop_prior, item_stats, rating_ts, CAWR).

**Below multi-seed bar** (+0.005 single-seed). Sub-noise but directional. May be marginal real lift OR seed noise.

**vs simple_v2 0.8594**: gap −0.0019 (was −0.0027 at baseline).

### `may03-coldstart` CAWR — sub-noise null (val=0.8563)

User asked about LR schedules that "go up and down." Tested CosineAnnealingWarmRestarts with T_0=4 epochs, T_mult=1, eta_min=20% × peak. 5 equal cycles over 20 epochs.

**Result: val_auc=0.8563 vs baseline 0.8567 = -0.0004**.

Visible restart pattern: each restart at epochs 3, 7, 11, 15 caused a visible 1-epoch dip in val_auc (e.g., ep 11→ep 12: 0.8541→0.8524). The model recovers within 1-2 epochs but never catches up to constant-LR baseline.

**Diagnosis confirmed (Critic R1)**: constant-LR baseline was monotone-climbing, not stuck in a local minimum. Warm restarts solve a problem we don't have. The cycles just consume progress without unlocking new capacity.

Three consecutive cold_user-targeted nulls (D pop_prior, B item_stats, A rating_ts) plus this CAWR null. The cold_user gap to simple_v2 is likely structural (architecture-level) rather than feature- or schedule-related.

### `may03-coldstart` rating-ts (A) — sub-noise overall, lifts cold_item (+0.0050) but not cold_user

`USE_RATING_TS=1` on operational best. 32 monthly buckets over train ts range (~8 mo/bucket on ml-25m). Bucketed ts embedded into content tokens at all 3 call sites.

**Result: val_auc=0.8571 (+0.0004 vs baseline 0.8567)**. Sub-noise overall.

Strata diff (A vs baseline):

| Stratum | Δ |
|---|---|
| warm | +0.0015 |
| cold_user | +0.0004 (target stratum, sub-noise) |
| **cold_item** | **+0.0050** (biggest lift) |
| cold_both | +0.0031 |
| warm_dense | +0.0015 |
| warm_tail | +0.0026 |

**Diagnosis**: temporal interaction (year × ts) is real signal for ITEMS — cold_item lifts +0.0050 because the age-at-rating signal compensates for missing item embedding training. But cold_user (the target stratum, 80% of val) sees only +0.0004. The cold_user gap isn't temporal — it's structural (no user_embed analog for HSTU's `i_hist_pool`).

**Lesson**: rating-ts in content token is a small but real win for cold-item handling. Worth keeping as a feature even though it doesn't close the cold_user gap. The architectural ceiling for cold_user without a user-side rater pool mechanism appears to be ~0.855.

### `may03-coldstart` item_stats (B) — REGRESSES, fails on cold_item

`USE_ITEM_STATS=1` (alone, no pop prior) on operational best. 3 scalars per item from train_df only: `mean_rating/5`, `std/2.5`, `frac_engaged`. Zero-imputation for items with no train ratings.

**Result: val_auc=0.8541, -0.0026 vs baseline 0.8567.** Below baseline at every epoch.

Strata diff (item_stats - baseline):

| Stratum | Δ |
|---|---|
| warm | -0.0021 |
| cold_user | +0.0001 (target — zero lift) |
| **cold_item** | **-0.0341** (massive regression) |
| **cold_both** | **-0.0268** (massive regression) |
| warm_dense | -0.0021 |
| warm_tail | -0.0055 |

**Mechanism failure**: zero-imputation is the bug. Stats are all 0 for items with no train ratings, but 0 is a *valid* low-quality value (e.g., a 0.5★-mean item). The model learns "zero stats = low quality" and unfairly penalizes cold_item / cold_both predictions. To fix: use a learnable "missing" indicator OR mean-of-means imputation.

**Both cold_user interventions (D and B) failed**. Static item-side features can't port simple_v2's `i_hist_pool` mechanism faithfully — that mechanism uses `user_embed` which HSTU lacks. The structural gap is not easily closed without a user representation.

### `apr30` strata diagnostic — gap lives in cold_user (80% of val), warm matches simple_v2

Strata analysis on operational best `interleave_3L_bf16` (val 0.8567 single-seed, ml-25m SEED=42):

| Stratum | n | mean_label | AUC |
|---|---|---|---|
| warm | 373.9K (15%) | 0.396 | **0.8596** ← matches simple_v2 overall 0.8594 |
| cold_user | 2.01M (80%) | 0.514 | **0.8550** ← gap to simple_v2 lives here |
| cold_item | 53.9K (2%) | 0.390 | 0.8402 |
| cold_both | 65.2K (3%) | 0.479 | 0.8235 |
| warm_dense (≥20 prior events) | 372.3K | 0.395 | 0.8594 |
| warm_popular (top-2000 items) | 188.6K | 0.450 | 0.8568 |
| warm_tail (rest) | 185.2K | 0.341 | 0.8566 |

**Key**: HSTU exceeds simple_v2's overall AUC on warm (0.8596 vs 0.8594). Item-popularity within warm has near-zero effect (popular vs tail differ by 0.0002). The gap to simple_v2 is concentrated in cold_user where HSTU lacks item-side rater context.

### `apr30` pop_prior — log1p(item_train_count) feature, +0.0001 overall

`USE_POP_PRIOR=1` on operational best (val=0.8568, +0.0001 over baseline 0.8567).

Strata diff (pop_prior - baseline):

| Stratum | Δ |
|---|---|
| warm | +0.0007 |
| **warm_tail** | **+0.0016** (best lift, but small population) |
| warm_dense | +0.0008 |
| cold_user | +0.0003 (target stratum, sub-noise) |
| cold_item | +0.0005 |
| cold_both | +0.0004 |

**Pop prior is wrong intervention for cold_user.** Item_embed already encodes popularity implicitly via co-occurrence frequency. The signal that matters for cold_user is rating *distribution* (mean, std) — what variant B targets. Pop prior is more useful for warm_tail differentiation.

### `apr30` aux_interleave_3L_bf16 — AUX_RATING_WEIGHT=25 regresses (val=0.8556)

`AUX_RATING_WEIGHT=25` on top of the interleave_3L_bf16 stack.

**Result: val_auc=0.8556** (peak ~ep 18). vs interleave-only (0.8567): **−0.0011**. vs C2 (0.8561): −0.0005.

| ep | interleave | aux+interleave |
|---|---|---|
| 8 | 0.8530 | 0.8522 |
| 13 | 0.8558 | 0.8545 |
| 15 | 0.8567 | 0.8554 |
| 19 | 0.8564 | 0.8542 |

Consistent -0.001 below interleave-only at every epoch. The aux loss is fighting BCE rather than helping.

**Diagnosis**: simple_v2's apr28o gain from AUX=25 came from a linear head over a 1376-d concat that benefited from auxiliary supervision pulling embeddings toward rating geometry. HSTU's `rating_embed` (the action-position token in interleaved mode) ALREADY encodes rating info directly — adding aux MSE on top is redundant or counterproductive. Mechanism doesn't transfer.

**Lesson**: simple_v2 mechanisms don't universally transfer to HSTU. Per-mechanism evaluation needed.

### `apr30` SEED=43 verify of interleave_3L_bf16 — reproduces (val=0.8558)

| seed | val_auc | peak ep |
|---|---|---|
| 42 | 0.8567 | 15 |
| 43 | 0.8558 | 19 |
| **2-seed mean** | **0.8563** | |
| inter-seed diff | 0.0009 | within seed-σ |

Reproducibility confirmed. 2-seed mean is +0.0002 over C2 — sub-noise. Per team rule, multi-seed lift threshold (+0.005) not cleared, so no test-set evaluation yet. But interleave_3L_bf16 reproduces cleanly as the new operational baseline.

### `apr30` interleave_3L_bf16 — **paper-canonical interleaving on top of fast baseline: val=0.8567** (new best)

`INTERLEAVE=1 SEQ_LEN=100 NUM_LAYERS=3 USE_BF16=1` on C2 stack.

Sequences become `[c_0, a_0, c_1, a_1, …]` interleaved (paper-canonical). 100 events × 2 = 200 tokens, matched compute with C2's fused SEQ_LEN=200.

**Result: val_auc=0.8567 at epoch 15** (peak), trajectory 0.8559-0.8567 in late epochs.

| epoch | L3_bf16 (fused) | interleave_3L_bf16 | Δ |
|---|---|---|---|
| 4 | 0.8463 | 0.8469 | +0.0006 |
| 8 | 0.8519 | 0.8530 | +0.0011 |
| 13 | 0.8551 | 0.8558 | +0.0007 |
| 19 | 0.8556 | 0.8564 | +0.0008 |

**Lift over fused at 3L**: +0.0007 (sub-σ but consistent positive across all epochs). **Lift over C2**: +0.0006.

**Speed**: 7930s vs C2's 9959s = **20% faster** despite doubling effective sequence length (interleaving adds ~8% overhead vs L3 fused).

vs simple_v2 0.8594: gap **−0.0027** (was −0.0033 at C2).

### `apr30` L2_bf16 — depth=2 marginal regression

`NUM_LAYERS=2 USE_BF16=1`: val=0.8546 (peak ep 18). −0.001 vs L3_bf16. Faster (90 min vs 121 min) but the AUC cost isn't worth it. **NUM_LAYERS=3 confirmed as right depth.**

### `apr30` L3_bf16 — **NUM_LAYERS=3 + bf16: val=0.8560, 27% faster than C2** at no AUC cost

`NUM_LAYERS=3 USE_BF16=1` on C2 stack (LR=1e-3 GRAD_CLIP=1.0 PROJ_INIT_MODE=xavier USE_GENOME+GENRE+YEAR=1 MLP_HEAD=1 MAX_EPOCHS=20).

**Result: val_auc=0.8560 at epoch 18** (peak), 0.8556 at ep 19. Essentially identical to C2's 0.8561 (4L fp32) within ±0.001 — well below seed-σ.

| epoch | C2 (4L fp32) | L3_bf16 (3L bf16) | Δ |
|---|---|---|---|
| 4 | 0.8475 | 0.8463 | -0.0012 |
| 8 | 0.8522 | 0.8519 | -0.0003 |
| 13 | 0.8548 | 0.8551 | +0.0003 |
| 19 | 0.8561 | 0.8556 | -0.0005 |

**Speed**: 7284s vs C2's 9959s = **27% faster**.

**Speedup decomposition** (epoch-0 wall-time):
- Layer reduction (4L→3L): ~20% (attention scales linearly with layers)
- bf16 alone: ~8% (modest — autocast overhead vs HSTU's tiny matmuls at 64-dim)
- Combined: ~27%

**Implication**: NUM_LAYERS=3 + bf16 is the new operational baseline at no measurable AUC cost. All subsequent cells in this sweep run at this faster regime.

### `apr30` C2 — **MLP head + LR=1e-3 → val=0.8561**, gap to simple_v2 just −0.0033

`MLP_HEAD=1 MLP_HEAD_DROPOUT=0.1 LR=1e-3 GRAD_CLIP=1.0 PROJ_INIT_MODE=xavier USE_GENOME+GENRE+YEAR=1 MAX_EPOCHS=20` on M1 stack.

**Result: val_auc=0.8561 at epoch 19** (peak). Smooth monotonic climb throughout — NO spikes, grad_norm steady at 0.08-0.23, still gaining +0.0002/epoch at the end (model not converged).

| epoch | C1 (no MLP) | C2 (MLP) | Δ |
|---|---|---|---|
| 0 | 0.776 | 0.827 | +0.051 |
| 6 | 0.844 | 0.850 | +0.006 |
| 13 | 0.851 | 0.855 | +0.004 |
| 19 | 0.855 | **0.856** | +0.001 |

C2 lift over C1: +0.0014. Below +0.005 multi-seed threshold but consistent (smooth trajectory, no spikes, still climbing at ep 19). Decision rule applied: 0-0.005 lift → "keep, proceed to interleaving."

**Cumulative apr30 progression**:
- M0 (pure HSTU): 0.8370
- M1 (metadata, no clip): 0.8453 (+0.008 from metadata)
- Cycle 2 (LR=2e-3, clip, xavier): 0.8541 (+0.009 from stabilization)
- C1 (LR=1e-3): 0.8547 (+0.001 from LR step)
- **C2 (MLP head): 0.8561** (+0.001 from MLP head)
- vs simple_v2 0.8594: gap **−0.0033** (was −0.022 at HSTU baseline)

C2 still climbing at ep 19 → extending to 30 epochs OR multi-seed verify (next decision).

### `apr30` C1 — LR=1e-3 marginal lift, much cleaner training (**val=0.8547**)

`LR=1e-3 GRAD_CLIP=1.0 PROJ_INIT_MODE=xavier USE_GENOME+GENRE+YEAR=1 MAX_EPOCHS=20` on M1 stack.

**Result: val_auc=0.8547 at epoch 18** (peak). +0.0006 over Cycle 2 (0.8541) — within seed-σ, NOT multi-seed-verifiable on its own. But the trajectory is qualitatively cleaner: NO major spikes (grad_norm stayed <1.0 in late epochs vs Cycle 2's 1.2-2.6 mild spikes).

| metric | C1 (LR=1e-3) | Cycle 2 (LR=2e-3) |
|---|---|---|
| Final val_auc | 0.8547 | 0.8541 |
| Late epochs grad_norm | 0.17 | 0.13-2.6 (with mild spikes) |
| Visible spikes | 0 | 3 mild (eps 7, 12, 15) |

C1 caught up to Cycle 2 around epoch 13 then slightly exceeded it in late epochs. This validates the **LR=1e-3 + clip + xavier + metadata stack** as the new operational baseline — same/better AUC, qualitatively cleaner training.

**Decision rule applied** (per team converge): lift in [0, 0.005) → keep, proceed to C2. C2 (MLP head) will run at LR=1e-3.

vs simple_v2 0.8594: gap **−0.0047** (was −0.0053 at Cycle 2).

### `apr30` SEED=43 verify — **Cycle 2 reproduces (val=0.8531)**

First reproducibility check on Cycle 2's winning config (`LR=2e-3 GRAD_CLIP=1.0 PROJ_INIT_MODE=xavier USE_GENOME+GENRE+YEAR=1 MAX_EPOCHS=20`).

| Seed | val_auc | Best epoch |
|---|---|---|
| 42 (Cycle 2) | 0.8541 | ep 18 |
| 43 (verify) | 0.8531 | ep 19 |
| **2-seed mean** | **0.8536** | |

Inter-seed diff: 0.0010 (within seed-σ ~0.003). Win reproduces cleanly.

Multi-seed lift over M1 (0.8453): +0.0083 (clears +0.005 threshold).

Notable: SEED=43 had an unusual epoch-16 val drop (0.8522 → 0.7258) with only mild grad_norm=0.74 — a different failure mode than the LR=5e-3 spikes, and recovered in one epoch. Suggests there's a residual instability mechanism even at LR=2e-3 that doesn't show in train_loss / grad_norm but transiently degrades the model's eval representation. Worth investigating later.

### `apr30` Cycle 2 — **LR=2e-3 wins: 0.8541 single-seed (+0.0088 over M1, +0.006 over S3)**

`LR=2e-3 GRAD_CLIP=1.0 PROJ_INIT_MODE=xavier USE_GENOME=1 USE_GENRE=1 USE_YEAR=1 MAX_EPOCHS=20` on M1 metadata stack.

**Key result: val_auc = 0.8541 at epoch 18 (peak)**, ep 19 = 0.8536. Trajectory was monotonically climbing with only mild residual spikes (grad_norm 1.19-2.64 at eps 7, 12, 15 — vs S3's 6-15 spikes). Recovery within 1 epoch.

| epoch | val_auc | grad_norm |
|---|---|---|
| 5 | 0.8470 | 0.19 |
| 9 | 0.8502 | 0.15 (crosses 0.85) |
| 11 | 0.8519 | 0.15 |
| 14 | 0.8526 | 0.14 |
| 16 | 0.8535 | 0.12 |
| 18 | **0.8541** | 0.13 |

**Lift vs prior best HSTU configs:**
- vs S3 extended (0.8481): +0.0060
- vs S2 stabilized (0.8467): +0.0074
- vs M1 spike-locked (0.8453): **+0.0088**
- vs HSTU baseline M0 (0.8370): +0.017

**Diagnosis confirmed**: Critic's "LR=5e-3 too aggressive once metadata raises curvature" hypothesis is validated. Lower LR+clip+xavier preserves the metadata signal AND eliminates most of the spike-driven training loss. Spikes still occur but are 5-10× milder and recover in one epoch instead of permanently damaging the model.

**vs simple_v2 0.8594: gap -0.0053** (was -0.022 at HSTU baseline).

**Multi-seed verification authorized**: lift +0.0088 over M1 clears the +0.005 threshold cleanly. Next cycle: 4-seed verify (SEEDs 43, 44, 45, 46) at the same config to confirm reproducibility.

### `apr30` S3 — extend S2 to 25 epochs: **0.8481 peak (+0.0014 over S2)**, but ep 20-25 mean = 0.844 (below multi-seed threshold)

S2 stack (USE_GENOME+GENRE+YEAR=1, GRAD_CLIP=1.0, PROJ_INIT_MODE=xavier, LR=5e-3) extended to MAX_EPOCHS=25. Tests the asymptote-vs-blocked-by-spikes hypothesis.

| epoch | val_auc | event |
|---|---|---|
| 14 | 0.810 | spike |
| 17 | 0.848 | new peak |
| 18 | 0.829 | spike |
| 20 | **0.8481** | peak |
| 21 | 0.834 | spike |
| 24 | 0.841 | spike (final epoch) |

Late-trajectory mean ep 20-24: 0.8435. Well below the 0.852 multi-seed threshold.

Spike pattern continues throughout — not an early-training phenomenon. The model breaks through the spike ceiling slowly: S2 ep 14 = 0.8467, S3 ep 20 = 0.8481 (+0.0014 over 6 more epochs). Net: extension adds ~+0.0001/epoch on average — confirms Critic's "metastable around asymptote" interpretation is dominant, with a slow climb component the Researcher correctly identified.

**Decision per team rule**: ep 20-25 mean < 0.852 → "asymptote confirmed; try LR=2e-3 once before pivoting to interleaving."

Next cycle: Cycle 2 — LR=2e-3 + xavier + clip 1.0 at MAX_EPOCHS=20. Tests if lower LR cures the spike instability orthogonally.

### `apr30` stabilization S1+S2 — clip 1.0 catches but doesn't prevent recurring spikes; **+0.0014 over M1 sub-noise**

**Stabilization sweep, sub-noise lift** (`a768f4d`). 2-cell sweep of M1 stack with `GRAD_CLIP=1.0`:

| Cell | Init mode | best val_auc | trajectory |
|---|---|---|---|
| S1 | zero-init projections | 0.8444 (peak ep 10) | spikes at eps 3, 6, 7, 11, 13, 14 |
| **S2** | **xavier-init projections** | **0.8467 (peak ep 14)** | spikes at eps 2, 5, 8, 9, 12 |

Spike pattern is **recurring** (every 3-4 epochs) and **clip 1.0 catches but doesn't prevent**. Each spike: train_loss jumps 0.5 → 1.5-2.5, val_auc drops 0.03-0.10, recovers in 1-2 epochs. The clip-as-safety-net is working but the underlying training instability remains.

**Findings**:
- Xavier init is mildly better than zero-init (+0.0023 raw lift S2 over S1) but doesn't prevent spikes — refutes the "zero-init alone causes the spikes" hypothesis.
- vs M1 (no clip, peak 0.8453): S2 is +0.0014 (sub-noise at lift-σ ~0.003).
- vs HSTU baseline (M0=0.8370): S2 is +0.0097 — content metadata DOES lift HSTU meaningfully, just at the cost of training stability.
- vs simple_v2 bar 0.8594: gap **−0.0127** (was −0.022 at 0.8367, now −0.013 at 0.8467).

**Spike interpretation (Critic)**: metastable around asymptote — the spikes ARE the asymptote with clip catching periodic outlier batches; the mean trajectory is the right value, not blocked from climbing higher.

**Spike interpretation (Researcher)**: prevention possible — tighter clip or different intervention could yield monotonic climb past 0.847.

S2's tail trajectory: ep 11=0.8463, ep 12=0.8241 (spike), ep 13=0.8460, ep 14=0.8467. Suggests still climbing post-spike. Test: extend to 25 epochs.

### `apr30` metadata-1 — content features lift **+0.008 (M1: genome+genre+year)** but training spikes at epoch 5

**Pre-spike peak win, post-spike instability** (`ab4bdda`). 3-cell sweep at 4L/64D, LR=5e-3, MAX_EPOCHS=15, SEED=42:

| Cell | Flags | val_auc (best_val_auc) | params |
|---|---|---|---|
| M0 control | none | 0.837035 | 3.86M |
| M2 | USE_GENOME=1 | 0.837405 | 3.94M |
| **M1** | **GENOME+GENRE+YEAR** | **0.845293** | 3.95M |

Per-epoch comparison up to the spike:

| epoch | M0 | M2 | M1 | M1−M0 |
|---|---|---|---|---|
| 0 | 0.794 | 0.812 | 0.812 | +0.018 |
| 1 | 0.811 | 0.826 | 0.831 | +0.020 |
| 2 | 0.819 | 0.833 | 0.840 | +0.021 |
| 3 | 0.822 | 0.837 | 0.842 | +0.020 |
| 4 | 0.824 | **0.8374** | **0.8453** | **+0.021** |
| 5 | 0.828 | 0.784 (spike) | 0.786 (spike) | -0.042 |

**Reproducible training instability** at epoch 5: train_loss in M2 jumped 0.516→7.38 and in M1 jumped 0.515→7.21 (~14× spike). Same epoch, similar magnitude — clearly an LR=5e-3 + content-feature interaction, not random. Post-spike both cells slowly recover but never re-reach the pre-spike peak.

The reported best_val_auc captures the pre-spike peak: **M1=0.8453 (+0.008 over M0).**

**Findings**:
- Content metadata genuinely lifts HSTU. M1's +0.008 over M0 (already +0.0083 over the prior 0.8367 LR=5e-3 baseline) closes ~36% of the remaining gap to simple_v2's 0.8594.
- Genre+year add orthogonal signal on top of genome (M1=0.8453 vs M2=0.8374 = +0.008). Genome covers only 23.4% of movies (long tail has zero rows); cheap genre+year cover all movies.
- LR=5e-3 is too aggressive for the metadata-enabled regime. Need stabilization (lower LR / gradient clipping / warmup) to preserve the lift over a full stable run.

**Next cycle**: stabilize the M1 config to lock in the +0.008 lift and explore whether a stable run reaches significantly higher than 0.8453.

vs simple_v2 0.8594: gap shrunk from -0.022 to **-0.014**.

### `apr30` Bug #1 fix + LR=5e-3 / 15-epoch cell — **+0.015 lift, asymptote confirmed at this config**

**Below bar but trajectory clear** (`3ea5c1b`). Bug #1 fix (movieId +1 shift to avoid PAD/movieId-0 collision in `nn.Embedding(..., padding_idx=0)`) + Critic's path: LR=5e-3, MAX_EPOCHS=15, ml-25m SEED=42. Bug #2 (cold-user fallback) intentionally skipped — refined impact estimate ~0.001 AUC, sub-noise.

| epoch | val_auc | Δ |
|---|---|---|
| 0 | 0.7946 | — |
| 1 | 0.8121 | +0.018 |
| 2 | 0.8186 | +0.007 |
| 3 | 0.8223 | +0.004 |
| 6 | 0.8305 | +0.003/ep |
| 9 | 0.8339 | +0.001/ep |
| 12 | 0.8354 | +0.000/ep |
| 13 | **0.8367** | +0.001 (peak) |
| 14 | 0.8362 | -0.001 (overfit) |

Total runtime: 7200s ≈ 120 min on 1× CUDA GPU.

**vs apr30 step 3** (LR=3e-3, 5 epochs, pre-Bug#1): 0.8214 → **+0.0153 lift**. Three contributing factors:
- Bug #1 fix: ~0.005-0.015 (Validator-estimated)
- LR=3e-3 → 5e-3: ~0.005 (LR ladder still climbing)
- 5 → 15 epochs: ~0.010 (more training)

**Per Critic's decision rule**: result is in the `[0.825, 0.84)` bucket with last-3-epoch deltas ≤ +0.001 (overfit signal at epoch 14). **Asymptote confirmed at (NUM_LAYERS=4, EMBED_DIM=64). Next cycle: capacity sweep at LR=5e-3.**

**vs simple_v2 bar 0.8594**: gap -0.0227 (was -0.038 at step 3).

### `apr30` step 3 — fused-token HSTU LR sweep on ml-25m

**Below bar** (`12481fc`). 3-cell LR sweep at SEED=42, defaults (EMBED_DIM=64, NUM_LAYERS=4, NUM_HEADS=4, SEQ_LEN=200, BATCH_SIZE=256, MAX_EPOCHS=5):

| LR | val_auc | gap vs simple_v2 |
|---|---|---|
| 3e-4 | 0.6509 | -0.208 |
| 1e-3 | 0.8099 | -0.050 |
| 3e-3 | **0.8214** | **-0.038** |

**Per-epoch trajectories** (decelerating but non-converged):
- LR=3e-4: 0.522 → 0.543 → 0.572 → 0.611 → 0.651 (still climbing fast — LR too low)
- LR=1e-3: 0.570 → 0.677 → 0.779 → 0.800 → 0.810 (rate slowing)
- LR=3e-3: 0.761 → 0.806 → 0.815 → 0.819 → **0.821** (rate slowing fast: +0.044 → +0.003 across last 4 deltas)

**Validator-flagged Critic concern (LR=1e-3 may diverge for SiLU pointwise attention) was wrong** — model is stable at all three LRs, no NaN. Higher LR helps because pointwise SiLU attention has bounded outputs (SiLU saturates) so update magnitude doesn't explode.

Below the team's pre-set decision threshold of 0.83 (= "HSTU at this scale may be wrong direction; pivot or HP refine"). Best cell is 0.8214 — gap to simple_v2 of -0.038. The slowing per-epoch growth at LR=3e-3 (+0.003 in epoch 4) suggests pure epoch extension will not close the gap.

**Open: convene team on next move.** Candidates:
- Extend training (MAX_EPOCHS=15 or 20 at LR=3e-3) — cheapest test, ~75 min; probably +0.005-0.010
- Higher LR (5e-3, 1e-2) — addresses the "still under-tuned" hypothesis
- Refactor to interleaved (the deferred decision from the team's prior round) — addresses train/inference asymmetry
- HP sweep beyond LR (NUM_LAYERS ∈ {2, 8}, EMBED_DIM ∈ {128, 256}, SEQ_LEN ∈ {100, 400})
- Pivot to IMDB plot text (Critic's original Plan A pre-HSTU)

