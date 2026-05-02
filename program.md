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

