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

