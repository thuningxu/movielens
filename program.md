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

