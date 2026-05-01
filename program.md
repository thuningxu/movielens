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

(none yet — train.py is a stub)
