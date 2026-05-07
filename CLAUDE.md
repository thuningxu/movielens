# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status: concluded

Three sequential attempts on MovieLens engagement prediction (predict whether a user will rate a movie ≥ 4 stars on ml-25m at SEED=42). Final result: **HSTU sliding-window val 0.8626 / test 0.8652**. See `README.md` for the project-wide arc and headline comparison.

| Attempt | val | test | Subdir |
|---|---|---|---|
| DLRM ceiling | 0.8284 | — | `legacy/` |
| Linear head + engineered features | 0.8594 | 0.8455 | `simple_v2/` |
| HSTU sequence model | **0.8626** | **0.8652** | `hstu/` |

Each subdirectory has its own `CLAUDE.md`, `program.md`, and `train.py`. Read the relevant per-attempt `CLAUDE.md` before working in that attempt.

## Layout

- **`prepare.py`** — Shared across all three attempts. `load_data()` returns raw `train`/`val`/`test` DataFrames; `evaluate(labels, scores)` is the AUC ground truth. **Do not modify.**
- **`pyproject.toml`** — Shared `uv` environment.
- **`legacy/`** — Attempt 1 (DLRM). Frozen archive.
- **`simple_v2/`** — Attempt 2 (linear head + engineered features). Frozen archive.
- **`hstu/`** — Attempt 3 (HSTU sequence model). Frozen at the may6 ceiling declaration; see `hstu/CLAUDE.md` for the operational-best config and the may6 cycle docs.
- **`data/`** — Auto-downloaded MovieLens datasets; gitignored.
- **`checkpoints/`, `logs/`** — Run artifacts; gitignored.

## When working in a subdirectory

Each attempt has its own conventions and operational config. Read the per-attempt `CLAUDE.md` first:
- `legacy/CLAUDE.md` — DLRM baseline, ~540 experiments of HP tuning history.
- `simple_v2/CLAUDE.md` — apr28 linear-head stack with eval-time dynamic-history mechanism.
- `hstu/CLAUDE.md` — HSTU operational best (D=128 sliding) plus the may6 ceiling declaration and structural follow-up ideas.

## Cross-cutting discipline

- **Multi-seed verification mandatory for any keep claim.** HSTU seed-noise floor σ ≈ 0.0001 (n=2 sliding regime) — sub-0.001 single-seed lifts routinely turn null on second seed.
- **`prepare.py:evaluate()` is the ground truth.** Never modify the AUC harness; cross-attempt comparisons depend on identical metric semantics.
- **Test-set generalization is the final arbiter.** simple_v2 val→test −0.0139; HSTU val→test +0.0026. Don't infer test rank from val rank across architecture classes.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **movielens** (1635 symbols, 1796 relationships, 10 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/movielens/context` | Codebase overview, check index freshness |
| `gitnexus://repo/movielens/clusters` | All functional areas |
| `gitnexus://repo/movielens/processes` | All execution flows |
| `gitnexus://repo/movielens/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
