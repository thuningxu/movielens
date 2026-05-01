# MovieLens — HSTU generative recommendation

Predict whether a user will rate a movie ≥ 4 stars (positive engagement). Same task and metric as the prior attempts — different model: **HSTU** (Hierarchical Sequential Transduction Units, Meta 2024 — *Actions Speak Louder than Words*).

## Why HSTU, why a fresh start?

Two prior attempts on this task are archived in `legacy/` and `simple_v2/`:

- `legacy/` — DLRM-style architecture (causal SA + DIN + tag-genome bottleneck + 4-layer MLP). Reached **val 0.8284** on ml-25m after ~540 experiments. Architecture family confirmed saturated by two ceiling tests (apr27, apr27c).
- `simple_v2/` — apr28 restart from a single Linear head over engineered features (rating-centered history pools, multi-hot genres, raw tag genome, manual cross fields, eval-time dynamic user history). Reached **val 0.8594 / test 0.8455** with ~6M params. Confirmed locked at apr28ah after 5 consecutive nulls — the engineered-feature representation has no remaining lift.

This attempt drops the "engineered features going into a scoring head" framing entirely. **HSTU treats the user as a token stream** of (item, action, time) events and uses pointwise causal attention to predict the next event's engagement target. No hand-specified history pools, no tag-genome concat, no cross fields. The model sees raw events; representations emerge from the sequence.

## Why HSTU specifically (not OneRec / PinRec / TIGER)

- **HSTU** is point-estimate scoring (per-event probability) — keeps AUC eval apples-to-apples vs simple_v2 / legacy.
- **OneRec / PinRec** are encoder-decoder generative *retrieval* (beam-search a ranked slate) — closer to "true generative" but force NDCG@K and are 2-3× the code.
- **TIGER** sits between the two, with semantic-ID quantization. Higher cold-start ceiling than HSTU but bigger build (RQ-VAE + encoder-decoder + decoding) and a metric change.

HSTU is the cleanest first generative baseline that lets us measure against the apr28ah locked number directly. If HSTU lifts meaningfully, we can layer semantic-ID quantization on top later.

## Layout

- **`prepare.py`** — Shared with both prior attempts. Data download + time-based train/val/test splits + `evaluate()` AUC harness. **Do not modify.**
- **`train.py`** — HSTU model + training loop. Currently a stub.
- **`program.md`** — Experiment log for this attempt (starts empty).
- **`legacy/`** — Frozen archive of the original DLRM project.
- **`simple_v2/`** — Frozen archive of the apr28 linear-head restart. The locked baseline (`val 0.859384`, `test 0.845497`) lives there for comparison.
- **`data/`** — Auto-downloaded MovieLens datasets; not in git.

## Quickstart

```bash
uv sync

# Smoke test (ml-100k, ~seconds)
DATASET=ml-100k uv run python train.py

# Standard experiment (ml-25m on the current CUDA GPU)
DATASET=ml-25m uv run python train.py
```

## Architecture

```mermaid
graph TD
    subgraph "Per-user event sequence (left-padded, length L)"
        EV["events[t] = (movieId_t, rating_bucket_t, timestamp_t)<br/>t = 0 .. L-1, real events at the right end"]
    end

    subgraph "Static metadata tables (apr30, opt-in via USE_*)"
        GMT["genome_table: (num_items+1, 1128)<br/>nn.Buffer, fixed at load"]
        GRT["genre_table: (num_items+1, 20)<br/>multi-hot, nn.Buffer"]
        YRT["year_id_table: (num_items+1,)<br/>year - 1850 clipped to [0, 199]"]
    end

    subgraph "item_full_embed(m) — symmetric helper"
        EV -.->|"movieId_t"| IFE["item_embed[m]<br/>+ USE_GENOME · genome_proj(genome_table[m])<br/>+ USE_GENRE · genre_proj(genre_table[m])<br/>+ USE_YEAR · year_embed(year_id_table[m])"]
        GMT -.-> IFE
        GRT -.-> IFE
        YRT -.-> IFE
        IFE --> IE["x = item_full_embed(m_t) + rating_embed[rating_bucket_t]<br/>→ (B, L, D)"]
        EV --> IE
    end

    subgraph "Time-delta bias (precomputed once per batch)"
        EV --> TD["pairwise Δt = |ts_i - ts_j|<br/>bucket = 0 if Δt=0, else 1+floor(log2(Δt))<br/>clamp to [0, 31] → (B, L, L) int"]
    end

    subgraph "HSTU block × NUM_LAYERS (=4)"
        IE --> LN1["LayerNorm"]
        LN1 --> UVQK["Linear(D, 4D) → SiLU<br/>split → U, V, Q, K  each (B, L, D)"]
        UVQK --> MH["reshape Q, K, V → (B, H=4, L, D/H=16)"]
        TD --> RB["rel_bias = Embedding(32, H)[bucket]<br/>→ (B, H, L, L)"]
        MH --> SCORE["scores = Q · Kᵀ / √(D/H) + rel_bias<br/>→ (B, H, L, L)"]
        RB --> SCORE
        SCORE --> POINT["SiLU(scores) ⊙ keep_mask<br/>(causal × pad-key) — POINTWISE, NO softmax"]
        POINT --> AV["AV = scores · V → reshape (B, L, D)"]
        AV --> GLU["LayerNorm(AV) ⊙ U<br/>(gated linear unit — no separate FFN)"]
        GLU --> WO["Linear(D, D)"]
        WO --> RES["x + out (residual)"]
    end

    subgraph "Heads (use the SAME item_full_embed helper)"
        RES --> TRH["Train: per-position<br/>dot(h_t, item_full_embed(events[t+1].movieId))<br/>→ BCE on engaged(events[t+1])"]
        RES --> EVH["Eval: dot(h_{L-1}, item_full_embed(candidate))<br/>→ sigmoid → P(engage)"]
        IFE -.->|"shared"| TRH
        IFE -.->|"shared"| EVH
    end

    style EV fill:#e1f5fe
    style IFE fill:#fff3e0
    style POINT fill:#fce4ec
    style GLU fill:#fff3e0
    style EVH fill:#c8e6c9
    style GMT fill:#f3e5f5
    style GRT fill:#f3e5f5
    style YRT fill:#f3e5f5
```

The signature HSTU departures from a vanilla causal transformer:

- **Pointwise (SiLU) attention, not softmax.** No row-wise normalization over keys.
- **Gated linear unit replaces FFN.** `LayerNorm(AV) ⊙ U` does both attention output and channel-mixing in one step.
- **Relative-position bias from log-bucketed time deltas.** Per-block, per-head learnable bias table over 32 time-gap buckets (0 = same instant, 31 = >2³⁰ s ≈ 34 yr). Init zeros so the model starts as a no-bias HSTU.

Per Meta 2024 §3.

### Content metadata injection (apr30, Idea 1)

Three opt-in flags wire MovieLens content into HSTU:

- `USE_GENOME=1` — MovieLens 1128-d tag-genome (relevance scores per tag-id, dense)
- `USE_GENRE=1` — 20-d genre multi-hot from `movies.csv`
- `USE_YEAR=1` — year embedding via title regex `(\d{4})`, bucketed `clip(year - 1850, 0, 199)`

Single symmetric helper `item_full_embed(m)` adds projected metadata to `item_embed[m]`. The same helper is used at three call sites:

1. **Sequence input** — every position's input is `item_full_embed(m_t) + rating_embed(r_t)`
2. **Per-position training target** — `dot(h_t, item_full_embed(events[t+1].movieId))` (no rating, since rating is the label)
3. **Eval candidate scoring** — `dot(h_{L-1}, item_full_embed(candidate))`

Whatever metadata-fused embedding the sequence sees as observations is exactly what the candidate gets at scoring time — symmetric by design, no separate "head augmentation" branch.

**Off-state byte-equivalence**: when all three flags are 0, the projection modules (`genome_proj`, `genre_proj`, `year_embed`) are not constructed at all (RNG state preserved). At ON state, projection weights are zero-init so step-0 logits match OFF; signal grows monotonically as projections train.

**Tables stored as `nn.Buffer(persistent=False)`**: 254 MB on GPU at ml-25m for the genome table; not in `state_dict` (rebuilt at load).

## Status

`apr30` branch implements the full pipeline:
- **Step 1** (`5bf86c6`): sequence-level training with per-position causal loss (SASRec/HSTU framing).
- **Step 2** (`e057e70`): real HSTU block — pointwise SiLU attention + gated linear unit + log-bucketed time-delta bias.
- **Bugfix** (`12481fc`): cold/short-history users were getting the wrong hidden state at eval — left-padding makes `seq_len-1` always the last real event.
- **Bug #1 fix** (`3ea5c1b`): movieId +1 shift to avoid PAD/movieId-0 collision in `nn.Embedding(..., padding_idx=0)`.
- **LR=5e-3, MAX_EPOCHS=15** (`a046154`): val 0.8367 with 4L/64D, no metadata. **Already beats heavily-tuned simple_v2 static (0.828) and legacy DLRM (0.8284) by +0.008 with no movie content features at all** — pure user-item interaction modeling.
- **Capacity sweep** (cells A/B): width-doubling at preserved depth gave zero lift; capacity along this axis is not the binding constraint.
- **Content metadata** (`ab4bdda`): USE_GENOME / USE_GENRE / USE_YEAR opt-in flags added to test the structural-content-gap hypothesis. Default off (byte-equivalent baseline).

Current ml-25m sweep in flight: 3-cell metadata-1 (M0 control / M2 genome-only / M1 all metadata) at 4L/64D LR=5e-3 MAX_EPOCHS=15.

## What carries over from the prior attempts

- `prepare.py` data pipeline + `evaluate()` AUC harness (the metric is the ground truth)
- The label scheme: rating ≥ 4 → positive (1); rating < 4 OR random unrated → negative (0)
- The dataset choice: ml-25m as the default; ml-100k for smoke tests
- The discipline learnings (multi-seed verification, deterministic SEED=42, etc.) — see `legacy/CLAUDE.md` learnings #14, #15

## What does NOT carry over

- Any model architecture or layer choices
- Any feature engineering (history pools, multi-hot fields, tag genome, cross products, eval-time dynamic-history mechanism)
- Any hyperparameter defaults — HSTU has its own (embed dim, num layers, num heads, sequence length)
- Any inherited assumption about input structure
