# MovieLens Recommendation — Hybrid Engagement Prediction

Predict whether a user will rate a movie >= 4 stars (positive engagement). Hybrid task with hard negatives (rated < 4) and easy negatives (random unrated). ~500 experiments on ml-25m.

**Current restart baseline: val_auc = 0.8284 at SEED=42 on ml-25m (apr27b 100-trial HP sweep, +0.0012 over apr26 cycle-8)**
**Best historical single model: val_auc = 0.821 (apr04 architecture summarized below)**
**Best historical ensemble: val_auc = 0.854 (HistGBM stacking of 59 diverse models, 3-fold CV)**

The checked-in `train.py` is the restart baseline (squeeze-and-excitation field reweighting + scalar-dot user-genome alignment) and differs from the apr04 best historical single-model architecture summarized below. Treat the code as the source of truth for the current implementation, and this README as a summary of historical results.

## AUC Progress

```mermaid
%%{init: {'theme': 'default'}}%%
xychart-beta
    title "AUC Improvement Over Time (~500 single-model experiments)"
    x-axis ["apr01", "apr01", "apr01", "apr01", "apr02", "apr02", "apr02", "apr02", "apr03c", "apr03c", "apr03e", "apr03e", "apr03e", "apr03e", "apr04", "apr25", "apr26", "apr27b"]
    y-axis "val_auc" 0.76 --> 0.83
    line "Single Model" [0.770, 0.781, 0.793, 0.799, 0.802, 0.804, 0.806, 0.806, 0.811, 0.814, 0.817, 0.820, 0.821, 0.821, 0.821, 0.8263, 0.8272, 0.8284]
```

### Key milestones

| Date | AUC | Experiments | What worked |
|------|-----|------------|-------------|
| Apr 1 | 0.770 | 0 | Baseline DLRM on ml-25m |
| Apr 1 | 0.799 | ~30 | HISTORY_LEN=100, 3 GDCN layers, causal self-attention |
| Apr 2 | 0.806 | ~120 | Rating histograms, 4 GDCN layers, embed_dim=28 |
| Apr 3 | 0.814 | ~170 | Tag genome with learned bottleneck compression |
| Apr 4 | 0.821 | ~250 | NEG_RATIO=1, WD=5e-5, ACCUM=4, LR=8e-5 (historical FinalMLP two-stream) |
| Apr 25 | 0.8263 | ~430 | Restart on SE field-reweighting baseline: anchor-pos-catalog negatives + post-recency neg resample + rating-pooled causal histories |
| Apr 26 | 0.8272 | ~440 | Per-user genome profile + scalar-dot user×item content alignment routed into `genome_field` (cycle-8 win, +0.000944 over 0.82628) |
| **Apr 27b** | **0.8284** | **~540** | **100-trial HP sweep stacking 4 sub-noise single-knob lifts: `RECENCY_FRAC=0.7`, `POST_RECENCY_EASY_NEG_PER_POS=0.4`, `GENOME_BOTTLENECK_DROPOUT=0.0`, `MLP_DROPOUT=0.3` — 5-seed mean lift +0.00170 (5/5 positive)** |

## Best Historical Single-Model Architecture

This section describes the best historical single-model variant from the Apr 4-5 experiments, not necessarily the exact checked-in `train.py`.

```mermaid
graph TD
    subgraph Inputs
        UID["userId"]
        MID["movieId"]
        HIST["User History<br/>(last 100 items + ratings)"]
        IHIST["Item History<br/>(last 30 raters + ratings)"]
        DENSE_RAW["Dense Features (17)<br/>timestamp, user rating hist (5-bin),<br/>user count, item rating hist (5-bin),<br/>item count, ug_dot, year,<br/>genre_count, movie_age"]
        GENRE["Genre Multi-Hot (20)"]
        GENOME_RAW["Tag Genome (1128)<br/>relevance scores<br/>(22% movie coverage)"]
    end

    subgraph "Embedding Layer"
        UID --> USER_E["User Embed<br/>dim=28, dropout=0.1"]
        MID --> ITEM_E["Item Embed<br/>dim=28, dropout=0.1"]
        HIST --> HIST_E["History Item Embed<br/>dim=28"]
        HIST --> HIST_RAT["Rating Proj<br/>Linear(1, 28)"]
        IHIST --> RATER_E["Rater Embed<br/>dim=28, dropout=0.1"]
        IHIST --> RATER_RAT["Rating Proj<br/>Linear(1, 28)"]
    end

    subgraph "User-Side Sequential Modeling"
        HIST_E --> CAUSAL["Causal Self-Attention<br/>(single head, Q/K/V linear)<br/>+ additive residual from raw embeds"]
        CAUSAL --> CONTEXTUAL["Contextual + Raw History<br/>(B, 100, 28)"]
        CONTEXTUAL --> DIN_CAT["Concat: contextual + ratings<br/>(B, 100, 56)"]
        HIST_RAT --> DIN_CAT
        DIN_CAT --> DIN["DIN Attention<br/>(target-item-aware)<br/>Linear(168, 64) → ReLU → Linear(64, 1)"]
        ITEM_E -.->|target query| DIN
        DIN --> USER_HIST_E["user_hist_e<br/>dim=28"]
    end

    subgraph "Item-Side DIN"
        RATER_E --> IDIN_CAT["Concat: raters + ratings<br/>(B, 30, 56)"]
        RATER_RAT --> IDIN_CAT
        IDIN_CAT --> IDIN["Item DIN Attention<br/>(target-user-aware)<br/>Linear(168, 64) → ReLU → Linear(64, 1)"]
        USER_E -.->|target query| IDIN
        IDIN --> ITEM_HIST_E["item_hist_e<br/>dim=28"]
    end

    subgraph "Feature Processing"
        DENSE_RAW --> BOTTOM_MLP["Bottom MLP<br/>17 → 128 → ReLU → 28 → ReLU"]
        BOTTOM_MLP --> DENSE_E["dense_e<br/>dim=28"]
        GENRE --> GENRE_PROJ["Genre Proj<br/>Linear(20, 28) → ReLU"]
        GENRE_PROJ --> GENRE_E["genre_e<br/>dim=28"]
    end

    subgraph "Tag Genome Compression"
        GENOME_RAW --> GENOME_PROJ["Bottleneck MLP<br/>1128 → 256 → ReLU<br/>→ 64 → ReLU → Dropout(0.1)<br/>→ 28"]
        GENOME_PROJ --> GENOME_E_RAW["genome_e (28)"]
        GENOME_E_RAW --> GATE["Sigmoid Gate<br/>Linear(28, 28)"]
        ITEM_E -.->|fallback for<br/>missing genome| GATE
        GATE --> GENOME_FIELD["genome_field (28)<br/>gate * genome_e +<br/>(1-gate) * item_e"]
    end

    subgraph "Field Attention (replaces GDCN)"
        USER_E --> STACK["Stack 7 fields<br/>(B, 7, 28)"]
        ITEM_E --> STACK
        USER_HIST_E --> STACK
        ITEM_HIST_E --> STACK
        DENSE_E --> STACK
        GENRE_E --> STACK
        GENOME_FIELD --> STACK
        STACK --> FATTN["Multi-Head Attention<br/>1 head, dropout=0.1<br/>+ additive residual"]
        FATTN --> X4["x4 (196)<br/>flatten 7 × 28"]
    end

    subgraph "FinalMLP Two-Stream"
        USER_E --> US_CAT["User Stream Input<br/>user_e + user_hist_e + dense_e<br/>(84)"]
        USER_HIST_E --> US_CAT
        DENSE_E --> US_CAT
        US_CAT --> US["User Stream MLP<br/>84 → 256 → ReLU → Dropout(0.2)<br/>→ 64 → ReLU"]
        US --> US_OUT["user_stream (64)"]

        ITEM_E --> IS_CAT["Item Stream Input<br/>item_e + item_hist_e +<br/>genre_e + genome_field<br/>(112)"]
        ITEM_HIST_E --> IS_CAT
        GENRE_E --> IS_CAT
        GENOME_FIELD --> IS_CAT
        IS_CAT --> IS["Item Stream MLP<br/>112 → 256 → ReLU → Dropout(0.2)<br/>→ 64 → ReLU"]
        IS --> IS_OUT["item_stream (64)"]

        US_OUT --> BILINEAR["Bilinear Interaction<br/>element-wise product"]
        IS_OUT --> BILINEAR
        BILINEAR --> BIL_OUT["bilinear (64)"]
    end

    subgraph "Top MLP → Output"
        X4 --> FINAL_CAT["Concat<br/>196 + 64 + 64 + 64 = 388"]
        US_OUT --> FINAL_CAT
        IS_OUT --> FINAL_CAT
        BIL_OUT --> FINAL_CAT
        FINAL_CAT --> TOP["Top MLP<br/>388 → 256 → 128 → 64 → 1<br/>(dropout 0.2)"]
        TOP --> LOGIT["logit (scalar)"]
        LOGIT --> SIGMOID["sigmoid"]
        SIGMOID --> PRED["P(engage)"]
    end

    subgraph "Training"
        LOGIT --> LOSS["BCEWithLogitsLoss<br/>label smoothing 0.1"]
        LOSS --> OPT["Adam, LR=7e-5<br/>weight_decay=1e-4<br/>AMP fp16, grad accum 2×<br/>batch=16384, eff. 33K<br/>NEG_RATIO=1, TRAIN_NEG_MODE=anchor_pos_catalog<br/>POST_RECENCY_NEG_RESAMPLE=1, EASY_NEG_PER_POS=0.75<br/>USER_HIST_MODE=rating<br/>USER_HIST_CONTEXT=causal_masked<br/>ITEM_HIST_CONTEXT=causal_masked<br/>patience=3, eval 3×/epoch"]
    end

    style Inputs fill:#e1f5fe
    style CAUSAL fill:#fff3e0
    style DIN fill:#fff3e0
    style IDIN fill:#fff3e0
    style GENOME_PROJ fill:#e8eaf6
    style GATE fill:#e8eaf6
    style FATTN fill:#f3e5f5
    style US fill:#e8f5e9
    style IS fill:#e8f5e9
    style BILINEAR fill:#e8f5e9
    style TOP fill:#fce4ec
    style PRED fill:#c8e6c9
```

\* **History residual:** Causal self-attention output is added to raw item embeddings before DIN, preserving item identity alongside contextual representation.

\*\* **Tag genome gating:** 78% of movies lack genome data. The sigmoid gate learns to fall back to `item_e` when genome features are zeros. PCA compression failed (0.798); the learned 3-layer bottleneck MLP succeeds (0.811→0.814).

\*\*\* **Field attention replaces GDCN:** 1-head multi-head attention across 7 feature fields with additive residual. Simpler than 4 gated cross layers, slightly better AUC (0.8207 vs 0.8201).
