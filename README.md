# Model Architecture

DLRM with rating-aware DIN + causal self-attention (with residual), item-side DIN, tag genome with learned bottleneck compression, 1-head field attention, FinalMLP two-stream with bilinear.

**Single model: val_auc = 0.821 on ml-25m** | ~13M params | ~7.6 GB VRAM on NVIDIA L4
**Ensemble (22 models): val_auc = 0.824 on ml-25m** | LogReg stacking of architecturally diverse variants
**~460 experiments total**

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
        LOSS --> OPT["Adam, LR=8e-5<br/>weight_decay=5e-5<br/>AMP fp16, grad accum 4×<br/>batch=16384, eff. 65K<br/>NEG_RATIO=1, patience=3<br/>eval 3×/epoch"]
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

## Ensemble

The best results come from ensembling architecturally diverse models. Key insight: models need **low prediction correlation** to provide complementary signal.

**Best ensemble: 0.8242 AUC** (LogReg stacking, 22 models, 5-fold CV)

Top ensemble members (by contribution):
- `fieldattn` (0.821) — current best single model
- `meanpool` (0.819) — no attention, mean pooling history (correlation 0.944)
- `ratingpool` (0.819) — rating-weighted mean pooling (correlation ~0.94)
- `noitemdin` (0.821) — no item-side DIN
- `nostream` (0.818) — no two-stream separation
- `dim16` (0.818) — embed_dim=16, much smaller model

Models with >0.97 prediction correlation with fieldattn (GDCN, nogenome) add little ensemble value. The most valuable partners are those with fundamentally different inductive biases.
