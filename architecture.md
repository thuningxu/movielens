# Model Architecture

DLRM with rating-aware DIN + causal self-attention, item-side DIN, 4 GDCN cross layers, FinalMLP two-stream with bilinear.

**val_auc = 0.806 on ml-25m** | ~13M params | ~9.7 GB VRAM on NVIDIA L4 | ~140 experiments

```mermaid
graph TD
    subgraph Inputs
        UID["userId"]
        MID["movieId"]
        HIST["User History<br/>(last 100 items + ratings)"]
        IHIST["Item History<br/>(last 30 raters + ratings)"]
        DENSE_RAW["Dense Features (17)<br/>timestamp, user rating hist (5-bin),<br/>user count, item rating hist (5-bin),<br/>item count, ug_dot, year,<br/>genre_count, movie_age"]
        GENRE["Genre Multi-Hot (20)"]
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
        HIST_E --> CAUSAL["Causal Self-Attention<br/>(single head, Q/K/V linear)<br/>items only *"]
        CAUSAL --> CONTEXTUAL["Contextual History<br/>(B, 100, 28)"]
        CONTEXTUAL --> DIN_CAT["Concat: contextual + ratings<br/>(B, 100, 56)"]
        HIST_RAT --> DIN_CAT
        DIN_CAT --> DIN["DIN Attention<br/>(target-item-aware)<br/>Linear(168, 64) - ReLU - Linear(64, 1)"]
        ITEM_E -.->|target query| DIN
        DIN --> USER_HIST_E["user_hist_e<br/>dim=28"]
    end

    subgraph "Item-Side DIN"
        RATER_E --> IDIN_CAT["Concat: raters + ratings<br/>(B, 30, 56)"]
        RATER_RAT --> IDIN_CAT
        IDIN_CAT --> IDIN["Item DIN Attention<br/>(target-user-aware)<br/>Linear(168, 64) - ReLU - Linear(64, 1)"]
        USER_E -.->|target query| IDIN
        IDIN --> ITEM_HIST_E["item_hist_e<br/>dim=28"]
    end

    subgraph "Feature Processing"
        DENSE_RAW --> BOTTOM_MLP["Bottom MLP<br/>17 - 128 - 28 - ReLU"]
        BOTTOM_MLP --> DENSE_E["dense_e<br/>dim=28"]
        GENRE --> GENRE_PROJ["Genre Proj<br/>Linear(20, 28) - ReLU"]
        GENRE_PROJ --> GENRE_E["genre_e<br/>dim=28"]
    end

    subgraph "GDCN Cross Network (4 layers)"
        USER_E --> CONCAT["Concat all 6 fields<br/>6 x 28 = 168 dim"]
        ITEM_E --> CONCAT
        USER_HIST_E --> CONCAT
        ITEM_HIST_E --> CONCAT
        DENSE_E --> CONCAT
        GENRE_E --> CONCAT
        CONCAT --> X0["x0 (168)"]
        X0 --> GCL1["Gated Cross Layer 1<br/>cross = x0 * (W @ xi + b)<br/>gate = sigmoid(G @ xi)<br/>xi+1 = gate * cross + (1-gate) * xi"]
        GCL1 --> GCL2["Gated Cross Layer 2"]
        GCL2 --> GCL3["Gated Cross Layer 3"]
        GCL3 --> GCL4["Gated Cross Layer 4"]
        GCL4 --> X4["x4 (168)"]
    end

    subgraph "FinalMLP Two-Stream"
        USER_E --> US_CAT["User Stream Input<br/>user_e + user_hist_e + dense_e"]
        USER_HIST_E --> US_CAT
        DENSE_E --> US_CAT
        US_CAT --> US["User Stream MLP<br/>84 - 256 - 64<br/>(dropout 0.2)"]
        US --> US_OUT["user_stream (64)"]

        ITEM_E --> IS_CAT["Item Stream Input<br/>item_e + item_hist_e + genre_e"]
        ITEM_HIST_E --> IS_CAT
        GENRE_E --> IS_CAT
        IS_CAT --> IS["Item Stream MLP<br/>84 - 256 - 64<br/>(dropout 0.2)"]
        IS --> IS_OUT["item_stream (64)"]

        US_OUT --> BILINEAR["Bilinear Interaction<br/>element-wise product"]
        IS_OUT --> BILINEAR
        BILINEAR --> BIL_OUT["bilinear (64)"]
    end

    subgraph "Top MLP - Output"
        X4 --> FINAL_CAT["Concat<br/>168 + 64 + 64 + 64 = 360"]
        US_OUT --> FINAL_CAT
        IS_OUT --> FINAL_CAT
        BIL_OUT --> FINAL_CAT
        FINAL_CAT --> TOP["Top MLP<br/>360 - 256 - 128 - 64 - 1<br/>(dropout 0.2)"]
        TOP --> LOGIT["logit (scalar)"]
        LOGIT --> SIGMOID["sigmoid"]
        SIGMOID --> PRED["P(engage)"]
    end

    subgraph "Training"
        LOGIT --> LOSS["BCEWithLogitsLoss<br/>label smoothing 0.1"]
        LOSS --> OPT["Adam, LR=1e-4<br/>weight_decay=1e-5<br/>AMP fp16, grad accum 8x<br/>batch=16384, eff. 131K"]
    end

    style Inputs fill:#e1f5fe
    style CAUSAL fill:#fff3e0
    style DIN fill:#fff3e0
    style IDIN fill:#fff3e0
    style GCL1 fill:#f3e5f5
    style GCL2 fill:#f3e5f5
    style GCL3 fill:#f3e5f5
    style GCL4 fill:#f3e5f5
    style US fill:#e8f5e9
    style IS fill:#e8f5e9
    style BILINEAR fill:#e8f5e9
    style TOP fill:#fce4ec
    style PRED fill:#c8e6c9
```

\* **Known design note:** Causal self-attention operates on item embeddings only, then the output is concatenated with per-position rating embeddings for DIN. This means the contextual embedding at position i (a weighted mix of items 0..i) is paired with the rating specifically for item i. A residual connection or combined input was tested but gave identical AUC (0.806), suggesting the model compensates for this misalignment.
