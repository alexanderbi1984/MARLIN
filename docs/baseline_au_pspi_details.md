# AU + PSPI Baseline Implementation Details

This document outlines the methodology used to establish a handcrafted feature baseline (Action Units + PSPI) for comparison against deep learning representations (Marlin, Remix/MMA) in the Syracuse Pain Analysis task.

## 1. Feature Extraction Pipeline

### 1.1 Raw Feature Extraction (OpenFace)
We utilized **OpenFace 2.0** to extract facial behavioral features from the raw video clips.
- **Input**: Raw MP4 videos (same source as Marlin/MMA).
- **Tool**: `FeatureExtraction` binary from OpenFace.
- **Outputs**: Frame-by-frame CSV files containing Action Units (Intensity & Presence), Pose, and Gaze.
- **Selected Features**:
  - We specifically selected **17 Action Units (Intensity)**: `AU01_r`, `AU02_r`, `AU04_r`, `AU05_r`, `AU06_r`, `AU07_r`, `AU09_r`, `AU10_r`, `AU12_r`, `AU14_r`, `AU15_r`, `AU17_r`, `AU20_r`, `AU23_r`, `AU25_r`, `AU26_r`, `AU45_r`.
  - **PSPI Calculation**: We computed the **Prkachin and Solomon Pain Intensity (PSPI)** score for each frame using the formula:
    $$PSPI = AU4 + \max(AU6, AU7) + \max(AU9, AU10) + AU43$$
    *(Note: OpenFace provides `AU45` (blink) which is used as a proxy for `AU43` (eyes closed) in this calculation.)*

### 1.2 Temporal Alignment & Preprocessing
To ensure a fair comparison with the **Marlin** baseline, we aligned the temporal structure of the AU features exactly with the Marlin feature bags.

*   **Marlin Strategy**:
    *   Window Size: 16 frames
    *   Stride: 16 frames (Non-overlapping)
    *   Pooling: Learned pooling via VideoMAE encoder $\rightarrow$ 768-dim vector per window.

*   **AU Strategy (Our Implementation)**:
    *   **Window Size**: 16 frames
    *   **Stride**: 16 frames
    *   **Flattening**: Instead of mean pooling (which loses micro-expression dynamics), we **concatenated** the features of all 16 frames in a window.
    *   **Input Dimension**: $(17 \text{ AUs} + 1 \text{ PSPI}) \times 16 \text{ frames} = 18 \times 16 = \mathbf{288}$ dimensions.

The resulting `.npy` feature files have the shape `(N_windows, 288)`, where `N_windows` matches the Marlin feature bags for the same video.

### 1.3 Augmentation Consistency
To maintain rigorous control variables, we applied the **same face-swap augmentation strategy** to the AU baseline.
1.  We ran OpenFace on the **augmented (face-swapped) videos**.
2.  We preprocessed these outputs using the same script (`preprocess_openface_au.py`) to generate `_windows.npy` files.
3.  The training pipeline loads these augmented views with the same probability (`train_aug_ratio`) as used in the Marlin experiments.

## 2. Model Configuration

We utilized the exact same **MIL-CORAL Transformer** architecture used for the deep features, with the only modification being the input projection layer.

*   **Architecture**:
    *   **Backbone**: Fixed (Offline features).
    *   **Input Projector**: Linear layer mapping $D_{in} \rightarrow 256$ (Marlin: $768 \rightarrow 256$; AU: $288 \rightarrow 256$).
    *   **Aggregator**: Transformer Encoder (2 layers, 4 heads) pooling instance embeddings into a bag embedding.
    *   **Heads**:
        *   **CORAL Head**: Ordinal regression head (if enabled).
        *   **CE Head**: Standard Cross-Entropy classification head.
        *   **Auxiliary Head**: Multitask learning on BioVid (if enabled).

*   **Key Parameters (Config: `syracuse_mil_coral_xformer_au.yaml`)**:
    *   `input_dim`: **288**
    *   `embed_dim`: 256
    *   `batch_size`: 64
    *   `task`: Ordinal (4-class or 5-class)
    *   `loocv`: True (Leave-One-Subject-Out Cross-Validation)

## 3. Comparison Protocol

The comparison is conducted under the following strict conditions:

1.  **Data Split**: Identical Leave-One-Subject-Out (LOOCV) splits.
2.  **Multitask Setup**: Both baselines are trained with the same Auxiliary Task (BioVid Pain Classification) to stabilize learning.
    *   Marlin uses BioVid Marlin features (768-dim).
    *   AU uses BioVid AU features (288-dim).
3.  **Metrics**: Evaluation uses Cohen's Kappa (QWK), Accuracy, and F1-Macro averaged across all subjects.

## 4. Script Usage

To run the AU baseline:

```bash
# 1. Preprocess OpenFace CSVs to NPY
python preprocess_openface_au.py \
  --csv_root /path/to/openface_csvs \
  --save_root /path/to/save_npy \
  --window_size 16 --stride 16

# 2. Run Training (LOOCV)
python train_syracuse_biovid_aux.py \
  --config config/syracuse_mil_coral_xformer_au.yaml \
  --loocv
```

## 5. Rationale for Choices

*   **Why Flattening vs. Pooling?**
    Action Units are low-level geometric/intensity features. Mean pooling them over 16 frames (~0.5s) would wash out transient micro-expressions (e.g., a quick wince). Concatenating them preserves the temporal evolution within the window, allowing the Transformer's projection layer to learn short-term temporal patterns.

*   **Why OpenFace?**
    OpenFace is the industry standard for open-source facial behavior analysis, providing a strong, reproducible "traditional CV" baseline.

*   **Why Include PSPI?**
    PSPI is a domain-specific engineered feature for pain. Including it explicitly ensures the baseline represents the "best possible" prior knowledge, making the comparison with latent deep features (Marlin) more meaningful.

