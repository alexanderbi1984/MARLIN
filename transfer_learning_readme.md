# Multi-Task Transfer Learning (Syracuse Pain + BioVid Stimulus + ShoulderPain)

This document describes how to set up and run the multi-task transfer learning experiment using the `evaluate_multitask.py` script. The goal is to train a model that learns from multiple datasets (Syracuse pain, BioVid stimulus, and optionally ShoulderPain), but evaluate its performance primarily on the Syracuse pain prediction task.

## Objective

Train a single model with a shared feature encoder and task-specific heads:
1.  **Pain Head:** Predicts ordinal pain levels using Syracuse dataset features and optionally ShoulderPain features.
2.  **Stimulus Head:** Predicts ordinal stimulus levels using BioVid dataset features.

The model is trained on a combined dataset including:
- Syracuse training split (original + augmented) derived from video ID splits
- The *entire* BioVid training split (defined by `train.txt`)
- The *entire* ShoulderPain training split (optional, defined by `train.txt`)

Evaluation (validation and testing) is performed *only* on the held-out Syracuse dataset splits (validation or test split derived from video IDs) to measure the model's ability to predict pain levels, potentially enhanced by the multi-task learning approach.

## Requirements

1.  **Python Environment:** Ensure you have a Python environment with the necessary libraries installed, including:
    *   `pytorch`
    *   `pytorch-lightning`
    *   `torchmetrics`
    *   `numpy`
    *   `pandas`
    *   `pyyaml`
    *   `tqdm`
    *   `scikit-learn` (for metrics/splitting, if applicable)
    *   (Potentially others required by `marlin-pytorch`, `ffmpeg-python` etc. - refer to project setup)

2.  **Data Setup:** You need two or three datasets prepared as follows:

    **a) Syracuse Dataset (Required):**

    *   **Root Directory (`--syracuse_data_path`):** The main directory containing Syracuse data.
    *   **Feature Directory (`syracuse_feature_dir` in YAML):** A subdirectory within the root directory containing pre-extracted features as `.npy` files (expected shape e.g., (4, 768) before temporal reduction).
        ```
        <syracuse_data_path>/
            <syracuse_feature_dir>/  # e.g., marlin_vit_small_patch16_224
                original_clip_1.npy
                augmented_clip_1a.npy
                ...
        ```
    *   **Metadata Base Directory (`--syracuse_marlin_base_dir`):** A directory containing the core Syracuse metadata file.
        *   `clips_json.json`: This JSON file is crucial. It should map clip filenames (like `original_clip_1.npy`) to metadata objects. Each object must contain:
            *   `video_id`: An identifier grouping original and augmented clips from the same source video. **Crucial for train/val/test splitting.**
            *   `video_type`: Either 'original' or 'aug'.
            *   `meta_info`: A dictionary containing the label information. For pain prediction, it needs a key like `pain_level` (for regression) or `class_N` (e.g., `class_5`) for multiclass classification, matching the `num_pain_classes` in the config. The `MultiTaskDataModule` specifically uses this `video_id` mapping to perform stratified train/validation/test splits, ensuring clips from the same video don't leak between sets for the primary Syracuse evaluation task.
            ```json
            // Example clips_json.json structure
            {
              "original_clip_1.npy": {
                "video_id": "video_001",
                "video_type": "original",
                "meta_info": {
                  "pain_level": "2.5", 
                  "class_5": "2" 
                }
              },
              "augmented_clip_1a.npy": {
                "video_id": "video_001",
                "video_type": "aug",
                 "meta_info": { ... } // Labels usually copied from original
              },
              ...
            }
            ```

    **b) BioVid Dataset (Required):**

    *   **Root Directory (`--biovid_data_path`):** The main directory containing BioVid data.
    *   **Feature Directory (`biovid_feature_dir` in YAML):** A subdirectory within the root directory containing pre-extracted features as `.npy` files.
        ```
        <biovid_data_path>/
            <biovid_feature_dir>/    # e.g., marlin_vit_small_patch16_224
                PartA_001_sx001t1.npy
                PartA_001_sx001t2.npy
                ...
            PartA_001/             # Original video structure (may not be needed if only using features)
                sx001t1.avi
                ...
            biovid_info.json       # Metadata file (see below)
            train.txt              # List of training clip filenames/paths relative to root
            val.txt                # List of validation clip filenames/paths (not used by multitask script)
            test.txt               # List of test clip filenames/paths (not used by multitask script)
        ```
    *   **Metadata File (`biovid_info.json`):** Located in the BioVid root directory. It maps *relative video paths* (like `PartA_001/sx001t1.avi`) to metadata. The relevant structure for stimulus labels is typically under `clips -> <video_path> -> attributes -> multiclass -> <num_stimulus_classes>`.
        ```json
        // Example biovid_info.json structure
        {
          "clips": {
            "PartA_001/sx001t1.avi": {
              "attributes": {
                "multiclass": {
                  "5": "3" // Stimulus level for 5 classes
                },
                "binary": "1"
              }
            },
            ...
          }
        }
        ```
        *Note: The `.npy` filenames in the feature directory must correspond to the video filenames listed in `train.txt` (e.g., `PartA_001_sx001t1.npy` corresponds to `PartA_001/sx001t1.avi` in `train.txt` and `biovid_info.json`).*
    *   **Split File (`train.txt`):** A text file in the BioVid root directory listing the relative paths of the video clips used for training (one per line, e.g., `PartA_001/sx001t1.avi`). The `MultiTaskDataModule` uses this to identify which BioVid features to load for the combined training dataset. **Validation (`val.txt`) and test (`test.txt`) splits for BioVid are ignored by this script.**

    **c) ShoulderPain Dataset (Optional):**

    *   **Root Directory (`--shoulder_pain_data_path`):** The main directory containing ShoulderPain data. Set the command-line argument to enable.
    *   **Feature Directory (`shoulder_pain_feature_dir` in YAML):** A subdirectory within the root directory containing pre-extracted features as `.npy` files. **Must be set in the YAML config (not `null`) to enable.**
        ```
        <shoulder_pain_data_path>/
            <shoulder_pain_feature_dir>/  # e.g., marlin_vit_small_patch16_224
                ll042t1aaaff.npy
                ll042t1aaunaff.npy
                ...
            shoulder_pain_info.json  # Metadata file (see below)
            train.txt                # List of training clip filenames
        ```
    *   **Metadata File (`shoulder_pain_info.json`):** Located in the ShoulderPain root directory. Similar structure to BioVid's metadata, but with VAS scores for pain (values 0-10).
        ```json
        // Example shoulder_pain_info.json structure
        {
          "clips": {
            "ll042t1aaaff.mp4": {
              "attributes": {
                "binary": 1,
                "multiclass": {
                  "6": 3,
                  "5": 2.0
                },
                "subject_id": "042-ll042",
                "sex": "",
                "age": "",
                "vas": "3.0",     // VAS score (0-10) used for pain classification
                "opr": "3.0",
                "source": "ShoulderPain"
              }
            },
            ...
          }
        }
        ```
        *Note: The VAS score is automatically binned into classes (matching `num_pain_classes`) by the `ShoulderPainLP` dataset class as follows:*
        - Class 0: Pain level 0.0 - 1.0
        - Class 1: Pain level 2.0 - 3.0
        - Class 2: Pain level 4.0 - 5.0
        - Class 3: Pain level 6.0 - 7.0
        - Class 4: Pain level 8.0 - 10.0
        *(This assumes `num_pain_classes` is 5. Adjust binning logic in `ShoulderPainLP` if needed for different class counts).*
    
    *   **Split File (`train.txt`):** A text file listing the relative paths of the video clips used for training (one per line, e.g., `ll042t1aaaff.mp4`). The `MultiTaskDataModule` uses this to load all ShoulderPain features for the combined training dataset.
    
    *Note: ShoulderPain data is only used for training. It is not used for validation or testing in this setup.*

## Configuration File (`--config`)

The experiment is configured using a YAML file. Create a copy of the template below and modify it for your run.

**Template (`configs/multitask_coral_template.yaml`):**

```yaml
# configs/multitask_coral_template.yaml

# --- Required --- 
model_name: multitask_pain_stimulus_run_01  # Name for checkpoints and logs

# Defines the number of pain classes and their boundaries based on raw scores (Syracuse: pain_level, ShoulderPain: vas)
# Example: 5 classes (0: <=1, 1: >1 & <=3, 2: >3 & <=5, 3: >5 & <=7, 4: >7)
pain_class_cutoffs: [1.0, 3.0, 5.0, 7.0] 

num_stimulus_classes: 5  # Number of ordinal stimulus levels (BioVid)

# Feature directory names (relative to the respective data paths provided via CLI)
syracuse_feature_dir: marlin_vit_small_patch16_224 # Example: Feature directory for Syracuse
biovid_feature_dir: marlin_vit_small_patch16_224   # Example: Feature directory for BioVid (can be the same or different)
shoulder_pain_feature_dir: null  # Optional: feature directory for ShoulderPain dataset. Set to a non-null string (e.g., "marlin_vit_small_patch16_224") AND provide --shoulder_pain_data_path via CLI to enable.

temporal_reduction: mean  # How to aggregate features temporally (mean, max, min, none)

learning_rate: 1e-4      # Learning rate for the optimizer

# --- Optional: Model & Loss Weights --- 
weight_decay: 0.0        # Weight decay for optimizer
label_smoothing: 0.0     # Label smoothing for loss function (0.0-1.0)
pain_loss_weight: 1.0    # Weight for the pain task loss in the combined loss
stim_loss_weight: 1.0    # Weight for the stimulus task loss in the combined loss

# --- Optional: Advanced CORAL Loss Options ---
use_distance_penalty: false  # Whether to penalize predictions farther from true label more
focal_gamma: null            # Focal loss gamma parameter (null = disabled)

# --- Optional: Training Options ---
patience: 200               # Early stopping patience (epochs of no improvement)
monitor_metric: "val_pain_QWK" # Metric to monitor for checkpoints/early stopping ("val_pain_QWK" or "val_pain_MAE")
use_class_weights: false    # Whether to use inverse frequency class weights in the PAIN loss calculation (based on Syracuse training split distribution)
balance_pain_classes: false # Whether to use weighted random sampling during dataloading to balance PAIN classes (based on Syracuse training split distribution)

# --- Optional: Encoder Architecture ---
# If null or empty, a single Linear layer with ReLU and dropout is used
encoder_hidden_dims: [512, 256] # Example: [hidden1_size, hidden2_size]

# --- Optional: Data Balancing (Training Set Construction) ---
balance_sources: false           # Deprecated/Not Implemented: Sample BioVid to match Syracuse training set size
balance_stimulus_classes: false  # Apply class balancing (via random sampling with replacement) within the BioVid training samples *before* combining datasets

# --- Optional: Stimulus Weight Scheduling ---
use_stim_weight_scheduler: false  # Whether to schedule stimulus loss weight dynamically during training
initial_stim_weight: 5.0          # Initial weight for stimulus loss (higher focuses on stimulus task early)
final_stim_weight: 1.0            # Final weight for stimulus loss after decay
stim_weight_decay_epochs: 50      # Number of epochs over which to decay the weight (default: epochs // 2)
stim_weight_sched_type: "cosine"  # Scheduling type ("cosine" or "linear")
```

**Parameter Explanations:**

*   `model_name`: Used for naming checkpoint directories and log files.
*   `pain_class_cutoffs: List[float]` (**Required**): List of upper boundaries for pain classes (e.g., `[1.0, 3.0, 5.0, 7.0]` for 5 classes). Used for both Syracuse (`pain_level`) and ShoulderPain (`vas`). `num_pain_classes` is derived from this list (`len(cutoffs) + 1`). **Ensure the list is sorted.**
*   `num_stimulus_classes: <int>` (**Required**): Number of ordinal classes for stimulus (BioVid).
*   `syracuse_feature_dir: <name>` (**Required**): Feature dir name for Syracuse (relative to `--syracuse_data_path`).
*   `biovid_feature_dir`: Name of the subdirectory under `--biovid_data_path` containing BioVid `.npy` features.
*   `shoulder_pain_feature_dir`: Name of the subdirectory under `--shoulder_pain_data_path` containing ShoulderPain `.npy` features. Set to a feature directory name (not `null`) to use ShoulderPain data. **Also requires setting `--shoulder_pain_data_path` via the command line.**
*   `temporal_reduction`: Method to aggregate frame-level features (shape e.g., `(4, 768)`) into a single vector (shape `(768,)`) before feeding to the model's linear layers. Options: `mean`, `max`, `min`. Use `none` if the model handles sequences directly (may require model changes).
*   `learning_rate`: Optimizer learning rate.
*   `weight_decay`: L2 regularization strength for the optimizer.
*   `label_smoothing`: Amount of label smoothing to apply (0.0-1.0) to the *pain task* loss during training.
*   `pain_loss_weight`, `stim_loss_weight`: Weights applied to the respective task losses before summing them for the total training loss. `stim_loss_weight` can be dynamically scheduled if `use_stim_weight_scheduler` is true.
*   `use_distance_penalty`: If `true`, the CORAL loss penalizes predictions farther from the true label more heavily.
*   `focal_gamma`: If set to a float (e.g., 2.0), applies focal weighting to the CORAL loss, focusing more on hard-to-classify examples. Set to `null` to disable.
*   `patience`: Number of epochs with no improvement on the `monitor_metric` after which training will stop early.
*   `monitor_metric`: The metric used for saving the best checkpoint and triggering early stopping. Typically `val_pain_QWK` (maximize) or `val_pain_MAE` (minimize). **Ensure the metric name matches the logged name exactly (case-sensitive).**
*   `use_class_weights`: If `true`, calculates inverse frequency weights based on the *Syracuse training split's pain class distribution* and applies them within the CORAL loss calculation for the pain task.
*   `balance_pain_classes`: If `true`, uses `WeightedRandomSampler` during training dataloading. Samples are weighted based on the inverse frequency of their *pain class* (derived from the Syracuse training split), aiming for a balanced distribution of pain classes per epoch. Samples from BioVid (and potentially ShoulderPain if not weighted by class) receive an average weight to maintain overall dataset proportions.
*   `encoder_hidden_dims`: List of integers defining the hidden layer sizes for the shared MLP encoder. Each linear layer is followed by ReLU and Dropout. If `null` or empty, a simpler encoder (Linear -> ReLU -> Dropout) is used.
*   `balance_sources`: **Deprecated/Not Implemented.** This option is present in the code but does not have an effect in the current `MultiTaskDataModule`.
*   `balance_stimulus_classes`: If `true`, the BioVid samples are first balanced by stimulus class (using random sampling with replacement to reach a target count per class) *before* being wrapped and concatenated into the final training set.
*   `use_stim_weight_scheduler`: Enables dynamic scheduling of the stimulus task weight during training via a callback.
*   `initial_stim_weight`: Starting weight for the stimulus loss (used if scheduler is enabled).
*   `final_stim_weight`: Final weight for the stimulus loss after decay (used if scheduler is enabled).
*   `stim_weight_decay_epochs`: Number of epochs over which to decay the stimulus weight (used if scheduler is enabled).
*   `stim_weight_sched_type`: Type of scheduling ("cosine" or "linear") to use for the stimulus weight decay (used if scheduler is enabled).

## Running the Experiment

Use the `evaluate_multitask.py` script. Provide paths to the datasets and the configuration file.

### Standard Training and Evaluation

```bash
python evaluate_multitask.py \
    --config configs/your_multitask_config.yaml \
    --syracuse_data_path /path/to/syracuse_features_root \
    --syracuse_marlin_base_dir /path/to/syracuse_metadata \
    --biovid_data_path /path/to/biovid_features_root \
    --shoulder_pain_data_path /path/to/shoulder_pain_features_root \  # Optional: Must be provided if shoulder_pain_feature_dir is set in config
    --n_gpus 1            `# Number of GPUs (0 for CPU)` \
    --batch_size 64       `# Training batch size` \
    --epochs 50           `# Maximum training epochs` \
    --num_workers 4       `# Dataloader workers` \
    --precision 32        `# Training precision (32, 16, bf16)` \
    # --predict_only        # Add this flag to skip training and only test/predict
    # --output_path predictions.csv # Add to save Syracuse test set predictions
```

### Cross-Validation Mode

The script also supports K-fold cross-validation for more robust evaluation:

```bash
python evaluate_multitask.py \
    --config configs/your_multitask_config.yaml \
    --syracuse_data_path /path/to/syracuse_features_root \
    --syracuse_marlin_base_dir /path/to/syracuse_metadata \
    --biovid_data_path /path/to/biovid_features_root \
    --shoulder_pain_data_path /path/to/shoulder_pain_features_root \  # Optional: Must be provided if shoulder_pain_feature_dir is set in config
    --cv_folds 5          `# Number of cross-validation folds (must be > 1)` \
    --n_gpus 1 \
    --batch_size 64 \
    --epochs 50 \
    --num_workers 4 \
    --precision 32
```

In cross-validation mode:
- The script performs stratified K-fold splitting based on unique Syracuse `video_id`s using the labels defined in `clips_json.json`.
- Each fold trains a *new* model on a combined dataset consisting of:
    - (K-1) folds of Syracuse data (original + augmented)
    - The *entire* BioVid training dataset
    - The *entire* ShoulderPain training dataset (if enabled)
- Each fold validates *only* on the held-out Syracuse fold (original clips).
- Checkpoints are saved based on the validation performance of the Syracuse fold (e.g., best `val_pain_QWK`).
- After all folds are trained, the script evaluates each fold's best checkpoint on its corresponding Syracuse validation set.
- Results (MAE, QWK, Acc, CM for the pain task) are aggregated across folds and reported as mean ± standard deviation.
- The best model checkpoint for each fold is saved separately in `ckpt/<model_name>/fold_<fold_idx>/`.
- An aggregated summary is saved to `results/<model_name>/<model_name>_cv_summary.json`.

## Output

*   **Checkpoints:** Saved in `ckpt/<model_name>_multitask/` (standard run) or `ckpt/<model_name>/fold_<fold_idx>/` (CV run). Includes the best checkpoint based on `monitor_metric` (e.g., `val_pain_QWK`) and potentially the last epoch checkpoint (only in standard run if `save_last=True` in `ModelCheckpoint`).
*   **Logs:** TensorBoard logs saved in `logs/<model_name>/version_X` (standard run) or `logs/<model_name>/fold_<fold_idx>/version_X` (CV run).
*   **Test Results (Standard Run):** Metrics from evaluating the best checkpoint on the Syracuse test set are printed to the console and saved to `ckpt/<model_name>_multitask/<model_name>_test_results_logged.json` (from `trainer.test`) and `<model_name>_test_results_manual.json` (manual calculation).
*   **Cross-Validation Results (CV Run):** Aggregate metrics (mean ± std dev for MAE, QWK, etc. across validation folds) are printed and saved to `results/<model_name>/<model_name>_cv_summary.json`. Individual fold metrics are included in this summary file.
*   **Predictions (Standard Run, Optional):** If `--output_path` is provided, predicted pain levels for the Syracuse test set are saved to the specified CSV file.

## Performance Metrics

The model reports several metrics for evaluating performance:

*   **MAE (Mean Absolute Error):** Average absolute difference between predicted and true pain/stimulus levels.
*   **QWK (Quadratic Weighted Kappa):** Agreement between predicted and true labels, accounting for the ordinal relationship between classes. Higher values (closer to 1.0) are better. This is typically the primary metric (`monitor_metric`) for model selection (checkpointing/early stopping).
*   **Accuracy:** Proportion of correctly predicted classes (less informative for ordinal tasks).
*   **Confusion Matrix:** Detailed breakdown of prediction outcomes (true label vs. predicted label).

QWK is the primary metric used for model selection during training, as it better accounts for the ordinal relationship between classes than simple accuracy.

## Technical Implementation Details

### Data Loading Architecture

The multi-task learning system uses a sophisticated data loading pipeline (`dataset/multitask.py`) to handle multiple datasets simultaneously:

#### 1. Dataset Classes

- **`SyracuseLP`**: Loads Syracuse pain level features and labels. Takes an explicit list of filenames (`name_list`) and the full metadata dictionary, enabling precise control for train/val/test/CV splits based on `video_id`. **Crucially, it now accepts `pain_class_cutoffs` and derives the integer class label by comparing the `pain_level` value (must exist and be numeric) from the metadata against these cutoffs using the `map_score_to_class` helper.** Includes validation to filter out clips missing features, metadata, or a valid `pain_level`.
- **`BioVidLP`**: Loads BioVid stimulus level features and labels based on the `train.txt` split file. Options `data_ratio` and `take_num` can subset this dataset.
- **`ShoulderPainLP`**: Loads ShoulderPain data based on its `train.txt` file. **It now accepts `pain_class_cutoffs` and derives the integer pain class label by comparing the `vas` score (must exist and be numeric, 0-10) from `shoulder_pain_info.json` against these cutoffs using the `map_score_to_class` helper.** Includes validation to filter out clips missing features, metadata, or a valid `vas` score.
- **`MultiTaskWrapper`**: A crucial component that wraps each base dataset (`SyracuseLP`, `BioVidLP`, `ShoulderPainLP`). It standardizes the output format to `(features, pain_label, stimulus_label)`. For a sample from a 'pain' dataset (Syracuse/ShoulderPain), `stimulus_label` is set to `-1`. For a sample from a 'stimulus' dataset (BioVid), `pain_label` is set to `-1`. This allows the model's training step to handle data from different sources within the same batch.

#### 2. `MultiTaskDataModule`

This PyTorch Lightning DataModule orchestrates the loading and combining of all datasets:

- **`__init__()`:** Accepts `pain_class_cutoffs` and derives `num_pain_classes = len(pain_class_cutoffs) + 1`. Stores configuration parameters.
- **`setup()` method:**
    - Loads Syracuse metadata using `SyracuseDataModule` logic. **This helper module now also accepts `pain_class_cutoffs` and generates `video_id_labels` (used for stratification) by deriving a class label from the average `pain_level` of each video's original clips.** Videos without valid original pain levels are excluded from the stratification map.
    - **Standard Mode:** Performs a stratified train/val/test split on Syracuse `video_id`s based on ratios (`syracuse_val_ratio`, `syracuse_test_ratio`) using the derived `video_id_labels`.
    - **CV Mode (Implicit via `evaluate_multitask.py`):** The module's `setup` is adapted within the CV loop. `evaluate_multitask.py` performs the K-fold split on Syracuse `video_id`s, and then constructs `SyracuseLP` datasets for the specific train/validation filenames of that fold, passing the `pain_class_cutoffs`.
    - **Dataset Construction:**
        - `train_dataset`: A `ConcatDataset` combining:
            - Wrapped Syracuse training samples (originals + augmentations from train video IDs), using `pain_class_cutoffs`.
            - Wrapped BioVid training samples (from `train.txt`, potentially balanced by class if `balance_stimulus_classes` is true).
            - Wrapped ShoulderPain training samples (from `train.txt`, if enabled), using `pain_class_cutoffs`.
        - `val_dataset`: Wrapped Syracuse validation samples (originals only, from validation video IDs), using `pain_class_cutoffs`.
        - `test_dataset`: Wrapped Syracuse test samples (originals only, from test video IDs), using `pain_class_cutoffs`.
- **Dataloaders:** Creates standard `DataLoader` instances for train, val, and test. The training dataloader can optionally use `WeightedRandomSampler` if `balance_pain_classes` is enabled.

```python
# Simplified DataModule structure within setup()
class MultiTaskDataModule(pl.LightningDataModule):
    def setup(self, stage):
        # --- 1. Load Syracuse Metadata & Perform Video ID Splits ---
        syracuse_dm_helper = SyracuseDataModule(..., pain_class_cutoffs=self.pain_class_cutoffs, ...)
        syracuse_dm_helper.setup()
        all_syracuse_metadata = syracuse_dm_helper.all_metadata
        video_id_labels = syracuse_dm_helper.video_id_labels # video_id -> derived class for stratification
        # ... (Get train/val/test filenames based on video_id splits) ...

        # --- 2. Create Base Datasets ---
        # Note: SyracuseLP and ShoulderPainLP now take pain_class_cutoffs
        syracuse_train_set = SyracuseLP(..., pain_class_cutoffs=self.pain_class_cutoffs, name_list=syracuse_train_filenames, metadata=all_syracuse_metadata, ...)
        syracuse_val_set = SyracuseLP(..., pain_class_cutoffs=self.pain_class_cutoffs, name_list=syracuse_val_filenames, metadata=all_syracuse_metadata, ...)
        syracuse_test_set = SyracuseLP(..., pain_class_cutoffs=self.pain_class_cutoffs, name_list=syracuse_test_filenames, metadata=all_syracuse_metadata, ...)

        biovid_train_set = BioVidLP(..., split='train', ...) # Loads based on train.txt
        # Optionally balance biovid_train_set by class here if balance_stimulus_classes=True

        shoulder_pain_train_set = ShoulderPainLP(..., pain_class_cutoffs=self.pain_class_cutoffs, ...) if self.use_shoulder_pain else None

        # --- 3. Wrap Datasets ---
        wrapped_syracuse_train = MultiTaskWrapper(syracuse_train_set, 'pain')
        self.val_dataset = MultiTaskWrapper(syracuse_val_set, 'pain') # Val uses only Syracuse
        self.test_dataset = MultiTaskWrapper(syracuse_test_set, 'pain') # Test uses only Syracuse

        wrapped_biovid_train = MultiTaskWrapper(biovid_train_set, 'stimulus')
        wrapped_shoulder_pain_train = MultiTaskWrapper(shoulder_pain_train_set, 'pain') if self.use_shoulder_pain else None

        # --- 4. Combine Training Datasets ---
        train_datasets = [wrapped_syracuse_train, wrapped_biovid_train]
        if self.use_shoulder_pain and wrapped_shoulder_pain_train is not None:
            train_datasets.append(wrapped_shoulder_pain_train)
        self.train_dataset = ConcatDataset(train_datasets)
```

#### 3. Class Balancing Mechanisms

The system implements two complementary approaches to handle class imbalance, primarily focused on the **pain task** using the **Syracuse training split** distribution:

**a) Class Weights for Loss Function** (`use_class_weights: true` in config):
- Calculates inverse frequency weights based on the pain class distribution in the Syracuse training set for the current run/fold.
- Passes these weights (`class_weights`) to the `MultiTaskCoralClassifier`.
- The `coral_loss` function uses these weights (`importance_weights`) to scale the loss contribution of each sample belonging to the pain task, giving higher weight to samples from less frequent classes.

```python
# In evaluate_multitask.py (simplified)
if use_class_weights:
    # Calculate weights based on Syracuse train labels
    class_counts = np.bincount(syracuse_train_labels, ...)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
# In MultiTaskCoralClassifier._calculate_loss_and_metrics (simplified)
if valid_pain_mask.any():
    importance_weights = None
    if self.class_weights is not None and stage == 'train':
         importance_weights = self.class_weights[valid_pain_labels] # Get weight for each sample's label
    pain_loss = self.coral_loss(..., importance_weights=importance_weights, ...)
```

**b) Weighted Random Sampling** (`balance_pain_classes: true` in config):
- Calculates sampling weights for each sample in the *combined* training dataset.
- **Syracuse samples:** Weights are based on the inverse frequency of their pain class (derived from the Syracuse training split distribution).
- **BioVid samples:** Receive a uniform average weight (mean of Syracuse weights) to maintain their overall proportion relative to Syracuse.
- **ShoulderPain samples (if enabled):** Receive weights based on their *pain class* (using the same distribution logic as Syracuse).
- Creates a `torch.utils.data.WeightedRandomSampler` using these combined weights.
- The training `DataLoader` uses this sampler, oversampling pain classes that were less frequent in the original Syracuse training split.

```python
# In evaluate_multitask.py, CV loop (simplified)
if balance_pain_classes:
    # Calculate sample_class_weights based on Syracuse train labels for the fold
    # Assign weights to Syracuse samples based on their class
    syracuse_weights = sample_class_weights[syracuse_labels]
    # Assign average weight to BioVid samples
    biovid_weights = np.ones(len(wrapped_biovid_train)) * np.mean(syracuse_weights)
    # Assign weights to ShoulderPain samples based on their class (if enabled)
    shoulder_pain_weights = sample_class_weights[shoulder_pain_labels] # if enabled
    # Concatenate weights
    combined_weights = np.concatenate([syracuse_weights, biovid_weights, shoulder_pain_weights])
    # Create sampler
    sampler = WeightedRandomSampler(weights=combined_weights, ...)
    # Create DataLoader with sampler
    fold_train_loader = DataLoader(..., sampler=sampler, shuffle=False) # Shuffle is False when sampler is used
else:
    # Standard DataLoader
    fold_train_loader = DataLoader(..., shuffle=True)
```

#### 4. Cross-Validation Implementation (`evaluate_multitask.py`)

Cross-validation (`--cv_folds > 1`) focuses on robust evaluation of the pain task on Syracuse data:

- **Splitting:** Uses `sklearn.model_selection.StratifiedKFold` on unique Syracuse `video_id`s, stratified by pain label.
- **Fold Data Construction:**
    - **Train:** `ConcatDataset` of (Syracuse K-1 folds + *all* BioVid train + *all* ShoulderPain train), appropriately wrapped.
    - **Validation:** Wrapped Syracuse data from the 1 held-out fold.
- **Training Loop:** A separate `Trainer` and `MultiTaskCoralClassifier` instance is created and trained for each fold.
- **Evaluation:** After all folds train, each fold's *best* checkpoint (based on its Syracuse validation performance) is loaded and evaluated *on its corresponding Syracuse validation set*. Metrics like MAE, QWK, Acc, CM are calculated for the pain task.
- **Aggregation:** Metrics are averaged across all folds, and standard deviations are reported. Results are saved to a JSON file.

```python
# In evaluate_multitask.py (simplified CV logic)
skf = StratifiedKFold(n_splits=n_splits, ...)
for fold_idx, (train_vid_indices, val_vid_indices) in enumerate(skf.split(unique_video_ids, video_labels)):
    # Get fold_train_video_ids, fold_val_video_ids
    # Create syracuse_train_filenames, syracuse_val_filenames for the fold
    
    # Create Syracuse datasets for fold
    # Note: SyracuseLP takes pain_class_cutoffs
    syracuse_train_set = SyracuseLP(..., pain_class_cutoffs=pain_class_cutoffs, name_list=syracuse_train_filenames, ...)
    syracuse_val_set = SyracuseLP(..., pain_class_cutoffs=pain_class_cutoffs, name_list=syracuse_val_filenames, ...)
    
    # Wrap datasets
    wrapped_syracuse_train = MultiTaskWrapper(syracuse_train_set, 'pain')
    wrapped_syracuse_val = MultiTaskWrapper(syracuse_val_set, 'pain')
    wrapped_biovid_train = MultiTaskWrapper(full_biovid_train_set, 'stimulus') # Use full BioVid train
    wrapped_shoulder_pain_train = MultiTaskWrapper(full_shoulder_pain_train_set, 'pain') if use_shoulder_pain else None # Use full ShoulderPain train if enabled

    # Combine Training data for the fold
    combined_train_dataset = ConcatDataset([wrapped_syracuse_train, wrapped_biovid_train, ...])
    
    # Create DataLoaders (train uses combined_train_dataset, val uses wrapped_syracuse_val)
    fold_train_loader = DataLoader(combined_train_dataset, ...) # potentially with sampler
    fold_val_loader = DataLoader(wrapped_syracuse_val, ...)
    
    # Initialize Model and Trainer for the fold
    model = MultiTaskCoralClassifier(...)
    trainer = Trainer(...)
    
    # Train the fold
    trainer.fit(model, train_dataloaders=fold_train_loader, val_dataloaders=fold_val_loader)
    
    # Store best checkpoint path
    fold_checkpoint_paths.append(checkpoint_callback.best_model_path)
    fold_val_filenames.append(syracuse_val_filenames) # Store val filenames for evaluation

# After loop: Evaluate each checkpoint on its val_filenames
all_fold_metrics = []
for ckpt_path, val_files in zip(fold_checkpoint_paths, fold_val_filenames):
    metrics = _evaluate_multitask_fold_checkpoint(ckpt_path, val_files, ...)
    all_fold_metrics.append(metrics)

# Aggregate metrics from all_fold_metrics
# ...
```

### Classifier Design (`model/multi_task_coral.py`)

The `MultiTaskCoralClassifier` implements a shared-encoder, dual-head architecture using the CORAL (Consistent Rank Logits) method for ordinal regression.

#### 1. Architecture Overview

```
Input Features (e.g., shape [BatchSize, FeatureDim])
      │
      ▼
┌───────────────────┐
│ Shared Encoder    │
│ (MLP or Linear    │
│  + ReLU + Dropout)│
└───────────────────┘
      │ (e.g., shape [BatchSize, EncoderOutputDim])
      ├─────────────────────────────────────────┐
      │                                         │
      ▼                                         ▼
┌─────────────┐                         ┌─────────────────┐
│ Pain Head   │                         │ Stimulus Head   │
│ (Linear)    │                         │ (Linear)        │
└─────────────┘                         └─────────────────┘
      │ (shape [BS, NumPainClasses-1])        │ (shape [BS, NumStimClasses-1])
      ▼                                         ▼
 Pain Logits                            Stimulus Logits
```

#### 2. Shared Encoder

- Configurable via `encoder_hidden_dims` in the YAML config.
- If `encoder_hidden_dims` is `null` or empty:
    - A simple sequence: `Linear(input_dim, input_dim) -> ReLU -> Dropout(0.5)`
- If `encoder_hidden_dims` is a list (e.g., `[512, 256]`):
    - An MLP is constructed: `Linear(input_dim, 512) -> ReLU -> Dropout(0.5) -> Linear(512, 256) -> ReLU -> Dropout(0.5)`
- The output dimension (`encoder_output_dim`) depends on the last layer of the encoder.

```python
# In MultiTaskCoralClassifier.__init__
if encoder_hidden_dims is None or len(encoder_hidden_dims) == 0:
    self.shared_encoder = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.Dropout(0.5) # Dropout added here
    )
    encoder_output_dim = input_dim
else:
    layers = []
    current_dim = input_dim
    for h_dim in encoder_hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5)) # Dropout added after each layer
        current_dim = h_dim
    self.shared_encoder = nn.Sequential(*layers)
    encoder_output_dim = current_dim
```

#### 3. CORAL Heads

- Each task (pain, stimulus) gets a separate, simple linear layer applied to the shared encoder's output.
- Each head outputs `num_classes - 1` logits, corresponding to the thresholds used in CORAL.

```python
# In MultiTaskCoralClassifier.__init__
self.pain_head = nn.Linear(encoder_output_dim, num_pain_classes - 1)
self.stimulus_head = nn.Linear(encoder_output_dim, num_stimulus_classes - 1)
```

#### 4. CORAL Loss Function (`coral_loss` static method)

The CORAL method models ordinal regression as a series of binary classifications P(y > k) for k = 0 to num_classes-2. The loss is calculated independently for samples with valid pain labels and valid stimulus labels.

- **Input:** `logits` (shape `[BatchSize, NumClasses-1]`), `levels` (ordinal labels, shape `[BatchSize]`).
- **Process:**
    1.  **Convert Labels to Binary Targets:** Creates a target tensor `levels_binary` (shape `[BatchSize, NumClasses-1]`) where `levels_binary[i, k] = 1` if `levels[i] > k`, and 0 otherwise.
    2.  **Label Smoothing (Optional):** If `label_smoothing > 0.0` (applied only to pain task during training), targets are adjusted: `target * (1 - smoothing) + smoothing / 2`.
    3.  **Base Loss:** Calculates binary cross-entropy loss for each threshold using `F.logsigmoid`.
    4.  **Distance Penalty (Optional):** If `use_distance_penalty=True`, multiplies the loss for each threshold `k` by `(1 + abs(true_label - k))`, penalizing errors on thresholds further from the true label more heavily.
    5.  **Focal Weighting (Optional):** If `focal_gamma` is set, multiplies the loss by a focal weight `(1-p)^gamma` for correctly classified thresholds and `p^gamma` for incorrectly classified ones, focusing on harder examples.
    6.  **Sum & Importance Weights:** Sums the loss across thresholds for each sample. If `importance_weights` (derived from `class_weights` for the pain task) are provided, multiplies each sample's loss by its corresponding weight.
    7.  **Reduction:** Averages the loss across the batch (`reduction='mean'`).
- **Handling Missing Labels:** The `_calculate_loss_and_metrics` method ensures `coral_loss` is only called on batches with valid labels (`!= -1`) for each task.

```python
# In MultiTaskCoralClassifier._calculate_loss_and_metrics (simplified)
# --- Pain Task ---
valid_pain_mask = pain_labels != -1
if valid_pain_mask.any():
    valid_pain_logits = pain_logits[valid_pain_mask]
    valid_pain_labels = pain_labels[valid_pain_mask]
    importance_weights = self.class_weights[valid_pain_labels] if use_class_weights else None
    smoothing = self.hparams.label_smoothing if stage == 'train' else 0.0
    pain_loss = self.coral_loss(
        valid_pain_logits, valid_pain_labels, 
        importance_weights=importance_weights, label_smoothing=smoothing,
        distance_penalty=self.hparams.use_distance_penalty, focal_gamma=self.hparams.focal_gamma
    )
    total_loss += self.hparams.pain_loss_weight * pain_loss
    # Update metrics...

# --- Stimulus Task --- (Similar logic, typically no label smoothing or class weights)
valid_stim_mask = stimulus_labels != -1
if valid_stim_mask.any():
    # ... call self.coral_loss without smoothing/weights ...
    total_loss += self.hparams.stim_loss_weight * stim_loss
    # Update metrics...
```

#### 5. Prediction Process (`prob_to_label` static method)

Prediction converts the sigmoid probabilities of the CORAL logits into a final class label.

- **Input:** `probs` (shape `[BatchSize, NumClasses-1]`), where `probs[i, k] = sigmoid(logits[i, k]) ≈ P(y > k)`.
- **Process:** Sums the number of thresholds `k` for which `P(y > k) > 0.5`. This count directly corresponds to the predicted ordinal label (0 to `NumClasses-1`).

```python
# In MultiTaskCoralClassifier.prob_to_label (static method)
@staticmethod
def prob_to_label(probs):
    # probs[b, k] = P(y > k)
    # Label = sum_{k=0}^{num_classes-2} I(P(y > k) > 0.5)
    return torch.sum(probs > 0.5, dim=1) # Result is integer label 0, ..., num_classes-1
```

#### 6. Stimulus Weight Scheduling (`StimWeightSchedulerCallback` in `evaluate_multitask.py`)

If `use_stim_weight_scheduler: true` in the config, this callback dynamically adjusts the `stim_loss_weight` hyperparameter of the `MultiTaskCoralClassifier` during training.

- **Mechanism:** At the start of each training epoch, it calculates a new weight based on the current epoch, `initial_stim_weight`, `final_stim_weight`, `stim_weight_decay_epochs`, and `stim_weight_sched_type` ('cosine' or 'linear').
- **Update:** It updates `pl_module.hparams.stim_loss_weight`. The loss calculation in `_calculate_loss_and_metrics` uses this hyperparameter directly.
- **Purpose:** Allows the model to initially focus more on the auxiliary stimulus task (using a higher weight) and gradually shift focus to the primary pain task (as the weight decays towards its final value).

```python
# In StimWeightSchedulerCallback.on_train_epoch_start (simplified)
class StimWeightSchedulerCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Calculate new_weight based on epoch, decay_epochs, type, initial/final weights
        # ... (cosine or linear decay logic) ...
        
        # Update the hyperparameter used in the loss calculation
        pl_module.hparams.stim_loss_weight = new_weight 
        # Log the weight...
```

This approach helps the model leverage the stimulus task more heavily in early training stages, then gradually focus more on the pain task as training progresses. The implementation ensures the weight is properly updated in the hyperparameters, so it's correctly used in the loss calculation. 