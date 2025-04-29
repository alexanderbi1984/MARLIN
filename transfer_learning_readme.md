# Multi-Task Transfer Learning (Syracuse Pain + BioVid Stimulus + ShoulderPain)

This document describes how to set up and run the multi-task transfer learning experiment using the `evaluate_multitask.py` script. The goal is to train a model that learns from multiple datasets (Syracuse pain, BioVid stimulus, and optionally ShoulderPain), but evaluate its performance primarily on the Syracuse pain prediction task.

## Objective

Train a single model with a shared feature encoder and task-specific heads:
1.  **Pain Head:** Predicts ordinal pain levels using Syracuse dataset features and optionally ShoulderPain features.
2.  **Stimulus Head:** Predicts ordinal stimulus levels using BioVid dataset features.

The model is trained on a combined dataset including:
- Syracuse (original + augmented) 
- BioVid samples
- ShoulderPain samples (optional)

Evaluation (validation and testing) is performed *only* on the Syracuse dataset splits to measure the model's ability to predict pain levels, potentially enhanced by the multi-task learning approach.

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
            *   `video_id`: An identifier grouping original and augmented clips from the same source video.
            *   `video_type`: Either 'original' or 'aug'.
            *   `meta_info`: A dictionary containing the label information. For pain prediction, it needs a key like `pain_level` (for regression) or `class_N` (e.g., `class_5`) for multiclass classification, matching the `num_pain_classes` in the config.
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
    *   **Split File (`train.txt`):** A text file in the BioVid root directory listing the relative paths of the video clips used for training (one per line). The `MultiTaskDataModule` uses this to identify which BioVid features to load for the training portion.

    **c) ShoulderPain Dataset (Optional):**

    *   **Root Directory (`--shoulder_pain_data_path`):** The main directory containing ShoulderPain data.
    *   **Feature Directory (`shoulder_pain_feature_dir` in YAML):** A subdirectory within the root directory containing pre-extracted features as `.npy` files.
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
        *Note: The VAS score is automatically binned into classes as follows:*
        - Class 0: Pain level 0.0 - 1.0
        - Class 1: Pain level 2.0 - 3.0
        - Class 2: Pain level 4.0 - 5.0
        - Class 3: Pain level 6.0 - 7.0
        - Class 4: Pain level 8.0 - 10.0
    
    *   **Split File (`train.txt`):** A text file listing the relative paths of the video clips used for training (same format as BioVid).
    
    *Note: ShoulderPain data is only used for training. It is not used for validation or testing.*

## Configuration File (`--config`)

The experiment is configured using a YAML file. Create a copy of the template below and modify it for your run.

**Template (`configs/multitask_coral_template.yaml`):**

```yaml
# configs/multitask_coral_template.yaml

# --- Required --- 
model_name: multitask_pain_stimulus_run_01  # Name for checkpoints and logs

num_pain_classes: 5  # Number of ordinal pain levels (Syracuse)
num_stimulus_classes: 5  # Number of ordinal stimulus levels (BioVid)

# Feature directory names (relative to the respective data paths provided via CLI)
syracuse_feature_dir: marlin_vit_small_patch16_224 # Example: Feature directory for Syracuse
biovid_feature_dir: marlin_vit_small_patch16_224   # Example: Feature directory for BioVid (can be the same or different)
shoulder_pain_feature_dir: null  # Optional: feature directory for ShoulderPain dataset, set to null to disable

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
use_class_weights: false    # Whether to use inverse frequency class weights in loss
balance_pain_classes: false # Whether to use weighted sampling for Syracuse pain classes 

# --- Optional: Encoder Architecture ---
# If null or empty, a single Linear layer with dropout is used
encoder_hidden_dims: [512, 256] # Example: [hidden1_size, hidden2_size] 

# --- Optional: Data Balancing --- 
balance_sources: false           # Sample BioVid to match Syracuse training set size
balance_stimulus_classes: false  # Apply class balancing within BioVid training samples

# --- Optional: Stimulus Weight Scheduling ---
use_stim_weight_scheduler: false  # Whether to schedule stimulus loss weight
initial_stim_weight: 5.0          # Initial weight (higher focuses on stimulus task early)
final_stim_weight: 1.0            # Final weight after decay
stim_weight_decay_epochs: 50      # Number of epochs over which to decay the weight
stim_weight_sched_type: "cosine"  # Scheduling type ("cosine" or "linear")
```

**Parameter Explanations:**

*   `model_name`: Used for naming checkpoint directories and log files.
*   `num_pain_classes`: Number of ordinal classes for the Syracuse/ShoulderPain pain task. Must match the label structure in `clips_json.json` (`class_N`).
*   `num_stimulus_classes`: Number of ordinal classes for the BioVid stimulus task. Must match the label structure in `biovid_info.json` (`multiclass[N]`).
*   `syracuse_feature_dir`: Name of the subdirectory under `--syracuse_data_path` containing Syracuse `.npy` features.
*   `biovid_feature_dir`: Name of the subdirectory under `--biovid_data_path` containing BioVid `.npy` features.
*   `shoulder_pain_feature_dir`: Name of the subdirectory under `--shoulder_pain_data_path` containing ShoulderPain `.npy` features. Set to `null` to disable ShoulderPain data.
*   `temporal_reduction`: Method to aggregate frame-level features (shape e.g., `(4, 768)`) into a single vector (shape `(768,)`) before feeding to the model's linear layers. Options: `mean`, `max`, `min`. Use `none` if the model handles sequences directly (may require model changes).
*   `learning_rate`: Optimizer learning rate.
*   `weight_decay`: L2 regularization strength for the optimizer.
*   `label_smoothing`: Amount of label smoothing to apply (0.0-1.0), which can improve generalization.
*   `pain_loss_weight`, `stim_loss_weight`: Weights applied to the respective task losses before summing them for the total training loss.
*   `use_distance_penalty`: When enabled, mistakes on predictions farther from the true label are penalized more heavily.
*   `focal_gamma`: Parameter for focal loss weighting. When set, focuses more on hard-to-classify examples. Set to null to disable.
*   `patience`: Number of epochs with no improvement after which training will be stopped. Higher values allow for more exploration.
*   `use_class_weights`: When enabled, less frequent classes are weighted more heavily in the loss function.
*   `balance_pain_classes`: When enabled, uses weighted random sampling to balance class distributions.
*   `encoder_hidden_dims`: List of integers defining the hidden layer sizes for the shared MLP encoder. If `null` or empty, a simple network with a single Linear layer and dropout is used.
*   `balance_sources`: If `true`, the number of BioVid samples used in training will be randomly sampled down (or up via replacement if `balance_stimulus_classes` is also true and requires it) to match the number of Syracuse training samples.
*   `balance_stimulus_classes`: If `true`, the BioVid samples included in the training set will be selected to achieve a more balanced distribution across stimulus classes.
*   `use_stim_weight_scheduler`: Enables dynamic scheduling of the stimulus task weight during training.
*   `initial_stim_weight`: Starting weight for the stimulus loss (typically higher than final).
*   `final_stim_weight`: Final weight for the stimulus loss after decay.
*   `stim_weight_decay_epochs`: Number of epochs over which to decay the stimulus weight.
*   `stim_weight_sched_type`: Type of scheduling to use ("cosine" or "linear").

## Running the Experiment

Use the `evaluate_multitask.py` script. Provide paths to the datasets and the configuration file.

### Standard Training and Evaluation

```bash
python evaluate_multitask.py \
    --config configs/your_multitask_config.yaml \
    --syracuse_data_path /path/to/syracuse_features_root \
    --syracuse_marlin_base_dir /path/to/syracuse_metadata \
    --biovid_data_path /path/to/biovid_features_root \
    --shoulder_pain_data_path /path/to/shoulder_pain_features_root \  # Optional
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
    --shoulder_pain_data_path /path/to/shoulder_pain_features_root \  # Optional
    --cv_folds 5          `# Number of cross-validation folds` \
    --n_gpus 1 \
    --batch_size 64 \
    --epochs 50 \
    --num_workers 4 \
    --precision 32
```

In cross-validation mode:
- The script performs stratified K-fold splitting based on unique Syracuse video IDs
- Each fold trains on K-1 folds of Syracuse + all BioVid/ShoulderPain data, and validates on 1 fold from Syracuse
- Results are aggregated and reported as mean ± standard deviation
- The best model for each fold is saved separately

## Output

*   **Checkpoints:** Saved in `ckpt/<model_name>_multitask/`. Includes the best checkpoint based on validation QWK and the last epoch checkpoint.
*   **Logs:** TensorBoard logs saved in `lightning_logs/`.
*   **Test Results:** Metrics from evaluating the best checkpoint on the Syracuse test set are printed to the console and saved to `ckpt/<model_name>_multitask/<model_name>_test_results_logged.json` and `<model_name>_test_results_manual.json`.
*   **Predictions (Optional):** If `--output_path` is provided, predicted pain levels for the Syracuse test set are saved to the specified CSV file.
*   **Cross-Validation Results:** When using cross-validation, aggregate metrics are saved to `results/<model_name>/<model_name>_cv_summary.json`.

## Performance Metrics

The model reports several metrics for evaluating performance:

*   **MAE (Mean Absolute Error):** Average absolute difference between predicted and true pain/stimulus levels.
*   **QWK (Quadratic Weighted Kappa):** Agreement between predicted and true labels, accounting for the ordinal relationship between classes. Higher values (closer to 1.0) are better.
*   **Accuracy:** Proportion of correctly predicted classes.
*   **Confusion Matrix:** Detailed breakdown of prediction outcomes (true label vs. predicted label).

QWK is the primary metric used for model selection during training, as it better accounts for the ordinal relationship between classes than simple accuracy.

## Technical Implementation Details

### Data Loading Architecture

The multi-task learning system uses a sophisticated data loading pipeline to handle multiple datasets simultaneously:

#### 1. Dataset Classes

- **SyracuseLP**: Loads Syracuse pain level features and labels. Handles filtering by specific filenames for cross-validation.
- **BioVidLP**: Loads BioVid stimulus level features and labels from the training split.
- **ShoulderPainLP**: Loads ShoulderPain data and converts VAS scores (0-10) to discrete pain classes using predefined bins.
- **MultiTaskWrapper**: Wraps each dataset to provide a consistent interface for the multi-task model, returning features along with pain and stimulus labels (with -1 for missing task labels).

#### 2. MultiTaskDataModule

This PyTorch Lightning DataModule orchestrates the loading and combining of all datasets:

```python
# Simplified DataModule structure
class MultiTaskDataModule(pl.LightningDataModule):
    def setup(self, stage):
        # For training: combine Syracuse, BioVid and optionally ShoulderPain
        syracuse_train = SyracuseLP(...)    # Load Syracuse training set
        biovid_train = BioVidLP(...)        # Load BioVid training set
        shoulder_pain_train = ShoulderPainLP(...) if use_shoulder_pain else None
        
        # Wrap datasets to handle multi-task format
        wrapped_syracuse_train = MultiTaskWrapper(syracuse_train, 'pain')
        wrapped_biovid_train = MultiTaskWrapper(biovid_train, 'stimulus')
        wrapped_shoulder_pain_train = MultiTaskWrapper(shoulder_pain_train, 'pain') if use_shoulder_pain else None
        
        # Combine datasets for training
        train_datasets = [wrapped_syracuse_train, wrapped_biovid_train]
        if use_shoulder_pain:
            train_datasets.append(wrapped_shoulder_pain_train)
            
        self.train_dataset = ConcatDataset(train_datasets)
        
        # For validation and testing: use only Syracuse
        self.val_dataset = MultiTaskWrapper(SyracuseLP(...), 'pain')
        self.test_dataset = MultiTaskWrapper(SyracuseLP(...), 'pain')
```

#### 3. Class Balancing Mechanisms

The system implements two complementary approaches to handle class imbalance:

**a) Class Weights for Loss Function** (`use_class_weights: true`):
```python
# Calculate inverse frequency class weights
class_counts = np.bincount(train_labels, minlength=num_classes)
weights = 1.0 / (class_counts + 1e-6)
weights = weights / weights.mean()  # Normalize to maintain loss scale
```

**b) Weighted Random Sampling** (`balance_pain_classes: true`):
```python
# Extract labels and calculate class frequencies
class_counts = np.bincount(syracuse_labels, minlength=num_pain_classes)
sample_weights = 1.0 / (class_counts + 1e-6)

# Create weighted sampler
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights[labels]).float(),
    num_samples=len(dataset),
    replacement=True
)
```

#### 4. Cross-Validation Implementation

For cross-validation, the script performs video-level stratified splitting to ensure that all clips from the same video stay in the same fold:

```python
# Stratified K-Fold based on unique video IDs
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_vid_idx, val_vid_idx in skf.split(unique_video_ids, video_labels):
    # Map indices to video IDs
    fold_train_video_ids = [unique_video_ids[i] for i in train_vid_idx]
    fold_val_video_ids = [unique_video_ids[i] for i in val_vid_idx]
    
    # Filter clips based on video IDs
    syracuse_train_filenames = [clip['filename'] for clip in all_clips 
                               if clip['video_id'] in fold_train_video_ids]
    # ...
```

### Classifier Design

The `MultiTaskCoralClassifier` implements a shared-encoder, dual-head architecture using the CORAL (Consistent Rank Logits) method for ordinal regression.

#### 1. Architecture Overview

```
Input Features
      │
      ▼
┌─────────────┐
│Shared Encoder│
└─────────────┘
      │
      ├─────────────┬─────────────┐
      │             │             │
      ▼             ▼             ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│ Pain Head │ │Stimulus Head│ │  Other    │
│ (CORAL)   │ │  (CORAL)   │ │  Heads    │
└───────────┘ └───────────┘ └───────────┘
```

#### 2. Shared Encoder

A configurable MLP with dropout after each layer:

```python
if encoder_hidden_dims is None or len(encoder_hidden_dims) == 0:
    self.shared_encoder = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ReLU(),
        nn.Dropout(0.5)
    )
else:
    layers = []
    current_dim = input_dim
    for h_dim in encoder_hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))
        current_dim = h_dim
    self.shared_encoder = nn.Sequential(*layers)
```

#### 3. CORAL Heads

Each task gets a single linear layer that outputs `num_classes - 1` logits:

```python
self.pain_head = nn.Linear(encoder_output_dim, num_pain_classes - 1)
self.stimulus_head = nn.Linear(encoder_output_dim, num_stimulus_classes - 1)
```

#### 4. CORAL Loss Function

The CORAL method models ordinal regression as a series of binary classifications:

```python
def coral_loss(logits, levels, importance_weights=None, reduction='mean',
               label_smoothing=0.0, distance_penalty=False, focal_gamma=None):
    # Convert ordinal labels to binary targets
    # For a 5-class problem with label 3:
    # Binary targets would be [1, 1, 1, 0] (1 for classes up to the label)
    levels_binary = (levels.unsqueeze(1) > 
                     torch.arange(num_classes_minus1, device=device).unsqueeze(0)).float()
    
    # Apply label smoothing if enabled
    if label_smoothing > 0.0:
        levels_binary = levels_binary * (1 - label_smoothing) + label_smoothing / 2
    
    # Core binary cross-entropy for each threshold
    log_sigmoid = F.logsigmoid(logits)
    base_loss_tasks = log_sigmoid * levels_binary + (log_sigmoid - logits) * (1 - levels_binary)
    
    # Apply distance penalty (optional)
    if distance_penalty:
        distance_matrix = torch.abs(levels.unsqueeze(1) - 
                                    torch.arange(num_classes_minus1, device=device).unsqueeze(0))
        base_loss_tasks = base_loss_tasks * (1.0 + distance_matrix)
    
    # Apply focal weighting (optional)
    if focal_gamma is not None:
        probs = torch.sigmoid(logits)
        focal_weight = torch.where(
            levels_binary > 0.5,
            (1 - probs) ** focal_gamma,
            probs ** focal_gamma
        )
        base_loss_tasks = focal_weight * base_loss_tasks
    
    # Sum across thresholds, apply importance weights if provided
    loss_per_sample = -torch.sum(base_loss_tasks, dim=1)
    if importance_weights is not None:
        loss_per_sample *= importance_weights
    
    # Apply reduction (mean/sum)
    return loss_per_sample.mean() if reduction == 'mean' else loss_per_sample.sum()
```

#### 5. Prediction Process

Prediction converts sigmoid probabilities to class labels:

```python
def prob_to_label(probs):
    # Count how many thresholds predict positive
    # For a 5-class problem, we have 4 thresholds
    # If probabilities are [0.9, 0.8, 0.4, 0.1]
    # The predicted class would be 2 (as 2 thresholds > 0.5)
    return torch.sum(probs > 0.5, dim=1)
```

#### 6. Stimulus Weight Scheduling

The model can dynamically adjust the weight of the stimulus task during training, starting high and decaying over time:

```python
class StimWeightSchedulerCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch >= self.decay_epochs:
            new_weight = self.final_weight
        else:
            progress = current_epoch / self.decay_epochs
            if self.scheduler_type == "cosine":
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                new_weight = self.final_weight + (self.initial_weight - self.final_weight) * cosine_decay
            else:
                new_weight = self.initial_weight - (self.initial_weight - self.final_weight) * progress
        
        # Update both the module attribute and the hyperparameters to ensure consistent usage
        pl_module.stim_loss_weight = new_weight
        pl_module.hparams.stim_loss_weight = new_weight
```

This approach helps the model leverage the stimulus task more heavily in early training stages, then gradually focus more on the pain task as training progresses. The implementation ensures the weight is properly updated in both the module attribute and the hyperparameters, so it's correctly used in the loss calculation. 