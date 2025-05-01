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

## Training Approaches

There are two ways to train the multi-task learning model:

### 1. Joint Training (Original Approach)

In this approach, all tasks are trained simultaneously from scratch. The model learns to encode features useful for both pain and stimulus tasks.

```bash
python evaluate_multitask.py \
    --config configs/your_multitask_config.yaml \
    --syracuse_data_path /path/to/syracuse_features_root \
    --syracuse_marlin_base_dir /path/to/syracuse_metadata \
    --biovid_data_path /path/to/biovid_features_root \
    --shoulder_pain_data_path /path/to/shoulder_pain_features_root \  # Optional
    --n_gpus 1 \
    --batch_size 64 \
    --epochs 50 \
    --num_workers 4
```

### 2. Pretrain + Fine-tune Approach

This approach involves two steps:

#### Step 1: Pretrain the shared encoder and stimulus head on BioVid

First, train the model using only BioVid data to learn stimulus classification:

```bash
python pretrain_biovid.py \
    --config configs/your_multitask_config.yaml \
    --biovid_data_path /path/to/biovid_features_root \
    --n_gpus 1 \
    --batch_size 64 \
    --epochs 100 \
    --num_workers 4
```

This will save the best model to `ckpt/<model_name>_pretrain/`.

#### Step 2: Fine-tune on the multi-task objective

Next, load the pretrained weights and fine-tune the model on the multi-task objective:

```bash
python evaluate_multitask.py \
    --config configs/your_multitask_config.yaml \
    --syracuse_data_path /path/to/syracuse_features_root \
    --syracuse_marlin_base_dir /path/to/syracuse_metadata \
    --biovid_data_path /path/to/biovid_features_root \
    --shoulder_pain_data_path /path/to/shoulder_pain_features_root \  # Optional
    --pretrained_checkpoint ckpt/<model_name>_pretrain/<model_name>_pretrain-last.ckpt \
    --freeze_stimulus_head \  # Optional: freeze stimulus head to preserve learned knowledge
    --freeze_encoder \  # Optional: freeze shared encoder to preserve learned features
    --encoder_lr_factor 0.1 \  # Use lower learning rate for pretrained encoder (ignored if freeze_encoder is used)
    --n_gpus 1 \
    --batch_size 64 \
    --epochs 50 \
    --num_workers 4
```

The fine-tuning stage supports several training strategies:

1. **Full Fine-tuning**: Use only `--pretrained_checkpoint` without freezing any components
   - Both encoder and heads are trained, starting from pretrained weights
   - Good when you want to adapt all parts of the network

2. **Partial Freezing**: Use `--freeze_stimulus_head` and/or `--freeze_encoder`
   - `--freeze_stimulus_head`: Keep the pretrained stimulus classifier fixed
   - `--freeze_encoder`: Keep the pretrained feature encoder fixed
   - `--freeze_encoder --freeze_stimulus_head`: Only train the pain head (linear probing)

3. **Discriminative Learning Rates**: Use `--encoder_lr_factor`
   - Lower learning rate for the encoder (e.g., 0.1×) to preserve pretrained features
   - Only applies when encoder is not frozen

This approach has several advantages:
- The encoder learns strong features from BioVid's stimulus task first
- Freezing options help preserve knowledge from pretraining
- Discriminative learning rates help preserve useful features during fine-tuning
- The pain head can leverage the pretrained encoder's knowledge

Cross-validation can also be used with pretraining by adding the `--cv_folds` parameter to the fine-tuning command.

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