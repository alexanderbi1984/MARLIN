# Multi-Task Transfer Learning (Syracuse Pain + BioVid Stimulus)

This document describes how to set up and run the multi-task transfer learning experiment using the `evaluate_multitask.py` script. The goal is to train a model that learns from both Syracuse pain level data and BioVid stimulus level data, but evaluate its performance primarily on the Syracuse pain prediction task.

## Objective

Train a single model with a shared feature encoder and two separate CORAL heads:
1.  **Pain Head:** Predicts ordinal pain levels using Syracuse dataset features.
2.  **Stimulus Head:** Predicts ordinal stimulus levels using BioVid dataset features.

The model is trained on a combined dataset including Syracuse (original + augmented) and BioVid samples. Evaluation (validation and testing) is performed *only* on the Syracuse dataset splits to measure the model's ability to predict pain levels, potentially enhanced by the multi-task learning.

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

2.  **Data Setup:** You need both the Syracuse and BioVid datasets prepared as follows:

    **a) Syracuse Dataset:**

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

    **b) BioVid Dataset:**

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

temporal_reduction: mean  # How to aggregate features temporally (mean, max, min, none)

learning_rate: 1e-4      # Learning rate for the optimizer

# --- Optional: Model & Loss Weights --- 
pain_loss_weight: 1.0  # Weight for the pain task loss in the combined loss
stim_loss_weight: 1.0  # Weight for the stimulus task loss in the combined loss

# Optional: Define hidden layers for the shared MLP encoder
# If null or empty, a single Linear layer is used.
encoder_hidden_dims: [512, 256] # Example: [hidden1_size, hidden2_size]
# encoder_hidden_dims: null

# --- Optional: Data Balancing --- 
balance_sources: false           # Set to true to sample BioVid to match Syracuse training set size
balance_stimulus_classes: false # Set to true to apply class balancing within the BioVid training samples

# --- Optional: Add other parameters if needed by model/datamodule ---
# Example: optimizer_name: AdamW (if you add this option to the model)
```

**Parameter Explanations:**

*   `model_name`: Used for naming checkpoint directories and log files.
*   `num_pain_classes`: Number of ordinal classes for the Syracuse pain task. Must match the label structure in `clips_json.json` (`class_N`).
*   `num_stimulus_classes`: Number of ordinal classes for the BioVid stimulus task. Must match the label structure in `biovid_info.json` (`multiclass[N]`).
*   `syracuse_feature_dir`: Name of the subdirectory under `--syracuse_data_path` containing Syracuse `.npy` features.
*   `biovid_feature_dir`: Name of the subdirectory under `--biovid_data_path` containing BioVid `.npy` features.
*   `temporal_reduction`: Method to aggregate frame-level features (shape e.g., `(4, 768)`) into a single vector (shape `(768,)`) before feeding to the model's linear layers. Options: `mean`, `max`, `min`. Use `none` if the model handles sequences directly (may require model changes).
*   `learning_rate`: Optimizer learning rate.
*   `pain_loss_weight`, `stim_loss_weight`: Weights applied to the respective task losses before summing them for the total training loss.
*   `encoder_hidden_dims`: List of integers defining the hidden layer sizes for the shared MLP encoder. If `null` or empty, a simple `nn.Linear` layer is used.
*   `balance_sources`: If `true`, the number of BioVid samples used in training will be randomly sampled down (or up via replacement if `balance_stimulus_classes` is also true and requires it) to match the number of Syracuse training samples.
*   `balance_stimulus_classes`: If `true`, the BioVid samples included in the training set will be selected to achieve a more balanced distribution across stimulus classes.

## Running the Experiment

Use the `evaluate_multitask.py` script. Provide paths to the datasets and the configuration file.

```bash
python evaluate_multitask.py \
    --config configs/your_multitask_config.yaml \
    --syracuse_data_path /path/to/syracuse_features_root \
    --syracuse_marlin_base_dir /path/to/syracuse_metadata \
    --biovid_data_path /path/to/biovid_features_root \
    --n_gpus 1            `# Number of GPUs (0 for CPU)` \
    --batch_size 64       `# Training batch size` \
    --epochs 50           `# Maximum training epochs` \
    --num_workers 4       `# Dataloader workers` \
    --precision 32        `# Training precision (32, 16, bf16)` \
    # --predict_only        # Add this flag to skip training and only test/predict
    # --output_path predictions.csv # Add to save Syracuse test set predictions
```

## Output

*   **Checkpoints:** Saved in `ckpt/<model_name>_multitask/`. Includes the best checkpoint based on `val_pain_mae` and the last epoch checkpoint.
*   **Logs:** TensorBoard logs saved in `lightning_logs/`.
*   **Test Results:** Metrics from evaluating the best checkpoint on the Syracuse test set are printed to the console and saved to `ckpt/<model_name>_multitask/<model_name>_test_results.json`.
*   **Predictions (Optional):** If `--output_path` is provided, predicted pain levels for the Syracuse test set are saved to the specified CSV file. 