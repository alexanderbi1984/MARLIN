# evaluate_multitask.py
"""
MARLIN Multi-Task Model Evaluation Script

This script handles training and evaluation for the multi-task CORAL model
using data from both Syracuse (Pain) and BioVid (Stimulus) datasets.

Workflow:
----------
1. Reads a multi-task configuration YAML file specified by `--config`.
2. Initializes the `MultiTaskDataModule`.
3. Initializes the `MultiTaskCoralClassifier` model.
4. Trains the model on the combined training set (Syracuse + BioVid).
5. Evaluates the best checkpoint (based on Syracuse validation performance) 
   on the Syracuse test set.
6. Prints evaluation metrics primarily focused on the pain task.

Configuration YAML File Keys (`--config`):
------------------------------------------
*   `model_name: <your_run_name>` (**Required**): Name for checkpoints/logs.
*   `num_pain_classes: <int>` (**Required**): Number of ordinal classes for pain (Syracuse).
*   `num_stimulus_classes: <int>` (**Required**): Number of ordinal classes for stimulus (BioVid).
*   `syracuse_feature_dir: <name>` (**Required**): Feature dir name for Syracuse (relative to `--syracuse_data_path`).
*   `biovid_feature_dir: <name>` (**Required**): Feature dir name for BioVid (relative to `--biovid_data_path`).
*   `temporal_reduction: <mean|max|min|none>` (**Required**): Temporal reduction for features.
*   `learning_rate: <float>` (**Required**).
*   `pain_loss_weight: <float>` (Optional, default: 1.0): Weight for pain task loss.
*   `stim_loss_weight: <float>` (Optional, default: 1.0): Weight for stimulus task loss.
*   `balance_sources: <bool>` (Optional, default: False): Balance Syracuse vs BioVid in training set.
*   `balance_stimulus_classes: <bool>` (Optional, default: False): Balance BioVid classes in training.
*   `encoder_hidden_dims`: (Optional, list[int], default: None): Hidden dims for MLP encoder.

Example Command:
----------------
```bash
python evaluate_multitask.py \
    --config configs/multitask_coral_config.yaml \
    --syracuse_data_path /path/to/syracuse_features_root \
    --syracuse_marlin_base_dir /path/to/syracuse_metadata \
    --biovid_data_path /path/to/biovid_features_root \
    --n_gpus 1 \
    --batch_size 64 \
    --epochs 50 
```
"""

import argparse
import torch
torch.set_float32_matmul_precision('high')
import os
import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm
# Add scikit-learn imports for metrics
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, mean_absolute_error
# Add KFold import
from sklearn.model_selection import StratifiedKFold
from dataset.multitask import MultiTaskWrapper # Ensure wrapper is imported

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import custom modules
from dataset.multitask import MultiTaskDataModule
from model.multi_task_coral import MultiTaskCoralClassifier
from marlin_pytorch.util import read_yaml # Assuming this is available
from util.lr_logger import LrLogger
from util.seed import Seed
from util.system_stats_logger import SystemStatsLogger
from dataset.syracuse import SyracuseLP, SyracuseDataModule
from dataset.biovid import BioVidLP
# from dataset.utils import BalanceSampler, balance_source_datasets # Commented out: Module not found
from torch.utils.data import ConcatDataset, DataLoader

def run_multitask_evaluation(args, config):
    """Configures and runs the multi-task training and evaluation pipeline."""
    Seed.set(42) # Ensure reproducibility

    # --- Configuration --- 
    model_name = config.get("model_name", "multitask_coral_run")
    num_pain_classes = config["num_pain_classes"]
    num_stimulus_classes = config["num_stimulus_classes"]
    syracuse_feature_dir = config["syracuse_feature_dir"]
    biovid_feature_dir = config["biovid_feature_dir"]
    temporal_reduction = config.get("temporal_reduction", "mean")
    learning_rate = config["learning_rate"]
    pain_loss_weight = config.get("pain_loss_weight", 1.0)
    stim_loss_weight = config.get("stim_loss_weight", 1.0)
    balance_sources = config.get("balance_sources", False)
    balance_stimulus_classes = config.get("balance_stimulus_classes", False)
    encoder_hidden_dims = config.get("encoder_hidden_dims", None)

    print("--- Multi-Task Configuration ---")
    print(f"Model Name: {model_name}")
    print(f"Pain Classes: {num_pain_classes}, Stimulus Classes: {num_stimulus_classes}")
    print(f"Syracuse Features: {syracuse_feature_dir}, BioVid Features: {biovid_feature_dir}")
    print(f"Temporal Reduction: {temporal_reduction}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Loss Weights (Pain/Stim): {pain_loss_weight}/{stim_loss_weight}")
    print(f"Balance Sources: {balance_sources}, Balance Stimulus Classes: {balance_stimulus_classes}")
    print(f"Encoder Hidden Dims: {encoder_hidden_dims}")
    print(f"---------------------------------")

    # --- DataModule Setup --- 
    print("Initializing MultiTaskDataModule...")
    dm = MultiTaskDataModule(
        # Syracuse Params
        syracuse_root_dir=args.syracuse_data_path,
        syracuse_feature_dir=syracuse_feature_dir,
        syracuse_marlin_base_dir=args.syracuse_marlin_base_dir,
        num_pain_classes=num_pain_classes,
        # BioVid Params
        biovid_root_dir=args.biovid_data_path,
        biovid_feature_dir=biovid_feature_dir,
        num_stimulus_classes=num_stimulus_classes,
        # Common Params
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        temporal_reduction=temporal_reduction,
        # Balancing
        balance_sources=balance_sources,
        balance_stimulus_classes=balance_stimulus_classes,
        # Optional subset params (can be added to args/config if needed)
        # data_ratio=args.data_ratio, 
        # take_train=args.take_train,
        # take_val=args.take_val,
        # take_test=args.take_test,
    )
    # Note: setup() is called internally by Trainer.fit/test

    # --- Model Setup --- 
    print("Initializing MultiTaskCoralClassifier...")
    # Assuming features are 768-dim after temporal reduction (unless reduction='none')
    input_dim = 768 if temporal_reduction != "none" else 768 # Adjust if temporal_reduction='none' changes dim
    if temporal_reduction == "none":
        # TODO: How should 'none' reduction interact with the Linear encoder?
        # Might need a Flatten layer or different encoder structure. 
        # For now, assume input_dim is fixed after some operation.
        print("Warning: temporal_reduction='none' might require model adjustments (e.g., Flatten layer). Assuming 768 input dim for now.")
        # If sequence length is fixed (e.g., 4), input_dim could be 4*768
        # input_dim = 4 * 768 

    model = MultiTaskCoralClassifier(
        input_dim=input_dim,
        num_pain_classes=num_pain_classes,
        num_stimulus_classes=num_stimulus_classes,
        learning_rate=learning_rate,
        pain_loss_weight=pain_loss_weight,
        stim_loss_weight=stim_loss_weight,
        encoder_hidden_dims=encoder_hidden_dims,
        distributed=(args.n_gpus > 1), # Pass the distributed flag
        # optimizer_name can be added to config/args if needed
    )
    
    # Perform a sanity check
    model.sanity_check()

    # --- Trainer Setup --- 
    n_gpus = args.n_gpus
    accelerator = "cpu" if n_gpus == 0 else "gpu"
    strategy = "auto" if n_gpus <= 1 else "ddp"
    max_epochs = args.epochs
    precision = args.precision

    # Checkpoint callback: Monitor validation performance on the pain task (Syracuse)
    # ckpt_monitor = "val_pain_MAE"
    ckpt_monitor = "val_pain_QWK"
    ckpt_mode = "max" # Higher QWK is better
    ckpt_filename = model_name + "-{epoch}-{" + ckpt_monitor + ":.3f}"
    checkpoint_dir = f"ckpt/{model_name}_multitask"
    print(f"Checkpoints will be saved in: {checkpoint_dir}")
    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, 
        save_last=True,
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        save_top_k=1, # Save only the best checkpoint
        verbose=True
    )

    # Early stopping callback
    early_stop_patience = 200 # Example patience
    print(f"Using EarlyStopping with patience={early_stop_patience}, monitoring '{ckpt_monitor}' ({ckpt_mode})")
    early_stop_callback = EarlyStopping(
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        patience=early_stop_patience,
        verbose=True
    )

    trainer = Trainer(
        log_every_n_steps=1, # Log less frequently than every step
        devices=n_gpus,
        accelerator=accelerator,
        strategy=strategy,
        max_epochs=max_epochs,
        precision=precision,
        logger=True, # Use default TensorBoardLogger
        callbacks=[
            ckpt_callback, # Restore ModelCheckpoint
            early_stop_callback, # Restore EarlyStopping
            LrLogger(), 
            SystemStatsLogger() 
        ],
        benchmark=True if n_gpus > 0 else False,
        num_sanity_val_steps=0 # Keep sanity check disabled for now, can re-enable later
    )

    # --- Training --- 
    if not args.predict_only:
        print("--- Starting Training --- ")
        trainer.fit(model, datamodule=dm)
        best_ckpt_path = ckpt_callback.best_model_path
        print(f"Training finished. Best checkpoint saved at: {best_ckpt_path}")
    else:
        print("Skipping training due to --predict_only flag.")
        # Try to find the best checkpoint if predict_only
        best_ckpt_path = ckpt_callback.best_model_path # Might be empty if never trained
        if not best_ckpt_path or not os.path.exists(best_ckpt_path):
            # Fallback to last checkpoint if best doesn't exist
            last_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
            if os.path.exists(last_ckpt_path):
                best_ckpt_path = last_ckpt_path
            else:
                 print("Error: No checkpoint found to use for prediction.")
                 return
        print(f"Using checkpoint for prediction: {best_ckpt_path}")

    # --- Testing --- 
    if best_ckpt_path:
        print("--- Starting Testing (on Syracuse Test Set) --- ")
        # test() automatically loads the best checkpoint and logs metrics defined in test_step
        test_results = trainer.test(model, datamodule=dm, ckpt_path=best_ckpt_path)
        print("Test Set Results (Logged Metrics):")
        # test_results is a list containing one dictionary
        if test_results:
            print(json.dumps(test_results[0], indent=4)) 
            # Save logged test results
            results_filename = os.path.join(checkpoint_dir, f"{model_name}_test_results_logged.json")
            try:
                with open(results_filename, 'w') as f:
                    json.dump(test_results[0], f, indent=4)
                print(f"Logged test results saved to: {results_filename}")
            except Exception as e:
                print(f"Error saving logged test results: {e}")
        else:
            print("No logged test results returned by trainer.test().")

        # --- Manual Calculation for Accuracy, Confusion Matrix, and QWK --- 
        print("--- Calculating Accuracy, CM & QWK (Syracuse Test Set) --- ")
        # Load the best model again or reuse the one loaded by trainer.test
        # It's safer to load explicitly to ensure we have the correct weights
        try:
            model = MultiTaskCoralClassifier.load_from_checkpoint(best_ckpt_path)
            model.eval()
            device = torch.device("cuda" if args.n_gpus > 0 else "cpu")
            model.to(device)

            all_true_pain_labels = []
            all_pred_pain_labels = []
            
            # Get the test dataloader again
            dm.setup('test') # Ensure test dataset/loader is ready
            test_loader = dm.test_dataloader()
            if test_loader is None:
                 raise ValueError("Test dataloader is None, cannot proceed with manual evaluation.")

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Manual Test Prediction"):
                    features, pain_labels, _ = batch # We only need pain labels from test set
                    
                    # Ensure only valid pain labels are considered (should be all in test set)
                    valid_mask = pain_labels != -1
                    if not valid_mask.any(): continue # Skip batch if no valid labels somehow
                    
                    features = features[valid_mask].to(device)
                    true_labels = pain_labels[valid_mask]
                    
                    pain_logits, _ = model(features)
                    pain_probs = torch.sigmoid(pain_logits)
                    # Use the static method directly for clarity
                    pred_labels = MultiTaskCoralClassifier.prob_to_label(pain_probs) 
                    
                    all_true_pain_labels.append(true_labels.cpu().numpy())
                    all_pred_pain_labels.append(pred_labels.cpu().numpy())
            
            # Concatenate results from all batches
            if all_true_pain_labels:
                y_true = np.concatenate(all_true_pain_labels)
                y_pred = np.concatenate(all_pred_pain_labels)
                
                # Calculate Metrics
                accuracy = accuracy_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)
                qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
                
                print(f"\nManual Test Set Calculation Results (Pain Task):")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}")
                print("  Confusion Matrix:")
                print(cm)
                
                # Optionally save these too
                manual_results = {
                    'test_pain_accuracy': accuracy,
                    'test_pain_qwk': qwk,
                    'test_pain_confusion_matrix': cm.tolist()
                }
                manual_results_filename = os.path.join(checkpoint_dir, f"{model_name}_test_results_manual.json")
                try:
                    with open(manual_results_filename, 'w') as f:
                        json.dump(manual_results, f, indent=4)
                    print(f"Manual test results saved to: {manual_results_filename}")
                except Exception as e:
                    print(f"Error saving manual test results: {e}")
                    
            else:
                 print("No valid pain labels found in the test set during manual calculation.")

        except Exception as e:
             print(f"Error during manual test evaluation: {e}")
             # Optionally re-raise if debugging
             # raise e
        # ------------------------------------------------------------

    else:
        print("Skipping testing because no valid checkpoint was found.")

    # --- Prediction (Optional) --- 
    if args.output_path:
        print(f"--- Generating Predictions (on Syracuse Test Set) --- ")
        if best_ckpt_path:
            # Load the best model explicitly for prediction
            # model = MultiTaskCoralClassifier.load_from_checkpoint(best_ckpt_path)
            # model.eval()
            # device = torch.device("cuda" if n_gpus > 0 else "cpu")
            # model.to(device)
            
            # Use trainer.predict to handle device placement etc.
            dm.setup('test') # Ensure test dataset is ready
            predictions = trainer.predict(model, dataloaders=dm.test_dataloader(), ckpt_path=best_ckpt_path)
            
            # Process predictions (which are outputs of predict_step in the model)
            all_pain_preds = []
            # We don't have ground truth easily here unless predict_step returns it
            # Or we reload the test dataloader separately
            
            for batch_preds in predictions:
                 # Assuming predict_step returns (pain_pred_labels, stim_pred_labels)
                 pain_preds, _ = batch_preds 
                 all_pain_preds.append(pain_preds.cpu().numpy())
            
            if all_pain_preds:
                all_pain_preds_np = np.concatenate(all_pain_preds)
                df = pd.DataFrame({'predicted_pain_level': all_pain_preds_np})
                
                # Try to add filenames if possible (requires modifying dataloader/wrapper)
                # try:
                #     test_filenames = [dm.test_dataset.dataset.name_list[i] for i in range(len(dm.test_dataset))]
                #     if len(test_filenames) == len(df):
                #          df['filename'] = test_filenames
                # except Exception:
                #      print("Warning: Could not retrieve filenames for predictions.")
                     
                df.to_csv(args.output_path, index=False)
                print(f"Predictions for Syracuse test set saved to: {args.output_path}")
            else:
                 print("No predictions were generated.")

        else:
            print("Skipping prediction because no valid checkpoint was found.")


def run_multitask_cv(args, config):
    """Runs K-Fold Cross-Validation for the multi-task model."""
    Seed.set(42)
    n_splits = args.cv_folds
    print(f"--- Starting {n_splits}-Fold Cross-Validation --- ")
    
    # --- Common Config --- 
    # Extract necessary params from config and args
    model_name = config.get("model_name", f"multitask_cv_{n_splits}fold")
    num_pain_classes = config["num_pain_classes"]
    num_stimulus_classes = config["num_stimulus_classes"]
    syracuse_feature_dir = config["syracuse_feature_dir"]
    biovid_feature_dir = config["biovid_feature_dir"]
    temporal_reduction = config.get("temporal_reduction", "mean")
    learning_rate = config["learning_rate"]
    pain_loss_weight = config.get("pain_loss_weight", 1.0)
    stim_loss_weight = config.get("stim_loss_weight", 1.0)
    balance_sources = config.get("balance_sources", False)
    balance_stimulus_classes = config.get("balance_stimulus_classes", False)
    encoder_hidden_dims = config.get("encoder_hidden_dims", None)
    input_dim = 768 # Assuming same as before, adjust if needed
    data_ratio = config.get("data_ratio", 1.0) # Use from config if available
    take_train = config.get("take_train", None)
    patience = config.get("patience", 10)
    monitor_metric = config.get("monitor_metric", "val_pain_MAE") # QWK or MAE
    monitor_mode = "min" if "MAE" in monitor_metric else "max"

    # --- Load Syracuse Metadata (once) --- 
    print("  Loading Syracuse metadata for splitting...")
    try:
        # DEBUG: Inspect SyracuseDataModule signature and arguments
        import inspect
        print("DEBUG: SyracuseDataModule __init__ signature:", inspect.signature(SyracuseDataModule.__init__))
        print("DEBUG: Preparing args for SyracuseDataModule:")
        print("    root_dir=", args.syracuse_data_path)
        print("    task=multiclass")
        print("    num_classes=", num_pain_classes)
        print("    batch_size=", 1)
        print("    feature_dir=", syracuse_feature_dir)
        print("    marlin_base_dir=", args.syracuse_marlin_base_dir)
        print("    temporal_reduction=", temporal_reduction)
        print("    num_workers=", 0)
        # Instantiate SyracuseDataModule directly to load metadata
        # Using positional arguments for required ones to avoid potential kwarg issues
        syracuse_dm_for_meta = SyracuseDataModule(
            args.syracuse_data_path,       # root_dir
            'multiclass',                 # task
            num_pain_classes,             # num_classes
            1,                            # batch_size (dummy)
            syracuse_feature_dir,         # feature_dir
            args.syracuse_marlin_base_dir, # marlin_base_dir 
            # Optional args as keywords
            temporal_reduction=temporal_reduction,
            num_workers=0                 # num_workers (dummy)
        )
        # Call setup just to load metadata and parse it
        syracuse_dm_for_meta.setup(stage=None) # Use stage=None to load everything
        # Access attributes from the SyracuseDataModule instance
        # Note: It's syracuse_dm_for_meta.all_metadata, not all_syracuse_metadata
        all_syracuse_metadata = syracuse_dm_for_meta.all_metadata 
        original_clips = syracuse_dm_for_meta.original_clips # dict: video_id -> [filenames]
        augmented_clips = syracuse_dm_for_meta.augmented_clips # dict: video_id -> [filenames]
        video_id_labels = syracuse_dm_for_meta.video_id_labels # dict: video_id -> label
        if not video_id_labels:
            raise ValueError("video_id_labels empty after Syracuse metadata setup.")
    except Exception as e:
        print(f"Error loading Syracuse metadata for CV: {e}")
        return

    unique_video_ids = sorted(list(video_id_labels.keys()))
    video_labels_for_stratify = [video_id_labels[vid] for vid in unique_video_ids]
    if len(unique_video_ids) < n_splits:
        raise ValueError(f"Number of unique Syracuse videos ({len(unique_video_ids)}) is less than the number of folds ({n_splits}).")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_checkpoint_paths = [] # Stores best ckpt path for each fold
    fold_val_filenames = [] # Stores list of val filenames for each fold
    
    # --- Load Full BioVid Training Data (once) --- 
    print("  Loading full BioVid training data...")
    try:
        full_biovid_train_set = BioVidLP(
             root_dir=args.biovid_data_path, 
             feature_dir=biovid_feature_dir,
             split='train', 
             task='multiclass', 
             num_classes=num_stimulus_classes,
             temporal_reduction=temporal_reduction,
             data_ratio=data_ratio, 
             take_num=take_train
        )
        print(f"    Full BioVid train size: {len(full_biovid_train_set)}")
    except Exception as e:
         print(f"Error loading BioVid training data: {e}. Cannot proceed with CV.")
         return

    # --- Cross-Validation Loop --- 
    for fold_idx, (train_vid_idx_indices, val_vid_idx_indices) in enumerate(skf.split(unique_video_ids, video_labels_for_stratify)):
        print(f"\n===== Starting Fold {fold_idx + 1}/{n_splits} =====")
        
        # Map indices back to video IDs
        fold_train_video_ids = [unique_video_ids[i] for i in train_vid_idx_indices]
        fold_val_video_ids = [unique_video_ids[i] for i in val_vid_idx_indices]
        
        # DEBUGGING - Print fold data statistics
        print(f"\nFOLD DEBUG: Total unique video IDs: {len(unique_video_ids)}")
        print(f"FOLD DEBUG: Train set should be ~{(n_splits-1)/n_splits:.1%} of videos")
        print(f"FOLD DEBUG: Actual train/val video split: {len(fold_train_video_ids)}/{len(fold_val_video_ids)} = {len(fold_train_video_ids)/len(unique_video_ids):.1%}/{len(fold_val_video_ids)/len(unique_video_ids):.1%}")
        
        print(f"  Train Video IDs: {len(fold_train_video_ids)}, Val Video IDs: {len(fold_val_video_ids)}")

        # DEBUG: Inspect video IDs and types
        print("DEBUG: fold_train_video_ids (first 10):", fold_train_video_ids[:10])
        print("DEBUG: type of video IDs in fold_train_video_ids:", type(fold_train_video_ids[0]) if fold_train_video_ids else None)
        print("DEBUG: sample original_clips video_id values (first 5):", [clip['video_id'] for clip in original_clips[:5]])

        # Filter Syracuse filenames for this fold based on clip_data['video_id']
        fold_train_video_ids_set = set(fold_train_video_ids)
        fold_val_video_ids_set = set(fold_val_video_ids)

        print("DEBUG: fold_train_video_ids_set (first 10):", list(fold_train_video_ids_set)[:10])
        print("DEBUG: check membership for first 5 original_clips:")
        for clip in original_clips[:5]:
            vid = clip.get('video_id')
            print(f"    clip video_id={vid}({type(vid)}), in train_set? {vid in fold_train_video_ids_set}")
        
        syracuse_train_filenames = []
        syracuse_val_filenames = []
        
        # Process original clips (assign to train or val based on video_id)
        for clip_data in original_clips:
            vid = clip_data.get('video_id')
            filename = clip_data.get('filename')
            if vid in fold_train_video_ids_set:
                syracuse_train_filenames.append(filename)
            elif vid in fold_val_video_ids_set:
                syracuse_val_filenames.append(filename)

        # Process augmented clips (only add to train if video_id matches)
        for clip_data in augmented_clips:
            vid = clip_data.get('video_id')
            filename = clip_data.get('filename')
            if vid in fold_train_video_ids_set:
                syracuse_train_filenames.append(filename)

        print(f"  Syracuse Train Files: {len(syracuse_train_filenames)}, Val Files: {len(syracuse_val_filenames)}")
        # Count files per video ID in each set
        train_files_per_vid = {}
        val_files_per_vid = {}
        for clip in original_clips:
            vid = clip.get('video_id')
            if vid in fold_train_video_ids_set:
                train_files_per_vid[vid] = train_files_per_vid.get(vid, 0) + 1
            elif vid in fold_val_video_ids_set:
                val_files_per_vid[vid] = val_files_per_vid.get(vid, 0) + 1
        
        for clip in augmented_clips:
            vid = clip.get('video_id')
            if vid in fold_train_video_ids_set:
                train_files_per_vid[vid] = train_files_per_vid.get(vid, 0) + 1
        
        print(f"FOLD DEBUG: Avg files per train video: {sum(train_files_per_vid.values())/len(train_files_per_vid) if train_files_per_vid else 0:.1f}")
        print(f"FOLD DEBUG: Avg files per val video: {sum(val_files_per_vid.values())/len(val_files_per_vid) if val_files_per_vid else 0:.1f}")
        print(f"FOLD DEBUG: Total clips ratio (train/all): {len(syracuse_train_filenames)/(len(syracuse_train_filenames)+len(syracuse_val_filenames)):.1%}")
        
        # Calculate expected batches per epoch
        expected_batches = len(syracuse_train_filenames) // args.batch_size
        if len(syracuse_train_filenames) % args.batch_size != 0:
            expected_batches += 1
        print(f"FOLD DEBUG: Expected batches per epoch: {expected_batches} (train_size={len(syracuse_train_filenames)}, batch_size={args.batch_size})")
        
        # Skip folds with empty train or validation splits
        if not syracuse_train_filenames or not syracuse_val_filenames:
            print(f"  WARNING: Fold {fold_idx + 1} has empty train or validation file list. Skipping this fold.")
            continue
        # Store validation filenames for later evaluation
        fold_val_filenames.append(syracuse_val_filenames)

        # Create Syracuse datasets for the fold using correct SyracuseLP signature
        syracuse_train_set = SyracuseLP(
            args.syracuse_data_path,      # root_dir
            syracuse_feature_dir,         # feature_dir
            'train',                      # split
            'multiclass',                 # task
            num_pain_classes,             # num_classes
            temporal_reduction,           # temporal_reduction
            syracuse_train_filenames,     # name_list of specific filenames
            all_syracuse_metadata         # metadata dict from SyracuseDataModule
        )
        syracuse_val_set = SyracuseLP(
            args.syracuse_data_path,      # root_dir
            syracuse_feature_dir,         # feature_dir
            'val',                        # split
            'multiclass',                 # task
            num_pain_classes,             # num_classes
            temporal_reduction,           # temporal_reduction
            syracuse_val_filenames,       # name_list of validation filenames
            all_syracuse_metadata         # metadata dict
        )

        # MultiTaskCoralClassifier expects 3 values (features, pain_labels, stim_labels)
        # So we need to wrap datasets with MultiTaskWrapper for training
        wrapped_syracuse_train = MultiTaskWrapper(syracuse_train_set, 'pain')
        wrapped_syracuse_val = MultiTaskWrapper(syracuse_val_set, 'pain')
        
        # Create DataLoaders with the wrapped datasets
        fold_train_loader = DataLoader(
            wrapped_syracuse_train, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        
        fold_val_loader = DataLoader(
            wrapped_syracuse_val, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        
        print(f"  Created DataLoaders for fold {fold_idx + 1}")
        
        # --- Configure Model & Trainer for the Fold --- 
        model = MultiTaskCoralClassifier(
            input_dim=input_dim,
            num_pain_classes=num_pain_classes,
            num_stimulus_classes=num_stimulus_classes,
            learning_rate=learning_rate,
            encoder_hidden_dims=encoder_hidden_dims,
            pain_loss_weight=pain_loss_weight,
            stim_loss_weight=stim_loss_weight
        )
        
        fold_checkpoint_dir = os.path.join("ckpt", f"{model_name}", f"fold_{fold_idx}")
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_checkpoint_dir,
            filename=f"{{epoch}}-{{{monitor_metric}:.4f}}",
            save_top_k=1,
            monitor=monitor_metric,
            mode=monitor_mode,
            save_last=False # Only save the best
        )
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            verbose=True,
            mode=monitor_mode
        )
        
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1, # Run each fold on a single device for simplicity
            logger=pl.loggers.TensorBoardLogger("logs", name=f"{model_name}/fold_{fold_idx}"),
            callbacks=[checkpoint_callback, early_stop_callback, LrLogger(), SystemStatsLogger()],
            precision=args.precision,
            log_every_n_steps=50, # Log less frequently during CV
            enable_progress_bar=True, # Show progress bar per fold
            # deterministic=True, # Could add for reproducibility if needed
            # benchmark=True, # Add if using GPU and input size is constant
            num_sanity_val_steps=0 # Disable sanity check for CV folds
        )
        
        # --- Train the Fold ---
        print(f"  Starting training for fold {fold_idx + 1}...")
        # Align with evaluate.py's parameter style
        trainer.fit(model, train_dataloaders=fold_train_loader, val_dataloaders=fold_val_loader)

        # --- Store Best Checkpoint Path ---
        best_ckpt_path = checkpoint_callback.best_model_path
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            print(f"  Fold {fold_idx + 1} finished. Best checkpoint: {best_ckpt_path}")
            fold_checkpoint_paths.append(best_ckpt_path)
        else:
            print(f"  WARNING: No best checkpoint found for fold {fold_idx + 1}. Using last checkpoint if available.")
            # Fallback logic if needed (e.g., use last.ckpt, but ModelCheckpoint isn't saving it here)
            last_ckpt = os.path.join(fold_checkpoint_dir, "last.ckpt") # Check if it exists
            if os.path.exists(last_ckpt):
                 fold_checkpoint_paths.append(last_ckpt)
                 print(f"  Using last checkpoint: {last_ckpt}")
            else:
                 print(f"  ERROR: Could not find any checkpoint for fold {fold_idx + 1}")
                 fold_checkpoint_paths.append(None) # Mark as failed/missing

    # After processing all folds, ensure at least one fold ran successfully
    if not fold_val_filenames or not fold_checkpoint_paths:
        print(f"\nWARNING: No valid folds were executed (train or val splits empty). Cross-validation aborted.")
        return
    print(f"\n===== Cross-Validation Finished =====")

    # --- Aggregate and Evaluate CV Results ---
    print(f"Stored best checkpoint paths for {len(fold_checkpoint_paths)} folds.")
    print(f"Validation filenames stored for {len(fold_val_filenames)} folds.")

    # Sanity check lengths
    if len(fold_checkpoint_paths) != n_splits or len(fold_val_filenames) != n_splits:
        print(f"Warning: Mismatch in number of checkpoints ({len(fold_checkpoint_paths)}) or validation sets ({len(fold_val_filenames)}) vs folds ({n_splits}).")

    # Evaluate each fold's best checkpoint on its corresponding validation set
    all_fold_metrics = []
    for i, (ckpt_path, val_files) in enumerate(zip(fold_checkpoint_paths, fold_val_filenames)):
        if ckpt_path is None:
            print(f"Skipping evaluation for Fold {i+1} due to missing checkpoint.")
            all_fold_metrics.append(None) # Placeholder for failed fold
            continue

        print(f"\n--- Evaluating Fold {i+1} Checkpoint on its Validation Set ---")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Validation Files ({len(val_files)}): {val_files[:5]}...") # Show first 5

        # We need a separate function to load a checkpoint and evaluate it
        # on a specific set of validation files (Syracuse only for validation)
        try:
            fold_metrics = _evaluate_multitask_fold_checkpoint(
                checkpoint_path=ckpt_path,
                val_filenames=val_files,
                args=args,
                config=config # Pass necessary config
            )
            if fold_metrics:
                print(f"  Fold {i+1} Metrics: {fold_metrics}")
                all_fold_metrics.append(fold_metrics)
            else:
                print(f"  Fold {i+1} evaluation failed or returned no metrics.")
                all_fold_metrics.append(None)
        except Exception as e:
            print(f"  Error evaluating Fold {i+1}: {e}")
            all_fold_metrics.append(None)

    # --- Aggregate Results ---
    valid_fold_metrics = [m for m in all_fold_metrics if m is not None]
    if not valid_fold_metrics:
        print("\nError: No valid fold metrics were collected. Cannot aggregate results.")
        return

    print(f"\n--- Aggregated Cross-Validation Results ({len(valid_fold_metrics)}/{n_splits} Folds) ---")

    # Example: Aggregate QWK and MAE
    avg_qwk = np.mean([m['val_pain_qwk'] for m in valid_fold_metrics])
    std_qwk = np.std([m['val_pain_qwk'] for m in valid_fold_metrics])
    avg_mae = np.mean([m['val_pain_mae'] for m in valid_fold_metrics])
    std_mae = np.std([m['val_pain_mae'] for m in valid_fold_metrics])
    # Add other metrics as needed

    print(f"  Average Validation Pain QWK: {avg_qwk:.4f} +/- {std_qwk:.4f}")
    print(f"  Average Validation Pain MAE: {avg_mae:.4f} +/- {std_mae:.4f}")

    # Optionally save aggregated results
    agg_results = {
        'model_name': model_name,
        'n_splits': n_splits,
        'num_successful_folds': len(valid_fold_metrics),
        'monitor_metric': monitor_metric,
        'monitor_mode': monitor_mode,
        'avg_val_pain_qwk': avg_qwk,
        'std_val_pain_qwk': std_qwk,
        'avg_val_pain_mae': avg_mae,
        'std_val_pain_mae': std_mae,
        'all_fold_metrics': all_fold_metrics # Include individual fold results
    }
    results_dir = os.path.join("results", model_name)
    os.makedirs(results_dir, exist_ok=True)
    agg_results_filename = os.path.join(results_dir, f"{model_name}_cv_summary.json")
    try:
        with open(agg_results_filename, 'w') as f:
            json.dump(agg_results, f, indent=4)
        print(f"Aggregated CV results saved to: {agg_results_filename}")
    except Exception as e:
        print(f"Error saving aggregated CV results: {e}")


def _evaluate_multitask_fold_checkpoint(checkpoint_path, val_filenames, args, config):
    """Loads a model from a checkpoint and evaluates it on the given validation filenames."""
    if not checkpoint_path or not os.path.exists(checkpoint_path) or not val_filenames:
        print("  Evaluation skipped: Missing checkpoint or validation files.")
        return None

    try:
        # --- Load Model ---
        # Need to know model parameters to load
        # Extract from config or assume defaults consistent with training
        num_pain_classes = config["num_pain_classes"]
        num_stimulus_classes = config["num_stimulus_classes"]
        syracuse_feature_dir = config["syracuse_feature_dir"]
        temporal_reduction = config.get("temporal_reduction", "mean")
        input_dim = 768 # Assuming same as before
        encoder_hidden_dims = config.get("encoder_hidden_dims", None)
        pain_loss_weight = config.get("pain_loss_weight", 1.0)
        stim_loss_weight = config.get("stim_loss_weight", 1.0)

        # Load the model from the checkpoint
        model = MultiTaskCoralClassifier.load_from_checkpoint(
            checkpoint_path,
            input_dim=input_dim,
            num_pain_classes=num_pain_classes,
            num_stimulus_classes=num_stimulus_classes,
            learning_rate=0, # Not needed for evaluation
            encoder_hidden_dims=encoder_hidden_dims,
            pain_loss_weight=pain_loss_weight,
            stim_loss_weight=stim_loss_weight
        )
        model.eval()
        
        # --- Load Syracuse metadata for evaluation ---
        syracuse_dm_meta = SyracuseDataModule(
            args.syracuse_data_path,
            'multiclass',
            num_pain_classes,
            1,
            syracuse_feature_dir,
            args.syracuse_marlin_base_dir,
            temporal_reduction=temporal_reduction,
            num_workers=args.num_workers
        )
        syracuse_dm_meta.setup(stage=None)
        metadata = syracuse_dm_meta.all_metadata

        # Create the validation dataset for this fold with correct SyracuseLP signature
        syracuse_val_set = SyracuseLP(
            args.syracuse_data_path,    # root_dir
            syracuse_feature_dir,       # feature_dir
            'val',                      # split
            'multiclass',               # task
            num_pain_classes,           # num_classes
            temporal_reduction,         # temporal_reduction
            val_filenames,              # name_list for this fold
            metadata                    # metadata dict
        )
        
        # Create a regular DataLoader without MultiTaskWrapper
        val_loader = DataLoader(
            syracuse_val_set, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Manual prediction loop
        all_pain_preds = []
        all_pain_labels = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold Validation"):
                features, true_labels = batch  # SyracuseLP returns (features, labels)
                features = features.to(device)
                
                # Get pain predictions - model returns tuple (pain_logits, stim_logits)
                pain_logits, _ = model(features)
                pain_probas = torch.sigmoid(pain_logits)
                
                # Get predicted classes using threshold method
                predicted_classes = torch.sum(pain_probas > 0.5, dim=1)
                
                # Add batch results to lists for aggregation
                all_pain_preds.append(predicted_classes.cpu())
                all_pain_labels.append(true_labels.cpu())

        # Convert prediction lists to numpy arrays
        if not all_pain_preds:
            print(f"  No predictions were generated for evaluation.")
            return None
            
        pain_preds_tensor = torch.cat(all_pain_preds)
        pain_labels_tensor = torch.cat(all_pain_labels)
        
        y_true = pain_labels_tensor.numpy()
        y_pred = pain_preds_tensor.numpy()

        # Calculate metrics for the pain task
        mae = mean_absolute_error(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_pain_classes)))

        print(f"    Fold Validation Results - MAE: {mae:.4f}, Acc: {accuracy:.4f}, QWK: {qwk:.4f}")
        
        return {
            'val_pain_mae': mae,
            'val_pain_accuracy': accuracy,
            'val_pain_qwk': qwk,
            'val_pain_confusion_matrix': cm.tolist()
        }

    except Exception as e:
        print(f"Error evaluating fold checkpoint: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MARLIN Multi-Task Model Evaluation")
    
    # --- Paths --- 
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the multi-task YAML configuration file.")
    parser.add_argument("--syracuse_data_path", type=str, required=True, 
                        help="Root directory for Syracuse features.")
    parser.add_argument("--syracuse_marlin_base_dir", type=str, required=True, 
                        help="Base directory containing Syracuse metadata (e.g., clips_json.json)." )
    parser.add_argument("--biovid_data_path", type=str, required=True, 
                        help="Root directory for BioVid features.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional path to save prediction CSV file (Syracuse test set).")

    # --- Training --- 
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs (0 for CPU).")
    parser.add_argument("--precision", type=str, default="32", help="Training precision (32, 16, bf16).")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    
    # --- Control --- 
    parser.add_argument("--predict_only", action="store_true", default=False,
                        help="Skip training, load best/last checkpoint and run testing/prediction.")
    parser.add_argument("--cv_folds", type=int, default=0, 
                        help="Number of folds for cross-validation. If > 1, runs CV mode instead of train/val/test split. Default: 0 (disabled).")

    args = parser.parse_args()

    # --- Run Evaluation --- 
    try:
        config = read_yaml(args.config)
        # Check if CV mode is requested
        if args.cv_folds > 1:
             print(f"--- Running {args.cv_folds}-Fold Cross-Validation ---")
             run_multitask_cv(args, config) # Call the new CV function
        else:
             print("--- Running Standard Train/Val/Test Evaluation ---")
             run_multitask_evaluation(args, config)
             
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
    except KeyError as e:
        print(f"Error: Missing required key in config file {args.config}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Optionally re-raise for debugging
        # raise e
 