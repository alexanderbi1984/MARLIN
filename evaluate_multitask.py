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
*   `shoulder_pain_feature_dir: <name>` (Optional): Feature dir name for ShoulderPain (relative to `--shoulder_pain_data_path`).
*   `temporal_reduction: <mean|max|min|none>` (**Required**): Temporal reduction for features.
*   `learning_rate: <float>` (**Required**).
*   `weight_decay: <float>` (Optional, default: 0.0): Weight decay for the optimizer.
*   `label_smoothing: <float>` (Optional, default: 0.0): Amount of label smoothing for Syracuse data (0.0-1.0).
*   `use_class_weights: <bool>` (Optional, default: False): Whether to use class weights to handle class imbalance.
*   `balance_pain_classes: <bool>` (Optional, default: False): Whether to use weighted sampling for Syracuse pain classes.
*   `pain_loss_weight: <float>` (Optional, default: 1.0): Weight for pain task loss.
*   `stim_loss_weight: <float>` (Optional, default: 1.0): Weight for stimulus task loss.
*   `balance_sources: <bool>` (Optional, default: False): Balance Syracuse vs BioVid in training set.
*   `balance_stimulus_classes: <bool>` (Optional, default: False): Balance BioVid classes in training.
*   `encoder_hidden_dims`: (Optional, list[int], default: None): Hidden dims for MLP encoder.
*   `use_distance_penalty: <bool>` (Optional, default: False): Whether to use distance penalty for CORAL loss.
*   `focal_gamma: <float>` (Optional, default: None): Focal loss gamma parameter for CORAL loss.

Example Command:
----------------
```bash
python evaluate_multitask.py \
    --config configs/multitask_coral_config.yaml \
    --syracuse_data_path /path/to/syracuse_features_root \
    --syracuse_marlin_base_dir /path/to/syracuse_metadata \
    --biovid_data_path /path/to/biovid_features_root \
    --shoulder_pain_data_path /path/to/shoulder_pain_features_root \  # Optional
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
from dataset.shoulder_pain import ShoulderPainLP  # Import ShoulderPainLP class
# from dataset.utils import BalanceSampler, balance_source_datasets # Commented out: Module not found
from torch.utils.data import ConcatDataset, DataLoader
# Import CombinedLoader for balanced sampling
# Try both import paths for CombinedLoader to support different PyTorch Lightning versions
try:
    # For newer PyTorch Lightning versions
    from lightning.pytorch.utilities.combined_loader import CombinedLoader
except ImportError:
    try:
        # For older PyTorch Lightning versions
        from pytorch_lightning.utilities.combined_loader import CombinedLoader
    except ImportError:
        try:
            # For even older PyTorch Lightning versions
            from pytorch_lightning.trainer.supporters import CombinedLoader
        except ImportError:
            raise ImportError("Could not import CombinedLoader from any known path in PyTorch Lightning. "
                            "Please check your version compatibility.")

# Add a custom callback for stimulus weight scheduling
class StimWeightSchedulerCallback(pl.Callback):
    """Callback to schedule stimulus weight during training."""
    
    def __init__(self, initial_weight=5.0, final_weight=1.0, decay_epochs=50, scheduler_type="cosine"):
        """
        Initialize the stimulus weight scheduler.
        
        Args:
            initial_weight: Starting weight for stimulus loss
            final_weight: Final weight for stimulus loss
            decay_epochs: Number of epochs over which to decay the weight
            scheduler_type: Type of scheduler ("cosine" or "linear")
        """
        super().__init__()
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.decay_epochs = decay_epochs
        self.scheduler_type = scheduler_type.lower()
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Update the stimulus weight at the start of each epoch."""
        current_epoch = trainer.current_epoch
        
        # Calculate the weight based on scheduler type
        if current_epoch >= self.decay_epochs:
            # After decay_epochs, use the final weight
            new_weight = self.final_weight
        else:
            # During decay period, calculate based on scheduler type
            progress = current_epoch / self.decay_epochs
            
            if self.scheduler_type == "cosine":
                # Cosine annealing: smoothly transitions from initial to final
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                new_weight = self.final_weight + (self.initial_weight - self.final_weight) * cosine_decay
            else:
                # Linear decay: straight line from initial to final
                new_weight = self.initial_weight - (self.initial_weight - self.final_weight) * progress
        
        # Update the model's stimulus weight in the CORRECT location
        # Fix: Update pl_module.hparams.stim_loss_weight instead of pl_module.stim_loss_weight
        pl_module.hparams.stim_loss_weight = new_weight
        
        # Also update the attribute directly for any code that might use it
        if hasattr(pl_module, 'stim_loss_weight'):
            pl_module.stim_loss_weight = new_weight
        
        # Log the current weight
        trainer.logger.experiment.add_scalar("stim_loss_weight", new_weight, current_epoch)
        
        # Every 10 epochs or at the start/end, print the current weight
        if current_epoch == 0 or current_epoch == trainer.max_epochs - 1 or current_epoch % 10 == 0:
            print(f"  Epoch {current_epoch}: Stimulus loss weight = {new_weight:.4f}")

def run_multitask_evaluation(args, config):
    """Configures and runs the multi-task training and evaluation pipeline."""
    Seed.set(42) # Ensure reproducibility

    # --- Configuration --- 
    model_name = config.get("model_name", "multitask_coral_run")
    num_pain_classes = config["num_pain_classes"]
    num_stimulus_classes = config["num_stimulus_classes"]
    syracuse_feature_dir = config["syracuse_feature_dir"]
    biovid_feature_dir = config["biovid_feature_dir"]
    shoulder_pain_feature_dir = config.get("shoulder_pain_feature_dir", None)
    temporal_reduction = config.get("temporal_reduction", "mean")
    learning_rate = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.0)  # Default to 0.0 (no weight decay)
    label_smoothing = config.get("label_smoothing", 0.0)  # Default to 0.0 (no label smoothing)
    use_class_weights = config.get("use_class_weights", False)  # Default to False (no class weighting)
    balance_pain_classes = config.get("balance_pain_classes", False)  # Default to False (no weighted sampling)
    pain_loss_weight = config.get("pain_loss_weight", 1.0)
    stim_loss_weight = config.get("stim_loss_weight", 1.0)
    balance_sources = config.get("balance_sources", False)
    balance_stimulus_classes = config.get("balance_stimulus_classes", False)
    encoder_hidden_dims = config.get("encoder_hidden_dims", None)
    # New parameters for coral_loss
    use_distance_penalty = config.get("use_distance_penalty", False)
    focal_gamma = config.get("focal_gamma", None)
    
    # Print pretrained model information if provided
    using_pretrained = args.pretrained_checkpoint is not None
    if using_pretrained:
        print(f"Using pretrained model from: {args.pretrained_checkpoint}")
        print(f"Freeze stimulus head: {args.freeze_stimulus_head}")
        print(f"Freeze encoder: {args.freeze_encoder}")
        print(f"Encoder learning rate factor: {args.encoder_lr_factor}")

    print("--- Multi-Task Configuration ---")
    print(f"Model Name: {model_name}")
    print(f"Pain Classes: {num_pain_classes}, Stimulus Classes: {num_stimulus_classes}")
    print(f"Syracuse Features: {syracuse_feature_dir}, BioVid Features: {biovid_feature_dir}")
    if shoulder_pain_feature_dir and args.shoulder_pain_data_path:
        print(f"ShoulderPain Features: {shoulder_pain_feature_dir}")
    print(f"Temporal Reduction: {temporal_reduction}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Label Smoothing: {label_smoothing}")
    print(f"Use Class Weights: {use_class_weights}")
    print(f"Balance Pain Classes: {balance_pain_classes}")
    print(f"Loss Weights (Pain/Stim): {pain_loss_weight}/{stim_loss_weight}")
    print(f"Balance Sources: {balance_sources}, Balance Stimulus Classes: {balance_stimulus_classes}")
    print(f"Encoder Hidden Dims: {encoder_hidden_dims}")
    print(f"Use Distance Penalty: {use_distance_penalty}")
    print(f"Focal Gamma: {focal_gamma}")
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
        # ShoulderPain Params (Optional)
        shoulder_pain_root_dir=args.shoulder_pain_data_path if hasattr(args, 'shoulder_pain_data_path') else None,
        shoulder_pain_feature_dir=shoulder_pain_feature_dir,
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

    # Calculate class weights if enabled
    class_weights = None
    if use_class_weights and not args.predict_only:
        print("Calculating class weights for Syracuse pain levels...")
        dm.setup('fit')  # Ensure train dataset is ready
        # Get the wrapped Syracuse train set from the training dataset
        # The MultiTaskDataModule's train_dataset is a ConcatDataset with Syracuse as first item
        wrapped_syracuse_train = dm.train_dataset.datasets[0]
        
        # Extract all labels
        train_labels = []
        # Use a temporary DataLoader to efficiently extract labels
        temp_loader = DataLoader(wrapped_syracuse_train, batch_size=64, shuffle=False)
        for batch in temp_loader:
            # The MultiTaskWrapper returns (features, pain_labels, stim_labels)
            _, labels, _ = batch
            valid_mask = labels != -1  # Filter out invalid labels if any
            if valid_mask.any():
                train_labels.append(labels[valid_mask])
        
        if train_labels:
            train_labels = torch.cat(train_labels).numpy()
            
            # Calculate class frequencies
            class_counts = np.bincount(train_labels, minlength=num_pain_classes)
            print(f"Class distribution: {class_counts}")
            
            # Calculate inverse frequency weights (with small epsilon to avoid division by zero)
            weights = 1.0 / (class_counts + 1e-6)
            
            # Normalize weights to maintain loss scale
            weights = weights / weights.mean()
            
            print(f"Class weights: {np.round(weights, 3)}")
            class_weights = torch.tensor(weights, dtype=torch.float32)
        else:
            print("Warning: Could not extract valid labels for class weight calculation.")

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
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        pain_loss_weight=pain_loss_weight,
        stim_loss_weight=stim_loss_weight,
        encoder_hidden_dims=encoder_hidden_dims,
        distributed=(args.n_gpus > 1), # Pass the distributed flag
        use_distance_penalty=use_distance_penalty,
        focal_gamma=focal_gamma,
        freeze_stimulus_head=args.freeze_stimulus_head if using_pretrained else False,
        freeze_encoder=args.freeze_encoder if using_pretrained else False,
        encoder_lr_factor=args.encoder_lr_factor if using_pretrained else 1.0
        # optimizer_name can be added to config/args if needed
    )
    
    # Load pretrained weights if provided
    if using_pretrained:
        model.load_from_pretrained(args.pretrained_checkpoint)
    
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
    Seed.set(42)  # Fixed seed for reproducibility
    
    # --- Configuration --- 
    model_name = config.get("model_name", "multitask_cv_run")
    num_pain_classes = config["num_pain_classes"]
    num_stimulus_classes = config["num_stimulus_classes"]
    syracuse_feature_dir = config["syracuse_feature_dir"]
    biovid_feature_dir = config.get("biovid_feature_dir", None)
    shoulder_pain_feature_dir = config.get("shoulder_pain_feature_dir", None)
    temporal_reduction = config.get("temporal_reduction", "mean")
    learning_rate = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.0)
    label_smoothing = config.get("label_smoothing", 0.0)
    use_class_weights = config.get("use_class_weights", False)
    balance_pain_classes = config.get("balance_pain_classes", False)
    pain_loss_weight = config.get("pain_loss_weight", 1.0)
    stim_loss_weight = config.get("stim_loss_weight", 1.0)
    balance_sources = config.get("balance_sources", False)
    balance_stimulus_classes = config.get("balance_stimulus_classes", False)
    encoder_hidden_dims = config.get("encoder_hidden_dims", None)
    # New coral_loss parameters
    use_distance_penalty = config.get("use_distance_penalty", False)
    focal_gamma = config.get("focal_gamma", None)
    # Early stopping parameters
    patience = config.get("patience", 200)  # Higher since we're using a combined dataset
    # Default to QWK for model selection, can be changed through config
    monitor_metric = config.get("monitor_metric", "val_pain_QWK")
    monitor_mode = config.get("monitor_mode", "max")  # Default to max for QWK
    
    # Stimulus Weight Scheduler
    use_stim_weight_scheduler = config.get("use_stim_weight_scheduler", False)
    initial_stim_weight = config.get("initial_stim_weight", 5.0)
    final_stim_weight = config.get("final_stim_weight", 1.0)
    stim_weight_decay_epochs = config.get("stim_weight_decay_epochs", 50)
    stim_weight_sched_type = config.get("stim_weight_sched_type", "cosine")
    
    # Print pretrained model information if provided
    using_pretrained = args.pretrained_checkpoint is not None
    if using_pretrained:
        print(f"Using pretrained model from: {args.pretrained_checkpoint}")
        print(f"Freeze stimulus head: {args.freeze_stimulus_head}")
        print(f"Freeze encoder: {args.freeze_encoder}")
        print(f"Encoder learning rate factor: {args.encoder_lr_factor}")

    # --- 1. Load Syracuse metadata first --- 
    # This must be done before anything else to get video_id_labels
    print("  Loading Syracuse metadata...")
    try:
        # Instantiate SyracuseDataModule to use its metadata loading logic
        syracuse_dm_for_meta = SyracuseDataModule(
            root_dir=args.syracuse_data_path,
            task='multiclass',
            num_classes=num_pain_classes,
            batch_size=1,  # Dummy value
            feature_dir=syracuse_feature_dir,
            marlin_base_dir=args.syracuse_marlin_base_dir,
            temporal_reduction=temporal_reduction,
            num_workers=0  # Dummy value
        )
        # Load metadata
        syracuse_dm_for_meta.setup(stage=None)
        
        # Get important metadata variables
        all_syracuse_metadata = syracuse_dm_for_meta.all_metadata
        original_clips = syracuse_dm_for_meta.original_clips
        augmented_clips = syracuse_dm_for_meta.augmented_clips
        video_id_labels = syracuse_dm_for_meta.video_id_labels  # Now correctly defined
        
        print(f"  Found {len(video_id_labels)} unique Syracuse videos with labels")
        print(f"  Dataset has {len(original_clips)} original clips and {len(augmented_clips)} augmented clips")
    except Exception as e:
        print(f"Error loading Syracuse metadata: {e}. Cannot proceed with CV.")
        return

    # --- 2. Load Full BioVid Training Data (once) --- 
    print("  Loading full BioVid training data...")
    try:
        full_biovid_train_set = BioVidLP(
             root_dir=args.biovid_data_path, 
             feature_dir=biovid_feature_dir,
             split='train', 
             task='multiclass', 
             num_classes=num_stimulus_classes,
             temporal_reduction=temporal_reduction,
             data_ratio=1.0, 
             take_num=None
        )
        print(f"    Full BioVid train size: {len(full_biovid_train_set)}")
    except Exception as e:
         print(f"Error loading BioVid training data: {e}. Cannot proceed with CV.")
         return
         
    # --- 3. Load ShoulderPain Training Data (once, if enabled) --- 
    full_shoulder_pain_train_set = None
    
    # Debug the shoulder pain parameters
    print(f"SHOULDER_PAIN PATH DEBUG:")
    print(f"  - hasattr(args, 'shoulder_pain_data_path'): {hasattr(args, 'shoulder_pain_data_path')}")
    if hasattr(args, 'shoulder_pain_data_path'):
        print(f"  - args.shoulder_pain_data_path: '{args.shoulder_pain_data_path}'")
        print(f"  - args.shoulder_pain_data_path is None: {args.shoulder_pain_data_path is None}")
        if args.shoulder_pain_data_path:
            print(f"  - Path exists: {os.path.exists(args.shoulder_pain_data_path)}")
    print(f"  - shoulder_pain_feature_dir: '{shoulder_pain_feature_dir}'")
    print(f"  - shoulder_pain_feature_dir is None: {shoulder_pain_feature_dir is None}")
    
    # Check what creates the use_shoulder_pain flag
    use_shoulder_pain = shoulder_pain_feature_dir is not None and hasattr(args, 'shoulder_pain_data_path') and args.shoulder_pain_data_path is not None
    print(f"  - Combined condition (use_shoulder_pain): {use_shoulder_pain}")
    
    # Additional directory structure checks
    if hasattr(args, 'shoulder_pain_data_path') and args.shoulder_pain_data_path and shoulder_pain_feature_dir:
        # Check if the full feature directory exists
        feature_dir_path = os.path.join(args.shoulder_pain_data_path, shoulder_pain_feature_dir)
        print(f"  - Full feature directory path: '{feature_dir_path}'")
        print(f"  - Feature directory exists: {os.path.exists(feature_dir_path)}")
        
        # Check if metadata and split files exist
        print(f"  - Checking required files:")
        metadata_path = os.path.join(args.shoulder_pain_data_path, 'shoulder_pain_info.json')
        train_split_path = os.path.join(args.shoulder_pain_data_path, 'train.txt')
        print(f"    - shoulder_pain_info.json exists: {os.path.exists(metadata_path)}")
        print(f"    - train.txt exists: {os.path.exists(train_split_path)}")
        
        # Check if any feature files exist
        if os.path.exists(feature_dir_path):
            try:
                feature_files = os.listdir(feature_dir_path)
                npy_files = [f for f in feature_files if f.endswith('.npy')]
                print(f"    - Number of .npy files in feature directory: {len(npy_files)}")
                if npy_files:
                    print(f"    - Example feature files: {npy_files[:3]}")
            except Exception as e:
                print(f"    - Error listing feature files: {e}")
    
    # Uncomment for testing: Override use_shoulder_pain flag if needed
    # use_shoulder_pain = True
    # print("  - TESTING: Forcing use_shoulder_pain to True for testing")
    
    if use_shoulder_pain:
        print("  Loading full ShoulderPain training data...")
        try:
            print(f"  SHOULDER_PAIN DEBUG: Creating dataset with params:")
            print(f"    - root_dir: {args.shoulder_pain_data_path}")
            print(f"    - feature_dir: {shoulder_pain_feature_dir}")
            print(f"    - num_classes: {num_pain_classes}")
            
            # First check if the required files exist
            metadata_exists = os.path.exists(os.path.join(args.shoulder_pain_data_path, 'shoulder_pain_info.json'))
            train_file_exists = os.path.exists(os.path.join(args.shoulder_pain_data_path, 'train.txt'))
            feature_dir_exists = os.path.exists(os.path.join(args.shoulder_pain_data_path, shoulder_pain_feature_dir))
            
            if not metadata_exists:
                raise FileNotFoundError(f"shoulder_pain_info.json not found in {args.shoulder_pain_data_path}")
            if not train_file_exists:
                raise FileNotFoundError(f"train.txt not found in {args.shoulder_pain_data_path}")
            if not feature_dir_exists:
                raise FileNotFoundError(f"Feature directory {shoulder_pain_feature_dir} not found in {args.shoulder_pain_data_path}")
                
            # Now try to create the dataset
            full_shoulder_pain_train_set = ShoulderPainLP(
                root_dir=args.shoulder_pain_data_path,
                feature_dir=shoulder_pain_feature_dir,
                split='train',
                task='multiclass',
                num_classes=num_pain_classes,
                temporal_reduction=temporal_reduction,
                data_ratio=1.0,
                take_num=None
            )
            print(f"    Full ShoulderPain train size: {len(full_shoulder_pain_train_set)}")
            
            # Print class distribution
            class_distribution = full_shoulder_pain_train_set.get_class_distribution()
            print(f"    ShoulderPain Class Distribution: {class_distribution}")
        except FileNotFoundError as e:
            print(f"  ERROR: ShoulderPain directory structure issue: {e}")
            print(f"  Continuing without ShoulderPain dataset.")
            use_shoulder_pain = False
            full_shoulder_pain_train_set = None
        except ImportError as e:
            print(f"  ERROR: ShoulderPainLP class not found: {e}")
            print(f"  Make sure dataset/shoulder_pain.py is properly implemented and imported.")
            use_shoulder_pain = False
            full_shoulder_pain_train_set = None
        except Exception as e:
            print(f"  ERROR: Loading ShoulderPain training data: {str(e)}")
            print(f"  Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"  Continuing without ShoulderPain dataset.")
            use_shoulder_pain = False
            full_shoulder_pain_train_set = None

    # Now video_id_labels is properly defined before it's used
    unique_video_ids = sorted(list(video_id_labels.keys()))
    video_labels_for_stratify = [video_id_labels[vid] for vid in unique_video_ids]
    if len(unique_video_ids) < args.cv_folds:
        raise ValueError(f"Number of unique Syracuse videos ({len(unique_video_ids)}) is less than the number of folds ({args.cv_folds}).")
    
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    fold_checkpoint_paths = [] # Stores best ckpt path for each fold
    fold_val_filenames = [] # Stores list of val filenames for each fold
    
    # --- Cross-Validation Loop --- 
    for fold_idx, (train_vid_idx_indices, val_vid_idx_indices) in enumerate(skf.split(unique_video_ids, video_labels_for_stratify)):
        print(f"\n===== Starting Fold {fold_idx + 1}/{args.cv_folds} =====")
        
        # Map indices back to video IDs
        fold_train_video_ids = [unique_video_ids[i] for i in train_vid_idx_indices]
        fold_val_video_ids = [unique_video_ids[i] for i in val_vid_idx_indices]
        
        # DEBUGGING - Print fold data statistics
        print(f"\nFOLD DEBUG: Total unique video IDs: {len(unique_video_ids)}")
        print(f"FOLD DEBUG: Train set should be ~{(args.cv_folds-1)/args.cv_folds:.1%} of videos")
        print(f"FOLD DEBUG: Actual train/val video split: {len(fold_train_video_ids)}/{len(fold_val_video_ids)} = {len(fold_train_video_ids)/len(unique_video_ids):.1%}/{len(fold_val_video_ids)/len(unique_video_ids):.1%}")
        
        print(f"  Train Video IDs: {len(fold_train_video_ids)}, Val Video IDs: {len(fold_val_video_ids)}")

        # Filter Syracuse filenames for this fold based on clip_data['video_id']
        fold_train_video_ids_set = set(fold_train_video_ids)
        fold_val_video_ids_set = set(fold_val_video_ids)

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
        
        # Calculate class weights if enabled (for loss weighting)
        fold_class_weights = None
        if use_class_weights:
            print(f"  Calculating class weights for loss function (fold {fold_idx + 1})...")
            # Get all labels from training set
            train_labels = []
            # Use a temporary DataLoader with batch_size=64 to efficiently extract labels
            temp_loader = DataLoader(syracuse_train_set, batch_size=64, shuffle=False)
            for batch in temp_loader:
                _, labels = batch
                # Filter out -1 labels (invalid)
                valid_mask = labels != -1
                if valid_mask.any():
                    train_labels.append(labels[valid_mask])
            
            if train_labels:
                train_labels = torch.cat(train_labels).numpy()
                
                # Calculate class frequencies
                class_counts = np.bincount(train_labels, minlength=num_pain_classes)
                print(f"  Class distribution for loss weighting: {class_counts}")
                
                # Calculate inverse frequency weights (with small epsilon to avoid division by zero)
                class_weights = 1.0 / (class_counts + 1e-6)
                
                # Normalize weights to maintain loss scale
                class_weights = class_weights / class_weights.mean()
                
                print(f"  Loss class weights: {np.round(class_weights, 3)}")
                fold_class_weights = torch.tensor(class_weights, dtype=torch.float32)
            else:
                print(f"  WARNING: No valid labels found for loss weighting. Using uniform weights.")
        
        # Add BioVid data to the training set (as this is a multi-task model)
        wrapped_biovid_train = MultiTaskWrapper(full_biovid_train_set, 'stimulus')
        
        # Add ShoulderPain data to the training set if available
        wrapped_shoulder_pain_train = None
        if use_shoulder_pain and full_shoulder_pain_train_set is not None and len(full_shoulder_pain_train_set) > 0:
            wrapped_shoulder_pain_train = MultiTaskWrapper(full_shoulder_pain_train_set, 'pain')
            print(f"  Including ShoulderPain dataset with {len(wrapped_shoulder_pain_train)} samples")
            
            # Add more detailed debug info about ShoulderPain dataset
            print(f"  SHOULDER_PAIN DEBUG: Dataset loaded successfully")
            print(f"  SHOULDER_PAIN DEBUG: Original dataset size: {len(full_shoulder_pain_train_set)}")
            print(f"  SHOULDER_PAIN DEBUG: Wrapped dataset size: {len(wrapped_shoulder_pain_train)}")
        else:
            print(f"  SHOULDER_PAIN DEBUG: Dataset NOT included in training")
            if not use_shoulder_pain:
                print(f"  SHOULDER_PAIN DEBUG: Reason - use_shoulder_pain flag is False")
            elif full_shoulder_pain_train_set is None:
                print(f"  SHOULDER_PAIN DEBUG: Reason - dataset object is None")
            elif len(full_shoulder_pain_train_set) == 0:
                print(f"  SHOULDER_PAIN DEBUG: Reason - dataset is empty (0 samples)")
        
        # Combine Syracuse and BioVid for training
        print(f"  Creating combined training dataset:")
        print(f"    - Syracuse train: {len(wrapped_syracuse_train)} samples")
        print(f"    - BioVid train: {len(wrapped_biovid_train)} samples")
        
        # Start with Syracuse and BioVid datasets
        train_datasets = [wrapped_syracuse_train, wrapped_biovid_train]
        
        # Add ShoulderPain if available
        if wrapped_shoulder_pain_train is not None:
            train_datasets.append(wrapped_shoulder_pain_train)
            print(f"    - ShoulderPain train: {len(wrapped_shoulder_pain_train)} samples")
        
        combined_train_dataset = ConcatDataset(train_datasets)
        print(f"    - Combined training set: {len(combined_train_dataset)} samples")
        
        # Add detailed debug information about ConcatDataset internals
        print(f"DATASET INTERNALS DEBUG:")
        if hasattr(combined_train_dataset, 'cumulative_sizes'):
            cum_sizes = combined_train_dataset.cumulative_sizes
            print(f"  - Cumulative sizes: {cum_sizes}")
            dataset_sizes = [cum_sizes[0]] + [cum_sizes[i] - cum_sizes[i-1] for i in range(1, len(cum_sizes))]
            print(f"  - Individual dataset sizes: {dataset_sizes}")
            print(f"  - Dataset start indices:")
            start_indices = [0] + [cum_sizes[i] for i in range(len(cum_sizes)-1)]
            for i, start_idx in enumerate(start_indices):
                end_idx = cum_sizes[i] - 1
                dataset_type = type(train_datasets[i]).__name__
                print(f"    Dataset {i+1} ({dataset_type}): indices {start_idx} to {end_idx} (size: {end_idx-start_idx+1})")
        else:
            print(f"  - combined_train_dataset does not have cumulative_sizes attribute")
            print(f"  - Number of datasets: {len(train_datasets)}")
            for i, ds in enumerate(train_datasets):
                print(f"    Dataset {i+1}: type={type(ds).__name__}, size={len(ds)}")
        
        # Calculate new expected batches with combined dataset
        combined_expected_batches = len(combined_train_dataset) // args.batch_size
        if len(combined_train_dataset) % args.batch_size != 0:
            combined_expected_batches += 1
        print(f"FOLD DEBUG: Expected batches with combined data: {combined_expected_batches}")
        
        # Check dataset composition
        print(f"DATASET DEBUG: Combined dataset contains {len(combined_train_dataset)} samples")
        print(f"DATASET DEBUG: Number of datasets in combined dataset: {len(train_datasets)}")
        for i, dataset in enumerate(train_datasets):
            print(f"DATASET DEBUG: Dataset {i+1} type: {type(dataset).__name__}, size: {len(dataset)}")
        
        # Verify total makes sense
        expected_total = len(wrapped_syracuse_train) + len(wrapped_biovid_train)
        if wrapped_shoulder_pain_train is not None:
            expected_total += len(wrapped_shoulder_pain_train)
        print(f"DATASET DEBUG: Expected combined size: {expected_total}, Actual size: {len(combined_train_dataset)}")
        if expected_total != len(combined_train_dataset):
            print(f"DATASET DEBUG: WARNING - Size mismatch! Dataset might not be properly combined!")
        
        # Create validation dataloader
        fold_val_loader = DataLoader(
            wrapped_syracuse_val, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )

        # For training - use WeightedRandomSampler for Syracuse if balance_pain_classes is enabled
        if balance_pain_classes:
            print(f"  Using weighted random sampler for Syracuse pain classes...")
            
            # Extract pain labels from Syracuse dataset
            syracuse_labels = []
            temp_loader = DataLoader(syracuse_train_set, batch_size=64, shuffle=False)
            for batch in temp_loader:
                _, labels = batch  # SyracuseLP returns (features, labels)
                valid_mask = labels != -1
                if valid_mask.any():
                    syracuse_labels.append(labels[valid_mask])
            
            if len(syracuse_labels) == 0:
                print("  WARNING: No valid labels found in Syracuse train set. Using uniform weights.")
                # Default to regular dataloader
                fold_train_loader = DataLoader(
                    combined_train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=True if args.num_workers > 0 else False
                )
            else:
                # Convert labels to numpy array for bincount
                syracuse_labels = torch.cat(syracuse_labels).numpy()
                
                # Calculate class frequencies and weights
                class_counts = np.bincount(syracuse_labels, minlength=num_pain_classes)
                print(f"  Syracuse sampling class distribution: {class_counts}")
                
                # Check for empty classes
                if np.any(class_counts == 0):
                    print(f"  WARNING: Some classes have zero samples - using frequency-based weights for sampling")
                    # Still use inverse frequency but don't try to make perfectly uniform
                    sample_class_weights = 1.0 / (class_counts + 1e-6)
                    sample_class_weights = sample_class_weights / sample_class_weights.sum()  # Normalize to sum to 1
                else:
                    # Calculate weights to create perfectly balanced classes
                    sample_class_weights = 1.0 / class_counts
                    sample_class_weights = sample_class_weights / sample_class_weights.sum()  # Normalize to sum to 1
                    
                print(f"  Syracuse sampling class weights: {np.round(sample_class_weights, 3)}")
                
                # We need to assign a weight to each sample in the combined dataset:
                # 1. Syracuse samples get weights based on their class
                # 2. BioVid samples get default weight to maintain dataset balance
                
                # Get weights for all Syracuse samples
                all_weights = []
                
                # First get all Syracuse sample labels
                all_syracuse_labels = []
                for batch in DataLoader(wrapped_syracuse_train, batch_size=128, shuffle=False):
                    # Parse MultiTaskWrapper output: (features, pain_labels, stim_labels)
                    _, pain_labels, _ = batch
                    all_syracuse_labels.append(pain_labels)
                
                # Convert to tensor and create weight array
                all_syracuse_labels = torch.cat(all_syracuse_labels)
                
                # Create sample weights array for Syracuse
                syracuse_weights = torch.ones_like(all_syracuse_labels, dtype=torch.float)
                # Only valid labels get class-based weights
                valid_mask = all_syracuse_labels != -1
                for i, (is_valid, label) in enumerate(zip(valid_mask, all_syracuse_labels)):
                    if is_valid:
                        syracuse_weights[i] = sample_class_weights[label]
                        
                # Convert to numpy
                syracuse_weights = syracuse_weights.numpy()
                
                # BioVid samples get average weight to maintain balance between datasets
                biovid_weight = np.mean(syracuse_weights)
                biovid_weights = np.ones(len(wrapped_biovid_train)) * biovid_weight
                
                # Combine weights in the same order as datasets in combined_train_dataset
                combined_weights = np.concatenate([syracuse_weights, biovid_weights])
                
                # Add ShoulderPain sample weights if available
                if wrapped_shoulder_pain_train is not None:
                    # ShoulderPain samples are pain samples too, so they get weights just like Syracuse
                    # Extract all ShoulderPain labels to calculate weights
                    all_shoulder_pain_labels = []
                    for batch in DataLoader(wrapped_shoulder_pain_train, batch_size=128, shuffle=False):
                        # Parse MultiTaskWrapper output: (features, pain_labels, stim_labels)
                        _, pain_labels, _ = batch
                        all_shoulder_pain_labels.append(pain_labels)
                    
                    # Convert to tensor
                    if all_shoulder_pain_labels:
                        all_shoulder_pain_labels = torch.cat(all_shoulder_pain_labels)
                        
                        # Create sample weights array for ShoulderPain
                        shoulder_pain_weights = torch.ones_like(all_shoulder_pain_labels, dtype=torch.float)
                        # Only valid labels get class-based weights
                        valid_mask = all_shoulder_pain_labels != -1
                        for i, (is_valid, label) in enumerate(zip(valid_mask, all_shoulder_pain_labels)):
                            if is_valid:
                                shoulder_pain_weights[i] = sample_class_weights[label]
                        
                        # Convert to numpy and combine with existing weights
                        shoulder_pain_weights = shoulder_pain_weights.numpy()
                        combined_weights = np.concatenate([syracuse_weights, biovid_weights, shoulder_pain_weights])
                        
                        print(f"  SHOULDER_PAIN DEBUG: Added weights for {len(shoulder_pain_weights)} ShoulderPain samples")
                        print(f"  SHOULDER_PAIN DEBUG: Mean ShoulderPain sample weight: {np.mean(shoulder_pain_weights):.4f}")
                        print(f"  SHOULDER_PAIN DEBUG: Mean Syracuse sample weight: {np.mean(syracuse_weights):.4f}")
                        print(f"  SHOULDER_PAIN DEBUG: Mean BioVid sample weight: {np.mean(biovid_weights):.4f}")
                    else:
                        print(f"  SHOULDER_PAIN DEBUG: No labels extracted from ShoulderPain dataset")
                
                # Verify the weights array matches the combined dataset size
                if len(combined_weights) != len(combined_train_dataset):
                    print(f"  WARNING: Weight array size ({len(combined_weights)}) doesn't match dataset size ({len(combined_train_dataset)})")
                    print(f"  WEIGHT DEBUG: Syracuse weights: {len(syracuse_weights)}, BioVid weights: {len(biovid_weights)}")
                    if wrapped_shoulder_pain_train is not None and 'shoulder_pain_weights' in locals():
                        print(f"  WEIGHT DEBUG: ShoulderPain weights: {len(shoulder_pain_weights)}")
                
                # Create a weighted sampler for the combined dataset
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=torch.from_numpy(combined_weights).float(),
                    num_samples=len(combined_train_dataset),
                    replacement=True
                )
                
                # Create dataloader with the weighted sampler
                fold_train_loader = DataLoader(
                    combined_train_dataset,
                    batch_size=args.batch_size,
                    sampler=sampler,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=True if args.num_workers > 0 else False
                )
                
                print(f"  Created balanced DataLoader for fold {fold_idx + 1} with {len(fold_train_loader)} batches")
                print(f"  SAMPLING DEBUG: Total samples: {len(combined_train_dataset)}, Weights array size: {len(combined_weights)}")
                print(f"  SAMPLING DEBUG: Dataset composition: Syracuse={len(wrapped_syracuse_train)}, BioVid={len(wrapped_biovid_train)}")
                if wrapped_shoulder_pain_train is not None:
                    print(f"  SAMPLING DEBUG: ShoulderPain={len(wrapped_shoulder_pain_train)}")
                print(f"  SAMPLING DEBUG: Expected batches: {combined_expected_batches}, Created batches: {len(fold_train_loader)}")
                
        else:
            # Standard DataLoader with combined dataset (default behavior)
            fold_train_loader = DataLoader(
                combined_train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False
            )
            print(f"  Created standard DataLoader for fold {fold_idx + 1} with {len(fold_train_loader)} batches")
            
            # Add debug prints for the standard DataLoader case
            print(f"  SAMPLING DEBUG: Using standard DataLoader (no class balancing)")
            print(f"  SAMPLING DEBUG: Total samples: {len(combined_train_dataset)}")
            print(f"  SAMPLING DEBUG: Dataset composition: Syracuse={len(wrapped_syracuse_train)}, BioVid={len(wrapped_biovid_train)}")
            if wrapped_shoulder_pain_train is not None:
                print(f"  SAMPLING DEBUG: ShoulderPain={len(wrapped_shoulder_pain_train)}")
            print(f"  SAMPLING DEBUG: Expected batches: {combined_expected_batches}, Created batches: {len(fold_train_loader)}")
            
            # Check if the batch calculation makes sense
            total_samples = len(combined_train_dataset)
            expected_batch_count = total_samples // args.batch_size
            if total_samples % args.batch_size != 0:
                expected_batch_count += 1
            if expected_batch_count != len(fold_train_loader):
                print(f"  SAMPLING DEBUG: WARNING - Batch count mismatch! Expected: {expected_batch_count}, Actual: {len(fold_train_loader)}")

        # --- Configure Model & Trainer for the Fold ---
        # Determine input dimension from a sample feature file
        sample_feature_path = os.path.join(args.syracuse_data_path, syracuse_feature_dir, 
                                         syracuse_train_filenames[0])
        if not os.path.exists(sample_feature_path):
            print(f"Error: Could not find sample feature file at {sample_feature_path}")
            return
            
        sample_feature = np.load(sample_feature_path)
        if temporal_reduction == "none":
            print(f"Warning: temporal_reduction='none' specified. "
                  f"Raw feature shape: {sample_feature.shape}")
            # Handle sequence data differently if needed
            input_dim = 768  # Default fallback
        else:
            # Apply temporal reduction
            if temporal_reduction == "mean":
                sample_feature = np.mean(sample_feature, axis=0)
            elif temporal_reduction == "max":
                sample_feature = np.max(sample_feature, axis=0)
            elif temporal_reduction == "min":
                sample_feature = np.min(sample_feature, axis=0)
                
            input_dim = sample_feature.shape[0]
            
        print(f"Using input dimension: {input_dim}")
        
        model = MultiTaskCoralClassifier(
            input_dim=input_dim,
            num_pain_classes=num_pain_classes,
            num_stimulus_classes=num_stimulus_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            class_weights=fold_class_weights,
            encoder_hidden_dims=encoder_hidden_dims,
            pain_loss_weight=pain_loss_weight,
            stim_loss_weight=stim_loss_weight,
            use_distance_penalty=use_distance_penalty,
            focal_gamma=focal_gamma,
            freeze_stimulus_head=args.freeze_stimulus_head if using_pretrained else False,
            freeze_encoder=args.freeze_encoder if using_pretrained else False,
            encoder_lr_factor=args.encoder_lr_factor if using_pretrained else 1.0
        )
        
        # Load pretrained weights if provided
        if using_pretrained:
            model.load_from_pretrained(args.pretrained_checkpoint)
        
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
        
        # Create a stimulus weight scheduler callback if enabled
        callbacks = [checkpoint_callback, early_stop_callback, LrLogger(), SystemStatsLogger()]
        if use_stim_weight_scheduler:
            stim_weight_scheduler = StimWeightSchedulerCallback(
                initial_weight=initial_stim_weight,
                final_weight=final_stim_weight,
                decay_epochs=stim_weight_decay_epochs,
                scheduler_type=stim_weight_sched_type
            )
            callbacks.append(stim_weight_scheduler)
            print(f"  Added stimulus weight scheduler callback")
        
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=1, # Run each fold on a single device for simplicity
            logger=pl.loggers.TensorBoardLogger("logs", name=f"{model_name}/fold_{fold_idx}"),
            callbacks=callbacks,
            precision=args.precision,
            log_every_n_steps=50, # Log less frequently during CV
            enable_progress_bar=True, # Show progress bar per fold
            # deterministic=True, # Could add for reproducibility if needed
            # benchmark=True, # Add if using GPU and input size is constant
            num_sanity_val_steps=0 # Disable sanity check for CV folds
        )
        
        # --- Train the Fold ---
        print(f"  Starting training for fold {fold_idx + 1}...")
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
    if len(fold_checkpoint_paths) != args.cv_folds or len(fold_val_filenames) != args.cv_folds:
        print(f"Warning: Mismatch in number of checkpoints ({len(fold_checkpoint_paths)}) or validation sets ({len(fold_val_filenames)}) vs folds ({args.cv_folds}).")

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

    print(f"\n--- Aggregated Cross-Validation Results ({len(valid_fold_metrics)}/{args.cv_folds} Folds) ---")

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
        'n_splits': args.cv_folds,
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
        label_smoothing = config.get("label_smoothing", 0.0)  # For loading checkpoint
        weight_decay = config.get("weight_decay", 0.0)  # For loading checkpoint
        use_distance_penalty = config.get("use_distance_penalty", False)  # New parameter
        focal_gamma = config.get("focal_gamma", None)  # New parameter

        # Load the model from the checkpoint
        model = MultiTaskCoralClassifier.load_from_checkpoint(
            checkpoint_path,
            input_dim=input_dim,
            num_pain_classes=num_pain_classes,
            num_stimulus_classes=num_stimulus_classes,
            learning_rate=0, # Not needed for evaluation
            encoder_hidden_dims=encoder_hidden_dims,
            pain_loss_weight=pain_loss_weight,
            stim_loss_weight=stim_loss_weight,
            label_smoothing=label_smoothing,
            weight_decay=weight_decay,
            use_distance_penalty=use_distance_penalty,
            focal_gamma=focal_gamma,
            freeze_stimulus_head=args.freeze_stimulus_head if using_pretrained else False,
            freeze_encoder=args.freeze_encoder if using_pretrained else False
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
    parser.add_argument("--shoulder_pain_data_path", type=str, default=None,
                        help="Root directory for ShoulderPain features (optional).")
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
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='Path to pretrained BioVid model checkpoint for fine-tuning')
    parser.add_argument('--freeze_stimulus_head', action='store_true',
                        help='Freeze the stimulus head when using a pretrained model')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze the shared encoder when using a pretrained model')
    parser.add_argument('--encoder_lr_factor', type=float, default=0.1,
                        help='Learning rate factor for the encoder when using a pretrained model (default: 0.1)')

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
 