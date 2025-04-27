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
import os
import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm

torch.set_float32_matmul_precision('high')
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import custom modules
from dataset.multitask import MultiTaskDataModule
from model.multi_task_coral import MultiTaskCoralClassifier
from marlin_pytorch.util import read_yaml # Assuming this is available
from util.lr_logger import LrLogger
from util.seed import Seed
from util.system_stats_logger import SystemStatsLogger


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
    ckpt_monitor = "val_pain_mae" 
    ckpt_mode = "min" # Lower MAE is better
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
    early_stop_patience = 20 # Example patience
    print(f"Using EarlyStopping with patience={early_stop_patience}, monitoring '{ckpt_monitor}' ({ckpt_mode})")
    early_stop_callback = EarlyStopping(
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        patience=early_stop_patience,
        verbose=True
    )

    trainer = Trainer(
        log_every_n_steps=10, # Log less frequently than every step
        devices=n_gpus,
        accelerator=accelerator,
        strategy=strategy,
        max_epochs=max_epochs,
        precision=precision,
        logger=True, # Use default TensorBoardLogger
        callbacks=[
            # ckpt_callback, # Temporarily disable for debugging
            # early_stop_callback, # Temporarily disable for debugging
            LrLogger(),
            SystemStatsLogger()
        ],
        benchmark=True if n_gpus > 0 else False
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
        # test() automatically loads the best checkpoint
        test_results = trainer.test(model, datamodule=dm, ckpt_path=best_ckpt_path)
        print("Test Set Results (Pain Task Metrics):")
        print(json.dumps(test_results, indent=4))
        
        # Save test results
        results_filename = os.path.join(checkpoint_dir, f"{model_name}_test_results.json")
        try:
            with open(results_filename, 'w') as f:
                json.dump(test_results, f, indent=4)
            print(f"Test results saved to: {results_filename}")
        except Exception as e:
            print(f"Error saving test results: {e}")
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

    args = parser.parse_args()

    # --- Run Evaluation --- 
    try:
        config = read_yaml(args.config)
        run_multitask_evaluation(args, config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
    except KeyError as e:
        print(f"Error: Missing required key in config file {args.config}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Optionally re-raise for debugging
        # raise e
 