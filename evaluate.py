"""
MARLIN Model Evaluation Script

This script handles training and evaluation for different datasets using MARLIN features
or fine-tuning MARLIN models. It currently supports:

1. BioVid Dataset (Linear Probing or Fine-tuning with Classifier)
2. Syracuse Dataset (Linear Probing with CORAL Ordinal Classifier - LightningMLP)

Workflow:
----------
1. Reads a configuration YAML file specified by `--config`.
2. Determines the dataset ('biovid' or 'syracuse') from the config.
3. Initializes the appropriate DataModule and Model based on the dataset and config.
4. Trains the model (unless `--predict_only` is used).
5. Evaluates the best checkpoint from training on the test set.
6. Prints evaluation metrics and optionally saves predictions.

Syracuse Dataset with CORAL (`LightningMLP`) Usage:
--------------------------------------------------

Command-line Arguments:

*   `--config PATH`: **Required**. Path to the YAML configuration file.
*   `--data_path PATH`: **Required**. Root directory of the Syracuse dataset. This directory
    should contain the feature subdirectory specified in the config's `backbone` field.
*   `--marlin_base_dir PATH`: **Required**. Path to the directory containing the
    `clips_json.json` metadata file used by `MarlinFeatures`.
*   `--n_gpus N`: (Optional) Number of GPUs (default: 1). Use 0 for CPU.
*   `--batch_size N`: (Optional) Batch size (default: 32).
*   `--num_workers N`: (Optional) Dataloader workers (default: 4).
*   `--epochs N`: (Optional) Max training epochs (default: 100).
*   `--precision P`: (Optional) Training precision ('32', '16', 'bf16') (default: '32').
*   `--val_split_ratio F`: (Optional) Ratio of videos for validation (default: 0.15).
*   `--test_split_ratio F`: (Optional) Ratio of videos for testing (default: 0.15).
*   `--predict_only`: (Optional) Flag to skip metric calculation and only save predictions.
    Requires `--output_path`.
*   `--output_path PATH`: (Optional) Path to save prediction CSV file when using
    `--predict_only` (defaults to 'syracuse_predictions.csv').

Configuration YAML File Keys (`--config`):

*   `dataset: syracuse` (**Required**)
*   `model_name: <your_run_name>` (**Required**): Name for checkpoints/logs.
*   `backbone: <feature_dir_name>` (**Required**): Name of the feature directory
    (relative to `--data_path`) containing `.npy` files.
    Example: `marlin_vit_small_patch16_224`
*   `temporal_reduction: <mean|max|min>` (**Required**): How to aggregate the
    (4, 768) features along the time dimension. The `LightningMLP` model
    expects the resulting (768,) feature vector per sample.
*   `learning_rate: <float>` (**Required**): Learning rate for the CORAL head.
*   `task: multiclass` (**Required**): Tells the DataModule to fetch ordinal labels
    (e.g., 'class_3', 'class_5') based on `num_classes`.
*   `num_classes: <int>` (**Required**): Number of ordinal levels/classes (e.g., 3, 4, 5).
    Determines the model output size and which label column is used.
*   `finetune: false` (Recommended): Explicitly state it's linear probing.

Example Command:

```bash
python evaluate.py \
    --config configs/syracuse_coral_5cls.yaml \
    --data_path /path/to/syracuse_features_root \
    --marlin_base_dir /path/to/syracuse_metadata \
    --n_gpus 1 \
    --batch_size 64 \
    --epochs 50
```

Note: The `predict_from_features` mode (triggered by providing `--feature_dir`,
`--checkpoint_path`, and `--output_path`) is separate and currently commented out
in the main execution block.
"""
import argparse
import torch
torch.set_float32_matmul_precision('high')

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm.auto import tqdm
from torch.nn.functional import softmax, sigmoid
import pandas as pd
from dataset.celebv_hq import CelebvHqDataModule
from dataset.biovid import BioVidDataModule
from dataset.syracuse import SyracuseDataModule
from dataset.syracuse import SyracuseLP
from model.config import resolve_config
# from marlin_pytorch.config import resolve_config
from marlin_pytorch.util import read_yaml
from model.classifier import Classifier
from model.CORAL import LightningMLP
from coral_pytorch.dataset import proba_to_label
from util.earlystop_lr import EarlyStoppingLR
from util.lr_logger import LrLogger
from util.seed import Seed
from util.system_stats_logger import SystemStatsLogger
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt # Import matplotlib
from matplotlib.ticker import MaxNLocator # For integer ticks
from torch.utils.data import DataLoader # Add DataLoader import globally



def train_celebvhq(args, config):
    data_path = args.data_path
    resume_ckpt = args.resume
    n_gpus = args.n_gpus
    max_epochs = args.epochs

    finetune = config["finetune"]
    learning_rate = config["learning_rate"]
    task = config["task"]

    if task == "appearance":
        num_classes = 40
    elif task == "action":
        num_classes = 35
    else:
        raise ValueError(f"Unknown task {task}")

    if finetune:
        backbone_config = resolve_config(config["backbone"])

        model = Classifier(
            num_classes, config["backbone"], True, args.marlin_ckpt, "multilabel", config["learning_rate"],
            args.n_gpus > 1,
        )

        dm = CelebvHqDataModule(
            data_path, finetune, task,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            clip_frames=backbone_config.n_frames,
            temporal_sample_rate=2
        )

    else:
        model = Classifier(
            num_classes, config["backbone"], False,
            None, "multilabel", config["learning_rate"], args.n_gpus > 1,
        )

        dm = CelebvHqDataModule(
            data_path, finetune, task,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            feature_dir=config["backbone"],
            temporal_reduction=config["temporal_reduction"]
        )

    if args.skip_train:
        dm.setup()
        return resume_ckpt, dm

    strategy = None if n_gpus <= 1 else "ddp"
    accelerator = "cpu" if n_gpus == 0 else "gpu"

    ckpt_filename = config["model_name"] + "-{epoch}-{val_auc:.3f}"
    ckpt_monitor = "val_auc"
    if args.task == "regression":
        ckpt_monitor = "val_mse"
    # ckpt_filename = config["model_name"] + "-{epoch}"+f"-{ckpt_monitor:.3f}"

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    ckpt_callback = ModelCheckpoint(dirpath=f"ckpt/{config['model_name']}", save_last=True,
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        mode="max")

    trainer = Trainer(log_every_n_steps=1, devices=n_gpus, accelerator=accelerator, benchmark=True,
        logger=True, precision=precision, max_epochs=max_epochs,
        strategy=strategy, resume_from_checkpoint=resume_ckpt,
        callbacks=[ckpt_callback, LrLogger(), EarlyStoppingLR(1e-6), SystemStatsLogger()])

    trainer.fit(model, dm)

    return ckpt_callback.best_model_path, dm

def train_biovid(args, config):
    data_path = args.data_path
    n_gpus = args.n_gpus
    max_epochs = args.epochs

    finetune = config["finetune"]
    learning_rate = config["learning_rate"]
    task = config["task"]

    if task == "binary" or task == "regression":
        # num_classes = 2
        num_classes = 1
    elif task == "multiclass":
        num_classes = config["num_classes"]
    else:
        raise ValueError(f"Unknown task {task}")

    if finetune:
        backbone_config = resolve_config(config["backbone"])

        model = Classifier(
            num_classes, config["backbone"], finetune, args.marlin_ckpt, task, config["learning_rate"],
            args.n_gpus > 1,
        )
        dm = BioVidDataModule(
            data_path, finetune, task, num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            clip_frames=backbone_config.n_frames,
            temporal_sample_rate=2
        )

    else:
        model = Classifier(
            num_classes, config["backbone"], False,
            args.marlin_ckpt, task, config["learning_rate"], args.n_gpus > 1,
        )
        if args.augmentation:
            dm = BioVidDataModule(
                data_path, finetune, task, num_classes,
                augmentation=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                feature_dir=config["backbone"],
                temporal_reduction=config["temporal_reduction"]
            )
        else:
            dm = BioVidDataModule(
                data_path, finetune, task, num_classes,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                feature_dir=config["backbone"],
                temporal_reduction=config["temporal_reduction"]
            )

    strategy = 'auto' if n_gpus <= 1 else "ddp"
    accelerator = "cpu" if n_gpus == 0 else "gpu"

    ckpt_filename = config["model_name"] + "-{epoch}-{val_auc:.3f}"
    ckpt_monitor = "val_auc"
    # ckpt_monitor = "val_acc"
    if task == "regression":
        ckpt_monitor = "val_mse"

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    ckpt_callback = ModelCheckpoint(dirpath=f"ckpt/{config['model_name']}", save_last=True,
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        mode="max")

    trainer = Trainer(
        log_every_n_steps=1,
        devices=n_gpus,
        accelerator=accelerator,
        benchmark=True,
        logger=True,
        precision=precision,
        max_epochs=max_epochs,
        strategy=strategy,
        callbacks=[ckpt_callback, LrLogger(), EarlyStoppingLR(1e-8), SystemStatsLogger()]
    )

    print("Starting BioVid training from scratch...")
    trainer.fit(model, dm)

    print(f"Training finished. Best model checkpoint: {ckpt_callback.best_model_path}")
    return ckpt_callback.best_model_path, dm

def evaluate_celebvhq(args, ckpt, dm):
    print("Load checkpoint", ckpt)
    model = Classifier.load_from_checkpoint(ckpt)
    accelerator = "cpu" if args.n_gpus == 0 else "gpu"
    trainer = Trainer(log_every_n_steps=1, devices=1 if args.n_gpus > 0 else 0, accelerator=accelerator, benchmark=True,
        logger=False, enable_checkpointing=False)
    Seed.set(42)
    model.eval()

    # collect predictions
    preds = trainer.predict(model, dm.test_dataloader())
    preds = torch.cat(preds)

    # collect ground truth
    ys = torch.zeros_like(preds, dtype=torch.bool)
    for i, (_, y) in enumerate(tqdm(dm.test_dataloader())):
        ys[i * args.batch_size: (i + 1) * args.batch_size] = y

    preds = preds.sigmoid()
    acc = ((preds > 0.5) == ys).float().mean()
    auc = model.auc_fn(preds, ys)
    results = {
        "acc": acc,
        "auc": auc
    }
    print(results)

# def evaluate_biovid(args, ckpt, dm):
#     print("Load checkpoint", ckpt)
#     model = Classifier.load_from_checkpoint(ckpt)
#     accelerator = "cpu" if args.n_gpus == 0 else "gpu"
#     trainer = Trainer(log_every_n_steps=1, devices=1 if args.n_gpus > 0 else 0, accelerator=accelerator, benchmark=True,
#         logger=False, enable_checkpointing=False)
#     Seed.set(42)
#     model.eval()
#     # Collect predictions
#     preds = trainer.predict(model, dm.test_dataloader())
#     preds = torch.cat(preds)  # Concatenate predictions from all batches
#     # Apply softmax for multiclass probabilities
#     prob = softmax(preds, dim=1)  # Apply softmax along the class dimension
#     # print("Predictions (softmax probabilities):", prob)
#
#     # Collect ground truth
#     ys = torch.zeros(len(preds), dtype=torch.long)  # Assuming labels are class indices
#     for i, (_, y) in enumerate(tqdm(dm.test_dataloader())):
#         ys[i * args.batch_size: (i + 1) * args.batch_size] = y.view(-1)  # Flatten the label tensor
#
#     # Calculate accuracy
#     # Get the predicted class by taking the argmax
#     preds_classes = torch.argmax(prob, dim=1)  # Get predicted classes
#     acc = (preds_classes == ys).float().mean()  # Calculate accuracy
#     acc_alt = model.acc_fn(prob, ys)  # Calculate AUC (assuming you are using a multiclass AUC calculation)
#     # Calculate AUC (assuming you are using a multiclass AUC calculation)
#     auc = model.auc_fn(prob, ys)  # This needs to be compatible with multiclass AUC calculation
#
#     # Store results
#     results = {
#         "acc": acc.item(),  # Convert to Python float for easier printing
#         "acc_alt": acc_alt.item(),  # Convert to Python float for easier printing
#         "auc": auc.item()  # Convert to Python float for easier printing
#     }
#"
#     print("Evaluation Results:", results)

def evaluate_biovid(args, ckpt, dm, config):
    """
    Evaluate a trained model on the BIOVID dataset.

    This function loads a model checkpoint, evaluates it on the test dataset, and computes
    various metrics including accuracy, AUC, per-class accuracy, per-class AUC, confusion matrix,
    and classification report.

    Args:
        args (argparse.Namespace): Command-line arguments containing `n_gpus` and `batch_size`.
        ckpt (str): Path to the model checkpoint file.
        dm (DataModule): DataModule containing the test dataset.

    Returns:
        dict: A dictionary containing evaluation results.
    """
    print("Loading checkpoint", ckpt)
    model = Classifier.load_from_checkpoint(ckpt)
    accelerator = "cpu" if args.n_gpus == 0 else "gpu"
    trainer = Trainer(
        log_every_n_steps=1,
        devices=1 if args.n_gpus > 0 else 0,
        accelerator=accelerator,
        benchmark=True,
        logger=False,
        enable_checkpointing=False,
    )
    Seed.set(42)
    model.eval()
    task = config["task"]
    # if args.predict_only:
    #     # Collect predictions
    #     preds_list = trainer.predict(model, dm.test_dataloader())
    #     preds = torch.cat(preds_list)  # Concatenate predictions from all batches
    #
    #     # Apply sigmoid if it's a binary classification task
    #     if task == "binary":
    #         probability = torch.sigmoid(preds)
    #     else:
    #         # For multiclass, apply softmax
    #         probability = softmax(preds, dim=1)
    #     # Collect ground truth and filenames
    #     filenames = []  # Assuming you have a way to collect filenames from your DataModule
    #     ys = torch.zeros(len(preds), dtype=torch.long)  # Assuming labels are class indices
    #     for i, (x, y, filename) in enumerate(tqdm(dm.test_dataloader())):
    #         ys[i * args.batch_size: (i + 1) * args.batch_size] = y.view(-1)  # Flatten the label tensor
    #         filenames.extend(filename)  # Collecting filenames
    #     # print(f"the length of filenames is {len(filenames)}")
    #     # print(f"the length of ys is {len(ys)}")
    #     # Create a DataFrame to save results
    #     results_df = pd.DataFrame({
    #         # 'filename': filenames,
    #         'predictions': probability.numpy().tolist(),
    #     })
    #
    #     # Save to CSV
    #     results_df.to_csv('predictions.csv', index=False)
    #     print("Predictions saved to predictions.csv")
    #     return results_df  # Return the DataFrame if needed

    if args.predict_only:
        # Collect predictions
        preds_list = trainer.predict(model, dm.test_dataloader())
        preds = torch.cat(preds_list)  # Concatenate predictions from all batches

        # Apply sigmoid if it's a binary classification task
        if task == "binary":
            probability = torch.sigmoid(preds)
            # Convert to binary predictions (0 or 1)
            pred_classes = (probability > 0.5).int()
        else:
            # For multiclass, apply softmax
            probability = softmax(preds, dim=1)
            # Get predicted class indices
            pred_classes = torch.argmax(probability, dim=1)

        # Collect ground truth and filenames
        filenames = []  # Assuming you have a way to collect filenames from your DataModule
        ys = torch.zeros(len(preds), dtype=torch.long)  # Assuming labels are class indices
        # for i, (x, y, filename) in enumerate(tqdm(dm.test_dataloader())):
        for i,(x,y) in enumerate(tqdm(dm.test_dataloader())):
            ys[i * args.batch_size: (i + 1) * args.batch_size] = y.view(-1)  # Flatten the label tensor
            # filenames.extend(filename)  # Collecting filenames

        # Calculate metrics for binary classification
        if task == "binary":
            # Convert tensors to numpy for sklearn metrics
            y_true = ys.cpu().numpy()
            print(f"the true labels are {y_true}")
            y_pred = pred_classes.cpu().numpy()
            y_score = probability.cpu().numpy()
            # print(f"Shape of y_true: {y_true.shape}")
            # print(f"Shape of y_pred: {y_pred.shape}")
            # assert y_true.shape == y_pred.shape, "Mismatch in shapes of y_true and y_pred"
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # Calculate accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Calculate AUC
            auc_score = roc_auc_score(y_true, y_score)

            # Calculate precision, recall, and F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Print metrics
            print("\nBinary Classification Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC: {auc_score:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("\nConfusion Matrix:")
            print(f"TN: {tn}, FP: {fp}")
            print(f"FN: {fn}, TP: {tp}")

            # Add metrics to results DataFrame
            results_df = pd.DataFrame({
                'filename': filenames,
                'true_label': y_true.tolist(),
                'predicted_label': y_pred.tolist(),
                'raw_prediction': preds.squeeze().cpu().numpy().tolist(),
                'probability': y_score.tolist(),  # Probability of class 1
            })

            # Add metrics as a separate DataFrame for reference
            metrics_df = pd.DataFrame({
                'metric': ['accuracy', 'auc', 'precision', 'recall', 'f1', 'tn', 'fp', 'fn', 'tp'],
                'value': [accuracy, auc_score, precision, recall, f1, tn, fp, fn, tp]
            })
            metrics_df.to_csv('metrics.csv', index=False)
            print("Metrics saved to metrics.csv")

        else:
            # For multiclass, just save basic prediction info
            results_df = pd.DataFrame({
                'filename': filenames,
                'true_label': ys.cpu().numpy().tolist(),
                'predicted_label': pred_classes.cpu().numpy().tolist(),
                'raw_predictions': preds.cpu().numpy().tolist(),
                'probabilities': probability.cpu().numpy().tolist(),
            })

        # Save to CSV
        results_df.to_csv('predictions.csv', index=False)
        print("Predictions saved to predictions.csv")
        return results_df  # Return the DataFrame if needed
    # if task == "binary" or task == "multiclass":
    #     # Collect predictions
    #     preds = trainer.predict(model, dm.test_dataloader())
    #     preds = torch.cat(preds)  # Concatenate predictions from all batches
    #     prob = softmax(preds, dim=1)  # Apply softmax to get probabilities
    #
    #     # Collect ground truth
    #     ys = torch.zeros(len(preds), dtype=torch.long)  # Assuming labels are class indices
    #     for i, (_, y) in enumerate(tqdm(dm.test_dataloader())):
    #         ys[i * args.batch_size: (i + 1) * args.batch_size] = y.view(-1)  # Flatten the label tensor
    #
    #     # Calculate overall accuracy
    #     preds_classes = torch.argmax(prob, dim=1)  # Get predicted classes
    #     acc = (preds_classes == ys).float().mean()  # Calculate accuracy
    #
    #     # Calculate AUC (one-vs-rest for multiclass)
    #     auc_scores = {}
    #     for class_idx in range(prob.shape[1]):  # Iterate over each class
    #         auc_scores[f"auc_class_{class_idx}"] = roc_auc_score(
    #             (ys == class_idx).int().numpy(), prob[:, class_idx].numpy()
    #         )
    #
    #     # Calculate per-class accuracy
    #     class_acc = {}
    #     for class_idx in range(prob.shape[1]):
    #         class_mask = ys == class_idx
    #         class_acc[f"acc_class_{class_idx}"] = (preds_classes[class_mask] == ys[class_mask]).float().mean().item()
    #
    #     # Generate confusion matrix
    #     cm = confusion_matrix(ys.numpy(), preds_classes.numpy())
    #
    #     # Generate classification report
    #     report = classification_report(
    #         ys.numpy(), preds_classes.numpy(), target_names=[f"Class_{i}" for i in range(prob.shape[1])], output_dict=True
    #     )
    #
    #     # Store results
    #     results = {
    #         "overall_accuracy": acc.item(),  # Overall accuracy
    #         "auc_scores": auc_scores,  # AUC for each class
    #         "class_accuracy": class_acc,  # Accuracy for each class
    #         "confusion_matrix": cm.tolist(),  # Confusion matrix as a list
    #         "classification_report": report,  # Classification report
    #     }
    #
    #     print("Evaluation Results:")
    #     print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    #     print("AUC Scores (One-vs-Rest):", results["auc_scores"])
    #     print("Per-Class Accuracy:", results["class_accuracy"])
    #     print("Confusion Matrix:")
    #     print(np.array2string(np.array(results["confusion_matrix"]), precision=4))
    #     print("Classification Report:")
    #     print(classification_report(ys.numpy(), preds_classes.numpy(), target_names=[f"Class_{i}" for i in range(prob.shape[1])]))

    ##top 2 classes considered as predicted class
    if task == "binary" or task == "multiclass":
        # Collect predictions
        preds = trainer.predict(model, dm.test_dataloader())
        preds = torch.cat(preds)  # Concatenate predictions from all batches
        prob = softmax(preds, dim=1)  # Apply softmax to get probabilities

        # Collect ground truth
        ys = torch.zeros(len(preds), dtype=torch.long)  # Assuming labels are class indices
        for i, (_, y) in enumerate(tqdm(dm.test_dataloader())):
            ys[i * args.batch_size: (i + 1) * args.batch_size] = y.view(-1)  # Flatten the label tensor

        # Calculate top 2 classes with max probabilities
        topk_probs, topk_indices = torch.topk(prob, k=2, dim=1)  # Get top 2 classes and their probabilities

        # Initialize lists to store differences
        correct_top1_differences = []
        correct_top2_differences = []

        for i in range(len(ys)):
            if ys[i] == topk_indices[i][0]:  # Check if true class is the top 1 prediction
                # Calculate the difference between top 1 and top 2 probabilities
                difference = topk_probs[i][0].item() - topk_probs[i][1].item()
                correct_top1_differences.append(difference)  # Store the difference

            if ys[i] == topk_indices[i][1]:  # Check if true class is the top 2 prediction
                # Calculate the difference between top 1 and top 2 probabilities
                difference = topk_probs[i][1].item() - topk_probs[i][0].item()
                correct_top2_differences.append(difference)  # Store the difference

        # Calculate statistics for differences
        def calculate_statistics(differences):
            if differences:
                mean_diff = np.mean(differences)
                std_diff = np.std(differences)
                median_diff = np.median(differences)
                q1_diff = np.percentile(differences, 25)
                q3_diff = np.percentile(differences, 75)
            else:
                mean_diff = std_diff = median_diff = q1_diff = q3_diff = 0
            return mean_diff, std_diff, median_diff, q1_diff, q3_diff

        # Statistics for top 1 hits
        top1_stats = calculate_statistics(correct_top1_differences)
        # Statistics for top 2 hits
        top2_stats = calculate_statistics(correct_top2_differences)

        # Store results
        results = {
            "top1_diff_mean": top1_stats[0],  # Mean difference for top 1 hits
            "top1_diff_std": top1_stats[1],  # Std deviation for top 1 hits
            "top1_diff_median": top1_stats[2],  # Median difference for top 1 hits
            "top1_diff_q1": top1_stats[3],  # Q1 for top 1 hits
            "top1_diff_q3": top1_stats[4],  # Q3 for top 1 hits
            "top2_diff_mean": top2_stats[0],  # Mean difference for top 2 hits
            "top2_diff_std": top2_stats[1],  # Std deviation for top 2 hits
            "top2_diff_median": top2_stats[2],  # Median difference for top 2 hits
            "top2_diff_q1": top2_stats[3],  # Q1 for top 2 hits
            "top2_diff_q3": top2_stats[4],  # Q3 for top 2 hits
        }

        print("Evaluation Results:")
        print(f"Mean Difference for Top 1 Hits: {results['top1_diff_mean']:.4f}")
        print(f"Std Deviation for Top 1 Hits: {results['top1_diff_std']:.4f}")
        print(f"Median Difference for Top 1 Hits: {results['top1_diff_median']:.4f}")
        print(f"Q1 for Top 1 Hits: {results['top1_diff_q1']:.4f}")
        print(f"Q3 for Top 1 Hits: {results['top1_diff_q3']:.4f}")

        print(f"Mean Difference for Top 2 Hits: {results['top2_diff_mean']:.4f}")
        print(f"Std Deviation for Top 2 Hits: {results['top2_diff_std']:.4f}")
        print(f"Median Difference for Top 2 Hits: {results['top2_diff_median']:.4f}")
        print(f"Q1 for Top 2 Hits: {results['top2_diff_q1']:.4f}")
        print(f"Q3 for Top 2 Hits: {results['top2_diff_q3']:.4f}")





    elif task == "regression":
        # Collect predictions
        preds = trainer.predict(model, dm.test_dataloader())
        preds = torch.cat(preds)  # Concatenate predictions from all batches

        # Collect ground truth
        ys = torch.zeros(len(preds), dtype=torch.float32)  # Assuming labels are continuous values
        for i, (_, y) in enumerate(tqdm(dm.test_dataloader())):
            ys[i * args.batch_size: (i + 1) * args.batch_size] = y.view(-1)  # Flatten the label tensor

        # Calculate regression metrics
        mae = mean_absolute_error(ys.numpy(), preds.numpy())
        mse = mean_squared_error(ys.numpy(), preds.numpy())
        rmse = np.sqrt(mse)
        r2 = r2_score(ys.numpy(), preds.numpy())

        # Store results
        results = {
            "mae": mae,  # Mean Absolute Error
            "mse": mse,  # Mean Squared Error
            "rmse": rmse,  # Root Mean Squared Error
            "r2": r2,  # R-squared
        }

        print("Evaluation Results for Regression:")
        print(f"Mean Absolute Error (MAE): {results['mae']:.4f}")
        print(f"Mean Squared Error (MSE): {results['mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {results['rmse']:.4f}")
        print(f"R-squared (R²): {results['r2']:.4f}")

    else:
        raise ValueError(
            f"Unsupported task type: {task}. Supported tasks are 'binary', 'multiclass', and 'regression'.")


    return results

def train_syracuse_cv(args, config):
    print("--- Starting Syracuse Cross-Validation Training --- ")
    data_path = args.data_path # root_dir for SyracuseDataModule
    n_gpus = args.n_gpus
    max_epochs = args.epochs

    # Syracuse uses feature-based approach (Linear Probing)
    finetune = False # Hardcoded as SyracuseDataModule is for features
    learning_rate = config["learning_rate"]
    task = config["task"]

    # For CORAL (LightningMLP), num_classes is the primary parameter defining the output structure.
    # The 'task' variable might still be used by the DataModule to fetch the correct label column.
    num_classes = config.get("num_classes")
    if num_classes is None:
         raise ValueError("config['num_classes'] must be specified in the config file for the Syracuse/CORAL workflow.")
    if not isinstance(num_classes, int) or num_classes <= 1:
         raise ValueError(f"config['num_classes'] must be an integer greater than 1 for CORAL. Found: {num_classes}")
         
    print(f"Using {num_classes} classes/levels for CORAL model.")

    # Instantiate LightningMLP model for CORAL
    print(f"Instantiating LightningMLP model with lr={learning_rate}")
    model = LightningMLP(
        num_classes=num_classes,
        learning_rate=learning_rate,
        distributed=args.n_gpus > 1
    )

    # Instantiate SyracuseDataModule
    print(f"Instantiating SyracuseDataModule with marlin_base_dir: {args.marlin_base_dir}")
    if not args.marlin_base_dir:
        raise ValueError("Missing required argument: --marlin_base_dir must be provided for Syracuse dataset.")

    dm = SyracuseDataModule(
        root_dir=data_path,
        task=task,
        num_classes=num_classes,
        batch_size=args.batch_size,
        feature_dir=config["backbone"], # Directory where .npy features are stored (relative to root_dir)
        temporal_reduction=config["temporal_reduction"],
        marlin_base_dir=args.marlin_base_dir, # Path containing clips_json.json etc.
        num_workers=args.num_workers
    )

    # Call setup to load all metadata
    dm.setup()

    # Get data needed for splitting
    unique_video_ids = list(dm.video_id_labels.keys())
    video_labels = [dm.video_id_labels[vid] for vid in unique_video_ids]
    n_splits = 3 # Define number of folds
    print(f"Preparing for {n_splits}-fold cross-validation on {len(unique_video_ids)} videos.")

    # Initialize StratifiedKFold
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # Use args.seed if available

    all_best_checkpoints = []
    # Rename list for clarity - it stores more than just checkpoints now
    fold_results_list = [] 

    # --- Cross-Validation Loop ---
    for fold_idx, (train_vid_indices, val_vid_indices) in enumerate(skf.split(unique_video_ids, video_labels)):
        print(f"\n===== Starting Fold {fold_idx + 1}/{n_splits} =====")

        # Get video IDs for this fold
        train_vids_np = np.array(unique_video_ids)[train_vid_indices]
        val_vids_np = np.array(unique_video_ids)[val_vid_indices]
        train_vids_set = set(train_vids_np)
        val_vids_set = set(val_vids_np)

        print(f"  Fold {fold_idx + 1}: Train Video IDs: {len(train_vids_set)}, Val Video IDs: {len(val_vids_set)}")

        # Assign filenames for this fold
        fold_train_names = []
        fold_val_names = []

        # Assign original clips
        for clip in dm.original_clips:
            if clip['video_id'] in train_vids_set:
                fold_train_names.append(clip['filename'])
            elif clip['video_id'] in val_vids_set:
                fold_val_names.append(clip['filename']) # Val gets only originals

        # Assign augmented clips to training
        num_aug_added = 0
        for clip in dm.augmented_clips:
            if clip['video_id'] in train_vids_set:
                fold_train_names.append(clip['filename'])
                num_aug_added += 1

        print(f"  Fold {fold_idx + 1}: Train clips: {len(fold_train_names)} ({num_aug_added} augmented), Val clips: {len(fold_val_names)}")

        if not fold_train_names or not fold_val_names:
             print(f"Warning: Fold {fold_idx + 1} resulted in empty train or val set. Skipping fold.")
             all_best_checkpoints.append(None) # Add placeholder for skipped fold
             continue

        # Instantiate Datasets and DataLoaders for the current fold
        common_lp_args = {
            "root_dir": dm.root_dir,
            "feature_dir": dm.feature_dir,
            "task": dm.task,
            "num_classes": dm.num_classes,
            "temporal_reduction": dm.temporal_reduction,
            "metadata": dm.all_metadata
        }
        fold_train_dataset = SyracuseLP(split=f"train_fold{fold_idx}", name_list=fold_train_names, **common_lp_args)
        fold_val_dataset = SyracuseLP(split=f"val_fold{fold_idx}", name_list=fold_val_names, **common_lp_args)

        fold_train_loader = DataLoader(fold_train_dataset, batch_size=dm.batch_size, shuffle=True, num_workers=dm.num_workers, pin_memory=True, drop_last=True, persistent_workers=dm.num_workers > 0)
        fold_val_loader = DataLoader(fold_val_dataset, batch_size=dm.batch_size, shuffle=False, num_workers=dm.num_workers, pin_memory=True, drop_last=False, persistent_workers=dm.num_workers > 0)

        # Instantiate a NEW model for each fold
        print(f"  Fold {fold_idx + 1}: Initializing new LightningMLP model...")
        model = LightningMLP(
            num_classes=num_classes,
            learning_rate=learning_rate,
            distributed=n_gpus > 1
        )

        # --- Define checkpoint strategy for the fold --- 
        ckpt_monitor = "val_acc" # Metric to monitor
        mode = "max"             # 'min' because lower MAE is better
        # Include fold index in the filename pattern
        ckpt_filename = config["model_name"] + f"-syracuse-fold{fold_idx}-{{epoch}}-{{{ckpt_monitor}:.3f}}" 
        # ----------------------------------------------

        # Checkpoint callback for the current fold
        fold_ckpt_dir = f"ckpt/{config['model_name']}_syracuse/fold_{fold_idx}"
        print(f"  Fold {fold_idx + 1}: Checkpoints will be saved in: {fold_ckpt_dir}")
        ckpt_callback = ModelCheckpoint(
            dirpath=fold_ckpt_dir, # Use fold-specific directory
            save_last=True,
            filename=ckpt_filename, # Filename includes metric and epoch
            monitor=ckpt_monitor,
            mode=mode # Mode should match the monitored metric
        )

        # Trainer setup for the fold
        strategy = 'auto' if n_gpus <= 1 else "ddp"
        accelerator = "cpu" if n_gpus == 0 else "gpu"

        try:
            precision = int(args.precision)
        except ValueError:
            precision = args.precision

        # Use a reasonable patience for EarlyStopping, e.g., 10 epochs
        early_stop_patience = 50
        print(f"Using EarlyStopping with patience={early_stop_patience}, monitoring '{ckpt_monitor}' ({mode})")

        trainer = Trainer(
            log_every_n_steps=1,
            devices=n_gpus,
            accelerator=accelerator,
            benchmark=True,
            logger=True,
            precision=precision,
            max_epochs=max_epochs,
            strategy=strategy,
            callbacks=[
                ckpt_callback,
                LrLogger(),
                EarlyStopping(
                    monitor="val_acc",
                    mode="max",
                    patience=50,
                    verbose=True
                ),
                SystemStatsLogger()
            ]
        )

        # Train the model for the current fold
        print(f"  Fold {fold_idx + 1}: Starting training...")
        trainer.fit(model, train_dataloaders=fold_train_loader, val_dataloaders=fold_val_loader)

        print(f"  Fold {fold_idx + 1}: Training finished. Best model path: {ckpt_callback.best_model_path}")
        # Store results for this fold as a dictionary
        fold_results_list.append({
            'ckpt_path': ckpt_callback.best_model_path,
            'val_filenames': fold_val_names # Make sure fold_val_names is correctly populated
        })
        # --------------------------

    print("\n===== Cross-Validation Training Complete =====")
    # Return the list of dictionaries
    return fold_results_list, dm

def _evaluate_fold_checkpoint(ckpt_path, val_filenames, dm, config):
    """Loads a fold's checkpoint and evaluates on its validation set."""
    print(f"\n--- Evaluating Fold Checkpoint: {os.path.basename(ckpt_path)} ---")
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("  Warning: Invalid checkpoint path provided. Skipping evaluation for this fold.")
        return None

    try:
        model = LightningMLP.load_from_checkpoint(ckpt_path)
        model.eval()
        eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(eval_device)
        num_classes = model.num_classes
    except Exception as e:
        print(f"  Warning: Failed to load checkpoint {ckpt_path}. Error: {e}. Skipping evaluation for this fold.")
        return None

    # Create validation dataset and dataloader for this fold
    common_lp_args = {
        "root_dir": dm.root_dir, "feature_dir": dm.feature_dir, "task": dm.task,
        "num_classes": dm.num_classes, "temporal_reduction": dm.temporal_reduction,
        "metadata": dm.all_metadata
    }
    fold_val_dataset = SyracuseLP(split="val_eval", name_list=val_filenames, **common_lp_args)
    if not fold_val_dataset:
        print("  Warning: Failed to create validation dataset for evaluation. Skipping fold.")
        return None
    fold_val_loader = DataLoader(fold_val_dataset, batch_size=dm.batch_size, shuffle=False, num_workers=dm.num_workers, pin_memory=True)

    # Manual prediction loop
    all_val_preds = []
    all_val_true = []
    try:
        with torch.no_grad():
            for batch in tqdm(fold_val_loader, desc=f"Predicting Fold Validation Set"):
                features, true_labels = batch
                features = features.to(eval_device)
                logits = model(features)
                probas = torch.sigmoid(logits)
                predicted_labels = proba_to_label(probas)
                all_val_preds.append(predicted_labels.cpu())
                all_val_true.append(true_labels.cpu())
    except Exception as e:
        print(f"  Warning: Error during prediction for this fold. Error: {e}. Skipping.")
        return None

    if not all_val_preds:
        print("  Warning: No predictions generated for this fold's validation set.")
        return None

    preds_tensor = torch.cat(all_val_preds)
    true_tensor = torch.cat(all_val_true)
    y_true_np = true_tensor.numpy()
    pred_classes = preds_tensor.numpy()

    # Calculate metrics
    try:
        mae = mean_absolute_error(y_true_np, pred_classes)
        acc = np.mean(pred_classes == y_true_np)
        print(f"  Fold Validation Metrics: MAE={mae:.4f}, Accuracy={acc:.4f}")
        return {'mae': mae, 'acc': acc, 'y_true': y_true_np, 'y_pred': pred_classes}
    except Exception as e:
        print(f"  Warning: Error calculating metrics for this fold. Error: {e}")
        return None

def evaluate(args):
    print(f"Loading config from: {args.config}")
    config = read_yaml(args.config)
    dataset_name = config["dataset"]
    print(f"Dataset specified in config: {dataset_name}")

    if dataset_name == "celebvhq":
        print("Running evaluation for CelebVHQ...")
        # ckpt, dm = train_celebvhq(args, config)
        # evaluate_celebvhq(args, ckpt, dm)
        print("CelebVHQ part currently commented out.")
        pass
    elif dataset_name == "biovid":
        print("Running evaluation for BioVid...")
        ckpt, dm = train_biovid(args, config)
        if not args.predict_only:
            evaluate_biovid(args, ckpt, dm, config)
        else:
             # Handle predict_only for biovid if needed, or rely on the evaluate_biovid logic
             evaluate_biovid(args, ckpt, dm, config) # evaluate_biovid handles predict_only
    elif dataset_name == "syracuse": # Added Syracuse block
        print("Running evaluation for Syracuse...")
        # Call the CV training function
        fold_results, dm = train_syracuse_cv(args, config) # Changed variable name
        print(f"\nCross-validation training complete.")
        # --- DEBUG PRINT --- 
        print(f"DEBUG: Type of fold_results: {type(fold_results)}")
        if isinstance(fold_results, list) and len(fold_results) > 0:
            print(f"DEBUG: Type of first item in fold_results: {type(fold_results[0])}")
        # Print limited content for brevity if it's long
        print(f"DEBUG: Content of fold_results (first few items): {str(fold_results)[:500]}") 
        # ---------------------
        print(f"Fold results (ckpt_path, val_filenames count):")
        for i, res in enumerate(fold_results):
            print(f"  Fold {i}: Ckpt={os.path.basename(res['ckpt_path']) if res['ckpt_path'] else 'None'}, Val Files={len(res['val_filenames'])}")

        # --- Aggregate and Report CV Metrics ---
        all_fold_mae = []
        all_fold_acc = []
        all_fold_true = []
        all_fold_pred = []

        for fold_idx, result in enumerate(fold_results):
            if result['ckpt_path']:
                fold_metrics = _evaluate_fold_checkpoint(
                    result['ckpt_path'],
                    result['val_filenames'],
                    dm,
                    config
                )
                if fold_metrics:
                    all_fold_mae.append(fold_metrics['mae'])
                    all_fold_acc.append(fold_metrics['acc'])
                    all_fold_true.extend(fold_metrics['y_true'])
                    all_fold_pred.extend(fold_metrics['y_pred'])
            else:
                 print(f"Skipping evaluation for Fold {fold_idx} as no checkpoint was generated.")

        print("\n--- Cross-Validation Summary ---")
        if all_fold_mae:
            avg_mae = np.mean(all_fold_mae)
            std_mae = np.std(all_fold_mae)
            avg_acc = np.mean(all_fold_acc)
            std_acc = np.std(all_fold_acc)
            print(f"Average Validation MAE across {len(all_fold_mae)} folds: {avg_mae:.4f} (+/- {std_mae:.4f})")
            print(f"Average Validation Accuracy across {len(all_fold_acc)} folds: {avg_acc:.4f} (+/- {std_acc:.4f})")

            # Combined Confusion Matrix
            if all_fold_true and all_fold_pred:
                 print("\nCombined Confusion Matrix (Validation Sets):")
                 try:
                    # Ensure labels are integers for confusion matrix if necessary
                    all_fold_true_np = np.array(all_fold_true, dtype=int)
                    all_fold_pred_np = np.array(all_fold_pred, dtype=int)
                    combined_cm = confusion_matrix(all_fold_true_np, all_fold_pred_np)
                    print(combined_cm)
                    # Optional: Print classification report on combined data
                    # print("\nCombined Classification Report (Validation Sets):")
                    # print(classification_report(all_fold_true_np, all_fold_pred_np, zero_division=0))
                 except Exception as cm_e:
                      print(f"  Warning: Could not generate combined confusion matrix. Error: {cm_e}")
            else:
                 print("\nCould not generate combined confusion matrix (no labels collected).")

        else:
            print("No valid fold results found to calculate average metrics or confusion matrix.")
        # --------------------------------------

        # Remove the old single evaluation call
        # if ckpt:
        #      evaluate_syracuse(args, ckpt, dm, config) # This function is essentially replaced by the loop above
        # else:
        #     print("Warning: No checkpoint available from training. Cannot evaluate.")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

# --- Helper Function for Plotting ---
def _plot_pain_vs_class(predicted_class_levels, true_class_levels, filenames, dm, config, split_name):
    """Generates and saves a plot of true pain level vs. predicted class level."""
    print(f"\n--- Generating True Pain vs. Predicted Class plot for {split_name} Set ---")
    try:
        # --- Get True Pain Levels from Metadata ---
        true_pain_levels_list = []
        valid_indices_for_plot = [] # Keep track of indices where pain level is valid
        skipped_pain_level = 0

        if len(filenames) != len(true_class_levels):
            print(f"  Warning: Mismatch between {split_name} filenames ({len(filenames)}) and true labels ({len(true_class_levels)}). Cannot generate accurate plot.")
            return # Skip plotting if counts don't match
        else:
            print(f"  Extracting true pain levels for {split_name} set...")
            for idx, filename in enumerate(filenames):
                meta = dm.all_metadata.get(filename)
                pain_level = None
                if meta and 'meta_info' in meta:
                    pain_level_val = meta['meta_info'].get('pain_level')
                    if pain_level_val is not None:
                        try:
                            pain_level = float(pain_level_val)
                            true_pain_levels_list.append(pain_level)
                            valid_indices_for_plot.append(idx) # Store index if valid
                        except (ValueError, TypeError):
                            pain_level = None # Treat conversion error as missing
                
                if pain_level is None:
                    skipped_pain_level += 1
            
            if skipped_pain_level > 0:
                 print(f"    (Skipped {skipped_pain_level} clips for plot due to missing/invalid 'pain_level' in metadata)")

        # Convert to numpy array and filter predictions accordingly
        true_pain_levels_np = np.array(true_pain_levels_list)
        # Filter predicted classes to only include those where true pain level was valid
        pred_classes_for_plot = predicted_class_levels[valid_indices_for_plot]

        if len(true_pain_levels_np) == 0:
             print("  Warning: No valid true pain levels found. Skipping plot generation.")
             return

        # --- Create Plot ---
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Optional: Add jitter
        jitter_strength_x = 0.1 
        jitter_strength_y = 0.1
        x_jitter = true_pain_levels_np + np.random.uniform(-jitter_strength_x, jitter_strength_x, size=true_pain_levels_np.shape)
        y_jitter = pred_classes_for_plot + np.random.uniform(-jitter_strength_y, jitter_strength_y, size=pred_classes_for_plot.shape)

        ax.scatter(x_jitter, y_jitter, alpha=0.5, label='Predicted Class Level')

        ax.set_xlabel("True Pain Level (from Metadata)")
        ax.set_ylabel("Predicted Ordinal Class Level")
        ax.set_title(f"True Pain Level vs. Predicted Class ({split_name} Set - {config.get('model_name', 'Unknown')})")

        # Set integer ticks for Y axis 
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Adjust limits based on actual data ranges
        ax.set_xlim(np.nanmin(true_pain_levels_np) - 0.5, np.nanmax(true_pain_levels_np) + 0.5)
        ax.set_ylim(np.nanmin(pred_classes_for_plot) - 0.5, np.nanmax(pred_classes_for_plot) + 0.5)
        
        ax.legend()
        ax.grid(True)
        
        # Save the plot
        plot_filename = f"{config.get('model_name', 'syracuse')}_{split_name.lower()}_true_pain_vs_pred_class.png"
        plt.savefig(plot_filename)
        print(f"  Saved plot to: {plot_filename}")
        plt.close(fig) # Close the figure to free memory
            
    except Exception as plot_e:
        print(f"\n  Warning: Failed to generate True Pain vs. Predicted plot for {split_name} set. Error: {plot_e}")
# ----------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MARLIN Model Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset root directory.")
    parser.add_argument("--marlin_ckpt", type=str, default=None,
        help="Path to MARLIN checkpoint (often unused for LP). Default: None, load from online.")
    parser.add_argument("--marlin_base_dir", type=str, default=None, # Added for Syracuse
                        help="Base directory for MARLIN features and metadata (required for Syracuse dataset).")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use (0 for CPU).")
    parser.add_argument("--precision", type=str, default="32", help="Training precision (e.g., 32, 16, bf16).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs to train.") # Reduced default epochs
    parser.add_argument("--predict_only", action="store_true", default=False,
                        help="Skip evaluation metrics. Save prediction results only (uses checkpoint from training).")
    parser.add_argument("--augmentation", action="store_true", default=False, 
                        help="Enable feature augmentation for BioVid LP (not used for Syracuse).")
    # Removed --val_split_ratio and --test_split_ratio as CV handles splits
    # Deprecated/Moved prediction-specific args to a separate mode maybe?
    # Keep them for now as predict_from_features uses them.
    parser.add_argument("--feature_dir", type=str, help="Directory containing pre-extracted MARLIN features (for predict_from_features mode).")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the trained model checkpoint (for predict_from_features mode).")
    parser.add_argument("--output_path", type=str, help="Path to save the predictions CSV (for predict_only or predict_from_features mode).")

    args = parser.parse_args()

    # Decide execution mode
    if args.feature_dir and args.checkpoint_path:
        print("--- Running Prediction from Features Mode ---")
        # Ensure output path is provided for this mode
        if not args.output_path:
             parser.error("Prediction from features mode requires --output_path.")
        config = read_yaml(args.config)
        # predict_from_features(args, config) # This function seems separate, maybe call it directly?
        print("Predict from features function call currently commented out.")
    else:
        print("--- Running Standard Training/Evaluation Mode ---")
        # Check for Syracuse specific requirement
        config_check = read_yaml(args.config)
        if config_check.get("dataset") == "syracuse" and not args.marlin_base_dir:
             parser.error("Syracuse dataset requires --marlin_base_dir to be specified.")
        evaluate(args)
