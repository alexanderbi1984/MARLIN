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

def train_syracuse(args, config):
    print("--- Starting Syracuse Training --- ")
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
    print("Instantiating SyracuseDataModule...")
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
        val_split_ratio=args.val_split_ratio,
        test_split_ratio=args.test_split_ratio,
        random_state=42, # Or use args.seed if available
        num_workers=args.num_workers
    )

    strategy = 'auto' if n_gpus <= 1 else "ddp"
    accelerator = "cpu" if n_gpus == 0 else "gpu"

    # CORAL typically uses MAE for validation monitoring
    ckpt_monitor = "val_acc" # Changed monitor metric
    mode = "min" # Minimize MAE
    ckpt_filename = config["model_name"] + f"-syracuse-{{epoch}}-{{{ckpt_monitor}:.3f}}"

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    ckpt_callback = ModelCheckpoint(
        dirpath=f"ckpt/{config['model_name']}_syracuse", # Add suffix to ckpt dir
        save_last=True,
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        mode=mode # Use 'min' for regression (MSE), 'max' otherwise (AUC)
    )

    # Use a reasonable patience for EarlyStopping, e.g., 10 epochs
    early_stop_patience = 10
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
                patience=10,
                verbose=True
            ),
            SystemStatsLogger()
        ]
    )

    # Always start training from scratch
    print("Starting Syracuse training from scratch...")
    trainer.fit(model, dm)

    print(f"Syracuse training finished. Best model checkpoint: {ckpt_callback.best_model_path}")
    return ckpt_callback.best_model_path, dm

def evaluate_syracuse(args, ckpt, dm, config):
    print("--- Starting Syracuse Evaluation --- ")
    print(f"Loading checkpoint: {ckpt}")

    try:
        # Load the LightningMLP model from checkpoint
        model = LightningMLP.load_from_checkpoint(ckpt)
    except Exception as e:
        print(f"Error loading checkpoint {ckpt}: {e}")
        # Attempt to load ignoring strict=False if it might be a partial save
        try:
             print("Attempting to load checkpoint with strict=False")
             model = LightningMLP.load_from_checkpoint(ckpt, strict=False)
        except Exception as e2:
             print(f"Failed to load checkpoint even with strict=False: {e2}")
             raise RuntimeError(f"Could not load checkpoint: {ckpt}")

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
    num_classes = model.num_classes # Get num_classes from the loaded model

    # --- Prediction Phase --- 
    print("Running prediction on the test set (manual loop for CORAL)...")
    # Manual prediction loop for CORAL model
    all_preds = [] 
    all_probas = [] # Store probabilities if needed
    all_logits = [] # Store raw logits if needed
    all_true_labels = []
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"): 
            features, true_labels = batch
            features = features.to(accelerator)

            # Forward pass to get logits
            logits = model(features)
            
            # Convert logits to probabilities and then to labels using CORAL logic
            probas = torch.sigmoid(logits)
            predicted_labels = proba_to_label(probas)

            all_preds.append(predicted_labels.cpu())
            all_true_labels.append(true_labels.cpu()) # Keep original labels
            # Optionally store logits/probas if needed for analysis
            all_logits.append(logits.cpu())
            all_probas.append(probas.cpu())

    if not all_preds:
        print("Error: Prediction list is empty after manual loop.")
        return {}
        
    preds_tensor = torch.cat(all_preds)
    ys = torch.cat(all_true_labels)
    logits_tensor = torch.cat(all_logits)
    probas_tensor = torch.cat(all_probas)

    # --- Ground Truth Collection --- 
    # Ground truth (ys) collected during the prediction loop above
     
    # Ensure preds and ys have compatible shapes/lengths
    if preds_tensor.shape[0] != ys.shape[0]:
        print(f"Warning: Mismatch between number of predictions ({preds_tensor.shape[0]}) and ground truth labels ({ys.shape[0]}). Check dataloader/prediction process.")
        # Attempt to truncate to the shorter length
        min_len = min(preds_tensor.shape[0], ys.shape[0])
        preds_tensor = preds_tensor[:min_len]
        ys = ys[:min_len]
        if min_len == 0:
            print("Error: No matching predictions and labels found.")
            return {}

    # --- Prediction Post-processing & Saving (if predict_only) --- 
    if args.predict_only:
        print("Predict only mode: Saving predictions...")
        # For CORAL, the direct prediction is the label from proba_to_label
        pred_classes = preds_tensor # Already calculated labels
        raw_logits_np = logits_tensor.cpu().numpy() # Save logits
        probas_np = probas_tensor.cpu().numpy() # Save sigmoid probabilities

        # Assuming dm.test_dataset.name_list contains filenames corresponding to ys/preds
        filenames = dm.test_dataset.name_list if hasattr(dm, 'test_dataset') and hasattr(dm.test_dataset, 'name_list') else [f"item_{i}" for i in range(len(ys))]
        if len(filenames) != len(ys):
             print(f"Warning: Filename list length ({len(filenames)}) doesn't match label count ({len(ys)}). Using generic names.")
             filenames = [f"item_{i}" for i in range(len(ys))]

        results_df = pd.DataFrame({
            'filename': filenames,
            'true_label': ys.cpu().numpy().tolist(),
            'predicted_label': pred_classes.tolist(),
            'raw_logits': raw_logits_np.tolist(),
            'probabilities': probas_np.tolist(), # Sigmoid outputs from CORAL layer
        })
        results_df.to_csv(args.output_path or 'syracuse_predictions.csv', index=False)
        print(f"Predictions saved to {args.output_path or 'syracuse_predictions.csv'}")
        # Return metrics calculation part if needed, or just the df
        # For predict_only, maybe return basic info or the df itself
        return {"status": "Predictions saved", "path": args.output_path or 'syracuse_predictions.csv'}

    # --- Metrics Calculation --- 
    print("Calculating evaluation metrics...")
    results = {}
    y_true_np = ys.cpu().numpy()

    # CORAL is for ordinal classification (task should be multiclass conceptually)
    if task == "multiclass" or task == "regression" or task == "binary": # Handle all cases, but metrics differ
        print(f"Calculating metrics for task: {task} using CORAL predictions...")
        pred_classes = preds_tensor.cpu().numpy()

        # --- Key Metric: MAE ---
        mae = mean_absolute_error(y_true_np, pred_classes)
        results['mae'] = mae
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        
        # --- Accuracy ---
        acc = np.mean(pred_classes == y_true_np)
        results['accuracy'] = acc
        print(f"  Accuracy: {acc:.4f}")

        # --- AUC (Not standard for CORAL, skip or use with caution) ---
        # probas_np = probas_tensor.cpu().numpy()
        # try:
        #      # Calculate AUC if needed, but interpret carefully
        #      # auc = roc_auc_score(y_true_np, probas_np, multi_class='ovr', average='weighted') 
        #      # print(f"  AUC (Weighted OvR - experimental): {auc:.4f}")
        #      # results['auc_experimental'] = auc
        # except ValueError as e:
        #      print(f"  AUC calculation failed: {e}")
        print("  AUC calculation skipped (not standard for CORAL output).")

        # Confusion Matrix & Classification Report
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true_np, pred_classes)
        print(cm)
        results['confusion_matrix'] = cm.tolist()

        print("\nClassification Report:")
        target_names = [f"Class_{i}" for i in range(num_classes)] if task == "multiclass" else ["Class_0", "Class_1"]
        # Ensure labels in report match actual classes present
        unique_labels = np.unique(np.concatenate((y_true_np, pred_classes))) 
        filtered_target_names = [f"Class_{i}" for i in unique_labels]
        try:
             report = classification_report(y_true_np, pred_classes, labels=unique_labels, target_names=filtered_target_names, output_dict=True, zero_division=0)
             print(classification_report(y_true_np, pred_classes, labels=unique_labels, target_names=filtered_target_names, zero_division=0))
             results['classification_report'] = report
        except Exception as e:
            print(f"Could not generate classification report: {e}")
            results['classification_report'] = {}

    else:
        raise ValueError(f"Unsupported task type: {task}")

    print("--- Syracuse Evaluation Complete ---")
    return results

def predict_from_features(args, config):
    """
    Make predictions using pre-extracted MARLIN features.
    
    Args:
        args: Command line arguments containing:
            - feature_dir: Directory containing .npy feature files
            - checkpoint_path: Path to the trained model checkpoint
            - output_path: Path to save the predictions CSV
            - batch_size: Batch size for predictions
            - n_gpus: Number of GPUs to use
        config: Model configuration dictionary
    """
    print("Loading checkpoint", args.checkpoint_path)
    model = Classifier.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    
    # Setup device
    device = "cuda" if args.n_gpus > 0 else "cpu"
    model = model.to(device)
    
    # Get list of all .npy files in feature directory
    feature_files = glob.glob(os.path.join(args.feature_dir, "*.npy"))
    print(f"Found {len(feature_files)} feature files")
    
    # Initialize lists to store results
    filenames = []
    predictions = []
    probabilities = []
    
    # Process features in batches
    batch_features = []
    batch_files = []
    
    # Get temporal reduction method from config
    temporal_reduction = config.get("temporal_reduction", "mean")
    
    for feature_file in tqdm(feature_files, desc="Processing features"):
        try:
            # Load feature
            feature = np.load(feature_file)
            feature = torch.from_numpy(feature).float()
            
            # Apply temporal reduction
            if temporal_reduction == "mean":
                feature = feature.mean(dim=0, keepdim=True)  # [T, D] -> [1, D]
            elif temporal_reduction == "max":
                feature = feature.max(dim=0, keepdim=True)[0]  # [T, D] -> [1, D]
            else:
                raise ValueError(f"Unsupported temporal reduction method: {temporal_reduction}")
            
            batch_features.append(feature)
            batch_files.append(os.path.basename(feature_file))
            
            # Process when batch is full or on last item
            if len(batch_features) == args.batch_size or feature_file == feature_files[-1]:
                # Stack features into a batch
                batch_tensor = torch.stack(batch_features).to(device)
                batch_tensor = batch_tensor.squeeze(1)  # Remove temporal dimension after reduction
                
                # Get predictions
                with torch.no_grad():
                    batch_preds = model(batch_tensor)
                    
                    if config["task"] == "binary":
                        batch_probs = torch.sigmoid(batch_preds)
                        batch_pred_classes = (batch_probs > 0.5).int()
                    else:  # multiclass
                        batch_probs = softmax(batch_preds, dim=1)
                        batch_pred_classes = torch.argmax(batch_probs, dim=1)
                
                # Store results
                filenames.extend(batch_files)
                predictions.extend(batch_pred_classes.cpu().numpy())
                probabilities.extend(batch_probs.cpu().numpy())
                
                # Clear batch
                batch_features = []
                batch_files = []
                
        except Exception as e:
            print(f"Error processing file {feature_file}: {str(e)}")
            continue
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'filename': filenames,
        'predicted_label': predictions,
        'probabilities': probabilities
    })
    
    # Save to CSV
    results_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")
    
    return results_df

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
        ckpt, dm = train_syracuse(args, config)
        # Evaluate the best checkpoint saved from the training run
        if ckpt: # Ensure training produced a checkpoint
             evaluate_syracuse(args, ckpt, dm, config)
        else:
            print("Warning: No checkpoint available from training. Cannot evaluate.")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")


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
    parser.add_argument("--val_split_ratio", type=float, default=0.15, # Added for Syracuse
                        help="Ratio of videos for the validation set (Syracuse only).")
    parser.add_argument("--test_split_ratio", type=float, default=0.15, # Added for Syracuse
                        help="Ratio of videos for the test set (Syracuse only).")
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
