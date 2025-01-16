import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm
from torch.nn.functional import softmax
import pandas as pd
from dataset.celebv_hq import CelebvHqDataModule
from dataset.biovid import BioVidDataModule
from marlin_pytorch.config import resolve_config
from marlin_pytorch.util import read_yaml
from model.classifier import Classifier
from util.earlystop_lr import EarlyStoppingLR
from util.lr_logger import LrLogger
from util.seed import Seed
from util.system_stats_logger import SystemStatsLogger
import torch
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm



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
    resume_ckpt = args.resume
    n_gpus = args.n_gpus
    max_epochs = args.epochs

    finetune = config["finetune"]
    learning_rate = config["learning_rate"]
    task = config["task"]

    if task == "binary" or task == "regression":
        # num_classes = 2
        num_classes = 1
    elif task == "multiclass":
        num_classes = 5
    else:
        raise ValueError(f"Unknown task {task}")

    if finetune:
        backbone_config = resolve_config(config["backbone"])

        model = Classifier(
            num_classes, config["backbone"], finetune, args.marlin_ckpt, task, config["learning_rate"],
            args.n_gpus > 1,
        )
        dm = BioVidDataModule(
            data_path, finetune, task,
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

        dm = BioVidDataModule(
            data_path, finetune, task,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            feature_dir=config["backbone"],
            temporal_reduction=config["temporal_reduction"]
        )

    if args.skip_train:
        dm.setup()
        return resume_ckpt, dm

    strategy = 'auto' if n_gpus <= 1 else "ddp"
    accelerator = "cpu" if n_gpus == 0 else "gpu"

    ckpt_filename = config["model_name"] + "-{epoch}-{val_auc:.3f}"
    # ckpt_monitor = "val_auc"
    ckpt_monitor = "val_acc"
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
## the way to resume training from a checkpoint has changed. The resume_from_checkpoint argument was removed in favor of a different approach.
    # trainer = Trainer(log_every_n_steps=1, devices=n_gpus, accelerator=accelerator, benchmark=True,
    #     logger=True, precision=precision, max_epochs=max_epochs,
    #     strategy=strategy, resume_from_checkpoint=resume_ckpt,
    #     callbacks=[ckpt_callback, LrLogger(), EarlyStoppingLR(1e-6), SystemStatsLogger()])
    # trainer.fit(model, dm)
    # Initialize the Trainer
    trainer = Trainer(
        log_every_n_steps=1,  # Log every n steps
        devices=n_gpus,  # Number of GPUs to use
        accelerator=accelerator,  # Accelerator type (e.g., 'gpu', 'cpu')
        benchmark=True,  # Enable benchmark mode
        logger=True,  # Whether to log
        precision=precision,  # Precision (e.g., 16 or 32)
        max_epochs=max_epochs,  # Maximum number of epochs
        strategy=strategy,  # Distributed strategy
        callbacks=[ckpt_callback, LrLogger(), EarlyStoppingLR(1e-8), SystemStatsLogger()]  # List of callbacks
    )

    # Fit the model, passing the checkpoint path if resuming training
    if resume_ckpt:  # Check if a checkpoint path is provided
        trainer.fit(model, dm, ckpt_path=resume_ckpt)  # Resume training from the checkpoint
    else:
        trainer.fit(model, dm)  # Start training from scratch
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
    if args.predict_only:
        # Collect predictions
        preds = trainer.predict(model, dm.test_dataloader())
        preds = torch.cat(preds)  # Concatenate predictions from all batches

        # Collect ground truth and filenames
        filenames = []  # Assuming you have a way to collect filenames from your DataModule
        ys = torch.zeros(len(preds), dtype=torch.long)  # Assuming labels are class indices
        for i, (x, y, filename) in enumerate(tqdm(dm.test_dataloader())):
            ys[i * args.batch_size: (i + 1) * args.batch_size] = y.view(-1)  # Flatten the label tensor
            filenames.extend(filename)  # Collecting filenames

        # Create a DataFrame to save results
        results_df = pd.DataFrame({
            'filename': filenames,
            'predictions': preds.numpy().tolist(),
        })

        # Save to CSV
        results_df.to_csv('predictions.csv', index=False)
        print("Predictions saved to predictions.csv")
        return results_df  # Return the DataFrame if needed
    if task == "binary" or task == "multiclass":
        # Collect predictions
        preds = trainer.predict(model, dm.test_dataloader())
        preds = torch.cat(preds)  # Concatenate predictions from all batches
        prob = softmax(preds, dim=1)  # Apply softmax to get probabilities

        # Collect ground truth
        ys = torch.zeros(len(preds), dtype=torch.long)  # Assuming labels are class indices
        for i, (_, y) in enumerate(tqdm(dm.test_dataloader())):
            ys[i * args.batch_size: (i + 1) * args.batch_size] = y.view(-1)  # Flatten the label tensor

        # Calculate overall accuracy
        preds_classes = torch.argmax(prob, dim=1)  # Get predicted classes
        acc = (preds_classes == ys).float().mean()  # Calculate accuracy

        # Calculate AUC (one-vs-rest for multiclass)
        auc_scores = {}
        for class_idx in range(prob.shape[1]):  # Iterate over each class
            auc_scores[f"auc_class_{class_idx}"] = roc_auc_score(
                (ys == class_idx).int().numpy(), prob[:, class_idx].numpy()
            )

        # Calculate per-class accuracy
        class_acc = {}
        for class_idx in range(prob.shape[1]):
            class_mask = ys == class_idx
            class_acc[f"acc_class_{class_idx}"] = (preds_classes[class_mask] == ys[class_mask]).float().mean().item()

        # Generate confusion matrix
        cm = confusion_matrix(ys.numpy(), preds_classes.numpy())

        # Generate classification report
        report = classification_report(
            ys.numpy(), preds_classes.numpy(), target_names=[f"Class_{i}" for i in range(prob.shape[1])], output_dict=True
        )

        # Store results
        results = {
            "overall_accuracy": acc.item(),  # Overall accuracy
            "auc_scores": auc_scores,  # AUC for each class
            "class_accuracy": class_acc,  # Accuracy for each class
            "confusion_matrix": cm.tolist(),  # Confusion matrix as a list
            "classification_report": report,  # Classification report
        }

        print("Evaluation Results:")
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print("AUC Scores (One-vs-Rest):", results["auc_scores"])
        print("Per-Class Accuracy:", results["class_accuracy"])
        print("Confusion Matrix:")
        print(np.array2string(np.array(results["confusion_matrix"]), precision=4))
        print("Classification Report:")
        print(classification_report(ys.numpy(), preds_classes.numpy(), target_names=[f"Class_{i}" for i in range(prob.shape[1])]))
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


def evaluate(args):
    config = read_yaml(args.config)
    dataset_name = config["dataset"]

    if dataset_name == "celebvhq":
        ckpt, dm = train_celebvhq(args, config)
        evaluate_celebvhq(args, ckpt, dm)
    elif dataset_name == "biovid":
        # implement biovid evaluation here
        ckpt, dm = train_biovid(args, config)
        evaluate_biovid(args, ckpt, dm, config)
        # pass

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CelebV-HQ evaluation")
    parser.add_argument("--config", type=str, help="Path to CelebV-HQ evaluation config file.")
    parser.add_argument("--data_path", type=str, help="Path to CelebV-HQ dataset.")
    parser.add_argument("--marlin_ckpt", type=str, default=None,
        help="Path to MARLIN checkpoint. Default: None, load from online.")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2000, help="Max epochs to train.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--skip_train", action="store_true", default=False,
        help="Skip training and evaluate only.")
    parser.add_argument("--predict_only", action="store_true", default=False,
                        help="Skip evaluation. Save prediction results only.")

    args = parser.parse_args()
    if args.skip_train:
        assert args.resume is not None

    evaluate(args)
