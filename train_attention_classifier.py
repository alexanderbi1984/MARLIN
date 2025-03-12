import argparse
import os
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.biovid import BioVidDataModule
from model.attention_classifier import AttentionClassifier
from util.seed import Seed
from util.earlystop_lr import EarlyStoppingLR
from util.lr_logger import LrLogger
from util.system_stats_logger import SystemStatsLogger


def main():
    parser = argparse.ArgumentParser("Train Attention-based Classifier for Pain Assessment")

    # Data and model settings
    parser.add_argument("--data_path", type=str, required=True, help="Path to BioVid dataset")
    parser.add_argument("--marlin_ckpt", type=str, default=None, help="Path to MARLIN checkpoint")
    parser.add_argument("--backbone", type=str, default="marlin_vit_base_ytf", help="Backbone model type")
    parser.add_argument("--output_dir", type=str, default="./runs", help="Output directory")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=2000, help="Maximum number of epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--finetune", action="store_true", help="Whether to finetune backbone")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")

    # Task settings
    parser.add_argument("--task", type=str, default="multiclass",
                        choices=["binary", "multiclass", "regression"],
                        help="Classification task type")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")

    # Attention settings
    parser.add_argument("--attention_dim", type=int, default=64, help="Attention dimension")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")

    # Parse arguments
    args = parser.parse_args()
    resume_ckpt = args.resume
    # Set random seed for reproducibility
    Seed.set(args.seed)

    # Determine acceleration settings
    accelerator = "cpu" if args.gpus == 0 else "gpu"
    strategy = "auto" if args.gpus <= 1 else "ddp"

    # Create model name based on settings
    model_name = f"attention_{args.backbone}_{args.task}_{args.num_classes}cls"
    if args.num_heads > 1:
        model_name += f"_mha{args.num_heads}"
    model_name += f"_lr{args.learning_rate}"

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)

    # Set up data module
    if args.finetune:
        # For finetuning, load raw video data
        from model.config import resolve_config
        backbone_config = resolve_config(args.backbone)

        dm = BioVidDataModule(
            root_dir=args.data_path,
            load_raw=True,
            task=args.task,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            clip_frames=backbone_config.n_frames,
            temporal_sample_rate=2
        )
    else:
        # For linear probing, load pre-extracted features
        # Important: Make sure we have sequence-preserving features (not temporally reduced)
        # This requires that features were extracted with keep_seq=True
        dm = BioVidDataModule(
            root_dir=args.data_path,
            load_raw=False,
            task=args.task,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            feature_dir=args.backbone,
            temporal_reduction="none"  # Critical: Don't reduce, we need clip sequence for attention
        )

    # Create attention-based classifier
    model = AttentionClassifier(
        num_classes=args.num_classes,
        backbone=args.backbone,
        finetune=args.finetune,
        marlin_ckpt=args.marlin_ckpt,
        task=args.task,
        learning_rate=args.learning_rate,
        distributed=args.gpus > 1,
        attention_dim=args.attention_dim,
        num_heads=args.num_heads,
        dropout=args.dropout
    )

    # Set up checkpointing
    monitor_metric = "val_auc" if args.task in ["binary", "multiclass"] else "val_mse"
    mode = "max" if args.task in ["binary", "multiclass"] else "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, model_name),
        save_last=True,
        filename=model_name + "-{epoch}-{" + monitor_metric + ":.3f}",
        monitor=monitor_metric,
        mode=mode
    )

    # Set up early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=100,
        mode=mode
    )

    # Set up logger
    logger = TensorBoardLogger(args.output_dir, name=model_name)

    # Create trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=args.gpus if args.gpus > 0 else None,
        strategy=strategy,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            LrLogger(),
            EarlyStoppingLR(1e-8),
            SystemStatsLogger()
        ],
        log_every_n_steps=10,
        precision=32
    )

    # Train model
    if resume_ckpt:  # Check if a checkpoint path is provided
        trainer.fit(model, dm, ckpt_path=resume_ckpt)  # Resume training from the checkpoint
    else:
        trainer.fit(model, dm)  # Start training from scratch

    # Test model
    test_results = trainer.test(model, datamodule=dm)
    print(f"Test results: {test_results}")

    # Print path to best model
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    main()