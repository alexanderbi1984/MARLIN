#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to pretrain the shared encoder and stimulus head on the BioVid dataset.
This pretrained model can then be used as a starting point for multi-task learning.
"""

import os
import json
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from dataset.biovid import BioVidLP
from model.multi_task_coral import MultiTaskCoralClassifier

class StimOnlyWrapper:
    """
    Simple wrapper to make BioVidLP dataset return data in the format expected by the multi-task model,
    but with dummy pain labels set to -1 (unused during pretraining).
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        features, stimulus_labels = self.dataset[idx]
        # Return features, dummy pain labels (-1), and stimulus labels
        return features, torch.tensor(-1, dtype=torch.long), stimulus_labels

class StimOnlyDataModule(pl.LightningDataModule):
    """
    DataModule for pretraining with BioVid stimulus data only.
    """
    def __init__(
        self,
        biovid_root_dir,
        biovid_feature_dir,
        num_stimulus_classes,
        batch_size=32,
        num_workers=4,
        temporal_reduction='mean',
        val_split=0.15  # Validation split ratio
    ):
        super().__init__()
        self.biovid_root_dir = biovid_root_dir
        self.biovid_feature_dir = biovid_feature_dir
        self.num_stimulus_classes = num_stimulus_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.temporal_reduction = temporal_reduction
        self.val_split = val_split

    def setup(self, stage=None):
        # Load full BioVid dataset
        full_dataset = BioVidLP(
            root_dir=self.biovid_root_dir,
            task='multiclass',
            num_classes=self.num_stimulus_classes,
            split='train',
            feature_dir=self.biovid_feature_dir,
            temporal_reduction=self.temporal_reduction
        )
        
        # Calculate train/val split sizes
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        
        # Split dataset into train and validation sets
        self.biovid_train_dataset, self.biovid_val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Wrap datasets for multi-task format
        self.train_dataset = StimOnlyWrapper(self.biovid_train_dataset)
        self.val_dataset = StimOnlyWrapper(self.biovid_val_dataset)
        
        print(f"BioVid dataset split: {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples ({self.val_split*100:.1f}%)")
        
        # Calculate class distribution for training set
        train_labels = [full_dataset[i][1] for i in self.biovid_train_dataset.indices]
        train_class_counts = np.bincount(train_labels, minlength=self.num_stimulus_classes)
        print(f"Training stimulus class distribution: {train_class_counts}")
        
        # Calculate class distribution for validation set
        val_labels = [full_dataset[i][1] for i in self.biovid_val_dataset.indices]
        val_class_counts = np.bincount(val_labels, minlength=self.num_stimulus_classes)
        print(f"Validation stimulus class distribution: {val_class_counts}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

def pretrain_model(args, config):
    # Extract configuration
    model_name = config.get('model_name', 'pretrain_biovid')
    pretrain_model_name = f"{model_name}_pretrain"
    num_pain_classes = config.get('num_pain_classes', 5)  # Needed for model init but not used
    num_stimulus_classes = config.get('num_stimulus_classes', 5)
    biovid_feature_dir = config.get('biovid_feature_dir')
    temporal_reduction = config.get('temporal_reduction', 'mean')
    learning_rate = float(config.get('learning_rate', 1e-4))
    weight_decay = float(config.get('weight_decay', 0.0))
    encoder_hidden_dims = config.get('encoder_hidden_dims', None)
    use_distance_penalty = config.get('use_distance_penalty', False)
    focal_gamma = config.get('focal_gamma', None)
    val_split = float(config.get('val_split', 0.15))  # Default 15% validation split
    
    # Set up checkpoint directory
    checkpoint_dir = os.path.join("ckpt", f"{pretrain_model_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Print configuration
    print(f"\n=== Pretraining Configuration ===")
    print(f"Model Name: {pretrain_model_name}")
    print(f"Stimulus Classes: {num_stimulus_classes}")
    print(f"BioVid Features: {biovid_feature_dir}")
    print(f"Temporal Reduction: {temporal_reduction}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Use Distance Penalty: {use_distance_penalty}")
    print(f"Focal Gamma: {focal_gamma}")
    print(f"Encoder Hidden Dims: {encoder_hidden_dims}")
    print(f"Validation Split: {val_split*100:.1f}%")
    print(f"---------------------------------")
    
    # Create data module
    dm = StimOnlyDataModule(
        biovid_root_dir=args.biovid_data_path,
        biovid_feature_dir=biovid_feature_dir,
        num_stimulus_classes=num_stimulus_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        temporal_reduction=temporal_reduction,
        val_split=val_split
    )
    
    # Setup data module to get feature dimension
    dm.setup()
    
    # Get sample features to determine input dimension
    sample_features, _, _ = dm.train_dataset[0]
    input_dim = sample_features.shape[0]
    
    # Create model
    model = MultiTaskCoralClassifier(
        input_dim=input_dim,
        num_pain_classes=num_pain_classes,
        num_stimulus_classes=num_stimulus_classes,
        encoder_hidden_dims=encoder_hidden_dims,
        learning_rate=learning_rate,
        optimizer_name='AdamW',
        pain_loss_weight=0.0,  # Zero weight for pain task during pretraining
        stim_loss_weight=1.0,  # Full weight for stimulus task
        weight_decay=weight_decay,
        use_distance_penalty=use_distance_penalty,
        focal_gamma=focal_gamma
    )
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{pretrain_model_name}-{{epoch:02d}}-{{val_stim_QWK:.4f}}",
        monitor="val_stim_QWK",  # Changed to monitor validation metric
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_stim_QWK",  # Changed to monitor validation metric
        mode="max",
        patience=15,  # Changed patience to 15 epochs
        verbose=True
    )
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=pretrain_model_name,
        default_hp_metric=False
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.n_gpus > 0 else "cpu",
        devices=args.n_gpus if args.n_gpus > 0 else None,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision=args.precision,
        deterministic=True,
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(model, dm)
    
    # Get best checkpoint path
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nBest model saved at: {best_model_path}")
    
    # Save additional metadata about the checkpoint
    checkpoint_info = {
        "best_model_path": best_model_path,
        "best_val_stim_qwk": checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None,
        "val_split": val_split,
        "config": config
    }
    
    with open(os.path.join(checkpoint_dir, f"{pretrain_model_name}_info.json"), 'w') as f:
        json.dump(checkpoint_info, f, indent=2)
    
    print(f"Pretraining completed. Best validation stimulus QWK: {checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else 'Unknown'}")
    
    return best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a model on BioVid stimulus data")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    parser.add_argument('--biovid_data_path', type=str, required=True, help='Path to BioVid data')
    parser.add_argument('--n_gpus', type=int, default=1, help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum number of epochs for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32, 64], help='Floating point precision')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run pretraining
    best_checkpoint = pretrain_model(args, config)
    print(f"Pretraining complete. Best checkpoint: {best_checkpoint}") 