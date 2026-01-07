import argparse
import os
import random
import pandas as pd
import yaml
import warnings
from typing import List, Optional, Dict, Any, Tuple
import math

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Specific filter for the pkg_resources warning if needed, but general UserWarning usually covers it in PL logs
# warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, F1Score

from model.pain_transformer import PainEstimationModel
from dataset.video_dataset import VideoDataset

# --- Helper to parse Excel Metadata (Aligning with your existing workflow) ---

def parse_metadata(
    meta_path: str, 
    video_root: str, 
    split_col: str = "split", 
    video_col: str = "file_name", 
    label_col: str = "pain_level",
    subject_col: str = "subject_id",
    task: str = "ordinal",
    num_classes: int = 5
) -> pd.DataFrame:
    """
    Parses the Excel/CSV metadata file.
    Adaptively finds video directories or files.
    """
    if meta_path.endswith('.csv'):
        df = pd.read_csv(meta_path)
    else:
        df = pd.read_excel(meta_path)
    
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    
    # Check required columns
    required = [video_col, label_col]
    for c in required:
        if c not in df.columns:
            # Try fuzzy matching or specific fallbacks
            if c == "file_name" and "filename" in df.columns:
                video_col = "filename"
            elif c == "file_name" and "video_id" in df.columns:
                video_col = "video_id"
            else:
                print(f"Column '{c}' not found. Available: {df.columns}")
                raise ValueError(f"Missing required column '{c}' in metadata file.")

    # Filter out missing labels or files
    df = df.dropna(subset=[video_col, label_col])
    
    # Normalize Paths
    def get_full_path(fname):
        fname = str(fname).strip()
        # Remove extension if present in metadata but directory assumes base name
        if "." in fname:
            base_name = fname.split(".")[0]
        else:
            base_name = fname
            
        # Prioritize Directory Search
        path_dir = os.path.join(video_root, base_name)
        if os.path.isdir(path_dir):
            return path_dir
            
        # Fallback to file search
        path_file = os.path.join(video_root, fname)
        if os.path.exists(path_file):
            return path_file
            
        # Fallback to trying extensions
        for ext in ['.mp4', '.avi', '.mov', '.bmp', '.png', '.jpg']:
            if os.path.exists(path_file + ext):
                return path_file + ext
                
        # Default to constructed dir path even if missing (filtered later)
        return path_dir

    df['full_path'] = df[video_col].apply(get_full_path)
    
    # Filter out non-existent paths (dir or file)
    def exists_check(p):
        return os.path.isdir(p) or os.path.isfile(p)
        
    exists_mask = df['full_path'].apply(exists_check)
    
    if not exists_mask.all():
        missing_count = (~exists_mask).sum()
        print(f"Warning: {missing_count} videos/directories not found in {video_root}. Examples:", df[~exists_mask][video_col].head().tolist())
        df = df[exists_mask]

    # Process Labels
    if task == "ordinal" or task == "multiclass":
        # Convert pain level to class index
        try:
             df['target'] = df[label_col].astype(float).astype(int)
             # Clamp to num_classes - 1
             df['target'] = df['target'].apply(lambda x: min(max(0, x), num_classes - 1))
        except:
             print("Error converting labels to integer targets.")
             raise
    else:
        raise ValueError("Only ordinal/multiclass supported in this demo.")

    # Ensure subject column exists for LOSO
    if subject_col not in df.columns:
        print(f"Warning: Subject column '{subject_col}' not found. Filling with 'unknown'.")
        df['subject'] = 'unknown'
    else:
        df['subject'] = df[subject_col].astype(str)

    return df

# --- Lightning Module ---

class PainTransformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        max_epochs: int = 200,
        warmup_epochs: int = 5,
        smoothing: float = 0.1,
        pretrained_checkpoint: Optional[str] = None,
        freeze_encoder: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = PainEstimationModel(num_classes=num_classes)
        
        # Load weights if provided
        if pretrained_checkpoint and os.path.isfile(pretrained_checkpoint):
            self.model.load_pretrained(pretrained_checkpoint)
            
        # Freeze if requested
        if freeze_encoder:
            self.model.freeze_backbone()
            
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

        # Metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average='micro')
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average='micro')
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Log basic metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        # Update torchmetrics
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        
        # Keep track of preds for simple distribution check
        if not hasattr(self, 'val_preds_cache'):
             self.val_preds_cache = []
        self.val_preds_cache.append(preds.cpu())
        
        return preds, y

    def on_validation_epoch_end(self):
        if hasattr(self, 'val_preds_cache') and self.val_preds_cache:
            # Concatenate all batches on this device
            all_preds = torch.cat(self.val_preds_cache)
            
            # Count occurrences
            unique, counts = torch.unique(all_preds, return_counts=True)
            dist = dict(zip(unique.tolist(), counts.tolist()))
            total = len(all_preds)
            
            # Log distribution ratios to TensorBoard
            if self.trainer.is_global_zero:
                print(f"\n[Epoch {self.current_epoch}] Validation Prediction Distribution: {dist}", flush=True)
                
                # Log ratios to TensorBoard
                for cls_idx in range(self.hparams.num_classes):
                    count = dist.get(cls_idx, 0)
                    self.log(f'debug_class_dist/class_{cls_idx}', count / total, on_epoch=True, rank_zero_only=True)

            self.val_preds_cache.clear()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        
        # Update torchmetrics
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, prog_bar=True, on_step=False, on_epoch=True)
        
        return preds, y

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Warmup + Cosine Decay
        try:
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
            
            warmup_iters = self.warmup_epochs # approximation if interval=epoch
            
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=self.max_epochs - warmup_iters, eta_min=1e-6
            )
            
            scheduler = SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_iters]
            )
        except ImportError:
            # Fallback
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Train End-to-End Pain Transformer (TNT + Temporal)")
    
    # Config File Support
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")

    # Path Args (Aligned with your existing usage)
    parser.add_argument("--meta_excel", type=str, default=None, help="Path to Excel file with columns.")
    parser.add_argument("--video_root", type=str, default=None, help="Root directory containing raw video files.")
    
    # Column Mappings
    parser.add_argument("--video_col", type=str, default="file_name")
    parser.add_argument("--label_col", type=str, default="pain_level")
    parser.add_argument("--subject_col", type=str, default="subject_id")
    parser.add_argument("--split_col", type=str, default="split", help="Column for train/val/test split if not using LOSO")
    
    # Training Args
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16", help="Precision (16, 32, etc.)")
    parser.add_argument("--gradient_clip_val", type=float, default=0.0, help="Gradient clipping value (0.0 to disable)")
    
    # Mode
    parser.add_argument("--loso", action="store_true", help="Run Leave-One-Subject-Out Cross-Validation")
    
    # Transfer Learning Args
    parser.add_argument("--pretrained_checkpoint", type=str, default=None, help="Path to BioVid .ckpt to load weights from")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze Spatial and Temporal encoders, training only the head")

    # Parse config first if present
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        if os.path.exists(temp_args.config):
            with open(temp_args.config, 'r') as f:
                config_data = yaml.safe_load(f)
                parser.set_defaults(**config_data)
            print(f"Loaded configuration from {temp_args.config}")
        else:
            print(f"Warning: Config file {temp_args.config} not found.")

    args = parser.parse_args()
    
    # Validation after merging config and CLI
    if not args.meta_excel or not args.video_root:
        parser.error("meta_excel and video_root are required (via CLI or config file)")

    pl.seed_everything(42)
    
    print(f"Loading metadata from {args.meta_excel}...")
    df = parse_metadata(
        args.meta_excel, 
        args.video_root, 
        video_col=args.video_col, 
        label_col=args.label_col,
        subject_col=args.subject_col,
        num_classes=args.num_classes
    )
    print(f"Loaded {len(df)} samples.")

    # PL compatibility: try converting to int (e.g. "16" -> 16), else keep string ("bf16")
    try:
        precision_val = int(args.precision)
    except ValueError:
        precision_val = args.precision

    # DDP Strategy config to fix unused params warning
    from pytorch_lightning.strategies import DDPStrategy
    # Only use strategy if multiple GPUs are requested
    strategy_obj = DDPStrategy(find_unused_parameters=False) if args.gpus > 1 else None

    if args.loso:
        subjects = df['subject'].unique()
        print(f"Starting LOSO evaluation on {len(subjects)} subjects: {subjects}")
        
        for test_subject in subjects:
            print(f"\n=== Fold: Subject {test_subject} ===")
            
            # LOSO Split
            train_df = df[df['subject'] != test_subject]
            test_df = df[df['subject'] == test_subject]
            
            if len(test_df) == 0:
                continue

            # In production, you might want to split train_df further for validation
            # Here we use the test fold as validation for simplicity/monitoring
            
            train_ds = VideoDataset(
                video_paths=train_df['full_path'].tolist(), 
                labels=train_df['target'].tolist(), 
                is_train=True
            )
            val_ds = VideoDataset(
                video_paths=test_df['full_path'].tolist(), 
                labels=test_df['target'].tolist(), 
                is_train=False
            )
            
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            model = PainTransformerLightningModule(
                num_classes=args.num_classes, 
                max_epochs=args.max_epochs,
                pretrained_checkpoint=args.pretrained_checkpoint,
                freeze_encoder=args.freeze_encoder
            )
            
            checkpoint_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True, filename=f"subject_{test_subject}"+"-{epoch}-{val_acc:.2f}")
            early_stop_cb = EarlyStopping(monitor="val_acc", mode="max", patience=args.patience, verbose=True)
            logger = TensorBoardLogger("logs_tnt_loso", name=f"subj_{test_subject}")
            
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=args.gpus,
                max_epochs=args.max_epochs,
                accumulate_grad_batches=args.accumulate_grad_batches,
                precision=precision_val,
                strategy=strategy_obj,
                logger=logger,
                callbacks=[checkpoint_cb, early_stop_cb],
                log_every_n_steps=10,
                gradient_clip_val=args.gradient_clip_val
            )
            
            trainer.fit(model, train_loader, val_loader)
            trainer.test(model, val_loader, ckpt_path="best")

    else:
        print("Running Fixed Split Training (using 'split' column if available)...")
        if args.split_col in df.columns:
            # Case insensitive match for 'train', 'val', 'test'
            train_df = df[df[args.split_col].astype(str).str.lower().str.contains('train')]
            val_df = df[df[args.split_col].astype(str).str.lower().str.contains('val')]
            test_df = df[df[args.split_col].astype(str).str.lower().str.contains('test')]
            
            print(f"Splits found: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        else:
            print(f"Split column '{args.split_col}' not found. Using random 80/20 split on available data.")
            # Simple random split
            msk = torch.rand(len(df)) < 0.8
            train_df = df[msk.numpy()]
            val_df = df[~msk.numpy()]
            test_df = val_df # Reuse val as test
        
        train_ds = VideoDataset(train_df['full_path'].tolist(), train_df['target'].tolist(), is_train=True)
        val_ds = VideoDataset(val_df['full_path'].tolist(), val_df['target'].tolist(), is_train=False)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        model = PainTransformerLightningModule(
            num_classes=args.num_classes, 
            max_epochs=args.max_epochs,
            pretrained_checkpoint=args.pretrained_checkpoint,
            freeze_encoder=args.freeze_encoder
        )
        
        checkpoint_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True)
        early_stop_cb = EarlyStopping(monitor="val_acc", mode="max", patience=args.patience, verbose=True)
        logger = TensorBoardLogger("logs_tnt_fixed", name="fixed_split")
        
        trainer = pl.Trainer(
            accelerator="gpu", devices=args.gpus, 
            max_epochs=args.max_epochs, accumulate_grad_batches=args.accumulate_grad_batches,
            precision=precision_val,
            strategy=strategy_obj,
            logger=logger, callbacks=[checkpoint_cb, early_stop_cb],
            gradient_clip_val=args.gradient_clip_val
        )
        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
