import argparse
import os
import random
import pandas as pd
import yaml
import warnings
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import math
import glob

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics import Accuracy, F1Score

from model.pain_transformer import PainEstimationModel
from dataset.video_dataset import VideoDataset

# --- Helper to parse Syracuse Metadata ---

def pain_to_class(pain: float, cutoffs: List[float]) -> Optional[int]:
    """Match baseline semantics in dataset/syracuse_bag.py:
    class = number of thresholds strictly less than pain (pain > th increments class).
    This means boundary values (e.g. pain==1.1) stay in the lower class.
    """
    if pain is None:
        return None
    c = 0
    for th in cutoffs:
        if pain > th:
            c += 1
        else:
            break
    return c

def parse_syracuse_metadata(
    meta_path: str, 
    video_root: str, 
    video_col: str = "file_name", 
    label_col: str = "pain_level",
    subject_col: str = "subject_id",
    cutoffs: List[float] = [1.1, 3.1, 5.1, 7.1],
    exclude_video_ids: List[str] = [],
    aug_root: Optional[str] = None
) -> pd.DataFrame:
    """
    Parses Syracuse metadata, handles dirty labels, performs binning, and finds aug paths.
    """
    print(f"Parsing metadata from {meta_path}...")
    if meta_path.endswith('.csv'):
        df = pd.read_csv(meta_path)
    else:
        df = pd.read_excel(meta_path)
    
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    
    # 1. Clean Labels (Strict Mode - Align with Baseline)
    # Baseline logic: try float(x), if fail -> None. No fancy parsing.
    def clean_label(val):
        try:
            v = float(val)
            if np.isnan(v):
                return None
            return v
        except:
            return None

    df['clean_label'] = df[label_col].apply(clean_label)
    
    # Drop rows with invalid labels
    initial_len = len(df)
    df = df.dropna(subset=['clean_label'])
    print(f"Dropped {initial_len - len(df)} samples due to invalid labels.")
    
    # 2. Binning (Regression -> Classification)
    # Class 0: < 1.1
    # Class 1: 1.1 - 3.1
    # ...
    # Match baseline: pain_to_class uses strict '>' thresholds
    df['target'] = df['clean_label'].apply(lambda x: pain_to_class(x, cutoffs))
    
    # 3. Clean Paths
    # Meta has "IMG_0003.MP4", Folder is "IMG_0003"
    def get_full_path(fname):
        fname = str(fname).strip()
        base_name = fname.split(".")[0] if "." in fname else fname
        
        path_dir = os.path.join(video_root, base_name)
        if os.path.isdir(path_dir):
            return path_dir
        return None

    df['full_path'] = df[video_col].apply(get_full_path)
    
    # Drop missing files
    missing_mask = df['full_path'].isnull()
    if missing_mask.sum() > 0:
        print(f"Warning: {missing_mask.sum()} videos not found on disk. Examples: {df[missing_mask][video_col].head().tolist()}")
        df = df[~missing_mask]
        
    # 4. Clean Subjects
    # Remove 'nan', handle '33?'
    def clean_subject(val):
        s = str(val).strip()
        if s.lower() == 'nan': return None
        if s.endswith('?'): return s[:-1]
        return s
        
    df['subject'] = df[subject_col].apply(clean_subject)
    df = df.dropna(subset=['subject'])
    
    # 5. Filter Excluded Videos (Strict Mode - Align with Baseline)
    if exclude_video_ids:
        print(f"Applying exclusion list: {len(exclude_video_ids)} videos")
        # Extract base name from exclude list just in case (e.g. IMG_0006)
        exclude_set = set([str(x).strip() for x in exclude_video_ids])
        
        # Helper to extract base name from df['video_col'] or df['full_path']
        # df['file_name'] is like 'IMG_0009.MP4'
        def get_base(fname):
            fname = str(fname).strip()
            if "." in fname:
                return fname.split(".")[0]
            return fname
            
        df['video_base'] = df[video_col].apply(get_base)
        
        before_excl = len(df)
        df = df[~df['video_base'].isin(exclude_set)]
        print(f"Excluded {before_excl - len(df)} videos based on exclude_video_ids.")
    
    # 6. Find Augmentation Paths
    # Look for IMG_XXXX_1, IMG_XXXX_2 in aug_root
    def find_aug_paths(full_path):
        if not aug_root or not os.path.isdir(aug_root):
            return []
        
        if not full_path: return []
        
        # full_path is like .../IMG_0003
        base_name = os.path.basename(full_path) # IMG_0003
        
        # Search for base_name_* in aug_root
        # Pattern: aug_root/IMG_0003_*
        # But be careful not to match IMG_00030
        pattern = os.path.join(aug_root, f"{base_name}_*")
        candidates = glob.glob(pattern)
        
        valid_augs = []
        for c in candidates:
            # Check if it is a directory
            if os.path.isdir(c):
                # Verify it's a swap aug (suffix is number)
                # e.g. IMG_0003_1
                c_base = os.path.basename(c)
                suffix = c_base.replace(base_name + "_", "")
                if suffix.isdigit():
                    valid_augs.append(c)
        
        return valid_augs

    if aug_root:
        print(f"Searching for augmented views in {aug_root}...")
        df['aug_paths'] = df['full_path'].apply(find_aug_paths)
        
        # Debug: check how many have augs
        has_aug = df['aug_paths'].apply(len) > 0
        print(f"Found augmented views for {has_aug.sum()} / {len(df)} videos.")
    else:
        df['aug_paths'] = [[] for _ in range(len(df))]

    print(f"Final dataset size: {len(df)}")
    return df

# --- Lightning Module (Same as before, but ensure head reset) ---

class PainTransformerFinetuneModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 50,
        warmup_epochs: int = 5,
        smoothing: float = 0.1,
        pretrained_checkpoint: Optional[str] = None,
        freeze_mode: str = "none",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model (will load ImageNet weights into backbone)
        self.model = PainEstimationModel(num_classes=num_classes)
        
        # Load BioVid Pretrained Weights if provided
        if pretrained_checkpoint and os.path.isfile(pretrained_checkpoint):
            print(f"Loading BioVid weights from {pretrained_checkpoint}...")
            checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            # Filter logic: Load everything EXCEPT head
            # BioVid might be 5 classes, Syracuse is 5 classes, BUT the meaning is different.
            # It's safer to re-init head.
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    key = k[6:]
                else:
                    key = k
                
                # Skip head weights
                if 'head' in key:
                    continue
                new_state_dict[key] = v
            
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"BioVid weights loaded. Missing (should include head): {len(missing)}")
            
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

        # Freeze strategy to mitigate overfitting (configurable)
        self.freeze_mode = str(freeze_mode).lower().strip()
        if self.freeze_mode not in ("none", "head", "temporal", "spatial"):
            raise ValueError(f"Invalid freeze_mode: {freeze_mode}. Expected one of: none/head/temporal/spatial")
        self._apply_freeze_mode()
        
        # Metrics
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, average='micro')
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average='micro')
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def _apply_freeze_mode(self) -> None:
        """Apply freeze strategy by toggling requires_grad on submodules.

        - none: train all
        - head: train head only (freeze spatial + temporal)
        - temporal: train temporal + head (freeze spatial)
        - spatial: train spatial + head (freeze temporal)
        """
        # default: unfreeze all
        for p in self.model.parameters():
            p.requires_grad = True

        if self.freeze_mode == "none":
            return

        if self.freeze_mode == "head":
            # use existing helper if present
            if hasattr(self.model, "freeze_backbone"):
                self.model.freeze_backbone()
            else:
                for p in self.model.spatial_encoder.parameters():
                    p.requires_grad = False
                for p in self.model.temporal_transformer.parameters():
                    p.requires_grad = False
                for p in self.model.head.parameters():
                    p.requires_grad = True
            return

        if self.freeze_mode == "temporal":
            for p in self.model.spatial_encoder.parameters():
                p.requires_grad = False
            for p in self.model.temporal_transformer.parameters():
                p.requires_grad = True
            for p in self.model.head.parameters():
                p.requires_grad = True
            return

        if self.freeze_mode == "spatial":
            for p in self.model.spatial_encoder.parameters():
                p.requires_grad = True
            for p in self.model.temporal_transformer.parameters():
                p.requires_grad = False
            for p in self.model.head.parameters():
                p.requires_grad = True
            return

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True)
        self.log('val_f1', self.val_f1, prog_bar=True, on_epoch=True)
        
        # Cache for distribution check
        if not hasattr(self, 'val_preds_cache'):
            self.val_preds_cache = []
        self.val_preds_cache.append(preds.detach().cpu())
        return preds, y

    def on_validation_epoch_end(self):
        if hasattr(self, 'val_preds_cache') and self.val_preds_cache:
            all_preds = torch.cat(self.val_preds_cache)
            unique, counts = torch.unique(all_preds, return_counts=True)
            dist = dict(zip(unique.tolist(), counts.tolist()))
            if self.trainer.is_global_zero:
                print(f"\n[Epoch {self.current_epoch}] Val Pred Dist: {dist}", flush=True)
            self.val_preds_cache.clear()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_epoch=True)
        return preds, y

    def predict_step(self, batch, batch_idx):
        x, y, sample_ids = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        # Keep keys compatible with downstream aggregation
        return {"sample_ids": sample_ids, "y_true": y, "y_pred": preds}

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()

    def configure_optimizers(self):
        # Layer-wise learning rate could be implemented here, but let's stick to global LR first
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters. Check freeze_mode configuration.")
        optimizer = torch.optim.AdamW(trainable_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--biovid_checkpoint", type=str, default=None, help="Path to best BioVid checkpoint")
    # Fold-level parallelization helpers (one process can run a subset of subjects)
    parser.add_argument("--list_subjects", action="store_true", help="Print discovered subject IDs then exit.")
    parser.add_argument("--only_subject", type=str, default=None, help="Run only one LOSO test subject (e.g. '11').")
    parser.add_argument("--shard_id", type=int, default=None, help="Shard index for subject slicing (0-based).")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards for subject slicing.")
    parser.add_argument("--no_summary", action="store_true", help="Skip global summary aggregation at the end.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42)

    # 1. Parse Metadata
    use_aug = bool(config.get("use_aug", False))
    aug_root = config.get('aug_video_root', None) if use_aug else None

    df = parse_syracuse_metadata(
        config['meta_excel'],
        config['video_root'],
        video_col=config['video_col'],
        label_col=config['label_col'],
        subject_col=config['subject_col'],
        cutoffs=config['cutoffs'],
        exclude_video_ids=config.get('exclude_video_ids', []),
        aug_root=aug_root
    )

    # 2. LOSO Loop
    subjects = sorted([str(s) for s in df['subject'].unique().tolist()])
    print(f"Found {len(subjects)} subjects: {subjects}")

    if args.list_subjects:
        # Stable printed list for scripting
        for s in subjects:
            print(s)
        return
    
    # Check if we should limit subjects (debug mode)
    limit = config.get('limit_subjects', None)
    if limit:
        subjects = subjects[:limit]
        print(f"Limiting to first {limit} subjects.")

    # Optional: run only one subject
    if args.only_subject is not None:
        only = str(args.only_subject).strip()
        if only not in subjects:
            raise ValueError(f"--only_subject={only} not found in discovered subjects: {subjects}")
        subjects = [only]

    # Optional: shard subjects for parallel runs
    if (args.shard_id is None) != (args.num_shards is None):
        raise ValueError("Both --shard_id and --num_shards must be provided together.")
    if args.shard_id is not None and args.num_shards is not None:
        shard_id = int(args.shard_id)
        num_shards = int(args.num_shards)
        if num_shards <= 0:
            raise ValueError("--num_shards must be > 0")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"--shard_id must be in [0, {num_shards-1}]")
        subjects = subjects[shard_id::num_shards]
        print(f"Sharded subjects: shard_id={shard_id}/{num_shards} -> {subjects}")

    for test_subj in subjects:
        print(f"\n=== LOSO Fold: Subject {test_subj} ===")
        
        # Split Data (Nested CV logic)
        # 1. Test set: Current subject
        test_df = df[df['subject'] == test_subj]
        
        # 2. Remaining subjects
        remaining_subjects = [s for s in subjects if s != test_subj]
        
        # 3. Validation Split (e.g. 15% of remaining subjects)
        # Deterministic shuffle for reproducibility
        rng = random.Random(42 + hash(test_subj) % 1000) # Seed based on fold
        rng.shuffle(remaining_subjects)
        
        n_val = max(1, int(0.15 * len(remaining_subjects)))
        val_subjects = remaining_subjects[:n_val]
        train_subjects = remaining_subjects[n_val:]
        
        val_df = df[df['subject'].isin(val_subjects)]
        train_df = df[df['subject'].isin(train_subjects)]
        
        print(f"Split: Train={len(train_subjects)} subj, Val={len(val_subjects)} subj, Test={test_subj}")
        
        clip_level = bool(config.get("clip_level", False))
        clip_len_frames = int(config.get("clip_len_frames", 150))
        clip_stride_frames = int(config.get("clip_stride_frames", 15))

        train_ds = VideoDataset(
            train_df['full_path'].tolist(), 
            train_df['target'].tolist(), 
            is_train=True,
            aug_paths_list=train_df['aug_paths'].tolist() if use_aug else None,
            aug_ratio=float(config.get('aug_ratio', 0.0)) if use_aug else 0.0,
            clip_level=clip_level,
            clip_len_frames=clip_len_frames,
            clip_stride_frames=clip_stride_frames,
        )
        val_ds = VideoDataset(
            val_df['full_path'].tolist(),
            val_df['target'].tolist(),
            is_train=False,
            clip_level=clip_level,
            clip_len_frames=clip_len_frames,
            clip_stride_frames=clip_stride_frames,
        )
        test_ds = VideoDataset(
            test_df['full_path'].tolist(),
            test_df['target'].tolist(),
            is_train=False,
            clip_level=clip_level,
            clip_len_frames=clip_len_frames,
            clip_stride_frames=clip_stride_frames,
        )
        
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        
        model = PainTransformerFinetuneModule(
            num_classes=config['num_classes'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            max_epochs=config['max_epochs'],
            pretrained_checkpoint=args.biovid_checkpoint,
            freeze_mode=config.get("freeze_mode", "none"),
        )
        
        checkpoint_cb = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_last=True,
            filename=f"subj_{test_subj}" + "-{epoch}-{val_acc:.2f}",
        )
        early_stop = EarlyStopping(monitor="val_acc", mode="max", patience=config['patience'])
        logger = TensorBoardLogger("logs_syracuse_finetune", name=f"subj_{test_subj}")
        
        strategy = DDPStrategy(find_unused_parameters=False) if config['gpus'] > 1 else None
        
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=config['gpus'],
            max_epochs=config['max_epochs'],
            accumulate_grad_batches=config['accumulate_grad_batches'],
            precision=config['precision'],
            strategy=strategy,
            logger=logger,
            callbacks=[checkpoint_cb, early_stop],
            log_every_n_steps=5
        )
        
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader, ckpt_path="best")
        
        # 3. Predict & Aggregate Results
        print(f"Predicting on subject {test_subj}...")
        predictions = trainer.predict(model, test_loader, ckpt_path="best")
        
        # Collect all clip predictions
        clip_results = []
        for batch in predictions:
            sample_ids = batch['sample_ids']
            y_ts = batch['y_true'].cpu().numpy()
            y_ps = batch['y_pred'].cpu().numpy()
            
            for i in range(len(sample_ids)):
                sid = sample_ids[i]
                # sid is like "IMG_0003_clip_0000"
                vid_id = str(sid).split("_clip_")[0]
                clip_results.append({
                    "video_id": vid_id,
                    "sample_id": str(sid),
                    "y_true": int(y_ts[i]),
                    "y_pred": int(y_ps[i])
                })
        
        # Aggregate to Video Level (Majority Vote)
        from collections import Counter
        video_level_preds = {}
        for res in clip_results:
            vid = res['video_id']
            if vid not in video_level_preds:
                video_level_preds[vid] = {"y_true": res['y_true'], "votes": []}
            video_level_preds[vid]["votes"].append(res['y_pred'])
            
        # Calculate Video Acc and Save CSV
        correct_vid = 0
        total_vid = 0
        
        save_dir = os.path.join("logs_syracuse_finetune", "results")
        os.makedirs(save_dir, exist_ok=True)
        
        csv_path = os.path.join(save_dir, f"preds_subj_{test_subj}.csv")
        with open(csv_path, 'w') as f:
            f.write("video_id,y_true,y_pred_majority,vote_ratio,num_clips\n")
            for vid, data in video_level_preds.items():
                y_true = data['y_true']
                votes = data['votes']
                if not votes: continue
                
                # Majority vote
                counts = Counter(votes)
                y_pred_maj = counts.most_common(1)[0][0]
                ratio = counts[y_pred_maj] / len(votes)
                
                if y_pred_maj == y_true:
                    correct_vid += 1
                total_vid += 1
                
                f.write(f"{vid},{y_true},{y_pred_maj},{ratio:.4f},{len(votes)}\n")
                
        vid_acc = correct_vid / total_vid if total_vid > 0 else 0
        print(f"Subject {test_subj} Video-Level Accuracy: {vid_acc:.4f} ({correct_vid}/{total_vid})")

    # --- 3. Global Summary ---
    # NOTE: When running subsets (only_subject/shards) in parallel, global summary can race and/or mix runs.
    if args.no_summary or args.only_subject is not None or (args.shard_id is not None):
        print("\nSkipping global summary (subset run).")
        return

    print("\n=== Generating Global Summary ===")
    results_dir = os.path.join("logs_syracuse_finetune", "results")
    all_preds_files = [f for f in os.listdir(results_dir) if f.startswith("preds_subj_") and f.endswith(".csv")]
    
    total_correct = 0
    total_videos = 0
    all_y_true = []
    all_y_pred = []
    
    # Store per-subject metrics
    subject_metrics = []

    for f in all_preds_files:
        path = os.path.join(results_dir, f)
        df_res = pd.read_csv(path)
        
        # Calculate subject-level acc
        subj_correct = (df_res['y_true'] == df_res['y_pred_majority']).sum()
        subj_total = len(df_res)
        subj_acc = subj_correct / subj_total if subj_total > 0 else 0
        
        # Extract subject ID from filename (preds_subj_XX.csv)
        subj_id = f.replace("preds_subj_", "").replace(".csv", "")
        subject_metrics.append({"subject": subj_id, "acc": subj_acc, "total": subj_total})
        
        total_correct += subj_correct
        total_videos += subj_total
        all_y_true.extend(df_res['y_true'].tolist())
        all_y_pred.extend(df_res['y_pred_majority'].tolist())
        
    # Global Metrics
    global_acc = total_correct / total_videos if total_videos > 0 else 0
    
    from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
    macro_f1 = f1_score(all_y_true, all_y_pred, average='macro')
    weighted_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
    
    print(f"\nGlobal Results across {len(all_preds_files)} subjects:")
    print(f"Total Videos: {total_videos}")
    print(f"Global Accuracy: {global_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    # Subject-level Statistics
    subj_accs = [m['acc'] for m in subject_metrics]
    print(f"Mean Subject Accuracy: {np.mean(subj_accs):.4f} +/- {np.std(subj_accs):.4f}")
    
    # Save Summary CSV
    summary_df = pd.DataFrame(subject_metrics)
    summary_df.loc['MEAN'] = summary_df.mean(numeric_only=True)
    summary_df.loc['MEAN', 'subject'] = 'MEAN'
    summary_df.to_csv(os.path.join(results_dir, "summary_metrics.csv"), index=False)
    print(f"Summary saved to {os.path.join(results_dir, 'summary_metrics.csv')}")

    # Save Global Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(results_dir, "global_confusion_matrix.csv"), index=False)
    print(f"Global Confusion Matrix saved to {os.path.join(results_dir, 'global_confusion_matrix.csv')}")

if __name__ == "__main__":
    main()
