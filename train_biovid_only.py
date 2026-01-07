import argparse
import os
import csv
from typing import Optional, Dict, Any, List, Set, Union

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import Dataset, DataLoader

from dataset.syracuse_bag import collate_mil_batch, pain_to_class
from model.mil_coral_xformer import MILCoralTransformer


class ExcelFeatureDataset(Dataset):
    def __init__(
        self,
        meta_df: pd.DataFrame,
        feature_root: str,
        feature_suffix: str = "_windows.npy",
        combo_name: str = "biovid",
        cutoffs: Optional[List[float]] = None,
        video_col: str = "video_id",
        label_col: str = "pain_level",
    ):
        super().__init__()
        self.feature_root = feature_root
        self.feature_suffix = feature_suffix
        self.combo_name = combo_name
        self.cutoffs = cutoffs
        
        self.entries: List[Dict[str, Any]] = []
        
        for _, row in meta_df.iterrows():
            vid = str(row[video_col]).strip()
            # Heuristic: remove '_aligned' suffix if present to match AU filenames
            if vid.endswith("_aligned"):
                vid = vid[:-8]
            if not vid:
                continue
            label = row[label_col]
            if pd.isna(label):
                continue
            self.entries.append({"video_id": vid, "label": float(label)})

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        video_id = entry["video_id"]
        label = entry["label"]
        path = os.path.join(self.feature_root, f"{video_id}{self.feature_suffix}")
        if not os.path.isfile(path):
            # Try appending combo_name or other heuristic if needed, 
            # but standard BioVid layout usually matches video_id directly.
            raise FileNotFoundError(f"Feature file missing: {path}")
            
        arr = np.load(path)
        # Handle feature shapes
        if arr.ndim == 3:
            N, T, D = arr.shape
            arr = arr.reshape(N, T * D)
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]
            
        x = torch.from_numpy(arr).float()
        
        if self.cutoffs is not None:
            y = pain_to_class(label, self.cutoffs)
            if y is None:
                y = -1
        else:
            y = int(round(label))
            
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x, y_tensor, video_id, self.combo_name


class ExcelBagDataModule(pl.LightningDataModule):
    def __init__(
        self,
        meta_path: str,
        feature_root: str,
        batch_size: int,
        num_workers: int,
        split_col: str,
        video_col: str,
        label_col: str,
        subject_col: str,
        feature_suffix: str,
        cutoffs: Optional[List[float]],
        # Data-efficiency subsampling (train only)
        train_subject_frac: float = 1.0,
        subsample_seed: int = 42,
        # Split args
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        # LOOCV args
        loocv_test_subject: Optional[str] = None,
        val_ratio: float = 0.15, 
        random_state: int = 42,
    ):
        super().__init__()
        self.meta_path = meta_path
        self.feature_root = feature_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_col = split_col
        self.video_col = video_col
        self.label_col = label_col
        self.subject_col = subject_col
        self.feature_suffix = feature_suffix
        self.cutoffs = cutoffs
        self.train_subject_frac = float(train_subject_frac)
        self.subsample_seed = int(subsample_seed)
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.loocv_test_subject = loocv_test_subject
        self.val_ratio = val_ratio
        self.random_state = random_state

        self.df: Optional[pd.DataFrame] = None

    def _label_to_class(self, label_val: Any) -> Optional[int]:
        """Map raw label value to a class index for stratified subsampling.

        Uses the same pain_to_class() rule as training when cutoffs are provided.
        Falls back to rounded int label when cutoffs is None.
        """
        try:
            v = float(label_val)
        except Exception:
            return None
        if np.isnan(v):
            return None
        if self.cutoffs is not None:
            return pain_to_class(v, self.cutoffs)
        return int(round(v))

    def _subsample_train_df(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Subsample training data within each subject, stratified by class.

        For each subject and each class present, keep approximately train_subject_frac of samples
        (at least 1 sample per class if present). This preserves class balance within each subject.
        """
        frac = float(self.train_subject_frac)
        if frac >= 1.0:
            return df_train
        if frac <= 0.0:
            raise ValueError("train_subject_frac must be > 0")

        rng = np.random.RandomState(self.subsample_seed)

        # Add class column for stratification
        df = df_train.copy()
        df["_cls"] = df[self.label_col_real].apply(self._label_to_class)
        df = df.dropna(subset=["_cls"])
        df["_cls"] = df["_cls"].astype(int)

        kept_rows = []
        for sid, df_s in df.groupby(self.subject_col_real):
            for cls, df_sc in df_s.groupby("_cls"):
                n = len(df_sc)
                if n <= 1:
                    kept_rows.append(df_sc)
                    continue
                k = int(round(frac * n))
                k = max(1, min(k, n))
                idx = rng.choice(df_sc.index.values, size=k, replace=False)
                kept_rows.append(df_sc.loc[idx])

        out = pd.concat(kept_rows, axis=0).drop(columns=["_cls"]).reset_index(drop=True)
        return out

    def _load_df(self):
        ext = os.path.splitext(self.meta_path)[-1].lower()
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(self.meta_path)
        else:
            df = pd.read_csv(self.meta_path)
        
        # normalize columns to lower case for lookup, but keep original for access if needed?
        # No, let's just create a normalized map for the specific columns we need
        col_map = {c.lower(): c for c in df.columns}
        
        # Ensure our target columns exist
        self.split_col_real = col_map.get(self.split_col.lower(), self.split_col)
        self.video_col_real = col_map.get(self.video_col.lower(), self.video_col)
        self.label_col_real = col_map.get(self.label_col.lower(), self.label_col)
        self.subject_col_real = col_map.get(self.subject_col.lower(), self.subject_col)
        
        return df

    def setup(self, stage: Optional[str] = None):
        if self.df is None:
            self.df = self._load_df()

        if self.loocv_test_subject is not None:
            # LOOCV Logic:
            # Test set: all rows where subject_id == loocv_test_subject
            # Remaining rows: split into Train/Val by subject
            
            # Normalize subject column to string
            self.df[self.subject_col_real] = self.df[self.subject_col_real].astype(str)
            
            test_mask = self.df[self.subject_col_real] == str(self.loocv_test_subject)
            df_test = self.df[test_mask].copy()
            df_remaining = self.df[~test_mask].copy()
            
            # Get remaining subjects
            remaining_subjects = df_remaining[self.subject_col_real].unique()
            
            # Deterministic split of remaining subjects for val
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(remaining_subjects)
            
            n_rem = len(remaining_subjects)
            n_val = max(1, int(round(self.val_ratio * n_rem)))
            
            val_subjects = set(remaining_subjects[:n_val])
            train_subjects = set(remaining_subjects[n_val:])
            
            df_val = df_remaining[df_remaining[self.subject_col_real].isin(val_subjects)].copy()
            df_train = df_remaining[df_remaining[self.subject_col_real].isin(train_subjects)].copy()
            
            print(f"[LOOCV] Subject={self.loocv_test_subject} | Train: {len(df_train)} ({len(train_subjects)} subs) | Val: {len(df_val)} ({len(val_subjects)} subs) | Test: {len(df_test)} (1 sub)")
            
        else:
            # Standard Fixed Split Logic (from CSV column)
            self.df[self.split_col_real] = self.df[self.split_col_real].astype(str).str.lower()
            
            df_train = self.df[self.df[self.split_col_real] == self.train_split.lower()].copy()
            df_val = self.df[self.df[self.split_col_real] == self.val_split.lower()].copy()
            df_test = self.df[self.df[self.split_col_real] == self.test_split.lower()].copy()

        # Optional train-only subsampling (data-efficiency study)
        before = len(df_train)
        df_train = self._subsample_train_df(df_train)
        after = len(df_train)
        if after != before:
            print(f"[Subsample] train_subject_frac={self.train_subject_frac} | Train: {before} -> {after}")

        # Build Datasets
        self.train_dataset = ExcelFeatureDataset(
            df_train, self.feature_root, self.feature_suffix, "biovid", self.cutoffs, 
            video_col=self.video_col_real, label_col=self.label_col_real
        )
        self.val_dataset = ExcelFeatureDataset(
            df_val, self.feature_root, self.feature_suffix, "biovid", self.cutoffs,
            video_col=self.video_col_real, label_col=self.label_col_real
        )
        self.test_dataset = ExcelFeatureDataset(
            df_test, self.feature_root, self.feature_suffix, "biovid", self.cutoffs,
            video_col=self.video_col_real, label_col=self.label_col_real
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_mil_batch,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_mil_batch,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_mil_batch,
        )
    
    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def discover_subjects_from_csv(meta_path: str, subject_col: str = "subject_id") -> List[str]:
    """Helper to list all unique subjects from the CSV for LOOCV."""
    ext = os.path.splitext(meta_path)[-1].lower()
    if ext in (".xls", ".xlsx"):
        df = pd.read_excel(meta_path)
    else:
        df = pd.read_csv(meta_path)
    
    col_map = {c.lower(): c for c in df.columns}
    real_col = col_map.get(subject_col.lower(), subject_col)
    
    if real_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found in metadata file.")
        
    subs = df[real_col].dropna().unique()
    # Return sorted strings
    return sorted([str(s) for s in subs if str(s).lower() != "nan"])


def main():
    parser = argparse.ArgumentParser(description="Train BioVid MIL (Single Task) - Fixed Split or LOOCV")
    parser.add_argument("--config", type=str, required=True)
    
    # Allow overriding key config params via CLI
    parser.add_argument("--meta_excel", type=str, default=None)
    parser.add_argument("--feature_root", type=str, default=None)
    parser.add_argument("--feature_suffix", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default_root_dir", type=str, default="BioVid/mil_runs")
    # Data-efficiency subsampling (train only)
    parser.add_argument("--train_subject_frac", type=float, default=None, help="Keep this fraction of samples per subject (stratified by class).")
    parser.add_argument("--subsample_seed", type=int, default=None, help="Random seed for subsampling.")
    
    # LOOCV
    parser.add_argument("--loocv", action="store_true", help="Run Leave-One-Subject-Out Cross Validation")
    parser.add_argument("--loocv_limit_subjects", type=int, default=None, help="Limit to N subjects for debugging")
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    pl.seed_everything(args.seed)

    # 1. Resolve Data Config
    meta_excel = args.meta_excel or cfg.get("meta_excel_path")
    feature_root = args.feature_root or cfg.get("feature_root")
    feature_suffix = args.feature_suffix or cfg.get("feature_suffix", "_windows.npy")
    
    if not meta_excel:
        raise ValueError("meta_excel_path must be provided via config or --meta_excel")
    if not feature_root:
        raise ValueError("feature_root must be provided via config or --feature_root")
        
    split_col = cfg.get("split_col", "split")
    video_col = cfg.get("video_col", "video_id")
    label_col = cfg.get("label_col", "pain_level")
    subject_col = cfg.get("subject_col", "subject_id")  # New config param
    
    # Default params
    train_split = cfg.get("train_split", "train")
    val_split = cfg.get("val_split", "val")
    test_split = cfg.get("test_split", "test")
    
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 64))
    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get("num_workers", 4))
    cutoffs = cfg.get("pain_class_cutoffs", None)
    num_classes = int(cfg.get("num_classes", 5))

    train_subject_frac = args.train_subject_frac if args.train_subject_frac is not None else float(cfg.get("train_subject_frac", 1.0))
    subsample_seed = args.subsample_seed if args.subsample_seed is not None else int(cfg.get("subsample_seed", args.seed))
    
    # Setup Paths
    save_dir = cfg.get("save_dir", args.default_root_dir)
    os.makedirs(save_dir, exist_ok=True)
    monitor_metric = cfg.get("monitor_metric", "val_qwk")

    # Helper to build model
    def _build_model():
        return MILCoralTransformer(
            input_dim=int(cfg.get("input_dim", 768)),
            embed_dim=int(cfg.get("embed_dim", 256)),
            num_classes=num_classes,
            attn_type=str(cfg.get("attn_type", "xformer")),
            xformer_heads=int(cfg.get("xformer_heads", 4)),
            xformer_latents=int(cfg.get("xformer_latents", 16)),
            xformer_cross_layers=int(cfg.get("xformer_cross_layers", 1)),
            xformer_self_layers=int(cfg.get("xformer_self_layers", 2)),
            xformer_dropout=float(cfg.get("xformer_dropout", 0.1)),
            coral_alpha=float(cfg.get("coral_alpha", 0.5)),
            ce_weight=float(cfg.get("ce_weight", 1.0)),
            learning_rate=args.learning_rate or float(cfg.get("learning_rate", 2e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-2)),
            class_weights=None,
            eval_head=str(cfg.get("eval_head", "coral")),
            aux_num_classes=None,
            aux_loss_weight=0.0,
        )

    # Mode 1: Standard Fixed Split
    if not args.loocv:
        print(f"[Info] Running Fixed Split Training (Train={train_split}, Val={val_split})")
        
        dm = ExcelBagDataModule(
            meta_path=meta_excel,
            feature_root=feature_root,
            batch_size=batch_size,
            num_workers=num_workers,
            split_col=split_col,
            video_col=video_col,
            label_col=label_col,
            subject_col=subject_col,
            feature_suffix=feature_suffix,
            cutoffs=cutoffs,
            train_subject_frac=train_subject_frac,
            subsample_seed=subsample_seed,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )
        dm.setup()
        
        model = _build_model()
        
        # Construct filename dynamically based on the monitored metric
        filename_format = "biovid-{epoch}-{" + monitor_metric + ":.3f}"
        
        checkpoint_cb = ModelCheckpoint(
            monitor=monitor_metric,
            mode="max",
            dirpath=save_dir,
            filename=filename_format,
            save_last=True,
        )
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            patience=int(cfg.get("patience", 20)),
            mode="max",
        )
        logger = TensorBoardLogger(
            save_dir=os.path.dirname(save_dir),
            name=os.path.basename(save_dir),
            default_hp_metric=False,
        )

        trainer = pl.Trainer(
            accelerator=args.accelerator or cfg.get("accelerator", "gpu"),
            devices=args.devices or cfg.get("devices", 1),
            max_epochs=args.max_epochs or int(cfg.get("max_epochs", 100)),
            precision=cfg.get("precision", 32),
            default_root_dir=save_dir,
            callbacks=[checkpoint_cb, early_stop],
            logger=logger,
            log_every_n_steps=10,
        )

        trainer.fit(model, datamodule=dm)

        best_path = checkpoint_cb.best_model_path
        if best_path:
            print(f"\n[Eval] Testing with best checkpoint: {best_path}")
            trainer.test(model, datamodule=dm, ckpt_path=best_path)
        else:
            print("\n[Eval] No best checkpoint recorded; testing with last weights.")
            trainer.test(model, datamodule=dm)
            
        # Optional: Save predictions
        if cfg.get("save_preds", False):
            preds_batches = trainer.predict(model, datamodule=dm, ckpt_path=best_path if best_path else None)
            all_preds = []
            if preds_batches:
                for batch_out in preds_batches:
                    vids = batch_out.get("video_ids", [])
                    y_true = batch_out.get("y_true", []).numpy().tolist()
                    y_pred = batch_out.get("y_pred", []).numpy().tolist()
                    for i in range(len(y_true)):
                        vid = vids[i] if i < len(vids) else ""
                        all_preds.append({"video_id": vid, "y_true": y_true[i], "y_pred": y_pred[i]})
            
            if all_preds:
                pred_df = pd.DataFrame(all_preds)
                pred_csv = os.path.join(save_dir, "test_predictions.csv")
                pred_df.to_csv(pred_csv, index=False)
                print(f"Predictions saved to {pred_csv}")

    # -----------------------------------------------------------
    # MODE 2: LOOCV
    # -----------------------------------------------------------
    else:
        # Discover subjects
        subjects = discover_subjects_from_csv(meta_excel, subject_col)
        if args.loocv_limit_subjects:
            subjects = subjects[:args.loocv_limit_subjects]
            
        print(f"[Info] Running LOOCV on {len(subjects)} subjects: {subjects}")
        
        # Prepare output metrics
        fields = ["subject", "test_qwk", "test_acc", "test_mae", "test_f1_macro"]
        rows = []
        
        # Global confusion matrix
        K = num_classes
        sum_conf = np.zeros((K, K), dtype=int)
        
        loocv_dir = os.path.join(save_dir, "loocv_runs")
        os.makedirs(loocv_dir, exist_ok=True)
        
        for i, sid in enumerate(subjects, 1):
            print(f"\n=== Fold {i}/{len(subjects)}: Test Subject {sid} ===")
            pl.seed_everything(args.seed)  # Reset seed for reproducibility per fold
            
            fold_dir = os.path.join(loocv_dir, f"subject_{sid}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Setup DM for this fold
            dm = ExcelBagDataModule(
                meta_path=meta_excel,
                feature_root=feature_root,
                batch_size=batch_size,
                num_workers=num_workers,
                split_col=split_col,
                video_col=video_col,
                label_col=label_col,
                subject_col=subject_col,
                feature_suffix=feature_suffix,
                cutoffs=cutoffs,
                train_subject_frac=train_subject_frac,
                subsample_seed=subsample_seed,
                # LOOCV params
                loocv_test_subject=sid,
                val_ratio=0.15, # Use 15% of remaining subjects for validation
                random_state=args.seed
            )
            dm.setup()
            
            model = _build_model()
            
            # Construct filename dynamically based on the monitored metric
            filename_format = f"sub_{sid}" + "-{epoch}-{" + monitor_metric + ":.3f}"
            
            checkpoint_cb = ModelCheckpoint(
                monitor=monitor_metric,
                mode="max",
                dirpath=fold_dir,
                filename=filename_format,
                save_last=True,
            )
            early_stop = EarlyStopping(
                monitor=monitor_metric,
                patience=int(cfg.get("patience", 15)),
                mode="max",
            )
            logger = TensorBoardLogger(
                save_dir=os.path.dirname(fold_dir),
                name=f"logs_sub_{sid}",
                default_hp_metric=False,
            )
            
            trainer = pl.Trainer(
                accelerator=args.accelerator or cfg.get("accelerator", "gpu"),
                devices=args.devices or cfg.get("devices", 1),
                max_epochs=args.max_epochs or int(cfg.get("max_epochs", 100)),
                precision=cfg.get("precision", 32),
                default_root_dir=fold_dir,
                callbacks=[checkpoint_cb, early_stop],
                logger=logger,
                log_every_n_steps=10,
                enable_progress_bar=True
            )
            
            trainer.fit(model, datamodule=dm)
            
            best_path = checkpoint_cb.best_model_path
            ckpt_choice = best_path if best_path else "last"
            print(f"[Fold {sid}] Testing with checkpoint: {ckpt_choice}")
            
            # Test
            test_results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_choice)
            metrics = test_results[0] if test_results else {}
            
            # Record metrics
            row = {"subject": str(sid)}
            for k in fields[1:]:
                row[k] = metrics.get(k, None)
            rows.append(row)
            
            # Predict for confusion matrix & detailed output
            preds_batches = trainer.predict(model, datamodule=dm, ckpt_path=ckpt_choice)
            
            y_true_all = []
            y_pred_all = []
            vids_all = []
            
            if preds_batches:
                for out in preds_batches:
                    yt = out.get("y_true")
                    yp = out.get("y_pred")
                    vids = out.get("video_ids", [])
                    
                    if yt is not None: y_true_all.extend([int(v) for v in yt.view(-1).tolist()])
                    if yp is not None: y_pred_all.extend([int(v) for v in yp.view(-1).tolist()])
                    vids_all.extend(vids)
                    
            # Update global confusion matrix
            if y_true_all and y_pred_all:
                from evaluate_syracuse_mil_loocv import _confusion_from_preds
                # Fallback implementation if import fails or just inline it
                def simple_conf(yt, yp, k):
                    c = np.zeros((k, k), dtype=int)
                    for t, p in zip(yt, yp):
                        if 0 <= t < k and 0 <= p < k:
                            c[t, p] += 1
                    return c
                
                conf = simple_conf(y_true_all, y_pred_all, K)
                sum_conf += conf
                
                # Save per-fold predictions
                preds_csv = os.path.join(fold_dir, f"preds_sub_{sid}.csv")
                with open(preds_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["video_id", "y_true", "y_pred"])
                    for i in range(len(y_true_all)):
                        w.writerow([vids_all[i] if i < len(vids_all) else "", y_true_all[i], y_pred_all[i]])

        # 3. Final Aggregation
        summary_csv = os.path.join(save_dir, "biovid_loocv_summary.csv")
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            if rows:
                mean_row = {"subject": "MEAN"}
                for k in fields[1:]:
                    vals = [float(r[k]) for r in rows if r.get(k) is not None]
                    mean_row[k] = sum(vals) / len(vals) if vals else None
                writer.writerow(mean_row)
                
        print(f"\n[Done] LOOCV Summary saved to {summary_csv}")
        
        # Save Global Confusion Matrix
        conf_csv = os.path.join(save_dir, "biovid_loocv_confusion_matrix.csv")
        with open(conf_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + [str(i) for i in range(K)])
            for i in range(K):
                w.writerow([str(i)] + [str(c) for c in sum_conf[i]])
        print(f"Global Confusion Matrix saved to {conf_csv}")

if __name__ == "__main__":
    main()
