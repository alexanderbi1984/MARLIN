import argparse
import os
import csv
from typing import Optional, Dict, Any, List, Union

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
try:
    from pytorch_lightning.trainer.supporters import CombinedLoader
except ImportError:
    from pytorch_lightning.utilities.combined_loader import CombinedLoader

from torch.utils.data import Dataset, DataLoader

from dataset.syracuse_bag import collate_mil_batch, pain_to_class
from model.mil_coral_xformer import MILCoralTransformer


def _split_indices_random(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Return (val_idx, test_idx) as two disjoint index lists."""
    n = int(n)
    if n <= 1:
        return list(range(n)), []
    r = float(val_ratio)
    r = max(0.0, min(1.0, r))
    rng = np.random.RandomState(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(r * n))
    n_val = max(1, min(n - 1, n_val))
    val_idx = idx[:n_val].tolist()
    test_idx = idx[n_val:].tolist()
    return val_idx, test_idx


def _split_dataset_for_val_test(
    dataset: "UniversalFeatureDataset",
    val_ratio: float,
    seed: int,
    strategy: str = "subject",
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Split a dataset into (val_subset, test_subset) with no overlap.

    - strategy='subject': split by subject_id groups when available.
    - strategy='random': split by samples.
    Falls back to random if subject split is not feasible.
    """
    n = len(dataset)
    if n == 0:
        return torch.utils.data.Subset(dataset, []), torch.utils.data.Subset(dataset, [])
    strat = str(strategy).lower().strip()
    if strat not in ("subject", "random"):
        strat = "subject"

    if strat == "random":
        val_idx, test_idx = _split_indices_random(n, val_ratio, seed)
        return torch.utils.data.Subset(dataset, val_idx), torch.utils.data.Subset(dataset, test_idx)

    # subject split
    # Build subject->indices mapping using dataset.entries
    subj_to_idx: Dict[str, List[int]] = {}
    for i, e in enumerate(getattr(dataset, "entries", [])):
        s = str(e.get("subject", "unknown"))
        subj_to_idx.setdefault(s, []).append(i)

    subjects = [s for s in subj_to_idx.keys() if s and s.lower() != "nan"]
    # If subjects are not informative, fallback to random
    if len(subjects) <= 1:
        val_idx, test_idx = _split_indices_random(n, val_ratio, seed)
        return torch.utils.data.Subset(dataset, val_idx), torch.utils.data.Subset(dataset, test_idx)

    # If "unknown" dominates, prefer random split
    if "unknown" in subj_to_idx and len(subj_to_idx["unknown"]) >= max(1, int(0.9 * n)):
        val_idx, test_idx = _split_indices_random(n, val_ratio, seed)
        return torch.utils.data.Subset(dataset, val_idx), torch.utils.data.Subset(dataset, test_idx)

    rng = np.random.RandomState(int(seed))
    rng.shuffle(subjects)
    n_sub = len(subjects)
    n_val_sub = int(round(float(val_ratio) * n_sub))
    n_val_sub = max(1, min(n_sub - 1, n_val_sub))
    val_subs = set(subjects[:n_val_sub])
    test_subs = set(subjects[n_val_sub:])

    val_idx: List[int] = []
    test_idx: List[int] = []
    for s, idxs in subj_to_idx.items():
        if s in val_subs:
            val_idx.extend(idxs)
        elif s in test_subs:
            test_idx.extend(idxs)
        else:
            # subjects like 'unknown' or filtered keys: assign to test by default
            test_idx.extend(idxs)

    # Ensure both sides non-empty; otherwise fallback to random
    if len(val_idx) == 0 or len(test_idx) == 0:
        val_idx, test_idx = _split_indices_random(n, val_ratio, seed)
    return torch.utils.data.Subset(dataset, val_idx), torch.utils.data.Subset(dataset, test_idx)


class UniversalFeatureDataset(Dataset):
    """
    A generic dataset that can handle Syracuse, BioVid, or any new clinical dataset.
    It maps continuous pain scores to classes based on provided cutoffs.
    """
    def __init__(
        self,
        meta_path: str,
        feature_root: str,
        split_names: List[str],  # e.g. ["train", "val"] or ["all"]
        split_col: str = "split",
        video_col: str = "video_id",
        label_col: str = "pain_level",
        subject_col: str = "subject_id",
        feature_suffix: str = "_windows.npy",
        feature_prefix: str = "", # Added feature_prefix
        remove_extension: bool = False, # Remove .mp4 etc from video_id
        dataset_name: str = "dataset",
        cutoffs: Optional[List[float]] = None,
        scale_factor: float = 1.0, # Multiply label by this before binning (if needed)
        exclude_subjects: Optional[List[str]] = None,
        limit_subjects: Optional[int] = None,
    ):
        super().__init__()
        self.feature_root = feature_root
        self.feature_suffix = feature_suffix
        self.feature_prefix = feature_prefix # Store feature_prefix
        self.remove_extension = remove_extension
        self.dataset_name = dataset_name
        self.cutoffs = cutoffs
        
        # Load Metadata
        ext = os.path.splitext(meta_path)[-1].lower()
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(meta_path)
        else:
            df = pd.read_csv(meta_path)
            
        # Normalize columns
        col_map = {c.lower(): c for c in df.columns}
        self.split_col = col_map.get(split_col.lower(), split_col)
        self.video_col = col_map.get(video_col.lower(), video_col)
        self.label_col = col_map.get(label_col.lower(), label_col)
        self.subject_col = col_map.get(subject_col.lower(), subject_col)
        
        # Filter by split names (if specific splits requested)
        # If split_names contains "all", we ignore the split column and take everything
        if "all" not in [s.lower() for s in split_names]:
            if self.split_col in df.columns:
                df = df[df[self.split_col].astype(str).str.lower().isin([s.lower() for s in split_names])]
            else:
                print(f"[Warn] Split column '{self.split_col}' not found in {dataset_name}. Using all data.")

        # Filter excluded subjects
        if exclude_subjects:
            if self.subject_col in df.columns:
                df = df[~df[self.subject_col].astype(str).isin([str(s) for s in exclude_subjects])]
        
        self.entries: List[Dict[str, Any]] = []
        
        # Iterate
        for _, row in df.iterrows():
            vid = str(row[self.video_col]).strip()
            
            # Optional: Remove file extension (e.g. .mp4) if present in CSV but not in feature filename
            if self.remove_extension and "." in vid:
                vid = os.path.splitext(vid)[0]

            # Heuristics for Syracuse/BioVid naming consistency
            if vid.endswith("_aligned"):
                vid = vid[:-8]
            if not vid:
                continue
                
            label_raw = row[self.label_col]
            if pd.isna(label_raw):
                continue
                
            label_val = float(label_raw)
            # Optional: Scale label before processing (though usually we adjust cutoffs instead)
            label_val *= scale_factor
            
            # Robust subject extraction
            subj = str(row[self.subject_col]) if (self.subject_col in df.columns and pd.notna(row.get(self.subject_col))) else "unknown"
            
            self.entries.append({
                "video_id": vid,
                "label": label_val,
                "subject": subj
            })
            
        # Optional: Limit subjects (for debugging or few-shot setup)
        if limit_subjects is not None:
            all_subs = sorted(list(set(e["subject"] for e in self.entries)))
            kept_subs = set(all_subs[:limit_subjects])
            self.entries = [e for e in self.entries if e["subject"] in kept_subs]
            print(f"[{dataset_name}] Limited to {limit_subjects} subjects: {len(self.entries)} samples.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        video_id = entry["video_id"]
        label = entry["label"]
        
        # Feature Loading
        path = os.path.join(self.feature_root, f"{self.feature_prefix}{video_id}{self.feature_suffix}")
        if not os.path.isfile(path):
            # Fallback for some naming conventions?
            raise FileNotFoundError(f"[{self.dataset_name}] Feature file missing: {path}")
            
        arr = np.load(path)
        if arr.ndim == 3:
            N, T, D = arr.shape
            arr = arr.reshape(N, T * D)
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]
        x = torch.from_numpy(arr).float()
        
        # Label Binning
        if self.cutoffs is not None:
            y = pain_to_class(label, self.cutoffs)
            if y is None: y = -1
        else:
            # Assume label is already class index
            y = int(round(label))
            
        return x, torch.tensor(y, dtype=torch.long), video_id, self.dataset_name


class CrossDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_datasets_cfg: List[Dict[str, Any]], 
        test_datasets_cfg: List[Dict[str, Any]],   # Now a list of test datasets
        aux_dataset_cfg: Optional[Dict[str, Any]],
        batch_size: int = 32,
        num_workers: int = 4,
        # Target-domain validation split config (no overlap with test)
        val_target: Optional[str] = None,
        val_ratio: float = 0.2,
        val_split_strategy: str = "subject",  # 'subject' | 'random'
        val_seed: int = 42,
    ):
        super().__init__()
        self.train_cfgs = train_datasets_cfg
        self.test_cfgs = test_datasets_cfg
        self.aux_cfg = aux_dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_datasets_map = {} # name -> dataset
        self.val_target = val_target
        self.val_ratio = float(val_ratio)
        self.val_split_strategy = str(val_split_strategy)
        self.val_seed = int(val_seed)

    def setup(self, stage: Optional[str] = None):
        # 1. Build Main Training Set (concatenation)
        train_dsets = []
        for cfg in self.train_cfgs:
            ds = UniversalFeatureDataset(**cfg)
            print(f"Loaded Train Dataset: {ds.dataset_name} ({len(ds)} samples)")
            train_dsets.append(ds)
        self.train_dataset = torch.utils.data.ConcatDataset(train_dsets)
        
        # 2. Build Test Datasets (Target Domains)
        # We hold multiple test datasets in a dict
        for cfg in self.test_cfgs:
            ds = UniversalFeatureDataset(**cfg)
            print(f"Loaded Test Dataset: {ds.dataset_name} ({len(ds)} samples)")
            self.test_datasets_map[ds.dataset_name] = ds
        
        # 3. Validation Set (Target-domain validation) with NO overlap with test:
        # Split one target dataset into val/test subsets.
        if len(self.test_datasets_map) > 0:
            default_val_target = list(self.test_datasets_map.keys())[0]
            val_target = self.val_target or default_val_target
            if val_target not in self.test_datasets_map:
                print(f"[Warn] val_target='{val_target}' not found among test targets; using '{default_val_target}'.")
                val_target = default_val_target

            full_target_ds = self.test_datasets_map[val_target]
            val_subset, test_subset = _split_dataset_for_val_test(
                full_target_ds,
                val_ratio=self.val_ratio,
                seed=self.val_seed,
                strategy=self.val_split_strategy,
            )
            self.val_dataset = val_subset
            self.test_datasets_map[val_target] = test_subset
            print(
                f"[Val/Test split] target='{val_target}' strategy={self.val_split_strategy} "
                f"val_ratio={self.val_ratio} -> val={len(self.val_dataset)} test={len(self.test_datasets_map[val_target])}"
            )
        else:
            self.val_dataset = self.train_dataset  # Fallback (should not happen)

        # 4. Build Aux Dataset (BioVid)
        self.aux_dataset = None
        if self.aux_cfg:
            self.aux_dataset = UniversalFeatureDataset(**self.aux_cfg)
            print(f"Loaded Aux Dataset: {self.aux_cfg.get('dataset_name')} ({len(self.aux_dataset)} samples)")

    def train_dataloader(self):
        loaders = {}
        loaders["main"] = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_mil_batch, drop_last=True
        )
        if self.aux_dataset:
            loaders["aux"] = DataLoader(
                self.aux_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, collate_fn=collate_mil_batch, drop_last=True
            )
            return CombinedLoader(loaders, mode="max_size_cycle")
        return loaders["main"]

    def val_dataloader(self):
        # Use first test dataset as validation
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_mil_batch
        )

    # We need a custom method to get specific test loaders
    def get_test_dataloader(self, name: str):
        if name not in self.test_datasets_map:
            raise ValueError(f"Test dataset {name} not found.")
        return DataLoader(
            self.test_datasets_map[name],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_mil_batch
        )


def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_feature_catalog(path: Optional[str]) -> Dict[str, Any]:
    """Load a feature catalog YAML (dataset × feature registry) if provided."""
    if not path:
        return {}
    import yaml
    if not os.path.isfile(path):
        raise FileNotFoundError(f"feature_catalog_path not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    parser = argparse.ArgumentParser(description="Cross-Database Training & Eval (One Source, Multi Target)")
    parser.add_argument("--config", type=str, required=True, help="Path to cross_db.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    pl.seed_everything(args.seed)
    
    # --- Configuration Parsing ---
    
    num_classes = int(cfg.get("num_classes", 5))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 4))
    
    base_cutoffs = cfg.get("pain_class_cutoffs")
    half_cutoffs = [c / 2.0 for c in base_cutoffs] if base_cutoffs else None
    
    # Config keys for selecting Source and Targets
    # train_sources: ["syracuse"]
    # test_targets: ["dataset_a", "dataset_b"]
    train_sources = cfg.get("train_sources", [])
    test_targets = cfg.get("test_targets", [])
    
    dbs_config = cfg.get("datasets", {})

    # Optional: centralized feature registry
    feature_catalog_path = cfg.get("feature_catalog_path", None)
    feature_catalog = load_feature_catalog(feature_catalog_path) if feature_catalog_path else {}
    feature_defs = feature_catalog.get("features", {}) if isinstance(feature_catalog, dict) else {}

    def _infer_input_dim_from_catalog(dataset_names: List[str], aux_cfg: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """Infer a single shared input_dim from feature catalog via feature_name.

        Returns None if it cannot infer.
        Raises if multiple different dims are detected.
        """
        dims: List[int] = []
        for dn in dataset_names:
            info = dbs_config.get(dn, {})
            feat_name = info.get("feature_name")
            if not feat_name:
                continue
            feat_def = feature_defs.get(feat_name)
            if isinstance(feat_def, dict) and feat_def.get("input_dim") is not None:
                try:
                    dims.append(int(feat_def["input_dim"]))
                except Exception:
                    pass
        if aux_cfg is not None:
            feat_name = aux_cfg.get("feature_name")
            if feat_name:
                feat_def = feature_defs.get(feat_name)
                if isinstance(feat_def, dict) and feat_def.get("input_dim") is not None:
                    try:
                        dims.append(int(feat_def["input_dim"]))
                    except Exception:
                        pass
        dims = [d for d in dims if d > 0]
        if not dims:
            return None
        uniq = sorted(set(dims))
        if len(uniq) > 1:
            raise ValueError(f"Multiple feature input_dims detected from catalog: {uniq}. Please set input_dim explicitly in config.")
        return uniq[0]
    
    # Helper to build dataset config
    def make_ds_cfg(name, split_mode="all"):
        if name not in dbs_config:
            raise ValueError(f"Dataset '{name}' not defined in 'datasets' section.")
        info = dbs_config[name]
        scale = info.get("scale", "0-10")
        cutoffs = base_cutoffs if scale == "0-10" else half_cutoffs

        # Resolve feature_* fields from catalog when `feature_name` is provided.
        # Explicit fields in `cross_db.yaml` override catalog values.
        resolved = dict(info)
        feat_name = resolved.get("feature_name", None)
        if feat_name:
            feat_def = feature_defs.get(feat_name)
            if feat_def is None:
                raise ValueError(f"feature_name '{feat_name}' not found in feature catalog.")
            per_ds = (feat_def.get("per_dataset") or {}).get(name, {})
            for k in ("feature_root", "feature_suffix", "feature_prefix", "remove_extension"):
                if (k not in resolved) or (resolved.get(k) is None):
                    if k in per_ds:
                        resolved[k] = per_ds[k]
            if (("feature_suffix" not in resolved) or (resolved.get("feature_suffix") is None)) and ("default_suffix" in feat_def):
                resolved["feature_suffix"] = feat_def["default_suffix"]

        return {
            "meta_path": resolved["meta_path"],
            "feature_root": resolved["feature_root"],
            "feature_suffix": resolved.get("feature_suffix", "_windows.npy"),
            "feature_prefix": resolved.get("feature_prefix", ""), # Pass prefix
            "remove_extension": resolved.get("remove_extension", False),
            "split_names": [split_mode],
            "dataset_name": name,
            "cutoffs": cutoffs,
            "label_col": resolved.get("label_col", "pain_level"),
            "video_col": resolved.get("video_col", "video_id"),
            "subject_col": resolved.get("subject_col", "subject_id"),
        }

    # Prepare Train Configs
    train_datasets_cfg = [make_ds_cfg(name) for name in train_sources]
    
    # Prepare Test Configs
    test_datasets_cfg = [make_ds_cfg(name) for name in test_targets]
    
    # Aux Config
    aux_config_raw = cfg.get("aux_dataset", None)
    aux_dataset_cfg = None
    if aux_config_raw and aux_config_raw.get("enabled", False):
        aux_resolved = dict(aux_config_raw)
        aux_feat_name = aux_resolved.get("feature_name", None)
        if aux_feat_name:
            feat_def = feature_defs.get(aux_feat_name)
            if feat_def is None:
                raise ValueError(f"aux feature_name '{aux_feat_name}' not found in feature catalog.")
            # use dataset_name in aux config to find per_dataset entry (default: 'biovid')
            aux_ds_name = str(aux_resolved.get("dataset_name", "biovid"))
            per_ds = (feat_def.get("per_dataset") or {}).get(aux_ds_name, {})
            for k in ("feature_root", "feature_suffix", "feature_prefix", "remove_extension"):
                if (k not in aux_resolved) or (aux_resolved.get(k) is None):
                    if k in per_ds:
                        aux_resolved[k] = per_ds[k]
            if (("feature_suffix" not in aux_resolved) or (aux_resolved.get("feature_suffix") is None)) and ("default_suffix" in feat_def):
                aux_resolved["feature_suffix"] = feat_def["default_suffix"]

        aux_dataset_cfg = {
            "meta_path": aux_resolved["meta_path"],
            "feature_root": aux_resolved["feature_root"],
            "feature_suffix": aux_resolved.get("feature_suffix", "_windows.npy"),
            "feature_prefix": aux_resolved.get("feature_prefix", ""), # Pass prefix for aux too
            "remove_extension": aux_resolved.get("remove_extension", False),
            "split_names": ["train", "val"], 
            "dataset_name": "biovid_aux",
            "cutoffs": None,
            "label_col": aux_resolved.get("label_col", "pain_level"),
            "video_col": aux_resolved.get("video_col", "video_id"),
        }

    # Resolve model input_dim:
    # - Prefer explicit config value
    # - Otherwise infer from feature catalog (feature_name)
    input_dim_cfg = cfg.get("input_dim", None)
    if input_dim_cfg is None:
        inferred = _infer_input_dim_from_catalog(train_sources + test_targets, aux_config_raw if aux_config_raw else None)
        input_dim = int(inferred) if inferred is not None else 288
        print(f"[Info] input_dim not set in config; inferred input_dim={input_dim}")
    else:
        input_dim = int(input_dim_cfg)

    # --- Target-domain val split config ---
    val_target = cfg.get("val_target", None)  # if None -> first in test_targets
    val_ratio = float(cfg.get("val_ratio", 0.2))
    val_split_strategy = str(cfg.get("val_split_strategy", "subject"))
    val_seed = int(cfg.get("val_seed", args.seed))

    # --- DataModule ---
    dm = CrossDBDataModule(
        train_datasets_cfg=train_datasets_cfg,
        test_datasets_cfg=test_datasets_cfg,
        aux_dataset_cfg=aux_dataset_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        val_target=val_target,
        val_ratio=val_ratio,
        val_split_strategy=val_split_strategy,
        val_seed=val_seed,
    )
    
    # --- Model ---
    model = MILCoralTransformer(
        input_dim=int(input_dim),
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
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
        eval_head=str(cfg.get("eval_head", "coral")),
        aux_num_classes=int(cfg.get("aux_num_classes", 5)) if aux_dataset_cfg else None,
        aux_loss_weight=float(cfg.get("aux_loss_weight", 0.1)) if aux_dataset_cfg else 0.0,
    )

    # --- Trainer ---
    save_dir = cfg.get("save_dir", "CrossDB/runs")
    os.makedirs(save_dir, exist_ok=True)
    
    monitor_metric = cfg.get("monitor_metric", "val_qwk")
    
    checkpoint_cb = ModelCheckpoint(
        monitor=monitor_metric,
        mode="max",
        dirpath=save_dir,
        filename=f"crossdb-{{epoch}}-{{{monitor_metric}:.3f}}",
        save_last=True,
    )
    
    logger = TensorBoardLogger(
        save_dir=os.path.dirname(save_dir),
        name=os.path.basename(save_dir),
        default_hp_metric=False,
    )
    
    trainer = pl.Trainer(
        accelerator=cfg.get("accelerator", "gpu"),
        devices=cfg.get("devices", 1),
        max_epochs=int(cfg.get("max_epochs", 50)),
        default_root_dir=save_dir,
        callbacks=[checkpoint_cb],
        logger=logger,
        log_every_n_steps=10
    )
    
    # Train
    print(f"\n[Start] Training Source(s): {train_sources}")
    print(f"Test Targets: {test_targets}")
    
    trainer.fit(model, datamodule=dm)
    
    best_path = checkpoint_cb.best_model_path
    if not best_path:
        best_path = "last"
    print(f"\n[Eval] Best Checkpoint: {best_path}")
    
    # Evaluation on ALL Targets
    for target in test_targets:
        print(f"\n>>> Evaluating on Target: {target} <<<")
        test_loader = dm.get_test_dataloader(target)
        test_res = trainer.test(model, dataloaders=test_loader, ckpt_path=best_path)
        
        # Save Metrics to CSV
        if test_res:
            metrics = test_res[0]
            metrics["target_db"] = target
            metrics["best_ckpt"] = best_path
            metrics_df = pd.DataFrame([metrics])
            metrics_csv = os.path.join(save_dir, f"metrics_{target}.csv")
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"Metrics for {target} saved to {metrics_csv}")
        
        # Save Predictions
        if cfg.get("save_preds", True):
            preds = trainer.predict(model, dataloaders=test_loader, ckpt_path=best_path)
            all_preds = []
            if preds:
                for out in preds:
                    vids = out.get("video_ids", [])
                    yt = out.get("y_true", []).numpy().tolist()
                    yp = out.get("y_pred", []).numpy().tolist()
                    for i in range(len(yt)):
                        all_preds.append({"video_id": vids[i] if i<len(vids) else "", "y_true": yt[i], "y_pred": yp[i]})
            if all_preds:
                csv_path = os.path.join(save_dir, f"preds_{target}.csv")
                pd.DataFrame(all_preds).to_csv(csv_path, index=False)
                print(f"Predictions for {target} saved to {csv_path}")

if __name__ == "__main__":
    main()
