import argparse
import csv
import os
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

try:
    from pytorch_lightning.trainer.supporters import CombinedLoader  # PL <2.0
except Exception:
    from pytorch_lightning.utilities.combined_loader import CombinedLoader  # PL >=2.0
from torch.utils.data import Dataset, DataLoader

from dataset.syracuse_bag import SyracuseBagDataModule, collate_mil_batch, pain_to_class
from model.mil_coral_xformer import MILCoralTransformer
from evaluate_syracuse_mil_loocv import discover_subjects, _confusion_from_preds


class ExcelFeatureDataset(Dataset):
    def __init__(
        self,
        meta_path: str,
        feature_root: str,
        split_name: str,
        split_col: str = "split",
        video_col: str = "video_id",
        label_col: str = "pain_level",
        feature_suffix: str = "_windows.npy",
        combo_name: str = "aux",
        cutoffs: Optional[List[float]] = None,
    ):
        super().__init__()
        self.feature_root = feature_root
        self.feature_suffix = feature_suffix
        self.combo_name = combo_name
        self.cutoffs = cutoffs
        ext = os.path.splitext(meta_path)[-1].lower()
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(meta_path)
        else:
            df = pd.read_csv(meta_path)
        df_columns = {c.lower(): c for c in df.columns}
        split_col = df_columns.get(split_col.lower(), split_col)
        video_col = df_columns.get(video_col.lower(), video_col)
        label_col = df_columns.get(label_col.lower(), label_col)
        df = df[df[split_col].astype(str).str.lower() == split_name.lower()]
        self.entries: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
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
            raise FileNotFoundError(f"Feature file missing: {path}")
        arr = np.load(path)
        # Auto-flatten 3D features (N, T, D) -> (N, T*D)
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
        train_split: str,
        val_split: str,
        test_split: str,
        split_col: str,
        video_col: str,
        label_col: str,
        feature_suffix: str,
        cutoffs: Optional[List[float]],
    ):
        super().__init__()
        self.meta_path = meta_path
        self.feature_root = feature_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.split_col = split_col
        self.video_col = video_col
        self.label_col = label_col
        self.feature_suffix = feature_suffix
        self.cutoffs = cutoffs

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ExcelFeatureDataset(
            self.meta_path,
            self.feature_root,
            self.train_split,
            split_col=self.split_col,
            video_col=self.video_col,
            label_col=self.label_col,
            feature_suffix=self.feature_suffix,
            combo_name="aux",
            cutoffs=self.cutoffs,
        )
        self.val_dataset = ExcelFeatureDataset(
            self.meta_path,
            self.feature_root,
            self.val_split,
            split_col=self.split_col,
            video_col=self.video_col,
            label_col=self.label_col,
            feature_suffix=self.feature_suffix,
            combo_name="aux",
            cutoffs=self.cutoffs,
        )
        self.test_dataset = ExcelFeatureDataset(
            self.meta_path,
            self.feature_root,
            self.test_split,
            split_col=self.split_col,
            video_col=self.video_col,
            label_col=self.label_col,
            feature_suffix=self.feature_suffix,
            combo_name="aux",
            cutoffs=self.cutoffs,
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


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Syracuse MIL with auxiliary CE head from Excel metadata.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--aux_meta_excel", type=str, default=None, help="Path to Excel file with columns video_id/pain_level/split/etc.")
    parser.add_argument("--aux_feature_root", type=str, default=None, help="Directory containing auxiliary *_windows.npy files.")
    parser.add_argument("--aux_feature_suffix", type=str, default=None)
    parser.add_argument("--aux_split_col", type=str, default=None)
    parser.add_argument("--aux_video_col", type=str, default=None)
    parser.add_argument("--aux_label_col", type=str, default=None)
    parser.add_argument("--aux_train_split", type=str, default=None)
    parser.add_argument("--aux_val_split", type=str, default=None)
    parser.add_argument("--aux_test_split", type=str, default=None)
    parser.add_argument("--aux_batch_size", type=int, default=None)
    parser.add_argument("--aux_num_workers", type=int, default=None)
    parser.add_argument("--aux_cutoffs", type=float, nargs="*", default=None)
    parser.add_argument("--aux_loss_weight", type=float, default=None)
    parser.add_argument("--default_root_dir", type=str, default="Syracuse/aux_multitask_runs")
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # LOOCV options for Syracuse main task (Biovid aux task unchanged)
    parser.add_argument(
        "--loocv",
        action="store_true",
        help="If set, run subject-level LOOCV on Syracuse (Biovid aux task kept fixed).",
    )
    parser.add_argument(
        "--loocv_limit_subjects",
        type=int,
        default=None,
        help="Optionally limit to first N Syracuse subjects when running with --loocv.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(args.seed)

    # Resolve arguments: CLI > Config > Default
    aux_meta_excel = args.aux_meta_excel or cfg.get("aux_meta_excel")
    aux_feature_root = args.aux_feature_root or cfg.get("aux_feature_root")
    
    if not aux_meta_excel:
        raise ValueError("aux_meta_excel must be provided via --aux_meta_excel or config.")
    if not aux_feature_root:
        raise ValueError("aux_feature_root must be provided via --aux_feature_root or config.")

    aux_feature_suffix = args.aux_feature_suffix or cfg.get("aux_feature_suffix", "_windows.npy")
    aux_split_col = args.aux_split_col or cfg.get("aux_split_col", "split")
    aux_video_col = args.aux_video_col or cfg.get("aux_video_col", "video_id")
    aux_label_col = args.aux_label_col or cfg.get("aux_label_col", "pain_level")
    aux_train_split = args.aux_train_split or cfg.get("aux_train_split", "train")
    aux_val_split = args.aux_val_split or cfg.get("aux_val_split", "val")
    aux_test_split = args.aux_test_split or cfg.get("aux_test_split", "test")
    
    aux_batch_size = args.aux_batch_size if args.aux_batch_size is not None else int(cfg.get("aux_batch_size", 64))
    aux_num_workers = args.aux_num_workers if args.aux_num_workers is not None else int(cfg.get("aux_num_workers", 4))

    aux_cutoffs = args.aux_cutoffs if args.aux_cutoffs else cfg.get("aux_cutoffs", None)

    # LOOCV logic
    run_loocv = args.loocv or bool(cfg.get("loocv", False))

    aux_dm = ExcelBagDataModule(
        meta_path=aux_meta_excel,
        feature_root=aux_feature_root,
        batch_size=aux_batch_size,
        num_workers=aux_num_workers,
        train_split=aux_train_split,
        val_split=aux_val_split,
        test_split=aux_test_split,
        split_col=aux_split_col,
        video_col=aux_video_col,
        label_col=aux_label_col,
        feature_suffix=aux_feature_suffix,
        cutoffs=aux_cutoffs,
    )
    aux_dm.setup()

    aux_loss_weight = args.aux_loss_weight if args.aux_loss_weight is not None else float(cfg.get("biovid_aux_loss_weight", 0.1))
    aux_num_classes = int(cfg.get("aux_num_classes", len(aux_cutoffs) + 1 if aux_cutoffs is not None else 5))
    monitor_metric = cfg.get("monitor_metric", "val_qwk")
    save_dir = cfg.get("save_dir", "Syracuse/xformer_mil_multitask")
    os.makedirs(save_dir, exist_ok=True)

    # Helper to build a fresh Syracuse DataModule (optionally LOOCV on a given subject)
    def _build_syracuse_dm(loocv_test_subject: Optional[str] = None) -> SyracuseBagDataModule:
        return SyracuseBagDataModule(
            feature_root=cfg.get("syracuse_feature_root"),
            meta_excel_path=cfg.get("meta_excel_path"),
            task=str(cfg.get("task", "ordinal")),
            num_classes=int(cfg.get("num_classes", 5)),
            pain_class_cutoffs=cfg.get("pain_class_cutoffs", [2.0, 4.0, 6.0, 8.0]),
            combos=cfg.get("combos"),
            baseline_combo=cfg.get("baseline_combo", "RGB"),
            batch_size=int(cfg.get("batch_size", 8)),
            num_workers=int(cfg.get("num_workers", 0)),
            val_split_ratio=float(cfg.get("val_split_ratio", 0.15)),
            test_split_ratio=float(cfg.get("test_split_ratio", 0.15)),
            random_state=int(cfg.get("random_state", 42)),
            max_bag_size=int(cfg.get("max_bag_size")) if cfg.get("max_bag_size") is not None else None,
            normalize_features=bool(cfg.get("normalize_features", True)),
            aug_feature_root=cfg.get("aug_feature_root"),
            use_aug_in_train=bool(cfg.get("use_aug_in_train", True)),
            train_aug_ratio=float(cfg.get("train_aug_ratio", 0.5)),
            use_weighted_sampler=bool(cfg.get("use_weighted_sampler", True)),
            train_epoch_multiplier=int(cfg.get("train_epoch_multiplier", 1)),
            train_enumerate_all_views=bool(cfg.get("train_enumerate_all_views", False)),
            train_max_aug_per_combo=int(cfg.get("train_max_aug_per_combo", 2)),
            train_include_original=bool(cfg.get("train_include_original", True)),
            clip_level=bool(cfg.get("clip_level", False)),
            clip_bag_size=int(cfg.get("clip_bag_size", 1)),
            clip_bag_stride=int(cfg.get("clip_bag_stride", 1)),
            exclude_video_ids=cfg.get("exclude_video_ids"),
            loocv_test_subject=loocv_test_subject,
            feature_suffix=cfg.get("syracuse_feature_suffix", "_windows.npy"),
        )

    # Helper to build a fresh model per run/fold
    def _build_model() -> MILCoralTransformer:
        return MILCoralTransformer(
            input_dim=int(cfg.get("input_dim", 768)),
            embed_dim=int(cfg.get("embed_dim", 256)),
            num_classes=int(cfg.get("num_classes", 5)),
            attn_type=str(cfg.get("attn_type", "xformer")),
            xformer_heads=int(cfg.get("xformer_heads", 4)),
            xformer_latents=int(cfg.get("xformer_latents", 16)),
            xformer_cross_layers=int(cfg.get("xformer_cross_layers", 1)),
            xformer_self_layers=int(cfg.get("xformer_self_layers", 2)),
            xformer_dropout=float(cfg.get("xformer_dropout", 0.1)),
            coral_alpha=float(cfg.get("coral_alpha", 0.5)),
            ce_weight=float(cfg.get("ce_weight", 1.0)),
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-2)),
            class_weights=None,
            eval_head=str(cfg.get("eval_head", "ce")),
            aux_num_classes=aux_num_classes,
            aux_loss_weight=aux_loss_weight,
        )

    if not run_loocv:
        # Standard single train/val/test split on Syracuse (subject-aware), Biovid aux unchanged.
        syracuse_dm = _build_syracuse_dm(loocv_test_subject=None)
        syracuse_dm.setup()

        model = _build_model()

        train_loader = CombinedLoader(
            {"syracuse": syracuse_dm.train_dataloader(), "aux": aux_dm.train_dataloader()},
            mode="max_size_cycle",
        )

        checkpoint_cb = ModelCheckpoint(
            monitor=monitor_metric,
            mode="max",
            dirpath=save_dir,
            filename="syracuse_aux-{epoch}-{val_qwk:.3f}",
            save_last=True,
        )
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            patience=int(cfg.get("patience", 50)),
            mode="max",
        )
        logger = TensorBoardLogger(
            save_dir=args.default_root_dir,
            name="syracuse_aux",
            default_hp_metric=False,
        )

        trainer = pl.Trainer(
            accelerator=args.accelerator or cfg.get("accelerator", "gpu"),
            devices=args.devices or cfg.get("devices", 1),
            max_epochs=args.max_epochs or int(cfg.get("max_epochs", 200)),
            precision=cfg.get("precision", 32),
            default_root_dir=args.default_root_dir,
            callbacks=[checkpoint_cb, early_stop],
            logger=logger,
            log_every_n_steps=10,
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=syracuse_dm.val_dataloader())

        best_path = checkpoint_cb.best_model_path
        if best_path:
            ckpt_choice = best_path
            print(f"\n[Eval] Testing Syracuse main task with best checkpoint: {best_path}")
        else:
            ckpt_choice = "last"
            print("\n[Eval] No best checkpoint recorded; testing with last epoch weights.")

        trainer.test(model, dataloaders=syracuse_dm.test_dataloader(), ckpt_path=ckpt_choice)
    else:
        # Subject-level LOOCV on Syracuse, Biovid aux task kept as-is across folds.
        feature_root = cfg.get("syracuse_feature_root")
        meta_excel = cfg.get("meta_excel_path")
        task = str(cfg.get("task", "ordinal"))
        num_classes = int(cfg.get("num_classes", 5))
        combo = cfg.get("baseline_combo", "RGB")
        exclude_videos = cfg.get("exclude_video_ids")
        feature_suffix = cfg.get("syracuse_feature_suffix", "_windows.npy")

        subjects = discover_subjects(feature_root, combo, meta_excel, task, num_classes, exclude_videos, feature_suffix=feature_suffix)
        if args.loocv_limit_subjects is not None:
            subjects = subjects[: int(args.loocv_limit_subjects)]

        print(f"Running Syracuse LOOCV with {len(subjects)} subjects (baseline combo={combo}); Biovid aux unchanged.")

        # Collect per-subject test metrics to a CSV (similar to evaluate_syracuse_mil_loocv.py)
        fields = [
            "subject",
            "test_qwk",
            "test_acc",
            "test_mae",
            "test_f1_macro",
            "test_f1_weighted",
            "test_f1_micro",
        ]
        rows: List[Dict[str, Any]] = []
        K = int(num_classes)
        sum_conf = np.zeros((K, K), dtype=int)

        for i, sid in enumerate(subjects, 1):
            print(f"\nFold {i}/{len(subjects)} — Syracuse test_subject={sid}")
            pl.seed_everything(args.seed)

            syracuse_dm = _build_syracuse_dm(loocv_test_subject=str(sid))
            syracuse_dm.setup()

            model = _build_model()

            train_loader = CombinedLoader(
                {"syracuse": syracuse_dm.train_dataloader(), "aux": aux_dm.train_dataloader()},
                mode="max_size_cycle",
            )

            checkpoint_cb = ModelCheckpoint(
                monitor=monitor_metric,
                mode="max",
                dirpath=save_dir,
                filename=f"syracuse_aux_loocv-subject={sid}" + "-{epoch}-{val_qwk:.3f}",
                save_last=True,
            )
            early_stop = EarlyStopping(
                monitor=monitor_metric,
                patience=int(cfg.get("patience", 50)),
                mode="max",
            )
            logger = TensorBoardLogger(
                save_dir=args.default_root_dir,
                name=f"syracuse_aux_loocv_subject_{sid}",
                default_hp_metric=False,
            )

            trainer = pl.Trainer(
                accelerator=args.accelerator or cfg.get("accelerator", "gpu"),
                devices=args.devices or cfg.get("devices", 1),
                max_epochs=args.max_epochs or int(cfg.get("max_epochs", 200)),
                precision=cfg.get("precision", 32),
                default_root_dir=args.default_root_dir,
                callbacks=[checkpoint_cb, early_stop],
                logger=logger,
                log_every_n_steps=10,
            )

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=syracuse_dm.val_dataloader())

            best_path = checkpoint_cb.best_model_path
            if best_path:
                ckpt_choice = best_path
                print(f"\n[Eval] LOOCV subject={sid}: testing Syracuse main task with checkpoint: {best_path}")
            else:
                ckpt_choice = "last"
                print(f"\n[Eval] LOOCV subject={sid}: no best checkpoint recorded; testing with last epoch weights.")

            # Test
            test_results = trainer.test(model, dataloaders=syracuse_dm.test_dataloader(), ckpt_path=ckpt_choice)
            if isinstance(test_results, list) and len(test_results) > 0:
                metrics = test_results[0]
                print(f"Syracuse LOOCV subject={sid} test metrics:", metrics)
                row: Dict[str, Any] = {"subject": str(sid)}
                for k in fields[1:]:
                    row[k] = metrics.get(k, None)
                rows.append(row)

            # Predict per-sample on Syracuse test split to derive confusion matrix and save preds/conf CSV
            preds_batches = trainer.predict(model, dataloaders=syracuse_dm.test_dataloader())
            y_true_all: List[int] = []
            y_pred_all: List[int] = []
            vids_all: List[str] = []
            if preds_batches:
                for out in preds_batches:
                    y_true = out.get("y_true")
                    y_pred = out.get("y_pred")
                    vids = out.get("video_ids") or []
                    if y_true is not None:
                        y_true_all.extend([int(v) for v in y_true.view(-1).tolist()])
                    if y_pred is not None:
                        y_pred_all.extend([int(v) for v in y_pred.view(-1).tolist()])
                    if vids:
                        vids_all.extend(list(vids))

            if y_true_all and y_pred_all:
                conf = _confusion_from_preds(y_true_all, y_pred_all, K)
                sum_conf += conf

                # Per-subject preds and confusion CSVs (under save_dir/multitask)
                fold_save_dir = os.path.join(save_dir, "multitask")
                os.makedirs(fold_save_dir, exist_ok=True)

                preds_csv = os.path.join(fold_save_dir, f"preds_multitask_subject_{sid}.csv")
                with open(preds_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["video_id", "y_true", "y_pred"])
                    for idx in range(len(y_pred_all)):
                        vid = vids_all[idx] if idx < len(vids_all) else ""
                        yt = y_true_all[idx] if idx < len(y_true_all) else None
                        yp = y_pred_all[idx]
                        w.writerow([vid, yt, yp])

                conf_csv = os.path.join(fold_save_dir, f"confusion_multitask_subject_{sid}.csv")
                with open(conf_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    header = ["true\\pred"] + [f"{i}" for i in range(K)]
                    w.writerow(header)
                    for ti in range(K):
                        row_vals = [str(ti)] + [str(int(v)) for v in conf[ti]]
                        w.writerow(row_vals)

        # Write LOOCV summary CSV with per-subject metrics and mean row
        out_csv = os.path.join(save_dir, "syracuse_loocv_multitask.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
            if rows:
                summary: Dict[str, Any] = {"subject": "MEAN"}
                for k in fields[1:]:
                    vals = [float(r[k]) for r in rows if r.get(k) is not None]
                    summary[k] = sum(vals) / len(vals) if vals else None
                writer.writerow(summary)
        print(f"\nSyracuse LOOCV (multitask) metrics saved to: {out_csv}")

        # Write aggregated confusion matrix across all subjects (clip-level)
        agg_conf_csv = os.path.join(save_dir, "syracuse_loocv_multitask_confusion.csv")
        with open(agg_conf_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["true\\pred"] + [f"{i}" for i in range(K)]
            w.writerow(header)
            for ti in range(K):
                row_vals = [str(ti)] + [str(int(v)) for v in sum_conf[ti]]
                w.writerow(row_vals)
        print(f"Syracuse LOOCV (multitask) aggregated confusion matrix saved to: {agg_conf_csv}")

        # Compute video-level majority-vote statistics across all LOOCV test videos
        multitask_dir = os.path.join(save_dir, "multitask")
        video_y_true: Dict[str, int] = {}
        video_pred_votes: Dict[str, List[int]] = {}

        if os.path.isdir(multitask_dir):
            from collections import defaultdict, Counter

            video_pred_votes = defaultdict(list)

            for fname in sorted(os.listdir(multitask_dir)):
                if not fname.startswith("preds_multitask_subject_") or not fname.endswith(".csv"):
                    continue
                fpath = os.path.join(multitask_dir, fname)
                with open(fpath, "r") as f:
                    reader = csv.reader(f)
                    _ = next(reader, None)  # header
                    for row in reader:
                        if not row:
                            continue
                        vid, yt, yp = row[0], row[1], row[2]
                        try:
                            yt_i = int(float(yt))
                            yp_i = int(float(yp))
                        except Exception:
                            continue
                        base = vid
                        if "_clip_" in base:
                            base = base.split("_clip_")[0]
                        if base not in video_y_true:
                            video_y_true[base] = yt_i
                        video_pred_votes[base].append(yp_i)

            correct_top1 = correct_top2 = correct_top3 = 0
            total_videos = 0
            # also build confusion matrix on video-level majority vote
            K_video = K
            video_conf = np.zeros((K_video, K_video), dtype=int)

            for vb, yt in video_y_true.items():
                preds = video_pred_votes.get(vb, [])
                if not preds:
                    continue
                cnt = Counter(preds)
                ordered = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))  # count desc, label asc
                top_labels = [lab for lab, _ in ordered]

                if len(top_labels) > 0 and yt == top_labels[0]:
                    correct_top1 += 1
                # confusion matrix: majority-vote prediction
                if len(top_labels) > 0 and 0 <= yt < K_video and 0 <= top_labels[0] < K_video:
                    video_conf[int(yt), int(top_labels[0])] += 1
                if yt in top_labels[:2]:
                    correct_top2 += 1
                if yt in top_labels[:3]:
                    correct_top3 += 1
                total_videos += 1

            acc1 = correct_top1 / total_videos if total_videos > 0 else 0.0
            acc2 = correct_top2 / total_videos if total_videos > 0 else 0.0
            acc3 = correct_top3 / total_videos if total_videos > 0 else 0.0

            print(
                f"\nSyracuse LOOCV (multitask) video-level majority-vote accuracy: "
                f"top-1={acc1:.4f} ({correct_top1}/{total_videos}), "
                f"top-2={acc2:.4f} ({correct_top2}/{total_videos}), "
                f"top-3={acc3:.4f} ({correct_top3}/{total_videos})"
            )

            # Persist video-level top-k metrics to a small CSV
            video_metrics_csv = os.path.join(save_dir, "syracuse_loocv_multitask_video_topk.csv")
            with open(video_metrics_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value", "correct", "total"])
                w.writerow(["top1_acc", f"{acc1:.6f}", correct_top1, total_videos])
                w.writerow(["top2_acc", f"{acc2:.6f}", correct_top2, total_videos])
                w.writerow(["top3_acc", f"{acc3:.6f}", correct_top3, total_videos])
            print(f"Syracuse LOOCV (multitask) video-level top-k metrics saved to: {video_metrics_csv}")

            # Persist video-level confusion matrix (majority-vote) across all videos
            video_conf_csv = os.path.join(save_dir, "syracuse_loocv_multitask_video_confusion.csv")
            with open(video_conf_csv, "w", newline="") as f:
                w = csv.writer(f)
                header = ["true\\pred"] + [f"{i}" for i in range(K_video)]
                w.writerow(header)
                for ti in range(K_video):
                    row_vals = [str(ti)] + [str(int(v)) for v in video_conf[ti]]
                    w.writerow(row_vals)
            print(f"Syracuse LOOCV (multitask) video-level confusion matrix saved to: {video_conf_csv}")

            # Persist per-video prediction details (majority vote, vote ratio, num_clips, pain_level, subject_id)
            # Load pain_level and subject_id from Syracuse meta Excel
            video_to_pain: Dict[str, Any] = {}
            video_to_subject: Dict[str, Any] = {}
            meta_excel_path = cfg.get("meta_excel_path")
            if meta_excel_path and os.path.isfile(meta_excel_path):
                try:
                    df_meta = pd.read_excel(meta_excel_path)
                    df_cols = {str(c).strip().lower(): c for c in df_meta.columns}
                    fname_col = None
                    for cand in ("file_name", "filename", "video_file", "video_name"):
                        if cand in df_cols:
                            fname_col = df_cols[cand]
                            break
                    subj_col = None
                    for cand in ("subject_id", "subject", "patient_id", "pid"):
                        if cand in df_cols:
                            subj_col = df_cols[cand]
                            break

                    if fname_col is not None:
                        def _video_base_from_file_name(x: Any) -> str:
                            base = os.path.basename(str(x))
                            if "." in base:
                                base = base.split(".")[0]
                            return base

                        df_meta["video_base"] = df_meta[fname_col].apply(_video_base_from_file_name)
                        for _, row in df_meta.iterrows():
                            vb = str(row["video_base"]).strip()
                            video_to_pain[vb] = row.get("pain_level")
                            if subj_col is not None:
                                video_to_subject[vb] = row.get(subj_col)
                except Exception:
                    # if meta loading fails, we still write detail CSV without pain/subject
                    pass

            detail_csv = os.path.join(save_dir, "syracuse_loocv_multitask_video_detail.csv")
            with open(detail_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["video_id", "subject_id", "true_class", "pred_class", "vote_ratio", "num_clips", "pain_level"]
                )
                for vb in sorted(video_y_true.keys()):
                    yt = video_y_true[vb]
                    preds = video_pred_votes.get(vb, [])
                    cnt = Counter(preds)
                    total = sum(cnt.values())
                    if total > 0 and cnt:
                        ordered = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
                        pred_cls, pred_count = ordered[0]
                        vote_ratio = float(pred_count) / float(total)
                    else:
                        pred_cls = ""
                        vote_ratio = ""
                    pain = video_to_pain.get(vb, "")
                    subj = video_to_subject.get(vb, "")
                    w.writerow([vb, subj, yt, pred_cls, vote_ratio, total, pain])
            print(f"Syracuse LOOCV (multitask) per-video prediction details saved to: {detail_csv}")


if __name__ == "__main__":
    main()
