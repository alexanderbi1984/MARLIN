import argparse
import os
from typing import List, Optional

import torch

import warnings
# Silence deprecation warning from PyTorch Lightning using pkg_resources
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="pytorch_lightning.utilities.imports",
)

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset.syracuse_bag import SyracuseBagDataModule
from model.mil_coral_xformer import MILCoralTransformer


def load_config(path: Optional[str]):
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is required to load config files. Install pyyaml or pass CLI args.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_class_weights(dm: SyracuseBagDataModule, num_classes: int) -> List[float]:
    # Build a temporary train dataset and count class labels
    ds = dm._make_dataset("train")
    counts = [0] * num_classes
    for entry in ds.index_items:
        vb = entry[0] if isinstance(entry, (list, tuple)) else entry
        y = ds._label_for(vb).item()
        if 0 <= y < num_classes:
            counts[int(y)] += 1
    total = sum(counts)
    # Inverse frequency weights
    weights = []
    for c in counts:
        if c == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * c))
    return weights


def main():
    parser = argparse.ArgumentParser(description="Evaluate Syracuse MIL + CORAL + Transformer")
    parser.add_argument("--config", type=str, default="config/syracuse_mil_coral_xformer.yaml")
    parser.add_argument("--feature_root", type=str, default=None)
    parser.add_argument("--meta_excel_path", type=str, default=None)
    parser.add_argument("--baseline_combo", type=str, default=None)
    parser.add_argument("--combos", type=str, nargs="*", default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    # Runtime/device overrides and resume support
    parser.add_argument("--accelerator", type=str, default=None, help="Override accelerator, e.g. 'gpu' or 'cpu'")
    parser.add_argument("--devices", type=int, default=None, help="Override number of devices, e.g. 1 for single GPU")
    parser.add_argument("--default_root_dir", type=str, default=None, help="Optional root dir for logs/checkpoints")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a .ckpt file to resume training from",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_videos", type=str, nargs="*", default=None, help="Optional video_base IDs to drop from all splits.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    feature_root = args.feature_root or cfg.get("syracuse_feature_root", "/data/Nbi/syracuse_fixed_combo_features")
    meta_excel_path = args.meta_excel_path or cfg.get("meta_excel_path", "/data/Nbi/meta_with_outcomes.xlsx")
    baseline_combo = args.baseline_combo or cfg.get("baseline_combo", "RGB")
    combos = args.combos or cfg.get("combos", ["RGB", "RGD", "RGT", "RBD", "RBT", "RDT", "GBD", "GBT", "GDT", "BDT"])
    task = args.task or cfg.get("task", "ordinal")
    num_classes = args.num_classes or int(cfg.get("num_classes", 5))

    # Prepare save_dir early and persist an effective config snapshot
    save_dir = cfg.get("save_dir", "Syracuse/xformer_mil_single")
    os.makedirs(save_dir, exist_ok=True)
    # Build effective config snapshot
    eff = dict(cfg)
    eff.update({
        "resolved": {
            "feature_root": feature_root,
            "meta_excel_path": meta_excel_path,
            "baseline_combo": baseline_combo,
            "combos": combos,
            "task": task,
            "num_classes": num_classes,
            "accelerator": cfg.get("accelerator", "gpu"),
            "devices": cfg.get("devices", 1),
            "script": "evaluate_syracuse_mil.py",
        }
    })
    try:
        if yaml is not None:
            with open(os.path.join(save_dir, "config_used.yaml"), "w") as f:
                yaml.safe_dump(eff, f, sort_keys=False)
    except Exception as e:
        print(f"Warning: failed to write config_used.yaml: {e}")

    exclude_videos = args.exclude_videos if args.exclude_videos is not None else cfg.get("exclude_video_ids")
    dm = SyracuseBagDataModule(
        feature_root=feature_root,
        meta_excel_path=meta_excel_path,
        task=task,
        num_classes=num_classes,
        pain_class_cutoffs=cfg.get("pain_class_cutoffs", [2.0, 4.0, 6.0, 8.0]),
        combos=combos,
        baseline_combo=baseline_combo,
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
        exclude_video_ids=exclude_videos,
    )
    dm.setup()

    # -------------------------
    # Print split sizes and label distributions
    # -------------------------
    def _vb_from_item(it):
        # item may be str (video_base) or (video_base, combo, path)
        if isinstance(it, (list, tuple)) and len(it) >= 1:
            return it[0]
        return it

    def summarize_split(name: str, ds) -> None:
        items = ds.index_items
        n = len(items)
        print(f"\n[{name}] samples: {n}")
        if n == 0:
            return
        if task in ("ordinal", "multiclass"):
            counts = [0] * num_classes
            for it in items:
                vb = _vb_from_item(it)
                y = ds._label_for(vb).item()
                if 0 <= y < num_classes:
                    counts[int(y)] += 1
            total = sum(counts)
            print("Label distribution (class: count | percent):")
            for k in range(num_classes):
                c = counts[k]
                pct = (100.0 * c / total) if total > 0 else 0.0
                print(f"  {k}: {c} | {pct:.1f}%")
        elif task == "binary":
            cnt0 = cnt1 = 0
            for it in items:
                vb = _vb_from_item(it)
                y = ds._label_for(vb).item()
                if int(y) == 1:
                    cnt1 += 1
                else:
                    cnt0 += 1
            total = cnt0 + cnt1
            p0 = 100.0 * cnt0 / total if total > 0 else 0.0
            p1 = 100.0 * cnt1 / total if total > 0 else 0.0
            print(f"Label distribution: 0 -> {cnt0} ({p0:.1f}%), 1 -> {cnt1} ({p1:.1f}%)")
        elif task == "regression":
            vals = []
            for it in items:
                vb = _vb_from_item(it)
                y = ds._label_for(vb).item()
                if not (y != y):  # filter NaN
                    vals.append(float(y))
            if not vals:
                print("No valid pain_level values in this split.")
            else:
                import math
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
                std = math.sqrt(var)
                vmin, vmax = min(vals), max(vals)
                print(f"pain_level stats: mean={mean:.3f}, std={std:.3f}, min={vmin:.3f}, max={vmax:.3f}")

    summarize_split("train", dm._make_dataset("train"))
    summarize_split("val", dm._make_dataset("val"))
    summarize_split("test", dm._make_dataset("test"))

    # Optional class weights for CE head
    class_weights = None
    if bool(cfg.get("class_weights", False)) and task in ("ordinal", "multiclass"):
        class_weights = compute_class_weights(dm, num_classes)

    model = MILCoralTransformer(
        input_dim=768,
        embed_dim=int(cfg.get("embed_dim", 256)),
        num_classes=num_classes,
        attn_type=str(cfg.get("attn_type", "xformer")),
        xformer_heads=int(cfg.get("xformer_heads", 4)),
        xformer_latents=int(cfg.get("xformer_latents", 16)),
        xformer_cross_layers=int(cfg.get("xformer_cross_layers", 1)),
        xformer_self_layers=int(cfg.get("xformer_self_layers", 2)),
        xformer_dropout=float(cfg.get("xformer_dropout", 0.1)),
        coral_alpha=float(cfg.get("coral_alpha", 1.0)),
        temperature=float(cfg.get("temperature", 1.0)),
        ce_weight=float(cfg.get("ce_weight", 0.25)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
        class_weights=class_weights,
    )

    pl.seed_everything(int(args.seed))

    ckpt_cb = ModelCheckpoint(monitor=str(cfg.get("monitor_metric", "val_qwk")), mode="max", save_top_k=1, save_last=True)
    es_cb = EarlyStopping(monitor=str(cfg.get("monitor_metric", "val_qwk")), mode="max", patience=int(cfg.get("patience", 100)))

    trainer = pl.Trainer(
        accelerator=(args.accelerator or str(cfg.get("accelerator", "gpu"))),
        devices=(args.devices or cfg.get("devices", 1)),
        max_epochs=int(args.max_epochs or cfg.get("max_epochs", 200)),
        precision=int(cfg.get("precision", 32)),
        callbacks=[ckpt_cb, es_cb],
        log_every_n_steps=10,
        default_root_dir=(args.default_root_dir or cfg.get("default_root_dir", None)),
    )

    # Optional resume from checkpoint
    resume_ckpt = None
    if args.resume_from is not None:
        if os.path.isfile(args.resume_from):
            resume_ckpt = args.resume_from
            print(f"Resuming training from checkpoint: {resume_ckpt}")
        else:
            print(f"--resume_from specified but file not found: {args.resume_from}")

    trainer.fit(
        model,
        datamodule=None,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
        ckpt_path=resume_ckpt,
    )
    # Test with best checkpoint when available; else fall back to last
    ckpt_choice = "best" if ckpt_cb.best_model_path else "last"
    trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path=ckpt_choice)

    # -------------------------
    # Export per-sample predictions and compute AUC
    # -------------------------
    save_preds = bool(cfg.get("save_preds", True))
    os.makedirs(save_dir, exist_ok=True)
    preds_batches = trainer.predict(model, dataloaders=dm.test_dataloader())
    y_true_all = []
    y_pred_all = []
    y_probs_all = []
    vids_all = []
    combos_all = []
    for out in preds_batches or []:
        y_true = out.get("y_true")
        y_pred = out.get("y_pred")
        y_probs = out.get("y_probs")
        vids = out.get("video_ids") or []
        cmbs = out.get("combos") or []
        if y_true is not None:
            y_true_all.extend([int(v) for v in y_true.view(-1).tolist()])
        if y_pred is not None:
            y_pred_all.extend([int(v) for v in y_pred.view(-1).tolist()])
        if y_probs is not None:
            y_probs_all.extend([list(map(float, row)) for row in y_probs.tolist()])
        if vids:
            vids_all.extend(list(vids))
        if cmbs:
            combos_all.extend(list(cmbs))

    # Save CSV
    if save_preds and y_pred_all:
        import csv
        out_csv = os.path.join(save_dir, "test_preds.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            # header
            header = ["video_id", "combo", "y_true", "y_pred"]
            K = int(num_classes)
            header += [f"p_{i}" for i in range(K)]
            writer.writerow(header)
            for i in range(len(y_pred_all)):
                row = [
                    vids_all[i] if i < len(vids_all) else "",
                    combos_all[i] if i < len(combos_all) else "",
                    y_true_all[i] if i < len(y_true_all) else "",
                    y_pred_all[i],
                ]
                probs = y_probs_all[i] if i < len(y_probs_all) else [""] * K
                writer.writerow(row + probs)
        print(f"Saved test predictions to {out_csv}")

    # Compute AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc_value = None
        if y_probs_all and y_true_all:
            import numpy as np
            y_true_np = np.array(y_true_all)
            y_probs_np = np.array(y_probs_all)
            if task in ("binary",):
                # assume class 1 probability is column 1 if K==2
                if y_probs_np.shape[1] >= 2:
                    auc_value = roc_auc_score(y_true_np, y_probs_np[:, 1])
            elif task in ("ordinal", "multiclass"):
                # macro-average One-vs-Rest AUC
                auc_value = roc_auc_score(y_true_np, y_probs_np, multi_class="ovr", average="macro")
        if auc_value is not None:
            print(f"Test AUC: {auc_value:.4f}")
        else:
            print("Test AUC not available (insufficient probability outputs or labels)")
    except Exception as e:
        print(f"AUC computation skipped: {e}")


if __name__ == "__main__":
    main()
