import argparse
import os
import csv
from typing import List, Dict, Any, Optional

import pytorch_lightning as pl

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from dataset.syracuse_bag import SyracuseBagDataModule
from model.mil_coral_xformer import MILCoralTransformer


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _load_existing_rows(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Load existing per-subject rows from a combo CSV, keyed by subject string."""
    if not os.path.isfile(csv_path):
        return {}
    by_subj: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = str(row.get("subject", "")).strip()
            if not sid or sid.upper() == "MEAN":
                continue
            by_subj[sid] = row
    return by_subj


def _read_confusion_csv(path: str, K: int):
    import numpy as np
    if not os.path.isfile(path):
        return np.zeros((K, K), dtype=int)
    mat = []
    with open(path, "r") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for row in rd:
            # row like [true_label, c0, c1, ...]
            if len(row) >= K + 1:
                vals = [int(float(v)) for v in row[1:K+1]]
                mat.append(vals)
    import numpy as np
    arr = np.array(mat, dtype=int)
    if arr.shape != (K, K):
        return np.zeros((K, K), dtype=int)
    return arr


def _compute_metrics_from_preds(preds_csv: str, K: int) -> Optional[Dict[str, Any]]:
    """Compute metrics used in LOOCV from a saved preds CSV for one subject."""
    if not os.path.isfile(preds_csv):
        return None
    y_true: List[int] = []
    y_pred: List[int] = []
    with open(preds_csv, "r") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        # Expect columns: video_id, y_true, y_pred (from this file's writer)
        # In this script it writes [video_id, y_true, y_pred]
        for row in rd:
            if not row:
                continue
            try:
                # handle possible header mismatch; be robust by using last two numeric columns
                vals = [v for v in row if v is not None and v != ""]
                yt = int(float(vals[-2]))
                yp = int(float(vals[-1]))
                y_true.append(yt)
                y_pred.append(yp)
            except Exception:
                continue
    if not y_true:
        return None
    try:
        from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, mean_absolute_error
        import numpy as np
        yt = np.array(y_true, dtype=int)
        yp = np.array(y_pred, dtype=int)
        qwk = cohen_kappa_score(yt, yp, weights="quadratic")
        acc = accuracy_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        f1_macro = f1_score(yt, yp, average="macro", labels=list(range(K)), zero_division=0)
        f1_weighted = f1_score(yt, yp, average="weighted", labels=list(range(K)), zero_division=0)
        f1_micro = f1_score(yt, yp, average="micro", labels=list(range(K)), zero_division=0)
        return {
            "test_qwk": float(qwk),
            "test_acc": float(acc),
            "test_mae": float(mae),
            "test_f1_macro": float(f1_macro),
            "test_f1_weighted": float(f1_weighted),
            "test_f1_micro": float(f1_micro),
        }
    except Exception:
        return None


def load_config(path: Optional[str]):
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is required to load config files. Install pyyaml or pass CLI args.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def discover_subjects(feature_root: str, combo: str, meta_excel_path: str, task: str, num_classes: int, exclude_videos: Optional[List[str]] = None, feature_suffix: str = "_windows.npy") -> List[str]:
    dm = SyracuseBagDataModule(
        feature_root=feature_root,
        meta_excel_path=meta_excel_path,
        task=task,
        num_classes=num_classes,
        combos=[combo],
        baseline_combo=combo,
        batch_size=1,  # not used here
        clip_level=False,
        exclude_video_ids=exclude_videos,
        feature_suffix=feature_suffix,
    )
    meta = dm._load_meta_excel()
    videos = dm._list_videos_for_combo(combo)
    # filter videos that have valid labels
    valid: List[str] = []
    for vb in videos:
        m = meta.get(vb, {})
        if task == "regression":
            ok = (m.get("pain_level") is not None)
        elif task in ("ordinal", "multiclass"):
            key = f"class_{num_classes}"
            ok = (m.get(key) is not None) or (m.get("pain_level") is not None)
        elif task == "binary":
            ok = (m.get("outcome") is not None) or (m.get("pain_level") is not None)
        else:
            ok = False
        if ok:
            valid.append(vb)
    # collect subjects
    subs = []
    for vb in valid:
        sid = meta.get(vb, {}).get("subject_id")
        if sid is None:
            continue
        subs.append(str(sid))
    uniq = sorted(list({s for s in subs if s.lower() != "nan"}))
    return uniq


def _confusion_from_preds(y_true: List[int], y_pred: List[int], num_classes: int):
    import numpy as np
    K = num_classes
    conf = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < K and 0 <= p < K:
            conf[t, p] += 1
    return conf


def _compute_class_weights(dm: SyracuseBagDataModule, num_classes: int) -> Optional[List[float]]:
    # Compute inverse-frequency class weights on the training split
    try:
        ds = dm._make_dataset("train")
        counts = [0] * num_classes
        for it in ds.index_items:
            vb = it[0] if isinstance(it, (list, tuple)) else it
            y = ds._label_for(vb).item()
            if 0 <= y < num_classes:
                counts[int(y)] += 1
        total = sum(counts)
        weights: List[float] = []
        for c in counts:
            weights.append((total / (num_classes * c)) if c > 0 else 0.0)
        return weights
    except Exception:
        return None


def run_fold(cfg: Dict[str, Any], combo: str, test_subject: str, seed: int, save_dir: Optional[str] = None) -> Dict[str, Any]:
    # DataModule with LOOCV split
    dm = SyracuseBagDataModule(
        feature_root=cfg.get("syracuse_feature_root"),
        meta_excel_path=cfg.get("meta_excel_path"),
        task=str(cfg.get("task", "ordinal")),
        num_classes=int(cfg.get("num_classes", 5)),
        pain_class_cutoffs=cfg.get("pain_class_cutoffs", [2.0, 4.0, 6.0, 8.0]),
        combos=[combo],
        baseline_combo=combo,
        batch_size=int(cfg.get("batch_size", 8)),
        num_workers=int(cfg.get("num_workers", 0)),
        val_split_ratio=float(cfg.get("val_split_ratio", 0.15)),
        test_split_ratio=float(cfg.get("test_split_ratio", 0.15)),
        random_state=seed,
        max_bag_size=int(cfg.get("max_bag_size")) if cfg.get("max_bag_size") is not None else None,
        normalize_features=bool(cfg.get("normalize_features", True)),
        aug_feature_root=cfg.get("aug_feature_root"),
        use_aug_in_train=bool(cfg.get("use_aug_in_train", True)),
        train_aug_ratio=float(cfg.get("train_aug_ratio", 0.5)),
        use_weighted_sampler=bool(cfg.get("use_weighted_sampler", False)),
        train_epoch_multiplier=int(cfg.get("train_epoch_multiplier", 1)),
        train_enumerate_all_views=bool(cfg.get("train_enumerate_all_views", True)),
        train_max_aug_per_combo=int(cfg.get("train_max_aug_per_combo", 2)),
        train_include_original=bool(cfg.get("train_include_original", True)),
        loocv_test_subject=str(test_subject),
        clip_level=bool(cfg.get("clip_level", False)),
        clip_bag_size=int(cfg.get("clip_bag_size", 1)),
        clip_bag_stride=int(cfg.get("clip_bag_stride", 1)),
        exclude_video_ids=cfg.get("exclude_video_ids"),
        feature_suffix=cfg.get("syracuse_feature_suffix", "_windows.npy"),
    )
    dm.setup()

    # Build model (optionally with CE class weights and/or a pretrained MIL backbone)
    class_weights = None
    if bool(cfg.get("class_weights", False)) and str(cfg.get("task", "ordinal")) in ("ordinal", "multiclass"):
        class_weights = _compute_class_weights(dm, int(cfg.get("num_classes", 5)))

    pretrained_mil_ckpt = cfg.get("pretrained_mil_ckpt")
    freeze_mil = bool(cfg.get("freeze_mil", False))

    if pretrained_mil_ckpt:
        # Use MME-pretrained Transformer MIL aggregator; train CE head on Syracuse.
        # We disable CORAL loss (coral_alpha=0) so that only CE contributes.
        model = MILCoralTransformer.from_mme_checkpoint(
            ckpt_path=str(pretrained_mil_ckpt),
            num_classes=int(cfg.get("num_classes", 5)),
            coral_alpha=0.0,
            ce_weight=float(cfg.get("ce_weight", 1.0)),
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-2)),
            class_weights=class_weights,
        )
        if freeze_mil:
            # Freeze MIL aggregator (and optionally CORAL head) to use as a fixed backbone.
            for p in model.aggregator.parameters():
                p.requires_grad = False
            if hasattr(model, "coral_head"):
                for p in model.coral_head.parameters():
                    p.requires_grad = False
    else:
        model = MILCoralTransformer(
            input_dim=int(cfg.get("input_dim", 768)),
            embed_dim=int(cfg.get("embed_dim", 256)),
            num_classes=int(cfg.get("num_classes", 5)),
            attn_type=str(cfg.get("attn_type", "xformer")),
            xformer_heads=int(cfg.get("xformer_heads", 4)),
            xformer_latents=int(cfg.get("xformer_latents", 16)),
            xformer_cross_layers=int(cfg.get("xformer_cross_layers", 1)),
            xformer_self_layers=int(cfg.get("xformer_self_layers", 2)),
            xformer_dropout=float(cfg.get("xformer_dropout", 0.1)),
            coral_alpha=float(cfg.get("coral_alpha", 1.0)),
            temperature=float(cfg.get("temperature", 1.0)),
            ce_weight=float(cfg.get("ce_weight", 0.0)),
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            weight_decay=float(cfg.get("weight_decay", 1e-2)),
            class_weights=class_weights,
        )

    pl.seed_everything(seed)

    ckpt_cb = pl.callbacks.ModelCheckpoint(monitor=str(cfg.get("monitor_metric", "val_qwk")), mode="max", save_top_k=1, save_last=True)
    es_cb = pl.callbacks.EarlyStopping(monitor=str(cfg.get("monitor_metric", "val_qwk")), mode="max", patience=int(cfg.get("patience", 100)))

    trainer = pl.Trainer(
        accelerator=str(cfg.get("accelerator", "gpu")),
        devices=cfg.get("devices", 1),
        max_epochs=int(cfg.get("max_epochs", 200)),
        precision=int(cfg.get("precision", 32)),
        callbacks=[ckpt_cb, es_cb],
        log_every_n_steps=10,
    )

    # Fit and test on the held-out subject
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    # Test with best checkpoint when available; else fall back to last
    ckpt_choice = "best" if ckpt_cb.best_model_path else "last"
    test_results = trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path=ckpt_choice)
    metrics = test_results[0] if isinstance(test_results, list) and len(test_results) > 0 else {}
    # Predict per-sample to derive confusion matrix and optionally save
    preds_batches = trainer.predict(model, dataloaders=dm.test_dataloader())
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
    conf = _confusion_from_preds(y_true_all, y_pred_all, int(cfg.get("num_classes", 5)))
    metrics["conf"] = conf.tolist()

    # Optional persistence
    if save_dir is not None:
        import csv
        os.makedirs(save_dir, exist_ok=True)
        # per-fold predictions
        preds_csv = os.path.join(save_dir, f"preds_{combo}_subject_{test_subject}.csv")
        with open(preds_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "y_true", "y_pred"])
            for i in range(len(y_pred_all)):
                vid = vids_all[i] if i < len(vids_all) else ""
                yt = y_true_all[i] if i < len(y_true_all) else None
                yp = y_pred_all[i]
                w.writerow([vid, yt, yp])
        # confusion matrix CSV
        conf_csv = os.path.join(save_dir, f"confusion_{combo}_subject_{test_subject}.csv")
        with open(conf_csv, "w", newline="") as f:
            w = csv.writer(f)
            K = int(cfg.get("num_classes", 5))
            header = ["true\\pred"] + [f"{i}" for i in range(K)]
            w.writerow(header)
            for i in range(K):
                row = [str(i)] + [str(int(v)) for v in conf[i]]
                w.writerow(row)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Subject-level LOOCV for Syracuse MIL (single combo)")
    parser.add_argument("--config", type=str, default="config/syracuse_mil_coral_xformer.yaml")
    parser.add_argument("--combo", type=str, required=True, help="One of the 10 combos, e.g., RGB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_subjects", type=int, default=None, help="Optionally limit to first N subjects for quick run")
    parser.add_argument("--save_dir", type=str, default="loocv_results")
    parser.add_argument("--resume", action="store_true", help="Skip subjects with existing preds/confusion files or rows in combo CSV")
    parser.add_argument("--accelerator", type=str, default=None, help="Override accelerator, e.g. 'gpu' or 'cpu'")
    parser.add_argument("--devices", type=int, default=None, help="Override number of devices, e.g. 1 for single GPU")
    parser.add_argument(
        "--exclude_videos", type=str, nargs="*", default=None, help="Video base IDs to exclude from all splits."
    )
    parser.add_argument(
        "--pretrained_mil_ckpt",
        type=str,
        default=None,
        help="Path to an MME-style MIL checkpoint to initialize the aggregator from.",
    )
    parser.add_argument(
        "--freeze_mil",
        action="store_true",
        help="If set together with --pretrained_mil_ckpt, freeze the MIL aggregator (and CORAL head).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Persist a copy of the used config and script metadata
    os.makedirs(args.save_dir, exist_ok=True)
    eff = dict(cfg)
    eff.update({
        "resolved": {
            "script": "evaluate_syracuse_mil_loocv.py",
            "combo": args.combo,
            "seed": args.seed,
            "save_dir": args.save_dir,
        }
    })
    try:
        if yaml is not None:
            with open(os.path.join(args.save_dir, "config_used.yaml"), "w") as f:
                yaml.safe_dump(eff, f, sort_keys=False)
    except Exception as e:
        print(f"Warning: failed to write config_used.yaml: {e}")
    # Optional runtime overrides (e.g., single-GPU resume)
    if args.accelerator is not None:
        cfg["accelerator"] = args.accelerator
    if args.devices is not None:
        cfg["devices"] = int(args.devices)
    if args.pretrained_mil_ckpt is not None:
        cfg["pretrained_mil_ckpt"] = args.pretrained_mil_ckpt
    if args.freeze_mil:
        cfg["freeze_mil"] = True
    exclude_videos = args.exclude_videos if args.exclude_videos is not None else cfg.get("exclude_video_ids")
    if exclude_videos is not None:
        cfg["exclude_video_ids"] = exclude_videos
    feature_root = cfg.get("syracuse_feature_root")
    meta_excel = cfg.get("meta_excel_path")
    task = str(cfg.get("task", "ordinal"))
    num_classes = int(cfg.get("num_classes", 5))
    combo = args.combo
    feature_suffix = cfg.get("syracuse_feature_suffix", "_windows.npy")

    subjects = discover_subjects(feature_root, combo, meta_excel, task, num_classes, exclude_videos, feature_suffix=feature_suffix)
    if args.limit_subjects is not None:
        subjects = subjects[: int(args.limit_subjects)]
    print(f"LOOCV over {len(subjects)} subjects with combo={combo}")

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"syracuse_loocv_{combo}.csv")

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

    # Resume support: load existing rows if present
    out_path = os.path.join(args.save_dir, f"syracuse_loocv_{combo}.csv")
    existing_by_subject: Dict[str, Dict[str, Any]] = _load_existing_rows(out_path) if args.resume else {}

    # accumulate confusion matrix across folds for this combo
    import numpy as np
    K = int(num_classes)
    sum_conf = np.zeros((K, K), dtype=int)

    # Pre-populate rows and confusion from existing results if resuming
    if args.resume and existing_by_subject:
        print(f"Resuming: found existing rows for {len(existing_by_subject)} subjects; will skip them.")
        fold_save_dir = os.path.join(args.save_dir, f"{combo}")
        for sid, erow in existing_by_subject.items():
            rows.append({**erow})
            conf_csv = os.path.join(fold_save_dir, f"confusion_{combo}_subject_{sid}.csv")
            if os.path.isfile(conf_csv):
                sum_conf += _read_confusion_csv(conf_csv, K)
            else:
                # fallback: derive conf from preds
                preds_csv = os.path.join(fold_save_dir, f"preds_{combo}_subject_{sid}.csv")
                if os.path.isfile(preds_csv):
                    # derive confusion
                    y_true: List[int] = []
                    y_pred: List[int] = []
                    with open(preds_csv, "r") as f:
                        rd = csv.reader(f)
                        _ = next(rd, None)
                        for r in rd:
                            try:
                                vals = [v for v in r if v is not None and v != ""]
                                yt = int(float(vals[-2]))
                                yp = int(float(vals[-1]))
                                if 0 <= yt < K and 0 <= yp < K:
                                    y_true.append(yt)
                                    y_pred.append(yp)
                            except Exception:
                                continue
                    if y_true:
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
                        sum_conf += cm.astype(int)

    # Determine which subjects still need to run
    remaining_subjects: List[str] = []
    for sid in subjects:
        if args.resume and sid in existing_by_subject:
            continue
        # also skip if fold-level preds already exist and resume is on
        if args.resume:
            fold_save_dir = os.path.join(args.save_dir, f"{combo}")
            preds_csv = os.path.join(fold_save_dir, f"preds_{combo}_subject_{sid}.csv")
            if os.path.isfile(preds_csv):
                # build metrics from preds and append
                m = _compute_metrics_from_preds(preds_csv, K)
                if m is not None:
                    rows.append({"subject": sid, **m})
                # add confusion
                conf_csv = os.path.join(fold_save_dir, f"confusion_{combo}_subject_{sid}.csv")
                if os.path.isfile(conf_csv):
                    sum_conf += _read_confusion_csv(conf_csv, K)
                else:
                    # derive confusion from preds
                    y_true: List[int] = []
                    y_pred: List[int] = []
                    with open(preds_csv, "r") as f:
                        rd = csv.reader(f)
                        _ = next(rd, None)
                        for r in rd:
                            try:
                                vals = [v for v in r if v is not None and v != ""]
                                yt = int(float(vals[-2]))
                                yp = int(float(vals[-1]))
                                if 0 <= yt < K and 0 <= yp < K:
                                    y_true.append(yt)
                                    y_pred.append(yp)
                            except Exception:
                                continue
                    if y_true:
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
                        sum_conf += cm.astype(int)
                print(f"Skipping existing subject {sid}")
                continue
        remaining_subjects.append(sid)

    # Run remaining subjects
    for i, sid in enumerate(remaining_subjects, 1):
        print(f"\nFold {i}/{len(remaining_subjects)} — test_subject={sid}")
        fold_save_dir = os.path.join(args.save_dir, f"{combo}")
        metrics = run_fold(cfg, combo, sid, args.seed, save_dir=fold_save_dir)
        row = {"subject": sid}
        for k in fields[1:]:
            row[k] = metrics.get(k, None)
        rows.append(row)
        print("Metrics:", row)
        # add to confusion accumulator
        if metrics.get("conf") is not None:
            arr = np.array(metrics["conf"], dtype=int)
            if arr.shape == (K, K):
                sum_conf += arr

    # Write CSV
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        # Summary row (means)
        if rows:
            summary: Dict[str, Any] = {"subject": "MEAN"}
            for k in fields[1:]:
                vals = [float(r[k]) for r in rows if r.get(k) is not None]
                summary[k] = sum(vals) / len(vals) if vals else None
            w.writerow(summary)

    print(f"\nLOOCV results saved to: {out_path}")


if __name__ == "__main__":
    main()
