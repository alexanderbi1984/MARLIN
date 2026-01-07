import argparse
import csv
import os
import random
import warnings
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd

# Silence noisy deprecation warning seen in PL imports on some systems.
warnings.filterwarnings("ignore", category=UserWarning)

import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from dataset.video_dataset import VideoDataset
from train_syracuse_finetune import parse_syracuse_metadata, PainTransformerFinetuneModule
from evaluate_syracuse_mil_loocv import _confusion_from_preds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Syracuse TNT fine-tune pipeline: LOSO folds -> per-subject video-level preds -> global summary."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--biovid_checkpoint", type=str, default=None, help="Path to best BioVid checkpoint")

    # Fold selection / parallelization
    parser.add_argument("--list_subjects", action="store_true", help="Print discovered subject IDs then exit.")
    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Do not train; only aggregate existing clip/video outputs under save_dir.",
    )
    parser.add_argument("--only_subject", type=str, default=None, help="Run only one LOSO test subject (e.g. '11').")
    parser.add_argument("--shard_id", type=int, default=None, help="Shard index for subject slicing (0-based).")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards for subject slicing.")

    # Output / summary
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override save_dir in YAML (directory to write baseline-aligned metrics).",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Skip global summary at the end (recommended for shard/only runs).",
    )
    return parser.parse_args()


def _select_subjects(
    subjects: List[str],
    limit_subjects: Optional[int],
    only_subject: Optional[str],
    shard_id: Optional[int],
    num_shards: Optional[int],
) -> List[str]:
    subs = list(subjects)
    if limit_subjects:
        subs = subs[: int(limit_subjects)]

    if only_subject is not None:
        only = str(only_subject).strip()
        if only not in subs:
            raise ValueError(f"--only_subject={only} not found in discovered subjects.")
        subs = [only]

    if (shard_id is None) != (num_shards is None):
        raise ValueError("Both --shard_id and --num_shards must be provided together.")
    if shard_id is not None and num_shards is not None:
        shard_id = int(shard_id)
        num_shards = int(num_shards)
        if num_shards <= 0:
            raise ValueError("--num_shards must be > 0")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"--shard_id must be in [0, {num_shards-1}]")
        subs = subs[shard_id::num_shards]
    return subs


def main() -> None:
    args = _parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pl.seed_everything(42)

    # Where to write baseline-aligned artifacts
    save_dir = args.save_dir or config.get("save_dir") or "/data/Nbi/Marlin/MARLIN/logs_syracuse_finetune/tnt_baseline_aligned"
    save_preds = bool(config.get("save_preds", True))
    clip_out_dir = os.path.join(save_dir, "multitask")  # keep baseline folder name for strict alignment
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(clip_out_dir, exist_ok=True)

    use_aug = bool(config.get("use_aug", False))
    aug_root = config.get("aug_video_root") if use_aug else None

    df = parse_syracuse_metadata(
        config["meta_excel"],
        config["video_root"],
        video_col=config["video_col"],
        label_col=config["label_col"],
        subject_col=config["subject_col"],
        cutoffs=config["cutoffs"],
        exclude_video_ids=config.get("exclude_video_ids", []),
        aug_root=aug_root,
    )

    subjects = sorted([str(s) for s in df["subject"].unique().tolist()])
    print(f"Found {len(subjects)} subjects: {subjects}")

    if args.list_subjects:
        for s in subjects:
            print(s)
        return

    run_subjects = _select_subjects(
        subjects=subjects,
        limit_subjects=config.get("limit_subjects"),
        only_subject=args.only_subject,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    print(f"Running subjects: {run_subjects}")

    # Clip-level protocol config
    clip_level = bool(config.get("clip_level", False))
    clip_len_frames = int(config.get("clip_len_frames", 150))
    clip_stride_frames = int(config.get("clip_stride_frames", 15))

    # Accumulate baseline-style LOOCV metrics & confusion across all processed subjects in this process
    K = int(config["num_classes"])
    fields = [
        "subject",
        "test_qwk",
        "test_acc",
        "test_mae",
        "test_f1_macro",
        "test_f1_weighted",
        "test_f1_micro",
    ]
    rows: List[dict] = []
    sum_conf = np.zeros((K, K), dtype=int)

    # Helper: compute metrics from clip-level predictions (baseline uses sklearn; we implement directly)
    def _qwk(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
        # Quadratic weighted kappa
        Kc = int(num_classes)
        conf = _confusion_from_preds(y_true, y_pred, Kc).astype(np.float64)
        n = conf.sum()
        if n <= 0:
            return 0.0
        # histograms
        hist_t = conf.sum(axis=1)
        hist_p = conf.sum(axis=0)
        # expected matrix
        expected = np.outer(hist_t, hist_p) / n
        # weight matrix
        w = np.zeros((Kc, Kc), dtype=np.float64)
        for i in range(Kc):
            for j in range(Kc):
                w[i, j] = ((i - j) ** 2) / float((Kc - 1) ** 2) if Kc > 1 else 0.0
        num = (w * conf).sum()
        den = (w * expected).sum()
        return float(1.0 - (num / den)) if den > 0 else 0.0

    def _f1_scores(y_true: List[int], y_pred: List[int], num_classes: int) -> tuple[float, float, float]:
        cm = _confusion_from_preds(y_true, y_pred, num_classes).astype(np.float64)
        supports = cm.sum(axis=1)
        total = supports.sum()
        f1s = []
        weighted = 0.0
        for k in range(num_classes):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            f1s.append(float(f1))
            if total > 0:
                weighted += float((supports[k] / total) * f1)
        macro = float(np.mean(f1s)) if f1s else 0.0
        # micro-F1 equals accuracy in single-label multiclass
        acc = float(np.trace(cm) / total) if total > 0 else 0.0
        micro = acc
        return macro, float(weighted), float(micro)

    def _load_clip_preds_csv(path: str) -> tuple[List[str], List[int], List[int]]:
        sids: List[str] = []
        yt: List[int] = []
        yp: List[int] = []
        with open(path, "r") as f:
            rd = csv.reader(f)
            _ = next(rd, None)
            for row in rd:
                if not row:
                    continue
                try:
                    vid = str(row[0]).strip()
                    yti = int(float(row[1]))
                    ypi = int(float(row[2]))
                except Exception:
                    continue
                sids.append(vid)
                yt.append(yti)
                yp.append(ypi)
        return sids, yt, yp

    def _write_global_outputs() -> None:
        """Write baseline-aligned global outputs from rows/sum_conf and existing per-subject preds."""
        # LOOCV metrics CSV + MEAN row
        if save_preds:
            out_csv = os.path.join(save_dir, "syracuse_loocv_multitask.csv")
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
                if rows:
                    summary_row = {"subject": "MEAN"}
                    for k in fields[1:]:
                        vals = [float(rr[k]) for rr in rows if rr.get(k) is not None]
                        summary_row[k] = float(sum(vals) / len(vals)) if vals else None
                    w.writerow(summary_row)
            print(f"\nSaved LOOCV metrics to: {out_csv}")

            # Aggregated clip-level confusion
            agg_conf_csv = os.path.join(save_dir, "syracuse_loocv_multitask_confusion.csv")
            with open(agg_conf_csv, "w", newline="") as f:
                w = csv.writer(f)
                header = ["true\\pred"] + [f"{i}" for i in range(K)]
                w.writerow(header)
                for ti in range(K):
                    row_vals = [str(ti)] + [str(int(v)) for v in sum_conf[ti]]
                    w.writerow(row_vals)
            print(f"Saved aggregated clip-level confusion to: {agg_conf_csv}")

        # Video-level majority vote across all saved clip-level preds
        video_y_true: dict[str, int] = {}
        from collections import defaultdict
        video_pred_votes: dict[str, List[int]] = defaultdict(list)

        for fname in sorted(os.listdir(clip_out_dir)):
            if not fname.startswith("preds_multitask_subject_") or not fname.endswith(".csv"):
                continue
            fpath = os.path.join(clip_out_dir, fname)
            sids, y_true_all, y_pred_all = _load_clip_preds_csv(fpath)
            for sid, yt, yp in zip(sids, y_true_all, y_pred_all):
                base = sid.split("_clip_")[0] if "_clip_" in sid else sid
                if base not in video_y_true:
                    video_y_true[base] = int(yt)
                video_pred_votes[base].append(int(yp))

        correct_top1 = correct_top2 = correct_top3 = 0
        total_videos = 0
        video_conf = np.zeros((K, K), dtype=int)
        for vb, yt in video_y_true.items():
            preds = video_pred_votes.get(vb, [])
            if not preds:
                continue
            cnt = Counter(preds)
            ordered = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
            top_labels = [lab for lab, _ in ordered]
            if top_labels and yt == top_labels[0]:
                correct_top1 += 1
            if top_labels and 0 <= yt < K and 0 <= top_labels[0] < K:
                video_conf[int(yt), int(top_labels[0])] += 1
            if yt in top_labels[:2]:
                correct_top2 += 1
            if yt in top_labels[:3]:
                correct_top3 += 1
            total_videos += 1

        acc1 = float(correct_top1 / total_videos) if total_videos > 0 else 0.0
        acc2 = float(correct_top2 / total_videos) if total_videos > 0 else 0.0
        acc3 = float(correct_top3 / total_videos) if total_videos > 0 else 0.0

        print(
            f"\nSyracuse LOOCV (TNT) video-level majority-vote accuracy: "
            f"top-1={acc1:.4f} ({correct_top1}/{total_videos}), "
            f"top-2={acc2:.4f} ({correct_top2}/{total_videos}), "
            f"top-3={acc3:.4f} ({correct_top3}/{total_videos})"
        )

        if not save_preds:
            return

        video_metrics_csv = os.path.join(save_dir, "syracuse_loocv_multitask_video_topk.csv")
        with open(video_metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value", "correct", "total"])
            w.writerow(["top1_acc", f"{acc1:.6f}", correct_top1, total_videos])
            w.writerow(["top2_acc", f"{acc2:.6f}", correct_top2, total_videos])
            w.writerow(["top3_acc", f"{acc3:.6f}", correct_top3, total_videos])
        print(f"Saved video-level top-k metrics to: {video_metrics_csv}")

        video_conf_csv = os.path.join(save_dir, "syracuse_loocv_multitask_video_confusion.csv")
        with open(video_conf_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["true\\pred"] + [f"{i}" for i in range(K)]
            w.writerow(header)
            for ti in range(K):
                row_vals = [str(ti)] + [str(int(v)) for v in video_conf[ti]]
                w.writerow(row_vals)
        print(f"Saved video-level confusion to: {video_conf_csv}")

        # Per-video detail CSV
        video_to_pain: dict[str, object] = {}
        video_to_subject: dict[str, object] = {}
        meta_path = config.get("meta_excel")
        if meta_path and os.path.isfile(meta_path):
            try:
                df_meta = pd.read_excel(meta_path)
                df_meta.columns = [str(c).strip() for c in df_meta.columns]
                cols = {str(c).strip().lower(): c for c in df_meta.columns}
                fname_col = cols.get("file_name") or cols.get("filename") or cols.get("video_file") or cols.get("video_name")
                subj_col = cols.get("subject_id") or cols.get("subject") or cols.get("patient_id") or cols.get("pid")
                if fname_col is not None:
                    def _base(x: object) -> str:
                        b = os.path.basename(str(x))
                        return b.split(".")[0] if "." in b else b
                    df_meta["video_base"] = df_meta[fname_col].apply(_base)
                    for _, row in df_meta.iterrows():
                        vb = str(row["video_base"]).strip()
                        video_to_pain[vb] = row.get("pain_level")
                        if subj_col is not None:
                            video_to_subject[vb] = row.get(subj_col)
            except Exception:
                pass

        detail_csv = os.path.join(save_dir, "syracuse_loocv_multitask_video_detail.csv")
        with open(detail_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "subject_id", "true_class", "pred_class", "vote_ratio", "num_clips", "pain_level"])
            for vb in sorted(video_y_true.keys()):
                yt = video_y_true[vb]
                preds = video_pred_votes.get(vb, [])
                cnt = Counter(preds)
                total = int(sum(cnt.values()))
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
        print(f"Saved per-video detail CSV to: {detail_csv}")

    if args.summarize_only:
        # Aggregate existing per-subject clip-level predictions under save_dir/multitask
        existing = sorted([fn for fn in os.listdir(clip_out_dir) if fn.startswith("preds_multitask_subject_") and fn.endswith(".csv")])
        if not existing:
            raise FileNotFoundError(f"No preds_multitask_subject_*.csv found under: {clip_out_dir}")
        rows.clear()
        sum_conf[:] = 0
        for fn in existing:
            sid = fn.replace("preds_multitask_subject_", "").replace(".csv", "")
            fpath = os.path.join(clip_out_dir, fn)
            _, y_true_all, y_pred_all = _load_clip_preds_csv(fpath)
            conf = _confusion_from_preds(y_true_all, y_pred_all, K)
            sum_conf += conf
            total = len(y_true_all)
            correct = int(np.sum(np.asarray(y_true_all) == np.asarray(y_pred_all))) if total > 0 else 0
            test_acc = float(correct / total) if total > 0 else 0.0
            test_mae = float(np.mean(np.abs(np.asarray(y_true_all) - np.asarray(y_pred_all)))) if total > 0 else 0.0
            test_qwk = _qwk(y_true_all, y_pred_all, K)
            f1_macro, f1_weighted, f1_micro = _f1_scores(y_true_all, y_pred_all, K)
            rows.append(
                {
                    "subject": str(sid),
                    "test_qwk": test_qwk,
                    "test_acc": test_acc,
                    "test_mae": test_mae,
                    "test_f1_macro": f1_macro,
                    "test_f1_weighted": f1_weighted,
                    "test_f1_micro": f1_micro,
                }
            )
        rows.sort(key=lambda r: str(r.get("subject", "")))
        _write_global_outputs()
        return

    for test_subj in run_subjects:
        print(f"\n=== LOSO Fold: Subject {test_subj} ===")

        test_df = df[df["subject"] == test_subj]
        remaining_subjects = [s for s in subjects if s != test_subj]

        # Nested CV: split remaining subjects into train/val
        rng = random.Random(42 + hash(test_subj) % 1000)
        rng.shuffle(remaining_subjects)
        n_val = max(1, int(0.15 * len(remaining_subjects)))
        val_subjects = remaining_subjects[:n_val]
        train_subjects = remaining_subjects[n_val:]

        train_df = df[df["subject"].isin(train_subjects)]
        val_df = df[df["subject"].isin(val_subjects)]

        print(f"Split: Train={len(train_subjects)} subj, Val={len(val_subjects)} subj, Test={test_subj}")

        train_ds = VideoDataset(
            train_df["full_path"].tolist(),
            train_df["target"].tolist(),
            is_train=True,
            aug_paths_list=train_df["aug_paths"].tolist() if use_aug else None,
            aug_ratio=float(config.get("aug_ratio", 0.0)) if use_aug else 0.0,
            clip_level=clip_level,
            clip_len_frames=clip_len_frames,
            clip_stride_frames=clip_stride_frames,
        )
        val_ds = VideoDataset(
            val_df["full_path"].tolist(),
            val_df["target"].tolist(),
            is_train=False,
            clip_level=clip_level,
            clip_len_frames=clip_len_frames,
            clip_stride_frames=clip_stride_frames,
        )
        test_ds = VideoDataset(
            test_df["full_path"].tolist(),
            test_df["target"].tolist(),
            is_train=False,
            clip_level=clip_level,
            clip_len_frames=clip_len_frames,
            clip_stride_frames=clip_stride_frames,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["num_workers"]),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
        )

        model = PainTransformerFinetuneModule(
            num_classes=int(config["num_classes"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            max_epochs=int(config["max_epochs"]),
            smoothing=float(config.get("smoothing", 0.1)),
            pretrained_checkpoint=args.biovid_checkpoint,
            freeze_mode=str(config.get("freeze_mode", "none")),
        )

        ckpt_name = f"subj_{test_subj}" + "-{epoch}-{val_acc:.2f}"
        checkpoint_cb = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True, filename=ckpt_name)
        early_stop = EarlyStopping(monitor="val_acc", mode="max", patience=int(config["patience"]))
        logger = TensorBoardLogger("logs_syracuse_finetune", name=f"subj_{test_subj}")

        strategy = DDPStrategy(find_unused_parameters=False) if int(config.get("gpus", 1)) > 1 else None

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=int(config.get("gpus", 1)),
            max_epochs=int(config["max_epochs"]),
            accumulate_grad_batches=int(config.get("accumulate_grad_batches", 1)),
            precision=int(config.get("precision", 32)),
            strategy=strategy,
            logger=logger,
            callbacks=[checkpoint_cb, early_stop],
            log_every_n_steps=5,
        )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader, ckpt_path="best")

        # Predict clip-level (baseline-style) and persist metrics/preds/conf
        print(f"Predicting on subject {test_subj}...")
        preds_batches = trainer.predict(model, test_loader, ckpt_path="best")

        sample_ids_all: List[str] = []
        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        for batch in preds_batches or []:
            sample_ids = batch["sample_ids"]
            y_ts = batch["y_true"].detach().cpu().view(-1).tolist()
            y_ps = batch["y_pred"].detach().cpu().view(-1).tolist()
            for i in range(len(sample_ids)):
                sid = str(sample_ids[i])
                sample_ids_all.append(sid)
                y_true_all.append(int(y_ts[i]))
                y_pred_all.append(int(y_ps[i]))

        # Per-subject clip-level preds CSV (baseline name + columns)
        if save_preds:
            preds_csv = os.path.join(clip_out_dir, f"preds_multitask_subject_{test_subj}.csv")
            with open(preds_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["video_id", "y_true", "y_pred"])
                for sid, yt, yp in zip(sample_ids_all, y_true_all, y_pred_all):
                    w.writerow([sid, yt, yp])

        # Confusion (clip-level)
        conf = _confusion_from_preds(y_true_all, y_pred_all, K)
        sum_conf += conf
        if save_preds:
            conf_csv = os.path.join(clip_out_dir, f"confusion_multitask_subject_{test_subj}.csv")
            with open(conf_csv, "w", newline="") as f:
                w = csv.writer(f)
                header = ["true\\pred"] + [f"{i}" for i in range(K)]
                w.writerow(header)
                for ti in range(K):
                    row_vals = [str(ti)] + [str(int(v)) for v in conf[ti]]
                    w.writerow(row_vals)

        # Per-subject test metrics row (computed from clip-level preds, baseline fields)
        total = len(y_true_all)
        correct = int(np.sum(np.asarray(y_true_all) == np.asarray(y_pred_all))) if total > 0 else 0
        test_acc = float(correct / total) if total > 0 else 0.0
        test_mae = float(np.mean(np.abs(np.asarray(y_true_all) - np.asarray(y_pred_all)))) if total > 0 else 0.0
        test_qwk = _qwk(y_true_all, y_pred_all, K)
        f1_macro, f1_weighted, f1_micro = _f1_scores(y_true_all, y_pred_all, K)
        rows.append(
            {
                "subject": str(test_subj),
                "test_qwk": test_qwk,
                "test_acc": test_acc,
                "test_mae": test_mae,
                "test_f1_macro": f1_macro,
                "test_f1_weighted": f1_weighted,
                "test_f1_micro": f1_micro,
            }
        )

    # Global summary (only meaningful when the process ran all subjects)
    if args.no_summary or args.only_subject is not None or args.shard_id is not None:
        print("\nSkipping global summary (subset/shard run).")
        return

    _write_global_outputs()


if __name__ == "__main__":
    main()

