import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SummaryMetrics:
    total_videos: int
    accuracy: float
    macro_f1: float
    weighted_f1: float


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def _confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yt < num_classes and 0 <= yp < num_classes:
            cm[int(yt), int(yp)] += 1
    return cm


def _f1_from_confusion(cm: np.ndarray) -> Tuple[float, float]:
    """Return (macro_f1, weighted_f1) from confusion matrix."""
    num_classes = int(cm.shape[0])
    supports = cm.sum(axis=1).astype(float)  # true counts
    total = float(supports.sum())
    f1s: List[float] = []
    w_f1 = 0.0
    for k in range(num_classes):
        tp = float(cm[k, k])
        fp = float(cm[:, k].sum() - cm[k, k])
        fn = float(cm[k, :].sum() - cm[k, k])
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * prec * rec, prec + rec) if (prec + rec) != 0 else 0.0
        f1s.append(f1)
        w_f1 += (supports[k] / total) * f1 if total > 0 else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return macro_f1, float(w_f1)


def _read_preds_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"video_id", "y_true", "y_pred_majority"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return df


def summarize(results_dir: str, num_classes: int) -> Tuple[pd.DataFrame, SummaryMetrics, np.ndarray]:
    files = sorted(
        [
            fn
            for fn in os.listdir(results_dir)
            if fn.startswith("preds_subj_") and fn.endswith(".csv")
        ]
    )
    if not files:
        raise FileNotFoundError(f"No preds_subj_*.csv found in: {results_dir}")

    per_subject_rows: List[Dict[str, object]] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for fn in files:
        subj = fn.replace("preds_subj_", "").replace(".csv", "")
        fpath = os.path.join(results_dir, fn)
        df = _read_preds_csv(fpath)

        # Coerce labels
        yt = [int(v) for v in df["y_true"].tolist()]
        yp = [int(v) for v in df["y_pred_majority"].tolist()]

        correct = int(np.sum(np.asarray(yt) == np.asarray(yp)))
        total = int(len(yt))
        acc = _safe_div(correct, total)
        per_subject_rows.append(
            {
                "subject": str(subj),
                "num_videos": total,
                "acc": float(acc),
            }
        )

        y_true_all.extend(yt)
        y_pred_all.extend(yp)

    total_videos = int(len(y_true_all))
    acc = _safe_div(int(np.sum(np.asarray(y_true_all) == np.asarray(y_pred_all))), total_videos)
    cm = _confusion_matrix(y_true_all, y_pred_all, num_classes=num_classes)
    macro_f1, weighted_f1 = _f1_from_confusion(cm)

    summary = SummaryMetrics(
        total_videos=total_videos,
        accuracy=float(acc),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
    )

    per_subject_df = pd.DataFrame(per_subject_rows).sort_values("subject")
    mean_row = {
        "subject": "MEAN",
        "num_videos": float(per_subject_df["num_videos"].mean()) if len(per_subject_df) else 0.0,
        "acc": float(per_subject_df["acc"].mean()) if len(per_subject_df) else 0.0,
    }
    per_subject_df = pd.concat([per_subject_df, pd.DataFrame([mean_row])], ignore_index=True)
    return per_subject_df, summary, cm


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Syracuse TNT finetune video-level results (majority-vote).")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="logs_syracuse_finetune/results",
        help="Directory containing preds_subj_*.csv",
    )
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="syracuse_tnt_video",
        help="Prefix for output files (written under results_dir).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    per_subject_df, summary, cm = summarize(results_dir=results_dir, num_classes=int(args.num_classes))

    # Write per-subject summary
    summary_csv = os.path.join(results_dir, f"{args.out_prefix}_per_subject.csv")
    per_subject_df.to_csv(summary_csv, index=False)

    # Write confusion matrix
    conf_csv = os.path.join(results_dir, f"{args.out_prefix}_confusion.csv")
    with open(conf_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["true\\pred"] + [str(i) for i in range(int(args.num_classes))]
        w.writerow(header)
        for i in range(int(args.num_classes)):
            w.writerow([str(i)] + [str(int(v)) for v in cm[i].tolist()])

    # Print global summary
    print("=== Syracuse TNT Video-level Summary (Majority Vote) ===")
    print(f"Results dir: {results_dir}")
    print(f"Subjects (files): {len(per_subject_df) - 1}")
    print(f"Total videos: {summary.total_videos}")
    print(f"Accuracy: {summary.accuracy:.4f}")
    print(f"Macro F1: {summary.macro_f1:.4f}")
    print(f"Weighted F1: {summary.weighted_f1:.4f}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {conf_csv}")


if __name__ == "__main__":
    main()

