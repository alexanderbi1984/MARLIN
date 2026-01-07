import argparse
import os
import csv
from typing import List, Dict, Any, Optional

import pytorch_lightning as pl

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from evaluate_syracuse_mil_loocv import (
    load_config as load_cfg,
    discover_subjects,
    run_fold,
)


DEFAULT_COMBOS = [
    "RGB", "RGD", "RGT", "RBD", "RBT", "RDT", "GBD", "GBT", "GDT", "BDT",
]


def main():
    parser = argparse.ArgumentParser(description="Run LOOCV over all combos and save to Syracuse/xformer_mil_loocv")
    parser.add_argument("--config", type=str, default="config/syracuse_mil_coral_xformer.yaml")
    parser.add_argument("--combos", type=str, nargs="*", default=None, help="Override combos list")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_subjects", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="Syracuse/xformer_mil_loocv")
    parser.add_argument("--resume", action="store_true", help="Skip folds already computed (detect by saved preds/confusion or existing combo CSV)")
    parser.add_argument("--accelerator", type=str, default=None, help="Override accelerator, e.g. 'gpu' or 'cpu'")
    parser.add_argument("--devices", type=int, default=None, help="Override number of devices, e.g. 1 for single GPU")
    parser.add_argument("--exclude_videos", type=str, nargs="*", default=None, help="Video base IDs to exclude from all combos.")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    combos: List[str] = args.combos or cfg.get("combos", DEFAULT_COMBOS)
    feature_root = cfg.get("syracuse_feature_root")
    meta_excel = cfg.get("meta_excel_path")
    task = str(cfg.get("task", "ordinal"))
    num_classes = int(cfg.get("num_classes", 5))

    os.makedirs(args.save_dir, exist_ok=True)
    # Persist a copy of the used config and script metadata
    eff = dict(cfg)
    eff.update({
        "resolved": {
            "script": "evaluate_syracuse_mil_loocv_all.py",
            "combos": combos,
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

    summary_rows: List[Dict[str, Any]] = []
    summary_fields = [
        "combo",
        "mean_test_qwk",
        "mean_test_acc",
        "mean_test_mae",
        "mean_test_f1_macro",
        "mean_test_f1_weighted",
        "mean_test_f1_micro",
        "n_subjects",
    ]

    # Runtime overrides propagate via cfg dict
    if args.accelerator is not None:
        cfg["accelerator"] = args.accelerator
    if args.devices is not None:
        cfg["devices"] = int(args.devices)
    exclude_videos = args.exclude_videos if args.exclude_videos is not None else cfg.get("exclude_video_ids")
    if exclude_videos is not None:
        cfg["exclude_video_ids"] = exclude_videos

    for combo in combos:
        print(f"\n=== Running LOOCV for combo: {combo} ===")
        subjects = discover_subjects(feature_root, combo, meta_excel, task, num_classes, exclude_videos)
        if args.limit_subjects is not None:
            subjects = subjects[: int(args.limit_subjects)]
        print(f"Subjects discovered: {len(subjects)}")
        if not subjects:
            print(f"No subjects found for combo {combo}; skipping.")
            continue

        rows: List[Dict[str, Any]] = []
        fields = [
            "subject",
            "test_qwk",
            "test_acc",
            "test_mae",
            "test_f1_macro",
            "test_f1_weighted",
            "test_f1_micro",
        ]

        # accumulate confusion matrix across folds for this combo
        import numpy as np
        K = int(num_classes)
        sum_conf = np.zeros((K, K), dtype=int)

        # Resume support: load existing per-combo CSV to skip
        out_csv = os.path.join(args.save_dir, f"syracuse_loocv_{combo}.csv")
        existing_by_subject: Dict[str, Dict[str, Any]] = {}
        if args.resume and os.path.isfile(out_csv):
            with open(out_csv, "r") as f:
                rd = csv.DictReader(f)
                for row in rd:
                    sid = str(row.get("subject", "")).strip()
                    if not sid or sid.upper() == "MEAN":
                        continue
                    existing_by_subject[sid] = row
        if existing_by_subject:
            print(f"Resume: found {len(existing_by_subject)} completed subjects for {combo}.")
            # Pre-populate rows and confusion from existing saved fold files
            fold_save_dir = os.path.join(args.save_dir, f"{combo}")
            for sid, erow in existing_by_subject.items():
                rows.append({**erow})
                conf_csv = os.path.join(fold_save_dir, f"confusion_{combo}_subject_{sid}.csv")
                if os.path.isfile(conf_csv):
                    # read confusion
                    mat = []
                    with open(conf_csv, "r") as f:
                        rd = csv.reader(f)
                        _ = next(rd, None)
                        for r in rd:
                            if len(r) >= K + 1:
                                vals = [int(float(v)) for v in r[1:K+1]]
                                mat.append(vals)
                    arr = np.array(mat, dtype=int)
                    if arr.shape == (K, K):
                        sum_conf += arr
                else:
                    # fallback: derive from preds
                    preds_csv = os.path.join(fold_save_dir, f"preds_{combo}_subject_{sid}.csv")
                    if os.path.isfile(preds_csv):
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

        # Determine which subjects need to run or can be skipped via fold files when resuming
        to_run: List[str] = []
        for sid in subjects:
            if args.resume and sid in existing_by_subject:
                continue
            if args.resume:
                fold_save_dir = os.path.join(args.save_dir, f"{combo}")
                preds_csv = os.path.join(fold_save_dir, f"preds_{combo}_subject_{sid}.csv")
                if os.path.isfile(preds_csv):
                    # derive metrics and confusion; append
                    # compute metrics from preds
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
                                y_true.append(yt)
                                y_pred.append(yp)
                            except Exception:
                                continue
                    if y_true:
                        from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, mean_absolute_error, confusion_matrix
                        import numpy as np
                        yt = np.array(y_true, dtype=int)
                        yp = np.array(y_pred, dtype=int)
                        qwk = cohen_kappa_score(yt, yp, weights="quadratic")
                        acc = accuracy_score(yt, yp)
                        mae = mean_absolute_error(yt, yp)
                        f1_macro = f1_score(yt, yp, average="macro", labels=list(range(K)), zero_division=0)
                        f1_weighted = f1_score(yt, yp, average="weighted", labels=list(range(K)), zero_division=0)
                        f1_micro = f1_score(yt, yp, average="micro", labels=list(range(K)), zero_division=0)
                        rows.append({
                            "subject": sid,
                            "test_qwk": float(qwk),
                            "test_acc": float(acc),
                            "test_mae": float(mae),
                            "test_f1_macro": float(f1_macro),
                            "test_f1_weighted": float(f1_weighted),
                            "test_f1_micro": float(f1_micro),
                        })
                        cm = confusion_matrix(yt, yp, labels=list(range(K)))
                        sum_conf += cm.astype(int)
                        print(f"Skipping existing subject {sid}")
                        continue
            to_run.append(sid)

        for i, sid in enumerate(to_run, 1):
            print(f"Fold {i}/{len(to_run)} — test_subject={sid}")
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
                if arr.shape == sum_conf.shape:
                    sum_conf += arr

        out_csv = os.path.join(args.save_dir, f"syracuse_loocv_{combo}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
            # Summary row per combo
            if rows:
                summary: Dict[str, Any] = {"subject": "MEAN"}
                for k in fields[1:]:
                    vals = [float(r[k]) for r in rows if r.get(k) is not None]
                    summary[k] = sum(vals) / len(vals) if vals else None
                w.writerow(summary)

        # Add to global summary
        srow: Dict[str, Any] = {"combo": combo, "n_subjects": len(subjects)}
        for k in fields[1:]:
            vals = [float(r[k]) for r in rows if r.get(k) is not None]
            srow[f"mean_{k}"] = sum(vals) / len(vals) if vals else None
        summary_rows.append(srow)
        print(f"Saved {out_csv}")

        # Save combined confusion matrix for this combo
        conf_csv = os.path.join(args.save_dir, f"confusion_summary_{combo}.csv")
        with open(conf_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["true\\pred"] + [f"{i}" for i in range(K)]
            w.writerow(header)
            for i in range(K):
                row = [str(i)] + [str(int(v)) for v in sum_conf[i]]
                w.writerow(row)
        print(f"Saved {conf_csv}")

    # Write global summary across combos
    summary_csv = os.path.join(args.save_dir, "syracuse_loocv_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\nAll-combo LOOCV summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
