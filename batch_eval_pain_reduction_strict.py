import pandas as pd
import argparse
import sys
import os
import glob
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_preds_strict(meta_df, preds_csv_path):
    try:
        preds_df = pd.read_csv(preds_csv_path)
    except Exception:
        return None

    if "video_id" not in preds_df.columns or "pred_class" not in preds_df.columns:
        return None

    preds_df["video_id"] = preds_df["video_id"].astype(str).str.strip()
    pred_lookup = dict(zip(preds_df["video_id"], preds_df["pred_class"]))
    
    true_class_lookup = {}
    if "true_class" in preds_df.columns:
        true_class_lookup = dict(zip(preds_df["video_id"], preds_df["true_class"]))

    results = []
    
    for subject_id, group in meta_df.groupby("subject_id"):
        for visit_num in ["1st", "2nd"]:
            pre_tag = f"{visit_num}-pre"
            post_tag = f"{visit_num}-post"
            
            row_pre = group[group["visit_type"] == pre_tag]
            row_post = group[group["visit_type"] == post_tag]
            
            if len(row_pre) == 1 and len(row_post) == 1:
                r_pre = row_pre.iloc[0]
                r_post = row_post.iloc[0]
                
                vid_pre = str(r_pre["file_name"]).split(".")[0]
                vid_post = str(r_post["file_name"]).split(".")[0]
                
                if vid_pre not in pred_lookup or vid_post not in pred_lookup:
                    continue
                
                pred_pre = pred_lookup[vid_pre]
                pred_post = pred_lookup[vid_post]
                diff = pred_pre - pred_post
                
                # Original Classification Logic
                pred_label = 1 if diff >= 2 else 0
                
                # Determine True Label
                t_pre = true_class_lookup.get(vid_pre)
                t_post = true_class_lookup.get(vid_post)
                if t_pre is None:
                     def map_pain(p):
                        try: p = float(p)
                        except: return -1
                        if p <= 1.0: return 0
                        if p <= 3.0: return 1
                        if p <= 5.0: return 2
                        if p <= 7.0: return 3
                        return 4
                     t_pre = map_pain(r_pre.get("pain_level"))
                if t_post is None:
                     def map_pain(p):
                        try: p = float(p)
                        except: return -1
                        if p <= 1.0: return 0
                        if p <= 3.0: return 1
                        if p <= 5.0: return 2
                        if p <= 7.0: return 3
                        return 4
                     t_post = map_pain(r_post.get("pain_level"))

                if t_pre == -1 or t_post == -1:
                    continue

                true_diff = t_pre - t_post
                true_label = 1 if true_diff >= 2 else 0
                
                # Strict Evaluation:
                # If diff < 0, we consider it WRONG regardless of true_label.
                # Otherwise, check pred_label == true_label.
                is_correct = False
                if diff < 0:
                    is_correct = False
                else:
                    is_correct = (pred_label == true_label)
                
                results.append({
                    "true": true_label,
                    "pred": pred_label,
                    "diff": diff,
                    "correct": is_correct
                })
                
    if not results:
        return None
        
    df_res = pd.DataFrame(results)
    
    # Original Metrics
    y_true = df_res["true"]
    y_pred = df_res["pred"]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Strict Accuracy
    n_correct = df_res["correct"].sum()
    n_total = len(df_res)
    acc_strict = n_correct / n_total if n_total > 0 else 0.0
    
    # Count negative diffs
    n_neg_diff = (df_res["diff"] < 0).sum()
    
    return {
        "acc_orig": acc,
        "f1_orig": f1,
        "acc_strict": acc_strict,
        "n_neg_diff": n_neg_diff,
        "n": n_total
    }

def main():
    root_dirs = [
        "/data/Nbi/Marlin/MARLIN/Syracuse/xformer_mil_mma_clip_level_auxiliary/",
    ]
    meta_path = "/data/Nbi/Syracuse/meta_with_outcomes.xlsx"
    out_csv = "batch_pain_reduction_strict_mma.csv"
    
    try:
        meta_df = pd.read_excel(meta_path)
        meta_df.columns = [c.strip() for c in meta_df.columns]
        if "subject_id" in meta_df.columns:
            meta_df["subject_id"] = meta_df["subject_id"].astype(str)
    except Exception:
        sys.exit(1)
        
    summary_rows = []
    
    for root in root_dirs:
        subdirs = sorted(glob.glob(os.path.join(root, "5class*")))
        for d in subdirs:
            dirname = os.path.basename(d)
            preds_path = os.path.join(d, "syracuse_loocv_multitask_video_detail.csv")
            
            if not os.path.isfile(preds_path):
                continue
                
            print(f"Processing {dirname} ...", end=" ")
            metrics = evaluate_preds_strict(meta_df, preds_path)
            
            if metrics:
                print(f"Acc={metrics['acc_orig']:.3f}, Strict={metrics['acc_strict']:.3f}, NegDiffs={metrics['n_neg_diff']}")
                row = {"experiment": dirname, **metrics}
                summary_rows.append(row)
            else:
                print("Failed.")
            
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(out_csv, index=False)
        print(f"\nSaved to {out_csv}")
        print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()





