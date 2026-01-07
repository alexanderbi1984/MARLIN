import pandas as pd
import argparse
import sys
import os
import glob
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_preds(meta_df, preds_csv_path):
    try:
        preds_df = pd.read_csv(preds_csv_path)
    except Exception as e:
        print(f"Error loading preds from {preds_csv_path}: {e}")
        return None

    if "video_id" not in preds_df.columns:
        print(f"Skipping {preds_csv_path}: 'video_id' column missing.")
        return None

    preds_df["video_id"] = preds_df["video_id"].astype(str).str.strip()
    # Need pred_class. If not present, try to infer or skip
    if "pred_class" not in preds_df.columns:
         print(f"Skipping {preds_csv_path}: 'pred_class' column missing.")
         return None

    pred_lookup = dict(zip(preds_df["video_id"], preds_df["pred_class"]))
    
    # We might need true_class from preds as well
    true_class_lookup = {}
    if "true_class" in preds_df.columns:
        true_class_lookup = dict(zip(preds_df["video_id"], preds_df["true_class"]))

    results = []
    
    # Iterate pairs from meta
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
                    # Missing prediction for one of the pair
                    continue
                
                pred_pre = pred_lookup[vid_pre]
                pred_post = pred_lookup[vid_post]
                diff = pred_pre - pred_post
                pred_significant = 1 if diff >= 2 else 0
                
                # Determine True Label
                true_pre = true_class_lookup.get(vid_pre)
                true_post = true_class_lookup.get(vid_post)
                
                if true_pre is None:
                    # Fallback to meta mapping
                    def map_pain(p):
                        try: p = float(p)
                        except: return -1
                        if p <= 1.0: return 0
                        if p <= 3.0: return 1
                        if p <= 5.0: return 2
                        if p <= 7.0: return 3
                        return 4
                    true_pre = map_pain(r_pre.get("pain_level"))
                
                if true_post is None:
                     def map_pain(p):
                        try: p = float(p)
                        except: return -1
                        if p <= 1.0: return 0
                        if p <= 3.0: return 1
                        if p <= 5.0: return 2
                        if p <= 7.0: return 3
                        return 4
                     true_post = map_pain(r_post.get("pain_level"))

                if true_pre == -1 or true_post == -1:
                    continue

                true_diff = true_pre - true_post
                true_significant = 1 if true_diff >= 2 else 0
                
                results.append({
                    "true": true_significant,
                    "pred": pred_significant
                })
                
    if not results:
        return None
        
    df_res = pd.DataFrame(results)
    y_true = df_res["true"]
    y_pred = df_res["pred"]
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        "acc": acc,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n": len(results)
    }

def main():
    root_dir = "/data/Nbi/Marlin/MARLIN/Syracuse/xformer_mil_marlin_clip_level_auxiliary/"
    meta_path = "/data/Nbi/Syracuse/meta_with_outcomes.xlsx"
    out_csv = "batch_pain_reduction_results_marlin.csv"
    
    print(f"Loading Meta: {meta_path}")
    try:
        meta_df = pd.read_excel(meta_path)
        meta_df.columns = [c.strip() for c in meta_df.columns]
        if "subject_id" in meta_df.columns:
            meta_df["subject_id"] = meta_df["subject_id"].astype(str)
    except Exception as e:
        print(f"Error loading meta: {e}")
        sys.exit(1)
        
    # Find all subdirectories starting with 5class
    subdirs = sorted(glob.glob(os.path.join(root_dir, "5class*")))
    
    print(f"Found {len(subdirs)} '5class' directories to process.")
    
    summary_rows = []
    
    for d in subdirs:
        dirname = os.path.basename(d)
        preds_path = os.path.join(d, "syracuse_loocv_multitask_video_detail.csv")
        
        if not os.path.isfile(preds_path):
            print(f"Skipping {dirname}: predictions file not found.")
            continue
            
        print(f"Processing {dirname} ...", end=" ")
        metrics = evaluate_preds(meta_df, preds_path)
        
        if metrics:
            print(f"Acc={metrics['acc']:.4f}, F1={metrics['f1']:.4f}, N={metrics['n']}")
            row = {"experiment": dirname, **metrics}
            summary_rows.append(row)
        else:
            print("Failed (No valid pairs or error).")
            
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(out_csv, index=False)
        print(f"\nAll Done. Results saved to {out_csv}")
        print("\nSummary Table:")
        print(df_summary.to_string(index=False))
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()





