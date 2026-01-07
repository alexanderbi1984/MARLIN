import pandas as pd
import argparse
import sys
import os
import glob
import numpy as np

def analyze_diff_distribution(meta_df, preds_csv_path):
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

    diffs_pred = []
    diffs_true = []
    
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
                
                # Pred Diff
                d_pred = pred_lookup[vid_pre] - pred_lookup[vid_post]
                diffs_pred.append(d_pred)
                
                # True Diff
                t_pre = true_class_lookup.get(vid_pre)
                t_post = true_class_lookup.get(vid_post)
                if t_pre is None:
                    # fallback map
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
                
                if t_pre != -1 and t_post != -1:
                    d_true = t_pre - t_post
                    diffs_true.append(d_true)

    return diffs_pred, diffs_true

def main():
    # Focus on the best model for this analysis
    target_dir = "/data/Nbi/Marlin/MARLIN/Syracuse/xformer_mil_mma_clip_level_auxiliary/5class_0.75_aug_ratio/"
    preds_csv = os.path.join(target_dir, "syracuse_loocv_multitask_video_detail.csv")
    meta_path = "/data/Nbi/Syracuse/meta_with_outcomes.xlsx"
    
    print(f"Analyzing Diff Distribution for: {target_dir}")
    
    try:
        meta_df = pd.read_excel(meta_path)
        meta_df.columns = [c.strip() for c in meta_df.columns]
        if "subject_id" in meta_df.columns:
            meta_df["subject_id"] = meta_df["subject_id"].astype(str)
    except Exception as e:
        print(e)
        sys.exit(1)
        
    d_pred, d_true = analyze_diff_distribution(meta_df, preds_csv)
    
    if not d_pred:
        print("No data found.")
        sys.exit(0)
        
    d_pred = np.array(d_pred)
    d_true = np.array(d_true)
    
    print(f"\nTotal Pairs Analyzed: {len(d_pred)}")
    
    print("\n--- True Diff Distribution ---")
    print(f"Range: [{d_true.min()}, {d_true.max()}]")
    print(f"Negative (< 0): {np.sum(d_true < 0)} ({np.mean(d_true < 0)*100:.1f}%)")
    print(f"Zero (= 0):     {np.sum(d_true == 0)} ({np.mean(d_true == 0)*100:.1f}%)")
    print(f"Positive (> 0): {np.sum(d_true > 0)} ({np.mean(d_true > 0)*100:.1f}%)")
    print(f"Significant (>= 2): {np.sum(d_true >= 2)} ({np.mean(d_true >= 2)*100:.1f}%)")
    
    print("\n--- Pred Diff Distribution ---")
    print(f"Range: [{d_pred.min()}, {d_pred.max()}]")
    print(f"Negative (< 0): {np.sum(d_pred < 0)} ({np.mean(d_pred < 0)*100:.1f}%)")
    print(f"Zero (= 0):     {np.sum(d_pred == 0)} ({np.mean(d_pred == 0)*100:.1f}%)")
    print(f"Positive (> 0): {np.sum(d_pred > 0)} ({np.mean(d_pred > 0)*100:.1f}%)")
    print(f"Significant (>= 2): {np.sum(d_pred >= 2)} ({np.mean(d_pred >= 2)*100:.1f}%)")
    
    # Joint analysis
    print("\n--- Joint Analysis (Negative Preds) ---")
    neg_indices = np.where(d_pred < 0)[0]
    if len(neg_indices) > 0:
        print("Cases where Pred < 0:")
        for idx in neg_indices:
            print(f"  Pair {idx}: Pred={d_pred[idx]}, True={d_true[idx]}")
    else:
        print("No negative predictions.")

if __name__ == "__main__":
    main()





