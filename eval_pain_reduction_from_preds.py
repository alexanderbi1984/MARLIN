import pandas as pd
import argparse
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pain Reduction (Diff) from Pre-computed Predictions")
    parser.add_argument("--meta_path", type=str, default="/data/Nbi/Syracuse/meta_with_outcomes.xlsx", help="Meta Excel")
    parser.add_argument("--preds_csv", type=str, required=True, help="Path to syracuse_loocv_multitask_video_detail.csv")
    parser.add_argument("--out_csv", type=str, default="pain_reduction_diff_results.csv", help="Output results")
    args = parser.parse_args()

    # 1. Load Meta
    print(f"Loading Meta: {args.meta_path}")
    try:
        meta_df = pd.read_excel(args.meta_path)
        meta_df.columns = [c.strip() for c in meta_df.columns]
        if "subject_id" in meta_df.columns:
            meta_df["subject_id"] = meta_df["subject_id"].astype(str)
    except Exception as e:
        print(f"Error loading meta: {e}")
        sys.exit(1)

    # 2. Load Predictions
    print(f"Loading Predictions: {args.preds_csv}")
    try:
        preds_df = pd.read_csv(args.preds_csv)
        # Ensure video_id exists
        if "video_id" not in preds_df.columns:
            print("Error: preds_csv must contain 'video_id' column.")
            sys.exit(1)
        # Clean video_id just in case
        preds_df["video_id"] = preds_df["video_id"].astype(str).str.strip()
    except Exception as e:
        print(f"Error loading preds: {e}")
        sys.exit(1)

    # 3. Create a lookup for predictions: video_id -> pred_class
    # Note: pred_class is in 0..4
    pred_lookup = dict(zip(preds_df["video_id"], preds_df["pred_class"]))

    # 4. Discover Pairs and Compute Diff
    results = []
    
    # We iterate subjects in meta
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
                
                # Check if we have predictions for both
                if vid_pre not in pred_lookup:
                    print(f"Warning: No prediction for Pre video {vid_pre} (Sub {subject_id})")
                    continue
                if vid_post not in pred_lookup:
                    print(f"Warning: No prediction for Post video {vid_post} (Sub {subject_id})")
                    continue
                
                pred_pre = pred_lookup[vid_pre]
                pred_post = pred_lookup[vid_post]
                
                # Compute Diff
                diff = pred_pre - pred_post
                
                # Determine "Significant Reduction"
                # Logic: significant if (pre - post) >= 2
                is_significant = 1 if diff >= 2 else 0
                
                # Ground Truth?
                # User scenario B: We define "significant" by the CLASS labels difference
                # But we need something to validate against?
                # User query said: "significant is class_label_pre - class_label_post >= 2"
                # If we just compute this from predictions, we get the model's opinion.
                # To EVALUATE, we need the TRUE significant label.
                # Assuming we derive TRUE significant from the TRUE pain levels (which are also in Meta or preds csv).
                
                # Let's get TRUE pain class
                # Option A: From Meta 'pain_level' -> map to class -> compute diff
                # Option B: The preds CSV also has 'true_class'. Let's use that if available.
                
                true_pre = None
                true_post = None
                
                # Try to get true class from preds lookup if possible, or meta
                # preds_csv has 'true_class'
                row_pred_pre = preds_df[preds_df["video_id"] == vid_pre]
                row_pred_post = preds_df[preds_df["video_id"] == vid_post]
                
                if not row_pred_pre.empty and not row_pred_post.empty:
                    true_pre = int(row_pred_pre.iloc[0]["true_class"])
                    true_post = int(row_pred_post.iloc[0]["true_class"])
                else:
                    # Fallback to meta 'pain_level' mapping (assuming user provided cutoffs in chat history)
                    # Class 0: 0-1, Class 1: 2-3, Class 2: 4-5, Class 3: 6-7, Class 4: 8-10
                    def map_pain(p):
                        try:
                            p = float(p)
                        except:
                            return -1
                        if p <= 1.0: return 0
                        if p <= 3.0: return 1
                        if p <= 5.0: return 2
                        if p <= 7.0: return 3
                        return 4
                    
                    true_pre = map_pain(r_pre.get("pain_level"))
                    true_post = map_pain(r_post.get("pain_level"))
                
                if true_pre == -1 or true_post == -1:
                    print(f"Skipping pair {subject_id} {visit_num}: Invalid true labels.")
                    continue

                true_diff = true_pre - true_post
                true_significant = 1 if true_diff >= 2 else 0
                
                results.append({
                    "subject_id": subject_id,
                    "visit": visit_num,
                    "vid_pre": vid_pre,
                    "vid_post": vid_post,
                    "pred_pre": pred_pre,
                    "pred_post": pred_post,
                    "pred_diff": diff,
                    "pred_significant": is_significant,
                    "true_pre": true_pre,
                    "true_post": true_post,
                    "true_diff": true_diff,
                    "true_significant": true_significant
                })

    if not results:
        print("No valid pairs found.")
        sys.exit(0)
        
    df_res = pd.DataFrame(results)
    print(f"Evaluated {len(df_res)} pairs.")
    
    # Metrics
    y_true = df_res["true_significant"]
    y_pred = df_res["pred_significant"]
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    print("\n=== Pain Reduction Classification Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Save
    df_res.to_csv(args.out_csv, index=False)
    print(f"\nDetailed results saved to {args.out_csv}")

if __name__ == "__main__":
    main()





