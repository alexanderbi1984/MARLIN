import os
import csv
import glob
import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Dict, Any, List, Optional

"""
Result Summarization Script for Syracuse Pain Analysis

Purpose:
    Aggregates experiment results from multiple models (Marlin, AU, MMA) across different 
    tasks (3/4/5-class) and augmentation ratios. It computes summary statistics (Mean, Std, 95% CI)
    for subject-level metrics and extracts video-level metrics.

Data Sources:
    Scans the following directories under `ROOT_DIR`:
    1. Marlin: `xformer_mil_marlin_clip_level_auxiliary` (Feature Type: RGB)
    2. AU: `xformer_mil_au_clip_level_auxiliary` (Feature Type: RGB)
    3. MMA: `xformer_mil_mma_clip_level_auxiliary` (Feature Types: RGB, Random)

    For each experiment subdirectory (e.g., `3class_0.25_aug_ratio_rgb_features`), it reads:
    - `syracuse_loocv_multitask.csv`: Subject-level LOOCV metrics (QWK, Acc, F1, MAE).
    - `syracuse_loocv_multitask_video_topk.csv`: Video-level Top-K accuracy.

Outputs:
    - `final_results_summary_with_stats.csv`: A comprehensive CSV table containing:
        - Experiment Params: Model, Feature Type, Task, Aug Ratio.
        - Subject Metrics: Mean, Std, and 95% CI for QWK, Accuracy, F1-Macro, MAE.
        - Video Metrics: Top-1, Top-2, Top-3 Accuracy.

Usage:
    Run with the environment's python:
    `/data/Nbi/marlin/bin/python summarize_results.py`
"""

# Configuration
ROOT_DIR = "/data/Nbi/Marlin/MARLIN/Syracuse"
RESULT_DIRS = {
    "Marlin": "xformer_mil_marlin_clip_level_auxiliary",
    "AU": "xformer_mil_au_clip_level_auxiliary",
    "MMA": "xformer_mil_mma_clip_level_auxiliary",
}

def compute_ci(data: List[float], confidence: float = 0.95):
    """Compute mean, std, and 95% CI for a list of values."""
    if len(data) == 0:
        return None, None, None
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    if n <= 1:
        return m, 0.0, 0.0
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, np.std(a, ddof=1), h

def parse_dir_name(dirname: str, model_type: str) -> Optional[Dict[str, str]]:
    """Parse experiment parameters from directory name."""
    # Pattern: {N}class_{RATIO}_aug_ratio_{TYPE}_features
    
    if "no_exclude" in dirname:
        return None
        
    try:
        parts = dirname.split("_")
        
        # 1. Parse Task (e.g., 3class)
        task = parts[0]
        if not task.endswith("class"):
            return None
            
        # 2. Parse Aug Ratio
        # Usually parts[1] is ratio, parts[2] is 'aug', parts[3] is 'ratio'
        # e.g., 0.25_aug_ratio
        aug_ratio = parts[1]
        
        # 3. Parse Feature Type
        # For Marlin and AU, force "RGB"
        # For MMA, extract from name (rgb or random)
        if model_type in ["Marlin", "AU"]:
            feature_type = "RGB"
        else:
            # MMA: look for 'rgb' or 'random' in the name
            if "random" in dirname:
                feature_type = "Random"
            elif "rgb" in dirname:
                feature_type = "RGB"
            else:
                return None # Skip other types like gdt
                
        return {
            "Task": task,
            "Aug_Ratio": aug_ratio,
            "Feature_Type": feature_type
        }
    except Exception:
        return None

def process_experiment(model: str, exp_path: str, params: Dict[str, str]) -> Optional[Dict[str, Any]]:
    row = {
        "Model": model,
        "Task": params["Task"],
        "Aug_Ratio": params["Aug_Ratio"],
        "Feature_Type": params["Feature_Type"]
    }
    
    # 1. Subject-Level Metrics
    subj_csv = os.path.join(exp_path, "syracuse_loocv_multitask.csv")
    if not os.path.exists(subj_csv):
        return None
        
    try:
        df = pd.read_csv(subj_csv)
        # Filter out MEAN row to calculate stats ourselves
        df = df[df["subject"].astype(str).str.upper() != "MEAN"]
        
        # Main metrics to summarize
        metrics = ["test_qwk", "test_acc", "test_f1_macro", "test_mae"]
        for m in metrics:
            if m in df.columns:
                vals = df[m].dropna().values
                mean, std, ci = compute_ci(vals)
                row[f"Subj_{m}_Mean"] = mean
                row[f"Subj_{m}_Std"] = std
                row[f"Subj_{m}_CI_HalfWidth"] = ci  # Rename to be explicit
    except Exception as e:
        print(f"Error reading subject CSV {subj_csv}: {e}")
        return None

    # 2. Video-Level Metrics
    video_csv = os.path.join(exp_path, "syracuse_loocv_multitask_video_topk.csv")
    if os.path.exists(video_csv):
        try:
            vdf = pd.read_csv(video_csv)
            # Expecting rows: top1_acc, top2_acc, top3_acc
            for _, r in vdf.iterrows():
                metric = r["metric"]
                val = r["value"]
                if metric == "top1_acc":
                    row["Video_Top1_Acc"] = float(val)
                elif metric == "top2_acc":
                    row["Video_Top2_Acc"] = float(val)
                elif metric == "top3_acc":
                    row["Video_Top3_Acc"] = float(val)
        except Exception as e:
            print(f"Error reading video CSV {video_csv}: {e}")
    else:
        row["Video_Top1_Acc"] = None
        row["Video_Top2_Acc"] = None
        row["Video_Top3_Acc"] = None
        
    return row

def main():
    all_rows = []
    
    print("Scanning results...")
    
    for model, rel_path in RESULT_DIRS.items():
        full_path = os.path.join(ROOT_DIR, rel_path)
        if not os.path.isdir(full_path):
            print(f"Warning: Directory not found: {full_path}")
            continue
            
        subdirs = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
        subdirs.sort()
        
        for d in subdirs:
            params = parse_dir_name(d, model)
            if not params:
                continue
                
            exp_path = os.path.join(full_path, d)
            row = process_experiment(model, exp_path, params)
            if row:
                all_rows.append(row)
                print(f"Processed: {model} {params['Feature_Type']} {params['Task']} Ratio={params['Aug_Ratio']}")

    if not all_rows:
        print("No results found.")
        return

    # Create DataFrame and Sort
    df_res = pd.DataFrame(all_rows)
    
    # Reorder columns
    cols = ["Model", "Feature_Type", "Task", "Aug_Ratio"]
    metric_cols = [c for c in df_res.columns if c not in cols]
    metric_cols.sort()
    # Put video top1 first among metrics for readability
    if "Video_Top1_Acc" in metric_cols:
        metric_cols.remove("Video_Top1_Acc")
        metric_cols = ["Video_Top1_Acc"] + metric_cols
        
    df_res = df_res[cols + metric_cols]
    
    # Sort rows
    df_res.sort_values(by=["Model", "Feature_Type", "Task", "Aug_Ratio"], inplace=True)
    
    out_path = os.path.join(ROOT_DIR, "final_results_summary_with_stats.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nSummary saved to: {out_path}")
    print(f"Total experiments processed: {len(df_res)}")

if __name__ == "__main__":
    main()

