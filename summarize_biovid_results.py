import os
import csv
import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Dict, Any, List, Optional

"""
Result Summarization Script for BioVid Pain Analysis

Purpose:
    Aggregates experiment results from 4 methods on the BioVid dataset.
    Computes summary statistics (Mean, Std, 95% CI) for subject-level metrics.

Data Sources:
    Scans the following directories under `ROOT_DIR`:
    1. AU: `biovid_coral_xformer_au_features`
    2. Marlin: `biovid_coral_xformer_marlin_rgb_features`
    3. MMA (Random): `biovid_coral_xformer_MMA_random_features`
    4. MMA (RGB): `biovid_coral_xformer_MMA_rgb_features`

    For each directory, it reads:
    - `biovid_loocv_summary.csv`: Subject-level metrics (QWK, Acc, F1, MAE).

Outputs:
    - `biovid_final_results_summary.csv`: Summary table with Mean, Std, and 95% CI.
"""

# Configuration
ROOT_DIR = "/data/Nbi/Marlin/MARLIN/BioVid/mil_runs_qwk"
RESULT_DIRS = {
    "AU_RGB": "biovid_coral_xformer_au_features",
    "Marlin_RGB": "biovid_coral_xformer_marlin_rgb_features",
    "MMA_Random": "biovid_coral_xformer_MMA_random_features",
    "MMA_RGB": "biovid_coral_xformer_MMA_rgb_features",
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

def process_experiment(name: str, rel_path: str) -> Optional[Dict[str, Any]]:
    exp_path = os.path.join(ROOT_DIR, rel_path)
    if not os.path.exists(exp_path):
        print(f"Warning: Directory not found: {exp_path}")
        return None

    # Parse Model and Feature from our internal name key (e.g., "MMA_RGB")
    parts = name.split("_")
    model_name = parts[0]
    feature_type = parts[1]

    row = {
        "Model": model_name,
        "Feature_Type": feature_type,
    }
    
    # Subject-Level Metrics
    subj_csv = os.path.join(exp_path, "biovid_loocv_summary.csv")
    if not os.path.exists(subj_csv):
        print(f"Warning: CSV not found in {exp_path}")
        return None
        
    try:
        df = pd.read_csv(subj_csv)
        # Filter out MEAN row to calculate stats ourselves
        # Check for 'subject' column case-insensitive or exact match
        if "subject" in df.columns:
            subj_col = "subject"
        else:
            # Fallback if column name is different, though sample showed 'subject'
            subj_col = df.columns[0]
            
        df = df[df[subj_col].astype(str).str.upper() != "MEAN"]
        
        # Main metrics to summarize
        # Sample columns: test_qwk,test_acc,test_mae,test_f1_macro
        metrics = ["test_qwk", "test_acc", "test_f1_macro", "test_mae"]
        
        for m in metrics:
            if m in df.columns:
                vals = df[m].dropna().values
                mean, std, ci = compute_ci(vals)
                row[f"{m}_Mean"] = mean
                row[f"{m}_Std"] = std
                row[f"{m}_CI"] = ci
            else:
                print(f"Warning: Metric {m} not found in {subj_csv}")
    except Exception as e:
        print(f"Error reading CSV {subj_csv}: {e}")
        return None

    return row

def main():
    all_rows = []
    
    print("Scanning BioVid results...")
    
    for name, rel_path in RESULT_DIRS.items():
        row = process_experiment(name, rel_path)
        if row:
            all_rows.append(row)
            print(f"Processed: {name}")

    if not all_rows:
        print("No results found.")
        return

    # Create DataFrame
    df_res = pd.DataFrame(all_rows)
    
    # Reorder columns
    # Base columns
    cols = ["Model", "Feature_Type"]
    
    # Metric columns (sorted)
    metric_cols = [c for c in df_res.columns if c not in cols]
    
    # Sort metric columns to group Mean/Std/CI together for each metric type if possible
    # Or just alphabetical
    metric_cols.sort()
    
    df_res = df_res[cols + metric_cols]
    
    # Sort rows by Model
    df_res.sort_values(by=["Model", "Feature_Type"], inplace=True)
    
    out_path = os.path.join(ROOT_DIR, "biovid_final_results_summary.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nSummary saved to: {out_path}")
    print(df_res.to_string())

if __name__ == "__main__":
    main()



