import argparse
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

"""
Pairwise Fold-Level Comparison Script

Purpose:
    Performs a rigorous statistical comparison between two experimental setups (e.g., Baseline vs. Proposed Method)
    based on their Leave-One-Subject-Out Cross-Validation (LOOCV) results. Instead of just comparing average metrics,
    this script analyzes the paired differences for each subject (fold).

Key Features:
    1.  **Data Loading**: Reads `syracuse_loocv_multitask.csv` from two provided result directories.
    2.  **Subject Alignment**: Ensures comparison is performed on the intersection of subjects present in both experiments.
    3.  **Statistical Testing**:
        *   **Wilcoxon Signed-Rank Test**: Non-parametric test for paired samples (Recommended).
        *   **Paired t-test**: Parametric test (Reference).
    4.  **Win/Tie/Loss Analysis**: Counts how many subjects were improved, unchanged, or degraded.
    5.  **Visualization**: Generates a scatter plot (Performance A vs B) and a bar plot (Difference per Subject).

Usage:
    /data/Nbi/marlin/bin/python analyze_pairwise_folds.py \
      --dir_a /path/to/baseline_dir \
      --dir_b /path/to/method_dir \
      --name_a "Baseline" --name_b "Ours" \
      --metric test_qwk

Outputs:
    Saved to `Syracuse/pairwise_comparisons/` by default:
    *   `pairwise_{A}_vs_{B}_{metric}.csv`: Raw data table.
    *   `plot_{A}_vs_{B}_{metric}.png`: Visualizations.
"""

def load_loocv_results(result_dir):
    """Load syracuse_loocv_multitask.csv and return a DataFrame indexed by subject."""
    csv_path = os.path.join(result_dir, "syracuse_loocv_multitask.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Result CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Filter out MEAN row
    df = df[df["subject"].astype(str).str.upper() != "MEAN"].copy()
    df["subject"] = df["subject"].astype(str)
    return df.set_index("subject")

def main():
    parser = argparse.ArgumentParser(description="Pairwise Subject-Level Comparison for LOOCV Results")
    parser.add_argument("--dir_a", type=str, required=True, help="Path to Baseline experiment directory")
    parser.add_argument("--dir_b", type=str, required=True, help="Path to Proposed experiment directory")
    parser.add_argument("--name_a", type=str, default="Baseline", help="Display name for A")
    parser.add_argument("--name_b", type=str, default="Method", help="Display name for B")
    parser.add_argument("--metric", type=str, default="test_qwk", help="Metric to compare (e.g. test_qwk, test_acc)")
    parser.add_argument("--baseline_val", type=float, default=None, help="Threshold for random/baseline performance (e.g. 0.2 for 5-class Acc, 0.0 for QWK). If set, excludes 'Double Fail' cases.")
    # Default output directory fixed to Syracuse/pairwise_comparisons
    parser.add_argument("--output_dir", type=str, default="Syracuse/pairwise_comparisons", help="Directory to save plots and stats")
    
    args = parser.parse_args()
    
    # 1. Load Data
    try:
        df_a = load_loocv_results(args.dir_a)
        df_b = load_loocv_results(args.dir_b)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Align Subjects
    common_subjects = sorted(list(set(df_a.index) & set(df_b.index)))
    if not common_subjects:
        print("Error: No common subjects found between the two experiments.")
        return
        
    print(f"Comparing {len(common_subjects)} common subjects.")
    
    # Extract Metric
    metric = args.metric
    if metric not in df_a.columns or metric not in df_b.columns:
        print(f"Error: Metric '{metric}' not found in CSVs. Available: {list(df_a.columns)}")
        return
        
    vals_a = df_a.loc[common_subjects, metric].values
    vals_b = df_b.loc[common_subjects, metric].values
    
    # --- Effective Comparison Logic ---
    if args.baseline_val is not None:
        thr = args.baseline_val
        # Identify Double Fails: Both A and B are worse than or equal to random/baseline
        is_double_fail = (vals_a <= thr) & (vals_b <= thr)
        num_double_fail = np.sum(is_double_fail)
        
        # Filter for effective pairs
        effective_mask = ~is_double_fail
        vals_a_eff = vals_a[effective_mask]
        vals_b_eff = vals_b[effective_mask]
        common_subjects_eff = np.array(common_subjects)[effective_mask]
        
        print(f"\n[Filter] Applied Baseline Threshold: {thr}")
        print(f"Total Subjects: {len(common_subjects)}")
        print(f"Double Fails (Both <= {thr}): {num_double_fail}")
        print(f"Effective Comparisons: {len(vals_a_eff)}")
        
        # Update variables for statistics
        vals_a_stat = vals_a_eff
        vals_b_stat = vals_b_eff
        diffs_stat = vals_b_stat - vals_a_stat
        subjects_stat = common_subjects_eff
    else:
        vals_a_stat = vals_a
        vals_b_stat = vals_b
        diffs_stat = vals_b - vals_a
        subjects_stat = np.array(common_subjects)
        num_double_fail = 0

    # 3. Calculate Differences (for all, but stats on effective)
    diffs_all = vals_b - vals_a
    
    # 4. Statistics (on Effective Pairs only)
    wins = np.sum(diffs_stat > 0)
    ties = np.sum(diffs_stat == 0)
    losses = np.sum(diffs_stat < 0)
    
    mean_a = np.mean(vals_a) # Keep global mean for reference
    mean_b = np.mean(vals_b)
    mean_diff = np.mean(diffs_all)
    
    # Mean on effective set
    mean_a_eff = np.mean(vals_a_stat) if len(vals_a_stat) > 0 else 0.0
    mean_b_eff = np.mean(vals_b_stat) if len(vals_b_stat) > 0 else 0.0
    
    # Statistical Tests
    # Wilcoxon Signed-Rank Test (Non-parametric)
    if len(diffs_stat) > 0:
        try:
            w_stat, w_p = stats.wilcoxon(vals_a_stat, vals_b_stat)
        except Exception:
            w_p = 1.0 
        t_stat, t_p = stats.ttest_rel(vals_a_stat, vals_b_stat)
    else:
        w_p, t_p = 1.0, 1.0
    
    print("\n" + "="*50)
    print(f"Comparison: {args.name_a} vs {args.name_b}")
    print(f"Metric: {metric}")
    print("="*50)
    print(f"Global Mean {args.name_a}: {mean_a:.4f}")
    print(f"Global Mean {args.name_b}: {mean_b:.4f}")
    print("-" * 30)
    if args.baseline_val is not None:
        print(f"Effective Mean {args.name_a}: {mean_a_eff:.4f}")
        print(f"Effective Mean {args.name_b}: {mean_b_eff:.4f}")
        print(f"Double Fails (Excluded): {num_double_fail}")
    print("-" * 30)
    print(f"Win / Tie / Loss (Effective): {wins} / {ties} / {losses}")
    print("-" * 30)
    print(f"Wilcoxon p-value: {w_p:.5f} {'*' if w_p < 0.05 else ''}")
    print(f"Paired t-test p:  {t_p:.5f} {'*' if t_p < 0.05 else ''}")
    print("="*50)
    
    # 5. Save Detailed CSV
    out_df = pd.DataFrame({
        "Subject": common_subjects,
        f"{args.name_a}": vals_a,
        f"{args.name_b}": vals_b,
        "Diff": diffs_all,
        "Is_Double_Fail": [(x <= (args.baseline_val if args.baseline_val else -999) and y <= (args.baseline_val if args.baseline_val else -999)) for x, y in zip(vals_a, vals_b)]
    })
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"pairwise_{args.name_a}_vs_{args.name_b}_{metric}.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nDetailed pairwise data saved to: {csv_path}")
    
    # 6. Visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Scatter with Diagonal
    plt.subplot(1, 2, 1)
    min_val = min(min(vals_a), min(vals_b))
    max_val = max(max(vals_a), max(vals_b))
    # Handle single point or zero range
    if max_val == min_val:
        margin = 0.1
    else:
        margin = (max_val - min_val) * 0.05
    
    plt.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'k--', alpha=0.5, label="Tie")
    
    if args.baseline_val is not None:
        # Plot Double Fails in gray
        fail_mask = np.array(out_df["Is_Double_Fail"])
        plt.scatter(vals_a[fail_mask], vals_b[fail_mask], alpha=0.3, color='gray', label="Double Fail")
        plt.scatter(vals_a[~fail_mask], vals_b[~fail_mask], alpha=0.7, color='blue', edgecolor='k', label="Effective")
        plt.axhline(args.baseline_val, color='r', linestyle=':', alpha=0.5)
        plt.axvline(args.baseline_val, color='r', linestyle=':', alpha=0.5)
    else:
        plt.scatter(vals_a, vals_b, alpha=0.7, color='blue', edgecolor='k')
    
    plt.xlabel(f"{args.name_a} {metric}")
    plt.ylabel(f"{args.name_b} {metric}")
    plt.title(f"Pairwise Performance ({metric})")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Plot 2: Bar Plot of Differences
    plt.subplot(1, 2, 2)
    # Sort by diff for better visualization
    sorted_indices = np.argsort(diffs_all)
    sorted_diffs = diffs_all[sorted_indices]
    sorted_subjs = np.array(common_subjects)[sorted_indices]
    
    colors = ['green' if x > 0 else 'red' for x in sorted_diffs]
    plt.bar(range(len(sorted_diffs)), sorted_diffs, color=colors, alpha=0.7)
    plt.axhline(0, color='k', linewidth=0.8)
    plt.xlabel("Subjects (Sorted by Improvement)")
    plt.ylabel(f"Difference ({args.name_b} - {args.name_a})")
    plt.title(f"Improvement per Subject\n(Mean Diff: {mean_diff:.3f})")
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"plot_{args.name_a}_vs_{args.name_b}_{metric}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Visualization saved to: {plot_path}")

if __name__ == "__main__":
    main()

