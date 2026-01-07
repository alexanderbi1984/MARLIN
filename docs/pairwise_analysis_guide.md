# Pairwise Fold-Level Analysis Guide

This guide explains how to use the `analyze_pairwise_folds.py` script to statistically validate the superiority of your proposed method over baselines using Leave-One-Subject-Out Cross-Validation (LOOCV) results.

## 1. Why Pairwise Comparison?

Standard evaluation often compares the **Average** accuracy across all subjects. While useful, averages can be misleading:
- A method might have a slightly higher average just because it performed exceptionally well on 1-2 easy subjects, while failing on many others.
- Averages ignore the **paired** nature of LOOCV: we test Method A and Method B on the **exact same subjects**.

**Pairwise Analysis** focuses on the difference $\Delta_i = Score_{Method, i} - Score_{Baseline, i}$ for each subject $i$. 
- If $\Delta_i > 0$ for most subjects, your method is robustly better.
- Statistical tests (like Wilcoxon) quantify the probability that this improvement happened by chance.

## 2. Usage

### Basic Command
```bash
/data/Nbi/marlin/bin/python analyze_pairwise_folds.py \
  --dir_a <PATH_TO_BASELINE_RESULT_DIR> \
  --dir_b <PATH_TO_YOUR_METHOD_RESULT_DIR> \
  --name_a "AU Baseline" \
  --name_b "MMA (Ours)" \
  --metric test_qwk
```

### Parameters
*   `--dir_a`: Path to the directory containing the baseline results (must contain `syracuse_loocv_multitask.csv`).
*   `--dir_b`: Path to your proposed method's results.
*   `--name_a`, `--name_b`: Readable names for the plot legends (e.g., "Marlin", "AU+PSPI", "MMA").
*   `--metric`: The metric to compare. Options: `test_qwk` (default), `test_acc`, `test_f1_macro`, `test_mae`.
*   `--output_dir`: Where to save results. Default: `Syracuse/pairwise_comparisons/`.

## 3. Interpreting Output

### Console Output
The script prints a statistical summary:
```text
Comparison: AU Baseline vs MMA (Ours)
Metric: test_qwk
==================================================
Mean AU Baseline: 0.4520
Mean MMA (Ours): 0.5100
Mean Diff (MMA (Ours) - AU Baseline): 0.0580
------------------------------
Win / Tie / Loss: 25 / 2 / 5
------------------------------
Wilcoxon p-value: 0.00123 *
Paired t-test p:  0.00456 *
==================================================
```
*   **Win/Tie/Loss**: Out of 32 subjects, MMA won 25 times. This is strong evidence of robustness.
*   **Wilcoxon p-value**: If $< 0.05$, the improvement is statistically significant. Use this for paper reporting.

### Visualizations
The script generates a PNG file (e.g., `plot_AU_vs_MMA_test_qwk.png`) with two subplots:

1.  **Scatter Plot**:
    *   X-axis: Baseline Score
    *   Y-axis: Method Score
    *   **Interpretation**: Points **above the diagonal line** indicate subjects where your method outperformed the baseline.

2.  **Difference Bar Plot**:
    *   Shows the improvement ($\Delta$) for each subject, sorted from negative to positive.
    *   **Interpretation**: A preponderance of **green bars** (positive diff) over red bars indicates consistent superiority.

## 4. Examples paths

*   **Marlin Baseline**: `Syracuse/xformer_mil_marlin_clip_level_auxiliary/5class_0.0_aug_ratio`
*   **AU Baseline**: `Syracuse/xformer_mil_au_clip_level_auxiliary/5class_0.0_aug_ratio_au_features`
*   **MMA (Ours)**: `Syracuse/xformer_mil_mma_clip_level_auxiliary/5class_0.0_aug_ratio_rgb_features`

## 5. Statistical Note
We prefer the **Wilcoxon Signed-Rank Test** over the Paired t-test because LOOCV scores (like Accuracy or QWK) are bounded [0,1] or [-1,1] and often not normally distributed. Wilcoxon is a non-parametric test that is robust to outliers and distributional assumptions.










