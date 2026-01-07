# Syracuse Pain Reduction Downstream Task Analysis

## 1. Task Definition
**Objective**: Predict whether a patient experiences a "Significant Reduction" in pain level after treatment (rTMS).
- **Input**: A pair of videos (Pre-treatment, Post-treatment) from the same subject and visit.
- **Methodology**: Leverage existing 5-class Pain Level predictions (Ordinal Classification 0-4).
- **Metric**:
  - Compute `Diff = Pred_Class_Pre - Pred_Class_Post`.
  - **Significant Reduction (Positive, 1)**: `Diff >= 2`.
  - **Not Significant (Negative, 0)**: `Diff < 2`.
  - **Constraint**: Since treatment is expected to reduce pain, `Diff < 0` (negative reduction, i.e., increased pain) is considered an **Incorrect Prediction** in strict evaluation.

## 2. Experimental Setup
- **Dataset**: Syracuse rTMS Dataset.
- **Samples**: 18 video pairs (Subject/Visit) after applying exclusion criteria (9 videos excluded).
- **Features**: 
  - **MMA Features**: Multi-modal (Face + Phys) features, processed via MIL-Transformer.
  - **Marlin Features**: Facial features extracted by Marlin encoder.
  - **Random Features**: Baseline for comparison.
- **Model Variants**: Evaluated different Augmentation Ratios (0.25, 0.5, 0.75, 1.0) during training of the base Pain Level estimator.

## 3. Key Findings

### 3.1 Best Model Configuration
The **MMA Features with 0.75 Augmentation Ratio** (`5class_0.75_aug_ratio`) consistently outperformed all other configurations.

| Configuration | Accuracy (Standard) | Accuracy (Strict) | F1 Score | Neg Diffs (Errors) |
| :--- | :--- | :--- | :--- | :--- |
| **MMA 0.75 Aug** | **0.667** | **0.500** | **0.625** | **4** |
| MMA 0.25 Aug | 0.611 | 0.444 | 0.462 | 6 |
| MMA 0.50 Aug | 0.611 | 0.444 | 0.462 | 5 |
| Marlin 0.75 Aug | 0.500 | N/A | 0.400 | N/A |

### 3.2 Standard vs. Strict Evaluation
- **Standard Accuracy**: Simply checks if `(Diff >= 2)` matches the ground truth label.
  - Best Result: **66.7%** (12/18 correct).
- **Strict Accuracy**: Treats any `Diff < 0` (prediction of increased pain) as an automatic failure.
  - Best Result: **50.0%** (9/18 correct and physiologically valid).
  - This metric reveals that while the model captures the general trend, it still suffers from temporal inconsistency in about 22-30% of cases (predicting pain increase).

### 3.3 Feature Comparison
- **MMA vs. Marlin**: MMA features significantly outperform Marlin features (Acc 0.67 vs 0.50).
- **Augmentation**: Higher augmentation ratios (0.75) generally improve robustness for this pairwise task, likely by helping the model generalize better across the variations between Pre and Post videos.
- **Random Features**: Random baselines achieved very low Strict Accuracy (< 10%), confirming the learned models are extracting meaningful pain signals.

## 4. Error Analysis
In the best model (MMA 0.75 Aug):
- **False Negatives (Missed Reduction)**: The primary source of error. The model tends to be conservative, often predicting `Diff = 1` (e.g., Class 3 -> 2) when the true reduction is significant (e.g., Class 3 -> 1).
- **Negative Diffs**: 4 out of 18 cases predicted an increase in pain. 
  - **Case 0**: True Diff = 3 (Significant), Pred Diff = -1. This is a critical failure where the model completely flipped the pain trend.
  - **Other Negatives**: True Diff = 0 or 1 (Not Significant), Pred Diff < 0. These are "correct" in binary classification (both < 2) but physiologically implausible.

## 5. Conclusion & Recommendations
1.  **Feasibility**: The task is solvable with current features, achieving significantly better-than-random performance (F1 0.625 vs Random Baseline).
2.  **Constraint Modeling**: The presence of negative predictions suggests that future models should explicitly incorporate a "non-increasing pain" constraint or joint Pre-Post modeling rather than independent predictions.
3.  **Data constraints**: The extremely small sample size (N=18) limits statistical power. Increasing the dataset (e.g., by relaxing exclusion criteria if data quality permits) is recommended.

## 6. Artifacts & Reproducibility
The detailed prediction and evaluation results are saved in the following CSV files:
- **Batch Evaluation Summary (MMA)**: `batch_pain_reduction_results.csv` (Standard metrics)
- **Batch Evaluation Summary (Strict)**: `batch_pain_reduction_strict.csv` (Strict metrics treating NegDiff as error)
- **Batch Evaluation Summary (Marlin)**: `batch_pain_reduction_results_marlin.csv`
- **Detailed Per-Sample Results (Best Model)**: `pain_reduction_diff_results.csv` (Generated from `5class_0.75_aug_ratio`)

All scripts used for this analysis:
- `eval_pain_reduction_from_preds.py`: Single experiment evaluation.
- `batch_eval_pain_reduction.py`: Batch evaluation standard.
- `batch_eval_pain_reduction_strict.py`: Batch evaluation strict.
- `analyze_diff_dist.py`: Distribution analysis of True vs Pred differences.

