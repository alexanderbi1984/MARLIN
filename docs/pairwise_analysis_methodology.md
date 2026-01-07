# Pairwise Comparison Methodology for LOOCV

This document details the statistical methodology used to compare the proposed method (e.g., MMA/Remix) against baselines (e.g., AU+PSPI, Marlin) using the results from Leave-One-Subject-Out Cross-Validation (LOOCV).

## 1. Motivation: Why Pairwise?

Standard evaluation metrics often report the **Average Performance** across all subjects (folds). While averages are useful, they have limitations:
*   **Outliers**: A method might perform exceptionally well on a few "easy" subjects, inflating the average even if it performs worse on the majority.
*   **Ignoring Paired Data**: LOOCV produces **paired samples**. For every test subject $S_i$, we have results from both Method A and Method B. We should analyze the *difference* $\Delta_i = Score_{B,i} - Score_{A,i}$ rather than independent distributions.

**Pairwise Analysis** allows us to:
1.  Quantify the consistency of improvement (Win/Tie/Loss counts).
2.  Perform rigorous paired statistical tests (Wilcoxon Signed-Rank Test) to determine significance.

## 2. The "Effective Comparison" Protocol

In challenging classification tasks (e.g., 4-class or 5-class pain estimation), the model performance can sometimes be close to or even below random chance. Comparing two methods when both fail completely (e.g., Accuracy < Random Guess) is noisy and statistically meaningless.

To address this, we introduce an **Effective Comparison** filter:

### 2.1 Definition of "Double Fail"
A test subject $S_i$ is considered a **Double Fail** if:
$$ Score_{A,i} \le \tau \quad \text{AND} \quad Score_{B,i} \le \tau $$
where $\tau$ is the random baseline threshold (e.g., 0.25 for 4-class accuracy, 0.0 for QWK).

### 2.2 Analysis Logic
1.  **Filter**: Exclude all "Double Fail" subjects.
2.  **Effective Set**: Perform statistical analysis only on the remaining subjects where **at least one method** achieved meaningful performance.
3.  **Rationale**: This focuses the evaluation on *competence*. We ask: *"In cases where the task is solvable by at least one model, which model performs better?"*

## 3. Metrics and Tests

We report the following for the Effective Set:

*   **Effective Mean**: The average score excluding Double Fails.
*   **Win / Tie / Loss**:
    *   **Win**: Proposed Method > Baseline
    *   **Loss**: Proposed Method < Baseline
*   **Wilcoxon Signed-Rank Test**: A non-parametric paired test. Returns a p-value indicating whether the median of differences is significantly different from zero.
    *   $p < 0.05$: Statistically significant improvement.
    *   $p > 0.05$: Improvement is not distinguishable from random noise.

## 4. Usage Example

We use the `analyze_pairwise_folds.py` script to perform this analysis.

### Command
Comparing **MMA (Random Features)** vs. **AU Baseline** on a 4-class task, filtering out random guess performance (< 0.25):

```bash
/data/Nbi/marlin/bin/python analyze_pairwise_folds.py \
  --dir_a "/path/to/AU_results/4class..." \
  --dir_b "/path/to/MMA_results/4class..." \
  --name_a "AU (Baseline)" \
  --name_b "MMA (Ours)" \
  --metric test_acc \
  --baseline_val 0.25
```

### Sample Output Interpretation
```text
Global Mean MMA: 0.3188
Global Mean AU:  0.2651
------------------------------
Effective Mean MMA: 0.3399
Effective Mean AU:  0.2781
Double Fails (Excluded): 4
------------------------------
Win / Tie / Loss (Effective): 16 / 0 / 12
------------------------------
Wilcoxon p-value: 0.27407
```
**Interpretation**:
*   The **Global Mean** shows MMA is +5.3% better.
*   After removing 4 subjects where both failed, the **Effective Mean** gap widens to +6.2%.
*   MMA wins 16 times vs 12 losses.
*   However, the p-value (0.27) suggests that given the sample size and variance, this result is not yet statistically significant, though it shows a strong positive trend.










