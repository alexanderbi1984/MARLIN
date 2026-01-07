# AU Feature Analysis for Pain Reduction Prediction

## Overview
This report presents the results of analyzing the effectiveness of Action Unit (AU) features in predicting pain reduction. The analysis was performed using 3-fold cross-validation with 5 selected AU features and multiple model types.

## Methodology

### 1. Data and Feature Selection
- **Dataset**: Syracuse pain dataset containing pre- and post-treatment data
- **Feature Selection**: Selected top 5 features based on absolute effect size:
  - AU12_r (0.494)
  - AU07_r (-0.481)
  - AU05_r (-0.415)
  - AU01_r (-0.257)
  - AU02_r (-0.254)
- **Target Variables**:
  - Binary outcome: Pain reduction ≥ 4 (yes/no)
  - Absolute pain reduction (pre_pain - post_pain)
  - Percentage pain reduction ((pre_pain - post_pain) / pre_pain * 100)
- **Sample Size**: 24 patients
- **Feature Engineering**:
  - Features were averaged along the time axis
  - Differences were calculated between post and pre conditions
  - Features were scaled after taking differences
- **Feature Range**: [-0.820, 0.801]
- **Target Range**: 
  - Binary: {0, 1}
  - Absolute: [0.000, 10.000]
  - Percentage: [0.000, 100.000]

### 2. Models Evaluated

#### 2.1 Binary Classification
- **Logistic Regression**: Standard logistic regression with L2 regularization
  - Maximum iterations: 1000
  - Cross-validation metrics: Accuracy and AUC

#### 2.2 Regression Models
- **Linear Regression**: Simple regression without regularization
- **Ridge Regression**: L2 regularization (alpha=1.0)
- **Lasso Regression**: L1 regularization (alpha=1.0)
- **GAM_4splines**: Basic model with 4 splines per feature
- **GAM_8splines**: More flexible model with 8 splines
- **GAM_4splines_lam0.1**: Less regularized (lambda=0.1)
- **GAM_4splines_lam10**: More regularized (lambda=10)

### 3. Validation Method
- 3-fold cross-validation
- Random splitting with seed 42 for reproducibility
- Performance metrics:
  - Binary: Accuracy and AUC
  - Regression: R² score
- Results reported as mean ± standard deviation across folds
- Feature scaling applied after taking differences between pre and post conditions

## Results

### 1. Binary Classification Results
| Metric | Score ± Std |
|--------|------------|
| Accuracy | 0.542 ± 0.118 |
| AUC | 0.489 ± 0.333 |

### 2. Absolute Pain Reduction Results
| Model Type | R² ± Std |
|------------|----------|
| Linear | -0.371 ± 0.404 |
| Ridge | -0.371 ± 0.404 |
| Lasso | -0.076 ± 0.093 |
| GAM_4splines | -0.371 ± 0.404 |
| GAM_8splines | -0.371 ± 0.404 |
| GAM_4splines_lam0.1 | -0.371 ± 0.404 |
| GAM_4splines_lam10 | -0.371 ± 0.404 |

### 3. Percentage Pain Reduction Results
| Model Type | R² ± Std |
|------------|----------|
| Linear | -0.371 ± 0.404 |
| Ridge | -0.371 ± 0.404 |
| Lasso | -0.076 ± 0.093 |
| GAM_4splines | -0.371 ± 0.404 |
| GAM_8splines | -0.371 ± 0.404 |
| GAM_4splines_lam0.1 | -0.371 ± 0.404 |
| GAM_4splines_lam10 | -0.371 ± 0.404 |

## Discussion

### 1. Binary Classification
- The model shows minimal predictive power (accuracy: 0.542)
- AUC below 0.5 indicates worse than random performance
- Suggests selected AU features cannot reliably predict significant pain reduction

### 2. Absolute Pain Reduction
- All models show negative R² values
- Lasso performs best but still below baseline
- Indicates AU features cannot predict magnitude of pain reduction

### 3. Percentage Pain Reduction
- Similar poor performance to absolute reduction
- No model shows meaningful predictive power
- Suggests AU features cannot capture relative pain reduction

## Conclusions
1. The selected AU features (AU12_r, AU07_r, AU05_r, AU01_r, AU02_r) show limited utility in predicting pain reduction
2. No model type (linear, regularized, or GAM) shows meaningful predictive power
3. The poor performance could be attributed to:
   - Small sample size (24 patients)
   - Complexity of pain reduction not well-captured by AU features
   - Possible need for different feature selection or combination

## Recommendations
1. Consider alternative feature sets or combinations
2. Explore non-linear relationships between features
3. Investigate temporal patterns in AU features
4. Consider larger sample size or different feature engineering approaches 
5. Use PSPI as a feature
6. Try PCA etc as features.