# Pain Reduction Regression Analysis Results

## Overview
This report presents the results of regression analysis for predicting pain reduction using different feature sets and models. The analysis was performed using 3-fold cross-validation with various numbers of features (3, 5, 7, and 10) and two feature types (raw features and probability scores).

## Methodology

### 1. Data and Feature Selection
- **Dataset**: Syracuse pain dataset containing pre- and post-treatment data
- **Feature Selection**: Selected top features based on effect size from previous analysis
- **Target Variable**: Absolute pain reduction (pre_pain - post_pain)
- **Sample Size**: 24 patients
- **Feature Range**: [-0.820, 0.801]
- **Target Range**: [0.000, 10.000]
- **Target Statistics**: 
  - Mean: 3.188
  - Standard Deviation: 2.649
  - Binary Distribution: 5 negative cases, 19 positive cases

### 2. Feature Engineering Approaches

#### 2.1 Raw Features Approach
- Direct use of selected features based on effect size
- Features are standardized before modeling
- Feature dimensions: (24, N) where N is the number of selected features
- Features represent differences between post and pre-treatment measurements

#### 2.2 Probability Scores Approach
- Binary classification target: 1 if pain reduced, 0 if not
- Logistic regression model trained on raw features to predict binary outcome
- Probability scores used as single feature for regression
- Feature dimensions: (24, 1)
- Probability score range: [0.754, 0.859]

### 3. Models Evaluated

#### 3.1 Linear Models
- **Linear Regression**: Simple regression without regularization
- **Ridge Regression**: L2 regularization (alpha=1.0)
- **Lasso Regression**: L1 regularization (alpha=1.0)

#### 3.2 Generalized Additive Models (GAM)
- **GAM_4splines**: Basic model with 4 splines per feature
- **GAM_8splines**: More flexible model with 8 splines
- **GAM_4splines_lam0.1**: Less regularized (lambda=0.1)
- **GAM_4splines_lam10**: More regularized (lambda=10)

### 4. Validation Method
- 3-fold cross-validation
- Random splitting with seed 42 for reproducibility
- Performance metrics:
  - Mean Squared Error (MSE)
  - R² score
- Results reported as mean ± standard deviation across folds

### 5. Experimental Design
- Number of features tested: 3, 5, 7, and 10
- Feature types tested: raw features and probability scores
- All models evaluated with all feature configurations
- Total combinations: 7 models × 4 feature counts × 2 feature types = 56 configurations

### 6. Feature Selection Process
Features were selected based on absolute effect size from previous analysis:
1. Features ranked by absolute effect size
2. Top N features selected where N ∈ {3, 5, 7, 10}
3. Effect sizes ranged from 0.91 to 1.33
4. All selected features had p-values < 1e-14

### 7. Implementation Details
- Python implementation using scikit-learn and pygam
- Feature standardization using StandardScaler
- Cross-validation using KFold with shuffle=True
- All models trained with default parameters except specified regularization terms

## Results Summary

### Raw Features Results

| Model Type | Features | MSE ± Std | R² ± Std |
|------------|----------|-----------|----------|
| Linear | 3 | 5.511 ± 1.086 | 0.176 ± 0.148 |
| Ridge | 3 | 5.299 ± 0.933 | 0.208 ± 0.125 |
| Lasso | 3 | 6.912 ± 1.060 | -0.029 ± 0.069 |
| GAM_4splines | 3 | 6.190 ± 1.264 | 0.083 ± 0.104 |
| GAM_8splines | 3 | 7.582 ± 1.409 | -0.130 ± 0.168 |
| GAM_4splines_lam0.1 | 3 | 6.309 ± 1.301 | 0.065 ± 0.110 |
| GAM_4splines_lam10 | 3 | 6.167 ± 1.256 | 0.086 ± 0.103 |
| Linear | 5 | 9.718 ± 5.723 | -0.483 ± 0.910 |
| Ridge | 5 | 8.148 ± 4.100 | -0.242 ± 0.657 |
| Lasso | 5 | 6.834 ± 0.641 | -0.022 ± 0.005 |
| GAM_4splines | 5 | 6.190 ± 1.264 | 0.083 ± 0.104 |
| GAM_8splines | 5 | 7.582 ± 1.409 | -0.130 ± 0.168 |
| GAM_4splines_lam0.1 | 5 | 6.309 ± 1.301 | 0.065 ± 0.110 |
| GAM_4splines_lam10 | 5 | 6.167 ± 1.256 | 0.086 ± 0.103 |
| Linear | 7 | 15.646 ± 5.803 | -1.434 ± 1.040 |
| Ridge | 7 | 11.850 ± 3.200 | -0.828 ± 0.610 |
| Lasso | 7 | 6.834 ± 0.643 | -0.022 ± 0.005 |
| GAM_4splines | 7 | 6.190 ± 1.264 | 0.083 ± 0.104 |
| GAM_8splines | 7 | 7.582 ± 1.409 | -0.130 ± 0.168 |
| GAM_4splines_lam0.1 | 7 | 6.309 ± 1.301 | 0.065 ± 0.110 |
| GAM_4splines_lam10 | 7 | 6.167 ± 1.256 | 0.086 ± 0.103 |
| Linear | 10 | 34.765 ± 12.273 | -4.386 ± 2.267 |
| Ridge | 10 | 17.499 ± 3.638 | -1.686 ± 0.749 |
| Lasso | 10 | 6.836 ± 0.641 | -0.023 ± 0.005 |
| GAM_4splines | 10 | 6.190 ± 1.264 | 0.083 ± 0.104 |
| GAM_8splines | 10 | 7.582 ± 1.409 | -0.130 ± 0.168 |
| GAM_4splines_lam0.1 | 10 | 6.309 ± 1.301 | 0.065 ± 0.110 |
| GAM_4splines_lam10 | 10 | 6.167 ± 1.256 | 0.086 ± 0.103 |

### Probability Scores Results

| Model Type | Features | MSE ± Std | R² ± Std |
|------------|----------|-----------|----------|
| Linear | 3 | 4.533 ± 0.679 | 0.325 ± 0.054 |
| Ridge | 3 | 4.524 ± 0.666 | 0.326 ± 0.048 |
| Lasso | 3 | 5.730 ± 1.018 | 0.149 ± 0.073 |
| GAM_4splines | 3 | 4.540 ± 0.683 | 0.323 ± 0.054 |
| GAM_8splines | 3 | 5.168 ± 1.652 | 0.242 ± 0.172 |
| GAM_4splines_lam0.1 | 3 | 4.580 ± 0.705 | 0.318 ± 0.058 |
| GAM_4splines_lam10 | 3 | 4.533 ± 0.679 | 0.325 ± 0.054 |
| Linear | 5 | 4.224 ± 0.329 | 0.359 ± 0.098 |
| Ridge | 5 | 4.129 ± 0.168 | 0.375 ± 0.077 |
| Lasso | 5 | 5.065 ± 0.925 | 0.247 ± 0.079 |
| GAM_4splines | 5 | 4.222 ± 0.337 | 0.359 ± 0.099 |
| GAM_8splines | 5 | 4.301 ± 0.717 | 0.342 ± 0.160 |
| GAM_4splines_lam0.1 | 5 | 4.212 ± 0.377 | 0.360 ± 0.106 |
| GAM_4splines_lam10 | 5 | 4.224 ± 0.329 | 0.359 ± 0.098 |
| Linear | 7 | 3.956 ± 0.653 | 0.412 ± 0.044 |
| Ridge | 7 | 3.926 ± 0.670 | 0.416 ± 0.046 |
| Lasso | 7 | 5.005 ± 0.912 | 0.256 ± 0.070 |
| GAM_4splines | 7 | 3.962 ± 0.645 | 0.411 ± 0.043 |
| GAM_8splines | 7 | 4.664 ± 0.607 | 0.304 ± 0.030 |
| GAM_4splines_lam0.1 | 7 | 3.992 ± 0.608 | 0.406 ± 0.036 |
| GAM_4splines_lam10 | 7 | 3.956 ± 0.653 | 0.412 ± 0.044 |
| Linear | 10 | 3.920 ± 0.837 | 0.419 ± 0.073 |
| Ridge | 10 | 3.916 ± 0.792 | 0.419 ± 0.067 |
| Lasso | 10 | 5.088 ± 0.845 | 0.243 ± 0.057 |
| GAM_4splines | 10 | 3.921 ± 0.836 | 0.419 ± 0.072 |
| GAM_8splines | 10 | 4.443 ± 1.517 | 0.348 ± 0.162 |
| GAM_4splines_lam0.1 | 10 | 3.926 ± 0.833 | 0.418 ± 0.072 |
| GAM_4splines_lam10 | 10 | 3.920 ± 0.836 | 0.419 ± 0.073 |

## Key Findings

1. **Feature Type Impact**:
   - Probability scores consistently outperformed raw features
   - Best R² scores with probability scores: 0.419 ± 0.073
   - Best R² scores with raw features: 0.208 ± 0.125

2. **Number of Features Impact**:
   - Raw features: Performance degraded with more features
   - Probability scores: Performance improved with more features (3 to 7 features)

3. **Model Performance**:
   - Best performing models: Linear, Ridge, and GAM_4splines with probability scores
   - Lasso consistently underperformed
   - GAM_8splines showed higher variance in performance

4. **Stability**:
   - Probability scores showed more stable performance (lower standard deviations)
   - Raw features showed high variance, especially with more features

## Best Model Configuration

The best performing configuration was:
- Feature Type: Probability scores
- Number of Features: 7-10
- Model: Ridge regression
- Performance: R² = 0.416 ± 0.046, MSE = 3.926 ± 0.670

## Conclusions

1. The probability score approach significantly outperformed raw features
2. More features improved performance for probability scores but degraded it for raw features
3. Simple models (Linear, Ridge) performed as well as or better than more complex models
4. The relationship between features and pain reduction appears to be better captured through the probability score transformation

## Potential Improvements in Feature Selection

### 1. Correlation-Based Selection
- **Current Limitation**: Effect size-based selection might not capture the direct relationship with pain reduction
- **Proposed Approach**: 
  - Calculate correlation between each feature and pain reduction
  - Select features with highest absolute correlation coefficients
  - Consider both linear and rank correlations (Pearson and Spearman)
  - Set correlation threshold based on statistical significance

### 2. Mutual Information-Based Selection
- **Current Limitation**: Effect size doesn't capture non-linear relationships
- **Proposed Approach**:
  - Use mutual information to measure feature-pain reduction relationships
  - Can capture non-linear dependencies
  - More robust to outliers
  - Better for percentage pain reduction where relationships might be non-linear

### 3. Forward/Backward Stepwise Selection
- **Current Limitation**: Fixed number of features might not be optimal
- **Proposed Approach**:
  - Start with empty model (forward) or full model (backward)
  - Add/remove features based on statistical significance
  - Use AIC/BIC for model selection
  - Automatically determine optimal number of features

### 4. LASSO-Based Feature Selection
- **Current Limitation**: Effect size doesn't consider feature interactions
- **Proposed Approach**:
  - Use LASSO regression with cross-validation
  - Features with non-zero coefficients are selected
  - Naturally handles feature interactions
  - Built-in regularization prevents overfitting

### 5. Ensemble Feature Selection
- **Current Limitation**: Single selection criterion might miss important features
- **Proposed Approach**:
  - Combine multiple selection methods:
    - Effect size
    - Correlation
    - Mutual information
    - LASSO coefficients
  - Select features that appear in multiple methods
  - Weight features based on consistency across methods

### 6. Domain-Specific Feature Selection
- **Current Limitation**: Pure statistical approach might miss clinically relevant features
- **Proposed Approach**:
  - Group features by anatomical regions or movement types
  - Ensure representation from each group
  - Consider clinical significance in selection
  - Balance statistical and clinical relevance

### 7. Feature Interaction Selection
- **Current Limitation**: Only individual features are considered
- **Proposed Approach**:
  - Generate interaction terms between top features
  - Use statistical tests to identify significant interactions
  - Consider polynomial terms for non-linear relationships
  - Regularize interaction terms to prevent overfitting

### Implementation Recommendations

1. **For Absolute Pain Reduction**:
   - Start with correlation-based selection
   - Add LASSO-based selection for feature interactions
   - Use forward stepwise selection to optimize feature count
   - Consider domain-specific grouping

2. **For Percentage Pain Reduction**:
   - Use mutual information for initial selection
   - Add polynomial terms for non-linear relationships
   - Implement ensemble selection for robustness
   - Consider log-transformed features

3. **General Improvements**:
   - Implement cross-validation in feature selection process
   - Use bootstrap to assess feature selection stability
   - Consider feature importance rankings from multiple models
   - Add feature interaction terms systematically

These improvements could potentially lead to:
- More stable feature selection
- Better capture of non-linear relationships
- More clinically interpretable feature sets
- Improved model performance for both absolute and percentage pain reduction 