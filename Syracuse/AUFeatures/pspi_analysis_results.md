# PSPI Feature Analysis for Pain Reduction Prediction

## Overview
This analysis investigates the effectiveness of the modified Prkachin & Solomon Pain Intensity (PSPI) score in predicting pain reduction outcomes. The PSPI score was modified to exclude AU43 (eye closure) due to data availability, resulting in:
PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10)

## Methodology

### 1. Data and Feature Selection
- **Dataset**: Syracuse pain dataset containing pre- and post-treatment data
- **Feature**: Modified PSPI score calculated from Action Units
  - AU4 (brow lowerer)
  - max(AU6, AU7) where:
    - AU6 (cheek raiser)
    - AU7 (lid tightener)
  - max(AU9, AU10) where:
    - AU9 (nose wrinkler)
    - AU10 (upper lip raiser)
- **Target Variables**:
  - Binary outcome: Pain reduction ≥ 4 (yes/no)
  - Absolute pain reduction (pre_pain - post_pain)
  - Percentage pain reduction ((pre_pain - post_pain) / pre_pain * 100)
- **Sample Size**: 24 patients
- **Feature Engineering**:
  - Features were averaged along the time axis
  - Differences were calculated between post and pre conditions
  - Features were scaled after taking differences
- **Feature Statistics**:
  - Mean PSPI difference: -0.283
  - Standard deviation: 1.693
  - Range: [-4.300, 3.276]
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
| Accuracy | 0.667 ± 0.118 |
| AUC | 0.578 ± 0.227 |

### 2. Absolute Pain Reduction Results
| Model Type | R² ± Std |
|------------|----------|
| Linear | -0.020 ± 0.230 |
| Ridge | -0.016 ± 0.212 |
| Lasso | -0.060 ± 0.077 |
| GAM_4splines | -0.219 ± 0.329 |
| GAM_8splines | -0.035 ± 0.662 |
| GAM_4splines_lam0.1 | -0.167 ± 0.384 |
| GAM_4splines_lam10 | -0.234 ± 0.318 |

### 3. Percentage Pain Reduction Results
| Model Type | R² ± Std |
|------------|----------|
| Linear | -0.177 ± 0.174 |
| Ridge | -0.166 ± 0.163 |
| Lasso | -0.149 ± 0.143 |
| GAM_4splines | -0.312 ± 0.158 |
| GAM_8splines | -0.260 ± 0.608 |
| GAM_4splines_lam0.1 | -0.350 ± 0.261 |
| GAM_4splines_lam10 | -0.305 ± 0.141 |

## Discussion

### 1. Binary Classification
- The model shows moderate predictive power (accuracy: 0.667)
- AUC of 0.578 suggests slightly better than random performance
- More stable performance compared to individual AU features

### 2. Absolute Pain Reduction
- All models show negative R² values
- Ridge performs best but still below baseline
- Indicates PSPI cannot predict magnitude of pain reduction

### 3. Percentage Pain Reduction
- Similar poor performance to absolute reduction
- No model shows meaningful predictive power
- Suggests PSPI cannot capture relative pain reduction

## Comparison with Individual AU Features

### Performance Metrics Comparison

#### Binary Classification Task
| Metric    | PSPI Feature | Top 5 AU Features |
|-----------|--------------|-------------------|
| Accuracy  | 0.667 ±0.118 | 0.542 ±0.118     |
| AUC       | 0.578 ±0.227 | 0.489 ±0.333     |

#### Absolute Pain Reduction Task
| Model     | PSPI Feature    | Top 5 AU Features |
|-----------|----------------|-------------------|
| Linear    | -0.020 ±0.230  | -0.371 ±0.404    |
| Ridge     | -0.016 ±0.212  | -0.371 ±0.404    |
| Lasso     | -0.060 ±0.077  | -0.076 ±0.093    |
| GAM       | -0.035 ±0.662  | -0.371 ±0.404    |

#### Percentage Pain Reduction Task
| Model     | PSPI Feature    | Top 5 AU Features |
|-----------|----------------|-------------------|
| Linear    | -0.177 ±0.174  | -0.371 ±0.404    |
| Ridge     | -0.166 ±0.163  | -0.371 ±0.404    |
| Lasso     | -0.149 ±0.143  | -0.076 ±0.093    |
| GAM       | -0.260 ±0.608  | -0.371 ±0.404    |

### Key Findings from Comparison

1. **Binary Classification**:
   - PSPI showed modest improvement in accuracy (12.5% increase)
   - AUC improved from 0.489 to 0.578, but still close to random
   - More stable performance with similar standard deviations

2. **Regression Tasks**:
   - Both approaches yielded negative R² scores
   - PSPI showed marginally better performance in absolute pain reduction
   - Neither approach achieved meaningful predictive power

3. **Feature Complexity**:
   - PSPI offers a simpler, single-feature approach compared to using multiple AUs
   - Reduced dimensionality did not significantly compromise performance
   - More interpretable due to established pain expression literature

4. **Advantages and Disadvantages**:
   
   PSPI Advantages:
   - Simpler feature space (single dimension)
   - Better binary classification performance
   - Based on established pain expression research
   - More interpretable results
   
   Raw AU Advantages:
   - Retains more granular facial expression information
   - Allows for feature-specific analysis
   - More flexible for alternative feature combinations

## Conclusions
1. The modified PSPI score shows limited utility in predicting pain reduction outcomes
2. While binary classification performance is slightly better than individual AUs, it remains modest
3. The negative R² scores across all regression models suggest that the PSPI difference is not a reliable predictor of pain reduction magnitude

## Recommendations
1. Consider incorporating additional facial features or modalities
2. Investigate temporal patterns in PSPI scores rather than just using mean values
3. Explore non-linear relationships between PSPI changes and pain reduction
4. Consider combining PSPI with other pain indicators for more robust prediction 