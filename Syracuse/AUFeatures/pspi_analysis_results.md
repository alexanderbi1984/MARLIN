# PSPI Feature Analysis for Pain Reduction Prediction

## Overview
This analysis investigates the effectiveness of the modified Prkachin & Solomon Pain Intensity (PSPI) score in predicting pain reduction outcomes. The PSPI score was modified to exclude AU43 (eye closure) due to data availability, resulting in:
PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10)

## Methodology
- **Dataset**: Syracuse pain dataset with 24 patients
- **Feature**: Modified PSPI score calculated from Action Units
- **Feature Engineering**: Difference in PSPI scores between post and pre conditions
- **Target Variables**:
  1. Binary outcome (significant pain reduction ≥ 4)
  2. Absolute pain reduction
  3. Percentage pain reduction
- **Models**:
  - Binary classification: Logistic Regression
  - Regression: Linear, Ridge, Lasso, and GAM variants
- **Validation**: 3-fold cross-validation

## Results

### PSPI Feature Statistics
- Mean PSPI difference: -0.283
- Standard deviation: 1.693
- Range: [-4.300, 3.276]

### Binary Classification (Significant Pain Reduction)
- Mean CV Accuracy: 0.667 (±0.118)
- Mean CV AUC: 0.578 (±0.227)

### Absolute Pain Reduction Prediction
Best performing models:
1. Linear Regression: R² = -0.020 (±0.230)
2. Ridge Regression: R² = -0.016 (±0.212)
3. Lasso Regression: R² = -0.060 (±0.077)

GAM models showed varying performance with R² scores ranging from -0.234 to -0.035.

### Percentage Pain Reduction Prediction
All models showed poor performance:
- Linear Regression: R² = -0.177 (±0.174)
- Ridge Regression: R² = -0.166 (±0.163)
- Lasso Regression: R² = -0.149 (±0.143)
- GAM models: R² ranging from -0.350 to -0.260

## Detailed Comparison with Raw AU Features

### Performance Metrics Comparison

#### Binary Classification Task
| Metric    | PSPI Feature | Top 5 AU Features |
|-----------|--------------|-------------------|
| Accuracy  | 0.667 ±0.118 | 0.542 ±0.118     |
| AUC       | 0.578 ±0.227 | 0.489 ±0.333     |

#### Absolute Pain Reduction Task
| Model     | PSPI Feature    | Top 5 AU Features |
|-----------|----------------|-------------------|
| Linear    | -0.020 ±0.230  | Negative R²      |
| Ridge     | -0.016 ±0.212  | Negative R²      |
| Lasso     | -0.060 ±0.077  | Negative R²      |
| GAM       | -0.035 ±0.662  | Negative R²      |

#### Percentage Pain Reduction Task
| Model     | PSPI Feature    | Top 5 AU Features |
|-----------|----------------|-------------------|
| Linear    | -0.177 ±0.174  | Negative R²      |
| Ridge     | -0.166 ±0.163  | Negative R²      |
| Lasso     | -0.149 ±0.143  | Negative R²      |
| GAM       | -0.260 ±0.608  | Negative R²      |

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

### Implications

1. **Feature Selection**:
   - While PSPI slightly outperforms individual AUs, the improvement is modest
   - The composite nature of PSPI may be losing some discriminative information

2. **Clinical Relevance**:
   - Neither approach achieves clinically reliable prediction
   - The slight advantage of PSPI in binary classification might be practically meaningful

3. **Future Directions**:
   - Consider hybrid approaches combining PSPI and individual AU features
   - Investigate temporal patterns in both PSPI and AU features
   - Explore alternative feature combinations based on pain expression literature

## Discussion
1. **Binary Classification Performance**:
   - The modified PSPI shows moderate performance in binary classification with 66.7% accuracy
   - AUC of 0.578 suggests only slightly better than random prediction

2. **Regression Performance**:
   - All models show negative R² scores, indicating worse performance than a horizontal line
   - The simpler models (Linear, Ridge) performed marginally better than more complex ones
   - GAM models with different spline configurations did not improve performance

3. **Feature Characteristics**:
   - The negative mean PSPI difference (-0.283) suggests a slight overall decrease in pain expression
   - Large standard deviation (1.693) indicates substantial variability in pain expression changes

## Comparison with Individual AU Features
Compared to our previous analysis using individual AU features:
1. The modified PSPI shows slightly better binary classification accuracy (0.667 vs 0.542)
2. Regression performance remains poor, similar to individual AU features
3. The composite PSPI score does not provide substantial improvement over individual AUs

## Conclusions
1. The modified PSPI score shows limited utility in predicting pain reduction outcomes
2. While binary classification performance is slightly better than individual AUs, it remains modest
3. The negative R² scores across all regression models suggest that the PSPI difference is not a reliable predictor of pain reduction magnitude

## Recommendations
1. Consider incorporating additional facial features or modalities
2. Investigate temporal patterns in PSPI scores rather than just using mean values
3. Explore non-linear relationships between PSPI changes and pain reduction
4. Consider combining PSPI with other pain indicators for more robust prediction 