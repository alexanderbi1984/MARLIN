# Analysis of Micro-expression Features in Pain Treatment Response

## Research Context
This study investigates the relationship between facial micro-expressions and pain treatment outcomes in a clinical setting. The research aims to understand how subtle facial expressions, captured through micro-expression analysis, might be associated with pain reduction after treatment.

## Dataset Description
- Total sample size: 24 participants
- Treatment outcome classification:
  - Positive responders: 9 participants (pain reduction ≥ 4)
  - Non-responders: 15 participants (pain reduction < 4)
- Features analyzed: 8 micro-expression logits
  - Surprise
  - Fear
  - Happiness
  - Neutral
  - Contempt
  - Sadness
  - Anger
  - Disgust

## Feature Analysis Results

### Feature Statistics
1. **Pre-treatment Features**:
   - Neutral: mean = 2.361, std = 0.327, range = [1.704, 3.011]
   - Sadness: mean = 1.702, std = 0.499, range = [0.844, 2.863]
   - Surprise: mean = 1.714, std = 0.479, range = [0.127, 2.486]
   - Fear: mean = 1.529, std = 0.569, range = [-0.035, 2.454]
   - Happiness: mean = 1.510, std = 0.503, range = [0.479, 3.019]
   - Anger: mean = 1.330, std = 0.289, range = [0.856, 1.832]
   - Contempt: mean = 0.730, std = 0.472, range = [-0.344, 1.762]
   - Disgust: mean = 0.332, std = 0.678, range = [-0.931, 1.526]

2. **Post-treatment Features**:
   - Similar patterns observed with slight variations in means and ranges
   - Feature differences (post - pre) used for reduction prediction tasks

### Outcome Statistics
1. **Pain Reduction**:
   - Mean: 3.19 ± 2.65 points
   - Range: 0 to 10 points
   - Distribution:
     * No improvement (0): 5 cases (21%)
     * Small improvement (1-3): 10 cases (42%)
     * Significant improvement (≥4): 8 cases (33%)
     * Complete improvement (10): 1 case (4%)

2. **Initial Pain Levels**:
   - Range: 0 to 10
   - Distribution across pain levels shows variation

## Task Performance Analysis

### Task 1: Binary Classification (Pain Reduction ≥4)
- Using pre-treatment features to predict significant pain reduction
- 3-fold cross-validation with balanced class weights
- Models: Logistic Regression and Random Forest
- Results show moderate performance with class imbalance challenges

### Task 2: Pain Level Prediction
- Predicting initial pain levels using pre-treatment features
- 3-fold cross-validation
- Models: Linear Regression and Random Forest
- Performance metrics include MSE, RMSE, and R²

### Task 3: Absolute Pain Reduction Prediction
- Predicting exact pain reduction using feature differences
- 3-fold cross-validation
- Models: Linear Regression and Random Forest
- Using post-pre feature differences to capture changes in expressions

### Task 4: Percentage Pain Reduction Prediction
- Predicting relative pain reduction using feature differences
- 3-fold cross-validation
- Models: Linear Regression and Random Forest
- Using post-pre feature differences normalized by initial pain level

## Methodological Considerations

### Data Processing
1. Feature Scaling:
   - StandardScaler applied to features before model training
   - Separate scaling for each fold to prevent data leakage

2. Model Selection:
   - Binary tasks: Logistic Regression and Random Forest with balanced weights
   - Regression tasks: Linear Regression and Random Forest
   - Ensemble approach by averaging predictions from multiple models

3. Cross-validation:
   - 3-fold cross-validation to account for small sample size
   - Random splitting with fixed seed for reproducibility

### Challenges and Limitations
1. Small sample size (n=24) limiting statistical power
2. Class imbalance in binary classification tasks
3. High variability in pain reduction outcomes
4. Complex relationships between features and outcomes

## Future Directions
1. Consider alternative feature representations
2. Explore non-linear model architectures
3. Investigate temporal patterns in micro-expressions
4. Consider incorporating additional clinical variables
5. Explore feature interaction terms
6. Investigate alternative thresholds for binary classification

## Conclusion
The analysis reveals moderate associations between micro-expressions and pain treatment outcomes. While the results show promise, the small sample size and class imbalance present significant challenges. The use of feature differences for reduction prediction tasks provides a more direct approach to capturing changes in emotional expressions. Further investigation with larger samples and refined methodologies is needed to establish robust associations. 