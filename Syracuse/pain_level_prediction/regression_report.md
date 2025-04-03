# Pain Level Regression Analysis Report

## 1. Dataset Overview

### 1.1 Data Collection
- Total videos in dataset: 96
- Videos with valid pain levels: 72
- Pain level range: 0-10 (continuous scale)
- Mean pain level: 4.76
- Median pain level: 4.50
- Standard deviation: 2.89

### 1.2 Pain Level Distribution
The pain levels show a relatively balanced distribution around the median, with:
- Minimum: 0.0
- Maximum: 10.0
- Mean: 4.76
- Median: 4.50
- Standard deviation: 2.89

## 2. Feature Selection

### 2.1 Feature Selection Method
We employed effect size (Cohen's d) to select the most discriminative features for pain level prediction. Cohen's d measures the standardized difference between means of two groups (high vs. low pain), providing a robust measure of feature importance that is:
- Scale-independent
- Sample-size robust
- Interpretable (d = 0.2: small effect, d = 0.5: medium effect, d = 0.8: large effect)

### 2.2 Selected Features
Top 5 features by effect size:
1. Feature 49: d = 0.944 (large effect)
2. Feature 386: d = 0.650 (medium-large effect)
3. Feature 417: d = 0.621 (medium-large effect)
4. Feature 94: d = 0.615 (medium-large effect)
5. Feature 580: d = 0.592 (medium effect)

All selected features show medium to large effect sizes, indicating strong discriminative power for pain level prediction.

## 3. Methodology

### 3.1 Data Processing
1. Video Processing:
   - Each video is segmented into 14 clips
   - Each clip is normalized to 4 frames
   - Features are extracted using a pre-trained model (768-dimensional feature vector)
   - Features are averaged across frames and clips to get one feature vector per video

2. Feature Selection:
   - Cohen's d effect size calculated for each feature
   - Top 5 features selected based on effect size magnitude
   - Features standardized using StandardScaler

### 3.2 Regression Setup
- Algorithms tested:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Elastic Net
  - Support Vector Regression (SVR)
  - Random Forest
  - Gradient Boosting
- Validation strategy: 3-fold cross-validation
  - No separate test set to maximize training data
  - Stratified splits to maintain pain level distribution

## 4. Results

### 4.1 Cross-validation Performance
Mean metrics with standard deviations (3-fold):

1. Support Vector Regression (SVR):
   - R²: 0.240 (±0.026)
   - Training MSE: 4.283
   - Testing MSE: 10.216

2. Random Forest:
   - R²: 0.173 (±0.194)
   - Training MSE: 0.921
   - Testing MSE: 13.538

3. Ridge Regression:
   - R²: 0.073 (±0.268)
   - Training MSE: 5.602
   - Testing MSE: 10.639

4. Linear Regression:
   - R²: 0.058 (±0.283)
   - Training MSE: 5.599
   - Testing MSE: 10.843

5. Gradient Boosting:
   - R²: 0.063 (±0.271)
   - Training MSE: 0.038
   - Testing MSE: 16.052

6. Elastic Net:
   - R²: 0.061 (±0.097)
   - Training MSE: 7.024
   - Testing MSE: 8.742

7. Lasso Regression:
   - R²: -0.037 (±0.042)
   - Training MSE: 8.044
   - Testing MSE: 9.397

### 4.2 Model Analysis
1. Best Performing Model: SVR
   - Highest and most stable R² score
   - Lowest variance in cross-validation
   - Moderate MSE values

2. Model Stability:
   - SVR shows most consistent performance across folds
   - Complex models (RandomForest, GradientBoosting) show signs of overfitting
   - Linear models show moderate performance with high variance

3. Error Analysis:
   - MSE ranges from 4.283 (SVR training) to 16.052 (GradientBoosting testing)
   - Testing MSE is consistently higher than training MSE
   - SVR shows best balance between training and testing performance

## 5. Analysis and Discussion

### 5.1 Model Performance Analysis
1. Overall Performance:
   - SVR shows best discrimination ability (R² = 0.240 ±0.026)
   - Most models show moderate to poor performance
   - High variance in performance across folds

2. Feature Importance:
   - Feature 49 shows strongest effect size (0.944)
   - All selected features have moderate to strong effects (>0.5)
   - Feature selection appears effective but may need refinement

3. Model Behavior:
   - Linear models show expected performance
   - Complex models show severe overfitting
   - SVR demonstrates best generalization

### 5.2 Limitations
1. Dataset Size:
   - 72 samples may limit model performance
   - Small fold sizes in 3-fold CV (24 samples per fold)

2. Model Stability:
   - High variance in performance across folds
   - Significant gap between training and testing performance

3. Feature Interpretation:
   - Limited understanding of feature semantics
   - Potential for spurious correlations

### 5.3 Future Directions
1. Model Improvements:
   - Tune SVR hyperparameters
   - Explore feature interactions
   - Consider ensemble approaches

2. Validation:
   - Collect additional data
   - Implement external validation
   - Explore different feature selection methods

3. Feature Analysis:
   - Investigate temporal aspects
   - Study feature combinations
   - Map features to interpretable pain indicators

## 6. Visualization

Generated plots are saved in `Syracuse/pain_level_prediction/regression_results/`:
1. Model performance comparisons
2. Prediction vs actual value plots
3. Error distribution plots
4. Feature importance visualizations 