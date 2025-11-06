# Pain Level Classification Analysis Report

## 1. Dataset Overview

### 1.1 Data Collection
- Total videos in dataset: 96
- Videos with valid pain levels: 72
- Pain level range: 0-10 (continuous scale)
- Mean pain level: 4.76
- Median pain level: 4.50
- Standard deviation: 2.91

### 1.2 Pain Level Distribution
The pain levels show a relatively balanced distribution around the median:

Detailed distribution of pain levels:
- Pain Level 0.0: 6 samples
- Pain Level 1.0: 3 samples
- Pain Level 2.0: 9 samples
- Pain Level 3.0: 10 samples
- Pain Level 4.0: 8 samples
- Pain Level 5.0: 8 samples
- Pain Level 6.0: 5 samples
- Pain Level 7.0: 8 samples
- Pain Level 8.0: 4 samples
- Pain Level 8.5: 1 sample
- Pain Level 9.0: 6 samples
- Pain Level 9.5: 1 sample
- Pain Level 10.0: 3 samples

Binary classification distribution:
- Low pain (< 5.0): 36 samples
- High pain (≥ 5.0): 36 samples

## 2. Feature Selection

### 2.1 Feature Selection Method
We employed effect size (Cohen's d) to select the most discriminative features for pain classification. Cohen's d measures the standardized difference between means of two groups (high vs. low pain), providing a robust measure of feature importance that is:
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

All selected features show medium to large effect sizes, indicating strong discriminative power between pain classes.

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

### 3.2 Classification Setup
- Algorithm: Logistic Regression
  - Maximum iterations: 1000
  - Random state: 42
- Pain threshold: 5.0 (creates balanced classes)
- Validation strategy: 3-fold cross-validation
  - No separate test set to maximize training data
  - Stratified splits to maintain class distribution

## 4. Results

### 4.1 Cross-validation Performance
Mean metrics with standard deviations (3-fold):
- Accuracy: 0.736 ±0.052
- Precision: 0.734 ±0.035
- Recall: 0.758 ±0.160
- F1 Score: 0.734 ±0.074
- AUC: 0.807 ±0.014

### 4.2 Fold-wise Performance
1. Fold 1:
   - Accuracy: 0.750
   - Precision: 0.692
   - Recall: 0.818
   - F1 Score: 0.750
   - AUC: 0.797

2. Fold 2:
   - Accuracy: 0.792
   - Precision: 0.733
   - Recall: 0.917
   - F1 Score: 0.815
   - AUC: 0.826

3. Fold 3:
   - Accuracy: 0.667
   - Precision: 0.778
   - Recall: 0.538
   - F1 Score: 0.636
   - AUC: 0.797

### 4.3 Final Model Parameters
Logistic Regression coefficients:
- Intercept: 0.005
- Feature weights:
  1. -0.436 (Feature 0)
  2. -0.152 (Feature 1)
  3. 0.465 (Feature 2)
  4. 0.583 (Feature 3)
  5. -0.914 (Feature 4)

## 5. Analysis and Discussion

### 5.1 Model Performance Analysis
1. Overall Performance:
   - Good discrimination ability (AUC = 0.807 ±0.014)
   - Consistent performance across most metrics
   - Notable variability in recall (0.758 ±0.160)

2. Fold Analysis:
   - Best performance in Fold 2 (Accuracy: 0.792, AUC: 0.826)
   - Most challenging in Fold 3 (Accuracy: 0.667, AUC: 0.797)
   - Recall varies significantly (0.538 to 0.917)

3. Feature Importance:
   - Feature 49 shows strongest effect size (0.944)
   - All selected features have moderate to strong effects (>0.5)
   - Model assigns highest weight to Feature 4 (-0.914)

### 5.2 Limitations
1. Dataset Size:
   - 72 samples may limit generalizability
   - Small fold sizes in 3-fold CV (24 samples per fold)

2. Model Stability:
   - High variance in recall across folds
   - Performance depends heavily on fold composition

3. Feature Interpretation:
   - Limited understanding of feature semantics
   - Potential for spurious correlations

### 5.3 Future Directions
1. Model Improvements:
   - Explore additional feature selection methods
   - Investigate more complex models (e.g., neural networks)
   - Consider ensemble approaches to reduce variance

2. Validation:
   - Collect additional data to increase sample size
   - Implement external validation on new datasets
   - Explore different pain thresholds

3. Feature Analysis:
   - Investigate temporal aspects of pain expression
   - Study feature interactions and combinations
   - Map features to interpretable pain indicators

## 6. Visualization

Generated plots are saved in `Syracuse/pain_level_prediction/classification_results/`:
1. `fold_performance.png`: Performance metrics across folds
2. `confusion_matrix.png`: Last fold confusion matrix
3. `roc_curve.png`: Last fold ROC curve 