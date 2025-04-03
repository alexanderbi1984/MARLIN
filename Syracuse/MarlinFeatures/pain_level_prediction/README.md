# Pain Level Prediction from Facial Expressions

This module focuses on predicting absolute pain levels (0-10 scale) from facial expression features extracted from videos, regardless of visit type (pre or post treatment).

## Background

In the Syracuse Pain Study, participants' facial expressions were recorded during pain experiences. These videos were processed using a pretrained video autoencoder to extract 768-dimensional feature vectors for each clip. The goal of this module is to predict self-reported pain levels (0-10 scale) from these facial expression features.

## Approach

The pain level prediction approach consists of two main steps:

1. **Feature Analysis**: Identify which facial expression features are most strongly correlated with pain levels
2. **Model Training**: Build regression models to predict pain levels using the most predictive features

## Scripts

### 1. Feature Analysis (`analyze_pain_level_features.py`)

This script analyzes which features are most significant for pain level prediction:

- Extracts features and pain levels from all videos
- Calculates multiple feature importance metrics:
  - F-regression for linear relationships
  - Mutual information for non-linear relationships
  - Pearson correlation
  - Spearman correlation
- Visualizes relationships between top features and pain levels
- Saves results to `feature_analysis/` directory

**Usage:**
```
python analyze_pain_level_features.py
```

### 2. Pain Level Prediction (`pain_level_predictor.py`)

This script builds and evaluates regression models for pain level prediction:

- Uses top features identified in the feature analysis
- Trains multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet
  - Support Vector Regression (SVR)
  - Random Forest
  - Gradient Boosting
- Performs 5-fold cross-validation
- Provides detailed evaluation metrics
- Visualizes model predictions
- Analyzes error patterns by subject and visit type

**Usage:**
```
python pain_level_predictor.py [OPTIONS]
```

**Options:**
- `--num_features N`: Number of top features to use (default: 20)
- `--model MODEL`: Model type to train (default: all)
  - Choices: all, Linear, Ridge, Lasso, ElasticNet, SVR, RandomForest, GradientBoosting
- `--optimize`: Optimize model hyperparameters using grid search

Examples:
```
# Train all models using top 20 features
python pain_level_predictor.py

# Train only Random Forest using top 30 features
python pain_level_predictor.py --num_features 30 --model RandomForest

# Train and optimize Gradient Boosting with top 15 features
python pain_level_predictor.py --num_features 15 --model GradientBoosting --optimize
```

## Results

The model evaluation results are saved to the `model_results/` directory:

- `average_results.csv`: Average metrics for each model across all folds
- `average_metrics.png`: Bar charts comparing model performance
- `fold_X/`: Results for each cross-validation fold
  - `model_predictions.png`: Scatter plots of actual vs. predicted pain levels
  - `error_by_subject.png`: Mean absolute error by subject
  - `error_by_visit_type.png`: Mean absolute error by visit type (pre/post)

## Data Structure

The underlying feature data for each video consists of:
- 14 clips per video
- 4 frames per clip
- 768 features per frame

For pain level prediction, features are averaged across frames and clips to get a single 768-dimensional feature vector per video.

## Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy 