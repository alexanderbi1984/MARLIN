import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import argparse
from pygam import LinearGAM, s
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Add parent directory to path to import syracuse_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from syracuse_dataset import SyracuseDataset

def load_feature_selection_results(num_features):
    """Load pre-calculated feature selection results based on effect sizes.
    
    Args:
        num_features: Number of top features to select based on absolute effect size
        
    Returns:
        DataFrame containing selected features with their effect sizes and p-values
    """
    feature_selection_path = '/Users/hd927/Documents/MARLIN/outcome_analysis_results/marlin_clip_outcome_analysis.csv'
    feature_df = pd.read_csv(feature_selection_path)
    
    # Sort by absolute effect size and select top features
    feature_df['abs_effect_size'] = feature_df['effect_size'].abs()
    selected_features = feature_df.nlargest(num_features, 'abs_effect_size')
    
    print("\nSelected features and their effect sizes:")
    print(selected_features[['feature_idx', 'effect_size', 'p_value']].to_string())
    
    return selected_features

def prepare_features(dataset, selected_features, target_type='absolute', feature_type='raw'):
    """Prepare features for regression using pre-selected features based on effect sizes.
    
    This function handles two types of feature preparation:
    1. Raw features: Uses the top N features (based on effect size) directly
    2. Probability scores: Uses the top N features to train logistic regression and get probability scores
    
    Args:
        dataset: SyracuseDataset instance containing pre and post treatment features
        selected_features: DataFrame with top N features selected based on effect size
        target_type: 'absolute' or 'percentage' for pain reduction calculation
            - 'absolute': pre_pain - post_pain (raw difference)
            - 'percentage': (pre_pain - post_pain) / pre_pain (0-1 scale)
        feature_type: 'raw' or 'probability' for feature type selection
            - 'raw': Uses the selected features directly (dimension = N)
            - 'probability': Uses selected features to train logistic regression and get probability scores (dimension = 1)
    
    Returns:
        X: Feature matrix (either raw features or probability scores)
            - For raw features: shape (n_samples, n_features)
            - For probability scores: shape (n_samples, 1)
        y: Target values (pain reduction)
        feature_indices: Array of feature indices used
    """
    pre_features, post_features, changes = dataset.get_all_features()
    
    # Get feature indices from selected features
    feature_indices = selected_features['feature_idx'].values
    
    # Select only the pre-selected features
    pre_selected = pre_features[:, :, :, feature_indices]
    post_selected = post_features[:, :, :, feature_indices]
    
    # Average across temporal dimensions (clips and frames)
    pre_avg = np.mean(pre_selected, axis=(1, 2))  # Average across clips and frames
    post_avg = np.mean(post_selected, axis=(1, 2))  # Average across clips and frames
    
    # Calculate feature differences (post - pre)
    X_raw = post_avg - pre_avg
    
    # Get pain values from pairs
    pairs_df = dataset.get_pair_info()
    pre_pain = pairs_df['pre_pain'].values
    post_pain = pairs_df['post_pain'].values
    
    # Calculate pain reduction based on target_type
    if target_type == 'absolute':
        y = pre_pain - post_pain  # Positive values indicate reduction
        target_label = "Pain reduction (absolute)"
    else:  # percentage
        y = (pre_pain - post_pain) / pre_pain  # Percentage reduction (0-1 scale)
        target_label = "Pain reduction (0-1 scale)"
    
    # Create binary target for logistic regression (1 if pain reduced, 0 if not)
    y_binary = (y > 0).astype(int)
    
    # Choose feature type based on argument
    if feature_type == 'raw':
        X = X_raw
        feature_label = f"Raw feature differences (top {len(feature_indices)} features)"
    else:  # probability
        # Train logistic regression using selected features to get probability scores
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_raw, y_binary)
        prob_scores = log_reg.predict_proba(X_raw)[:, 1]  # Probability of pain reduction
        X = prob_scores.reshape(-1, 1)
        feature_label = f"Logistic regression probability scores (using top {len(feature_indices)} features)"
    
    print(f"\nFeature dimensions:")
    print(f"Original features shape: {pre_avg.shape}")
    print(f"Selected features shape: {X.shape}")
    print(f"Number of features used in regression: {X.shape[1]}")
    
    # Add diagnostic prints
    print("\nDiagnostic information:")
    print(f"Original features range: [{np.min(pre_avg):.3f}, {np.max(pre_avg):.3f}]")
    if feature_type == 'probability':
        print(f"Probability scores range: [{np.min(prob_scores):.3f}, {np.max(prob_scores):.3f}]")
    print(f"Target ({target_label}) range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    print(f"Target mean: {np.mean(y):.3f}")
    print(f"Target std: {np.std(y):.3f}")
    print(f"Binary target distribution: {np.bincount(y_binary)}")
    
    # Print first few samples
    print("\nFirst sample:")
    print(f"Original features: {pre_avg[0]}")
    if feature_type == 'probability':
        print(f"Probability score: {prob_scores[0]:.3f}")
    print(f"Pre pain: {pre_pain[0]:.3f}")
    print(f"Post pain: {post_pain[0]:.3f}")
    print(f"Binary target: {y_binary[0]}")
    print(f"{target_label} (y): {y[0]:.3f}")
    print(f"Selected features ({feature_label}): {X[0]}")
    
    return X, y, feature_indices

def train_regression(X, y, feature_names, num_features, model_type='all'):
    """Train regression models with 3-fold cross-validation.
    
    This function trains various regression models to predict pain reduction.
    When using probability scores (X.shape[1] == 1), all models effectively perform
    the same task of finding the best relationship between probability and pain reduction.
    
    Args:
        X: Feature matrix (either raw features or probability scores)
            - For raw features: shape (n_samples, n_features)
            - For probability scores: shape (n_samples, 1)
        y: Target values (pain reduction)
        feature_names: Names of features (not used currently)
        num_features: Number of features used (not used currently)
        model_type: Type of model to train
            - 'all': Train all models
            - 'Linear': Simple linear regression without regularization
            - 'Ridge': Ridge regression with L2 regularization
            - 'Lasso': Lasso regression with L1 regularization
            - 'GAM_4splines': Generalized Additive Model with 4 splines
            - 'GAM_8splines': Generalized Additive Model with 8 splines
            - 'GAM_4splines_lam0.1': GAM with 4 splines and low regularization
            - 'GAM_4splines_lam10': GAM with 4 splines and high regularization
    
    Returns:
        Dictionary containing results for each model including:
        - MSE and R² scores with standard deviations
        - Trained model
        - Feature importance (if available)
    """
    # Store results
    results = {}
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models based on model_type
    if model_type == 'all':
        models = {
            'Linear': LinearRegression(),  # Simple linear regression without regularization
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'GAM_4splines': LinearGAM(s(0, n_splines=4)),
            'GAM_8splines': LinearGAM(s(0, n_splines=8)),
            'GAM_4splines_lam0.1': LinearGAM(s(0, n_splines=4, lam=0.1)),
            'GAM_4splines_lam10': LinearGAM(s(0, n_splines=4, lam=10))
        }
    else:
        # Only create the specified model
        if model_type == 'Linear':
            models = {'Linear': LinearRegression()}
        elif model_type == 'Ridge':
            models = {'Ridge': Ridge(alpha=1.0)}
        elif model_type == 'Lasso':
            models = {'Lasso': Lasso(alpha=1.0)}
        elif model_type == 'GAM_4splines':
            models = {'GAM_4splines': LinearGAM(s(0, n_splines=4))}
        elif model_type == 'GAM_8splines':
            models = {'GAM_8splines': LinearGAM(s(0, n_splines=8))}
        elif model_type == 'GAM_4splines_lam0.1':
            models = {'GAM_4splines_lam0.1': LinearGAM(s(0, n_splines=4, lam=0.1))}
        elif model_type == 'GAM_4splines_lam10':
            models = {'GAM_4splines_lam10': LinearGAM(s(0, n_splines=4, lam=10))}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Use 3-fold Cross-Validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        # Perform 3-fold CV
        mse_scores = []
        r2_scores = []
        
        for train_idx, val_idx in kf.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Make prediction
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            mse_scores.append(mse)
            r2_scores.append(r2)
        
        # Calculate mean and std of metrics
        results[name] = {
            'mse': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'r2': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'model': model,
            'feature_importance': np.abs(model.coef_) if hasattr(model, 'coef_') else None
        }
        
        print(f"{name} Results:")
        print(f"MSE: {results[name]['mse']:.3f} ± {results[name]['mse_std']:.3f}")
        print(f"R2: {results[name]['r2']:.3f} ± {results[name]['r2_std']:.3f}")
    
    return results

def analyze_feature_importance(model, feature_indices, selected_features, feature_importance=None):
    """Analyze and plot feature importance.
    
    This function handles two cases:
    1. Single feature (probability score): Creates a simple bar plot of the coefficient
    2. Multiple features: Creates a bar plot of the top 20 most important features
    
    Args:
        model: The trained model
        feature_indices: Array of feature indices
        selected_features: DataFrame with selected features and their effect sizes
        feature_importance: Pre-calculated feature importance (optional)
    
    Returns:
        DataFrame containing feature importance analysis
            - For probability score: Single row with coefficient value
            - For raw features: DataFrame with feature indices, importance, and effect sizes
    """
    # Get feature importance from model coefficients or pre-calculated values
    if feature_importance is not None:
        importance = feature_importance
    else:
        importance = np.abs(model.coef_)
    
    # If we have only one feature (probability score), create a simple plot
    if len(importance) == 1:
        plt.figure(figsize=(8, 6))
        plt.bar(['Probability Score'], importance)
        plt.title('Model Coefficient for Probability Score')
        plt.ylabel('Coefficient Value')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Create a simple DataFrame with just the probability score
        importance_df = pd.DataFrame({
            'feature_idx': ['Probability Score'],
            'importance': importance,
            'effect_size': [0.0]  # No effect size for probability score
        })
        return importance_df
    
    # For raw features, create the original analysis
    importance_df = pd.DataFrame({
        'feature_idx': feature_indices,
        'importance': importance
    })
    
    # Merge with selected features to get effect sizes
    importance_df = importance_df.merge(selected_features, on='feature_idx')
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature_idx')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importance_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train regression models for pain reduction prediction')
    parser.add_argument('--num_features', type=int, default=10,
                      help='Number of top features to select based on effect size')
    parser.add_argument('--target_type', type=str, choices=['absolute', 'percentage'],
                      default='absolute',
                      help='Type of target variable: absolute change or percentage change')
    parser.add_argument('--model_type', type=str, 
                      choices=['all', 'Linear', 'Ridge', 'Lasso', 'GAM_4splines', 'GAM_8splines', 
                              'GAM_4splines_lam0.1', 'GAM_4splines_lam10'],
                      default='all',
                      help='Type of model to use: all, Ridge, Lasso, or GAM variants')
    parser.add_argument('--feature_type', type=str,
                      choices=['raw', 'probability'],
                      default='raw',
                      help='Type of features to use: raw feature differences or logistic regression probability scores')
    args = parser.parse_args()
    
    # Load pre-selected features
    selected_features = load_feature_selection_results(args.num_features)
    print(f"Selected {len(selected_features)} features based on effect sizes")
    
    # Initialize dataset
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Prepare features using selected features
    X, y, feature_indices = prepare_features(dataset, selected_features, args.target_type, args.feature_type)
    
    # Train regression models
    results = train_regression(X, y, feature_indices, args.num_features, args.model_type)
    
    # # Analyze feature importance
    # for name, result in results.items():
    #     print(f"\nFeature importance for {name}:")
    #     importance_df = analyze_feature_importance(
    #         result['model'], 
    #         feature_indices, 
    #         selected_features,
    #         feature_importance=result['feature_importance']
    #     )
    #     print(importance_df[['feature_idx', 'importance', 'effect_size']].head())
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('regression_results.csv')

if __name__ == "__main__":
    main() 