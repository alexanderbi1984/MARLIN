import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import sys
import os
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

def load_data(data_dir):
    """
    Load features and pain levels from saved data.
    
    Args:
        data_dir: Directory containing features.npy and metadata.csv
        
    Returns:
        Tuple of (features, pain_levels, metadata_df)
    """
    features_path = os.path.join(data_dir, 'features.npy')
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    
    if not os.path.exists(features_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}. Run extract_all_pain_data.py first.")
    
    features = np.load(features_path)
    metadata_df = pd.read_csv(metadata_path)
    pain_levels = metadata_df['pain_level'].values
    
    return features, pain_levels, metadata_df

def analyze_feature_importance(features, pain_levels):
    """
    Analyze which features are most important for pain level prediction.
    
    Args:
        features: Array of features, shape (n_samples, n_features)
        pain_levels: Array of pain levels, shape (n_samples,)
        
    Returns:
        DataFrame with feature importance metrics
    """
    print(f"Analyzing feature importance for {features.shape[1]} features across {features.shape[0]} samples")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # F-regression for linear relationships
    f_values, p_values = f_regression(features_scaled, pain_levels)
    
    # Mutual information for non-linear relationships
    mi_values = mutual_info_regression(features_scaled, pain_levels)
    
    # Pearson correlation
    pearson_corrs = []
    pearson_p_values = []
    for i in range(features.shape[1]):
        corr, p_value = pearsonr(features_scaled[:, i], pain_levels)
        pearson_corrs.append(corr)
        pearson_p_values.append(p_value)
    
    # Spearman correlation (rank correlation, less sensitive to outliers)
    spearman_corrs = []
    spearman_p_values = []
    for i in range(features.shape[1]):
        corr, p_value = spearmanr(features_scaled[:, i], pain_levels)
        spearman_corrs.append(corr)
        spearman_p_values.append(p_value)
    
    # Create DataFrame with results
    importance_df = pd.DataFrame({
        'feature_idx': np.arange(features.shape[1]),
        'f_value': f_values,
        'p_value': p_values,
        'mutual_info': mi_values,
        'pearson_corr': pearson_corrs,
        'pearson_p_value': pearson_p_values,
        'spearman_corr': spearman_corrs,
        'spearman_p_value': spearman_p_values,
        'abs_pearson_corr': np.abs(pearson_corrs),
        'abs_spearman_corr': np.abs(spearman_corrs)
    })
    
    # Sort by absolute Pearson correlation
    importance_df = importance_df.sort_values('abs_pearson_corr', ascending=False)
    
    # Apply Benjamini-Hochberg FDR correction for multiple testing
    from statsmodels.stats.multitest import multipletests
    
    # Correct p-values for Pearson correlation
    _, pearson_p_adjusted, _, _ = multipletests(
        importance_df['pearson_p_value'], alpha=0.05, method='fdr_bh'
    )
    importance_df['pearson_p_adjusted'] = pearson_p_adjusted
    
    # Correct p-values for Spearman correlation
    _, spearman_p_adjusted, _, _ = multipletests(
        importance_df['spearman_p_value'], alpha=0.05, method='fdr_bh'
    )
    importance_df['spearman_p_adjusted'] = spearman_p_adjusted
    
    # Correct p-values for F-regression
    _, f_p_adjusted, _, _ = multipletests(
        importance_df['p_value'], alpha=0.05, method='fdr_bh'
    )
    importance_df['p_value_adjusted'] = f_p_adjusted
    
    return importance_df

def visualize_top_features(features, pain_levels, importance_df, n_top=10, output_dir=None):
    """
    Visualize the relationship between top features and pain levels.
    
    Args:
        features: Array of features
        pain_levels: Array of pain levels
        importance_df: DataFrame with feature importance metrics
        n_top: Number of top features to visualize
        output_dir: Directory to save visualizations
    """
    # Get top features
    top_features = importance_df.head(n_top)
    
    # Create directory for visualizations if not provided
    if output_dir is None:
        output_dir = Path('Syracuse/pain_level_prediction/feature_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a figure
    plt.figure(figsize=(15, 12))
    
    # Plot each feature vs pain level
    for i, (_, row) in enumerate(top_features.iterrows()):
        feature_idx = int(row['feature_idx'])
        pearson_corr = row['pearson_corr']
        spearman_corr = row['spearman_corr']
        p_value = row['pearson_p_adjusted']  # Use adjusted p-value
        
        plt.subplot(3, 4, i+1)
        plt.scatter(features[:, feature_idx], pain_levels, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(features[:, feature_idx], pain_levels, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(features[:, feature_idx]), p(np.sort(features[:, feature_idx])), "r--")
        
        plt.title(f"Feature {feature_idx}\nPearson: {pearson_corr:.3f} (p={p_value:.3f})", fontsize=10)
        plt.xlabel(f"Feature Value")
        plt.ylabel("Pain Level (0-10)")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_features_vs_pain.png')
    plt.close()
    
    # Create correlation heatmap between top features
    plt.figure(figsize=(12, 10))
    feature_indices = top_features['feature_idx'].astype(int).tolist()
    top_feature_data = features[:, feature_indices]
    
    # Create column names
    cols = [f"F{idx}" for idx in feature_indices]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(top_feature_data.T)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=cols, yticklabels=cols)
    plt.title("Correlation Between Top Pain-Predicting Features")
    plt.tight_layout()
    plt.savefig(output_dir / 'top_features_correlation.png')
    plt.close()
    
    # Scatter plot matrix for top 5 features
    if len(feature_indices) >= 5:
        top5_indices = feature_indices[:5]
        top5_data = features[:, top5_indices]
        top5_cols = [f"F{idx}" for idx in top5_indices]
        
        # Create DataFrame with top 5 features and pain levels
        df = pd.DataFrame(top5_data, columns=top5_cols)
        df['Pain'] = pain_levels
        
        # Create scatter plot matrix
        sns.set(style="ticks")
        sns.pairplot(df, x_vars=top5_cols, y_vars=['Pain'], kind='reg', height=3)
        plt.suptitle("Top 5 Features vs Pain Level", y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'top5_features_scatter_matrix.png')
        plt.close()
    
    return top_features

def analyze_feature_by_visit_type(features, pain_levels, visit_types, importance_df, output_dir):
    """
    Analyze whether feature importance differs between pre and post treatment.
    
    Args:
        features: Array of features
        pain_levels: Array of pain levels
        visit_types: Array of visit types ('pre' or 'post')
        importance_df: DataFrame with feature importance metrics
        output_dir: Directory to save visualizations
    """
    # Get indices for pre and post treatment
    pre_idx = np.where(visit_types == 'pre')[0]
    post_idx = np.where(visit_types == 'post')[0]
    
    if len(pre_idx) == 0 or len(post_idx) == 0:
        print("Warning: Not enough pre or post treatment samples for separate analysis")
        return
    
    # Get top 5 features
    top_features = importance_df.head(5)
    feature_indices = top_features['feature_idx'].astype(int).tolist()
    
    # Plot each top feature by visit type
    for i, feature_idx in enumerate(feature_indices):
        plt.figure(figsize=(10, 6))
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Feature Value': np.concatenate([features[pre_idx, feature_idx], features[post_idx, feature_idx]]),
            'Pain Level': np.concatenate([pain_levels[pre_idx], pain_levels[post_idx]]),
            'Visit Type': np.concatenate([['Pre'] * len(pre_idx), ['Post'] * len(post_idx)])
        })
        
        # Plot scatter with regression lines for each visit type
        sns.lmplot(x='Feature Value', y='Pain Level', hue='Visit Type', data=df, height=5, aspect=1.5)
        
        plt.title(f"Feature {feature_idx} vs Pain Level by Visit Type")
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_{feature_idx}_by_visit_type.png')
        plt.close()
    
    # Analyze correlation differences
    pre_corrs = []
    post_corrs = []
    diff_corrs = []
    
    for feature_idx in range(features.shape[1]):
        pre_corr, _ = pearsonr(features[pre_idx, feature_idx], pain_levels[pre_idx])
        post_corr, _ = pearsonr(features[post_idx, feature_idx], pain_levels[post_idx])
        
        pre_corrs.append(pre_corr)
        post_corrs.append(post_corr)
        diff_corrs.append(abs(pre_corr) - abs(post_corr))
    
    # Find features with largest differences in correlation
    diff_df = pd.DataFrame({
        'feature_idx': np.arange(features.shape[1]),
        'pre_corr': pre_corrs,
        'post_corr': post_corrs,
        'diff': diff_corrs,
        'abs_diff': np.abs(diff_corrs)
    })
    
    # Get top 10 features with largest difference in correlation
    top_diff = diff_df.sort_values('abs_diff', ascending=False).head(10)
    
    # Plot features with largest differences
    plt.figure(figsize=(12, 8))
    
    for i, (_, row) in enumerate(top_diff.iterrows()):
        feature_idx = int(row['feature_idx'])
        pre_corr = row['pre_corr']
        post_corr = row['post_corr']
        
        plt.subplot(3, 4, i+1)
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Feature Value': np.concatenate([features[pre_idx, feature_idx], features[post_idx, feature_idx]]),
            'Pain Level': np.concatenate([pain_levels[pre_idx], pain_levels[post_idx]]),
            'Visit Type': np.concatenate([['Pre'] * len(pre_idx), ['Post'] * len(post_idx)])
        })
        
        # Plot scatter points colored by visit type
        sns.scatterplot(x='Feature Value', y='Pain Level', hue='Visit Type', data=df, alpha=0.7)
        
        plt.title(f"Feature {feature_idx}\nPre corr: {pre_corr:.3f}, Post corr: {post_corr:.3f}", fontsize=9)
        plt.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'features_with_diff_correlation.png')
    plt.close()
    
    return diff_df

def main():
    # Create output directory
    output_dir = Path('Syracuse/pain_level_prediction/feature_analysis')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading data...")
    data_dir = 'Syracuse/pain_level_prediction/all_videos_data'
    features, pain_levels, metadata_df = load_data(data_dir)
    
    print(f"Loaded {len(pain_levels)} samples with {features.shape[1]} features each")
    print(f"Pain level range: {np.min(pain_levels)} to {np.max(pain_levels)}")
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    importance_df = analyze_feature_importance(features, pain_levels)
    
    # Save feature importance results
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Print top 20 features
    print("\nTop 20 features by Pearson correlation magnitude:")
    print(importance_df[['feature_idx', 'pearson_corr', 'pearson_p_adjusted', 'mutual_info']].head(20).to_string())
    
    # Count significant features (after multiple testing correction)
    sig_pearson = np.sum(importance_df['pearson_p_adjusted'] < 0.05)
    sig_spearman = np.sum(importance_df['spearman_p_adjusted'] < 0.05)
    sig_f = np.sum(importance_df['p_value_adjusted'] < 0.05)
    
    print(f"\nNumber of significant features (adjusted p < 0.05):")
    print(f"  Pearson correlation: {sig_pearson} of {len(importance_df)}")
    print(f"  Spearman correlation: {sig_spearman} of {len(importance_df)}")
    print(f"  F-regression: {sig_f} of {len(importance_df)}")
    
    # Visualize top features
    print("\nVisualizing top features...")
    top_features = visualize_top_features(features, pain_levels, importance_df, output_dir=output_dir)
    
    # If visit type information is available, analyze features by visit type
    if 'visit_type' in metadata_df.columns:
        print("\nAnalyzing features by visit type...")
        visit_types = metadata_df['visit_type'].values
        diff_df = analyze_feature_by_visit_type(features, pain_levels, visit_types, importance_df, output_dir)
        diff_df.to_csv(output_dir / 'feature_correlation_by_visit_type.csv', index=False)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 