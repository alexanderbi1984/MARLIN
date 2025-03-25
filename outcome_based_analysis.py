"""
Outcome-based Feature Analysis

This script analyzes MARLIN features considering treatment outcomes. It:
1. Groups patients by treatment outcome (success/failure)
2. Analyzes feature changes within each outcome group
3. Identifies features that distinguish successful from unsuccessful treatments
4. Visualizes feature patterns specific to each outcome group

Outcome Definitions:
- Positive (Success): Treatment led to improvement in pain symptoms
- Negative (Failure): Treatment did not lead to improvement in pain symptoms

The outcome should be specified in the metadata Excel file with a column 'outcome'
containing values 'positive' or 'negative'.

Author: Nan Bi
Date: March 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze features based on treatment outcomes')
    parser.add_argument('--features_dir', type=str, 
                      default=r'C:\pain\syracus\openface_clips\clips\multimodal_marlin_base',
                      help='Directory containing feature files')
    parser.add_argument('--meta_path', type=str,
                      default=r'C:\pain\syracus\openface_clips\clips\multimodal_marlin_base\meta_with_outcomes.xlsx',
                      help='Path to metadata file with outcomes')
    parser.add_argument('--output_dir', type=str,
                      default='outcome_analysis_results',
                      help='Directory to save analysis results')
    parser.add_argument('--features', type=str, nargs='+',
                      default=['marlin'],
                      help='Feature types to analyze')
    return parser.parse_args()

def load_metadata(meta_path):
    """Load metadata with outcomes"""
    try:
        df = pd.read_excel(meta_path)
        logging.info(f"Loaded metadata from {meta_path}")
        logging.info(f"Outcome distribution:\n{df['outcome'].value_counts()}")
        return df
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        raise

def load_features(features_dir, filename, feature_type='marlin'):
    """Load features for a given file"""
    try:
        feature_path = Path(features_dir) / f"{filename}_{feature_type}.npy"
        return np.load(str(feature_path))
    except Exception as e:
        logging.error(f"Error loading features for {filename}: {e}")
        return None

def compute_effect_size(group1, group2):
    """Compute Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / s if s != 0 else np.nan

def analyze_features(features_pos, features_neg, feature_type):
    """Analyze differences between positive and negative outcome groups"""
    n_features = features_pos.shape[1]
    results = []
    
    for i in tqdm(range(n_features), desc=f"Analyzing {feature_type} features"):
        pos_vals = features_pos[:, i]
        neg_vals = features_neg[:, i]
        
        # Compute statistics
        t_stat, p_val = stats.ttest_ind(pos_vals, neg_vals)
        effect_size = compute_effect_size(pos_vals, neg_vals)
        
        results.append({
            'feature_idx': i,
            'effect_size': effect_size,
            'p_value': p_val,
            'pos_mean': np.mean(pos_vals),
            'neg_mean': np.mean(neg_vals),
            'pos_std': np.std(pos_vals),
            'neg_std': np.std(neg_vals)
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    _, p_corrected = fdrcorrection(results_df['p_value'])
    results_df['p_value_fdr'] = p_corrected
    
    return results_df

def plot_feature_comparison(feature_idx, pos_vals, neg_vals, results_df, output_dir, feature_type):
    """Create box plot comparing feature values between outcomes"""
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    data = {
        'Outcome': ['Positive'] * len(pos_vals) + ['Negative'] * len(neg_vals),
        'Feature Value': np.concatenate([pos_vals, neg_vals])
    }
    df_plot = pd.DataFrame(data)
    
    # Create box plot
    sns.boxplot(x='Outcome', y='Feature Value', data=df_plot)
    
    # Add statistical annotations
    result = results_df[results_df['feature_idx'] == feature_idx].iloc[0]
    plt.title(f"Feature {feature_idx} Distribution by Outcome\n" + 
              f"Effect Size: {result['effect_size']:.2f}, " +
              f"p-value (FDR): {result['p_value_fdr']:.3f}")
    
    # Save plot
    plt.savefig(Path(output_dir) / f"{feature_type}_feature_{feature_idx}_comparison.png")
    plt.close()

def kernel_mmd_test(X, Y, kernel='rbf', num_permutations=1000):
    """
    Perform Kernel Maximum Mean Discrepancy (MMD) test between positive and negative outcomes.
    """
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    
    def rbf_kernel(X, Y):
        X_norm = torch.sum(X ** 2, dim=1, keepdim=True)
        Y_norm = torch.sum(Y ** 2, dim=1, keepdim=True)
        dist = X_norm + Y_norm.T - 2 * torch.mm(X, Y.T)
        sigma = torch.median(dist[dist > 0])
        return torch.exp(-dist / (2 * sigma))
    
    def compute_mmd(X, Y):
        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)
        
        n_x = X.size(0)
        n_y = Y.size(0)
        
        mmd = (K_XX.sum() - torch.diag(K_XX).sum()) / (n_x * (n_x - 1))
        mmd += (K_YY.sum() - torch.diag(K_YY).sum()) / (n_y * (n_y - 1))
        mmd -= 2 * K_XY.mean()
        
        return mmd.item()
    
    # Compute original MMD
    original_mmd = compute_mmd(X, Y)
    
    # Permutation test
    combined = torch.cat([X, Y], dim=0)
    n_x = X.size(0)
    permutation_mmds = []
    
    for _ in tqdm(range(num_permutations), desc="Running permutation test"):
        perm = torch.randperm(combined.size(0))
        X_perm = combined[perm[:n_x]]
        Y_perm = combined[perm[n_x:]]
        permutation_mmds.append(compute_mmd(X_perm, Y_perm))
    
    # Compute p-value
    p_value = sum(mmd >= original_mmd for mmd in permutation_mmds) / num_permutations
    
    return original_mmd, p_value

def visualize_distributions(X_pos, X_neg, output_dir, feature_type):
    """Create dimensionality reduction visualizations."""
    # Combine data for dimensionality reduction
    X_combined = np.vstack([X_pos, X_neg])
    labels = ['Positive'] * len(X_pos) + ['Negative'] * len(X_neg)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    # Create PCA plot
    fig = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=labels,
        title='PCA Visualization of Positive vs Negative Outcomes',
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
    )
    
    fig.write_html(Path(output_dir) / f"{feature_type}_pca_visualization.html")
    
    # UMAP
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_combined)
    
    # Create UMAP plot
    fig = px.scatter(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        color=labels,
        title='UMAP Visualization of Positive vs Negative Outcomes',
        labels={'x': 'UMAP1', 'y': 'UMAP2'}
    )
    
    fig.write_html(Path(output_dir) / f"{feature_type}_umap_visualization.html")
    
    # Print explained variance ratio for PCA
    logging.info("\nPCA Explained Variance Ratio:")
    logging.info(f"First component: {pca.explained_variance_ratio_[0]:.3f}")
    logging.info(f"Second component: {pca.explained_variance_ratio_[1]:.3f}")
    logging.info(f"Total explained variance: {sum(pca.explained_variance_ratio_[:2]):.3f}")

def analyze_feature_importance(X_pre, X_post, feature_type, output_dir):
    """Analyze individual feature importance using effect size and statistical tests."""
    n_features = X_pre.shape[1]
    effect_sizes = []
    p_values = []
    
    logging.info("\nAnalyzing individual features...")
    for i in tqdm(range(n_features)):
        # Calculate Cohen's d effect size
        d = compute_cohens_d(X_pre[:, i], X_post[:, i])
        effect_sizes.append(abs(d))
        
        # Perform Mann-Whitney U test
        stat, p = stats.mannwhitneyu(X_pre[:, i], X_post[:, i], alternative='two-sided')
        p_values.append(p)
    
    # Convert to numpy arrays
    effect_sizes = np.array(effect_sizes)
    p_values = np.array(p_values)
    
    # Apply Benjamini-Hochberg correction
    sorted_p_idx = np.argsort(p_values)
    sorted_p_values = p_values[sorted_p_idx]
    n_features = len(p_values)
    fdr = 0.05  # False Discovery Rate threshold
    
    # Calculate critical values
    critical_values = np.arange(1, n_features + 1) * fdr / n_features
    significant_features = sorted_p_values <= critical_values
    
    if np.any(significant_features):
        last_significant = np.where(significant_features)[0][-1]
        significant_threshold = sorted_p_values[last_significant]
        significant_features = p_values <= significant_threshold
    else:
        significant_features = np.zeros_like(p_values, dtype=bool)
    
    # Create visualization of top features
    n_top_features = 20
    top_idx = np.argsort(effect_sizes)[-n_top_features:]
    
    # Create interactive box plot of top feature
    fig = go.Figure()
    
    # Add positive distribution
    fig.add_trace(go.Box(
        y=X_pre[:, top_idx[-1]],
        name='Positive',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker_color='blue',
        showlegend=True
    ))
    
    # Add negative distribution
    fig.add_trace(go.Box(
        y=X_post[:, top_idx[-1]],
        name='Negative',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker_color='red',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'Distribution of Top Discriminative Feature (Effect Size = {effect_sizes[top_idx[-1]]:.3f})',
        yaxis_title='Feature Value',
        boxmode='group'
    )
    
    fig.write_html(Path(output_dir) / f"{feature_type}_top_feature_distribution.html")
    
    # Create heatmap of top features for positive outcomes
    top_features_pos = X_pre[:, top_idx]
    top_features_neg = X_post[:, top_idx]
    
    fig = go.Figure()
    
    # Add heatmap for positive
    fig.add_trace(go.Heatmap(
        z=top_features_pos.T,
        name='Positive',
        colorscale='Blues',
        showscale=True,
        xaxis='x',
        yaxis='y'
    ))
    
    fig.update_layout(
        title='Top 20 Most Discriminative Features Heatmap (Positive Outcomes)',
        xaxis_title='Samples',
        yaxis_title='Features',
        width=1000,
        height=600
    )
    
    fig.write_html(Path(output_dir) / f"{feature_type}_top_features_heatmap_positive.html")
    
    # Create heatmap for negative outcomes
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=top_features_neg.T,
        name='Negative',
        colorscale='Reds',
        showscale=True,
        xaxis='x',
        yaxis='y'
    ))
    
    fig.update_layout(
        title='Top 20 Most Discriminative Features Heatmap (Negative Outcomes)',
        xaxis_title='Samples',
        yaxis_title='Features',
        width=1000,
        height=600
    )
    
    fig.write_html(Path(output_dir) / f"{feature_type}_top_features_heatmap_negative.html")
    
    logging.info(f"\nFeature Importance Analysis:")
    logging.info(f"Number of significant features (FDR < 0.05): {np.sum(significant_features)}")
    logging.info(f"\nTop 5 most discriminative features:")
    for i, idx in enumerate(top_idx[-5:]):
        logging.info(f"Feature {idx}: Effect size = {effect_sizes[idx]:.3f}, p-value = {p_values[idx]:.6f}")
    
    return effect_sizes, p_values, significant_features

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load metadata
    metadata = load_metadata(args.meta_path)
    
    # Process each feature type
    for feature_type in args.features:
        logging.info(f"\nAnalyzing {feature_type} features...")
        
        # Load features for each group
        features_pos = []
        features_neg = []
        
        for _, row in metadata[metadata['outcome'].notna()].iterrows():
            features = load_features(args.features_dir, row['file_name'], feature_type)
            if features is not None:
                if row['outcome'] == 'positive':
                    features_pos.append(features)
                else:
                    features_neg.append(features)
        
        features_pos = np.vstack(features_pos)
        features_neg = np.vstack(features_neg)
        
        logging.info(f"Loaded {len(features_pos)} positive and {len(features_neg)} negative samples")
        
        # Standardize features
        scaler = StandardScaler()
        features_pos_scaled = scaler.fit_transform(features_pos)
        features_neg_scaled = scaler.transform(features_neg)
        
        # Perform MMD test
        logging.info("\nPerforming Kernel MMD test...")
        mmd_value, p_value = kernel_mmd_test(features_pos_scaled, features_neg_scaled)
        logging.info(f"MMD value: {mmd_value:.6f}")
        logging.info(f"p-value: {p_value:.6f}")
        logging.info(f"Statistically significant difference: {p_value < 0.05}")
        
        # Create distribution visualizations
        logging.info("\nCreating distribution visualizations...")
        visualize_distributions(features_pos_scaled, features_neg_scaled, output_dir, feature_type)
        
        # Analyze feature importance
        effect_sizes, p_values, significant_features = analyze_feature_importance(
            features_pos_scaled, 
            features_neg_scaled, 
            feature_type,
            output_dir
        )

if __name__ == "__main__":
    main() 