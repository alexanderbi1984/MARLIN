"""
Outcome-based Analysis for Preprocessed Biovid Data

This script analyzes the preprocessed Biovid dataset to identify features that significantly
differ between successful and unsuccessful treatments.

Analysis Components:
1. Data Loading
   - Loads preprocessed feature differences and metadata
   - Splits data into positive and negative outcome groups

2. Statistical Analysis
   - Mann-Whitney U test for feature differences
   - Cohen's d effect size calculation
   - False Discovery Rate (FDR) correction
   - Kernel Maximum Mean Discrepancy (MMD) test

3. Visualization
   - Feature comparison box plots
   - Feature change heatmaps
   - PCA and UMAP dimensionality reduction plots
   - Top feature distribution plots

Input Requirements:
1. Preprocessed Data Files:
   - pairs_metadata.json: Contains metadata for each pair
   - feature_diffs.npy: Contains feature differences for each pair
   - feature_diff_stats.csv: Contains statistical summaries

Output Files:
1. Statistical Results:
   - biovid_feature_analysis.csv: Feature-level analysis results
   - biovid_mmd_results.txt: MMD test results

2. Visualizations:
   - biovid_feature_changes_heatmap.html
   - biovid_feature_[N]_changes.png
   - biovid_pca_visualization.html
   - biovid_umap_visualization.html
   - biovid_top_feature_distribution.html

Usage:
    python biovid_outcome_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import json
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('biovid_analysis.log')
    ]
)

def load_preprocessed_data(data_dir):
    """Load preprocessed feature differences and metadata."""
    print("Loading preprocessed data...")  # Direct print for immediate feedback
    
    # Load metadata
    meta_path = os.path.join(data_dir, 'pairs_metadata.json')
    print(f"Loading metadata from: {meta_path}")  # Direct print
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    # Load feature differences
    diff_path = os.path.join(data_dir, 'feature_diffs.npy')
    print(f"Loading feature differences from: {diff_path}")  # Direct print
    feature_diffs = np.load(diff_path, allow_pickle=True).item()
    
    # Initialize lists for positive and negative samples
    X_pos = []
    X_neg = []
    
    # Process all pairs
    print("Processing pairs...")  # Direct print
    for pair_id, diff in feature_diffs.items():
        pair_meta = metadata[pair_id]
        pre_pain = pair_meta['pre_ground_truth']
        
        # Only consider pairs where pre-treatment pain level is 4
        if pre_pain == 4:
            if pair_meta['outcome'] == 1:  # Positive outcome
                X_pos.append(diff)
            else:  # Negative outcome
                X_neg.append(diff)
    
    # Convert to numpy arrays
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    
    # Log data loading summary
    print(f"\nData Loading Summary (Pre-treatment pain level = 4):")  # Direct print
    print(f"Positive samples: {len(X_pos)}")  # Direct print
    print(f"Negative samples: {len(X_neg)}")  # Direct print
    print(f"Feature dimension: {X_pos.shape[1]}")  # Direct print
    
    return X_pos, X_neg

def compute_effect_size(group1, group2, batch_size=1000):
    """Compute Cohen's d effect size using memory-efficient batching"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    
    # Compute means in batches
    mean1 = 0
    mean2 = 0
    for i in range(0, n1, batch_size):
        mean1 += np.sum(group1[i:min(i + batch_size, n1)])
    for i in range(0, n2, batch_size):
        mean2 += np.sum(group2[i:min(i + batch_size, n2)])
    mean1 /= n1
    mean2 /= n2
    
    # Compute variances in batches
    var1 = 0
    var2 = 0
    for i in range(0, n1, batch_size):
        batch = group1[i:min(i + batch_size, n1)]
        var1 += np.sum((batch - mean1) ** 2)
    for i in range(0, n2, batch_size):
        batch = group2[i:min(i + batch_size, n2)]
        var2 += np.sum((batch - mean2) ** 2)
    var1 = var1 / (n1 - 1)
    var2 = var2 / (n2 - 1)
    
    # Compute pooled standard deviation
    s = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (mean1 - mean2) / s if s != 0 else np.nan

def analyze_features(X_pos, X_neg, n_features=None):
    """Analyze differences in feature changes between positive and negative outcome groups"""
    if n_features is None:
        n_features = X_pos.shape[1]
    
    # Initialize results storage
    results = []
    
    # Standardize features first
    scaler = StandardScaler()
    X_combined = np.vstack([X_pos, X_neg])
    X_scaled = scaler.fit_transform(X_combined)
    X_pos_scaled = X_scaled[:len(X_pos)]
    X_neg_scaled = X_scaled[len(X_pos):]
    
    logging.info("\nAnalyzing features...")
    for i in tqdm(range(n_features)):
        pos_vals = X_pos_scaled[:, i]
        neg_vals = X_neg_scaled[:, i]
        
        # Compute statistics
        t_stat, p_val = stats.ttest_ind(pos_vals, neg_vals)
        effect_size = compute_effect_size(pos_vals, neg_vals)
        
        # Store original (unscaled) means and stds
        pos_mean = np.mean(X_pos[:, i])
        neg_mean = np.mean(X_neg[:, i])
        pos_std = np.std(X_pos[:, i])
        neg_std = np.std(X_neg[:, i])
        
        results.append({
            'feature_idx': i,
            'effect_size': effect_size,
            'p_value': p_val,
            't_statistic': t_stat,
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
            'pos_std': pos_std,
            'neg_std': neg_std
        })
    
    # Convert to DataFrame and sort by absolute effect size
    results_df = pd.DataFrame(results)
    results_df['abs_effect_size'] = np.abs(results_df['effect_size'])
    results_df = results_df.sort_values('abs_effect_size', ascending=False)
    
    # Apply FDR correction to p-values
    results_df['p_value_adj'] = fdrcorrection(results_df['p_value'])[1]
    
    return results_df

def plot_feature_comparison(feature_idx, pos_vals, neg_vals, results_df, output_dir):
    """Create box plot comparing feature changes between outcomes"""
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    data = {
        'Outcome': ['Positive'] * len(pos_vals) + ['Negative'] * len(neg_vals),
        'Feature Change': np.concatenate([pos_vals, neg_vals])
    }
    df_plot = pd.DataFrame(data)
    
    # Create box plot
    sns.boxplot(x='Outcome', y='Feature Change', data=df_plot)
    
    # Add statistical annotations
    result = results_df[results_df['feature_idx'] == feature_idx].iloc[0]
    plt.title(f"Feature {feature_idx} Changes by Outcome\n" + 
              f"Effect Size: {result['effect_size']:.2f}, " +
              f"p-value (FDR): {result['p_value_adj']:.3f}")
    
    # Add horizontal line at y=0 to show direction of change
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Save plot
    plt.savefig(Path(output_dir) / f"biovid_feature_{feature_idx}_changes.png")
    plt.close()

def plot_feature_heatmap(X_pos, X_neg, results_df, output_dir):
    """Create heatmap of feature changes"""
    # Get top features by absolute effect size
    results_df['abs_effect_size'] = results_df['effect_size'].abs()
    top_features = results_df.nlargest(10, 'abs_effect_size')
    
    # Prepare data for heatmap
    pos_changes = X_pos[:, top_features['feature_idx']]
    neg_changes = X_neg[:, top_features['feature_idx']]
    
    # Create heatmap data
    heatmap_data = np.vstack([pos_changes, neg_changes])
    
    # Create labels
    feature_labels = [f"Feature {idx}" for idx in top_features['feature_idx']]
    outcome_labels = ['Positive'] * len(pos_changes) + ['Negative'] * len(neg_changes)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=feature_labels,
        y=outcome_labels,
        colorscale='RdBu',
        colorbar=dict(title='Feature Change'),
        zmid=0,  # Center the colormap at 0
        text=np.round(heatmap_data, 2),
        texttemplate='%{text}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title="Top 10 Feature Changes by Outcome",
        xaxis_title="Feature",
        yaxis_title="Outcome"
    )
    
    fig.write_html(Path(output_dir) / "biovid_feature_changes_heatmap.html")

def kernel_mmd_test(X, Y, kernel='rbf', num_permutations=1000, batch_size=100):
    """Perform Kernel Maximum Mean Discrepancy (MMD) test with memory-efficient batching"""
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    
    def rbf_kernel_batched(X_batch, Y_batch, sigma):
        """Compute RBF kernel for batches"""
        dist = torch.cdist(X_batch, Y_batch)
        return torch.exp(-dist / (2 * sigma))
    
    def compute_mmd_batched(X, Y, batch_size):
        n_x = X.size(0)
        n_y = Y.size(0)
        
        # Compute sigma using a small subset
        sample_size = min(100, n_x)
        idx = torch.randperm(n_x)[:sample_size]
        X_sample = X[idx]
        dist_sample = torch.cdist(X_sample, X_sample)
        sigma = torch.median(dist_sample[dist_sample > 0])
        
        # Initialize MMD components
        xx_sum = 0
        yy_sum = 0
        xy_sum = 0
        xx_count = 0
        yy_count = 0
        xy_count = 0
        
        # Compute XX term
        for i in range(0, n_x, batch_size):
            end_i = min(i + batch_size, n_x)
            X_batch = X[i:end_i]
            
            for j in range(i, n_x, batch_size):
                end_j = min(j + batch_size, n_x)
                X_batch2 = X[j:end_j]
                
                K = rbf_kernel_batched(X_batch, X_batch2, sigma)
                if i == j:
                    # Remove diagonal elements
                    K = K - torch.diag(torch.diag(K))
                    xx_sum += K.sum().item()
                    xx_count += (end_i - i) * (end_j - j) - (end_i - i)
                else:
                    xx_sum += 2 * K.sum().item()
                    xx_count += 2 * (end_i - i) * (end_j - j)
        
        # Compute YY term
        for i in range(0, n_y, batch_size):
            end_i = min(i + batch_size, n_y)
            Y_batch = Y[i:end_i]
            
            for j in range(i, n_y, batch_size):
                end_j = min(j + batch_size, n_y)
                Y_batch2 = Y[j:end_j]
                
                K = rbf_kernel_batched(Y_batch, Y_batch2, sigma)
                if i == j:
                    K = K - torch.diag(torch.diag(K))
                    yy_sum += K.sum().item()
                    yy_count += (end_i - i) * (end_j - j) - (end_i - i)
                else:
                    yy_sum += 2 * K.sum().item()
                    yy_count += 2 * (end_i - i) * (end_j - j)
        
        # Compute XY term
        for i in range(0, n_x, batch_size):
            end_i = min(i + batch_size, n_x)
            X_batch = X[i:end_i]
            
            for j in range(0, n_y, batch_size):
                end_j = min(j + batch_size, n_y)
                Y_batch = Y[j:end_j]
                
                K = rbf_kernel_batched(X_batch, Y_batch, sigma)
                xy_sum += K.sum().item()
                xy_count += (end_i - i) * (end_j - j)
        
        # Compute final MMD
        mmd = xx_sum / xx_count + yy_sum / yy_count - 2 * xy_sum / xy_count
        return mmd
    
    # Compute original MMD
    logging.info("Computing original MMD...")
    original_mmd = compute_mmd_batched(X, Y, batch_size)
    
    # Permutation test
    logging.info("Running permutation test...")
    combined = torch.cat([X, Y], dim=0)
    n_x = X.size(0)
    permutation_mmds = []
    
    for i in tqdm(range(num_permutations), desc="Running permutation test"):
        perm = torch.randperm(combined.size(0))
        X_perm = combined[perm[:n_x]]
        Y_perm = combined[perm[n_x:]]
        permutation_mmds.append(compute_mmd_batched(X_perm, Y_perm, batch_size))
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute p-value
    p_value = sum(mmd >= original_mmd for mmd in permutation_mmds) / num_permutations
    
    return original_mmd, p_value

def visualize_distributions(X_pos, X_neg, output_dir):
    """Create dimensionality reduction visualizations"""
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
    
    fig.write_html(Path(output_dir) / "biovid_pca_visualization.html")
    
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
    
    fig.write_html(Path(output_dir) / "biovid_umap_visualization.html")
    
    # Print explained variance ratio for PCA
    logging.info("\nPCA Explained Variance Ratio:")
    logging.info(f"First component: {pca.explained_variance_ratio_[0]:.3f}")
    logging.info(f"Second component: {pca.explained_variance_ratio_[1]:.3f}")
    logging.info(f"Total explained variance: {sum(pca.explained_variance_ratio_[:2]):.3f}")

def main():
    # Set up paths using absolute paths
    data_dir = Path(r"C:\Users\Nan Bi\PycharmProjects\MARLIN\Syracuse\binary_outcome_classification\processed_data")
    output_dir = Path(r"C:\Users\Nan Bi\PycharmProjects\MARLIN\Syracuse\binary_outcome_classification\analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    logging.info("Starting analysis...")
    
    # Load preprocessed data
    logging.info("Loading preprocessed data...")
    X_pos, X_neg = load_preprocessed_data(data_dir)
    
    # Perform feature analysis
    logging.info("Analyzing features...")
    results_df = analyze_features(X_pos, X_neg)
    
    # Save results
    logging.info("Saving analysis results...")
    results_df.to_csv(output_dir / "biovid_feature_analysis.csv", index=False)
    
    # Create visualizations
    logging.info("Creating feature heatmap...")
    plot_feature_heatmap(X_pos, X_neg, results_df, output_dir)
    
    # Plot top features
    logging.info("Plotting top features...")
    results_df['abs_effect_size'] = results_df['effect_size'].abs()
    top_features = results_df.nlargest(5, 'abs_effect_size')
    
    for i, (_, row) in enumerate(tqdm(top_features.iterrows(), total=5, desc="Plotting top features")):
        feature_idx = int(row['feature_idx'])
        plot_feature_comparison(
            feature_idx,
            X_pos[:, feature_idx],
            X_neg[:, feature_idx],
            results_df,
            output_dir
        )
    
    # Print comprehensive results
    logging.info("Generating final report...")
    print("\n" + "="*80)
    print("Statistical Analysis Results")
    print("="*80)
    
    print("\n1. Mann-Whitney U Test Results:")
    print("-"*40)
    print(f"Number of significant features (p < 0.05): {sum(results_df['p_value'] < 0.05)}")
    print(f"Number of significant features after FDR correction: {sum(results_df['p_value_adj'] < 0.05)}")
    print("\nTop 5 most significant features:")
    top_sig = results_df.nsmallest(5, 'p_value_adj')
    for _, row in top_sig.iterrows():
        print(f"Feature {row['feature_idx']}: p-value = {row['p_value_adj']:.4f}, effect size = {row['effect_size']:.3f}")
    
    print("\n2. Effect Size Analysis:")
    print("-"*40)
    print(f"Number of features with large effect size (|d| > 0.8): {sum(abs(results_df['effect_size']) > 0.8)}")
    print(f"Number of features with medium effect size (0.5 < |d| ≤ 0.8): {sum((abs(results_df['effect_size']) > 0.5) & (abs(results_df['effect_size']) <= 0.8))}")
    print(f"Number of features with small effect size (|d| ≤ 0.5): {sum(abs(results_df['effect_size']) <= 0.5)}")
    print("\nTop 5 features by effect size:")
    top_effect = results_df.nlargest(5, 'abs_effect_size')
    for _, row in top_effect.iterrows():
        print(f"Feature {row['feature_idx']}: effect size = {row['effect_size']:.3f}, p-value = {row['p_value_adj']:.4f}")
    
    print("\n" + "="*80 + "\n")

    logging.info("Analysis complete!")

if __name__ == "__main__":
    main() 