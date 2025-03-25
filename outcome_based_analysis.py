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
import re
import os
import traceback

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

def parse_filename(filename):
    """Parse video ID and clip number from a MARLIN feature filename."""
    match = re.match(r'(IMG_\d+)_clip_(\d+)_aligned\.npy', filename)
    if match:
        video_id = match.group(1)
        clip_num = int(match.group(2))
        return video_id, clip_num
    return None, None

def load_metadata(meta_path):
    """Load and preprocess metadata from the Excel file."""
    try:
        metadata = pd.read_excel(meta_path)
        # Remove .MP4 extension from file_name column
        metadata['video_id'] = metadata['file_name'].str.replace('.MP4', '')
        
        logging.info(f"\nLoaded metadata from {meta_path}")
        logging.info("\nOutcome distribution:")
        logging.info(metadata['outcome'].value_counts())
        
        return metadata
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        return None

def load_features_with_metadata(directory, metadata):
    """Load features and combine with metadata information."""
    features_list = []
    video_ids = []
    clip_nums = []
    
    logging.info("\nLoading features...")
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.npy'):
            feature_path = os.path.join(directory, filename)
            try:
                feature = np.load(feature_path)
                # If feature is 3D, take mean across temporal dimension
                if len(feature.shape) == 3:
                    feature = np.mean(feature, axis=1)
                # If feature is still 2D, take mean across remaining temporal dimension
                if len(feature.shape) == 2:
                    feature = np.mean(feature, axis=0)
                
                video_id, clip_num = parse_filename(filename)
                if video_id is not None:
                    features_list.append(feature)
                    video_ids.append(video_id)
                    clip_nums.append(clip_num)
                
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
    
    # Convert to numpy arrays
    features_array = np.array(features_list)
    
    # Create a DataFrame with features information
    features_df = pd.DataFrame({
        'video_id': video_ids,
        'clip_num': clip_nums
    })
    
    # Merge with metadata
    merged_df = pd.merge(
        features_df,
        metadata,
        on='video_id',
        how='inner'
    )
    
    # Filter only the rows where we have both feature data and metadata
    valid_indices = merged_df.index
    features_array = features_array[valid_indices]
    
    logging.info("\nData Loading Summary:")
    logging.info(f"Total number of features: {len(features_array)}")
    logging.info(f"Feature dimensionality: {features_array.shape[1]}")
    logging.info("\nSample sizes by outcome:")
    logging.info(merged_df['outcome'].value_counts())
    
    return features_array, merged_df

def compute_cohens_d(x1, x2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    
    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    return (np.mean(x1) - np.mean(x2)) / pooled_se

def analyze_features(features_pos, features_neg, feature_type):
    """Analyze differences between positive and negative outcome groups"""
    n_features = features_pos.shape[1]
    results = []
    
    for i in tqdm(range(n_features), desc=f"Analyzing {feature_type} features"):
        pos_vals = features_pos[:, i]
        neg_vals = features_neg[:, i]
        
        # Compute statistics
        t_stat, p_val = stats.ttest_ind(pos_vals, neg_vals)
        effect_size = compute_cohens_d(pos_vals, neg_vals)
        
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

def aggregate_video_features(features_array, merged_df):
    """Aggregate features by video, taking the mean of all clips for each video."""
    # Group by video_id and outcome
    video_groups = merged_df.groupby(['video_id', 'outcome'])
    
    video_features = []
    video_outcomes = []
    
    for (video_id, outcome), group in video_groups:
        # Get indices for this video
        video_indices = group.index
        # Average features across all clips of this video
        video_feature = np.mean(features_array[video_indices], axis=0)
        video_features.append(video_feature)
        video_outcomes.append(outcome)
    
    return np.array(video_features), video_outcomes

def analyze_feature_stability(features_array, merged_df, significant_features, effect_sizes, output_dir, feature_type):
    """Analyze the stability of significant features across clips within each video."""
    # Get top significant features
    n_top = 5
    top_idx = np.argsort(np.abs(effect_sizes))[-n_top:][::-1]
    
    # Group by video
    video_groups = merged_df.groupby(['video_id', 'outcome'])
    
    # For each top feature
    for feat_idx in top_idx:
        # Prepare data for box plot
        video_values = []
        video_outcomes = []
        video_ids = []
        
        for (video_id, outcome), group in video_groups:
            clip_values = features_array[group.index, feat_idx]
            video_values.extend(clip_values)
            video_outcomes.extend([outcome] * len(clip_values))
            video_ids.extend([video_id] * len(clip_values))
        
        # Create DataFrame
        df_plot = pd.DataFrame({
            'Value': video_values,
            'Outcome': video_outcomes,
            'Video': video_ids
        })
        
        # Create box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Video', y='Value', hue='Outcome', data=df_plot)
        plt.xticks(rotation=45)
        plt.title(f'Feature {feat_idx} Values Across Videos\nEffect Size = {effect_sizes[feat_idx]:.3f}')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{feature_type}_feature_{feat_idx}_stability.png")
        plt.close()
        
        # Calculate within-video statistics
        stability_stats = []
        for (video_id, outcome), group in video_groups:
            clip_values = features_array[group.index, feat_idx]
            stability_stats.append({
                'video_id': video_id,
                'outcome': outcome,
                'mean': np.mean(clip_values),
                'std': np.std(clip_values),
                'n_clips': len(clip_values)
            })
        
        # Save stability statistics
        df_stats = pd.DataFrame(stability_stats)
        df_stats.to_csv(Path(output_dir) / f"{feature_type}_feature_{feat_idx}_stability_stats.csv", index=False)
        
        # Log summary statistics
        logging.info(f"\nFeature {feat_idx} Stability Analysis:")
        logging.info("Mean within-video standard deviation:")
        logging.info(f"Positive outcomes: {df_stats[df_stats['outcome'] == 'positive']['std'].mean():.3f}")
        logging.info(f"Negative outcomes: {df_stats[df_stats['outcome'] == 'negative']['std'].mean():.3f}")

def main():
    try:
        args = parse_args()
        
        logging.info("Starting analysis...")
        logging.info(f"Features directory: {args.features_dir}")
        logging.info(f"Metadata path: {args.meta_path}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")
        
        # Load metadata
        logging.info("Loading metadata...")
        metadata = load_metadata(args.meta_path)
        if metadata is None:
            logging.error("Failed to load metadata")
            return
        
        # Process each feature type
        for feature_type in args.features:
            try:
                logging.info(f"\nAnalyzing {feature_type} features...")
                
                # Load features with metadata
                features_array, merged_df = load_features_with_metadata(args.features_dir, metadata)
                
                # Clip-level analysis
                logging.info("\n=== Clip-level Analysis ===")
                
                # Split into positive and negative groups
                pos_mask = merged_df['outcome'] == 'positive'
                neg_mask = merged_df['outcome'] == 'negative'
                
                features_pos = features_array[pos_mask]
                features_neg = features_array[neg_mask]
                
                logging.info(f"Loaded {len(features_pos)} positive and {len(features_neg)} negative clips")
                logging.info(f"Feature dimensionality: {features_pos.shape[1]}")
                
                # Standardize features
                scaler = StandardScaler()
                features_pos_scaled = scaler.fit_transform(features_pos)
                features_neg_scaled = scaler.transform(features_neg)
                
                # Perform MMD test
                logging.info("\nPerforming Kernel MMD test (clip-level)...")
                mmd_value, p_value = kernel_mmd_test(features_pos_scaled, features_neg_scaled)
                logging.info(f"MMD value: {mmd_value:.6f}")
                logging.info(f"p-value: {p_value:.6f}")
                logging.info(f"Statistically significant difference: {p_value < 0.05}")
                
                # Create distribution visualizations
                logging.info("\nCreating distribution visualizations (clip-level)...")
                visualize_distributions(features_pos_scaled, features_neg_scaled, output_dir, f"{feature_type}_clip")
                
                # Analyze feature importance
                effect_sizes_clip, p_values_clip, significant_features_clip = analyze_feature_importance(
                    features_pos_scaled, 
                    features_neg_scaled, 
                    f"{feature_type}_clip",
                    output_dir
                )
                
                # Video-level analysis
                logging.info("\n=== Video-level Analysis ===")
                
                # Aggregate features by video
                video_features, video_outcomes = aggregate_video_features(features_array, merged_df)
                
                # Split into positive and negative groups
                video_pos_mask = np.array(video_outcomes) == 'positive'
                video_neg_mask = np.array(video_outcomes) == 'negative'
                
                video_features_pos = video_features[video_pos_mask]
                video_features_neg = video_features[video_neg_mask]
                
                logging.info(f"Aggregated to {len(video_features_pos)} positive and {len(video_features_neg)} negative videos")
                
                # Standardize features
                scaler_video = StandardScaler()
                video_features_pos_scaled = scaler_video.fit_transform(video_features_pos)
                video_features_neg_scaled = scaler_video.transform(video_features_neg)
                
                # Perform MMD test
                logging.info("\nPerforming Kernel MMD test (video-level)...")
                mmd_value_video, p_value_video = kernel_mmd_test(video_features_pos_scaled, video_features_neg_scaled)
                logging.info(f"MMD value: {mmd_value_video:.6f}")
                logging.info(f"p-value: {p_value_video:.6f}")
                logging.info(f"Statistically significant difference: {p_value_video < 0.05}")
                
                # Create distribution visualizations
                logging.info("\nCreating distribution visualizations (video-level)...")
                visualize_distributions(video_features_pos_scaled, video_features_neg_scaled, output_dir, f"{feature_type}_video")
                
                # Analyze feature importance
                effect_sizes_video, p_values_video, significant_features_video = analyze_feature_importance(
                    video_features_pos_scaled, 
                    video_features_neg_scaled, 
                    f"{feature_type}_video",
                    output_dir
                )
                
                # Analyze stability of significant features
                logging.info("\nAnalyzing stability of significant features...")
                analyze_feature_stability(
                    features_array,
                    merged_df,
                    significant_features_video,
                    effect_sizes_video,
                    output_dir,
                    f"{feature_type}_video"
                )
                
            except Exception as e:
                logging.error(f"Error processing feature type {feature_type}:")
                logging.error(traceback.format_exc())
    
    except Exception as e:
        logging.error("Fatal error in main:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 