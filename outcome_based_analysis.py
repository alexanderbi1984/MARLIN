"""
Outcome-based Feature Analysis

This script analyzes MARLIN features considering treatment outcomes. It performs comprehensive statistical analysis
to identify features that significantly differ between successful and unsuccessful pain treatments.

Analysis Components:
1. Feature Loading and Preprocessing
   - Loads MARLIN features from .npy files (768-dimensional vectors)
   - Calculates pre-post differences for each subject
   - Handles both clip-level (5-second clips) and video-level (~1-minute videos) analysis
   - Normalizes and pairs pre-post data for comparison

2. Statistical Analysis
   - Mann-Whitney U test for feature differences
   - Cohen's d effect size calculation
   - False Discovery Rate (FDR) correction for multiple comparisons
   - Kernel Maximum Mean Discrepancy (MMD) test for distribution differences

3. Visualization
   - Feature comparison box plots showing pre-post changes
   - Feature change heatmaps for top discriminative features
   - PCA and UMAP dimensionality reduction plots
   - Top feature distribution plots with statistical annotations
   - Interactive visualizations using Plotly

Outcome Definitions:
- Positive (Success): Pain reduction ≥4 points on the pain scale
- Negative (Failure): Pain reduction <4 points on the pain scale
- Pain scale range: 0-10 (self-reported)

Input Requirements:
1. Feature Files:
   - Directory containing .npy files with MARLIN features
   - Filename format: IMG_[ID]_clip_[N]_aligned.npy
   - Features: 768-dimensional vectors from pretrained video autoencoder
   - Clips: 5-second duration with 1-second overlap

2. Metadata File (Excel):
   - Required columns:
     * file_name: Video file identifiers (e.g., IMG_xxxx.MP4)
     * visit_type: Visit type (1st-pre, 1st-post, 2nd-pre, 2nd-post)
     * subject_id: Unique subject identifier
     * pain_level: Pain score (0-10)
     * outcome: Treatment outcome ('positive' or 'negative')
   - Optional columns:
     * creation_time: Video creation timestamp
     * duration: Video duration
     * comment: Additional notes

Output Files:
1. Statistical Results:
   - [feature_type]_clip_outcome_analysis.csv:
     * Feature-wise statistics for clip-level analysis
     * Effect sizes, p-values (raw and FDR-corrected)
     * Mean and std of changes for each outcome group
   - [feature_type]_video_outcome_analysis.csv:
     * Same as clip analysis but for video-level features
   - [feature_type]_mmd_results.txt:
     * MMD test results comparing outcome distributions

2. Visualizations:
   - Interactive Plots (HTML):
     * [feature_type]_clip/video_feature_changes_heatmap.html
     * [feature_type]_pca_visualization.html
     * [feature_type]_umap_visualization.html
     * [feature_type]_top_feature_distribution.html
     * [feature_type]_top_features_heatmap_[outcome].html
   - Static Plots (PNG):
     * [feature_type]_clip/video_feature_[N]_changes.png

Usage:
    python outcome_based_analysis.py \\
        --features_dir /path/to/features \\
        --meta_path /path/to/metadata.xlsx \\
        --output_dir outcome_analysis_results \\
        --features marlin

Dataset Statistics (as of March 2024):
- Total videos: 97
- Unique subjects: 37
- Visit distribution:
  * 1st-pre: 28 videos
  * 1st-post: 26 videos
  * 2nd-pre: 2 videos
  * 2nd-post: 13 videos
- Outcomes (complete pairs):
  * Positive (significant reduction): 18 cases
  * Negative (no significant reduction): 34 cases

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
import os
import re

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

def load_features_with_metadata(directory, metadata):
    """Load features and calculate pre-post differences for each subject."""
    features_list = []
    video_ids = []
    clip_nums = []
    subject_ids = []
    outcomes = []
    visit_types = []
    
    # First, ensure we have video_id column in metadata
    if 'video_id' not in metadata.columns:
        metadata['video_id'] = metadata['file_name'].str.replace('.MP4', '')
    
    # Only consider samples that have an outcome label
    valid_metadata = metadata[metadata['outcome'].notna()]
    
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
                if video_id is not None and video_id in valid_metadata['video_id'].values:
                    matching_row = valid_metadata[valid_metadata['video_id'] == video_id].iloc[0]
                    features_list.append(feature)
                    video_ids.append(video_id)
                    clip_nums.append(clip_num)
                    subject_ids.append(matching_row['subject_id'])
                    outcomes.append(matching_row['outcome'])
                    visit_type = 'pre' if 'pre' in matching_row['visit_type'] else 'post'
                    visit_types.append(visit_type)
                
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                continue
    
    if not features_list:
        logging.error("No features were loaded successfully")
        return None, None, None, None
    
    # Convert to numpy arrays
    features_array = np.array(features_list)
    
    # Create a DataFrame with features information
    features_df = pd.DataFrame({
        'video_id': video_ids,
        'subject_id': subject_ids,
        'clip_num': clip_nums,
        'outcome': outcomes,
        'visit_type': visit_types
    })
    
    # Calculate pre-post differences for each subject
    subject_differences = {}
    for i, row in features_df.iterrows():
        subject_id = row['subject_id']
        if subject_id not in subject_differences:
            subject_differences[subject_id] = {'pre': [], 'post': [], 'outcome': row['outcome']}
        subject_differences[subject_id][row['visit_type']].append(features_array[i])
    
    # Calculate differences and split by outcome
    clip_features_pos = []
    clip_features_neg = []
    for subject_id, data in subject_differences.items():
        if data['pre'] and data['post']:  # If we have both pre and post clips
            # Ensure we have equal numbers of pre and post clips
            n_pairs = min(len(data['pre']), len(data['post']))
            if n_pairs > 0:
                # Randomly select clips to pair
                pre_indices = np.random.choice(len(data['pre']), n_pairs, replace=False)
                post_indices = np.random.choice(len(data['post']), n_pairs, replace=False)
                
                # Calculate differences for paired clips
                for pre_idx, post_idx in zip(pre_indices, post_indices):
                    diff = data['post'][post_idx] - data['pre'][pre_idx]
                    if data['outcome'] == 'positive':
                        clip_features_pos.append(diff)
                    else:
                        clip_features_neg.append(diff)
    
    # Convert to numpy arrays
    clip_features_pos = np.array(clip_features_pos)
    clip_features_neg = np.array(clip_features_neg)
    
    # Video-level analysis: First average clips within each video, then calculate differences
    video_features_pos = []
    video_features_neg = []
    for subject_id, data in subject_differences.items():
        if data['pre'] and data['post']:  # If we have both pre and post clips
            # Average all clips within each visit
            pre_avg = np.mean(data['pre'], axis=0)
            post_avg = np.mean(data['post'], axis=0)
            diff = post_avg - pre_avg  # Calculate post-pre difference at video level
            if data['outcome'] == 'positive':
                video_features_pos.append(diff)
            else:
                video_features_neg.append(diff)
    
    # Convert to numpy arrays
    video_features_pos = np.array(video_features_pos)
    video_features_neg = np.array(video_features_neg)
    
    logging.info("\nData Loading Summary:")
    logging.info("Clip-level analysis (pre-post differences):")
    logging.info(f"Positive: {len(clip_features_pos)} clip pairs")
    logging.info(f"Negative: {len(clip_features_neg)} clip pairs")
    logging.info("\nVideo-level analysis (pre-post differences):")
    logging.info(f"Positive: {len(video_features_pos)} videos")
    logging.info(f"Negative: {len(video_features_neg)} videos")
    
    return clip_features_pos, clip_features_neg, video_features_pos, video_features_neg

def compute_effect_size(group1, group2):
    """Compute Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / s if s != 0 else np.nan

def analyze_features(features_pos, features_neg, feature_type):
    """Analyze differences in pre-post changes between positive and negative outcome groups"""
    # Convert inputs to numpy arrays if they aren't already
    features_pos = np.array(features_pos)
    features_neg = np.array(features_neg)
    
    # Check if we have any features to analyze
    if len(features_pos) == 0 or len(features_neg) == 0:
        logging.error(f"No features to analyze for {feature_type}")
        return pd.DataFrame()  # Return empty DataFrame if no features
    
    n_features = features_pos.shape[1]
    results = []
    
    for i in tqdm(range(n_features), desc=f"Analyzing {feature_type} features"):
        pos_vals = features_pos[:, i]  # Pre-post differences for positive outcomes
        neg_vals = features_neg[:, i]  # Pre-post differences for negative outcomes
        
        # Compute statistics
        t_stat, p_val = stats.ttest_ind(pos_vals, neg_vals)
        effect_size = compute_effect_size(pos_vals, neg_vals)
        
        results.append({
            'feature_idx': i,
            'effect_size': effect_size,
            'p_value': p_val,
            'pos_mean_change': np.mean(pos_vals),  # Mean change in positive group
            'neg_mean_change': np.mean(neg_vals),  # Mean change in negative group
            'pos_std_change': np.std(pos_vals),    # Std of changes in positive group
            'neg_std_change': np.std(neg_vals)     # Std of changes in negative group
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    _, p_corrected = fdrcorrection(results_df['p_value'])
    results_df['p_value_fdr'] = p_corrected
    
    return results_df

def plot_feature_comparison(feature_idx, pos_vals, neg_vals, results_df, output_dir, feature_type):
    """Create box plot comparing pre-post changes between outcomes"""
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    data = {
        'Outcome': ['Positive'] * len(pos_vals) + ['Negative'] * len(neg_vals),
        'Feature Change (Post-Pre)': np.concatenate([pos_vals, neg_vals])
    }
    df_plot = pd.DataFrame(data)
    
    # Create box plot
    sns.boxplot(x='Outcome', y='Feature Change (Post-Pre)', data=df_plot)
    
    # Add statistical annotations
    # Convert feature_idx to integer for DataFrame lookup
    result = results_df[results_df['feature_idx'] == int(feature_idx)].iloc[0]
    plt.title(f"Feature {feature_idx} Pre-Post Changes by Outcome\n" + 
              f"Effect Size: {result['effect_size']:.2f}, " +
              f"p-value (FDR): {result['p_value_fdr']:.3f}")
    
    # Add horizontal line at y=0 to show direction of change
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Save plot
    plt.savefig(Path(output_dir) / f"{feature_type}_feature_{feature_idx}_changes.png")
    plt.close()

def plot_feature_heatmap(features_pos, features_neg, results_df, output_dir, feature_type):
    """Create heatmap of feature changes"""
    # Get top features by absolute effect size
    results_df['abs_effect_size'] = results_df['effect_size'].abs()  # Create column with absolute values
    top_features = results_df.nlargest(10, 'abs_effect_size')  # Use the absolute values column
    
    # Prepare data for heatmap
    pos_changes = features_pos[:, top_features['feature_idx']]
    neg_changes = features_neg[:, top_features['feature_idx']]
    
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
        colorbar=dict(title='Feature Change (Post-Pre)'),
        zmid=0,  # Center the colormap at 0
        text=np.round(heatmap_data, 2),
        texttemplate='%{text}',
        textfont={"size": 8}
    ))
    
    fig.update_layout(
        title=f"Top 10 Feature Changes by Outcome ({feature_type})",
        xaxis_title="Feature",
        yaxis_title="Outcome"
    )
    
    fig.write_html(Path(output_dir) / f"{feature_type}_feature_changes_heatmap.html")

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

def analyze_feature_importance(X_pos, X_neg, feature_type, output_dir):
    """Analyze individual feature importance using effect size and statistical tests."""
    n_features = X_pos.shape[1]
    effect_sizes = []
    p_values = []
    
    logging.info("\nAnalyzing individual features...")
    for i in tqdm(range(n_features)):
        # Calculate Cohen's d effect size for feature changes
        d = compute_effect_size(X_pos[:, i], X_neg[:, i])
        effect_sizes.append(abs(d))
        
        # Perform Mann-Whitney U test on feature changes
        stat, p = stats.mannwhitneyu(X_pos[:, i], X_neg[:, i], alternative='two-sided')
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
        y=X_pos[:, top_idx[-1]],
        name='Positive',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker_color='blue',
        showlegend=True
    ))
    
    # Add negative distribution
    fig.add_trace(go.Box(
        y=X_neg[:, top_idx[-1]],
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
    top_features_pos = X_pos[:, top_idx]
    top_features_neg = X_neg[:, top_idx]
    
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

def parse_filename(filename):
    """Parse video ID and clip number from a MARLIN feature filename."""
    match = re.match(r'(IMG_\d+)_clip_(\d+)_aligned\.npy', filename)
    if match:
        video_id = match.group(1)
        clip_num = int(match.group(2))
        return video_id, clip_num
    return None, None

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load metadata
    metadata = load_metadata(args.meta_path)
    
    # Process each feature type
    for feature_type in args.features:
        logging.info(f"\nProcessing {feature_type} features...")
        
        # Load features and calculate pre-post differences
        clip_features_pos, clip_features_neg, video_features_pos, video_features_neg = \
            load_features_with_metadata(args.features_dir, metadata)
        
        if clip_features_pos is None:
            continue
        
        # Perform clip-level analysis
        clip_results = analyze_features(clip_features_pos, clip_features_neg, f"{feature_type}_clip")
        
        # Perform video-level analysis
        video_results = analyze_features(video_features_pos, video_features_neg, f"{feature_type}_video")
        
        # Save results
        clip_results.to_csv(output_dir / f"{feature_type}_clip_outcome_analysis.csv", index=False)
        video_results.to_csv(output_dir / f"{feature_type}_video_outcome_analysis.csv", index=False)
        
        # Create visualizations
        plot_feature_heatmap(clip_features_pos, clip_features_neg, clip_results, output_dir, f"{feature_type}_clip")
        plot_feature_heatmap(video_features_pos, video_features_neg, video_results, output_dir, f"{feature_type}_video")
        
        # Plot top features
        for level in ['clip', 'video']:
            results = clip_results if level == 'clip' else video_results
            features_pos = clip_features_pos if level == 'clip' else video_features_pos
            features_neg = clip_features_neg if level == 'clip' else video_features_neg
            
            # Create absolute effect size column and get top features
            results['abs_effect_size'] = results['effect_size'].abs()
            top_features = results.nlargest(5, 'abs_effect_size')
            
            for _, row in top_features.iterrows():
                feature_idx = int(row['feature_idx'])
                plot_feature_comparison(
                    feature_idx,
                    features_pos[:, feature_idx],
                    features_neg[:, feature_idx],
                    results,
                    output_dir,
                    f"{feature_type}_{level}"
                )
        
        # Perform MMD test
        clip_mmd, clip_pval = kernel_mmd_test(clip_features_pos, clip_features_neg)
        video_mmd, video_pval = kernel_mmd_test(video_features_pos, video_features_neg)
        
        # Print comprehensive test results
        print("\n" + "="*80)
        print(f"Statistical Analysis Results for {feature_type} Features")
        print("="*80)
        
        print("\n1. Mann-Whitney U Test Results:")
        print("-"*40)
        for level in ['clip', 'video']:
            results = clip_results if level == 'clip' else video_results
            print(f"\n{level.title()}-level analysis:")
            print(f"Number of significant features (p < 0.05): {sum(results['p_value'] < 0.05)}")
            print(f"Number of significant features after FDR correction: {sum(results['p_value_fdr'] < 0.05)}")
            print("\nTop 5 most significant features:")
            top_sig = results.nsmallest(5, 'p_value_fdr')
            for _, row in top_sig.iterrows():
                print(f"Feature {row['feature_idx']}: p-value = {row['p_value_fdr']:.4f}, effect size = {row['effect_size']:.3f}")
        
        print("\n2. Effect Size Analysis:")
        print("-"*40)
        for level in ['clip', 'video']:
            results = clip_results if level == 'clip' else video_results
            print(f"\n{level.title()}-level analysis:")
            print(f"Number of features with large effect size (|d| > 0.8): {sum(abs(results['effect_size']) > 0.8)}")
            print(f"Number of features with medium effect size (0.5 < |d| ≤ 0.8): {sum((abs(results['effect_size']) > 0.5) & (abs(results['effect_size']) <= 0.8))}")
            print(f"Number of features with small effect size (|d| ≤ 0.5): {sum(abs(results['effect_size']) <= 0.5)}")
            print("\nTop 5 features by effect size:")
            top_effect = results.nlargest(5, 'abs_effect_size')
            for _, row in top_effect.iterrows():
                print(f"Feature {row['feature_idx']}: effect size = {row['effect_size']:.3f}, p-value = {row['p_value_fdr']:.4f}")
        
        print("\n3. MMD Test Results:")
        print("-"*40)
        print(f"Clip-level MMD: {clip_mmd:.4f}, p-value: {clip_pval:.4f}")
        print(f"Video-level MMD: {video_mmd:.4f}, p-value: {video_pval:.4f}")
        
        print("\n" + "="*80 + "\n")
        
        # Save MMD results
        with open(output_dir / f"{feature_type}_mmd_results.txt", 'w') as f:
            f.write(f"Clip-level MMD: {clip_mmd:.6f}, p-value: {clip_pval:.6f}\n")
            f.write(f"Video-level MMD: {video_mmd:.6f}, p-value: {video_pval:.6f}\n")

if __name__ == "__main__":
    main() 