import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
from scipy.stats import norm
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly.graph_objects as go

# Path to the features - using a relative path
FEATURES_DIR = r"C:\pain\syracus\openface_clips\clips\multimodal_marlin_base"

def compute_cohens_d(x1, x2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    
    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    return (np.mean(x1) - np.mean(x2)) / pooled_se

def parse_filename(filename):
    """Parse video ID and clip number from a MARLIN feature filename."""
    match = re.match(r'(IMG_\d+)_clip_(\d+)_aligned\.npy', filename)
    if match:
        video_id = match.group(1)
        clip_num = int(match.group(2))
        return video_id, clip_num
    return None, None

def load_metadata(directory):
    """Load and preprocess metadata from the Excel file."""
    excel_files = [f for f in os.listdir(directory) if f.endswith('.xlsx') and f == 'meta.xlsx']
    if not excel_files:
        print("meta.xlsx not found in the directory!")
        return None
    
    excel_path = os.path.join(directory, excel_files[0])
    try:
        metadata = pd.read_excel(excel_path)
        # Remove .MP4 extension from file_name column
        metadata['video_id'] = metadata['file_name'].str.replace('.MP4', '')
        
        print(f"\nLoaded metadata from {excel_files[0]}")
        print("\nVisit type distribution:")
        print(metadata['visit_type'].value_counts())
        
        return metadata
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def load_features_with_metadata(directory, metadata):
    """Load features and combine with metadata information."""
    features_list = []
    video_ids = []
    clip_nums = []
    
    print("\nLoading features...")
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
                print(f"Error loading {filename}: {e}")
    
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
    
    print("\nData Loading Summary:")
    print(f"Total number of features: {len(features_array)}")
    print(f"Feature dimensionality: {features_array.shape[1]}")
    print("\nSample sizes by visit type:")
    print(merged_df['visit_type'].value_counts())
    
    return features_array, merged_df

def kernel_mmd_test(X, Y, kernel='rbf', num_permutations=1000):
    """
    Perform Kernel Maximum Mean Discrepancy (MMD) test.
    
    Args:
        X: Features from first distribution (pre)
        Y: Features from second distribution (post)
        kernel: Kernel type ('rbf' or 'linear')
        num_permutations: Number of permutations for p-value computation
    
    Returns:
        mmd_value: The MMD statistic
        p_value: The p-value from permutation test
    """
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    
    def rbf_kernel(X, Y):
        """RBF kernel computation"""
        X_norm = torch.sum(X ** 2, dim=1, keepdim=True)
        Y_norm = torch.sum(Y ** 2, dim=1, keepdim=True)
        dist = X_norm + Y_norm.T - 2 * torch.mm(X, Y.T)
        sigma = torch.median(dist[dist > 0])
        return torch.exp(-dist / (2 * sigma))
    
    def compute_mmd(X, Y):
        """Compute MMD value"""
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

def analyze_feature_importance(X_pre, X_post):
    """Analyze individual feature importance using effect size and statistical tests."""
    n_features = X_pre.shape[1]
    effect_sizes = []
    p_values = []
    
    print("\nAnalyzing individual features...")
    for i in tqdm(range(n_features)):
        # Calculate Cohen's d effect size
        d = compute_cohens_d(X_pre[:, i], X_post[:, i])
        effect_sizes.append(abs(d))
        
        # Perform Mann-Whitney U test
        stat, p = mannwhitneyu(X_pre[:, i], X_post[:, i], alternative='two-sided')
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
    
    # Create interactive heatmap of top features
    fig = go.Figure()
    
    # Add pre distribution
    fig.add_trace(go.Box(
        y=X_pre[:, top_idx[-1]],
        name='Pre',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        marker_color='blue',
        showlegend=True
    ))
    
    # Add post distribution
    fig.add_trace(go.Box(
        y=X_post[:, top_idx[-1]],
        name='Post',
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
    
    fig.write_html("top_feature_distribution.html")
    
    # Create heatmap of top features
    top_features_pre = X_pre[:, top_idx]
    top_features_post = X_post[:, top_idx]
    
    fig = go.Figure()
    
    # Add heatmap for pre
    fig.add_trace(go.Heatmap(
        z=top_features_pre.T,
        name='Pre',
        colorscale='Blues',
        showscale=True,
        xaxis='x',
        yaxis='y'
    ))
    
    fig.update_layout(
        title='Top 20 Most Discriminative Features Heatmap (Pre)',
        xaxis_title='Samples',
        yaxis_title='Features',
        width=1000,
        height=600
    )
    
    fig.write_html("top_features_heatmap_pre.html")
    
    # Create heatmap for post
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=top_features_post.T,
        name='Post',
        colorscale='Reds',
        showscale=True,
        xaxis='x',
        yaxis='y'
    ))
    
    fig.update_layout(
        title='Top 20 Most Discriminative Features Heatmap (Post)',
        xaxis_title='Samples',
        yaxis_title='Features',
        width=1000,
        height=600
    )
    
    fig.write_html("top_features_heatmap_post.html")
    
    print(f"\nFeature Importance Analysis:")
    print(f"Number of significant features (FDR < 0.05): {np.sum(significant_features)}")
    print(f"\nTop 5 most discriminative features:")
    for i, idx in enumerate(top_idx[-5:]):
        print(f"Feature {idx}: Effect size = {effect_sizes[idx]:.3f}, p-value = {p_values[idx]:.6f}")
    
    return effect_sizes, p_values, significant_features

def visualize_distributions(X_pre, X_post):
    """Create dimensionality reduction visualizations."""
    # Combine data for dimensionality reduction
    X_combined = np.vstack([X_pre, X_post])
    labels = ['Pre'] * len(X_pre) + ['Post'] * len(X_post)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    # Create PCA plot
    fig = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=labels,
        title='PCA Visualization of Pre vs Post Features',
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
    )
    
    fig.write_html("pca_visualization.html")
    
    # UMAP
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_combined)
    
    # Create UMAP plot
    fig = px.scatter(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        color=labels,
        title='UMAP Visualization of Pre vs Post Features',
        labels={'x': 'UMAP1', 'y': 'UMAP2'}
    )
    
    fig.write_html("umap_visualization.html")
    
    # Print explained variance ratio for PCA
    print("\nPCA Explained Variance Ratio:")
    print(f"First component: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"Second component: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_[:2]):.3f}")

def main():
    # Load metadata
    metadata = load_metadata(FEATURES_DIR)
    if metadata is None:
        return
    
    # Load features with metadata
    features, merged_df = load_features_with_metadata(FEATURES_DIR, metadata)
    
    # Filter for 1st-pre and 1st-post
    first_pre_mask = merged_df['visit_type'] == '1st-pre'
    first_post_mask = merged_df['visit_type'] == '1st-post'
    
    X_pre = features[first_pre_mask]
    X_post = features[first_post_mask]
    
    print("\nPreparing for MMD test:")
    print(f"Number of 1st-pre samples: {len(X_pre)}")
    print(f"Number of 1st-post samples: {len(X_post)}")
    print(f"Feature dimensionality: {X_pre.shape[1]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_pre_scaled = scaler.fit_transform(X_pre)
    X_post_scaled = scaler.transform(X_post)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    effect_sizes, p_values, significant_features = analyze_feature_importance(X_pre_scaled, X_post_scaled)
    
    # Create distribution visualizations
    print("\nCreating distribution visualizations...")
    visualize_distributions(X_pre_scaled, X_post_scaled)
    
    # Perform MMD test
    print("\nPerforming Kernel MMD test...")
    mmd_value, p_value = kernel_mmd_test(X_pre_scaled, X_post_scaled)
    
    print("\nResults:")
    print(f"MMD value: {mmd_value:.6f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Statistically significant difference: {p_value < 0.05}")

if __name__ == "__main__":
    main() 