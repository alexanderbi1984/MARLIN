"""
K-means Clustering Analysis for Treatment Outcome Prediction

This script implements k-means clustering (k=2) to analyze treatment outcomes using MARLIN features.
It uses the top 3 features based on effect size from video-level analysis to cluster patients
into two groups, which are then compared with the ground truth outcomes.

Key Features:
1. Feature Selection
   - Selects top 3 features based on effect size from video-level analysis
   - Features are chosen based on absolute effect size in pre-post treatment differences
   - Uses the same feature selection approach as binary classification

2. Data Processing
   - Uses SyracuseDataset class for proper feature loading and processing
   - Handles 3D features by taking means across temporal dimensions
   - Averages features across multiple clips for each video
   - Computes pre-post differences for selected features
   - Handles missing values and standardizes features

3. Clustering Analysis
   - Applies k-means clustering with k=2
   - Compares cluster assignments with ground truth outcomes
   - Evaluates clustering quality using silhouette score and adjusted rand index
   - Visualizes clusters in 3D space

4. Visualization
   - Creates 3D scatter plots of clusters
   - Generates confusion matrix comparing clusters with outcomes
   - Saves results and plots to specified output directory

Usage:
    python kmeans_clustering.py

Input Requirements:
1. Feature Files:
   - Directory containing .npy files with MARLIN features
   - Filename format: IMG_[ID]_clip_[N]_aligned.npy
   - Features should be pre-computed using MARLIN model
   - Multiple clips per video are supported and averaged

2. Metadata File (Excel):
   - Required columns:
     * subject_id: Unique subject identifier
     * file_name: Video file name (e.g., "IMG_123.MP4")
     * visit_type: Visit type (e.g., '1st-pre', '1st-post')
     * outcome: Treatment outcome ('positive' or 'negative')

3. Analysis Results:
   - outcome_analysis_results/marlin_video_outcome_analysis.csv
   - Contains effect sizes and p-values for feature selection

Output:
1. Clustering Results:
   - clustering_results.csv: Summary of clustering performance
   - clustering_results.png: Visualization of clusters
   - confusion_matrix.png: Comparison with ground truth

Author: Nan Bi
Date: 2025
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from syracuse_dataset import SyracuseDataset

# Constants
FEATURES_DIR = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
META_PATH = os.path.join(FEATURES_DIR, "meta_with_outcomes.xlsx")
OUTPUT_DIR = "Syracuse/clustering_results"

# Global variable for selected features
SELECTED_FEATURES = None

def get_selected_features(num_features=3):
    """Get top features based on effect size from video-level analysis."""
    # Read the video-level analysis results
    analysis_path = os.path.join('outcome_analysis_results', 'marlin_video_outcome_analysis.csv')
    df = pd.read_csv(analysis_path)
    
    # Sort by absolute effect size
    df['abs_effect_size'] = df['effect_size'].abs()
    df_sorted = df.sort_values('abs_effect_size', ascending=False)
    
    # Get top num_features
    top_features = df_sorted.head(num_features)
    
    # Convert to list of tuples (feature_idx, effect_size, p_value)
    selected_features = []
    for _, row in top_features.iterrows():
        selected_features.append((
            int(row['feature_idx']),
            float(row['effect_size']),
            float(row['p_value'])
        ))
    
    print(f"\nSelected top {num_features} features:")
    for idx, effect_size, p_value in selected_features:
        print(f"Feature {idx}: effect_size = {effect_size:.3f}, p_value = {p_value:.6f}")
    
    return selected_features

def prepare_data(dataset, metadata_df):
    """Prepare feature data and labels for clustering using SyracuseDataset."""
    try:
        print("\nStarting prepare_data function...")
        
        # Get all features and changes
        print("Getting all features...")
        pre_features, post_features, changes = dataset.get_all_features()
        
        print("\nDebug: Feature shapes")
        print(f"pre_features shape: {pre_features.shape}")
        print(f"post_features shape: {post_features.shape}")
        print(f"changes shape: {changes.shape}")
        
        # Get pair information
        print("\nGetting pair information...")
        pair_info = dataset.get_pair_info()
        print(f"Debug: Pair info shape: {pair_info.shape}")
        print("\nDebug: First few rows of pair_info:")
        print(pair_info.head())
        
        # Create feature matrix for selected features
        n_samples = len(pair_info)
        n_features = len(SELECTED_FEATURES)
        print(f"\nDebug: Creating feature matrix with shape ({n_samples}, {n_features})")
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        
        print("\nComputing feature differences...")
        # Process each pair
        for i, (_, row) in enumerate(pair_info.iterrows()):
            try:
                print(f"\nDebug: Processing pair {i}")
                # Get pre and post features for this pair
                pre_feat = pre_features[i]  # Shape: (14, 4, 768)
                post_feat = post_features[i]  # Shape: (14, 4, 768)
                
                print(f"pre_feat shape: {pre_feat.shape}")
                print(f"post_feat shape: {post_feat.shape}")
                
                # Take mean across temporal dimensions
                pre_feat = np.mean(pre_feat, axis=(0, 1))  # Shape: (768,)
                post_feat = np.mean(post_feat, axis=(0, 1))  # Shape: (768,)
                
                print(f"After mean - pre_feat shape: {pre_feat.shape}")
                print(f"After mean - post_feat shape: {post_feat.shape}")
                
                # Calculate pre-post differences for selected features
                for j, (feature_idx, _, _) in enumerate(SELECTED_FEATURES):
                    X[i, j] = post_feat[feature_idx] - pre_feat[feature_idx]
                
                # Get outcome from metadata
                subject_data = metadata_df[metadata_df['subject_id'] == row['subject']]
                print(f"Debug: Subject {row['subject']} data shape: {subject_data.shape}")
                outcome = subject_data['outcome'].iloc[0]
                y[i] = 1 if outcome == 'positive' else 0
                
            except Exception as e:
                print(f"Error processing pair {i}: {str(e)}")
                print(f"Row data: {row}")
                raise
        
        print("\nDebug: Final shapes")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        # Remove samples with missing features
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print("\nDebug: After removing missing values")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        # Print class distribution
        y_int = y.astype(int)
        counts = np.bincount(y_int)
        print("\nClass distribution:")
        print(f"Negative (0): {counts[0]} samples")
        print(f"Positive (1): {counts[1]} samples")
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("\nFeatures have been standardized (mean=0, std=1)")
        
        return X, y
        
    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        raise

def perform_clustering(X, y):
    """Perform k-means clustering and evaluate results.
    
    Args:
        X: Feature matrix
        y: True labels
        
    Returns:
        cluster_labels: Predicted cluster assignments
        metrics: Dictionary of clustering metrics
    """
    # Initialize and fit k-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate clustering metrics
    silhouette = silhouette_score(X, cluster_labels)
    ari = adjusted_rand_score(y, cluster_labels)
    
    # Create confusion matrix
    cm = confusion_matrix(y, cluster_labels)
    
    # Calculate accuracy (assuming cluster 0 corresponds to negative outcome)
    accuracy = np.mean(cluster_labels == y)
    
    metrics = {
        'silhouette_score': silhouette,
        'adjusted_rand_index': ari,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }
    
    return cluster_labels, metrics

def plot_results(X, y, cluster_labels, metrics, output_dir):
    """Plot and save clustering results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 2 subplots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: 3D scatter plot of clusters
    ax = plt.subplot(1, 2, 1, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                        c=cluster_labels, cmap='viridis',
                        alpha=0.6)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('K-means Clustering Results')
    plt.colorbar(scatter)
    
    # Plot 2: Confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Outcome')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_results.png'))
    plt.close()

def main():
    """Main function to run the clustering pipeline."""
    try:
        print("Starting main function...")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Get selected features (top 3)
        print("Getting selected features...")
        global SELECTED_FEATURES
        SELECTED_FEATURES = get_selected_features(num_features=3)
        
        # Initialize dataset
        print("Initializing dataset...")
        dataset = SyracuseDataset(META_PATH, FEATURES_DIR)
        
        # Load metadata
        print("Loading metadata...")
        metadata_df = pd.read_excel(META_PATH)
        metadata_df = metadata_df[metadata_df['outcome'].notna()]
        print(f"Loaded {len(metadata_df)} samples with outcomes")
        
        # Prepare data
        print("Preparing feature data...")
        X, y = prepare_data(dataset, metadata_df)
        print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        
        # Perform clustering
        print("\nPerforming k-means clustering...")
        cluster_labels, metrics = perform_clustering(X, y)
        
        # Print results
        print("\nClustering Results:")
        print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        
        # Save results
        results = {
            'silhouette_score': metrics['silhouette_score'],
            'adjusted_rand_index': metrics['adjusted_rand_index'],
            'accuracy': metrics['accuracy'],
            'n_samples': len(y),
            'n_features': len(SELECTED_FEATURES),
            'selected_features': [feature_idx for feature_idx, _, _ in SELECTED_FEATURES]
        }
        
        pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR, 'clustering_results.csv'), index=False)
        
        # Plot results
        print("\nPlotting results...")
        plot_results(X, y, cluster_labels, metrics, OUTPUT_DIR)
        print(f"Results saved to {OUTPUT_DIR}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {e}")
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 