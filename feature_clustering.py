import numpy as np
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import umap
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import re
import argparse

# Path to the features
FEATURES_DIR = r"C:\pain\syracus\openface_clips\clips\multimodal_marlin_base"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cluster MARLIN features and analyze distributions.')
    parser.add_argument(
        '--n_clusters', 
        type=int, 
        default=5,
        help='Number of clusters for K-means or Hierarchical clustering (default: 5)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['kmeans', 'hierarchical', 'dbscan'],
        default='kmeans',
        help='Clustering model to use: kmeans, hierarchical, or dbscan (default: kmeans)'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=0.5,
        help='Epsilon parameter for DBSCAN (default: 0.5)'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=5,
        help='Min samples parameter for DBSCAN (default: 5)'
    )
    return parser.parse_args()

def load_metadata(directory):
    """Load and preprocess metadata from an Excel file containing pain levels and subject information.

    This function reads a 'meta.xlsx' file from the specified directory and processes it for use in
    clustering analysis. It performs the following operations:
    1. Loads the Excel file
    2. Removes .MP4 extension from video filenames
    3. Converts pain levels to numeric values
    4. Prints summary statistics of the metadata

    Args:
        directory (str): Path to the directory containing 'meta.xlsx'

    Returns:
        pandas.DataFrame or None: A DataFrame containing the processed metadata with columns:
            - video_id: Video identifier (IMG_XXXX format)
            - pain_level: Numeric pain level values
            - subject_id: Subject identifier
            - Other columns from the original Excel file
            Returns None if the file is not found or there's an error loading it.

    Prints:
        - Available columns in the metadata
        - Total number of entries
        - Pain level distribution
        - Subject ID distribution
    """
    excel_files = [f for f in os.listdir(directory) if f.endswith('.xlsx') and f == 'meta.xlsx']
    if not excel_files:
        print("meta.xlsx not found in the directory!")
        return None
    
    excel_path = os.path.join(directory, excel_files[0])
    try:
        metadata = pd.read_excel(excel_path)
        # Remove .MP4 extension from file_name column
        metadata['video_id'] = metadata['file_name'].str.replace('.MP4', '')
        
        # Convert pain_level to numeric, non-numeric values will become NaN
        metadata['pain_level'] = pd.to_numeric(metadata['pain_level'], errors='coerce')
        
        print(f"Loaded metadata from {excel_files[0]}")
        # Print the column names to verify the structure
        print("\nAvailable columns in metadata:")
        print(metadata.columns.tolist())
        # Print some basic statistics
        print("\nMetadata summary:")
        print(f"Total entries: {len(metadata)}")
        print("\nPain level distribution (numeric only):")
        print(metadata['pain_level'].value_counts(dropna=False).sort_index())
        print("\nSubject ID distribution:")
        print(metadata['subject_id'].value_counts(dropna=False))
        return metadata
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def parse_filename(filename):
    """Parse video ID and clip number from a MARLIN feature filename.

    Extracts the video identifier and clip number from filenames following the pattern:
    'IMG_XXXX_clip_YYY_aligned.npy' where XXXX is the video ID and YYY is the clip number.

    Args:
        filename (str): Name of the .npy file to parse

    Returns:
        tuple: (video_id, clip_num) where:
            - video_id (str): Video identifier in IMG_XXXX format
            - clip_num (int): Clip number
            Returns (None, None) if the filename doesn't match the expected pattern
    """
    # Extract video ID (IMG_XXXX) and clip number (XXX)
    match = re.match(r'(IMG_\d+)_clip_(\d+)_aligned\.npy', filename)
    if match:
        video_id = match.group(1)
        clip_num = int(match.group(2))
        return video_id, clip_num
    return None, None

def load_features(directory):
    """Load and preprocess MARLIN features from .npy files in the specified directory.

    This function loads all .npy files containing MARLIN features and processes them for clustering.
    For each feature file, it:
    1. Loads the .npy file
    2. Handles different dimensional features (2D or 3D) by taking means across temporal dimensions
    3. Extracts video ID and clip number from the filename
    4. Collects features and metadata for successful loads

    Args:
        directory (str): Path to the directory containing .npy feature files

    Returns:
        tuple: (features, filenames, video_ids, clip_nums) where:
            - features (np.ndarray): Array of processed features, shape (n_samples, n_features)
            - filenames (list): List of original filenames
            - video_ids (list): List of video identifiers
            - clip_nums (list): List of clip numbers

    Prints:
        - Progress bar during loading
        - Total number of clips loaded
        - Number of unique videos
        - Average clips per video
        - Maximum and minimum clip numbers
        - Any errors encountered during loading
    """
    features_list = []
    filenames = []
    video_ids = []
    clip_nums = []
    
    print("Loading features...")
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
                    filenames.append(filename)
                    video_ids.append(video_id)
                    clip_nums.append(clip_num)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Print statistics about the loaded features
    print("\nFeature Loading Statistics:")
    print("=" * 50)
    print(f"Total number of clips loaded: {len(features_list)}")
    print(f"Number of unique videos: {len(set(video_ids))}")
    print(f"Average clips per video: {len(features_list) / len(set(video_ids)):.2f}")
    print(f"Maximum clip number: {max(clip_nums)}")
    print(f"Minimum clip number: {min(clip_nums)}")
    
    return np.array(features_list), filenames, video_ids, clip_nums

def create_pain_level_bins(pain_levels, n_clusters):
    """Create pain level bins based on the number of clusters for analysis.

    Creates evenly spaced bins for pain levels to analyze pain level distribution
    across clusters. The bins are created to match the number of clusters for
    easier comparison.

    Args:
        pain_levels (pandas.Series): Series containing pain level values
        n_clusters (int): Number of clusters used in the clustering

    Returns:
        tuple: (bins, labels) where:
            - bins (np.ndarray): Array of bin edges
            - labels (list): List of string labels for the bins
            Returns (None, None) if no valid pain levels are found
    """
    # Remove NaN values
    valid_pain_levels = pain_levels.dropna()
    
    if len(valid_pain_levels) == 0:
        return None, None
    
    # Create bins based on number of clusters
    min_pain = valid_pain_levels.min()
    max_pain = valid_pain_levels.max()
    
    # Create bin edges
    bins = np.linspace(min_pain - 0.1, max_pain + 0.1, n_clusters + 1)
    
    # Create labels for the bins
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    return bins, labels

def analyze_pain_and_subject_distribution(cluster_labels, video_ids, clip_nums, metadata, n_clusters):
    """Analyze and visualize the distribution of pain levels and subjects across clusters.

    This function creates detailed analyses and visualizations of how pain levels and subjects
    are distributed across the clusters, including:
    1. Pain level distribution counts and percentages
    2. Subject distribution across clusters
    3. Multiple interactive heatmap visualizations

    Args:
        cluster_labels (np.ndarray): Array of cluster assignments
        video_ids (list): List of video identifiers
        clip_nums (list): List of clip numbers
        metadata (pandas.DataFrame): DataFrame containing pain and subject information
        n_clusters (int): Number of clusters used in the clustering

    Creates HTML files:
        - pain_level_distribution_counts_{n_clusters}clusters.html
        - pain_level_distribution_within_clusters_{n_clusters}clusters.html
        - cluster_distribution_within_pain_levels_{n_clusters}clusters.html
        - subject_distribution_{n_clusters}clusters.html

    Returns:
        pandas.DataFrame: Merged DataFrame containing cluster assignments and metadata
    """
    if metadata is None:
        return
    
    # Create a DataFrame with clustering results
    cluster_df = pd.DataFrame({
        'video_id': video_ids,
        'clip_num': clip_nums,
        'cluster': cluster_labels
    })
    
    # Merge with metadata
    merged_df = pd.merge(
        cluster_df,
        metadata,
        left_on=['video_id'],
        right_on=['video_id'],
        how='left'
    )
    
    # Create pain level bins
    bins, bin_labels = create_pain_level_bins(merged_df['pain_level'], n_clusters)
    if bins is not None:
        merged_df['pain_level_binned'] = pd.cut(merged_df['pain_level'], 
                                               bins=bins, 
                                               labels=bin_labels, 
                                               include_lowest=True)
    else:
        print("No valid pain levels found for binning!")
        return merged_df
    
    # Analyze pain level distribution
    print("\nPain Level Distribution Across Clusters:")
    print("=" * 50)
    
    # Get counts
    pain_dist = merged_df.pivot_table(
        index='cluster',
        columns='pain_level_binned',
        values='clip_num',
        aggfunc='count',
        fill_value=0
    )
    print("\nPain level counts per cluster:")
    print(pain_dist)
    
    # Calculate percentage of pain levels within each cluster (row-wise)
    pain_dist_pct_within_cluster = pain_dist.div(pain_dist.sum(axis=1), axis=0) * 100
    print("\nPain level distribution (%) within each cluster (row-wise):")
    print(pain_dist_pct_within_cluster.round(2))
    
    # Calculate percentage of clusters within each pain level (column-wise)
    pain_dist_pct_within_pain = pain_dist.div(pain_dist.sum(axis=0), axis=1) * 100
    print("\nCluster distribution (%) within each pain level (column-wise):")
    print(pain_dist_pct_within_pain.round(2))
    
    # Create three heatmaps: counts and both percentage views
    # 1. Pain level counts heatmap
    fig_pain_counts = go.Figure(data=go.Heatmap(
        z=pain_dist.values,
        x=[str(x) for x in pain_dist.columns],
        y=[f"Cluster {i}" for i in pain_dist.index],
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig_pain_counts.update_layout(
        title=f'Pain Level Distribution Across Clusters (Counts) - {n_clusters} Clusters',
        xaxis_title='Pain Level Range',
        yaxis_title='Clusters',
        width=1000,
        height=600
    )
    fig_pain_counts.write_html(f"pain_level_distribution_counts_{n_clusters}clusters.html")
    
    # 2. Pain level percentages within clusters heatmap (row-wise)
    fig_pain_pct_within_cluster = go.Figure(data=go.Heatmap(
        z=pain_dist_pct_within_cluster.values,
        x=[str(x) for x in pain_dist_pct_within_cluster.columns],
        y=[f"Cluster {i}" for i in pain_dist_pct_within_cluster.index],
        colorscale='RdBu',
        hoverongaps=False,
        text=np.round(pain_dist_pct_within_cluster.values, 1),
        hovertemplate="Cluster %{y}<br>Pain Level: %{x}<br>% of Cluster: %{z:.1f}%<extra></extra>"
    ))
    
    fig_pain_pct_within_cluster.update_layout(
        title=f'Pain Level Distribution (%) within Each Cluster - {n_clusters} Clusters',
        xaxis_title='Pain Level Range',
        yaxis_title='Clusters',
        width=1000,
        height=600
    )
    fig_pain_pct_within_cluster.write_html(f"pain_level_distribution_within_clusters_{n_clusters}clusters.html")
    
    # 3. Cluster percentages within pain levels heatmap (column-wise)
    fig_pain_pct_within_pain = go.Figure(data=go.Heatmap(
        z=pain_dist_pct_within_pain.values,
        x=[str(x) for x in pain_dist_pct_within_pain.columns],
        y=[f"Cluster {i}" for i in pain_dist_pct_within_pain.index],
        colorscale='RdBu',
        hoverongaps=False,
        text=np.round(pain_dist_pct_within_pain.values, 1),
        hovertemplate="Cluster %{y}<br>Pain Level: %{x}<br>% of Pain Level: %{z:.1f}%<extra></extra>"
    ))
    
    fig_pain_pct_within_pain.update_layout(
        title=f'Cluster Distribution (%) within Each Pain Level - {n_clusters} Clusters',
        xaxis_title='Pain Level Range',
        yaxis_title='Clusters',
        width=1000,
        height=600
    )
    fig_pain_pct_within_pain.write_html(f"cluster_distribution_within_pain_levels_{n_clusters}clusters.html")
    
    # Subject distribution
    print("\nSubject Distribution Across Clusters:")
    print("=" * 50)
    subject_dist = merged_df.pivot_table(
        index='cluster',
        columns='subject_id',
        values='clip_num',
        aggfunc='count',
        fill_value=0
    )
    print("\nSubject counts per cluster:")
    print(subject_dist)
    
    # Subject distribution heatmap
    fig_subject = go.Figure(data=go.Heatmap(
        z=subject_dist.values,
        x=[f"Subject {i}" for i in subject_dist.columns],
        y=[f"Cluster {i}" for i in subject_dist.index],
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig_subject.update_layout(
        title=f'Subject Distribution Across Clusters - {n_clusters} Clusters',
        xaxis_title='Subject ID',
        yaxis_title='Clusters',
        width=1200,
        height=600
    )
    fig_subject.write_html(f"subject_distribution_{n_clusters}clusters.html")
    
    return merged_df

def analyze_cluster_composition(cluster_labels, video_ids, clip_nums, n_clusters):
    """Analyze and visualize the composition of each cluster in terms of video clips.

    This function provides detailed analysis of how video clips are distributed across
    clusters, including:
    1. Detailed per-cluster breakdown of video clips
    2. Interactive heatmap visualization of clip distribution
    3. Statistics about cluster composition

    Args:
        cluster_labels (np.ndarray): Array of cluster assignments
        video_ids (list): List of video identifiers
        clip_nums (list): List of clip numbers
        n_clusters (int): Number of clusters used in the clustering

    Creates HTML file:
        - cluster_composition_heatmap_{n_clusters}clusters.html

    Returns:
        dict: Nested dictionary containing cluster statistics:
            {cluster_label: {video_id: [clip_numbers]}}
    """
    # Create a dictionary to store cluster statistics
    cluster_stats = defaultdict(lambda: defaultdict(list))
    
    # Collect clips for each video in each cluster
    for cluster_label, video_id, clip_num in zip(cluster_labels, video_ids, clip_nums):
        cluster_stats[cluster_label][video_id].append(clip_num)
    
    # Print detailed analysis
    print("\nDetailed Cluster Analysis:")
    print("=" * 50)
    
    # Create a DataFrame to store the distribution
    video_counts = defaultdict(lambda: defaultdict(int))
    
    for cluster_label in sorted(cluster_stats.keys()):
        print(f"\nCluster {cluster_label}:")
        print("-" * 30)
        
        # Count clips per video in this cluster
        videos_in_cluster = cluster_stats[cluster_label]
        for video_id, clips in sorted(videos_in_cluster.items()):
            num_clips = len(clips)
            video_counts[video_id][f'Cluster_{cluster_label}'] = num_clips
            print(f"{video_id}: {num_clips} clips (clip numbers: {sorted(clips)})")
    
    # Create a DataFrame for the heatmap
    df_counts = pd.DataFrame.from_dict(video_counts, orient='index').fillna(0)
    
    # Create a heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=df_counts.values,
        x=df_counts.columns,
        y=df_counts.index,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'Distribution of Video Clips Across Clusters - {n_clusters} Clusters',
        xaxis_title='Clusters',
        yaxis_title='Video IDs',
        width=1000,
        height=max(600, len(video_counts) * 20)  # Adjust height based on number of videos
    )
    
    # Save the heatmap
    fig.write_html(f"cluster_composition_heatmap_{n_clusters}clusters.html")
    print(f"\nCluster composition heatmap saved as 'cluster_composition_heatmap_{n_clusters}clusters.html'")
    
    return cluster_stats

def perform_clustering(features_scaled, args):
    """Perform clustering using the specified clustering algorithm.

    Supports multiple clustering methods:
    1. K-means: Traditional centroid-based clustering
    2. Hierarchical: Agglomerative clustering with specified number of clusters
    3. DBSCAN: Density-based clustering with eps and min_samples parameters

    Args:
        features_scaled (np.ndarray): Standardized feature array
        args (argparse.Namespace): Command line arguments containing:
            - model: Clustering method ('kmeans', 'hierarchical', or 'dbscan')
            - n_clusters: Number of clusters for kmeans/hierarchical
            - eps: Epsilon parameter for DBSCAN
            - min_samples: Minimum samples parameter for DBSCAN

    Returns:
        tuple: (cluster_labels, model) where:
            - cluster_labels (np.ndarray): Array of cluster assignments
            - model: The fitted clustering model object
    """
    print(f"Performing {args.model} clustering...")
    
    if args.model == 'kmeans':
        model = KMeans(n_clusters=args.n_clusters, random_state=42)
        cluster_labels = model.fit_predict(features_scaled)
    
    elif args.model == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=args.n_clusters)
        cluster_labels = model.fit_predict(features_scaled)
    
    elif args.model == 'dbscan':
        model = DBSCAN(eps=args.eps, min_samples=args.min_samples)
        cluster_labels = model.fit_predict(features_scaled)
        # DBSCAN uses -1 for noise points, we'll keep this for analysis
    
    return cluster_labels, model

def cluster_and_visualize(features, filenames, video_ids, metadata=None, args=None):
    """Perform clustering on MARLIN features and create interactive visualizations.

    This function handles the complete clustering and visualization pipeline:
    1. Feature standardization
    2. Clustering using specified method
    3. UMAP dimensionality reduction
    4. Interactive scatter plot creation with metadata integration

    Args:
        features (np.ndarray): Array of MARLIN features
        filenames (list): List of original filenames
        video_ids (list): List of video identifiers
        metadata (pandas.DataFrame, optional): DataFrame containing pain and subject information
        args (argparse.Namespace): Command line arguments for clustering parameters

    Creates HTML file:
        - marlin_features_clustering_{model}_{n_clusters}clusters.html
        For DBSCAN: includes eps and min_samples in filename

    Returns:
        np.ndarray: Array of cluster assignments
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    cluster_labels, model = perform_clustering(features_scaled, args)
    
    # Perform dimensionality reduction with UMAP
    print("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features_scaled)
    
    # Create interactive scatter plot
    df = {
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Cluster': [f'Cluster {i}' if i != -1 else 'Noise' for i in cluster_labels],
        'Video': video_ids,
        'Filename': filenames
    }
    
    # Add pain level and subject information if available
    if metadata is not None:
        temp_df = pd.DataFrame(df)
        merged_df = pd.merge(
            temp_df,
            metadata,
            left_on=['Video'],
            right_on=['video_id'],
            how='left'
        )
        df['Pain Level'] = merged_df['pain_level'].fillna('Unknown')
        df['Subject'] = merged_df['subject_id']
    
    fig = px.scatter(
        df,
        x='UMAP1',
        y='UMAP2',
        color='Cluster',
        hover_data=['Video', 'Filename'] + (['Pain Level', 'Subject'] if metadata is not None else []),
        title=f'MARLIN Features Clustering Visualization - {args.model} ({args.n_clusters} clusters)'
    )
    
    # Update layout for better visualization
    fig.update_layout(
        plot_bgcolor='white',
        width=1200,
        height=800
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Save the plot
    output_name = f"marlin_features_clustering_{args.model}_{args.n_clusters}clusters"
    if args.model == 'dbscan':
        output_name += f"_eps{args.eps}_min{args.min_samples}"
    fig.write_html(f"{output_name}.html")
    print(f"Visualization saved as '{output_name}.html'")
    
    return cluster_labels

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load metadata
    metadata = load_metadata(FEATURES_DIR)
    
    # Load features
    features, filenames, video_ids, clip_nums = load_features(FEATURES_DIR)
    print(f"Loaded {len(features)} features")
    
    # Perform clustering and visualization
    cluster_labels = cluster_and_visualize(features, filenames, video_ids, metadata, args)
    
    # Analyze cluster composition
    cluster_stats = analyze_cluster_composition(cluster_labels, video_ids, clip_nums, args.n_clusters)
    
    # Analyze pain level and subject distribution
    if metadata is not None:
        analyze_pain_and_subject_distribution(cluster_labels, video_ids, clip_nums, metadata, args.n_clusters)
    
    # Print overall statistics
    print("\nOverall Clustering Statistics:")
    print("=" * 50)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"Noise points: {count} samples")
        else:
            print(f"Cluster {label}: {count} samples")
    
    # Print number of unique videos
    unique_videos = len(set(video_ids))
    print(f"\nTotal number of unique videos: {unique_videos}")

if __name__ == "__main__":
    main() 