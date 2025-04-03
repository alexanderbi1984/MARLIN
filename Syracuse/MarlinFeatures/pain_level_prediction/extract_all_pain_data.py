import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
from scipy import stats

# Add parent directory to path to import syracuse_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from syracuse_dataset import SyracuseDataset

def extract_all_videos_with_pain_levels(meta_path, feature_dir):
    """
    Extract features and corresponding pain levels from all available videos,
    regardless of whether they form complete pre-post pairs.
    
    Args:
        meta_path: Path to the meta data file
        feature_dir: Directory containing feature files
        
    Returns:
        Dictionary containing dataset information
    """
    print("Loading metadata...")
    # Load metadata directly
    meta_df = pd.read_excel(meta_path)
    
    # Convert pain_level to numeric, handling non-numeric values
    meta_df['pain_level'] = pd.to_numeric(meta_df['pain_level'], errors='coerce')
    
    # Extract visit number and type
    meta_df['visit_number'] = meta_df['visit_type'].str.extract('(\d+)')
    meta_df['visit_type'] = meta_df['visit_type'].str.extract('(pre|post)')
    
    # Filter out rows with missing pain levels
    valid_df = meta_df.dropna(subset=['pain_level'])
    print(f"Found {len(valid_df)} videos with valid pain levels out of {len(meta_df)} total videos")
    
    # Collect data from all videos with valid pain levels
    all_features = []
    all_pain_levels = []
    all_subject_ids = []
    all_visit_types = []
    all_file_names = []
    
    for idx, row in valid_df.iterrows():
        file_name = row['file_name']
        pain_level = row['pain_level']
        subject_id = row['subject_id']
        visit_type = row['visit_type']
        
        # Get clips for this video
        try:
            clips = sorted([f for f in os.listdir(feature_dir) 
                            if f.startswith(file_name.replace('.MP4', '_clip_')) and f.endswith('_aligned.npy')])[:14]
        except:
            print(f"Warning: Could not find clips for {file_name}, skipping")
            continue
        
        if len(clips) < 14:
            print(f"Warning: Not enough clips for {file_name}, found {len(clips)}, skipping")
            continue
        
        # Load and process video features
        video_features = []
        for clip in clips:
            clip_path = os.path.join(feature_dir, clip)
            try:
                features = np.load(clip_path)
            except:
                print(f"Warning: Could not load clip {clip}, skipping")
                continue
                
            # Check feature dimensions
            if features.shape[1] != 768:
                print(f"Warning: Clip {clip} has unexpected feature dimension {features.shape[1]}, skipping")
                continue
                
            # Normalize to 4 frames if needed
            if features.shape[0] != 4:
                n_frames = features.shape[0]
                if features.shape[0] > 4:
                    indices = np.linspace(0, n_frames-1, 4, dtype=int)
                    features = features[indices]
                else:
                    indices = np.linspace(0, n_frames-1, 4)
                    interpolated_features = np.zeros((4, features.shape[1]))
                    for j in range(features.shape[1]):
                        interpolated_features[:, j] = np.interp(indices, np.arange(n_frames), features[:, j])
                    features = interpolated_features
            
            video_features.append(features)
        
        if len(video_features) < 14:
            print(f"Warning: Not enough valid clips for {file_name}, found {len(video_features)}, skipping")
            continue
            
        # Stack and average across time and clips
        video_features = np.stack(video_features)  # (14, 4, 768)
        features_avg = np.mean(video_features, axis=(0, 1))  # (768,)
        
        all_features.append(features_avg)
        all_pain_levels.append(pain_level)
        all_subject_ids.append(subject_id)
        all_visit_types.append(visit_type)
        all_file_names.append(file_name)
    
    # Convert to arrays
    features = np.array(all_features)
    pain_levels = np.array(all_pain_levels)
    subject_ids = np.array(all_subject_ids)
    visit_types = np.array(all_visit_types)
    file_names = np.array(all_file_names)
    
    print(f"Successfully extracted {len(pain_levels)} videos with valid features and pain levels")
    
    return {
        'features': features,
        'pain_levels': pain_levels,
        'subject_ids': subject_ids,
        'visit_types': visit_types,
        'file_names': file_names
    }

def analyze_pain_level_distribution(pain_levels, output_dir):
    """
    Analyze the distribution of pain levels.
    
    Args:
        pain_levels: Array of pain levels
        output_dir: Directory to save visualizations
    """
    print("\n=== Pain Level Distribution ===")
    
    # Basic statistics
    min_pain = np.min(pain_levels)
    max_pain = np.max(pain_levels)
    mean_pain = np.mean(pain_levels)
    median_pain = np.median(pain_levels)
    std_pain = np.std(pain_levels)
    
    print(f"Number of samples: {len(pain_levels)}")
    print(f"Range: {min_pain} to {max_pain}")
    print(f"Mean: {mean_pain:.2f}")
    print(f"Median: {median_pain:.2f}")
    print(f"Standard deviation: {std_pain:.2f}")
    
    # Count for each pain level
    pain_counts = {}
    for level in range(int(min_pain), int(max_pain) + 1):
        count = np.sum((pain_levels >= level) & (pain_levels < level + 1))
        pain_counts[level] = count
        print(f"Pain level {level}: {count} samples ({count/len(pain_levels)*100:.1f}%)")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(pain_levels, bins=range(int(min_pain), int(max_pain) + 2), kde=True)
    plt.title("Distribution of Pain Levels")
    plt.xlabel("Pain Level (0-10)")
    plt.ylabel("Count")
    plt.xticks(range(int(min_pain), int(max_pain) + 1))
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'pain_level_histogram.png')
    plt.close()
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(pain_counts.keys(), pain_counts.values())
    plt.xlabel("Pain Level")
    plt.ylabel("Number of Samples")
    plt.title("Pain Level Distribution")
    plt.xticks(range(int(min_pain), int(max_pain) + 1))
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(pain_counts.values()):
        plt.text(list(pain_counts.keys())[i] - 0.1, v + 0.5, str(v), fontweight='bold')
    plt.savefig(output_dir / 'pain_level_bar.png')
    plt.close()
    
    return pain_counts

def analyze_feature_characteristics(features, output_dir):
    """
    Analyze the characteristics of the features.
    
    Args:
        features: Array of features, shape (n_samples, n_features)
        output_dir: Directory to save visualizations
    """
    print("\n=== Feature Characteristics ===")
    
    # Calculate global feature statistics
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Global feature mean: {np.mean(feature_means):.6f}")
    print(f"Global feature std: {np.mean(feature_stds):.6f}")
    print(f"Feature means range: {np.min(feature_means):.6f} to {np.max(feature_means):.6f}")
    print(f"Feature stds range: {np.min(feature_stds):.6f} to {np.max(feature_stds):.6f}")
    
    # Identify features with highest variance
    top_var_indices = np.argsort(-feature_stds)[:10]
    print("\nFeatures with highest variance:")
    for i, idx in enumerate(top_var_indices):
        print(f"  {i+1}. Feature {idx}: std = {feature_stds[idx]:.6f}, mean = {feature_means[idx]:.6f}")
    
    # PCA to check feature correlation
    from sklearn.decomposition import PCA
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    features_scaled = StandardScaler().fit_transform(features)
    
    # Apply PCA
    pca = PCA()
    pca.fit(features_scaled)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% explained variance')
    
    # Find number of components for 90% variance
    n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
    plt.axvline(x=n_components_90, color='g', linestyle='--', 
               label=f'{n_components_90} components for 90% variance')
    
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA: Cumulative Explained Variance')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'pca_explained_variance.png')
    plt.close()
    
    print(f"\nPCA Analysis:")
    print(f"  Number of components for 90% variance: {n_components_90}")
    print(f"  Top component explains {pca.explained_variance_ratio_[0]*100:.2f}% of variance")
    print(f"  Top 5 components explain {np.sum(pca.explained_variance_ratio_[:5])*100:.2f}% of variance")
    
    return {
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'n_components_90': n_components_90,
        'top_var_indices': top_var_indices
    }

def save_data(data, output_dir):
    """
    Save the processed data for later use.
    
    Args:
        data: Dictionary containing dataset information
        output_dir: Directory to save data
    """
    features = data['features']
    pain_levels = data['pain_levels']
    subject_ids = data['subject_ids']
    visit_types = data['visit_types']
    file_names = data['file_names']
    
    # Create a DataFrame with all information
    df = pd.DataFrame({
        'file_name': file_names,
        'subject_id': subject_ids,
        'visit_type': visit_types,
        'pain_level': pain_levels
    })
    
    # Save features as numpy array
    np.save(output_dir / 'features.npy', features)
    
    # Save metadata as CSV
    df.to_csv(output_dir / 'metadata.csv', index=False)
    
    print(f"Saved features and metadata to {output_dir}")

def main():
    # Create output directory
    output_dir = Path('Syracuse/pain_level_prediction/all_videos_data')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract all videos with pain levels
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    
    data = extract_all_videos_with_pain_levels(meta_path, feature_dir)
    features = data['features']
    pain_levels = data['pain_levels']
    subject_ids = data['subject_ids']
    visit_types = data['visit_types']
    
    print(f"\nExtracted {len(pain_levels)} samples total")
    print(f"Features shape: {features.shape}")
    
    # Analyze pain level distribution
    print("\nAnalyzing pain level distribution...")
    pain_counts = analyze_pain_level_distribution(pain_levels, output_dir)
    
    # Analyze feature characteristics
    print("\nAnalyzing feature characteristics...")
    feature_stats = analyze_feature_characteristics(features, output_dir)
    
    # Save data for later use
    save_data(data, output_dir)
    
    # Generate summary report
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write("=== SYRACUSE PAIN LEVEL DATASET SUMMARY (ALL VIDEOS) ===\n\n")
        
        f.write("Dataset Overview:\n")
        f.write(f"  Total videos with valid pain levels: {len(pain_levels)}\n")
        f.write(f"  Number of unique subjects: {len(np.unique(subject_ids))}\n")
        
        if 'pre' in visit_types and 'post' in visit_types:
            f.write(f"  Pre-treatment videos: {np.sum(visit_types == 'pre')}\n")
            f.write(f"  Post-treatment videos: {np.sum(visit_types == 'post')}\n\n")
        
        f.write("Pain Level Statistics:\n")
        f.write(f"  Range: {np.min(pain_levels)} to {np.max(pain_levels)}\n")
        f.write(f"  Mean: {np.mean(pain_levels):.2f}\n")
        f.write(f"  Median: {np.median(pain_levels):.2f}\n")
        f.write(f"  Standard deviation: {np.std(pain_levels):.2f}\n\n")
        
        f.write("Pain Level Distribution:\n")
        for level, count in pain_counts.items():
            f.write(f"  Level {level}: {count} samples ({count/len(pain_levels)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("Feature Information:\n")
        f.write(f"  Feature dimension: {features.shape[1]}\n")
        f.write(f"  Mean feature value: {np.mean(feature_stats['feature_means']):.6f}\n")
        f.write(f"  Mean feature standard deviation: {np.mean(feature_stats['feature_stds']):.6f}\n")
        f.write(f"  PCA components for 90% variance: {feature_stats['n_components_90']}\n\n")
        
        f.write("Top 10 features with highest variance:\n")
        for i, idx in enumerate(feature_stats['top_var_indices']):
            f.write(f"  {i+1}. Feature {idx}: std = {feature_stats['feature_stds'][idx]:.6f}\n")
    
    print(f"\nDataset exploration complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 