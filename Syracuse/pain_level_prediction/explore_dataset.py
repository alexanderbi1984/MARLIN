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

def extract_features_and_pain_levels(dataset):
    """
    Extract features and corresponding pain levels from all videos regardless of visit type.
    
    Args:
        dataset: SyracuseDataset instance
        
    Returns:
        Dictionary containing dataset information
    """
    # Get pairs information to extract pain levels and files
    pairs_df = dataset.get_pair_info()
    
    # Collect both pre and post treatment data for analysis
    all_features = []
    all_pain_levels = []
    all_subject_ids = []
    all_visit_types = []  # Will store "pre" or "post"
    
    # Process pre-treatment data
    for idx, row in pairs_df.iterrows():
        # Load pre-treatment features
        pre_file = row['pre_file']
        pre_pain = row['pre_pain']
        
        # Get clips for pre video
        pre_clips = sorted([f for f in os.listdir(dataset.feature_dir) 
                          if f.startswith(pre_file.replace('.MP4', '_clip_')) and f.endswith('_aligned.npy')])[:14]
        
        if len(pre_clips) < 14:
            print(f"Warning: Not enough clips for {pre_file}, skipping")
            continue
        
        # Load and process pre-treatment features
        pre_features = []
        for clip in pre_clips:
            clip_path = os.path.join(dataset.feature_dir, clip)
            features = np.load(clip_path)
            
            # Check feature dimensions
            if features.shape[1] != 768:
                print(f"Warning: Clip {clip} has unexpected feature dimension, skipping")
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
            
            pre_features.append(features)
        
        if not pre_features:
            continue
            
        # Stack and average across time and clips
        pre_features = np.stack(pre_features)  # (14, 4, 768)
        pre_features_avg = np.mean(pre_features, axis=(0, 1))  # (768,)
        
        all_features.append(pre_features_avg)
        all_pain_levels.append(pre_pain)
        all_subject_ids.append(row['subject'])
        all_visit_types.append('pre')
        
        # Load post-treatment features
        post_file = row['post_file']
        post_pain = row['post_pain']
        
        # Get clips for post video
        post_clips = sorted([f for f in os.listdir(dataset.feature_dir) 
                           if f.startswith(post_file.replace('.MP4', '_clip_')) and f.endswith('_aligned.npy')])[:14]
        
        if len(post_clips) < 14:
            print(f"Warning: Not enough clips for {post_file}, skipping")
            continue
        
        # Load and process post-treatment features
        post_features = []
        for clip in post_clips:
            clip_path = os.path.join(dataset.feature_dir, clip)
            features = np.load(clip_path)
            
            # Check feature dimensions
            if features.shape[1] != 768:
                print(f"Warning: Clip {clip} has unexpected feature dimension, skipping")
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
            
            post_features.append(features)
        
        if not post_features:
            continue
            
        # Stack and average across time and clips
        post_features = np.stack(post_features)  # (14, 4, 768)
        post_features_avg = np.mean(post_features, axis=(0, 1))  # (768,)
        
        all_features.append(post_features_avg)
        all_pain_levels.append(post_pain)
        all_subject_ids.append(row['subject'])
        all_visit_types.append('post')
    
    # Convert to arrays
    features = np.array(all_features)
    pain_levels = np.array(all_pain_levels)
    subject_ids = np.array(all_subject_ids)
    visit_types = np.array(all_visit_types)
    
    return {
        'features': features,
        'pain_levels': pain_levels,
        'subject_ids': subject_ids,
        'visit_types': visit_types
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
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(pain_counts.values(), labels=[f"Level {k}" for k in pain_counts.keys()], autopct='%1.1f%%', startangle=90)
    plt.title("Pain Level Distribution")
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.savefig(output_dir / 'pain_level_pie.png')
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

def analyze_pain_levels_by_visit_type(pain_levels, visit_types, output_dir):
    """
    Analyze pain levels separated by visit type.
    
    Args:
        pain_levels: Array of pain levels
        visit_types: Array of visit types ('pre' or 'post')
        output_dir: Directory to save visualizations
    """
    print("\n=== Pain Levels by Visit Type ===")
    
    # Separate pre and post visit data
    pre_pain = pain_levels[visit_types == 'pre']
    post_pain = pain_levels[visit_types == 'post']
    
    # Calculate statistics
    print(f"Pre-treatment visits: {len(pre_pain)} samples")
    print(f"  Mean pain: {np.mean(pre_pain):.2f}")
    print(f"  Median pain: {np.median(pre_pain):.2f}")
    print(f"  Standard deviation: {np.std(pre_pain):.2f}")
    print(f"  Range: {np.min(pre_pain)} to {np.max(pre_pain)}")
    
    print(f"Post-treatment visits: {len(post_pain)} samples")
    print(f"  Mean pain: {np.mean(post_pain):.2f}")
    print(f"  Median pain: {np.median(post_pain):.2f}")
    print(f"  Standard deviation: {np.std(post_pain):.2f}")
    print(f"  Range: {np.min(post_pain)} to {np.max(post_pain)}")
    
    # Test if the difference is statistically significant
    t_stat, p_value = stats.ttest_ind(pre_pain, post_pain, equal_var=False)
    print(f"T-test for difference: t = {t_stat:.3f}, p-value = {p_value:.5f}")
    
    # Create box plot
    plt.figure(figsize=(8, 6))
    data = pd.DataFrame({
        'Pain Level': np.concatenate([pre_pain, post_pain]),
        'Visit Type': np.concatenate([['Pre'] * len(pre_pain), ['Post'] * len(post_pain)])
    })
    sns.boxplot(x='Visit Type', y='Pain Level', data=data)
    plt.title("Pain Levels by Visit Type")
    plt.ylabel("Pain Level (0-10)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_dir / 'pain_by_visit_boxplot.png')
    plt.close()
    
    # Create violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Visit Type', y='Pain Level', data=data, inner='box')
    plt.title("Pain Level Distribution by Visit Type")
    plt.ylabel("Pain Level (0-10)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_dir / 'pain_by_visit_violin.png')
    plt.close()
    
    # Create overlapping histograms
    plt.figure(figsize=(12, 6))
    sns.histplot(pre_pain, bins=range(int(np.min(pain_levels)), int(np.max(pain_levels)) + 2), 
                alpha=0.6, label='Pre-treatment', kde=True)
    sns.histplot(post_pain, bins=range(int(np.min(pain_levels)), int(np.max(pain_levels)) + 2), 
                alpha=0.6, label='Post-treatment', kde=True)
    plt.title("Pain Level Distribution by Visit Type")
    plt.xlabel("Pain Level (0-10)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'pain_by_visit_histogram.png')
    plt.close()
    
    return {
        'pre_pain': pre_pain,
        'post_pain': post_pain
    }

def analyze_pain_levels_by_subject(pain_levels, subject_ids, output_dir):
    """
    Analyze pain levels by subject.
    
    Args:
        pain_levels: Array of pain levels
        subject_ids: Array of subject IDs
        output_dir: Directory to save visualizations
    """
    print("\n=== Pain Levels by Subject ===")
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame({
        'subject_id': subject_ids,
        'pain_level': pain_levels
    })
    
    # Get subject pain statistics
    subject_stats = df.groupby('subject_id')['pain_level'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    subject_stats = subject_stats.sort_values('mean', ascending=False)
    
    # Print statistics
    print(f"Number of unique subjects: {len(subject_stats)}")
    print(f"Average samples per subject: {np.mean(subject_stats['count']):.2f}")
    print(f"Range of subject mean pain: {subject_stats['mean'].min():.2f} to {subject_stats['mean'].max():.2f}")
    
    # Plot subject means
    plt.figure(figsize=(14, 7))
    plt.bar(range(len(subject_stats)), subject_stats['mean'])
    plt.errorbar(range(len(subject_stats)), subject_stats['mean'], yerr=subject_stats['std'], fmt='none', color='red', capsize=5)
    plt.xticks(range(len(subject_stats)), subject_stats['subject_id'], rotation=90)
    plt.xlabel("Subject ID")
    plt.ylabel("Mean Pain Level")
    plt.title("Mean Pain Level by Subject (with standard deviation)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pain_by_subject.png')
    plt.close()
    
    # Calculate number of samples per subject
    plt.figure(figsize=(14, 7))
    sample_counts = df['subject_id'].value_counts().sort_index()
    plt.bar(range(len(sample_counts)), sample_counts.values)
    plt.xticks(range(len(sample_counts)), sample_counts.index, rotation=90)
    plt.xlabel("Subject ID")
    plt.ylabel("Number of Samples")
    plt.title("Number of Samples per Subject")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'samples_per_subject.png')
    plt.close()
    
    return subject_stats

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
    feature_mins = np.min(features, axis=0)
    feature_maxs = np.max(features, axis=0)
    
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Global feature mean: {np.mean(feature_means):.6f}")
    print(f"Global feature std: {np.mean(feature_stds):.6f}")
    print(f"Feature means range: {np.min(feature_means):.6f} to {np.max(feature_means):.6f}")
    print(f"Feature stds range: {np.min(feature_stds):.6f} to {np.max(feature_stds):.6f}")
    
    # Plot feature means distribution
    plt.figure(figsize=(10, 6))
    plt.hist(feature_means, bins=50)
    plt.title("Distribution of Feature Means")
    plt.xlabel("Feature Mean Value")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'feature_means_distribution.png')
    plt.close()
    
    # Plot feature standard deviations distribution
    plt.figure(figsize=(10, 6))
    plt.hist(feature_stds, bins=50)
    plt.title("Distribution of Feature Standard Deviations")
    plt.xlabel("Feature Standard Deviation")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'feature_stds_distribution.png')
    plt.close()
    
    # Create scatter plot of means vs stds
    plt.figure(figsize=(10, 8))
    plt.scatter(feature_means, feature_stds, alpha=0.5)
    plt.title("Feature Means vs Standard Deviations")
    plt.xlabel("Feature Mean")
    plt.ylabel("Feature Standard Deviation")
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'feature_means_vs_stds.png')
    plt.close()
    
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

def main():
    # Create output directory
    output_dir = Path('Syracuse/pain_level_prediction/dataset_exploration')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize dataset
    print("Initializing Syracuse dataset...")
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Get pair information
    pairs_df = dataset.get_pair_info()
    print(f"\n=== Dataset Overview ===")
    print(f"Number of pre-post pairs: {len(pairs_df)}")
    print(f"Number of unique subjects: {pairs_df['subject'].nunique()}")
    print(f"Number of 1st visits: {len(pairs_df[pairs_df['visit_number'] == '1'])}")
    print(f"Number of 2nd visits: {len(pairs_df[pairs_df['visit_number'] == '2'])}")
    
    # Extract pain level data
    print("Extracting pain levels and features for all videos...")
    data = extract_features_and_pain_levels(dataset)
    features = data['features']
    pain_levels = data['pain_levels']
    subject_ids = data['subject_ids']
    visit_types = data['visit_types']
    
    print(f"\nExtracted {len(pain_levels)} samples total")
    print(f"Features shape: {features.shape}")
    
    # Analyze pain level distribution
    print("\nAnalyzing pain level distribution...")
    pain_counts = analyze_pain_level_distribution(pain_levels, output_dir)
    
    # Analyze pain levels by visit type
    print("\nAnalyzing pain levels by visit type...")
    visit_data = analyze_pain_levels_by_visit_type(pain_levels, visit_types, output_dir)
    
    # Analyze pain levels by subject
    print("\nAnalyzing pain levels by subject...")
    subject_stats = analyze_pain_levels_by_subject(pain_levels, subject_ids, output_dir)
    
    # Analyze feature characteristics
    print("\nAnalyzing feature characteristics...")
    feature_stats = analyze_feature_characteristics(features, output_dir)
    
    # Generate summary report
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write("=== SYRACUSE PAIN LEVEL DATASET SUMMARY ===\n\n")
        
        f.write("Dataset Overview:\n")
        f.write(f"  Number of pre-post pairs: {len(pairs_df)}\n")
        f.write(f"  Number of unique subjects: {pairs_df['subject'].nunique()}\n")
        f.write(f"  Number of 1st visits: {len(pairs_df[pairs_df['visit_number'] == '1'])}\n")
        f.write(f"  Number of 2nd visits: {len(pairs_df[pairs_df['visit_number'] == '2'])}\n\n")
        
        f.write(f"Total extracted samples: {len(pain_levels)}\n")
        f.write(f"  Pre-treatment: {len(visit_data['pre_pain'])}\n")
        f.write(f"  Post-treatment: {len(visit_data['post_pain'])}\n\n")
        
        f.write("Pain Level Statistics:\n")
        f.write(f"  Range: {np.min(pain_levels)} to {np.max(pain_levels)}\n")
        f.write(f"  Mean: {np.mean(pain_levels):.2f}\n")
        f.write(f"  Median: {np.median(pain_levels):.2f}\n")
        f.write(f"  Standard deviation: {np.std(pain_levels):.2f}\n\n")
        
        f.write("Pain Level Distribution:\n")
        for level, count in pain_counts.items():
            f.write(f"  Level {level}: {count} samples ({count/len(pain_levels)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("Pre vs Post Treatment:\n")
        f.write(f"  Pre-treatment mean pain: {np.mean(visit_data['pre_pain']):.2f}\n")
        f.write(f"  Post-treatment mean pain: {np.mean(visit_data['post_pain']):.2f}\n")
        t_stat, p_value = stats.ttest_ind(visit_data['pre_pain'], visit_data['post_pain'], equal_var=False)
        f.write(f"  T-test for difference: t = {t_stat:.3f}, p-value = {p_value:.5f}\n\n")
        
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