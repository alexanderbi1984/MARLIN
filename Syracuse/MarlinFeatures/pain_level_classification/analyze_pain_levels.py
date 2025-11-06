import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
from Syracuse.MarlinFeatures.syracuse_dataset import SyracuseDataset
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_pain_levels():
    # Initialize dataset
    dataset = SyracuseDataset(
        meta_path='/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx',
        feature_dir='/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    )
    
    # Get meta data
    meta_df = dataset.meta_df
    
    # Basic statistics about pain levels
    print("\n=== Pain Level Statistics ===")
    print(meta_df['pain_level'].describe())
    
    # Count of videos with valid pain levels
    valid_pain_count = meta_df['pain_level'].notna().sum()
    print(f"\nNumber of videos with valid pain levels: {valid_pain_count}")
    
    # Distribution of pain levels
    print("\n=== Pain Level Distribution ===")
    pain_dist = meta_df['pain_level'].value_counts().sort_index()
    print(pain_dist)
    
    # Feature statistics
    print("\n=== Feature Statistics ===")
    feature_stats = dataset.get_feature_statistics()
    print(f"Number of pre-post pairs: {feature_stats['num_pairs']}")
    print(f"Number of clips per video: {feature_stats['num_clips_per_video']}")
    print(f"Feature dimension: {feature_stats['feature_dimension']}")
    
    # Load all features and pain levels
    pre_features, post_features, changes = dataset.get_all_features()
    
    # Get pain levels for all videos
    pain_levels = []
    for pair in dataset.pairs:
        pre_pain = pair['pre_pain']
        post_pain = pair['post_pain']
        if pd.notna(pre_pain):
            pain_levels.append(pre_pain)
        if pd.notna(post_pain):
            pain_levels.append(post_pain)
    
    pain_levels = np.array(pain_levels)
    
    print("\n=== Pain Level Statistics for Feature Set ===")
    print(f"Number of pain level measurements: {len(pain_levels)}")
    print(f"Mean pain level: {np.mean(pain_levels):.2f}")
    print(f"Median pain level: {np.median(pain_levels):.2f}")
    print(f"Standard deviation: {np.std(pain_levels):.2f}")
    
    # Plot pain level distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pain_levels, bins=11)
    plt.title('Distribution of Pain Levels')
    plt.xlabel('Pain Level')
    plt.ylabel('Count')
    plt.savefig('pain_level_distribution.png')
    plt.close()

if __name__ == "__main__":
    analyze_pain_levels() 