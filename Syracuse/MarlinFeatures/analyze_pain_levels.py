"""
Analysis of Pre and Post Treatment Pain Levels in Syracuse Pain Study

This script analyzes and visualizes the pain levels before and after treatment in the Syracuse pain study.
It provides separate statistical analyses for pre and post treatment measurements, as well as
visualizations to help understand the treatment effects.

Key Features:
- Separate analysis of pre and post treatment pain levels
- Statistical summaries for both pre and post measurements
- Three visualizations:
  1. Pre-treatment pain distribution histogram
  2. Post-treatment pain distribution histogram
  3. Scatter plot of pre vs post pain levels with improvement analysis

The pain levels are measured on a scale of 0-10, where:
- 0 represents no pain
- 10 represents maximum pain

Output:
- Generates two plots:
  1. 'pain_levels_distribution.png': Side-by-side histograms of pre and post pain levels
  2. 'pre_vs_post_pain.png': Scatter plot showing treatment effects
- Prints detailed statistical summaries

Usage:
    python analyze_pain_levels.py

Dependencies:
    - syracuse_dataset.py (for data loading)
    - numpy
    - matplotlib
    - seaborn
    - pandas
"""

from syracuse_dataset import SyracuseDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def analyze_pain_levels():
    """
    Analyze and visualize pre and post treatment pain levels in the Syracuse study.
    
    This function performs the following operations:
    1. Loads the pain measurement data from the Syracuse dataset
    2. Calculates separate statistics for pre and post treatment
    3. Creates three visualizations:
       - Pre-treatment pain distribution histogram
       - Post-treatment pain distribution histogram
       - Scatter plot of pre vs post pain levels
    4. Prints detailed distribution analysis for both pre and post measurements
    
    The visualizations include:
    - Histograms:
      * Separate pre and post treatment distributions
      * Mean and median lines
      * Count-based visualization
    - Scatter plot:
      * Pre vs post pain levels
      * Diagonal line showing no-change reference
      * Annotations for overlapping points
      * Points below diagonal indicate improvement
    
    Returns:
        None. Results are saved as:
        - Statistics printed to console
        - Two plot files:
          1. pain_levels_distribution.png
          2. pre_vs_post_pain.png
    
    Statistics calculated for both pre and post:
    - Mean pain level
    - Median pain level
    - Standard deviation
    - Range (min to max)
    - Distribution percentages for each pain level
    """
    # Initialize dataset
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Get pair information
    pairs_df = dataset.get_pair_info()
    
    # Calculate statistics
    pre_stats = {
        'mean': np.mean(pairs_df['pre_pain']),
        'median': np.median(pairs_df['pre_pain']),
        'std': np.std(pairs_df['pre_pain']),
        'min': np.min(pairs_df['pre_pain']),
        'max': np.max(pairs_df['pre_pain'])
    }
    
    post_stats = {
        'mean': np.mean(pairs_df['post_pain']),
        'median': np.median(pairs_df['post_pain']),
        'std': np.std(pairs_df['post_pain']),
        'min': np.min(pairs_df['post_pain']),
        'max': np.max(pairs_df['post_pain'])
    }
    
    # Print statistics
    print("\n=== Pain Level Distribution Analysis ===")
    print("\nPre-treatment Pain Statistics:")
    print(f"  * Mean: {pre_stats['mean']:.2f}")
    print(f"  * Median: {pre_stats['median']:.2f}")
    print(f"  * Standard deviation: {pre_stats['std']:.2f}")
    print(f"  * Range: {pre_stats['min']:.1f} to {pre_stats['max']:.1f}")
    
    print("\nPost-treatment Pain Statistics:")
    print(f"  * Mean: {post_stats['mean']:.2f}")
    print(f"  * Median: {post_stats['median']:.2f}")
    print(f"  * Standard deviation: {post_stats['std']:.2f}")
    print(f"  * Range: {post_stats['min']:.1f} to {post_stats['max']:.1f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot pre-treatment pain distribution
    plt.subplot(1, 2, 1)
    sns.histplot(data=pairs_df['pre_pain'], bins=range(0, 12))
    plt.axvline(x=pre_stats['mean'], color='r', linestyle='--', label='Mean')
    plt.axvline(x=pre_stats['median'], color='g', linestyle='--', label='Median')
    plt.xlabel('Pre-treatment Pain Level')
    plt.ylabel('Count')
    plt.title('Distribution of Pre-treatment Pain Levels')
    plt.legend()
    
    # Plot post-treatment pain distribution
    plt.subplot(1, 2, 2)
    sns.histplot(data=pairs_df['post_pain'], bins=range(0, 12))
    plt.axvline(x=post_stats['mean'], color='r', linestyle='--', label='Mean')
    plt.axvline(x=post_stats['median'], color='g', linestyle='--', label='Median')
    plt.xlabel('Post-treatment Pain Level')
    plt.ylabel('Count')
    plt.title('Distribution of Post-treatment Pain Levels')
    plt.legend()
    
    # Save the plots
    os.makedirs('Syracuse/analysis_results', exist_ok=True)
    plt.savefig('Syracuse/analysis_results/pain_levels_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot of pre vs post pain levels
    plt.figure(figsize=(10, 6))
    plt.scatter(pairs_df['pre_pain'], pairs_df['post_pain'])
    
    # Add diagonal line representing no change
    min_val = min(pairs_df['pre_pain'].min(), pairs_df['post_pain'].min())
    max_val = max(pairs_df['pre_pain'].max(), pairs_df['post_pain'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='No Change Line')
    
    plt.xlabel('Pre-treatment Pain Level')
    plt.ylabel('Post-treatment Pain Level')
    plt.title('Pre vs Post Treatment Pain Levels\n(Points below line indicate improvement)')
    plt.legend()
    
    # Add count annotations for overlapping points
    from collections import defaultdict
    point_counts = defaultdict(int)
    for pre, post in zip(pairs_df['pre_pain'], pairs_df['post_pain']):
        point_counts[(pre, post)] += 1
    
    for (pre, post), count in point_counts.items():
        if count > 1:
            plt.annotate(f'n={count}', (pre, post), 
                        xytext=(5, 5), textcoords='offset points')
    
    plt.savefig('Syracuse/analysis_results/pre_vs_post_pain.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print distribution details
    print("\nPre-treatment Pain Level Distribution:")
    pre_counts = pairs_df['pre_pain'].value_counts().sort_index()
    for pain_level, count in pre_counts.items():
        print(f"  * Level {pain_level}: {count} cases ({count/len(pairs_df)*100:.1f}%)")
    
    print("\nPost-treatment Pain Level Distribution:")
    post_counts = pairs_df['post_pain'].value_counts().sort_index()
    for pain_level, count in post_counts.items():
        print(f"  * Level {pain_level}: {count} cases ({count/len(pairs_df)*100:.1f}%)")

if __name__ == "__main__":
    analyze_pain_levels() 