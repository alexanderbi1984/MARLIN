"""
Analysis of Pain Level Distribution in Syracuse Pain Study

This script analyzes and visualizes the distribution of pain levels from the Syracuse pain study,
combining both pre-treatment and post-treatment measurements. It provides statistical analysis
and visualization of how pain levels are distributed across all measurements.

Key Features:
- Combines pre and post treatment pain measurements
- Calculates basic statistics (mean, median, std, range)
- Creates a detailed histogram visualization
- Provides detailed distribution analysis with percentages

The pain levels are measured on a scale of 0-10, where:
- 0 represents no pain
- 10 represents maximum pain

Output:
- Generates a histogram plot saved as 'pain_levels_histogram.png'
- Prints statistical summary and detailed distribution

Usage:
    python analyze_pain_levels_combined.py

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
    Analyze and visualize the distribution of pain levels in the Syracuse study.
    
    This function performs the following operations:
    1. Loads the pain measurement data from the Syracuse dataset
    2. Combines pre and post treatment pain measurements
    3. Calculates descriptive statistics (mean, median, std, range)
    4. Creates a detailed histogram visualization
    5. Prints detailed distribution analysis
    
    The visualization includes:
    - Histogram with 0.5-point width bins
    - Mean and median lines
    - Count labels on bars
    - Grid for better readability
    
    Returns:
        None. Results are saved as:
        - Statistics printed to console
        - Histogram plot saved as 'pain_levels_histogram.png'
    
    Statistics calculated:
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
    
    # Combine pre and post pain levels into a single series
    all_pain_levels = pd.concat([pairs_df['pre_pain'], pairs_df['post_pain']])
    
    # Calculate statistics
    stats = {
        'mean': np.mean(all_pain_levels),
        'median': np.median(all_pain_levels),
        'std': np.std(all_pain_levels),
        'min': np.min(all_pain_levels),
        'max': np.max(all_pain_levels)
    }
    
    # Print statistics
    print("\n=== Overall Pain Level Distribution Analysis ===")
    print(f"Total measurements: {len(all_pain_levels)}")
    print(f"  * Mean: {stats['mean']:.2f}")
    print(f"  * Median: {stats['median']:.2f}")
    print(f"  * Standard deviation: {stats['std']:.2f}")
    print(f"  * Range: {stats['min']:.1f} to {stats['max']:.1f}")
    
    # Set style
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # Create visualization
    plt.figure(figsize=(12, 7))
    
    # Plot pain level distribution
    bins = np.arange(0, 11.5, 0.5)  # Create bins from 0 to 11 with 0.5 intervals
    n, bins, patches = plt.hist(all_pain_levels, bins=bins, edgecolor='black', 
                              alpha=0.7, color='skyblue', rwidth=0.8)
    
    # Add mean and median lines
    plt.axvline(x=stats['mean'], color='red', linestyle='--', linewidth=2, 
                label=f"Mean ({stats['mean']:.1f})")
    plt.axvline(x=stats['median'], color='green', linestyle='--', linewidth=2, 
                label=f"Median ({stats['median']:.1f})")
    
    # Add count labels on top of each bar
    for i in range(len(n)):
        if n[i] > 0:  # Only add label if there are cases
            plt.text(bins[i], n[i], int(n[i]), 
                    horizontalalignment='center',
                    verticalalignment='bottom')
    
    # Customize plot
    plt.xlabel('Pain Level', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.title('Distribution of Pain Level Measurements\n(Pre and Post Treatment Combined)', 
              fontsize=14, pad=20)
    plt.legend(fontsize=10)
    
    # Set x-axis ticks to whole numbers
    plt.xticks(range(0, 11))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('Syracuse/analysis_results', exist_ok=True)
    plt.savefig('Syracuse/analysis_results/pain_levels_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print distribution details
    print("\nPain Level Distribution:")
    value_counts = all_pain_levels.value_counts().sort_index()
    for pain_level, count in value_counts.items():
        print(f"  * Level {pain_level}: {count} cases ({count/len(all_pain_levels)*100:.1f}%)")

if __name__ == "__main__":
    analyze_pain_levels() 