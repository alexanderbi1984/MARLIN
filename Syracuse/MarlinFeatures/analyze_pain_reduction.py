from syracuse_dataset import SyracuseDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

def analyze_pain_reduction():
    # Initialize dataset with correct paths
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    
    dataset = SyracuseDataset(meta_path, feature_dir)
    pairs_df = dataset.get_pair_info()
    pain_changes = pairs_df['change'].values
    
    # Calculate basic statistics
    print("\n=== Pain Change Distribution Analysis ===")
    print(f"Total complete pre-post pairs: {len(pain_changes)}")
    print("\nOverall statistics:")
    print(f"  * Mean change: {np.mean(pain_changes):.2f} points")
    print(f"  * Median change: {np.median(pain_changes):.2f} points")
    print(f"  * Standard deviation: {np.std(pain_changes):.2f} points")
    print(f"  * Range: {np.min(pain_changes):.1f} to {np.max(pain_changes):.1f} points")
    
    # Distribution of changes
    print("\nDistribution of changes:")
    change_counts = Counter(pain_changes)
    for change in sorted(change_counts.keys()):
        print(f"  * {change:.1f} points: {change_counts[change]} cases" + 
              (" (no improvement)" if change == 0 else 
               " (most common)" if change_counts[change] == max(change_counts.values()) else 
               " (complete improvement)" if change == 10 else ""))
    
    # Categorize improvements
    no_improvement = sum(1 for x in pain_changes if x == 0)
    small_improvement = sum(1 for x in pain_changes if 0 < x <= 3)
    significant_improvement = sum(1 for x in pain_changes if x >= 4 and x < 10)
    complete_improvement = sum(1 for x in pain_changes if x == 10)
    
    total = len(pain_changes)
    print("\nCategorized improvements:")
    print(f"  * No improvement (0): {no_improvement} cases ({(no_improvement/total*100):.0f}%)")
    print(f"  * Small improvement (1-3): {small_improvement} cases ({(small_improvement/total*100):.0f}%)")
    print(f"  * Significant improvement (≥4): {significant_improvement} cases ({(significant_improvement/total*100):.0f}%)")
    print(f"  * Complete improvement (10): {complete_improvement} cases ({(complete_improvement/total*100):.0f}%)")
    
    # Visit-specific analysis
    first_visits = pairs_df[pairs_df['visit_number'] == '1']
    second_visits = pairs_df[pairs_df['visit_number'] == '2']
    
    print("\nVisit-specific analysis:")
    print(f"  * 1st visits ({len(first_visits)} pairs):")
    print(f"    - Mean change: {first_visits['change'].mean():.2f} points")
    print(f"    - Range: {first_visits['change'].min():.1f} to {first_visits['change'].max():.1f} points")
    print(f"  * 2nd visits ({len(second_visits)} pairs):")
    print(f"    - Mean change: {second_visits['change'].mean():.2f} points")
    print(f"    - Range: {second_visits['change'].min():.1f} to {second_visits['change'].max():.1f} points")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pain_changes, bins=20)
    plt.axvline(x=np.mean(pain_changes), color='r', linestyle='--', label='Mean')
    plt.axvline(x=np.median(pain_changes), color='g', linestyle='--', label='Median')
    plt.xlabel('Pain Reduction (Pre - Post)')
    plt.ylabel('Count')
    plt.title('Distribution of Pain Reduction\n(Positive values indicate improvement)')
    plt.legend()
    
    # Save the plot
    os.makedirs('Syracuse/analysis_results', exist_ok=True)
    plt.savefig('Syracuse/analysis_results/pain_reduction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_pain_reduction() 