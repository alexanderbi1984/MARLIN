"""
Analysis of Pain Reduction in Syracuse Pain Study

This script analyzes pain reduction in the Syracuse pain study using both absolute and percentage-based
measurements. It provides comprehensive statistical analysis and visualizations to understand the
effectiveness of the treatment across different visits.

Key Features:
- Dual analysis approach: absolute and percentage-based pain reduction
- Visit-specific analysis (first vs second visits)
- Categorized improvement levels
- Multiple visualizations:
  1. Distribution of absolute pain reduction
  2. Distribution of percentage pain reduction
  3. Correlation between absolute and percentage changes

Improvement Categories:
- Absolute Changes:
  * No improvement (0 points)
  * Small improvement (1-3 points)
  * Significant improvement (4-9 points)
  * Complete improvement (10 points)
- Percentage Changes:
  * No improvement (0%)
  * Small improvement (1-25%)
  * Significant improvement (26-75%)
  * Complete improvement (>75%)

Output:
- Generates two plots:
  1. 'pain_reduction_comparison.png': Side-by-side histograms of absolute and percentage changes
  2. 'pain_reduction_correlation.png': Scatter plot of absolute vs percentage changes
- Prints detailed statistical summaries and categorized results

Usage:
    python analyze_pain_reduction_percentage.py

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
import os
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple

def calculate_pain_statistics(pairs_df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive pain reduction statistics for both absolute and percentage changes.
    
    Args:
        pairs_df (pd.DataFrame): DataFrame containing pre and post pain measurements
        
    Returns:
        Dict: Dictionary containing statistics for both absolute and percentage changes:
            - absolute:
                * mean: Average absolute pain reduction
                * median: Median absolute pain reduction
                * std: Standard deviation of absolute changes
                * min: Minimum absolute change
                * max: Maximum absolute change
                * changes: Array of all absolute changes
            - percentage:
                * mean: Average percentage pain reduction
                * median: Median percentage pain reduction
                * std: Standard deviation of percentage changes
                * min: Minimum percentage change
                * max: Maximum percentage change
                * changes: Array of all percentage changes
    """
    pain_changes = pairs_df['change'].values
    pre_pain = pairs_df['pre_pain'].values
    percentage_changes = (pain_changes / pre_pain) * 100
    
    stats = {
        'absolute': {
            'mean': np.mean(pain_changes),
            'median': np.median(pain_changes),
            'std': np.std(pain_changes),
            'min': np.min(pain_changes),
            'max': np.max(pain_changes),
            'changes': pain_changes
        },
        'percentage': {
            'mean': np.mean(percentage_changes),
            'median': np.median(percentage_changes),
            'std': np.std(percentage_changes),
            'min': np.min(percentage_changes),
            'max': np.max(percentage_changes),
            'changes': percentage_changes
        }
    }
    return stats

def categorize_improvements(stats: Dict, total: int) -> Dict:
    """
    Categorize improvements into different levels for both absolute and percentage changes.
    
    Args:
        stats (Dict): Dictionary containing pain reduction statistics
        total (int): Total number of measurements
        
    Returns:
        Dict: Dictionary containing categorized improvements with both counts and percentages:
            - absolute:
                * no_improvement: Count and percentage of no improvement cases
                * small_improvement: Count and percentage of small improvements
                * significant_improvement: Count and percentage of significant improvements
                * complete_improvement: Count and percentage of complete improvements
            - percentage:
                * no_improvement: Count and percentage of no improvement cases
                * small_improvement: Count and percentage of small improvements
                * significant_improvement: Count and percentage of significant improvements
                * complete_improvement: Count and percentage of complete improvements
    """
    categories = {
        'absolute': {
            'no_improvement': sum(1 for x in stats['absolute']['changes'] if x == 0),
            'small_improvement': sum(1 for x in stats['absolute']['changes'] if 0 < x <= 3),
            'significant_improvement': sum(1 for x in stats['absolute']['changes'] if 4 <= x < 10),
            'complete_improvement': sum(1 for x in stats['absolute']['changes'] if x == 10)
        },
        'percentage': {
            'no_improvement': sum(1 for x in stats['percentage']['changes'] if x == 0),
            'small_improvement': sum(1 for x in stats['percentage']['changes'] if 0 < x <= 25),
            'significant_improvement': sum(1 for x in stats['percentage']['changes'] if 25 < x <= 75),
            'complete_improvement': sum(1 for x in stats['percentage']['changes'] if x > 75)
        }
    }
    
    # Add percentages without modifying the original dictionary during iteration
    result = {}
    for metric in categories:
        result[metric] = categories[metric].copy()
        for category in categories[metric]:
            result[metric][category + '_pct'] = (categories[metric][category] / total) * 100
    
    return result

def analyze_visit_specific_changes(pairs_df: pd.DataFrame) -> Dict:
    """
    Analyze pain reduction changes specific to first and second visits.
    
    Args:
        pairs_df (pd.DataFrame): DataFrame containing pre and post pain measurements with visit numbers
        
    Returns:
        Dict: Dictionary containing analysis for first and second visits:
            - first_visits:
                * absolute: Mean, min, max of absolute changes
                * percentage: Mean, min, max of percentage changes
            - second_visits:
                * absolute: Mean, min, max of absolute changes
                * percentage: Mean, min, max of percentage changes
    """
    first_visits = pairs_df[pairs_df['visit_number'] == '1']
    second_visits = pairs_df[pairs_df['visit_number'] == '2']
    
    analysis = {
        'first_visits': {
            'absolute': {
                'mean': first_visits['change'].mean(),
                'min': first_visits['change'].min(),
                'max': first_visits['change'].max()
            },
            'percentage': {
                'mean': (first_visits['change'] / first_visits['pre_pain']).mean() * 100,
                'min': (first_visits['change'] / first_visits['pre_pain']).min() * 100,
                'max': (first_visits['change'] / first_visits['pre_pain']).max() * 100
            }
        },
        'second_visits': {
            'absolute': {
                'mean': second_visits['change'].mean(),
                'min': second_visits['change'].min(),
                'max': second_visits['change'].max()
            },
            'percentage': {
                'mean': (second_visits['change'] / second_visits['pre_pain']).mean() * 100,
                'min': (second_visits['change'] / second_visits['pre_pain']).min() * 100,
                'max': (second_visits['change'] / second_visits['pre_pain']).max() * 100
            }
        }
    }
    return analysis

def create_visualizations(stats: Dict, pairs_df: pd.DataFrame):
    """
    Create visualizations for pain reduction analysis.
    
    Generates two plots:
    1. Side-by-side histograms of absolute and percentage changes
    2. Scatter plot showing correlation between absolute and percentage changes
    
    Args:
        stats (Dict): Dictionary containing pain reduction statistics
        pairs_df (pd.DataFrame): DataFrame containing pain measurements
        
    Output:
        Saves two plot files:
        - pain_reduction_comparison.png
        - pain_reduction_correlation.png
    """
    # Create directory for results
    os.makedirs('Syracuse/analysis_results', exist_ok=True)
    
    # Plot absolute and percentage changes side by side
    plt.figure(figsize=(15, 5))
    
    # Plot absolute changes
    plt.subplot(1, 2, 1)
    sns.histplot(data=stats['absolute']['changes'], bins=20)
    plt.axvline(x=stats['absolute']['mean'], color='r', linestyle='--', label='Mean')
    plt.axvline(x=stats['absolute']['median'], color='g', linestyle='--', label='Median')
    plt.xlabel('Absolute Pain Reduction (Pre - Post)')
    plt.ylabel('Count')
    plt.title('Distribution of Absolute Pain Reduction')
    plt.legend()
    
    # Plot percentage changes
    plt.subplot(1, 2, 2)
    sns.histplot(data=stats['percentage']['changes'], bins=20)
    plt.axvline(x=stats['percentage']['mean'], color='r', linestyle='--', label='Mean')
    plt.axvline(x=stats['percentage']['median'], color='g', linestyle='--', label='Median')
    plt.xlabel('Percentage Pain Reduction')
    plt.ylabel('Count')
    plt.title('Distribution of Percentage Pain Reduction')
    plt.legend()
    
    plt.savefig('Syracuse/analysis_results/pain_reduction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot of absolute vs percentage changes
    plt.figure(figsize=(10, 6))
    plt.scatter(stats['absolute']['changes'], stats['percentage']['changes'])
    plt.xlabel('Absolute Pain Reduction')
    plt.ylabel('Percentage Pain Reduction')
    plt.title('Absolute vs Percentage Pain Reduction')
    
    # Add correlation coefficient
    correlation = np.corrcoef(stats['absolute']['changes'], stats['percentage']['changes'])[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes)
    
    plt.savefig('Syracuse/analysis_results/pain_reduction_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_analysis_results(stats: Dict, categories: Dict, visit_analysis: Dict, total: int):
    """
    Print comprehensive analysis results in a formatted way.
    
    Args:
        stats (Dict): Dictionary containing pain reduction statistics
        categories (Dict): Dictionary containing categorized improvements
        visit_analysis (Dict): Dictionary containing visit-specific analysis
        total (int): Total number of measurements
        
    Output:
        Prints formatted results to console including:
        - Overall statistics
        - Categorized improvements
        - Visit-specific analysis
    """
    print("\n=== Pain Change Distribution Analysis ===")
    print(f"Total complete pre-post pairs: {total}")
    
    # Print absolute change statistics
    print("\nAbsolute Change Statistics:")
    print(f"  * Mean change: {stats['absolute']['mean']:.2f} points")
    print(f"  * Median change: {stats['absolute']['median']:.2f} points")
    print(f"  * Standard deviation: {stats['absolute']['std']:.2f} points")
    print(f"  * Range: {stats['absolute']['min']:.1f} to {stats['absolute']['max']:.1f} points")
    
    # Print percentage change statistics
    print("\nPercentage Change Statistics:")
    print(f"  * Mean reduction: {stats['percentage']['mean']:.2f}%")
    print(f"  * Median reduction: {stats['percentage']['median']:.2f}%")
    print(f"  * Standard deviation: {stats['percentage']['std']:.2f}%")
    print(f"  * Range: {stats['percentage']['min']:.1f}% to {stats['percentage']['max']:.1f}%")
    
    # Print categorized improvements
    print("\nCategorized improvements (Absolute):")
    print(f"  * No improvement (0): {categories['absolute']['no_improvement']} cases ({categories['absolute']['no_improvement_pct']:.0f}%)")
    print(f"  * Small improvement (1-3): {categories['absolute']['small_improvement']} cases ({categories['absolute']['small_improvement_pct']:.0f}%)")
    print(f"  * Significant improvement (≥4): {categories['absolute']['significant_improvement']} cases ({categories['absolute']['significant_improvement_pct']:.0f}%)")
    print(f"  * Complete improvement (10): {categories['absolute']['complete_improvement']} cases ({categories['absolute']['complete_improvement_pct']:.0f}%)")
    
    print("\nCategorized improvements (Percentage):")
    print(f"  * No improvement (0%): {categories['percentage']['no_improvement']} cases ({categories['percentage']['no_improvement_pct']:.0f}%)")
    print(f"  * Small improvement (1-25%): {categories['percentage']['small_improvement']} cases ({categories['percentage']['small_improvement_pct']:.0f}%)")
    print(f"  * Significant improvement (26-75%): {categories['percentage']['significant_improvement']} cases ({categories['percentage']['significant_improvement_pct']:.0f}%)")
    print(f"  * Complete improvement (>75%): {categories['percentage']['complete_improvement']} cases ({categories['percentage']['complete_improvement_pct']:.0f}%)")
    
    # Print visit-specific analysis
    print("\nVisit-specific analysis (Absolute):")
    print(f"  * 1st visits:")
    print(f"    - Mean change: {visit_analysis['first_visits']['absolute']['mean']:.2f} points")
    print(f"    - Range: {visit_analysis['first_visits']['absolute']['min']:.1f} to {visit_analysis['first_visits']['absolute']['max']:.1f} points")
    print(f"  * 2nd visits:")
    print(f"    - Mean change: {visit_analysis['second_visits']['absolute']['mean']:.2f} points")
    print(f"    - Range: {visit_analysis['second_visits']['absolute']['min']:.1f} to {visit_analysis['second_visits']['absolute']['max']:.1f} points")
    
    print("\nVisit-specific analysis (Percentage):")
    print(f"  * 1st visits:")
    print(f"    - Mean reduction: {visit_analysis['first_visits']['percentage']['mean']:.2f}%")
    print(f"    - Range: {visit_analysis['first_visits']['percentage']['min']:.1f}% to {visit_analysis['first_visits']['percentage']['max']:.1f}%")
    print(f"  * 2nd visits:")
    print(f"    - Mean reduction: {visit_analysis['second_visits']['percentage']['mean']:.2f}%")
    print(f"    - Range: {visit_analysis['second_visits']['percentage']['min']:.1f}% to {visit_analysis['second_visits']['percentage']['max']:.1f}%")

def analyze_pain_reduction():
    """
    Main function to analyze pain reduction using both absolute and percentage changes.
    
    This function orchestrates the complete analysis process:
    1. Loads the dataset
    2. Calculates pain reduction statistics
    3. Categorizes improvements
    4. Analyzes visit-specific changes
    5. Creates visualizations
    6. Prints comprehensive results
    
    The analysis provides insights into:
    - Overall treatment effectiveness
    - Distribution of improvements
    - Visit-specific patterns
    - Correlation between absolute and percentage changes
    
    Output:
        - Generates visualization plots
        - Prints detailed statistical analysis
    """
    # Initialize dataset
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Get pair information
    pairs_df = dataset.get_pair_info()
    total = len(pairs_df)
    
    # Calculate statistics
    stats = calculate_pain_statistics(pairs_df)
    
    # Categorize improvements
    categories = categorize_improvements(stats, total)
    
    # Analyze visit-specific changes
    visit_analysis = analyze_visit_specific_changes(pairs_df)
    
    # Create visualizations
    create_visualizations(stats, pairs_df)
    
    # Print results
    print_analysis_results(stats, categories, visit_analysis, total)

if __name__ == "__main__":
    analyze_pain_reduction() 