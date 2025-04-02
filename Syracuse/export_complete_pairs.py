"""
Export Complete Pre-Post Pain Measurement Pairs

This script exports the 24 complete pre-post pairs from the Syracuse pain study to a CSV file.
A complete pair is defined as having both pre and post pain measurements for the same visit.

Output:
    - CSV file containing complete pairs with columns:
        * subject_id: Unique identifier for each subject
        * visit_number: Visit number (1 or 2)
        * pre_pain: Pain level before treatment
        * post_pain: Pain level after treatment
        * change: Absolute change in pain level (pre - post)
        * percentage_change: Percentage reduction in pain level
        * visit_type: Type of visit (1st-pre, 1st-post, 2nd-pre, 2nd-post)
        * file_name: Original video file name
"""

from syracuse_dataset import SyracuseDataset
import pandas as pd
import os

def export_complete_pairs():
    """
    Export complete pre-post pain measurement pairs to a CSV file.
    
    This function:
    1. Loads the dataset
    2. Gets all pre-post pairs
    3. Exports complete pairs to a CSV file
    
    Output:
        Saves a CSV file 'complete_pain_pairs.csv' in the analysis_results directory
    """
    # Initialize dataset
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Get pair information
    pairs_df = dataset.get_pair_info()
    
    # Create output directory if it doesn't exist
    os.makedirs('Syracuse/analysis_results', exist_ok=True)
    
    # Save to CSV
    output_path = 'Syracuse/analysis_results/complete_pain_pairs.csv'
    pairs_df.to_csv(output_path, index=False)
    
    print(f"\nExported {len(pairs_df)} complete pairs to {output_path}")
    print("\nDataset Summary:")
    print(f"Total pairs: {len(pairs_df)}")
    print("\nVisit Distribution:")
    print(pairs_df['visit_number'].value_counts())
    print("\nPain Change Statistics:")
    print(f"Mean change: {pairs_df['change'].mean():.2f} points")
    print(f"Median change: {pairs_df['change'].median():.2f} points")
    print(f"Range: {pairs_df['change'].min():.1f} to {pairs_df['change'].max():.1f} points")

if __name__ == "__main__":
    export_complete_pairs() 