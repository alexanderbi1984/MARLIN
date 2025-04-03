"""
Preprocessing script for Biovid dataset for binary outcome classification.

This script preprocesses the Biovid dataset for binary outcome classification by:
1. Loading metadata from biovid_biovidGan.json
2. Constructing pre-post pairs based on ground truth differences:
   - Positive pairs: difference >= 3
   - Negative pairs: difference <= 2 (only when pre-treatment ground truth >= 3)
3. Loading MARLIN features for each clip
4. Computing feature differences between pre and post clips
5. Saving processed data in an efficient format:
   - pairs_metadata.json: Contains all metadata for each pair
   - feature_diffs.npy: Contains all feature differences in a dictionary format

Pair Construction Logic:
1. For each subject, consider all possible pre-post clip combinations
2. Filtering criteria:
   - Exclude self-pairs (pre_clip != post_clip)
   - Only consider pairs where pre_ground_truth >= post_ground_truth
   - For positive pairs: difference >= 3
   - For negative pairs: 
     * difference <= 2
     * AND pre_ground_truth >= 3 (only consider negative pairs when pre-treatment is painful)
   - Skip pairs where pre-treatment is not painful (ground_truth < 3) and improvement is small

Example:
  Pre-treatment ground truth: 4, Post-treatment: 1 -> Positive pair (diff = 3)
  Pre-treatment ground truth: 4, Post-treatment: 2 -> Negative pair (diff = 2)
  Pre-treatment ground truth: 2, Post-treatment: 1 -> Skipped (pre not painful)
  Pre-treatment ground truth: 3, Post-treatment: 1 -> Positive pair (diff = 2)

Output Format:
- pairs_metadata.json: Dictionary with pair IDs as keys and metadata as values
  {
    "pair_000001": {
      "subject_id": "...",
      "pre_clip": "...",
      "post_clip": "...",
      "pre_ground_truth": int,
      "post_ground_truth": int,
      "difference": int,
      "outcome": int (0 or 1),
      "sex": str,
      "age": int
    },
    ...
  }
- feature_diffs.npy: Dictionary with pair IDs as keys and feature differences as values
  {
    "pair_000001": array([...]),  # shape: (768,)
    ...
  }

Usage:
    python biovid_preprocess.py

Dependencies:
    - pandas
    - numpy
    - json
    - logging
    - os
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_biovid_metadata():
    """Load and preprocess Biovid metadata.
    
    Returns:
        pd.DataFrame: Processed metadata with outcomes
    """
    # Load JSON metadata
    meta_path = r"C:\Users\Nan Bi\PycharmProjects\MARLIN\dataset_meta\biovid_biovidGan.json"
    logging.info("Loading metadata file...")
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    # Extract clips data
    clips_data = meta_data['clips']
    
    # Create a list to store processed data
    processed_data = []
    
    # Process each clip
    logging.info("Processing clips...")
    for clip_name, clip_info in clips_data.items():
        # Only process BioVid clips (not BioVidGan)
        if clip_info['attributes']['source'] != 'BioVid':
            continue
            
        processed_data.append({
            'clip_name': clip_name,
            'subject_id': clip_info['attributes']['subject_id'],
            'ground_truth': clip_info['attributes']['ground_truth'],
            'sex': clip_info['attributes']['sex'],
            'age': clip_info['attributes']['age']
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Print initial data statistics
    logging.info("\nInitial Data Statistics:")
    logging.info(f"Total number of clips: {len(df)}")
    logging.info(f"Number of unique subjects: {df['subject_id'].nunique()}")
    logging.info("\nGround truth distribution:")
    logging.info(df['ground_truth'].value_counts().sort_index())
    
    # Create pairs
    pairs = []
    subjects = df['subject_id'].unique()
    total_subjects = len(subjects)
    
    logging.info("\nCreating pairs...")
    for i, subject in enumerate(subjects, 1):
        if i % 10 == 0:  # Progress update every 10 subjects
            logging.info(f"Processing subject {i}/{total_subjects}")
            
        subject_clips = df[df['subject_id'] == subject]
        
        # Get all possible pre-post combinations
        for _, pre_clip in subject_clips.iterrows():
            for _, post_clip in subject_clips.iterrows():
                if pre_clip['clip_name'] != post_clip['clip_name']:  # Avoid self-pairs
                    # Only consider pairs where pre >= post
                    if pre_clip['ground_truth'] >= post_clip['ground_truth']:
                        diff = pre_clip['ground_truth'] - post_clip['ground_truth']
                        
                        # Determine outcome based on difference
                        if diff >= 3:
                            outcome = 'positive'
                        elif pre_clip['ground_truth'] >= 3:  # Only consider negative pairs where pre is painful
                            outcome = 'negative'
                        else:
                            continue  # Skip pairs where pre is not painful and improvement is small
                        
                        pairs.append({
                            'subject_id': subject,
                            'pre_clip': pre_clip['clip_name'],
                            'post_clip': post_clip['clip_name'],
                            'pre_ground_truth': pre_clip['ground_truth'],
                            'post_ground_truth': post_clip['ground_truth'],
                            'difference': diff,
                            'outcome': outcome,
                            'sex': pre_clip['sex'],
                            'age': pre_clip['age']
                        })
    
    # Convert pairs to DataFrame
    pairs_df = pd.DataFrame(pairs)
    
    # Print pair statistics
    logging.info("\nPair Construction Statistics:")
    logging.info(f"Total number of pairs: {len(pairs_df)}")
    logging.info(f"Number of unique subjects with pairs: {pairs_df['subject_id'].nunique()}")
    
    # Print outcome distribution
    logging.info("\nOutcome distribution:")
    outcome_counts = pairs_df['outcome'].value_counts()
    logging.info(outcome_counts)
    logging.info(f"Positive/Negative ratio: {outcome_counts['positive']/outcome_counts['negative']:.2f}")
    
    # Print detailed pair statistics
    logging.info("\nDetailed Pair Statistics:")
    
    # Pre-post combinations
    logging.info("\nPre-post ground truth combinations:")
    combo_counts = pairs_df.groupby(['pre_ground_truth', 'post_ground_truth']).size()
    logging.info(combo_counts)
    
    # Outcome by pre-post combinations
    logging.info("\nOutcome by pre-post combinations:")
    outcome_by_combo = pairs_df.groupby(['pre_ground_truth', 'post_ground_truth', 'outcome']).size()
    logging.info(outcome_by_combo)
    
    # Average pairs per subject
    pairs_per_subject = pairs_df.groupby('subject_id').size()
    logging.info(f"\nAverage pairs per subject: {pairs_per_subject.mean():.2f}")
    logging.info(f"Min pairs per subject: {pairs_per_subject.min()}")
    logging.info(f"Max pairs per subject: {pairs_per_subject.max()}")
    
    # Difference distribution
    logging.info("\nDifference distribution:")
    logging.info(pairs_df['difference'].value_counts().sort_index())
    
    return pairs_df

def load_features(features_dir, clip_names):
    """Load MARLIN features for given clip names.
    
    Args:
        features_dir (str): Directory containing feature files
        clip_names (list): List of clip names to load features for
        
    Returns:
        pd.DataFrame: DataFrame with features indexed by clip names
    """
    features_list = []
    shapes = set()  # Track unique shapes
    
    for clip_name in clip_names:
        # Convert .mp4 to .npy extension
        feature_file = clip_name.replace('.mp4', '.npy')
        feature_path = os.path.join(features_dir, feature_file)
        
        if os.path.exists(feature_path):
            # Load .npy file
            features = np.load(feature_path)
            shapes.add(features.shape)
            
            # Average across temporal dimension to get 768-dim vector
            features_avg = np.mean(features, axis=0)  # Shape: (768,)
            
            # Ensure features are 1D
            if len(features_avg.shape) > 1:
                features_avg = features_avg.flatten()
            
            features_list.append({
                'clip_name': clip_name,
                'features': features_avg
            })
        else:
            logging.warning(f"Feature file not found: {feature_path}")
    
    if not features_list:
        raise ValueError("No feature files found!")
    
    # Log shape information
    logging.info(f"Found {len(shapes)} different original shapes:")
    for shape in shapes:
        logging.info(f"Original shape {shape}")
    logging.info(f"After averaging: all features have shape (768,)")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    features_df.set_index('clip_name', inplace=True)
    
    # Verify feature shapes
    for clip_name, row in features_df.iterrows():
        if row['features'].shape != (768,):
            logging.error(f"Unexpected feature shape for {clip_name}: {row['features'].shape}")
    
    return features_df

def process_features():
    """Process MARLIN features for Biovid dataset.
    
    Returns:
        np.ndarray: Processed feature matrix
    """
    # TODO: Implement feature processing
    pass

def create_outcome_labels():
    """Create binary outcome labels for Biovid dataset.
    
    Returns:
        np.ndarray: Binary outcome labels
    """
    # TODO: Implement outcome labeling
    pass

def save_pair_data(pairs_df, output_dir):
    """Save each pair's data in separate files.
    
    Args:
        pairs_df (pd.DataFrame): DataFrame containing all pairs
        output_dir (str): Directory to save the files
    """
    # Create metadata dictionary for all pairs
    metadata_dict = {}
    feature_diffs_dict = {}
    
    # Process each pair
    total_pairs = len(pairs_df)
    for i, (_, row) in enumerate(pairs_df.iterrows(), 1):
        if i % 1000 == 0:  # Progress update every 1000 pairs
            logging.info(f"Processing pair {i}/{total_pairs}")
            
        # Create metadata dictionary for this pair
        pair_id = f"pair_{i:06d}"
        metadata_dict[pair_id] = {
            'subject_id': row['subject_id'],
            'pre_clip': row['pre_clip'],
            'post_clip': row['post_clip'],
            'pre_ground_truth': row['pre_ground_truth'],
            'post_ground_truth': row['post_ground_truth'],
            'difference': row['difference'],
            'outcome': row['outcome'],
            'sex': row['sex'],
            'age': row['age']
        }
        
        # Store feature difference in dictionary
        feature_diffs_dict[pair_id] = row['feature_diff']
    
    # Save all metadata in a single JSON file
    meta_path = os.path.join(output_dir, 'pairs_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)  # indent=2 for pretty printing
    
    # Save all feature differences in a single NPY file
    diff_path = os.path.join(output_dir, 'feature_diffs.npy')
    np.save(diff_path, feature_diffs_dict)
    
    logging.info(f"\nSaved {total_pairs} pairs:")
    logging.info(f"- Metadata file: {meta_path}")
    logging.info(f"- Feature differences: {diff_path}")
    
    # Print some statistics about the saved data
    logging.info("\nSaved Data Statistics:")
    logging.info(f"Total pairs: {total_pairs}")
    logging.info(f"Feature dimension: {list(feature_diffs_dict.values())[0].shape}")
    logging.info(f"Memory usage of feature differences: {sum(x.nbytes for x in feature_diffs_dict.values()) / (1024*1024):.2f} MB")

def main():
    """Main preprocessing pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting Biovid data preprocessing...")
    
    # Load metadata and create pairs
    pairs_df = load_biovid_metadata()
    
    # Load and process features
    logging.info("\nProcessing features...")
    features_path = r"C:\pain\BioVid_224_video\multimodal_marlin_base"
    
    # Create pre and post feature DataFrames
    pre_features = load_features(features_path, pairs_df['pre_clip'].unique())
    post_features = load_features(features_path, pairs_df['post_clip'].unique())
    
    # Print feature statistics
    logging.info("\nFeature Statistics:")
    logging.info(f"Total number of pre clips: {len(pre_features)}")
    logging.info(f"Total number of post clips: {len(post_features)}")
    
    # Check for any clips that appear in both pre and post
    common_clips = set(pre_features.index) & set(post_features.index)
    logging.info(f"\nNumber of clips that appear in both pre and post: {len(common_clips)}")
    
    # Merge features with pairs
    pairs_df = pairs_df.merge(pre_features, left_on='pre_clip', right_index=True)
    pairs_df = pairs_df.merge(post_features, left_on='post_clip', right_index=True)
    
    # Calculate feature differences
    logging.info("\nCalculating feature differences...")
    pairs_df['feature_diff'] = pairs_df.apply(
        lambda row: row['features_x'] - row['features_y'], axis=1
    )
    
    # Sample a few differences to verify
    logging.info("\nSample feature differences (first 3 pairs):")
    for i in range(3):
        sample_row = pairs_df.iloc[i]
        logging.info(f"\nPair {i+1}:")
        logging.info(f"Pre clip: {sample_row['pre_clip']}")
        logging.info(f"Post clip: {sample_row['post_clip']}")
        logging.info(f"Feature diff shape: {sample_row['feature_diff'].shape}")
        logging.info(f"Feature diff mean: {np.mean(sample_row['feature_diff']):.6f}")
        logging.info(f"Feature diff std: {np.std(sample_row['feature_diff']):.6f}")
    
    # Validate feature differences
    logging.info("\nFeature Difference Statistics:")
    diff_stats = pairs_df['feature_diff'].apply(lambda x: {
        'mean': np.mean(x),
        'std': np.std(x),
        'min': np.min(x),
        'max': np.max(x),
        'zeros': np.sum(x == 0),
        'non_zeros': np.sum(x != 0)
    })
    
    # Print summary statistics
    logging.info("\nFeature Difference Summary:")
    logging.info(f"Mean of means: {diff_stats.apply(lambda x: x['mean']).mean():.6f}")
    logging.info(f"Mean of stds: {diff_stats.apply(lambda x: x['std']).mean():.6f}")
    logging.info(f"Overall min: {diff_stats.apply(lambda x: x['min']).min():.6f}")
    logging.info(f"Overall max: {diff_stats.apply(lambda x: x['max']).max():.6f}")
    logging.info(f"Total zeros: {diff_stats.apply(lambda x: x['zeros']).sum()}")
    logging.info(f"Total non-zeros: {diff_stats.apply(lambda x: x['non_zeros']).sum()}")
    
    # Check for any suspicious patterns
    zero_ratios = diff_stats.apply(lambda x: x['zeros'] / (x['zeros'] + x['non_zeros']))
    if zero_ratios.mean() > 0.5:
        logging.warning("High proportion of zero differences detected!")
        logging.info(f"Mean zero ratio: {zero_ratios.mean():.4f}")
    
    # Create outcome labels
    logging.info("\nCreating outcome labels...")
    pairs_df['outcome'] = (pairs_df['outcome'] == 'positive').astype(int)
    
    # Save processed data
    logging.info("\nSaving processed data...")
    output_dir = os.path.join(os.path.dirname(__file__), 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each pair in a separate file
    save_pair_data(pairs_df, output_dir)
    
    # Save feature difference statistics for reference
    diff_stats.to_csv(os.path.join(output_dir, 'feature_diff_stats.csv'))
    
    logging.info("Preprocessing completed successfully!")

if __name__ == "__main__":
    main() 