import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pathlib import Path

def load_features_for_subject(subject_id, file_name, feature_indices, features_dir):
    """Load specific features for a video."""
    video_id = file_name.replace('.MP4', '')
    feature_files = []
    
    # Look for feature files matching the pattern
    for f in os.listdir(features_dir):
        if f.startswith(f"{video_id}_clip_") and f.endswith("_aligned.npy"):
            feature_files.append(f)
    
    if not feature_files:
        print(f"No feature files found for {video_id}")
        return None
        
    # Load and extract specified features across clips
    features_list = []
    for f in sorted(feature_files):  # Sort to maintain temporal order
        try:
            feature = np.load(os.path.join(features_dir, f))
            # If feature is 3D (clips, frames, features), take mean across frames only
            if len(feature.shape) == 3:
                feature = np.mean(feature, axis=1)  # Average across frames, keep clips
            # If feature is 2D (frames, features), reshape to add clip dimension
            elif len(feature.shape) == 2:
                feature = feature.reshape(1, -1)  # Add clip dimension
                
            # Extract only the features we want
            selected_features = feature[:, feature_indices]
            features_list.append(selected_features)
            
        except Exception as e:
            print(f"Error loading {f}: {str(e)}")
            continue
    
    if not features_list:
        return None
        
    # Concatenate all clips
    return np.concatenate(features_list, axis=0)

def visualize_features(metadata_df, features_dir, feature_indices, output_dir):
    """Visualize how the selected features change over time in videos."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all differences for population statistics
    all_diffs = {idx: [] for idx in feature_indices}
    positive_diffs = {idx: [] for idx in feature_indices}
    negative_diffs = {idx: [] for idx in feature_indices}
    
    # Create figure for each subject
    for subject_id in tqdm(metadata_df['subject_id'].unique(), desc="Processing subjects"):
        subject_data = metadata_df[metadata_df['subject_id'] == subject_id]
        
        # Get pre and post visits
        pre_visit = subject_data[subject_data['visit_type'].fillna('').str.contains('-pre', na=False)]
        post_visit = subject_data[subject_data['visit_type'].fillna('').str.contains('-post', na=False)]
        
        if pre_visit.empty or post_visit.empty:
            print(f"Skipping subject {subject_id}: Missing pre or post visit")
            continue
        
        # Load features for pre and post visits
        pre_features = load_features_for_subject(
            subject_id, 
            pre_visit['file_name'].iloc[0], 
            feature_indices, 
            features_dir
        )
        post_features = load_features_for_subject(
            subject_id, 
            post_visit['file_name'].iloc[0], 
            feature_indices, 
            features_dir
        )
        
        if pre_features is None or post_features is None:
            continue
            
        # Calculate differences
        # Normalize to fixed length for comparison
        target_length = 100  # Fixed length for all sequences
        t_norm = np.linspace(0, 1, target_length)
        
        # Create time points for interpolation
        t_pre = np.linspace(0, 1, len(pre_features))
        t_post = np.linspace(0, 1, len(post_features))
        
        # Interpolate both sequences to target length
        pre_interp = np.zeros((target_length, len(feature_indices)))
        post_interp = np.zeros((target_length, len(feature_indices)))
        
        for i in range(len(feature_indices)):
            pre_interp[:, i] = np.interp(t_norm, t_pre, pre_features[:, i])
            post_interp[:, i] = np.interp(t_norm, t_post, post_features[:, i])
            
        feature_diffs = post_interp - pre_interp
        
        # Store differences based on outcome
        outcome = subject_data['outcome'].iloc[0]
        for i, idx in enumerate(feature_indices):
            if outcome == 'positive':
                positive_diffs[idx].append(feature_diffs[:, i])
            else:
                negative_diffs[idx].append(feature_diffs[:, i])
            all_diffs[idx].append(feature_diffs[:, i])
    
    # Create summary visualization
    fig, axes = plt.subplots(len(feature_indices), 1, figsize=(15, 5*len(feature_indices)))
    if len(feature_indices) == 1:
        axes = [axes]
    
    for i, (feature_idx, ax) in enumerate(zip(feature_indices, axes)):
        # Convert lists to arrays for calculations
        pos_array = np.array(positive_diffs[feature_idx])
        neg_array = np.array(negative_diffs[feature_idx])
        
        # Calculate statistics
        pos_mean = np.mean(pos_array, axis=0) if len(pos_array) > 0 else None
        pos_std = np.std(pos_array, axis=0) if len(pos_array) > 0 else None
        neg_mean = np.mean(neg_array, axis=0) if len(neg_array) > 0 else None
        neg_std = np.std(neg_array, axis=0) if len(neg_array) > 0 else None
        
        # Plot means and confidence intervals
        t = np.linspace(0, 1, target_length)
        
        if pos_mean is not None:
            ax.plot(t, pos_mean, 'b-', label='Positive Outcome (mean)', linewidth=2)
            ax.fill_between(t, pos_mean - pos_std, pos_mean + pos_std, 
                          color='blue', alpha=0.2, label='Positive Outcome (±1 std)')
        
        if neg_mean is not None:
            ax.plot(t, neg_mean, 'r-', label='Negative Outcome (mean)', linewidth=2)
            ax.fill_between(t, neg_mean - neg_std, neg_mean + neg_std, 
                          color='red', alpha=0.2, label='Negative Outcome (±1 std)')
        
        ax.set_title(f'Feature {feature_idx} - Pre-Post Differences Over Time')
        ax.set_xlabel('Normalized Time (0-1)')
        ax.set_ylabel('Feature Difference (Post - Pre)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add zero line for reference
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_differences_summary.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary visualization saved as 'feature_differences_summary.png'")

def main():
    # Constants
    FEATURES_DIR = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    META_PATH = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    OUTPUT_DIR = "Syracuse/feature_visualization"
    
    # Load metadata
    metadata_df = pd.read_excel(META_PATH)
    
    # Get top features from analysis results
    analysis_path = os.path.join('outcome_analysis_results', 'marlin_video_outcome_analysis.csv')
    df = pd.read_csv(analysis_path)
    df['abs_effect_size'] = df['effect_size'].abs()
    df_sorted = df.sort_values('abs_effect_size', ascending=False)
    top_features = df_sorted.head(3)
    feature_indices = top_features['feature_idx'].astype(int).tolist()
    
    print("\nSelected top 3 features:")
    for _, row in top_features.iterrows():
        print(f"Feature {int(row['feature_idx'])}: effect_size = {row['effect_size']:.3f}, p_value = {row['p_value']:.6f}")
    
    # Create visualizations
    visualize_features(metadata_df, FEATURES_DIR, feature_indices, OUTPUT_DIR)
    print(f"\nVisualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 