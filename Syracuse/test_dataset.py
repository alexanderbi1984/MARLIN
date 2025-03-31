from syracuse_dataset import SyracuseDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def test_dataset():
    # Initialize dataset
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Test pair creation
    pairs_df = dataset.get_pair_info()
    print("\n=== Pair Information ===")
    print(f"Total number of pairs: {len(pairs_df)}")
    print("\nPair statistics:")
    print(pairs_df.describe())
    
    # Analyze pain reduction distribution
    print("\n=== Pain Reduction Analysis ===")
    pain_changes = pairs_df['change'].values
    print("\nPain reduction statistics:")
    print(f"Mean reduction: {np.mean(pain_changes):.2f}")
    print(f"Median reduction: {np.median(pain_changes):.2f}")
    print(f"Standard deviation: {np.std(pain_changes):.2f}")
    print(f"Min reduction: {np.min(pain_changes):.2f}")
    print(f"Max reduction: {np.max(pain_changes):.2f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pain_changes, bins=20)
    plt.axvline(x=np.mean(pain_changes), color='r', linestyle='--', label='Mean')
    plt.axvline(x=np.median(pain_changes), color='g', linestyle='--', label='Median')
    plt.xlabel('Pain Reduction')
    plt.ylabel('Count')
    plt.title('Distribution of Pain Reduction')
    plt.legend()
    
    # Save the plot
    os.makedirs('analysis_results', exist_ok=True)
    plt.savefig('analysis_results/pain_reduction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test feature loading for one pair
    print("\n=== Testing Feature Loading ===")
    pair = dataset.pairs[0]
    pre_file = pair['pre_file']
    post_file = pair['post_file']
    
    # Print clip information
    pre_clips = sorted([f for f in os.listdir(feature_dir) 
                       if f.startswith(pre_file.replace('.MP4', ''))])
    post_clips = sorted([f for f in os.listdir(feature_dir) 
                        if f.startswith(post_file.replace('.MP4', ''))])
    
    print(f"\nPre video clips ({pre_file}):")
    print(f"Number of clips: {len(pre_clips)}")
    print("Clip names:", pre_clips)
    
    print(f"\nPost video clips ({post_file}):")
    print(f"Number of clips: {len(post_clips)}")
    print("Clip names:", post_clips)
    
    # Load and print feature shapes
    pre_features, post_features = dataset.load_features_for_pair(pair)
    print("\nFeature shapes:")
    print(f"Pre features shape: {pre_features.shape}")
    print(f"Post features shape: {post_features.shape}")
    
    # Print individual clip shapes
    print("\nIndividual clip shapes:")
    for clip in pre_clips:
        clip_path = os.path.join(feature_dir, clip)
        features = np.load(clip_path)
        print(f"{clip}: {features.shape}")
    
    print("\nPost clips:")
    for clip in post_clips:
        clip_path = os.path.join(feature_dir, clip)
        features = np.load(clip_path)
        print(f"{clip}: {features.shape}")

if __name__ == "__main__":
    test_dataset() 