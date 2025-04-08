import os
from syracuse_dataset import SyracuseDataset
import numpy as np
from collections import defaultdict

def test_clip_loading():
    # Initialize dataset with correct paths
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes_and_classes.xlsx'
    feature_dir = "/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2"
    
    print("Initializing SyracuseDataset...")
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Test loading a single clip
    print("\n=== Testing Single Clip Loading ===")
    try:
        # Get first video from meta data
        first_video = dataset.meta_df.iloc[0]
        file_name = first_video['file_name']
        video_name = file_name.replace('.MP4', '')
        
        # Find first clip for this video
        clips = sorted([f for f in os.listdir(feature_dir) 
                       if f.startswith(f"{video_name}_clip_") and f.endswith('_aligned.npy')],
                      key=lambda x: int(x.split('_clip_')[1].split('_')[0]))
        
        if clips:
            first_clip = clips[0]
            print(f"Testing with video: {file_name}")
            print(f"Testing with clip: {first_clip}")
            print(f"Total clips for this video: {len(clips)}")
            
            # Load clip data
            clip_data = dataset.load_features_for_clip(file_name, first_clip)
            
            # Print feature information
            print("\nFeature shape:", clip_data['features'].shape)
            print("Expected shape: (4, 768)")
            
            # Print metadata
            print("\nMetadata:")
            for key, value in clip_data['metadata'].items():
                print(f"{key}: {value}")
                
            print("\nSingle clip loading test passed!")
        else:
            print(f"No clips found for video: {file_name}")
            
    except Exception as e:
        print(f"Error in single clip loading test: {str(e)}")
    
    # Test loading all clips
    print("\n=== Testing All Clips Loading ===")
    try:
        all_clips = dataset.load_all_clips()
        
        # Print summary of loaded clips
        print(f"\nTotal clips loaded: {len(all_clips)}")
        
        # Print feature shapes for first few clips
        print("\nFeature shapes for first 3 clips:")
        for i, clip in enumerate(all_clips[:3]):
            print(f"Clip {i+1} shape: {clip['features'].shape}")
        
        # Analyze clips per video
        clips_per_video = defaultdict(int)
        for clip in all_clips:
            video_name = clip['metadata']['video_name']
            clips_per_video[video_name] += 1
        
        print("\nClips per video statistics:")
        print(f"Number of unique videos: {len(clips_per_video)}")
        print(f"Minimum clips per video: {min(clips_per_video.values())}")
        print(f"Maximum clips per video: {max(clips_per_video.values())}")
        print(f"Average clips per video: {np.mean(list(clips_per_video.values())):.2f}")
        print(f"Median clips per video: {np.median(list(clips_per_video.values())):.2f}")
        
        # Print distribution of clips per video
        print("\nDistribution of clips per video:")
        clip_counts = defaultdict(int)
        for count in clips_per_video.values():
            clip_counts[count] += 1
        for count in sorted(clip_counts.keys()):
            print(f"{count} clips: {clip_counts[count]} videos")
        
        # Print pain level distribution
        pain_levels = [clip['metadata']['pain_level'] for clip in all_clips]
        unique_pain_levels = sorted(set(pain_levels))
        print("\nPain level distribution:")
        for level in unique_pain_levels:
            count = pain_levels.count(level)
            print(f"Pain level {level}: {count} clips")
        
        print("\nAll clips loading test passed!")
        
    except Exception as e:
        print(f"Error in all clips loading test: {str(e)}")

if __name__ == "__main__":
    test_clip_loading() 