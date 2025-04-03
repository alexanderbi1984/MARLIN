import os
from syracuse_dataset import SyracuseDataset

def check_clip_lengths():
    feature_dir = "/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2"
    meta_path = os.path.join(feature_dir, "meta_with_outcomes.xlsx")
    
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    for pair in dataset.pairs:
        subject_id = pair['subject']
        visit_num = pair['visit_number']
        pre_file = pair['pre_file']
        post_file = pair['post_file']
        
        # Get base filename without extension
        pre_base = os.path.splitext(pre_file)[0]
        post_base = os.path.splitext(post_file)[0]
        
        # List all clips in the feature directory
        all_files = os.listdir(feature_dir)
        
        # Filter for pre and post clips
        pre_clips = [f for f in all_files if f.startswith(pre_base + "_clip_") and f.endswith("_aligned.npy")]
        post_clips = [f for f in all_files if f.startswith(post_base + "_clip_") and f.endswith("_aligned.npy")]
        
        print(f"\nSubject {subject_id} (Visit {visit_num}):")
        print(f"Pre file: {pre_file} ({len(pre_clips)} clips)")
        print(f"Post file: {post_file} ({len(post_clips)} clips)")
        
        # Skip if either pre or post clips are missing
        if not pre_clips or not post_clips:
            print("Skipping due to missing clips")
            continue
            
        # Truncate to 14 clips if necessary
        pre_clips = sorted(pre_clips)[:14]
        post_clips = sorted(post_clips)[:14]
        
        if len(pre_clips) != len(post_clips):
            print("WARNING: Mismatched lengths after truncation!")
            print(f"Pre clips: {len(pre_clips)}")
            print(f"Post clips: {len(post_clips)}")

if __name__ == "__main__":
    check_clip_lengths() 