import os
import json
import pandas as pd
import re
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
EXCEL_PATH = os.path.join(BASE_DIR, 'meta_with_outcomes_and_classes.xlsx')
OUTPUT_JSON = os.path.join(BASE_DIR, 'clips_json.json')

def parse_filename(filename):
    """Parse filename to extract video_id, clip_id, and type."""
    # Type 1 pattern: IMG_0005_4_aligned_clip_007.npy
    type1_pattern = r'IMG_(\d+)_\d+_aligned_clip_(\d+)\.npy'
    # Type 2 pattern: IMG_0103_clip_007_aligned.npy
    type2_pattern = r'IMG_(\d+)_clip_(\d+)_aligned\.npy'
    
    type1_match = re.match(type1_pattern, filename)
    type2_match = re.match(type2_pattern, filename)
    
    if type1_match:
        video_id = type1_match.group(1)
        clip_id = type1_match.group(2)
        video_type = 'aug'
    elif type2_match:
        video_id = type2_match.group(1)
        clip_id = type2_match.group(2)
        video_type = 'original'
    else:
        return None
    
    return {
        'video_id': video_id,
        'clip_id': clip_id,
        'video_type': video_type
    }

def convert_to_serializable(obj):
    """Convert pandas objects to JSON serializable format."""
    if pd.isna(obj):
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
        return float(obj)
    return obj

def main():
    # Read Excel file
    try:
        df = pd.read_excel(EXCEL_PATH)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Initialize results dictionary
    clips_data = {}
    unmatched_clips = []
    
    # Process all .npy files
    npy_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.npy')]
    total_clips = len(npy_files)
    
    for filename in npy_files:
        parsed = parse_filename(filename)
        if not parsed:
            print(f"Warning: Could not parse filename: {filename}")
            continue
            
        video_id = parsed['video_id']
        clip_id = parsed['clip_id']
        video_type = parsed['video_type']
        
        # Find matching row in Excel
        excel_filename = f"IMG_{video_id}.MP4"
        meta_row = df[df['file_name'] == excel_filename]
        
        if meta_row.empty:
            unmatched_clips.append(filename)
            continue
            
        # Convert meta info to serializable format
        meta_info = meta_row.iloc[0].to_dict()
        meta_info = {k: convert_to_serializable(v) for k, v in meta_info.items()}
            
        # Create entry for this clip
        clips_data[filename] = {
            'filename': filename,
            'video_id': video_id,
            'clip_id': clip_id,
            'video_type': video_type,
            'meta_info': meta_info
        }
    
    # Save to JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(clips_data, f, indent=2)
    
    # Print statistics
    print(f"\nProcessing complete!")
    print(f"Total clips found: {total_clips}")
    print(f"Successfully processed clips: {len(clips_data)}")
    print(f"Unmatched clips: {len(unmatched_clips)}")
    
    if unmatched_clips:
        print("\nUnmatched clips:")
        for clip in unmatched_clips:
            print(f"- {clip}")

if __name__ == "__main__":
    main() 