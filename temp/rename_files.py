import os
import re
from pathlib import Path

def rename_files(directory):
    # Convert directory to Path object
    dir_path = Path(directory)
    
    # Counter for renamed files
    renamed_count = 0
    
    # Pattern to match files like IMG_0003_1_aligned_clip_001.npy
    pattern = re.compile(r'(.+)_aligned_clip_(\d+)\.npy$')
    
    # Find all .npy files in the directory
    npy_files = list(dir_path.glob('*.npy'))
    matching_files = [f for f in npy_files if pattern.match(f.name)]
    
    print(f"Found {len(matching_files)} files matching the pattern '_aligned_clip_'")
    
    # Rename matching files
    for file_path in matching_files:
        match = pattern.match(file_path.name)
        if match:
            base_name = match.group(1)
            clip_number = match.group(2)
            new_name = f"{base_name}_clip_{clip_number}_aligned.npy"
            new_path = file_path.parent / new_name
            
            try:
                file_path.rename(new_path)
                renamed_count += 1
                print(f"Renamed: {file_path.name} -> {new_name}")
            except Exception as e:
                print(f"Error renaming {file_path.name}: {str(e)}")
    
    print(f"\nRenaming complete. {renamed_count} files renamed successfully.")

if __name__ == "__main__":
    # Directory containing the files to rename
    target_dir = "/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2"
    
    # Confirm with user before proceeding
    print(f"This script will rename .npy files in: {target_dir}")
    print("Files matching the pattern '*_aligned_clip_*.npy' will be renamed to '*_clip_*_aligned.npy'")
    response = input("Do you want to proceed? (yes/no): ")
    
    if response.lower() == 'yes':
        rename_files(target_dir)
    else:
        print("Operation cancelled.") 