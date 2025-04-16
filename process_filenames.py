import os
import csv
from pathlib import Path

def process_filenames(folder_path):
    """
    Process filenames in the specified folder:
    1. Remove file extensions
    2. Remove '_aligned' suffix
    3. Save results to a CSV file
    """
    # Convert folder path to Path object
    folder = Path(folder_path)
    
    # Check if folder exists
    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    # Get all files in the folder
    files = [f for f in folder.iterdir() if f.is_file()]
    
    # Process filenames
    processed_names = []
    for file in files:
        # Get filename without extension
        name_without_ext = file.stem
        
        # Remove '_aligned' suffix if it exists
        if name_without_ext.endswith('_aligned'):
            name_without_ext = name_without_ext[:-8]  # Remove '_aligned'
        
        processed_names.append(name_without_ext)
    
    # Save to CSV file
    output_csv = folder / 'processed_filenames.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original Filename', 'Processed Filename'])
        for file, processed_name in zip(files, processed_names):
            writer.writerow([file.name, processed_name])
    
    print(f"Processed {len(processed_names)} filenames")
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    folder_path = r"C:\pain\syracus\aug_crop_videos_new"
    process_filenames(folder_path) 