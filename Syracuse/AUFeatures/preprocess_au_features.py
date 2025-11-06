import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def preprocess_au_features(input_dir, output_dir, confidence_threshold=0.9):
    """
    Preprocess Action Units (AU) features from CSV files.
    
    Args:
        input_dir (str): Directory containing raw AU feature CSV files
        output_dir (str): Directory to save processed features
        confidence_threshold (float): Minimum confidence threshold for keeping frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize statistics dictionary
    stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'confidence_threshold': confidence_threshold,
        'files_processed': {}
    }
    
    # Get all CSV files in input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    total_files = len(csv_files)
    
    print(f"Found {total_files} CSV files to process")
    
    # Read first file to check structure
    if csv_files:
        first_file = os.path.join(input_dir, csv_files[0])
        try:
            df = pd.read_csv(first_file)
            print("\nColumns in the first file:")
            print(df.columns.tolist())
            print("\nFirst few rows of data:")
            print(df.head())
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Error reading first file: {str(e)}")
            return
    
    input("Press Enter to continue with processing all files...")
    
    for idx, csv_file in enumerate(csv_files, 1):
        print(f"\nProcessing file {idx}/{total_files}: {csv_file}")
        
        try:
            # Read CSV file
            df = pd.read_csv(os.path.join(input_dir, csv_file))
            
            # Clean column names by stripping whitespace
            df.columns = df.columns.str.strip()
            
            # Record initial number of rows
            initial_rows = len(df)
            
            # Apply filtering criteria
            filtered_df = df[
                (df['confidence'] >= confidence_threshold) & 
                (df['success'] == 1)
            ]
            
            # Record filtered number of rows
            filtered_rows = len(filtered_df)
            
            # Sort by timestamp to ensure chronological order
            filtered_df = filtered_df.sort_values('timestamp')
            
            # Save processed features
            output_path = os.path.join(output_dir, f"processed_{csv_file}")
            filtered_df.to_csv(output_path, index=False)
            
            # Calculate statistics
            rows_removed = initial_rows - filtered_rows
            percentage_removed = round((rows_removed / initial_rows * 100), 2) if initial_rows > 0 else 0
            
            # Record statistics for this file
            stats['files_processed'][csv_file] = {
                'initial_rows': initial_rows,
                'filtered_rows': filtered_rows,
                'rows_removed': rows_removed,
                'percentage_removed': percentage_removed
            }
            
            print(f"Initial rows: {initial_rows}")
            print(f"Rows after filtering: {filtered_rows}")
            print(f"Removed {rows_removed} rows ({percentage_removed}%)")
            print(f"Saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            stats['files_processed'][csv_file] = {
                'error': str(e)
            }
            continue
    
    # Calculate and add summary statistics
    successful_files = [f for f in stats['files_processed'].keys() if 'error' not in stats['files_processed'][f]]
    stats['summary'] = {
        'total_files': total_files,
        'successful_files': len(successful_files),
        'failed_files': total_files - len(successful_files),
        'total_initial_rows': sum(stats['files_processed'][f]['initial_rows'] for f in successful_files),
        'total_filtered_rows': sum(stats['files_processed'][f]['filtered_rows'] for f in successful_files),
    }
    
    # Save statistics to a JSON file
    stats_file = os.path.join(output_dir, 'preprocessing_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nPreprocessing complete!")
    print(f"Successfully processed: {stats['summary']['successful_files']}/{total_files} files")
    print(f"Statistics saved to: {stats_file}")

if __name__ == "__main__":
    # Set paths
    input_dir = r"C:\pain\syracus\AU_features"  # Source directory with raw AU features
    output_dir = os.path.join(os.path.dirname(__file__), "processed")  # Output in a 'processed' subdirectory
    
    # Run preprocessing
    preprocess_au_features(input_dir, output_dir) 