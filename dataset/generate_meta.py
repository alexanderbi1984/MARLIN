import os
import csv
import argparse
import random
from pathlib import Path
import shutil


def scan_and_split_subfolders(root_dir, output_dir, train_ratio=0.8, copy_files=False):
    """
    Scan all subfolders within the given directory, split them into training and testing sets,
    and create CSV files with folder paths and number of files in each folder.

    Args:
        root_dir (str): Path to the root directory to scan
        output_dir (str): Directory to save the CSV files
        train_ratio (float): Ratio of folders to include in the training set (0.0-1.0)
        copy_files (bool): Whether to copy files to new train/test directories
    """
    # Convert paths to absolute paths
    root_dir = os.path.abspath(root_dir)
    output_dir = os.path.abspath(output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Lists to store results
    all_folder_data = []

    # Walk through all subfolders
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the root directory itself
        if dirpath == root_dir:
            continue

        # Get relative path from the root directory
        rel_path = os.path.relpath(dirpath, root_dir)

        # Count the number of files in this directory (not including subdirectories)
        file_count = len(filenames)

        # Only include directories that have files
        if file_count > 0:
            all_folder_data.append((rel_path, file_count))

    # Randomize the data
    random.shuffle(all_folder_data)

    # Calculate split point
    split_index = int(len(all_folder_data) * train_ratio)

    # Split into training and testing sets
    train_data = all_folder_data[:split_index]
    test_data = all_folder_data[split_index:]

    # Write results to CSV files
    # All data
    all_csv_path = os.path.join(output_dir, "all_metadata.csv")
    with open(all_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'len'])  # Header row
        writer.writerows(all_folder_data)

    # Training data
    train_csv_path = os.path.join(output_dir, "train_metadata.csv")
    with open(train_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'len'])  # Header row
        writer.writerows(train_data)

    # Testing data
    test_csv_path = os.path.join(output_dir, "test_metadata.csv")
    with open(test_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'len'])  # Header row
        writer.writerows(test_data)

    print(f"Scan complete. Found {len(all_folder_data)} subfolders with files.")
    print(f"Split into {len(train_data)} training folders and {len(test_data)} testing folders.")
    print(f"All metadata saved to {all_csv_path}")
    print(f"Training metadata saved to {train_csv_path}")
    print(f"Testing metadata saved to {test_csv_path}")

    # If requested, copy files to train/test directories
    if copy_files:
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")

        # Create directories
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Copy training files
        for rel_path, _ in train_data:
            src_dir = os.path.join(root_dir, rel_path)
            dst_dir = os.path.join(train_dir, rel_path)

            # Create destination directory
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)

            # Copy directory
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)

        # Copy testing files
        for rel_path, _ in test_data:
            src_dir = os.path.join(root_dir, rel_path)
            dst_dir = os.path.join(test_dir, rel_path)

            # Create destination directory
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)

            # Copy directory
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)

        print(f"Files copied to {train_dir} and {test_dir}")


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Scan subfolders, split into train/test sets, and count files")
    parser.add_argument("root_dir", help="Root directory to scan")
    parser.add_argument("--output-dir", "-o", default="./metadata",
                        help="Directory to save output CSV files (default: ./metadata)")
    parser.add_argument("--train-ratio", "-t", type=float, default=0.8,
                        help="Ratio of folders to include in training set (default: 0.8)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--copy-files", "-c", action="store_true",
                        help="Copy files to train/test directories")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Check if the root directory exists
    if not os.path.isdir(args.root_dir):
        print(f"Error: Directory '{args.root_dir}' does not exist.")
        exit(1)

    # Check if train ratio is valid
    if args.train_ratio <= 0.0 or args.train_ratio >= 1.0:
        print(f"Error: Train ratio must be between 0.0 and 1.0 (got {args.train_ratio})")
        exit(1)

    scan_and_split_subfolders(args.root_dir, args.output_dir, args.train_ratio, args.copy_files)