import os
import csv
import argparse
from pathlib import Path


def scan_subfolders(root_dir, output_file):
    """
    Scan all subfolders within the given directory and create a CSV file
    with folder paths and number of files in each folder.

    Args:
        root_dir (str): Path to the root directory to scan
        output_file (str): Path to the output CSV file
    """
    # Convert root_dir to an absolute path
    root_dir = os.path.abspath(root_dir)

    # List to store results
    folder_data = []

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
            folder_data.append((rel_path, file_count))

    # Write results to CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'len'])  # Header row
        writer.writerows(folder_data)

    print(f"Scan complete. Found {len(folder_data)} subfolders with files.")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Scan subfolders and count files")
    parser.add_argument("root_dir", help="Root directory to scan")
    parser.add_argument("--output", "-o", default="folder_metadata.csv",
                        help="Output CSV file (default: folder_metadata.csv)")

    args = parser.parse_args()

    # Check if the root directory exists
    if not os.path.isdir(args.root_dir):
        print(f"Error: Directory '{args.root_dir}' does not exist.")
        exit(1)

    scan_subfolders(args.root_dir, args.output)