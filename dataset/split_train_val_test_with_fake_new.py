import os
import random
from collections import defaultdict

# Helper functions
def is_type1(filename):
    """Check if a file is Type 1."""
    return "_aligned.mp4" in filename and not any(prefix in filename for prefix in ["F", "M"])

def is_type2(filename):
    """Check if a file is Type 2."""
    return any(prefix in filename for prefix in ["F", "M"]) and "_aligned.mp4" not in filename

def extract_core_id(filename):
    """Extract the core identifier from a filename."""
    if is_type1(filename):
        return filename.replace("_aligned.mp4", "")
    elif is_type2(filename):
        return "-".join(filename.split("-")[1:]).replace(".mp4", "")
    return None

def is_related(type2_video, type1_video):
    """Check if a Type 2 video is related to a Type 1 video."""
    return extract_core_id(type2_video) == extract_core_id(type1_video)

def ensure_equal_distribution(train_type2, val_type2):
    """Ensure equal distribution of source and subject IDs in train and val sets."""
    # Count source and subject IDs in train and val
    train_source_counts = defaultdict(int)
    train_subject_counts = defaultdict(int)
    val_source_counts = defaultdict(int)
    val_subject_counts = defaultdict(int)

    for video in train_type2:
        source_id = video.split("-")[0]
        subject_id = video.split("-")[1].split("_")[0]
        train_source_counts[source_id] += 1
        train_subject_counts[subject_id] += 1

    for video in val_type2:
        source_id = video.split("-")[0]
        subject_id = video.split("-")[1].split("_")[0]
        val_source_counts[source_id] += 1
        val_subject_counts[subject_id] += 1

    # Print distribution
    print("Train Source ID Distribution:", dict(train_source_counts))
    print("Train Subject ID Distribution:", dict(train_subject_counts))
    print("Val Source ID Distribution:", dict(val_source_counts))
    print("Val Subject ID Distribution:", dict(val_subject_counts))

def save_to_file(file_list, filename):
    """Save a list of files to a text file."""
    with open(filename, "w") as f:
        for file in file_list:
            f.write(file + "\n")

# Main script
def main(folder_path):
    # Step 1: Read all .mp4 files
    all_videos = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    print(f"Total videos: {len(all_videos)}")

    # Step 2: Separate Type 1 and Type 2 videos
    type1_videos = [video for video in all_videos if is_type1(video)]
    type2_videos = [video for video in all_videos if is_type2(video)]
    print(f"Type 1 videos: {len(type1_videos)}")
    print(f"Type 2 videos: {len(type2_videos)}")

    # Step 3: Create the test set (1152 Type 1 videos)
    test_set = random.sample(type1_videos, 1152)
    remaining_type1 = [video for video in type1_videos if video not in test_set]

    # Step 4: Ensure no related Type 2 videos are in train/val
    test_core_ids = set(extract_core_id(video) for video in test_set)
    filtered_type2 = [video for video in type2_videos if not any(is_related(video, test_video) for test_video in test_set)]

    # Step 5: Split remaining Type 1 videos into train and val
    train_type1 = remaining_type1[:5380]
    val_type1 = remaining_type1[5380:5380 + 1152]

    # Step 6: Randomly sample Type 2 videos for train and val
    # Ensure the number of Type 2 videos matches the number of Type 1 videos in each set
    train_type2 = random.sample(filtered_type2, 5380)
    remaining_type2 = [video for video in filtered_type2 if video not in train_type2]
    val_type2 = random.sample(remaining_type2, 1152)

    # Step 7: Combine train and val sets
    train_set = train_type1 + train_type2
    val_set = val_type1 + val_type2

    # Step 8: Ensure equal distribution of source and subject IDs
    ensure_equal_distribution(train_type2, val_type2)

    # Step 9: Print composition of each set
    print("\nComposition of Sets:")
    print(f"Test Set: {len(test_set)} (Type 1: {len(test_set)}, Type 2: 0)")
    print(f"Train Set: {len(train_set)} (Type 1: {len(train_type1)}, Type 2: {len(train_type2)})")
    print(f"Val Set: {len(val_set)} (Type 1: {len(val_type1)}, Type 2: {len(val_type2)})")

    # Step 10: Save train, val, and test sets to separate files
    save_to_file(train_set, "train_set.txt")
    save_to_file(val_set, "val_set.txt")
    save_to_file(test_set, "test_set.txt")
    print("\nTrain, val, and test sets saved to train_set.txt, val_set.txt, and test_set.txt.")


# Run the script
if __name__ == "__main__":
    folder_path = r"C:\pain\BioVid_224_video"  # Replace with your folder path
    main(folder_path)