import json
import os
import random
import re
from collections import defaultdict


def normalize_subject_id(subject_id):
    """
    Normalize subject IDs to handle variant naming formats.
    For example, "F08-082909" and "082909" should be considered the same subject.

    Args:
        subject_id: Original subject ID string

    Returns:
        Normalized subject ID
    """
    # Extract numeric part if it exists within various formats
    # This handles cases like "F08-082909" -> "082909" or "071309_w_21" -> "071309"
    numeric_match = re.search(r'(\d+)', subject_id)
    if numeric_match:
        return numeric_match.group(1)
    return subject_id


def create_subject_independent_splits(json_path, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                      balance_classes=False, exclude_prefix=None):
    """
    Create subject-independent train/val/test splits from BioVid dataset.
    Ensures that samples from the same subject appear in only one split.

    Args:
        json_path: Path to the biovid_info.json file
        output_dir: Directory to save the split files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        balance_classes: Whether to balance classes in each split
        exclude_prefix: Prefix to exclude from subject IDs (e.g., "fa")
    """
    # Ensure ratios add up to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Split ratios must sum to 1"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter to only include samples with multiclass "5" value of 0 or 3
    filtered_clips = {}
    for clip_name, clip_info in data['clips'].items():
        multiclass_value = clip_info['attributes']['multiclass'].get('5')
        if multiclass_value in [0, 4]:
            filtered_clips[clip_name] = clip_info

    # Group clips by subject AND class
    subject_class_clips = defaultdict(lambda: defaultdict(list))

    # Track subject IDs and their mapping to normalized IDs
    subject_to_normalized = {}
    normalized_subjects = set()
    excluded_subjects = set()

    for clip_name, clip_info in filtered_clips.items():
        subject_id = clip_info['attributes'].get('subject_id', '')

        # Normalize the subject ID and track the mapping
        normalized_id = normalize_subject_id(subject_id)
        subject_to_normalized[subject_id] = normalized_id

        # Skip subjects with the excluded prefix
        if exclude_prefix and (subject_id.lower().startswith(exclude_prefix.lower()) or
                               normalized_id.lower().startswith(exclude_prefix.lower())):
            excluded_subjects.add(normalized_id)
            continue

        normalized_subjects.add(normalized_id)

        # Get the class and add to the appropriate group
        class_label = clip_info['attributes']['multiclass']['5']
        subject_class_clips[normalized_id][class_label].append(clip_name)

    # Print overall dataset statistics
    print(f"Total subjects (after filtering): {len(normalized_subjects)}")
    if exclude_prefix:
        print(f"Excluded subjects with prefix '{exclude_prefix}': {len(excluded_subjects)}")
    print(f"Total clips (after filtering for classes 0 and 3): {len(filtered_clips)}")

    class_counts = defaultdict(int)
    for subject_clips in subject_class_clips.values():
        for class_label, clips in subject_clips.items():
            class_counts[class_label] += len(clips)

    print("\nClass distribution:")
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} samples")

    # Split subjects into train/val/test groups
    normalized_subjects_list = list(normalized_subjects)
    random.shuffle(normalized_subjects_list)

    n_subjects = len(normalized_subjects_list)
    n_train_subjects = int(n_subjects * train_ratio)
    n_val_subjects = int(n_subjects * val_ratio)

    train_subjects = set(normalized_subjects_list[:n_train_subjects])
    val_subjects = set(normalized_subjects_list[n_train_subjects:n_train_subjects + n_val_subjects])
    test_subjects = set(normalized_subjects_list[n_train_subjects + n_val_subjects:])

    # Collect clips for each split
    train_clips_by_class = defaultdict(list)
    val_clips_by_class = defaultdict(list)
    test_clips_by_class = defaultdict(list)

    for subject, class_clips in subject_class_clips.items():
        target_split = None
        if subject in train_subjects:
            target_split = train_clips_by_class
        elif subject in val_subjects:
            target_split = val_clips_by_class
        elif subject in test_subjects:
            target_split = test_clips_by_class

        if target_split is not None:
            for class_label, clips in class_clips.items():
                target_split[class_label].extend(clips)

    # If balancing classes, equalize the number of samples per class in each split
    if balance_classes:
        for class_split in [train_clips_by_class, val_clips_by_class, test_clips_by_class]:
            min_samples = min(len(clips) for clips in class_split.values()) if class_split else 0
            for class_label in list(class_split.keys()):
                if len(class_split[class_label]) > min_samples:
                    # Randomly subsample to the minimum size
                    random.shuffle(class_split[class_label])
                    class_split[class_label] = class_split[class_label][:min_samples]

    # Combine all classes for each split
    train_set = []
    val_set = []
    test_set = []

    for class_label in class_counts.keys():
        if class_label in train_clips_by_class:
            train_set.extend(train_clips_by_class[class_label])
        if class_label in val_clips_by_class:
            val_set.extend(val_clips_by_class[class_label])
        if class_label in test_clips_by_class:
            test_set.extend(test_clips_by_class[class_label])

    # Shuffle each set
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    # Print split statistics
    print("\nSubject distribution:")
    print(f"Training set: {len(train_subjects)} subjects")
    print(f"Validation set: {len(val_subjects)} subjects")
    print(f"Test set: {len(test_subjects)} subjects")

    print("\nSample distribution:")
    print(f"Training set: {len(train_set)} samples")
    for class_label in class_counts.keys():
        class_count = sum(
            1 for clip in train_set if filtered_clips[clip]['attributes']['multiclass']['5'] == class_label)
        print(f"  Class {class_label}: {class_count} samples")

    print(f"Validation set: {len(val_set)} samples")
    for class_label in class_counts.keys():
        class_count = sum(1 for clip in val_set if filtered_clips[clip]['attributes']['multiclass']['5'] == class_label)
        print(f"  Class {class_label}: {class_count} samples")

    print(f"Test set: {len(test_set)} samples")
    for class_label in class_counts.keys():
        class_count = sum(
            1 for clip in test_set if filtered_clips[clip]['attributes']['multiclass']['5'] == class_label)
        print(f"  Class {class_label}: {class_count} samples")

    # Write to output files
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_set))

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_set))

    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_set))

    print(f"\nSubject-independent dataset splits successfully saved to {output_dir}")

    return train_set, val_set, test_set


def create_balanced_dataset_splits(json_path, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create balanced train/val/test splits from BioVid dataset

    Args:
        json_path: Path to the biovid_info.json file
        output_dir: Directory to save the split files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    # Ensure ratios add up to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Split ratios must sum to 1"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Group clips by class (multiclass "5" value of 0 or 3)
    clips_by_class = defaultdict(list)

    for clip_name, clip_info in data['clips'].items():
        multiclass_value = clip_info['attributes']['multiclass'].get('5')

        # Only include clips with multiclass "5" of 0 or 3
        if multiclass_value in [0, 3]:
            clips_by_class[multiclass_value].append(clip_name)

    # Print class distribution
    print("Class distribution:")
    for class_label, clips in clips_by_class.items():
        print(f"Class {class_label}: {len(clips)} samples")

    # Calculate split sizes for each class
    train_samples = {}
    val_samples = {}
    test_samples = {}

    for class_label, clips in clips_by_class.items():
        # Shuffle clips for randomization
        random.shuffle(clips)

        # Calculate split indices
        n_samples = len(clips)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        # Split the data
        train_samples[class_label] = clips[:n_train]
        val_samples[class_label] = clips[n_train:n_train + n_val]
        test_samples[class_label] = clips[n_train + n_val:]

    # Combine all classes for each split
    train_set = []
    val_set = []
    test_set = []

    for class_label in clips_by_class.keys():
        train_set.extend(train_samples[class_label])
        val_set.extend(val_samples[class_label])
        test_set.extend(test_samples[class_label])

    # Shuffle the combined sets
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    # Print split statistics
    print("\nSplit statistics:")
    print(f"Training set: {len(train_set)} samples")
    for class_label in clips_by_class.keys():
        class_count = sum(
            1 for clip in train_set if data['clips'][clip]['attributes']['multiclass']['5'] == class_label)
        print(f"  Class {class_label}: {class_count} samples")

    print(f"Validation set: {len(val_set)} samples")
    for class_label in clips_by_class.keys():
        class_count = sum(1 for clip in val_set if data['clips'][clip]['attributes']['multiclass']['5'] == class_label)
        print(f"  Class {class_label}: {class_count} samples")

    print(f"Test set: {len(test_set)} samples")
    for class_label in clips_by_class.keys():
        class_count = sum(1 for clip in test_set if data['clips'][clip]['attributes']['multiclass']['5'] == class_label)
        print(f"  Class {class_label}: {class_count} samples")

    # Write to output files
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_set))

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_set))

    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_set))

    print(f"\nDataset splits successfully saved to {output_dir}")

    return train_set, val_set, test_set


def create_perfectly_balanced_splits(json_path, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create perfectly balanced train/val/test splits from BioVid dataset,
    ensuring each split has the exact same number of samples from each class.

    Args:
        json_path: Path to the biovid_info.json file
        output_dir: Directory to save the split files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    # Ensure ratios add up to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Split ratios must sum to 1"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Group clips by class (multiclass "5" value of 0 or 3)
    clips_by_class = defaultdict(list)

    for clip_name, clip_info in data['clips'].items():
        multiclass_value = clip_info['attributes']['multiclass'].get('5')

        # Only include clips with multiclass "5" of 0 or 3
        if multiclass_value in [0, 3]:
            clips_by_class[multiclass_value].append(clip_name)

    # Print class distribution
    print("Class distribution:")
    class_sizes = []
    for class_label, clips in clips_by_class.items():
        print(f"Class {class_label}: {len(clips)} samples")
        class_sizes.append(len(clips))

    # Find minimum class size for perfect balancing
    min_class_size = min(class_sizes)
    print(f"\nUsing {min_class_size} samples per class for balanced dataset")

    # Create balanced splits
    train_set = []
    val_set = []
    test_set = []

    for class_label, clips in clips_by_class.items():
        # Shuffle clips for randomization
        random.shuffle(clips)

        # Limit to minimum class size
        balanced_clips = clips[:min_class_size]

        # Calculate split indices
        n_train = int(min_class_size * train_ratio)
        n_val = int(min_class_size * val_ratio)

        # Split the data
        train_set.extend(balanced_clips[:n_train])
        val_set.extend(balanced_clips[n_train:n_train + n_val])
        test_set.extend(balanced_clips[n_train + n_val:])

    # Shuffle the combined sets
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    # Print split statistics
    print("\nSplit statistics:")
    print(f"Training set: {len(train_set)} samples")
    for class_label in clips_by_class.keys():
        class_count = sum(
            1 for clip in train_set if data['clips'][clip]['attributes']['multiclass']['5'] == class_label)
        print(f"  Class {class_label}: {class_count} samples")

    print(f"Validation set: {len(val_set)} samples")
    for class_label in clips_by_class.keys():
        class_count = sum(1 for clip in val_set if data['clips'][clip]['attributes']['multiclass']['5'] == class_label)
        print(f"  Class {class_label}: {class_count} samples")

    print(f"Test set: {len(test_set)} samples")
    for class_label in clips_by_class.keys():
        class_count = sum(1 for clip in test_set if data['clips'][clip]['attributes']['multiclass']['5'] == class_label)
        print(f"  Class {class_label}: {class_count} samples")

    # Write to output files
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_set))

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_set))

    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_set))

    print(f"\nPerfectly balanced dataset splits successfully saved to {output_dir}")

    return train_set, val_set, test_set


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create balanced train/val/test splits for BioVid dataset")
    parser.add_argument("--json_path", type=str, required=True, help="Path to biovid_info.json file")
    parser.add_argument("--output_dir", type=str, default="./splits", help="Directory to save split files")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion for training set")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proportion for validation set")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proportion for test set")
    parser.add_argument("--perfect_balance", action="store_true", help="Create perfectly balanced splits")
    parser.add_argument("--subject_independent", action="store_true",
                        help="Create subject-independent splits (same subject won't appear in multiple sets)")
    parser.add_argument("--balance_classes", action="store_true",
                        help="Balance classes in each split (used with subject_independent)")
    parser.add_argument("--exclude_prefix", type=str, default=None,
                        help="Exclude subjects with this prefix (e.g., 'fa')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-10:
        parser.error("Split ratios must sum to 1")

    # Create splits
    if args.subject_independent:
        create_subject_independent_splits(
            args.json_path,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.balance_classes,
            args.exclude_prefix
        )
    elif args.perfect_balance:
        create_perfectly_balanced_splits(
            args.json_path,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )
    else:
        create_balanced_dataset_splits(
            args.json_path,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )