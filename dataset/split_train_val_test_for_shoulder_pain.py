import json
import random
from collections import defaultdict


def generate_train_val_test_sets(metadata_file, output_dir):
    # Load metadata from JSON
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    clips = metadata["clips"]

    # Organize the data by subject and class
    subject_class_map = defaultdict(lambda: defaultdict(list))

    for filename, attributes in clips.items():
        subject_id = attributes["attributes"]["subject_id"]
        class_label = attributes["attributes"]["multiclass"].get("6")  # Get the class label where "6" is the key

        if class_label is not None:
            subject_class_map[subject_id][class_label].append(filename)

    # Prepare lists to hold the final train, val, test sets
    train_set, val_set, test_set = set(), set(), set()
    subjects = list(subject_class_map.keys())

    # Ensure that each set has at least one example from every class
    class_labels = set()
    for subject in subjects:
        class_labels.update(subject_class_map[subject].keys())

    # Shuffle subjects for randomness
    random.shuffle(subjects)

    # Allocate subjects to train, val, and test sets
    for subject in subjects:
        # Randomly decide where to put the subject
        if len(train_set) < len(class_labels) * 2:  # Ensure at least one example of each class
            train_set.add(subject)
        elif len(val_set) < len(class_labels) and len(train_set) >= len(class_labels) * 2:
            val_set.add(subject)
        else:
            test_set.add(subject)

    # Collect filenames for each set
    train_filenames = []
    val_filenames = []
    test_filenames = []

    for subject in train_set:
        for class_label, filenames in subject_class_map[subject].items():
            if filenames:
                train_filenames.append(random.choice(filenames))  # Select one random filename

    for subject in val_set:
        for class_label, filenames in subject_class_map[subject].items():
            if filenames:
                val_filenames.append(random.choice(filenames))  # Select one random filename

    for subject in test_set:
        for class_label, filenames in subject_class_map[subject].items():
            if filenames:
                test_filenames.append(random.choice(filenames))  # Select one random filename

    # Save to text files
    with open(f"{output_dir}/train.txt", 'w') as f:
        for filename in train_filenames:
            f.write(f"{filename}\n")

    with open(f"{output_dir}/val.txt", 'w') as f:
        for filename in val_filenames:
            f.write(f"{filename}\n")

    with open(f"{output_dir}/test.txt", 'w') as f:
        for filename in test_filenames:
            f.write(f"{filename}\n")


# Call the function with the appropriate paths
generate_train_val_test_sets(r"C:\pain\ShoulderPain_video\cropped\biovid_info.json", r"C:\pain\ShoulderPain_video\cropped")  # Update the output directory as needed
