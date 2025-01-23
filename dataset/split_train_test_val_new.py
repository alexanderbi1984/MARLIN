import os
import random
import json
from collections import defaultdict

# Define paths
video_directory = r"C:\pain\json4marlin"  # Change to your directory
output_train_file = os.path.join(video_directory, 'train.txt')
output_test_file = os.path.join(video_directory, 'test.txt')
output_val_file = os.path.join(video_directory, 'val.txt')

# Prepare lists to hold filenames
train_files = []
test_files = []
val_files = []

# Load clips and their attributes from the provided JSON file
json_file_path = os.path.join(video_directory, 'biovid_info.json')  # Change to your JSON file path

# Specify the source to exclude clips
exclude_source = "BioVidGan"  # Change this to the source you want to exclude

with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
    clips = data.get("clips", {})  # Get the clips dictionary

    # Filter clips to exclude the specified source
    filtered_clips = {
        filename: attributes for filename, attributes in clips.items()
        if attributes["attributes"].get("source") != exclude_source  # Exclude the specified source
           and attributes["attributes"].get("multiclass", {}).get("5") in {0, 2, 4}
        # Ensure multiclass["5"] is 0, 2, or 4
    }

# Dictionary to hold filenames by subject ID
subject_files = {}

# Process the filtered clips to organize by subject_id
for filename, attributes in filtered_clips.items():
    subject_id = attributes["attributes"].get("subject_id", "")  # Get subject_id from attributes

    if subject_id:  # Ensure subject_id is not empty
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append((filename, attributes))

# Split the subjects into training, testing, and validation sets
all_subjects = list(subject_files.keys())
random.shuffle(all_subjects)  # Shuffle the subjects for random assignment

# Determine how many subjects to assign to each set
num_subjects = len(all_subjects)
train_subject_count = int(0.7 * num_subjects)
test_subject_count = int(0.15 * num_subjects)

# Assign subjects to training, testing, and validation sets
train_subjects = all_subjects[:train_subject_count]
test_subjects = all_subjects[train_subject_count:train_subject_count + test_subject_count]
val_subjects = all_subjects[train_subject_count + test_subject_count:]

# Collect files for each set based on the assigned subjects
for subject_id in train_subjects:
    train_files.extend(subject_files[subject_id])

for subject_id in test_subjects:
    test_files.extend(subject_files[subject_id])

for subject_id in val_subjects:
    val_files.extend(subject_files[subject_id])

# Function to calculate class distribution
def calculate_class_distribution(file_list):
    class_distribution = defaultdict(int)
    for filename, attributes in file_list:
        class_id = attributes["attributes"].get("multiclass", {}).get("5")  # Assuming '5' is the class key
        if class_id is not None:
            class_distribution[class_id] += 1
    return class_distribution

# Calculate class distributions
train_distribution = calculate_class_distribution(train_files)
test_distribution = calculate_class_distribution(test_files)
val_distribution = calculate_class_distribution(val_files)

# Print debugging information
print(f"Total number of files in JSON: {len(clips)}")
print(f"Total number of filtered clips: {len(filtered_clips)}")
print(f"Total number of subjects: {len(subject_files)}")
print(f"Number of training files: {len(train_files)}")
print(f"Number of testing files: {len(test_files)}")
print(f"Number of validation files: {len(val_files)}")

# Print class distributions
print("Class distribution in training set:", dict(train_distribution))
print("Class distribution in testing set:", dict(test_distribution))
print("Class distribution in validation set:", dict(val_distribution))

# Check for duplicates
train_set = set(file[0] for file in train_files)
test_set = set(file[0] for file in test_files)
val_set = set(file[0] for file in val_files)

# Check for overlaps
overlap_train_test = train_set.intersection(test_set)
overlap_train_val = train_set.intersection(val_set)
overlap_test_val = test_set.intersection(val_set)

print(f"Number of overlapping files between train and test: {len(overlap_train_test)}")
print(f"Number of overlapping files between train and validation: {len(overlap_train_val)}")
print(f"Number of overlapping files between test and validation: {len(overlap_test_val)}")

# Write the filenames to the output files
def write_file(file_list, filename):
    with open(filename, 'w') as f:
        for file in file_list:
            f.write(f"{file[0]}\n")  # Write only the filename

write_file(train_files, output_train_file)
write_file(test_files, output_test_file)
write_file(val_files, output_val_file)

print("Train, test, and validation files have been generated.")
