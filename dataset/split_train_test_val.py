import os
import random
import json

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
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
    clips = data.get("clips", {})  # Get the clips dictionary

# Dictionary to hold filenames by subject ID
subject_files = {}

# Process the clips to organize by subject_id
for filename, attributes in clips.items():
    subject_id = attributes["attributes"].get("subject_id", "")  # Get subject_id from attributes

    if subject_id:  # Ensure subject_id is not empty
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(filename)

# Split the data into train, test, and validation sets
for subject_id, files in subject_files.items():
    # Handle subjects with only one video
    if len(files) == 1:
        val_files.extend(files)  # Assign to validation set
    else:
        random.shuffle(files)  # Shuffle the filenames
        total_files = len(files)

        train_count = int(0.7 * total_files)
        test_count = int(0.15 * total_files)

        train_files.extend(files[:train_count])
        test_files.extend(files[train_count:train_count + test_count])
        val_files.extend(files[train_count + test_count:])

# Print debugging information
print(f"Total number of files in JSON: {len(clips)}")
print(f"Total number of subjects: {len(subject_files)}")
print(f"Number of training files: {len(train_files)}")
print(f"Number of testing files: {len(test_files)}")
print(f"Number of validation files: {len(val_files)}")


# Write the filenames to the output files
def write_file(file_list, filename):
    with open(filename, 'w') as f:
        for file in file_list:
            f.write(f"{file}\n")


write_file(train_files, output_train_file)
write_file(test_files, output_test_file)
write_file(val_files, output_val_file)

print("Train, test, and validation files have been generated.")
