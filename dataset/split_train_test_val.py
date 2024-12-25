import os
import random

# Define paths
video_directory = r"C:\pain\BioVid_224_video"  # Change to your directory
output_train_file = r'C:\pain\BioVid_224_video\train.txt'
output_test_file = r'C:\pain\BioVid_224_video\test.txt'
output_val_file = r'C:\pain\BioVid_224_video\val.txt'

# Prepare lists to hold filenames
train_files = []
test_files = []
val_files = []

# Dictionary to hold filenames by subject
subject_files = {}

# Read video files from the directory
for filename in os.listdir(video_directory):
    if filename.endswith(".mp4"):  # Adjust based on your video file extension
        subject_id = filename.split('_')[0]
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(filename)

# Split the data into train, test, and validation sets
for subject_id, files in subject_files.items():
    random.shuffle(files)  # Shuffle the filenames
    total_files = len(files)
    train_count = int(0.7 * total_files)
    test_count = int(0.15 * total_files)

    train_files.extend(files[:train_count])
    test_files.extend(files[train_count:train_count + test_count])
    val_files.extend(files[train_count + test_count:])


# Write the filenames to the output files
def write_file(file_list, filename):
    with open(filename, 'w') as f:
        for file in file_list:
            f.write(f"{file}\n")


write_file(train_files, output_train_file)
write_file(test_files, output_test_file)
write_file(val_files, output_val_file)

print("Train, test, and validation files have been generated.")
