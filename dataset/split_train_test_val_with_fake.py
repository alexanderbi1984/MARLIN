import os
import random

# Assuming subject_files is already populated with filenames
train_files = []
test_files = []
val_files = []

type_1_files = {}
type_2_files = {}
# Dictionary to hold filenames by subject
subject_files = {}
# Define paths
video_directory = r"C:\pain\BioVid_224_video"  # Change to your directory
output_train_file = r'C:\pain\BioVid_224_video\train.txt'
output_test_file = r'C:\pain\BioVid_224_video\test.txt'
output_val_file = r'C:\pain\BioVid_224_video\val.txt'

# Read video files from the directory
for filename in os.listdir(video_directory):
    if filename.endswith(".mp4"):  # Adjust based on your video file extension
        subject_id = filename.split('_')[0]
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(filename)

# Separate Type 1 and Type 2 videos
for subject_id, files in subject_files.items():
    type_1_files[subject_id] = []
    type_2_files[subject_id] = []

    for filename in files:
        if filename.startswith('F') or filename.startswith('M'):  # Type 2 starts with 'F' or 'M'
            type_2_files[subject_id].append(filename)
        else:  # Type 1
            type_1_files[subject_id].append(filename)

# Shuffle Type 1 files for randomness
for subject_id in type_1_files:
    random.shuffle(type_1_files[subject_id])

# Process Type 1 files for train/test/val allocation
for subject_id, files in type_1_files.items():
    total_files = len(files)
    train_count = int(0.7 * total_files)
    test_count = int(0.15 * total_files)

    # Allocate Type 1 to train/test/val
    train_files.extend(files[:train_count])
    test_files.extend(files[train_count:train_count + test_count])
    val_files.extend(files[train_count + test_count:])

# Count Type 1 videos in the training set
train_type_1_count = len(train_files)

# Initialize lists for Type 2 training and validation videos
type_2_train = []
type_2_val = []

# Process Type 2 files
for subject_id, files in type_2_files.items():
    if not files:  # No Type 2 files for this subject
        continue

    # Shuffle Type 2 files
    random.shuffle(files)

    # Allocate Type 2 videos to training set (up to the number of Type 1 videos in train)
    type_2_to_train = min(len(files), train_type_1_count)
    type_2_train.extend(files[:type_2_to_train])

    # Allocate remaining Type 2 videos to validation set (up to the count of Type 1 in train set)
    remaining_type_2 = files[type_2_to_train:]
    type_2_to_val = min(len(remaining_type_2), train_type_1_count - len(type_2_train))
    type_2_val.extend(remaining_type_2[:type_2_to_val])

# Finalize allocations respecting the exclusivity condition
# Check for Type 1 videos in test set and remove Type 2 counterparts
for t1_file in test_files:
    subject_id = t1_file.split('_')[0]  # Extracting subject ID
    # Remove corresponding Type 2 video
    corresponding_t2_f = f"F{subject_id}-" + t1_file.split('_', 1)[1].replace('.mp4', '') + '.mp4'
    corresponding_t2_m = f"M{subject_id}-" + t1_file.split('_', 1)[1].replace('.mp4', '') + '.mp4'

    type_2_train = [file for file in type_2_train if file != corresponding_t2_f and file != corresponding_t2_m]
    type_2_val = [file for file in type_2_val if file != corresponding_t2_f and file != corresponding_t2_m]

# Extend the final train and val files
train_files.extend(type_2_train)
val_files.extend(type_2_val)

# Shuffle final train, test, and validation sets
random.shuffle(train_files)
random.shuffle(test_files)
random.shuffle(val_files)


# Print the total number of videos in each set and their compositions
def print_set_composition(set_name, video_list):
    type_1_count = sum(1 for f in video_list if not f.startswith('F') and not f.startswith('M'))  # Count Type 1 videos
    type_2_count = sum(1 for f in video_list if f.startswith('F') or f.startswith('M'))   # Count Type 2 videos
    total_count = len(video_list)

    print(f"{set_name} Set:")
    print(f"  Total Videos: {total_count}")
    print(f"  Type 1 Videos: {type_1_count}")
    print(f"  Type 2 Videos: {type_2_count}")
    print("")

# Print compositions for each set
print_set_composition("Training", train_files)
print_set_composition("Testing", test_files)
print_set_composition("Validation", val_files)

# Function to save a list of filenames to a text file
def save_file_list(file_list, filename):
    with open(filename, 'w') as f:
        for file in file_list:
            f.write(f"{file}\n")  # Write each filename on a new line

# Save the train, test, and validation sets to files
save_file_list(train_files, output_train_file)
save_file_list(val_files, output_val_file)
save_file_list(test_files, output_test_file)

print("Train, validation, and test sets have been saved to text files.")