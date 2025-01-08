import os

# Define paths
video_directory = r"C:\pain\BioVid_224_video"  # Change to your directory
output_train_file = r'C:\pain\BioVid_224_video\train.txt'
output_val_file = r'C:\pain\BioVid_224_video\val.txt'

# Prepare lists to hold filenames
train_files = []
val_files = []

# Dictionary to hold filenames by subject
subject_files = {}

# Define low expression subjects and normal expression subjects for validation
low_expression_subjects = [
    "100914_m_39", "101114_w_37", "082315_w_60", "083114_w_55", "083109_m_60"
]

normal_expression_subjects = {
    '20-35': [
        "072514_m_27", "080309_m_29", "112016_m_25", "112310_m_20",
        "092813_w_24", "112809_w_23", "112909_w_20"
    ],
    '36-50': [
        "071313_m_41", "101309_m_48", "101609_m_36", "091809_w_43",
        "102214_w_36", "102316_w_50", "112009_w_43"
    ],
    '51-65': [
        "101814_m_58", "101908_m_61", "102309_m_61", "112209_m_51",
        "112610_w_60", "112914_w_51", "120514_w_56"
    ]
}
excluded_subjects = [
    "082315_w_60", "082414_m_64", "082909_m_47", "083009_w_42",
    "083013_w_47", "083109_m_60", "083114_w_55", "091914_m_46",
    "092009_m_54", "092014_m_56", "092509_w_51", "092714_m_64",
    "100514_w_51", "100914_m_39", "101114_w_37", "101209_w_61",
    "101809_m_59", "101916_m_40", "111313_m_64", "120614_w_61"
]
# Combine all validation subjects into a single list
validation_subjects = set(low_expression_subjects)
for subjects in normal_expression_subjects.values():
    validation_subjects.update(subjects)
print(f"Total validation subjects: {len(validation_subjects)}")
# Read video files from the directory
for filename in os.listdir(video_directory):
    if filename.endswith(".mp4"):  # Adjust based on your video file extension
        subject_id = filename.split('-')[0]
        experiment_type = filename.split('-')[1]
        # construct a binary classification problem only using no pain and extreme pain
        # if experiment_type not in ["BL1", "PA4"]:
        #     continue  # Skip files that do not meet the criteria

        # exclude subjects who are in low expression sets.
        if subject_id in excluded_subjects:
            continue
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(filename)

# Assign files to validation and training sets based on specified subjects
for subject_id, files in subject_files.items():
    if subject_id in validation_subjects:
        val_files.extend(files)  # Add to validation if it's a specified subject
    else:
        train_files.extend(files)  # Otherwise, add to training

# Write the filenames to the output files
def write_file(file_list, filename):
    with open(filename, 'w') as f:
        for file in file_list:
            f.write(f"{file}\n")

write_file(train_files, output_train_file)
write_file(val_files, output_val_file)

# Debugging information
print(f"Total training files: {len(train_files)}")
print(f"Total validation files: {len(val_files)}")

print("Train and validation files have been generated.")
