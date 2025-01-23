# import json
#
# # Load the first JSON file
# with open(r"C:\pain\ShoulderPain_bp4d_bp4d+_video\cropped\biovid_info.json", 'r') as f1:
#     json_data_1 = json.load(f1)
#
# # Load the second JSON file
# with open(r"C:\pain\BP4D_BP4D+_PAIN_VIDEO\biovid_info.json", 'r') as f2:
#     json_data_2 = json.load(f2)
#
# # Combine the "clips" from both files
# combined_clips = {**json_data_1["clips"], **json_data_2["clips"]}
#
# # Create the combined JSON object
# combined_json = {"clips": combined_clips}
#
# # Save the combined JSON to a new file (optional)
# with open(r"C:\pain\ShoulderPain_bp4d_bp4d+_video\cropped\biovid_info.json", 'w') as outfile:
#     json.dump(combined_json, outfile, indent=4)
#
# # Print the total number of clips
# total_clips = len(combined_clips)
# print(f"Total clips: {total_clips}")

import os
import json

# Define the directory containing the JSON files
json_directory = r"C:\pain\json4marlin"  # Change to your directory

# Initialize an empty dictionary to combine clips
combined_clips = {}

# Initialize a counter for the total number of clips
total_clips = 0

# Iterate over all JSON files in the specified directory
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):  # Process only JSON files
        file_path = os.path.join(json_directory, filename)

        # Load the JSON file
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        # Get the clips from the current JSON file
        clips = json_data.get("clips", {})

        # Count and print the number of records in the current JSON file
        num_clips = len(clips)
        print(f"Number of records in '{filename}': {num_clips}")

        # Combine the clips into the main dictionary
        combined_clips.update(clips)

        # Update the total number of clips
        total_clips += num_clips

# Create the combined JSON object
combined_json = {"clips": combined_clips}

# Save the combined JSON to a new file (optional)
combined_file_path = os.path.join(json_directory, 'combined_biovid_info.json')
with open(combined_file_path, 'w') as outfile:
    json.dump(combined_json, outfile, indent=4)

# Print the total number of clips in the combined JSON
print(f"Total records in the combined JSON file: {total_clips}")
