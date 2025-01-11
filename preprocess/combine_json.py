import json

# Load the first JSON file
with open(r"C:\pain\BioVid_224_video\fake_vid_info.json", 'r') as f1:
    json_data_1 = json.load(f1)

# Load the second JSON file
with open(r"C:\pain\BioVid_224_video\biovid_info.json", 'r') as f2:
    json_data_2 = json.load(f2)

# Combine the "clips" from both files
combined_clips = {**json_data_1["clips"], **json_data_2["clips"]}

# Create the combined JSON object
combined_json = {"clips": combined_clips}

# Save the combined JSON to a new file (optional)
with open('combined_file.json', 'w') as outfile:
    json.dump(combined_json, outfile, indent=4)

# Print the total number of clips
total_clips = len(combined_clips)
print(f"Total clips: {total_clips}")