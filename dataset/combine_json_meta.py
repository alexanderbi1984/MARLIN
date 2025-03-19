import json
import os

# Define input and output paths
biovid_path = r"C:\Users\Nan Bi\PycharmProjects\MARLIN\dataset_meta\biovid_biovidGan.json"  # Contains both BioVid and BioVidGan
mgh_path = r"C:\Users\Nan Bi\PycharmProjects\MARLIN\dataset_meta\mgh.json"
shoulder_pain_path = r"C:\Users\Nan Bi\PycharmProjects\MARLIN\dataset_meta\ShoulderPain.json"
bp4d_path = r"C:\Users\Nan Bi\PycharmProjects\MARLIN\dataset_meta\bp4d.json"
bp4d_plus_path = r"C:\Users\Nan Bi\PycharmProjects\MARLIN\dataset_meta\bp4d+.json"
output_path = r"C:\Users\Nan Bi\PycharmProjects\MARLIN\dataset_meta\all_dataset.json"

# Initialize the combined structure
combined_data = {
    "clips": {},
    "metadata": {
        "sources": {}
    }
}

# Initialize counters for each source
source_counters = {
    "BioVid": 0,
    "BioVidGan": 0,
    "MGH": 0,
    "ShoulderPain": 0,
    "BP4D": 0,
    "BP4D+": 0
}

# Process BioVid data (which includes both BioVid and BioVidGan)
with open(biovid_path, 'r') as f:
    biovid_data = json.load(f)

for clip_name, clip_info in biovid_data["clips"].items():
    # Copy the clip info
    combined_data["clips"][clip_name] = clip_info


    if "source" in clip_info and clip_info["source"] == "BioVidGan":
        combined_data["clips"][clip_name]["source"] = "BioVidGan"
        source_counters["BioVidGan"] += 1
    else:
        combined_data["clips"][clip_name]["source"] = "BioVid"
        source_counters["BioVid"] += 1

    # Calculate pain_score: Scale multiclass "5" (0-4) to range 0-10
    multiclass_value = int(clip_info["attributes"]["multiclass"]["5"])
    pain_score = multiclass_value * 2.5  # Multiply by 2.5 to map 0-4 to 0-10

    # Add pain_score to attributes
    combined_data["clips"][clip_name]["attributes"]["pain_score"] = float(pain_score)

# Process MGH data
with open(mgh_path, 'r') as f:
    mgh_data = json.load(f)

for clip_name, clip_info in mgh_data["clips"].items():
    # Copy the clip info
    combined_data["clips"][clip_name] = clip_info

    # Add source information
    combined_data["clips"][clip_name]["source"] = "MGH"
    source_counters["MGH"] += 1

    # Get VAS directly
    pain_score = float(clip_info["attributes"]["vas"])

    # Add pain_score to attributes
    combined_data["clips"][clip_name]["attributes"]["pain_score"] = float(pain_score)

# Process ShoulderPain data
with open(shoulder_pain_path, 'r') as f:
    shoulder_pain_data = json.load(f)

for clip_name, clip_info in shoulder_pain_data["clips"].items():
    # Copy the clip info
    combined_data["clips"][clip_name] = clip_info

    # Add source information
    combined_data["clips"][clip_name]["source"] = "ShoulderPain"
    source_counters["ShoulderPain"] += 1

    # Get VAS directly
    pain_score = float(clip_info["attributes"]["vas"])

    # Add pain_score to attributes
    combined_data["clips"][clip_name]["attributes"]["pain_score"] = float(pain_score)

# Process BP4D data
with open(bp4d_path, 'r') as f:
    bp4d_data = json.load(f)

for clip_name, clip_info in bp4d_data["clips"].items():
    # Copy the clip info
    combined_data["clips"][clip_name] = clip_info

    # Add source information
    combined_data["clips"][clip_name]["source"] = "BP4D"
    source_counters["BP4D"] += 1

    # Calculate pain_score: Multiply VAS by 2
    vas_value = float(clip_info["attributes"]["vas"])
    pain_score = vas_value * 2

    # Add pain_score to attributes
    combined_data["clips"][clip_name]["attributes"]["pain_score"] = float(pain_score)

# Process BP4D+ data
with open(bp4d_plus_path, 'r') as f:
    bp4d_plus_data = json.load(f)

for clip_name, clip_info in bp4d_plus_data["clips"].items():
    # Copy the clip info
    combined_data["clips"][clip_name] = clip_info

    # Add source information
    combined_data["clips"][clip_name]["source"] = "BP4D+"
    source_counters["BP4D+"] += 1

    # Calculate pain_score: Multiply VAS by 2
    vas_value = float(clip_info["attributes"]["vas"])
    pain_score = vas_value * 2

    # Add pain_score to attributes
    combined_data["clips"][clip_name]["attributes"]["pain_score"] = float(pain_score)

# Add source statistics to metadata
combined_data["metadata"]["sources"] = source_counters
combined_data["metadata"]["total_clips"] = sum(source_counters.values())

# Print source statistics
print("Source statistics:")
for source, count in source_counters.items():
    print(f"  {source}: {count} clips")
print(f"Total: {sum(source_counters.values())} clips")

# Save the combined data
with open(output_path, 'w') as f:
    json.dump(combined_data, f, indent=2)

print(f"Combined data saved to {output_path}")