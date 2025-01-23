import os
import json

# Define the path to your JSON file
json_file_path = r"C:\pain\json4marlin\biovid_info.json"  # Change to your JSON file path

# Load the JSON data
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Iterate through the clips to update the multiclass dictionary based on the source
for filename, attributes in data["clips"].items():
    # Get the source from attributes
    source = attributes["attributes"].get("source", "")
    # print(f"Processing file '{filename}' with source '{source}'")

    # Access the multiclass dictionary
    multiclass = attributes["attributes"].get("multiclass", {})
    if not isinstance(multiclass, dict):
        # print(f"Warning: multiclass for file '{filename}' is not a dictionary. Initializing as an empty dict.")
        multiclass = {}

    # Example (replace this with your logic):
    if source == "BioVid" or source == "BioVidGan":
        ground_truth = attributes["attributes"].get("ground_truth", "")
        multiclass["5"] = ground_truth  # Update logic example
    if source == "mgh":
        # print(f"Processing file '{filename}' with source '{source}'")
        vas = attributes["attributes"].get("vas", "")
        class_label_5 = int(vas/2)-1
        if class_label_5 < 0:
            class_label_5 = 0
        multiclass["5"] = class_label_5
        # multiclass = {
        #     "5": class_label_5
        # }
        # print(f"after updating: {}")
    if source == "bp4d" or source == "bp4d+":
        vas = attributes["attributes"].get("vas", "")
        class_label_5 = vas-1
        if class_label_5 < 0:
            class_label_5 = 0
        multiclass = {
            "5": class_label_5
        }
    if source == "ShoulderPain":
        vas = attributes["attributes"].get("vas", "")
        class_label_5 = int(float(vas))-1
        if class_label_5 < 0:
            class_label_5 = 0
        multiclass["5"] = class_label_5
    # Save the updated multiclass back into the attributes
    multiclass["3"] = multiclass["5"]  # Update the 3-class multiclass
    if multiclass["3"] == 2:
        multiclass["3"] = 1
    if multiclass["3"] == 4:
        multiclass["3"] = 2
    attributes["attributes"]["multiclass"] = multiclass


# Save the updated JSON data back to the file
with open(json_file_path, 'w') as f:
    json.dump(data, f, indent=4)

print("JSON file has been updated.")
