import csv
import json


def construct_json_from_csv(csv_file, output_json_file, multiclass_source):
    clips = {}

    # Read from the CSV file
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = row['file_name'].replace('.txt', '.mp4')  # Replace .txt with .mp4
            subject_id = row['subject_id']

            # Determine the multiclass value based on the specified source (VAS or OPR)
            if multiclass_source == 'VAS':
                multiclass_value = row['VAS']
            elif multiclass_source == 'OPR':
                multiclass_value = row['OPR']
            else:
                multiclass_value = None

            # Set default values for sex and age
            sex = ""
            age = ""

            # Create the multiclass dictionary with a fixed number of classes
            multiclass_dict = {}
            if multiclass_value is not None:
                multiclass_value = float(multiclass_value)
                multiclass_dict["6"] = multiclass_value # Use the value directly from the column
                multiclass_value_5 = multiclass_value-1 if multiclass_value > 1 else 0
                multiclass_dict["5"] = multiclass_value_5
                binary_value = 0 if multiclass_value < 3 else 1

            # Create the attributes dictionary
            attributes = {
                "binary": binary_value,
                "multiclass": multiclass_dict,
                "subject_id": subject_id,
                "sex": sex,
                "age": age
            }

            # Create the clip entry
            clips[filename] = {
                "attributes": attributes
            }

    # Wrap clips in the final structure
    final_output = {
        "clips": clips
    }

    # Write to the JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(final_output, json_file, indent=4)


# Call the function with the appropriate paths
construct_json_from_csv(r"C:\pain\ShoulderPain_video\label.csv", r'C:\pain\ShoulderPain_video\cropped\biovid_info.json', 'OPR')  # Change 'VAS' to 'OPR' as needed
