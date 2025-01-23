import csv
import json
import pandas as pd

def read_excel_file(excel_file):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file)

    # Convert the DataFrame to a list of dictionaries
    data = df.to_dict(orient='records')

    return data

def construct_json_from_csv(csv_file, output_json_file, multiclass_source):
    clips = {}

    # Read from the CSV file
    reader = read_excel_file(csv_file)

    for row in reader:
        # filename = row['Subject']+'_11.avi'  # Replace .txt with .mp4
        filename = row['Subject'] + '_11.avi'
        subject_id = row['Subject']+multiclass_source

        # Determine the multiclass value based on the specified source (VAS or OPR)
        if multiclass_source == 'VAS':
            multiclass_value = row['VAS']
        elif multiclass_source == 'OPR':
            multiclass_value = row['OPR']
        else:
            multiclass_value = row['Pained']

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
            "age": age,
            "source": 'bp4d+',
            "vas": row['Pained'],
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
construct_json_from_csv(r"C:\pain\Pain\BP4D+\bp4d+_pain.xlsx", r"C:\pain\Pain\BP4D+\bp4d+.json", 'bp4d')  # Change 'VAS' to 'OPR' as needed
