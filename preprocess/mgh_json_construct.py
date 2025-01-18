import pandas as pd
import json
import os

def set_attributes(row):
    # Set binary value based on vas
    binary = 1 if row['vas'] >= 5 else 0

    # Set multiclass based on vas ranges
    if 0 <= row['vas'] < 2:
        multiclass = 0
    elif 2 <= row['vas'] < 4:
        multiclass = 1
    elif 4 <= row['vas'] < 6:
        multiclass = 2
    elif 6 <= row['vas'] < 8:
        multiclass = 3
    elif 8 <= row['vas'] <= 10:
        multiclass = 4
    else:
        multiclass = -1  # Handle out-of-bound values if necessary

    return binary, multiclass

def csv_to_custom_json_and_txt(csv_file_path: str, json_file_path: str, txt_file_path: str):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)  # Using tab as the delimiter based on the input
    # print row names
    print(df.columns)

    # Create the structured data for the JSON
    clips = {}
    file_names = []  # List to store file names for the TXT file

    for index, row in df.iterrows():

        # Construct the video file name and replace the CSV extension with MP4
        file_name = row['file_name'].split('/')[-1].replace('.csv', '.mp4')  # Replace CSV extension with MP4

        # Set binary and multiclass attributes
        binary, multiclass = set_attributes(row)

        # Create the attributes dictionary, including temp, vas, and temp_cut
        attributes = {
            "binary": binary,  # Set based on vas
            "multiclass": multiclass,  # Set based on vas ranges
            "subject_id": row['id'],  # Assuming 'id' is the subject ID
            "sex": row['sex'],
            "age": row['age'],
            "temp": row['temp'],  # Adding temp
            "vas": row['vas'],  # Adding vas
            "temp_cut": row['temp_cut']  # Adding temp_cut
        }

        # Add to the clips dictionary
        clips[file_name] = {
            "attributes": attributes
        }

        # Add file_name to the list
        file_names.append(file_name)

    # Wrap the clips in an outer dictionary
    output_data = {
        "clips": clips
    }

    # Write the data to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Successfully converted {csv_file_path} to {json_file_path}")

    # Write the file names to a TXT file
    with open(txt_file_path, 'w') as txt_file:
        for name in file_names:
            txt_file.write(f"{name}\n")

    print(f"Successfully created {txt_file_path} with file names.")

if __name__ == "__main__":
    # Define paths for input CSV, output JSON, and output TXT
    input_csv_path = r"C:\pain\mgh\cropped\pred_data1.csv"  # Change this to your actual CSV file path
    output_json_path = r"C:\pain\mgh\cropped\biovid_info.json"  # Change this to your desired output JSON file path
    output_txt_path = r"C:\pain\mgh\cropped\test.txt"  # Change this to your desired output TXT file path

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    # Convert CSV to JSON and generate TXT file
    csv_to_custom_json_and_txt(input_csv_path, output_json_path, output_txt_path)
