import os
import csv


def extract_data_to_csv(root_directory, output_csv_file):
    # Dictionary to hold the data structured by subject_id and file_name
    data_dict = {}

    # Iterate through all outer folders (AFF, VAS, OPR, SEN)
    for outer_folder in ['AFF', 'VAS', 'OPR', 'SEN']:
        outer_folder_path = os.path.join(root_directory, outer_folder)

        # Check if the outer folder exists
        if not os.path.exists(outer_folder_path):
            continue

        # Iterate through all subject folders within the outer folder
        for foldername, subfolders, filenames in os.walk(outer_folder_path):
            subject_id = os.path.basename(foldername)

            for filename in filenames:
                if filename.endswith('.txt'):
                    # Construct the full path of the txt file
                    filepath = os.path.join(foldername, filename)

                    # Read the contents of the txt file
                    with open(filepath, 'r') as file:
                        content = file.read().strip()  # Remove leading/trailing whitespace
                        print(f"filename: {filename}, content: '{content}'")  # Debugging output

                        # Convert the content to a float value
                        value = float(content)
                        print(f"after conversion: {value}")  # Debugging output

                        # Initialize the subject_id and filename in the dictionary if not present
                        if (subject_id, filename) not in data_dict:
                            # filename = filename.replace('.txt', '.mp4')
                            data_dict[(subject_id, filename)] = {
                                'AFF': None,
                                'VAS': None,
                                'OPR': None,
                                'SEN': None
                            }

                        # Store the value in the appropriate outer type key
                        if outer_folder == 'VAS':
                            data_dict[(subject_id, filename)][outer_folder] = value / 2  # Divide VAS value by 2
                        else:
                            data_dict[(subject_id, filename)][outer_folder] = value

    # Write the data to a CSV file
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(['subject_id', 'file_name', 'AFF', 'VAS', 'OPR', 'SEN'])

        # Write the data rows
        for (subject_id, filename), values in data_dict.items():
            csv_writer.writerow([
                subject_id,
                filename.replace('.txt', '.mp4'),
                values['AFF'],
                values['VAS'],
                values['OPR'],
                values['SEN']
            ])


# def extract_data_to_csv_deepseek(root_directory, output_csv):
#     # Open the CSV file for writing
#     with open(output_csv, mode='w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#
#         # Write the header row
#         csv_writer.writerow(['subject_id', 'file_name', 'value'])
#
#         # Traverse through all folders and files
#         for root, dirs, files in os.walk(root_directory):
#             for file in files:
#                 if file.endswith('.txt'):
#                     # Extract subject ID from the folder name
#                     folder_name = os.path.basename(root)
#                     subject_id = folder_name.split('-')[1]
#
#                     # Extract file name
#                     file_name = file
#
#                     # Read the value from the txt file
#                     file_path = os.path.join(root, file)
#                     with open(file_path, 'r') as txt_file:
#                         print(f"filename:{file_name}, filepath:{file_path} before removing whitespace: {txt_file.read()}")  # Debugging
#                         content = txt_file.read().strip()  # Remove leading/trailing whitespace
#                         print(f"filename: {file_name}, content: '{content}'")  # Debugging
#
#                         try:
#                             # Convert the content to a float value
#                             vas = float(content)
#                             print(f"after conversion: {vas}")  # Debugging
#                         except ValueError as e:
#                             print(f"Error converting '{content}' to float in file {file_name}: {e}")
#                             continue  # Skip this file if conversion fails
#
#                     # Write the data to the CSV file
#                     csv_writer.writerow([subject_id, file_name, vas])
#
#     print(f"Data extraction complete. Check the '{output_csv}' file.")

# Call the function with the appropriate paths
extract_data_to_csv("C:\pain\ShoulderPain\Sequence_Labels\Sequence_Labels", r'C:\pain\ShoulderPain\Sequence_Labels\Sequence_Labels\data.csv')
# extract_data_to_csv_deepseek("C:\pain\ShoulderPain\Sequence_Labels\Sequence_Labels\VAS", r'C:\pain\ShoulderPain\Sequence_Labels\Sequence_Labels\vas_deepseek.csv')
