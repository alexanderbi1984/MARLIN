import os
import json


# Function to parse video file name and extract relevant data
# def parse_video_file(file_name):
#     # Remove the extension
#     file_name_no_ext = os.path.splitext(file_name)[0]
#
#     # Split the name into parts based on your naming convention
#     parts = file_name_no_ext.split('_')
#     subject_id = parts[0]
#     sex = 'female' if parts[1] == 'w' else 'male'
#
#     # Further split the class label and video number
#     age, class_label, video_serial_number = parts[2].split('-')
#
#     # Determine binary class based on multiclass label
#     if class_label in ['BL1', 'PA1', 'PA2']:
#         binary = 0
#     elif class_label in ['PA3', 'PA4']:
#         binary = 1
#     else:
#         raise ValueError(f"Unexpected class label: {class_label}")
#
#     # Return parsed data in the required format
#     return {
#         file_name: {
#             "attributes": {
#                 "binary": binary,
#                 "multiclass": class_label,
#                 "subject_id": subject_id,
#                 "sex": sex,
#                 "age": age,
#             }
#         }
#     }
import os


# Function to parse video file name and extract relevant data
def parse_video_file(file_name):
    # Remove the extension
    file_name_no_ext = os.path.splitext(file_name)[0]
    # if the file_name_no_ext starts with a number, source = 'BioVid'
    if file_name_no_ext[0].isdigit():
        source = 'BioVid'
    else:
        source = 'BioVidGan'

    # Split the name into parts based on your naming convention
    parts = file_name_no_ext.split('_')
    subject_id = parts[0]
    sex = 'female' if parts[1] == 'w' else 'male'

    # Further split the class label and video number
    age, class_label, video_serial_number = parts[2].split('-')

    # Define the mapping for multiclass labels
    class_mapping = {
        'BL1': 0,
        'PA1': 1,
        'PA2': 2,
        'PA3': 3,
        'PA4': 4
    }

    # Determine the integer class based on the class label
    if class_label not in class_mapping:
        raise ValueError(f"Unexpected class label: {class_label}")

    multiclass = class_mapping[class_label]
    if multiclass < 2:
        multiclass_3 = 0
    elif 2 < multiclass < 4:  # Use 'and' to check the range
        multiclass_3 = 1
    elif multiclass == 4:
        multiclass_3 = 2
    else:
        multiclass_3 = None  # Optionally handle the case for multiclass >= 5 or other values

    multiclass_dict = {
        "5": multiclass,
        "3": multiclass_3
    }

    # You can also determine the binary class if needed
    binary = 0 if class_label in ['BL1', 'PA1', 'PA2'] else 1

    # Return parsed data in the required format
    return {
        file_name: {
            "attributes": {
                "binary": binary,
                "multiclass": multiclass_dict,  # Now storing the numeric multiclass
                "subject_id": subject_id,
                "sex": sex,
                "age": age,
                "ground_truth": multiclass,
                "source": source
            }
        }
    }


# Function to process all video files in a directory
def process_videos(directory):
    clips = {}

    # Iterate through all files in the given directory
    for file_name in os.listdir(directory):
        # Skip files that are not .mp4
        if not file_name.endswith('.mp4'):
            continue

        # Parse the file name and add to the clips dictionary
        try:
            parsed_data = parse_video_file(file_name)
            clips.update(parsed_data)
        except ValueError as e:
            print(f"Error processing file {file_name}: {e}")

    # Print the total number of clips processed
    total_clips = len(clips)
    print(f"Total number of clips processed: {total_clips}")

    # Return the final JSON structure
    return {"clips": clips}


# Main function to run the script
def main():
    # Specify directory containing the video files
    video_directory = r"C:\pain\BioVid_224_video"

    # Process the videos and construct the JSON
    json_data = process_videos(video_directory)

    # Write the JSON data to a file
    output_file = r"C:\pain\BioVid_224_video\biovid_new.json"
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"JSON file created: {output_file}")


# Run the script
if __name__ == "__main__":
    main()

# import os
# import json
#
#
# # Function to parse video file name and extract relevant data
# def parse_video_file(file_name):
#     # Remove the extension
#     file_name_no_ext = os.path.splitext(file_name)[0]
#
#     # Split the name into parts based on your naming convention
#     parts = file_name_no_ext.split('_')
#     subject_id = parts[0]
#     sex = 'female' if parts[1] == 'w' else 'male'
#
#     # Further split the class label and video number
#     age, class_label, video_serial_number = parts[2].split('-')
#
#     # Determine binary class based on multiclass label
#     if class_label in ['BL1', 'PA1', 'PA2']:
#         binary = 0
#     elif class_label in ['PA3', 'PA4']:
#         binary = 1
#     else:
#         raise ValueError(f"Unexpected class label: {class_label}")
#
#     # Return parsed data in the required format
#     return {
#         file_name_no_ext: {
#             "attributes": {
#                 "binary": binary,
#                 "multiclass": class_label,
#                 "subject_id": subject_id,
#                 "sex": sex,
#                 "age": age,
#             }
#         }
#     }
#
#
# # Function to process all video files in a directory
# def process_videos(directory):
#     clips = {}
#
#     # Iterate through all files in the given directory
#     for file_name in os.listdir(directory):
#         # Skip files that are not .mp4
#         if not file_name.endswith('.mp4'):
#             continue
#
#         # Parse the file name and add to the clips dictionary
#         try:
#             parsed_data = parse_video_file(file_name)
#             clips.update(parsed_data)
#         except ValueError as e:
#             print(f"Error processing file {file_name}: {e}")
#
#     # Return the final JSON structure
#     return {"clips": clips}
#
#
# # Main function to run the script
# def main():
#     # Specify directory containing the video files
#     video_directory = r"C:\pain\BioVid_224_video"
#
#     # Process the videos and construct the JSON
#     json_data = process_videos(video_directory)
#
#     # Write the JSON data to a file
#     output_file = r"C:\pain\BioVid_224_video\biovid_info.json"
#     with open(output_file, "w") as f:
#         json.dump(json_data, f, indent=4)
#
#     print(f"JSON file created: {output_file}")
#
#
# # Run the script
# if __name__ == "__main__":
#     main()
