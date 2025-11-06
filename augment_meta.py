import pandas as pd
import os

# Read the original meta file
input_file = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes_and_classes.xlsx'
output_file = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/aug_meta_with_outcomes_and_classes.xlsx'

# Read the Excel file
df = pd.read_excel(input_file)

# Create a list to store the duplicated rows
duplicated_rows = []

# Process each row
for _, row in df.iterrows():
    # Get the original filename
    original_filename = row['file_name']
    
    # Create two new filenames
    base_name = original_filename.rsplit('.', 1)[0]
    extension = original_filename.rsplit('.', 1)[1]
    
    filename1 = f"{base_name}_1.{extension}"
    filename2 = f"{base_name}_2.{extension}"
    
    # Create two copies of the row with modified filenames
    row1 = row.copy()
    row2 = row.copy()
    
    row1['file_name'] = filename1
    row2['file_name'] = filename2
    
    # Add the duplicated rows to our list
    duplicated_rows.extend([row1, row2])

# Create a new dataframe with all rows
new_df = pd.concat([df, pd.DataFrame(duplicated_rows)], ignore_index=True)

# Save the new dataframe to Excel
new_df.to_excel(output_file, index=False)

print(f"Original file: {input_file}")
print(f"Augmented file saved as: {output_file}")
print(f"Original number of rows: {len(df)}")
print(f"New number of rows: {len(new_df)}") 