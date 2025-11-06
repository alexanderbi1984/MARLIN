import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the meta file
meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
df = pd.read_excel(meta_path)

# Convert pain_level to numeric, setting non-numeric values to NaN
df['pain_level'] = pd.to_numeric(df['pain_level'], errors='coerce')

# Create a copy of the dataframe for analysis
analysis_df = df.copy()

# Extract visit number and type (pre/post) from visit_type
analysis_df['visit_number'] = analysis_df['visit_type'].str.extract('(\d+)')
analysis_df['visit_type'] = analysis_df['visit_type'].str.extract('(pre|post)')

# Create pairs of pre-post visits
pairs = []
for subject in analysis_df['subject_id'].unique():
    subject_data = analysis_df[analysis_df['subject_id'] == subject]
    
    # Analyze 1st visit
    first_pre = subject_data[(subject_data['visit_number'] == '1') & (subject_data['visit_type'] == 'pre')]
    first_post = subject_data[(subject_data['visit_number'] == '1') & (subject_data['visit_type'] == 'post')]
    
    if not first_pre.empty and not first_post.empty:
        pairs.append({
            'subject_id': subject,
            'visit_number': '1',
            'pre_pain': first_pre['pain_level'].iloc[0],
            'post_pain': first_post['pain_level'].iloc[0],
            'pain_reduction': first_pre['pain_level'].iloc[0] - first_post['pain_level'].iloc[0]
        })
    
    # Analyze 2nd visit
    second_pre = subject_data[(subject_data['visit_number'] == '2') & (subject_data['visit_type'] == 'pre')]
    second_post = subject_data[(subject_data['visit_number'] == '2') & (subject_data['visit_type'] == 'post')]
    
    if not second_pre.empty and not second_post.empty:
        pairs.append({
            'subject_id': subject,
            'visit_number': '2',
            'pre_pain': second_pre['pain_level'].iloc[0],
            'post_pain': second_post['pain_level'].iloc[0],
            'pain_reduction': second_pre['pain_level'].iloc[0] - second_post['pain_level'].iloc[0]
        })

# Convert pairs to DataFrame
pairs_df = pd.DataFrame(pairs)

# Print statistics
print("\n=== Pre-Post Pain Analysis ===")
print(f"Total number of valid pre-post pairs: {len(pairs_df)}")
print("\nPain reduction statistics:")
print(pairs_df['pain_reduction'].describe())

print("\nDistribution of pain reduction:")
print(pairs_df['pain_reduction'].value_counts().sort_index())

# Calculate success rate (reduction >= 4)
success_rate = (pairs_df['pain_reduction'] >= 4).mean() * 100
print(f"\nSuccess rate (reduction >= 4): {success_rate:.2f}%")

# Create visualizations
plt.figure(figsize=(15, 10))

# Pain reduction distribution
plt.subplot(2, 2, 1)
sns.histplot(data=pairs_df, x='pain_reduction', bins=20)
plt.title('Distribution of Pain Reduction')
plt.xlabel('Pain Reduction')
plt.ylabel('Count')

# Pre vs Post pain levels
plt.subplot(2, 2, 2)
plt.scatter(pairs_df['pre_pain'], pairs_df['post_pain'])
plt.plot([0, 10], [0, 10], 'r--')  # Diagonal line
plt.title('Pre vs Post Pain Levels')
plt.xlabel('Pre Pain Level')
plt.ylabel('Post Pain Level')

# Pain reduction by visit number
plt.subplot(2, 2, 3)
sns.boxplot(data=pairs_df, x='visit_number', y='pain_reduction')
plt.title('Pain Reduction by Visit Number')
plt.xlabel('Visit Number')
plt.ylabel('Pain Reduction')

# Initial pain level vs reduction
plt.subplot(2, 2, 4)
plt.scatter(pairs_df['pre_pain'], pairs_df['pain_reduction'])
plt.title('Initial Pain Level vs Reduction')
plt.xlabel('Initial Pain Level')
plt.ylabel('Pain Reduction')

plt.tight_layout()
plt.savefig('prepost_analysis.png')
plt.close()

print("\nVisualizations have been saved to 'prepost_analysis.png'") 