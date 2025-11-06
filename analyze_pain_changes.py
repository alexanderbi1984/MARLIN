import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the meta file
meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
df = pd.read_excel(meta_path)

# Convert pain_level to numeric, setting non-numeric values to NaN
df['pain_level'] = pd.to_numeric(df['pain_level'], errors='coerce')

# Create pairs of pre-post visits
complete_pairs = []

for subj in df['subject_id'].unique():
    subject_data = df[df['subject_id'] == subj]
    
    # Analyze 1st visit
    first_pre = subject_data[subject_data['visit_type'] == '1st-pre']
    first_post = subject_data[subject_data['visit_type'] == '1st-post']
    
    if len(first_pre) > 0 and len(first_post) > 0:
        pre_pain = first_pre['pain_level'].iloc[0] if not first_pre['pain_level'].isna().all() else None
        post_pain = first_post['pain_level'].iloc[0] if not first_post['pain_level'].isna().all() else None
        
        if pre_pain is not None and post_pain is not None:
            if pd.notna(pre_pain) and pd.notna(post_pain):
                change = pre_pain - post_pain
                complete_pairs.append({
                    'subject': subj,
                    'visit_number': '1',
                    'pre_pain': pre_pain,
                    'post_pain': post_pain,
                    'change': change
                })
    
    # Analyze 2nd visit
    second_pre = subject_data[subject_data['visit_type'] == '2nd-pre']
    second_post = subject_data[subject_data['visit_type'] == '2nd-post']
    
    if len(second_pre) > 0 and len(second_post) > 0:
        pre_pain = second_pre['pain_level'].iloc[0] if not second_pre['pain_level'].isna().all() else None
        post_pain = second_post['pain_level'].iloc[0] if not second_post['pain_level'].isna().all() else None
        
        if pre_pain is not None and post_pain is not None:
            if pd.notna(pre_pain) and pd.notna(post_pain):
                change = pre_pain - post_pain
                complete_pairs.append({
                    'subject': subj,
                    'visit_number': '2',
                    'pre_pain': pre_pain,
                    'post_pain': post_pain,
                    'change': change
                })

# Convert pairs to DataFrame
pairs_df = pd.DataFrame(complete_pairs)

# Print detailed statistics
print("\n=== Pain Change Distribution Analysis ===")
print(f"Total number of complete pairs: {len(complete_pairs)}")

print("\nDetailed pain change statistics:")
print(pairs_df['change'].describe())

print("\nDistribution of pain changes:")
print(pairs_df['change'].value_counts().sort_index())

# Create visualizations
plt.figure(figsize=(15, 10))

# Pain change distribution
plt.subplot(2, 2, 1)
sns.histplot(data=pairs_df, x='change', bins=20)
plt.title('Distribution of Pain Changes')
plt.xlabel('Pain Change (Pre - Post)')
plt.ylabel('Count')
plt.axvline(x=4, color='r', linestyle='--', label='Threshold (4)')
plt.legend()

# Pain change by visit number
plt.subplot(2, 2, 2)
sns.boxplot(data=pairs_df, x='visit_number', y='change')
plt.title('Pain Changes by Visit Number')
plt.xlabel('Visit Number')
plt.ylabel('Pain Change')
plt.axhline(y=4, color='r', linestyle='--', label='Threshold (4)')
plt.legend()

# Initial pain level vs change
plt.subplot(2, 2, 3)
plt.scatter(pairs_df['pre_pain'], pairs_df['change'])
plt.title('Initial Pain Level vs Change')
plt.xlabel('Initial Pain Level')
plt.ylabel('Pain Change')
plt.axhline(y=4, color='r', linestyle='--', label='Threshold (4)')
plt.legend()

# Post pain level vs change
plt.subplot(2, 2, 4)
plt.scatter(pairs_df['post_pain'], pairs_df['change'])
plt.title('Final Pain Level vs Change')
plt.xlabel('Final Pain Level')
plt.ylabel('Pain Change')
plt.axhline(y=4, color='r', linestyle='--', label='Threshold (4)')
plt.legend()

plt.tight_layout()
plt.savefig('pain_changes_distribution.png')
plt.close()

print("\nVisualizations have been saved to 'pain_changes_distribution.png'")

# Print detailed breakdown of changes
print("\n=== Detailed Breakdown of Pain Changes ===")
print("\nChanges by visit number:")
print(pairs_df.groupby('visit_number')['change'].describe())

print("\nNumber of cases with different levels of improvement:")
print("No improvement (0):", len(pairs_df[pairs_df['change'] == 0]))
print("Small improvement (1-3):", len(pairs_df[(pairs_df['change'] > 0) & (pairs_df['change'] < 4)]))
print("Significant improvement (≥4):", len(pairs_df[pairs_df['change'] >= 4]))
print("Complete improvement (10):", len(pairs_df[pairs_df['change'] == 10])) 