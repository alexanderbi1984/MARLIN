import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the meta file
meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
df = pd.read_excel(meta_path)

# Convert pain_level to numeric, setting non-numeric values to NaN
df['pain_level'] = pd.to_numeric(df['pain_level'], errors='coerce')

# Function to check if a value can be converted to float
def is_numeric(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

# Create pairs of pre-post visits
complete_pairs = []
non_numeric_pairs = []

for subj in df['subject_id'].unique():
    subject_data = df[df['subject_id'] == subj]
    
    # Analyze 1st visit
    first_pre = subject_data[subject_data['visit_type'] == '1st-pre']
    first_post = subject_data[subject_data['visit_type'] == '1st-post']
    
    if len(first_pre) > 0 and len(first_post) > 0:
        pre_pain = first_pre['pain_level'].iloc[0] if not first_pre['pain_level'].isna().all() else None
        post_pain = first_post['pain_level'].iloc[0] if not first_post['pain_level'].isna().all() else None
        
        if pre_pain is not None and post_pain is not None:
            if is_numeric(pre_pain) and is_numeric(post_pain):
                pre_val = float(pre_pain)
                post_val = float(post_pain)
                change = pre_val - post_val  # Positive change means reduction in pain
                pct_change = (change / pre_val * 100) if pre_val != 0 else float('inf')
                
                outcome = 'positive' if change >= 4 else 'negative'
                complete_pairs.append({
                    'subject': subj,
                    'visit_number': '1',
                    'pre_pain': pre_val,
                    'post_pain': post_val,
                    'change': change,
                    'pct_change': pct_change,
                    'outcome': outcome
                })
            else:
                non_numeric_pairs.append({
                    'subject': subj,
                    'visit_number': '1',
                    'pre_pain': pre_pain,
                    'post_pain': post_pain,
                    'outcome': 'negative'
                })
    
    # Analyze 2nd visit
    second_pre = subject_data[subject_data['visit_type'] == '2nd-pre']
    second_post = subject_data[subject_data['visit_type'] == '2nd-post']
    
    if len(second_pre) > 0 and len(second_post) > 0:
        pre_pain = second_pre['pain_level'].iloc[0] if not second_pre['pain_level'].isna().all() else None
        post_pain = second_post['pain_level'].iloc[0] if not second_post['pain_level'].isna().all() else None
        
        if pre_pain is not None and post_pain is not None:
            if is_numeric(pre_pain) and is_numeric(post_pain):
                pre_val = float(pre_pain)
                post_val = float(post_pain)
                change = pre_val - post_val
                pct_change = (change / pre_val * 100) if pre_val != 0 else float('inf')
                
                outcome = 'positive' if change >= 4 else 'negative'
                complete_pairs.append({
                    'subject': subj,
                    'visit_number': '2',
                    'pre_pain': pre_val,
                    'post_pain': post_val,
                    'change': change,
                    'pct_change': pct_change,
                    'outcome': outcome
                })
            else:
                non_numeric_pairs.append({
                    'subject': subj,
                    'visit_number': '2',
                    'pre_pain': pre_pain,
                    'post_pain': post_pain,
                    'outcome': 'negative'
                })

# Convert pairs to DataFrame
pairs_df = pd.DataFrame(complete_pairs)

# Print detailed statistics
print("\n=== Pain Reduction Analysis ===")
print(f"Total number of valid pre-post pairs: {len(complete_pairs)}")
print(f"Number of non-numeric pairs: {len(non_numeric_pairs)}")

print("\nPain reduction statistics:")
print(pairs_df['change'].describe())

print("\nDistribution of pain reduction:")
print(pairs_df['change'].value_counts().sort_index())

# Calculate success rate (reduction >= 4)
success_rate = (pairs_df['change'] >= 4).mean() * 100
print(f"\nSuccess rate (reduction >= 4): {success_rate:.2f}%")

# Create visualizations
plt.figure(figsize=(15, 10))

# Pain reduction distribution
plt.subplot(2, 2, 1)
sns.histplot(data=pairs_df, x='change', bins=20)
plt.title('Distribution of Pain Reduction')
plt.xlabel('Pain Reduction')
plt.ylabel('Count')
plt.axvline(x=4, color='r', linestyle='--', label='Threshold (4)')
plt.legend()

# Pre vs Post pain levels
plt.subplot(2, 2, 2)
plt.scatter(pairs_df['pre_pain'], pairs_df['post_pain'])
plt.plot([0, 10], [0, 10], 'r--')  # Diagonal line
plt.title('Pre vs Post Pain Levels')
plt.xlabel('Pre Pain Level')
plt.ylabel('Post Pain Level')

# Pain reduction by visit number
plt.subplot(2, 2, 3)
sns.boxplot(data=pairs_df, x='visit_number', y='change')
plt.title('Pain Reduction by Visit Number')
plt.xlabel('Visit Number')
plt.ylabel('Pain Reduction')
plt.axhline(y=4, color='r', linestyle='--', label='Threshold (4)')
plt.legend()

# Initial pain level vs reduction
plt.subplot(2, 2, 4)
plt.scatter(pairs_df['pre_pain'], pairs_df['change'])
plt.title('Initial Pain Level vs Reduction')
plt.xlabel('Initial Pain Level')
plt.ylabel('Pain Reduction')
plt.axhline(y=4, color='r', linestyle='--', label='Threshold (4)')
plt.legend()

plt.tight_layout()
plt.savefig('pain_reduction_analysis.png')
plt.close()

print("\nVisualizations have been saved to 'pain_reduction_analysis.png'")

# Print detailed outcomes
print("\n=== Detailed Outcomes ===")
print("\nPositive outcomes (improvement ≥ 4 points):")
positive_cases = pairs_df[pairs_df['outcome'] == 'positive']
for _, p in positive_cases.iterrows():
    print(f"Subject {p['subject']:2d} (Visit {p['visit_number']}): "
          f"Pre={p['pre_pain']:.1f}, Post={p['post_pain']:.1f}, "
          f"Change={p['change']:.1f} points ({p['pct_change']:.1f}%)")

print("\nNegative outcomes (improvement < 4 points):")
negative_cases = pairs_df[pairs_df['outcome'] == 'negative']
for _, p in negative_cases.iterrows():
    print(f"Subject {p['subject']:2d} (Visit {p['visit_number']}): "
          f"Pre={p['pre_pain']:.1f}, Post={p['post_pain']:.1f}, "
          f"Change={p['change']:.1f} points ({p['pct_change']:.1f}%)")

# Summary statistics
print("\n=== Summary Statistics ===")
print(f"Total subjects with complete data: {len(complete_pairs) + len(non_numeric_pairs)}")
print(f"Subjects with numeric data: {len(complete_pairs)}")
print(f"Subjects with non-numeric data: {len(non_numeric_pairs)}")
print(f"\nPositive outcomes: {len(positive_cases)}")
print(f"Negative outcomes: {len(negative_cases) + len(non_numeric_pairs)}")
print(f"Ratio (positive:negative): 1:{(len(negative_cases) + len(non_numeric_pairs))/len(positive_cases):.2f}")

if len(positive_cases) > 0:
    print("\nPositive outcomes statistics:")
    print(f"Mean improvement: {positive_cases['change'].mean():.1f} points")
    print(f"Mean percentage improvement: {positive_cases['pct_change'].mean():.1f}%")
    print(f"Range of improvement: {positive_cases['change'].min():.1f} to {positive_cases['change'].max():.1f} points")

if len(negative_cases) > 0:
    print("\nNegative outcomes statistics (numeric cases only):")
    print(f"Mean change: {negative_cases['change'].mean():.1f} points")
    print(f"Range of change: {negative_cases['change'].min():.1f} to {negative_cases['change'].max():.1f} points") 