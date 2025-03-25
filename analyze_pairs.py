import pandas as pd
import numpy as np
from pathlib import Path

# Read the data
meta_path = r'C:\pain\syracus\openface_clips\clips\multimodal_marlin_base\meta.xlsx'
df = pd.read_excel(meta_path)

# Create a copy of the dataframe for modifications
df_with_outcomes = df.copy()

# Add outcome column initialized as None
df_with_outcomes['outcome'] = None

# Filter for 1st visits only for analysis
df_first = df[df['visit_type'].isin(['1st-pre', '1st-post'])]

print(f"Total 1st visit samples: {len(df_first)}")
print(f"Unique subjects in 1st visits: {len(df_first['subject_id'].unique())}")

# Print unique pain level values to understand the data
print("\nUnique pain level values:")
print(df_first['pain_level'].unique())

# Function to check if a value can be converted to float
def is_numeric(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

# Analyze complete pairs and store outcomes
complete_pairs = []
non_numeric_pairs = []

for subj in df_first['subject_id'].unique():
    pre = df_first[(df_first['subject_id'] == subj) & (df_first['visit_type'] == '1st-pre')]
    post = df_first[(df_first['subject_id'] == subj) & (df_first['visit_type'] == '1st-post')]
    
    if len(pre) > 0 and len(post) > 0:
        pre_pain = pre['pain_level'].iloc[0] if not pre['pain_level'].isna().all() else None
        post_pain = post['pain_level'].iloc[0] if not post['pain_level'].isna().all() else None
        
        outcome = None
        if pre_pain is not None and post_pain is not None:
            if is_numeric(pre_pain) and is_numeric(post_pain):
                pre_val = float(pre_pain)
                post_val = float(post_pain)
                change = post_val - pre_val
                pct_change = (change / pre_val * 100) if pre_val != 0 else float('inf')
                
                outcome = 'positive' if change <= -4 else 'negative'
                complete_pairs.append({
                    'subject': subj,
                    'pre_pain': pre_val,
                    'post_pain': post_val,
                    'change': change,
                    'pct_change': pct_change,
                    'outcome': outcome
                })
            else:
                outcome = 'negative'  # Conservative approach
                non_numeric_pairs.append({
                    'subject': subj,
                    'pre_pain': pre_pain,
                    'post_pain': post_pain,
                    'outcome': outcome
                })
        
        # Update outcome in the full dataframe for both pre and post visits
        if outcome is not None:
            mask = (df_with_outcomes['subject_id'] == subj) & (df_with_outcomes['visit_type'].isin(['1st-pre', '1st-post']))
            df_with_outcomes.loc[mask, 'outcome'] = outcome

# Save the updated metadata
output_path = Path(meta_path).parent / 'meta_with_outcomes.xlsx'
df_with_outcomes.to_excel(output_path, index=False)
print(f"\nSaved metadata with outcomes to: {output_path}")

# Continue with the existing analysis...
print(f"\nNumber of complete pairs with numeric pain data: {len(complete_pairs)}")

# Sort pairs by change amount
sorted_pairs = sorted(complete_pairs, key=lambda x: x['change'])

print("\n=== Analysis with -4 Point Threshold ===")
print("\nDetailed outcomes for numeric cases:")
print("\nPositive outcomes (improvement ≥ 4 points):")
positive_cases = [p for p in sorted_pairs if p['outcome'] == 'positive']
for p in positive_cases:
    print(f"Subject {p['subject']:2d}: Pre={p['pre_pain']:.1f}, Post={p['post_pain']:.1f}, "
          f"Change={p['change']:.1f} points ({p['pct_change']:.1f}%)")

print("\nNegative outcomes (improvement < 4 points):")
negative_cases = [p for p in sorted_pairs if p['outcome'] == 'negative']
for p in negative_cases:
    print(f"Subject {p['subject']:2d}: Pre={p['pre_pain']:.1f}, Post={p['post_pain']:.1f}, "
          f"Change={p['change']:.1f} points ({p['pct_change']:.1f}%)")

print("\nNon-numeric cases (counted as negative):")
for p in non_numeric_pairs:
    print(f"Subject {p['subject']}: Pre={p['pre_pain']}, Post={p['post_pain']}")

# Summary statistics
positive_count = len(positive_cases)
negative_count = len(negative_cases) + len(non_numeric_pairs)

print("\n=== Summary Statistics ===")
print(f"Total subjects with complete data: {len(complete_pairs) + len(non_numeric_pairs)}")
print(f"Subjects with numeric data: {len(complete_pairs)}")
print(f"Subjects with non-numeric data: {len(non_numeric_pairs)}")
print(f"\nPositive outcomes: {positive_count}")
print(f"Negative outcomes: {negative_count}")
print(f"Ratio (positive:negative): 1:{negative_count/positive_count:.2f}")

if positive_cases:
    pos_changes = [p['change'] for p in positive_cases]
    pos_pct_changes = [p['pct_change'] for p in positive_cases]
    print("\nPositive outcomes statistics:")
    print(f"Mean improvement: {abs(np.mean(pos_changes)):.1f} points")
    print(f"Mean percentage improvement: {abs(np.mean(pos_pct_changes)):.1f}%")
    print(f"Range of improvement: {abs(min(pos_changes)):.1f} to {abs(max(pos_changes)):.1f} points")

if negative_cases:
    neg_changes = [p['change'] for p in negative_cases]
    print("\nNegative outcomes statistics (numeric cases only):")
    print(f"Mean change: {abs(np.mean(neg_changes)):.1f} points")
    print(f"Range of change: {min(neg_changes):.1f} to {max(neg_changes):.1f} points")

# Print outcome distribution in full dataset
print("\n=== Full Dataset Outcome Distribution ===")
outcome_counts = df_with_outcomes['outcome'].value_counts()
print(outcome_counts)
print("\nPercentage of samples with outcomes assigned:")
print(f"{(df_with_outcomes['outcome'].notna().sum() / len(df_with_outcomes) * 100):.1f}%") 