import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('outcome_analysis_results/marlin_video_outcome_analysis.csv')

# Sort by absolute effect size
df['abs_effect_size'] = df['effect_size'].abs()
df_sorted = df.sort_values('abs_effect_size', ascending=False)

# Print top 20 features
print("\nTop 20 features by absolute effect size:")
print("========================================")
for _, row in df_sorted.head(20).iterrows():
    print(f"Feature {row['feature_idx']}: |d| = {row['abs_effect_size']:.3f} (p = {row['p_value']:.6f})") 