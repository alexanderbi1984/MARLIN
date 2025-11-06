import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8')

# Read the meta file
meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
df = pd.read_excel(meta_path)

# Print data info to see column types
print("\n=== Data Info ===")
print(df.info())

# Check unique values in pain_level column
print("\n=== Unique values in pain_level ===")
print(df['pain_level'].unique())

# Convert pain_level to numeric, setting non-numeric values to NaN
df['pain_level'] = pd.to_numeric(df['pain_level'], errors='coerce')

# Basic Statistics
print("\n=== Basic Statistics ===")
print(f"Total number of subjects: {df['subject_id'].nunique()}")
print(f"Total number of videos: {len(df)}")
print("\nPain level statistics:")
print(df['pain_level'].describe())

# Visit type distribution
print("\n=== Visit Type Distribution ===")
visit_type_counts = df['visit_type'].value_counts()
print(visit_type_counts)

# Outcome distribution
print("\n=== Outcome Distribution ===")
outcome_counts = df['outcome'].value_counts()
print(outcome_counts)

# Missing data analysis
print("\n=== Missing Data Analysis ===")
missing_data = df.isnull().sum()
print("Missing values in each column:")
print(missing_data[missing_data > 0])

# Create visualizations
plt.figure(figsize=(15, 10))

# Pain level distribution (only for numeric values)
plt.subplot(2, 2, 1)
# Remove NaN values before plotting
sns.histplot(data=df[df['pain_level'].notna()], x='pain_level', bins=11)
plt.title('Pain Level Distribution')
plt.xlabel('Pain Level')
plt.ylabel('Count')

# Visit type distribution
plt.subplot(2, 2, 2)
visit_type_counts.plot(kind='bar')
plt.title('Visit Type Distribution')
plt.xlabel('Visit Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Outcome distribution
plt.subplot(2, 2, 3)
outcome_counts.plot(kind='bar')
plt.title('Outcome Distribution')
plt.xlabel('Outcome')
plt.ylabel('Count')

# Pain level by visit type (only for numeric values)
plt.subplot(2, 2, 4)
sns.boxplot(data=df[df['pain_level'].notna()], x='visit_type', y='pain_level')
plt.title('Pain Level by Visit Type')
plt.xlabel('Visit Type')
plt.ylabel('Pain Level')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('dataset_statistics.png')
plt.close()

print("\nVisualizations have been saved to 'dataset_statistics.png'") 