import pandas as pd
import argparse

def check_dist(meta_path, label_col):
    if meta_path.endswith('.csv'):
        df = pd.read_csv(meta_path)
    else:
        df = pd.read_excel(meta_path)
    
    print(f"Loaded {len(df)} samples.")
    if label_col not in df.columns:
        print(f"Error: {label_col} not in columns.")
        return

    # Convert to int
    try:
        y = df[label_col].astype(float).astype(int)
        # Cap at 4
        y = y.apply(lambda x: min(x, 4))
        counts = y.value_counts().sort_index()
        print("\nLabel Distribution:")
        print(counts)
        print("\nPercentage:")
        print(counts / len(df))
    except:
        print("Could not convert labels to int stats.")

if __name__ == "__main__":
    check_dist("/data/Nbi/biovid/biovid_pain_labels.csv", "pain_level")
