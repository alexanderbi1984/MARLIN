import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MarlinFeatures.syracuse_dataset import SyracuseDataset
from typing import Tuple

class AUFeatureAnalyzer(SyracuseDataset):
    def __init__(self, meta_path: str, feature_dir: str, au_features_dir: str):
        """
        Initialize the AU feature analyzer.
        
        Args:
            meta_path: Path to the meta_with_outcomes.xlsx file
            feature_dir: Directory containing the MARLIN feature files
            au_features_dir: Directory containing the processed AU feature files
        """
        super().__init__(meta_path, feature_dir)
        self.au_features_dir = au_features_dir
        self.au_features = self._load_au_features()
        print(f"Loaded {len(self.au_features)} AU feature files")
        
    def _load_au_features(self) -> dict:
        """Load all processed AU feature files."""
        features = {}
        for file in os.listdir(self.au_features_dir):
            if file.startswith('processed_') and file.endswith('.csv'):
                # Extract IMG_xxxx from filename (e.g., processed_IMG_0003.csv -> IMG_0003)
                video_id = file.split('_')[1] + '_' + file.split('_')[2].split('.')[0]
                df = pd.read_csv(os.path.join(self.au_features_dir, file))
                features[video_id] = df
        return features
    
    def get_au_feature_differences(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Calculate differences in AU features between pre and post treatment."""
        feature_diffs = []
        outcomes = []
        
        print(f"Processing {len(self.pairs)} pre-post pairs")
        print(f"Available AU features: {list(self.au_features.keys())[:5]}...")  # Print first 5 available features
        
        # Check columns in first file
        first_file = list(self.au_features.keys())[0]
        print("\nColumns in first file:")
        print(self.au_features[first_file].columns.tolist())
        
        for pair in self.pairs:
            pre_file = pair['pre_file'].split('.')[0]  # e.g., IMG_0003
            post_file = pair['post_file'].split('.')[0]  # e.g., IMG_0004
            
            if pre_file not in self.au_features or post_file not in self.au_features:
                print(f"Skipping pair: {pre_file} or {post_file} not found in AU features")
                continue
            
            # Get only real-valued AU-related columns (ending with _r)
            pre_df = self.au_features[pre_file]
            post_df = self.au_features[post_file]
            
            # Only use real-valued AU features (ending with _r)
            au_columns = [col for col in pre_df.columns if col.endswith('_r')]
            print(f"\nReal-valued AU columns found: {len(au_columns)}")
            print(au_columns)
            
            # Calculate feature differences only for real-valued AU columns
            pre_features = pre_df[au_columns].mean()
            post_features = post_df[au_columns].mean()
            feature_diff = post_features - pre_features
            
            # Get outcome (pain reduction)
            outcome = 1 if pair['change'] >= 4 else 0
            
            feature_diffs.append(feature_diff)
            outcomes.append(outcome)
        
        print(f"Successfully processed {len(feature_diffs)} pairs")
        return pd.DataFrame(feature_diffs), np.array(outcomes)
    
    def calculate_au_effect_sizes(self) -> pd.DataFrame:
        """Calculate Cohen's d effect size for each AU feature."""
        feature_diffs, outcomes = self.get_au_feature_differences()
        
        if len(feature_diffs) == 0:
            print("No feature differences calculated. Check if AU feature files are properly loaded.")
            return pd.DataFrame()
            
        print(f"Calculating effect sizes for {len(feature_diffs.columns)} features")
        effect_sizes = {}
        p_values = {}
        
        # Store raw p-values for FDR correction
        raw_p_values = []
        features_list = []
        
        for feature in feature_diffs.columns:
            # Split data based on outcome
            pos_group = feature_diffs[feature][outcomes == 1]
            neg_group = feature_diffs[feature][outcomes == 0]
            
            if len(pos_group) == 0 or len(neg_group) == 0:
                print(f"Skipping feature {feature}: No data in one of the groups")
                continue
            
            # Calculate Cohen's d
            n1, n2 = len(pos_group), len(neg_group)
            s1, s2 = np.std(pos_group, ddof=1), np.std(neg_group, ddof=1)
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
            d = (np.mean(pos_group) - np.mean(neg_group)) / pooled_std
            
            # Calculate p-value using t-test
            t_stat, p_val = stats.ttest_ind(pos_group, neg_group)
            
            effect_sizes[feature] = d
            raw_p_values.append(p_val)
            features_list.append(feature)
        
        # Apply FDR correction
        from statsmodels.stats.multitest import fdrcorrection
        reject, p_adjusted = fdrcorrection(raw_p_values, alpha=0.05)
        
        # Map adjusted p-values back to features
        for feature, p_adj in zip(features_list, p_adjusted):
            p_values[feature] = p_adj
        
        return pd.DataFrame({
            'feature': effect_sizes.keys(),
            'effect_size': effect_sizes.values(),
            'p_value': p_values.values(),
            'significant': [p_values[feature] < 0.05 for feature in effect_sizes.keys()]
        })
    
    def plot_au_effect_sizes(self, top_n=20, save_path='effect_sizes.png'):
        """Plot the top N AU features by effect size."""
        results_df = self.calculate_au_effect_sizes()
        
        if len(results_df) == 0:
            print("No results to plot. Check if effect sizes were calculated properly.")
            return results_df
            
        results_df = results_df.sort_values('effect_size', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_features = results_df.nlargest(top_n, 'effect_size')
        
        sns.barplot(x='effect_size', y='feature', data=top_features)
        plt.title(f'Top {top_n} AU Features by Cohen\'s d Effect Size')
        plt.xlabel('Cohen\'s d')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return results_df

def main():
    # Set paths
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    au_features_dir = '/Users/hd927/Documents/syracuse_pain_research/AUFeatures/processed'
    
    # Initialize analyzer
    print("Initializing AU feature analyzer...")
    analyzer = AUFeatureAnalyzer(meta_path, feature_dir, au_features_dir)
    
    # Calculate and plot effect sizes
    print("Calculating effect sizes...")
    results_df = analyzer.plot_au_effect_sizes()
    
    # Save results
    if len(results_df) > 0:
        results_df.to_csv('au_feature_effect_sizes.csv', index=False)
        print("Analysis complete! Results saved to au_feature_effect_sizes.csv and effect_sizes.png")
    else:
        print("Analysis failed. No results to save.")

if __name__ == "__main__":
    main() 