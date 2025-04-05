import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Syracuse.MicroExprFeatures.micro_expr_dataset import MicroExprDataset

def calculate_effect_sizes(X, y, feature_names):
    """Calculate Cohen's d effect size and p-values for each feature."""
    effect_sizes = {}
    p_values = {}
    raw_p_values = []
    features_list = []
    
    for i, feature in enumerate(feature_names):
        # Split data based on outcome
        pos_group = X[y == 1, i]
        neg_group = X[y == 0, i]
        
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

def plot_effect_sizes(results_df, top_n=20, save_path='micro_expr_effect_sizes.png'):
    """Plot the top N features by effect size."""
    if len(results_df) == 0:
        print("No results to plot. Check if effect sizes were calculated properly.")
        return results_df
        
    results_df = results_df.sort_values('effect_size', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = results_df.nlargest(top_n, 'effect_size')
    
    sns.barplot(x='effect_size', y='feature', data=top_features)
    plt.title(f'Top {top_n} Micro-expression Features by Cohen\'s d Effect Size')
    plt.xlabel('Cohen\'s d')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return results_df

def analyze_feature_distributions(X, y, feature_names, results_df, save_dir='feature_distributions'):
    """Analyze and plot distributions of significant features."""
    os.makedirs(save_dir, exist_ok=True)
    
    significant_features = results_df[results_df['significant']]['feature'].tolist()
    
    for feature in significant_features:
        idx = feature_names.index(feature)
        pos_group = X[y == 1, idx]
        neg_group = X[y == 0, idx]
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=['Positive', 'Negative'], y=[pos_group, neg_group])
        plt.title(f'Distribution of {feature} by Outcome')
        plt.ylabel('Feature Value')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{feature}_distribution.png'))
        plt.close()

def main():
    # Initialize dataset
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    micro_expr_dir = '/Users/hd927/Documents/syracuse_pain_research/MicroExprFeatures'
    
    dataset = MicroExprDataset(meta_path, feature_dir, micro_expr_dir)
    pre_features, post_features, changes = dataset.get_micro_expr_features()
    feature_names = dataset.get_feature_names()
    
    # Prepare binary outcome
    y_binary = (changes >= 4).astype(int)
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(y_binary)}")
    print(f"Positive cases: {np.sum(y_binary)}")
    print(f"Negative cases: {np.sum(y_binary == 0)}")
    print(f"Total features: {len(feature_names)}")
    
    # Calculate effect sizes
    print("\nCalculating effect sizes...")
    effect_sizes_df = calculate_effect_sizes(pre_features, y_binary, feature_names)
    
    # Plot effect sizes
    print("\nPlotting effect sizes...")
    effect_sizes_df = plot_effect_sizes(effect_sizes_df)
    
    # Analyze distributions of significant features
    print("\nAnalyzing feature distributions...")
    analyze_feature_distributions(pre_features, y_binary, feature_names, effect_sizes_df)
    
    # Save results
    effect_sizes_df.to_csv('micro_expr_effect_sizes.csv', index=False)
    
    # Print summary
    print("\nFeature Analysis Summary:")
    print(f"Total features analyzed: {len(effect_sizes_df)}")
    print(f"Significant features (p < 0.05 after FDR): {np.sum(effect_sizes_df['significant'])}")
    print("\nTop 5 features by effect size:")
    top_5 = effect_sizes_df.nlargest(5, 'effect_size')
    for _, row in top_5.iterrows():
        print(f"- {row['feature']}: d = {row['effect_size']:.3f}, p = {row['p_value']:.3f}")

if __name__ == "__main__":
    main() 