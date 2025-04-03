from syracuse_dataset import SyracuseDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def plot_temporal_patterns(analysis: Dict, pair: Dict, feature_indices: List[int]):
    """Plot temporal patterns for selected features."""
    pre_stats = analysis['pre_stats']
    post_stats = analysis['post_stats']
    
    n_features = len(feature_indices)
    fig, axes = plt.subplots(n_features, 3, figsize=(15, 5*n_features))
    
    for i, feat_idx in enumerate(feature_indices):
        # Plot mean feature values over time
        axes[i, 0].plot(pre_stats['means'][:, feat_idx], label='Pre')
        axes[i, 0].plot(post_stats['means'][:, feat_idx], label='Post')
        axes[i, 0].set_title(f'Feature {feat_idx} Mean Over Time')
        axes[i, 0].set_xlabel('Clip Number')
        axes[i, 0].set_ylabel('Feature Value')
        axes[i, 0].legend()
        
        # Plot rate of change
        axes[i, 1].plot(pre_stats['rate_of_change'][:, feat_idx], label='Pre')
        axes[i, 1].plot(post_stats['rate_of_change'][:, feat_idx], label='Post')
        axes[i, 1].set_title(f'Feature {feat_idx} Rate of Change')
        axes[i, 1].set_xlabel('Clip Number')
        axes[i, 1].set_ylabel('Rate of Change')
        axes[i, 1].legend()
        
        # Plot acceleration
        axes[i, 2].plot(pre_stats['acceleration'][:, feat_idx], label='Pre')
        axes[i, 2].plot(post_stats['acceleration'][:, feat_idx], label='Post')
        axes[i, 2].set_title(f'Feature {feat_idx} Acceleration')
        axes[i, 2].set_xlabel('Clip Number')
        axes[i, 2].set_ylabel('Acceleration')
        axes[i, 2].legend()
    
    plt.suptitle(f'Temporal Patterns for Subject {pair["subject"]} (Pain Change: {pair["change"]:.1f})')
    plt.tight_layout()
    return fig

def analyze_temporal_patterns():
    # Initialize dataset
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Analyze all temporal patterns
    analyses, correlations = dataset.analyze_all_temporal_patterns()
    
    # Print overall correlations
    print("\n=== Feature-Pain Correlations ===")
    for metric, corr in correlations.items():
        print(f"{metric}: {corr:.3f}")
    
    # Find features with strongest temporal patterns
    mean_diffs = np.array([analysis['mean_diff'] for analysis in analyses])
    temporal_corrs = np.array([analysis['temporal_correlation'] for analysis in analyses])
    stability_changes = np.array([analysis['stability_change'] for analysis in analyses])
    
    # Compute feature-wise correlations with pain reduction
    changes = np.array([pair['change'] for pair in dataset.pairs])
    feature_correlations = {
        'mean_diff': np.array([np.corrcoef(changes, mean_diffs[:, i])[0, 1] 
                             for i in range(mean_diffs.shape[1])]),
        'temporal_corr': np.array([np.corrcoef(changes, temporal_corrs[:, i])[0, 1] 
                                 for i in range(temporal_corrs.shape[1])]),
        'stability': np.array([np.corrcoef(changes, stability_changes[:, i])[0, 1] 
                             for i in range(stability_changes.shape[1])])
    }
    
    # Find top correlated features
    top_features = {}
    for metric, corrs in feature_correlations.items():
        top_idx = np.argsort(np.abs(corrs))[-5:]  # Top 5 features
        top_features[metric] = {
            'indices': top_idx,
            'correlations': corrs[top_idx]
        }
    
    # Print top correlated features
    print("\n=== Top Correlated Features ===")
    for metric, features in top_features.items():
        print(f"\n{metric}:")
        for idx, corr in zip(features['indices'], features['correlations']):
            print(f"Feature {idx}: {corr:.3f}")
    
    # Plot temporal patterns for a few example pairs
    # Select pairs with different levels of pain reduction
    changes = np.array([pair['change'] for pair in dataset.pairs])
    example_indices = [
        np.argmax(changes),  # Maximum reduction
        np.argmin(changes),  # Minimum reduction
        len(changes)//2      # Median reduction
    ]
    
    for idx in example_indices:
        pair = dataset.pairs[idx]
        analysis = analyses[idx]
        
        # Plot patterns for top features
        top_feature_indices = top_features['mean_diff']['indices'][-3:]  # Use top 3 features
        fig = plot_temporal_patterns(analysis, pair, top_feature_indices)
        plt.savefig(f'temporal_patterns_subj{pair["subject"]}.png')
        plt.close()

if __name__ == "__main__":
    analyze_temporal_patterns() 