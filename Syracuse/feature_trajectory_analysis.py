import numpy as np
from scipy import stats
from syracuse_dataset import SyracuseDataset
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def compute_pre_post_changes(pre_features: np.ndarray, post_features: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute changes between pre and post treatment features.
    
    Args:
        pre_features: Array of shape (14, 4, 768) for pre-treatment
        post_features: Array of shape (14, 4, 768) for post-treatment
        
    Returns:
        Dictionary containing various change metrics
    """
    # Average across frames in each clip
    pre_clip_features = np.mean(pre_features, axis=1)  # Shape: (14, 768)
    post_clip_features = np.mean(post_features, axis=1)  # Shape: (14, 768)
    
    metrics = {}
    
    # 1. Overall change (post - pre)
    metrics['overall_change'] = post_clip_features - pre_clip_features
    
    # 2. Relative change (percentage change)
    metrics['relative_change'] = (post_clip_features - pre_clip_features) / (np.abs(pre_clip_features) + 1e-6)
    
    # 3. Maximum change in any clip
    metrics['max_change'] = np.max(np.abs(post_clip_features - pre_clip_features), axis=0)
    
    # 4. Change in variability
    pre_std = np.std(pre_clip_features, axis=0)
    post_std = np.std(post_clip_features, axis=0)
    metrics['std_change'] = post_std - pre_std
    
    # 5. Mean feature values (pre and post)
    metrics['pre_mean'] = np.mean(pre_clip_features, axis=0)
    metrics['post_mean'] = np.mean(post_clip_features, axis=0)
    
    return metrics

def analyze_pre_post_changes():
    """
    Analyze changes between pre and post treatment videos.
    """
    # Initialize dataset
    feature_dir = "/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2"
    meta_path = os.path.join(feature_dir, "meta_with_outcomes.xlsx")
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Store changes for all pairs
    all_changes = []
    subjects = []
    pain_changes = []
    
    for pair in dataset.pairs:
        try:
            # Load features
            pre_features, post_features = dataset.load_features_for_pair(pair)
            
            # Compute changes
            changes = compute_pre_post_changes(pre_features, post_features)
            
            all_changes.append(changes)
            subjects.append(pair['subject'])
            pain_changes.append(pair['change'])
            
        except Exception as e:
            print(f"Error processing Subject {pair['subject']}, Visit {pair['visit_number']}: {str(e)}")
            continue
    
    # Analyze changes
    analyze_changes_with_pain(all_changes, subjects, pain_changes)
    
    # Visualize changes
    visualize_changes(all_changes, subjects, pain_changes)

def analyze_changes_with_pain(changes: List[Dict], subjects: List[int], pain_changes: List[float]):
    """
    Analyze how feature changes relate to pain reduction.
    """
    print("\nAnalyzing relationships between feature changes and pain reduction:")
    
    # Convert changes to arrays for analysis
    change_types = ['overall_change', 'relative_change', 'max_change', 'std_change']
    
    for change_type in change_types:
        try:
            # Check if the metric is already averaged
            sample_value = changes[0][change_type]
            if len(sample_value.shape) == 2:  # Non-averaged metric (clips × features)
                values = np.array([np.mean(c[change_type], axis=0) for c in changes])
            else:  # Already averaged metric (just features)
                values = np.array([c[change_type] for c in changes])
            
            # Skip if no values
            if values.size == 0:
                continue
                
            # Find features with strongest correlations
            correlations = []
            for feat_idx in range(values.shape[1]):
                corr, p_value = stats.spearmanr(values[:, feat_idx], pain_changes)
                if abs(corr) > 0.5 and p_value < 0.05:
                    correlations.append((feat_idx, corr, p_value))
            
            if correlations:
                print(f"\n{change_type} features with strong correlations:")
                for feat_idx, corr, p_value in sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:5]:
                    print(f"Feature {feat_idx}:")
                    print(f"  Correlation: {corr:.3f}, p-value: {p_value:.3f}")
                    
                    # Calculate mean change
                    mean_change = np.mean(values[:, feat_idx])
                    print(f"  Mean change: {mean_change:.4f}")
                    
                    # For overall changes, also show pre and post means
                    if change_type == 'overall_change':
                        pre_mean = np.mean([c['pre_mean'][feat_idx] for c in changes])
                        post_mean = np.mean([c['post_mean'][feat_idx] for c in changes])
                        print(f"  Pre-treatment mean: {pre_mean:.4f}")
                        print(f"  Post-treatment mean: {post_mean:.4f}")
        except Exception as e:
            print(f"Error processing {change_type}: {str(e)}")
            continue

def visualize_changes(changes: List[Dict], subjects: List[int], pain_changes: List[float]):
    """
    Create visualizations of feature changes between pre and post treatment.
    """
    # Sort subjects by pain change
    sorted_indices = np.argsort(pain_changes)
    
    # Select features with strongest correlations
    # We'll focus on the overall change metric
    values = np.array([np.mean(c['overall_change'], axis=0) for c in changes])
    correlations = []
    for feat_idx in range(values.shape[1]):
        corr, p_value = stats.spearmanr(values[:, feat_idx], pain_changes)
        if abs(corr) > 0.5 and p_value < 0.05:
            correlations.append((feat_idx, corr, p_value))
    
    # Select top 4 features
    top_features = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:4]
    
    plt.figure(figsize=(15, 10))
    for i, (feat_idx, corr, p_value) in enumerate(top_features):
        plt.subplot(2, 2, i+1)
        
        # Plot changes for each subject
        for idx in sorted_indices:
            alpha = 0.5 + 0.5 * (pain_changes[idx] - min(pain_changes)) / (max(pain_changes) - min(pain_changes))
            plt.plot(changes[idx]['overall_change'][:, feat_idx], 
                    'o-', alpha=alpha, label=f'Subject {subjects[idx]}')
        
        plt.xlabel('Clip Number')
        plt.ylabel('Feature Change (Post - Pre)')
        plt.title(f'Feature {feat_idx} Changes\nCorrelation: {corr:.3f} (p={p_value:.3f})')
        
        # Add horizontal line at zero
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_results/pre_post_changes.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_pre_post_changes() 