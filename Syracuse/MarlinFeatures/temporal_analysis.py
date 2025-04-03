import numpy as np
from syracuse_dataset import SyracuseDataset
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def compute_temporal_features(features):
    """
    Compute temporal features from the raw feature sequence.
    
    Args:
        features: Array of shape (N, 4, 768) containing features for N clips
        
    Returns:
        Dictionary containing various temporal features
    """
    # Ensure we use exactly 14 clips
    features = features[:14]
    
    # Average across the 4 frames in each clip
    clip_features = np.mean(features, axis=1)  # (14, 768)
    
    # Compute temporal statistics
    temporal_stats = {
        'mean': np.mean(clip_features, axis=0),  # (768,)
        'std': np.std(clip_features, axis=0),  # (768,)
        'trend': np.polyfit(np.arange(len(clip_features)), clip_features, deg=1)[0],  # (768,)
        'max_diff': np.max(np.abs(np.diff(clip_features, axis=0)), axis=0),  # (768,)
        'temporal_std': np.std(clip_features, axis=0),  # (768,)
    }
    
    return temporal_stats

def analyze_temporal_patterns():
    # Initialize dataset
    feature_dir = "/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2"
    meta_path = os.path.join(feature_dir, "meta_with_outcomes.xlsx")
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Collect temporal features and pain changes
    pre_temporal_features = []
    post_temporal_features = []
    pain_changes = []
    
    for pair in dataset.pairs:
        try:
            # Load features
            pre_features, post_features = dataset.load_features_for_pair(pair)
            
            # Compute temporal statistics
            pre_stats = compute_temporal_features(pre_features)
            post_stats = compute_temporal_features(post_features)
            
            # Collect features and pain change
            pre_temporal_features.append(np.concatenate([
                pre_stats['mean'],
                pre_stats['std'],
                pre_stats['trend'],
                pre_stats['max_diff'],
                pre_stats['temporal_std']
            ]))
            
            post_temporal_features.append(np.concatenate([
                post_stats['mean'],
                post_stats['std'],
                post_stats['trend'],
                post_stats['max_diff'],
                post_stats['temporal_std']
            ]))
            
            pain_changes.append(pair['change'])
            
        except Exception as e:
            print(f"Error processing Subject {pair['subject']}, Visit {pair['visit_number']}: {str(e)}")
            continue
    
    # Convert to numpy arrays
    pre_temporal_features = np.array(pre_temporal_features)
    post_temporal_features = np.array(post_temporal_features)
    pain_changes = np.array(pain_changes)
    
    # Standardize features
    scaler = StandardScaler()
    pre_temporal_features_scaled = scaler.fit_transform(pre_temporal_features)
    post_temporal_features_scaled = scaler.fit_transform(post_temporal_features)
    
    # Analyze relationship with pain change
    pre_model = LinearRegression()
    pre_model.fit(pre_temporal_features_scaled, pain_changes)
    pre_r2 = r2_score(pain_changes, pre_model.predict(pre_temporal_features_scaled))
    
    post_model = LinearRegression()
    post_model.fit(post_temporal_features_scaled, pain_changes)
    post_r2 = r2_score(pain_changes, post_model.predict(post_temporal_features_scaled))
    
    print("\nTemporal Analysis Results:")
    print(f"Pre-treatment temporal features R² with pain change: {pre_r2:.3f}")
    print(f"Post-treatment temporal features R² with pain change: {post_r2:.3f}")
    
    # Plot pain changes vs predicted pain changes
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    pre_pred = pre_model.predict(pre_temporal_features_scaled)
    plt.scatter(pain_changes, pre_pred)
    plt.plot([min(pain_changes), max(pain_changes)], [min(pain_changes), max(pain_changes)], 'r--')
    plt.xlabel('Actual Pain Change')
    plt.ylabel('Predicted Pain Change')
    plt.title('Pre-treatment Temporal Features')
    
    plt.subplot(1, 2, 2)
    post_pred = post_model.predict(post_temporal_features_scaled)
    plt.scatter(pain_changes, post_pred)
    plt.plot([min(pain_changes), max(pain_changes)], [min(pain_changes), max(pain_changes)], 'r--')
    plt.xlabel('Actual Pain Change')
    plt.ylabel('Predicted Pain Change')
    plt.title('Post-treatment Temporal Features')
    
    plt.tight_layout()
    plt.savefig('temporal_analysis.png')
    plt.close()

if __name__ == "__main__":
    analyze_temporal_patterns() 