import pandas as pd
import numpy as np
import plotly.express as px
from syracuse_dataset import SyracuseDataset
import os

def load_feature_importance():
    """Load and sort features by effect size."""
    feature_importance = pd.read_csv('/Users/hd927/Documents/MARLIN/outcome_analysis_results/marlin_clip_outcome_analysis.csv')
    # Sort by absolute effect size and get top 3
    feature_importance['abs_effect_size'] = feature_importance['effect_size'].abs()
    top_features = feature_importance.nlargest(3, 'abs_effect_size')
    return top_features

def create_3d_visualization():
    # Initialize dataset
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    dataset = SyracuseDataset(meta_path, feature_dir)
    
    # Get top 3 features
    top_features = load_feature_importance()
    feature_indices = top_features['feature_idx'].values.astype(int)
    
    # Get all features and changes
    pre_features, post_features, changes = dataset.get_all_features()
    
    # Print shapes for debugging
    print(f"Pre features shape: {pre_features.shape}")
    print(f"Post features shape: {post_features.shape}")
    
    # Calculate mean features across clips and frames for each video
    pre_features_mean = np.mean(pre_features, axis=(1, 2))  # Average across clips and frames
    post_features_mean = np.mean(post_features, axis=(1, 2))  # Average across clips and frames
    
    # Calculate feature differences (post - pre)
    feature_diffs = post_features_mean - pre_features_mean  # Shape: (N, 768)
    
    # Extract top 3 features
    selected_features = feature_diffs[:, feature_indices]  # Shape: (N, 3)
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Feature 1': selected_features[:, 0],
        'Feature 2': selected_features[:, 1],
        'Feature 3': selected_features[:, 2],
        'Pain Reduction': changes
    })
    
    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        plot_data,
        x='Feature 1',
        y='Feature 2',
        z='Feature 3',
        color='Pain Reduction',
        color_continuous_scale='RdYlBu_r',  # Red for high reduction, blue for low
        title='Top 3 Features vs Pain Reduction',
        labels={
            'Feature 1': f'Feature {feature_indices[0]} (Effect Size: {top_features.iloc[0].effect_size:.3f})',
            'Feature 2': f'Feature {feature_indices[1]} (Effect Size: {top_features.iloc[1].effect_size:.3f})',
            'Feature 3': f'Feature {feature_indices[2]} (Effect Size: {top_features.iloc[2].effect_size:.3f})'
        }
    )
    
    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            aspectmode='cube',  # Make the plot cubic for better visualization
        ),
        coloraxis_colorbar_title='Pain Reduction',
        showlegend=False
    )
    
    # Update marker properties
    fig.update_traces(marker=dict(size=8))
    
    # Save the plot as an HTML file for interactive viewing
    fig.write_html('Syracuse/top_features_3d.html')
    
    print(f"\nSelected features: {feature_indices}")
    print(f"Their effect sizes: {top_features.effect_size.values}")
    print(f"Number of pairs analyzed: {len(changes)}")

if __name__ == '__main__':
    create_3d_visualization() 