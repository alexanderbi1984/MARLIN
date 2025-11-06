import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sys
import os

# Add the binary_classification directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'binary_outcome_classification'))
from binary_classification import load_metadata, load_features, SELECTED_FEATURES, get_selected_features

def create_feature_visualization():
    # Load metadata
    metadata_df = load_metadata()
    
    # Convert pain_level to numeric, coercing errors to NaN
    metadata_df['pain_level'] = pd.to_numeric(metadata_df['pain_level'], errors='coerce')
    
    # Get selected features
    selected_features = get_selected_features(5)  # Get top 5 features
    feature_indices = [idx for idx, _, _ in selected_features]
    effect_sizes = [effect for _, effect, _ in selected_features]
    
    # Lists to store data
    feature_diffs = []
    outcomes = []
    pain_changes = []
    subject_ids = []
    
    print("\nProcessing subjects...")
    # Process each subject
    for subject_id in metadata_df[metadata_df['outcome'].notna()]['subject_id'].unique():
        try:
            subject_data = metadata_df[metadata_df['subject_id'] == subject_id]
            
            # Find pre and post visits (handling NaN values)
            pre_visit = subject_data[subject_data['visit_type'].fillna('').str.contains('-pre', na=False)]
            post_visit = subject_data[subject_data['visit_type'].fillna('').str.contains('-post', na=False)]
            
            if pre_visit.empty or post_visit.empty:
                print(f"Skipping subject {subject_id}: Missing pre or post visit")
                continue
                
            pre_file = pre_visit['file_name'].iloc[0]
            post_file = post_visit['file_name'].iloc[0]
            
            # Load features
            pre_features = load_features(subject_id, pre_file, 'pre')
            post_features = load_features(subject_id, post_file, 'post')
            
            if pre_features is None or post_features is None:
                print(f"Skipping subject {subject_id}: Missing features")
                continue
            
            # Calculate differences for selected features
            diffs = []
            for idx in feature_indices:
                diff = post_features[idx] - pre_features[idx]
                diffs.append(diff)
                
            if len(diffs) == len(feature_indices):
                feature_diffs.append(diffs)
                outcomes.append(subject_data['outcome'].iloc[0])
                subject_ids.append(subject_id)
                
                # Calculate pain change if available
                pre_pain = pre_visit['pain_level'].iloc[0]
                post_pain = post_visit['pain_level'].iloc[0]
                if pd.notna(pre_pain) and pd.notna(post_pain):
                    pain_changes.append(float(pre_pain - post_pain))
                else:
                    print(f"Missing pain levels for subject {subject_id}")
                    pain_changes.append(np.nan)
        except Exception as e:
            print(f"Error processing subject {subject_id}: {str(e)}")
            continue
    
    print(f"\nProcessed {len(feature_diffs)} subjects successfully")
    
    if len(feature_diffs) == 0:
        print("No valid subjects found for visualization")
        return
    
    # Convert to numpy arrays
    X = np.array(feature_diffs)
    y = np.array(outcomes)
    pain_changes = np.array(pain_changes)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression model on top 3 features only
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled[:, :3], y)  # Use only top 3 features
    
    # Create mesh grid for the decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    z_min, z_max = X_scaled[:, 2].min() - 1, X_scaled[:, 2].max() + 1
    
    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, 50),
        np.linspace(y_min, y_max, 50),
        np.linspace(z_min, z_max, 50)
    )
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'bar'}],
               [{'type': 'scatter', 'colspan': 2}, None]],
        subplot_titles=(
            '3D Feature Visualization with Decision Boundary',
            'Feature Effect Sizes',
            'Pain Change Distribution'
        )
    )
    
    # 1. 3D scatter plot of top 3 features with decision boundary
    # Add scatter points
    scatter = go.Scatter3d(
        x=X_scaled[:, 0],
        y=X_scaled[:, 1],
        z=X_scaled[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=pain_changes,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title='Pain Reduction')
        ),
        text=[f'Subject {sid}<br>Outcome: {out}<br>Pain Change: {pc:.1f}' 
              for sid, out, pc in zip(subject_ids, outcomes, pain_changes)],
        hoverinfo='text',
        name='Subjects'
    )
    fig.add_trace(scatter, row=1, col=1)
    
    # Add decision boundary surface
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Create isosurface for decision boundary (probability = 0.5)
    fig.add_trace(
        go.Volume(
            x=xx.flatten(),
            y=yy.flatten(),
            z=zz.flatten(),
            value=Z.flatten(),
            isomin=0.45,
            isomax=0.55,
            opacity=0.1,
            surface_count=1,
            colorscale='RdBu',
            name='Decision Boundary'
        ),
        row=1, col=1
    )
    
    # 2. Bar plot of effect sizes
    bar = go.Bar(
        x=[f'Feature {idx}' for idx in feature_indices],
        y=effect_sizes,
        marker_color=['red' if es < 0 else 'blue' for es in effect_sizes]
    )
    fig.add_trace(bar, row=1, col=2)
    
    # 3. Pain change distribution
    hist = go.Histogram(
        x=pain_changes[~np.isnan(pain_changes)],  # Remove NaN values for histogram
        nbinsx=10,
        name='Pain Change Distribution'
    )
    fig.add_trace(hist, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        showlegend=True,
        title_text='Classification Results Visualization with Decision Boundary',
    )
    
    # Update 3D axis labels
    fig.update_scenes(
        dict(
            xaxis_title=f'Feature {feature_indices[0]}',
            yaxis_title=f'Feature {feature_indices[1]}',
            zaxis_title=f'Feature {feature_indices[2]}'
        )
    )
    
    # Update bar plot axis
    fig.update_xaxes(title_text='Features', row=1, col=2)
    fig.update_yaxes(title_text='Effect Size', row=1, col=2)
    
    # Update histogram axis
    fig.update_xaxes(title_text='Pain Reduction', row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2, col=1)
    
    # Save the interactive plot
    fig.write_html('Syracuse/classification_results/interactive_visualization.html')
    print("\nInteractive visualization saved to 'Syracuse/classification_results/interactive_visualization.html'")

if __name__ == '__main__':
    create_feature_visualization() 