"""
Binary Classification for Treatment Outcome Prediction

This script implements a binary classification model to predict treatment outcomes using MARLIN features.
It performs 3-fold stratified cross-validation with logistic regression to classify patients
into positive (improved) or negative (not improved) outcomes.

Key Features:
1. Dynamic Feature Selection
   - Selects top N features based on effect size from video-level analysis
   - Features are chosen based on absolute effect size in pre-post treatment differences
   - N is specified via command line argument (default: 5)

2. Data Processing
   - Loads MARLIN features from pre and post treatment videos
   - Handles 3D features by taking means across temporal dimensions
   - Averages features across multiple clips for each video
   - Computes pre-post differences for selected features
   - Handles missing values and standardizes features

3. Model Training
   - Uses logistic regression with balanced class weights
   - Implements 3-fold stratified cross-validation
   - Evaluates performance using accuracy and AUC metrics

4. Visualization
   - Generates accuracy and AUC distribution plots
   - Creates ROC curve visualization
   - Saves results and plots to specified output directory

Usage:
    python binary_classification.py [--num_features N]

Arguments:
    --num_features: Number of top features to select based on effect size (default: 5)

Input Requirements:
1. Feature Files:
   - Directory containing .npy files with MARLIN features
   - Filename format: IMG_[ID]_clip_[N]_aligned.npy
   - Features should be pre-computed using MARLIN model
   - Multiple clips per video are supported and averaged

2. Metadata File (Excel):
   - Required columns:
     * subject_id: Unique subject identifier
     * file_name: Video file name (e.g., "IMG_123.MP4")
     * visit_type: Visit type (e.g., '1st-pre', '1st-post')
     * outcome: Treatment outcome ('positive' or 'negative')

3. Analysis Results:
   - outcome_analysis_results/marlin_video_outcome_analysis.csv
   - Contains effect sizes and p-values for feature selection

Output:
1. Classification Results:
   - classification_results.csv: Summary of model performance
   - classification_results.png: Visualization of results

Author: Nan Bi
Date: 2025
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import argparse
import plotly.graph_objects as go

def parse_args():
    parser = argparse.ArgumentParser(description='Binary classification for treatment outcomes')
    parser.add_argument('--num_features', type=int, default=5,
                      help='Number of top features to select based on effect size')
    return parser.parse_args()

# Constants
# FEATURES_DIR = r"C:\pain\syracus\openface_clips\clips\multimodal_marlin_base"
FEATURES_DIR = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
# META_PATH = os.path.join(FEATURES_DIR, "meta_with_outcomes.xlsx")
META_PATH = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
OUTPUT_DIR = "Syracuse/classification_results"

# Global variable for selected features
SELECTED_FEATURES = None

def load_metadata():
    """Load and preprocess metadata file."""
    df = pd.read_excel(META_PATH)
    # Filter for samples with outcomes
    df = df[df['outcome'].notna()]
    
    # Print outcome distribution
    print("\nOutcome distribution in metadata:")
    print(df['outcome'].value_counts())
    
    return df

def get_selected_features(num_features):
    """Get top features based on effect size from video-level analysis."""
    # Read the video-level analysis results
    analysis_path = os.path.join('outcome_analysis_results', 'marlin_video_outcome_analysis.csv')
    df = pd.read_csv(analysis_path)
    
    # Sort by absolute effect size
    df['abs_effect_size'] = df['effect_size'].abs()
    df_sorted = df.sort_values('abs_effect_size', ascending=False)
    
    # Get top num_features
    top_features = df_sorted.head(num_features)
    
    # Convert to list of tuples (feature_idx, effect_size, p_value)
    selected_features = []
    for _, row in top_features.iterrows():
        selected_features.append((
            int(row['feature_idx']),
            float(row['effect_size']),
            float(row['p_value'])
        ))
    
    print(f"\nSelected top {num_features} features:")
    for idx, effect_size, p_value in selected_features:
        print(f"Feature {idx}: effect_size = {effect_size:.3f}, p_value = {p_value:.6f}")
    
    return selected_features

def load_features(subject_id, file_name, visit_type):
    """Load features for a specific subject and visit type.
    
    Args:
        subject_id: Subject ID
        file_name: Video file name
        visit_type: 'pre' or 'post'
        
    Returns:
        features: numpy array of features or None if not found
    """
    # Convert file_name to video_id format (remove .MP4)
    video_id = file_name.replace('.MP4', '')
    
    # Look for feature files matching the pattern
    feature_files = []
    for f in os.listdir(FEATURES_DIR):
        if f.startswith(f"{video_id}_clip_") and f.endswith("_aligned.npy"):
            feature_files.append(f)
    
    if not feature_files:
        print(f"No feature files found for {video_id}")
        return None
    
    # Load and average features across clips
    features_list = []
    for f in feature_files:
        try:
            feature = np.load(os.path.join(FEATURES_DIR, f))
            # If feature is 3D (clips, frames, features), take mean across clips and frames
            if len(feature.shape) == 3:
                feature = np.mean(feature, axis=(0, 1))
            # If feature is 2D (frames, features), take mean across frames
            elif len(feature.shape) == 2:
                feature = np.mean(feature, axis=0)
            # If feature is 1D (features), use as is
            elif len(feature.shape) == 1:
                pass
            else:
                print(f"Unexpected feature shape for {f}: {feature.shape}")
                continue
                
            if np.any(np.isnan(feature)):
                print(f"NaN values found in features for {f}")
                continue
                
            features_list.append(feature)
        except Exception as e:
            print(f"Error loading {f}: {str(e)}")
            continue
    
    if not features_list:
        print(f"No valid features found for {video_id}")
        return None
    
    # Average across all clips
    features = np.mean(features_list, axis=0)
    
    if np.any(np.isnan(features)):
        print(f"NaN values in final features for {video_id}")
        return None
        
    return features

def prepare_data(metadata_df):
    """Prepare feature data and labels for classification.
    
    Args:
        metadata_df: DataFrame containing metadata with outcomes
        
    Returns:
        X: Feature matrix (n_samples, n_features), standardized
        y: Labels (n_samples,)
    """
    # First, get unique subjects with outcomes
    subjects_with_outcomes = metadata_df[metadata_df['outcome'].notna()]['subject_id'].unique()
    
    # Lists to store valid samples
    feature_list = []
    label_list = []
    
    print("\nLoading and computing feature differences...")
    
    # Process each subject
    for subject_id in subjects_with_outcomes:
        try:
            # Get pre and post visits for this subject
            subject_data = metadata_df[metadata_df['subject_id'] == subject_id]
            
            # Get the outcome (should be the same for both visits)
            outcome = subject_data['outcome'].iloc[0]
            
            # Find pre and post visits
            pre_visit = subject_data[subject_data['visit_type'].str.contains('-pre')]
            post_visit = subject_data[subject_data['visit_type'].str.contains('-post')]
            
            if pre_visit.empty or post_visit.empty:
                print(f"Skipping subject {subject_id}: Missing pre or post visit")
                continue
                
            pre_file = pre_visit['file_name'].iloc[0]
            post_file = post_visit['file_name'].iloc[0]
            
            # Load features
            pre_features = load_features(subject_id, pre_file, 'pre')
            post_features = load_features(subject_id, post_file, 'post')
            
            if pre_features is None or post_features is None:
                print(f"Skipping subject {subject_id}: Missing feature files")
                continue
            
            # Calculate pre-post differences for selected features
            feature_diffs = []
            for feature_idx, _, _ in SELECTED_FEATURES:
                diff = post_features[feature_idx] - pre_features[feature_idx]
                feature_diffs.append(diff)
            
            # Only add if we have all features
            if len(feature_diffs) == len(SELECTED_FEATURES):
                feature_list.append(feature_diffs)
                label_list.append(1 if outcome == 'positive' else 0)
            else:
                print(f"Skipping subject {subject_id}: Incomplete feature differences")
                
        except Exception as e:
            print(f"Error processing subject {subject_id}: {str(e)}")
            continue
    
    if not feature_list:
        raise ValueError("No valid samples found after processing")
    
    # Convert lists to numpy arrays
    X = np.array(feature_list)
    y = np.array(label_list)
    
    print(f"\nProcessed data shape: X={X.shape}, y={y.shape}")
    
    # Print class distribution
    y_int = y.astype(int)
    counts = np.bincount(y_int)
    print("\nClass distribution:")
    print(f"Negative (0): {counts[0]} samples")
    print(f"Positive (1): {counts[1]} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("\nFeatures have been standardized (mean=0, std=1)")
    
    return X, y

def stratified_cross_validation(X, y, n_splits=3):
    """Implement stratified K-fold cross-validation.
    
    Args:
        X: Feature matrix
        y: Labels
        n_splits: Number of folds
        
    Returns:
        mean_acc, std_acc: Mean and std of accuracy scores
        mean_auc, std_auc: Mean and std of AUC scores
        all_probs: Predicted probabilities for all samples
        all_true: True labels for all samples
    """
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store metrics
    accuracies = []
    aucs = []
    all_probs = np.zeros(len(y))  # Store probabilities for all samples
    all_true = np.zeros(len(y))   # Store true labels for all samples
    
    print("\nRunning 3-fold stratified cross-validation...")
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Print fold information
        print(f"\nFold {fold}/{n_splits}")
        print(f"Training set: {len(y_train)} samples")
        print(f"Test set: {len(y_test)} samples")
        
        # Print class distribution
        train_counts = np.bincount(y_train.astype(int))
        test_counts = np.bincount(y_test.astype(int))
        print(f"Training set distribution - Negative: {train_counts[0]}, Positive: {train_counts[1]}")
        print(f"Test set distribution - Negative: {test_counts[0]}, Positive: {test_counts[1]}")
        
        # Train model
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Store predictions
        all_probs[test_idx] = y_prob
        all_true[test_idx] = y_test
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)
        
        print(f"Fold {fold} - Accuracy: {acc:.3f}, AUC: {auc_score:.3f}")
        
        accuracies.append(acc)
        aucs.append(auc_score)
    
    # Calculate mean and std of metrics
    mean_acc, std_acc = np.mean(accuracies), np.std(accuracies)
    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    
    return mean_acc, std_acc, mean_auc, std_auc, all_probs, all_true

def plot_results(accuracies, aucs, all_probs, all_true, output_dir, X=None, y=None, feature_indices=None):
    """Plot and save classification results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create static plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Accuracy distribution
    plt.subplot(1, 3, 1)
    plt.hist(accuracies, bins=10)
    plt.title('Accuracy Distribution')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    
    # Plot 2: AUC distribution
    plt.subplot(1, 3, 2)
    plt.hist(aucs, bins=10)
    plt.title('AUC Distribution')
    plt.xlabel('AUC')
    plt.ylabel('Count')
    
    # Plot 3: ROC curve
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(all_true, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_results.png'))
    plt.close()
    
    # Create feature coefficients plot
    if X is not None and y is not None and feature_indices is not None:
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        coefficients = model.coef_[0]
        
        plt.figure(figsize=(10, 6))
        plt.bar([f'Feature {idx}' for idx in feature_indices], coefficients,
                color=['red' if coef < 0 else 'blue' for coef in coefficients])
        plt.title('Feature Coefficients')
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_coefficients.png'))
        plt.close()
    
    # Create interactive 3D visualization if we have 3 features
    if X is not None and y is not None and feature_indices is not None and len(feature_indices) == 3:
        # Train logistic regression model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Create mesh grid for the decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
        
        xx, yy, zz = np.meshgrid(
            np.linspace(x_min, x_max, 50),
            np.linspace(y_min, y_max, 50),
            np.linspace(z_min, z_max, 50)
        )
        
        # Create 3D scatter plot with decision boundary
        fig = go.Figure()
        
        # Add scatter points
        scatter = go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=y,
                colorscale=[[0, 'red'], [1, 'blue']],
                showscale=False  # Hide the colorbar
            ),
            text=[f'Subject {i}<br>Outcome: {"Positive" if yi == 1 else "Negative"}' 
                  for i, yi in enumerate(y)],
            hoverinfo='text',
            name='Subjects'
        )
        
        # Split into two traces for better legend
        pos_mask = y == 1
        neg_mask = y == 0
        
        # Positive class points
        pos_scatter = go.Scatter3d(
            x=X[pos_mask, 0],
            y=X[pos_mask, 1],
            z=X[pos_mask, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='blue'
            ),
            text=[f'Subject {i}<br>Outcome: Positive' 
                  for i in np.where(pos_mask)[0]],
            hoverinfo='text',
            name='Positive Outcome'
        )
        
        # Negative class points
        neg_scatter = go.Scatter3d(
            x=X[neg_mask, 0],
            y=X[neg_mask, 1],
            z=X[neg_mask, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='red'
            ),
            text=[f'Subject {i}<br>Outcome: Negative' 
                  for i in np.where(neg_mask)[0]],
            hoverinfo='text',
            name='Negative Outcome'
        )
        
        fig.add_trace(pos_scatter)
        fig.add_trace(neg_scatter)
        
        # Calculate and add decision boundary surface
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
                colorscale=[[0, 'gray'], [1, 'gray']],  # Single color for the boundary
                showscale=False,  # Hide the colorbar
                name='Decision Boundary'
            )
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            title_text='3D Feature Visualization with Decision Boundary',
            scene=dict(
                xaxis_title=f'Feature {feature_indices[0]}',
                yaxis_title=f'Feature {feature_indices[1]}',
                zaxis_title=f'Feature {feature_indices[2]}'
            )
        )
        
        # Save the interactive plot
        fig.write_html(os.path.join(output_dir, 'interactive_3d_visualization.html'))
        print(f"\nInteractive 3D visualization saved to {os.path.join(output_dir, 'interactive_3d_visualization.html')}")
        print(f"Static plots saved to {output_dir}")

def main():
    """Main function to run the classification pipeline."""
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Parse command line arguments
        args = parse_args()
        
        # Get selected features based on command line argument
        global SELECTED_FEATURES
        SELECTED_FEATURES = get_selected_features(args.num_features)
        
        # Load metadata
        print("Loading metadata...")
        metadata_df = load_metadata()
        print(f"Loaded {len(metadata_df)} samples with outcomes")
        
        # Prepare data
        print("Preparing feature data...")
        X, y = prepare_data(metadata_df)
        print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        
        # Run stratified cross-validation
        mean_acc, std_acc, mean_auc, std_auc, all_probs, all_true = stratified_cross_validation(X, y, n_splits=3)
        
        # Print results
        print("\nClassification Results:")
        print(f"Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        
        # Save results
        results = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'n_samples': len(y),
            'n_features': len(SELECTED_FEATURES),
            'selected_features': [feature_idx for feature_idx, _, _ in SELECTED_FEATURES]
        }
        
        pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR, 'classification_results.csv'), index=False)
        
        # Plot results
        print("\nPlotting results...")
        # Create arrays of repeated values for plotting
        accuracies = np.repeat(mean_acc, 3)  # 3 folds
        aucs = np.repeat(mean_auc, 3)        # 3 folds
        plot_results(accuracies, aucs, all_probs, all_true, OUTPUT_DIR, X, y, 
                    [feature_idx for feature_idx, _, _ in SELECTED_FEATURES])
        print(f"Results saved to {OUTPUT_DIR}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {e}")
    except Exception as e:
        print(f"Error during classification: {e}")

if __name__ == "__main__":
    main() 