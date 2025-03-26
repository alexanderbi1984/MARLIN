"""
Binary classification script for predicting treatment outcomes using MARLIN features.
This script implements Leave-One-Out-Video (LOOV) cross-validation with logistic regression
to classify positive vs negative treatment outcomes.

Author: [Your Name]
Date: 2024
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

# Constants
FEATURES_DIR = r"C:\pain\syracus\openface_clips\clips\multimodal_marlin_base"
META_PATH = os.path.join(FEATURES_DIR, "meta_with_outcomes.xlsx")
OUTPUT_DIR = "Syracuse/classification_results"

# Features selected based on effect size from video-level analysis
# Format: (feature_idx, effect_size, p_value)
SELECTED_FEATURES = [
    (662, 1.613, 0.000921),  # Largest effect size
    (316, 1.372, 0.003643),  # Second largest effect size
    (587, 1.318, 0.004919),  # Third largest effect size
    (629, -1.293, 0.005661),  # Fourth largest effect size
    (143, -1.281, 0.006031)   # Fifth largest effect size
]

def load_metadata():
    """Load and preprocess metadata file."""
    df = pd.read_excel(META_PATH)
    # Filter for samples with outcomes
    df = df[df['outcome'].notna()]
    
    # Print outcome distribution
    print("\nOutcome distribution in metadata:")
    print(df['outcome'].value_counts())
    
    return df

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
        return None
    
    # Load and average features across clips
    features_list = []
    for f in feature_files:
        try:
            feature = np.load(os.path.join(FEATURES_DIR, f))
            # If feature is 3D, take mean across temporal dimension
            if len(feature.shape) == 3:
                feature = np.mean(feature, axis=1)
            # If feature is still 2D, take mean across remaining temporal dimension
            if len(feature.shape) == 2:
                feature = np.mean(feature, axis=0)
            features_list.append(feature)
        except Exception as e:
            continue
    
    if not features_list:
        return None
    
    # Average across all clips
    return np.mean(features_list, axis=0)

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
    n_samples = len(subjects_with_outcomes)
    n_features = len(SELECTED_FEATURES)
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    
    print("\nLoading and computing feature differences...")
    # Process each subject
    for i, subject_id in enumerate(subjects_with_outcomes):
        # Get pre and post visits for this subject
        subject_data = metadata_df[metadata_df['subject_id'] == subject_id]
        
        # Get the outcome (should be the same for both visits)
        outcome = subject_data['outcome'].iloc[0]
        
        # Find pre and post visits
        pre_visit = subject_data[subject_data['visit_type'].str.contains('-pre')]
        post_visit = subject_data[subject_data['visit_type'].str.contains('-post')]
        
        if pre_visit.empty or post_visit.empty:
            continue
            
        pre_file = pre_visit['file_name'].iloc[0]
        post_file = post_visit['file_name'].iloc[0]
        
        # Load features
        pre_features = load_features(subject_id, pre_file, 'pre')
        post_features = load_features(subject_id, post_file, 'post')
        
        if pre_features is not None and post_features is not None:
            # Calculate pre-post differences for selected features
            for j, (feature_idx, _, _) in enumerate(SELECTED_FEATURES):
                X[i, j] = post_features[feature_idx] - pre_features[feature_idx]
            
            # Get outcome (1 for positive, 0 for negative)
            y[i] = 1 if outcome == 'positive' else 0
    
    # Remove samples with missing features
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
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

def plot_results(accuracies, aucs, all_probs, all_true, output_dir):
    """Plot and save classification results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 subplots
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

def main():
    """Main function to run the classification pipeline."""
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
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
        plot_results(accuracies, aucs, all_probs, all_true, OUTPUT_DIR)
        print(f"Results saved to {OUTPUT_DIR}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {e}")
    except Exception as e:
        print(f"Error during classification: {e}")

if __name__ == "__main__":
    main() 