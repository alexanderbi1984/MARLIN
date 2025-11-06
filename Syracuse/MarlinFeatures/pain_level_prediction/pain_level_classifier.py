import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from scipy import stats
import os
from pathlib import Path
import argparse

def extract_features_and_pain_levels(meta_path, feature_dir, pain_threshold=5.0):
    """
    Extract features and corresponding pain levels from all available videos,
    converting pain levels to binary classification based on threshold.
    
    Args:
        meta_path: Path to the meta data file
        feature_dir: Directory containing feature files
        pain_threshold: Threshold for binary classification
        
    Returns:
        Dictionary containing dataset information
    """
    print("Loading metadata...")
    # Load metadata directly
    meta_df = pd.read_excel(meta_path)
    
    # Convert pain_level to numeric, handling non-numeric values
    meta_df['pain_level'] = pd.to_numeric(meta_df['pain_level'], errors='coerce')
    
    # Filter out rows with missing pain levels
    valid_df = meta_df.dropna(subset=['pain_level'])
    print(f"Found {len(valid_df)} videos with valid pain levels out of {len(meta_df)} total videos")
    
    # Convert pain levels to binary classification
    valid_df['pain_class'] = (valid_df['pain_level'] >= pain_threshold).astype(int)
    
    # Collect data from all videos with valid pain levels
    all_features = []
    all_pain_classes = []
    all_file_names = []
    
    for idx, row in valid_df.iterrows():
        file_name = row['file_name']
        pain_class = row['pain_class']
        
        # Get clips for this video
        try:
            clips = sorted([f for f in os.listdir(feature_dir) 
                            if f.startswith(file_name.replace('.MP4', '_clip_')) and f.endswith('_aligned.npy')])[:14]
        except:
            print(f"Warning: Could not find clips for {file_name}, skipping")
            continue
        
        if len(clips) < 14:
            print(f"Warning: Not enough clips for {file_name}, found {len(clips)}, skipping")
            continue
        
        # Load and process video features
        video_features = []
        for clip in clips:
            clip_path = os.path.join(feature_dir, clip)
            try:
                features = np.load(clip_path)
            except:
                print(f"Warning: Could not load clip {clip}, skipping")
                continue
                
            # Check feature dimensions
            if features.shape[1] != 768:
                print(f"Warning: Clip {clip} has unexpected feature dimension {features.shape[1]}, skipping")
                continue
                
            # Normalize to 4 frames if needed
            if features.shape[0] != 4:
                n_frames = features.shape[0]
                if features.shape[0] > 4:
                    indices = np.linspace(0, n_frames-1, 4, dtype=int)
                    features = features[indices]
                else:
                    indices = np.linspace(0, n_frames-1, 4)
                    interpolated_features = np.zeros((4, features.shape[1]))
                    for j in range(features.shape[1]):
                        interpolated_features[:, j] = np.interp(indices, np.arange(n_frames), features[:, j])
                    features = interpolated_features
            
            video_features.append(features)
        
        if len(video_features) < 14:
            print(f"Warning: Not enough valid clips for {file_name}, found {len(video_features)}, skipping")
            continue
            
        # Stack and average across time and clips
        video_features = np.stack(video_features)  # (14, 4, 768)
        features_avg = np.mean(video_features, axis=(0, 1))  # (768,)
        
        all_features.append(features_avg)
        all_pain_classes.append(pain_class)
        all_file_names.append(file_name)
    
    # Convert to arrays
    features = np.array(all_features)
    pain_classes = np.array(all_pain_classes)
    
    return {
        'features': features,
        'pain_classes': pain_classes,
        'file_names': all_file_names
    }

def select_features_by_effect_size(features, pain_classes, n_features=5, save_top_n=100):
    """
    Select features based on effect size (Cohen's d) for binary classification.
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        pain_classes: numpy array of binary pain classes
        n_features: number of features to select for model training
        save_top_n: number of top features to save to CSV
    
    Returns:
        selected_features: numpy array of selected features
        selected_indices: indices of selected features
        effect_sizes: effect sizes of all features
    """
    n_samples, n_features_total = features.shape
    effect_sizes = np.zeros(n_features_total)
    
    # Calculate Cohen's d for each feature
    for i in range(n_features_total):
        # Split features by class
        feature_values = features[:, i]
        class_0_values = feature_values[pain_classes == 0]
        class_1_values = feature_values[pain_classes == 1]
        
        # Calculate means and standard deviations
        mean_0 = np.mean(class_0_values)
        mean_1 = np.mean(class_1_values)
        std_0 = np.std(class_0_values, ddof=1)  # ddof=1 for sample standard deviation
        std_1 = np.std(class_1_values, ddof=1)
        
        # Calculate pooled standard deviation
        n_0 = len(class_0_values)
        n_1 = len(class_1_values)
        pooled_std = np.sqrt(((n_0 - 1) * std_0**2 + (n_1 - 1) * std_1**2) / (n_0 + n_1 - 2))
        
        # Calculate Cohen's d
        if pooled_std == 0:
            effect_sizes[i] = 0
        else:
            effect_sizes[i] = abs(mean_1 - mean_0) / pooled_std
    
    # Get indices of top N features by effect size
    top_n_indices = np.argsort(effect_sizes)[-save_top_n:]
    
    # Create DataFrame with feature indices and effect sizes
    effect_size_df = pd.DataFrame({
        'feature_index': top_n_indices,
        'effect_size': effect_sizes[top_n_indices]
    })
    effect_size_df = effect_size_df.sort_values('effect_size', ascending=False)
    
    # Save to CSV
    output_dir = Path('Syracuse/pain_level_prediction/classification_results')
    output_dir.mkdir(exist_ok=True, parents=True)
    effect_size_df.to_csv(os.path.join(output_dir, 'top_100_effect_sizes.csv'), index=False)
    print(f"\nSaved top {save_top_n} effect sizes to {os.path.join(output_dir, 'top_100_effect_sizes.csv')}")
    
    # Select top features for model training
    selected_indices = top_n_indices[-n_features:]
    selected_features = features[:, selected_indices]
    
    # Print effect sizes of selected features
    print(f"\nSelected features by effect size (Cohen's d):")
    for i, idx in enumerate(selected_indices):
        print(f"Feature {idx}: {effect_sizes[idx]:.3f}")
    
    return selected_features, selected_indices, effect_sizes

def reduce_dimensions(features, n_components=5):
    """
    Reduce dimensions using PCA.
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        n_components: number of PCA components to keep
    
    Returns:
        pca_features: numpy array of reduced features
        pca: fitted PCA object
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    
    print(f"Reduced from {features.shape[1]} features to {pca.n_components_} principal components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.3f}")
    
    return pca_features, pca

def train_and_evaluate_model(features, pain_classes, n_folds=3, random_state=42):
    """
    Train and evaluate a model using k-fold cross-validation.
    
    Args:
        features: numpy array of selected features
        pain_classes: numpy array of binary pain classes
        n_folds: number of folds for cross-validation
        random_state: random seed for reproducibility
    
    Returns:
        Dictionary containing model performance metrics and fold-specific results
    """
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Initialize model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    
    # Perform k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Store results for each fold
    fold_results = []
    
    # Metrics across all folds
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_aucs = []
    
    # Train and evaluate on each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(features_scaled)):
        X_train, X_val = features_scaled[train_idx], features_scaled[val_idx]
        y_train, y_val = pain_classes[train_idx], pain_classes[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        # Store metrics
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        all_aucs.append(auc_score)
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'train_indices': train_idx,
            'val_indices': val_idx,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score
        })
        
        print(f"\nFold {fold + 1} Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"AUC: {auc_score:.3f}")
    
    # Calculate mean metrics
    mean_accuracy = np.mean(all_accuracies)
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_f1 = np.mean(all_f1_scores)
    mean_auc = np.mean(all_aucs)
    
    # Calculate standard deviations
    std_accuracy = np.std(all_accuracies)
    std_precision = np.std(all_precisions)
    std_recall = np.std(all_recalls)
    std_f1 = np.std(all_f1_scores)
    std_auc = np.std(all_aucs)
    
    print(f"\nCross-Validation Results ({n_folds}-fold):")
    print(f"Mean Accuracy: {mean_accuracy:.3f} (±{std_accuracy:.3f})")
    print(f"Mean Precision: {mean_precision:.3f} (±{std_precision:.3f})")
    print(f"Mean Recall: {mean_recall:.3f} (±{std_recall:.3f})")
    print(f"Mean F1 Score: {mean_f1:.3f} (±{std_f1:.3f})")
    print(f"Mean AUC: {mean_auc:.3f} (±{std_auc:.3f})")
    
    # Train final model on all data
    final_model = LogisticRegression(random_state=random_state, max_iter=1000)
    final_model.fit(features_scaled, pain_classes)
    
    # Print model coefficients
    print("\nModel Coefficients:")
    print(f"Intercept: {final_model.intercept_[0]:.3f}")
    print("Feature coefficients:")
    for i, coef in enumerate(final_model.coef_[0]):
        print(f"Feature {i}: {coef:.3f}")
    
    return {
        'model': final_model,
        'scaler': scaler,
        'fold_results': fold_results,
        'mean_accuracy': mean_accuracy,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'mean_auc': mean_auc,
        'std_accuracy': std_accuracy,
        'std_precision': std_precision,
        'std_recall': std_recall,
        'std_f1': std_f1,
        'std_auc': std_auc
    }

def plot_results(results, output_dir, features, pain_classes):
    """
    Generate plots for model performance visualization.
    
    Args:
        results: Dictionary containing model results
        output_dir: Directory to save plots
        features: Original feature array
        pain_classes: Original pain class array
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot metrics across folds
    plt.figure(figsize=(10, 6))
    folds = [r['fold'] for r in results['fold_results']]
    accuracies = [r['accuracy'] for r in results['fold_results']]
    precisions = [r['precision'] for r in results['fold_results']]
    recalls = [r['recall'] for r in results['fold_results']]
    f1_scores = [r['f1_score'] for r in results['fold_results']]
    aucs = [r['auc'] for r in results['fold_results']]
    
    plt.plot(folds, accuracies, 'o-', label='Accuracy')
    plt.plot(folds, precisions, 'o-', label='Precision')
    plt.plot(folds, recalls, 'o-', label='Recall')
    plt.plot(folds, f1_scores, 'o-', label='F1 Score')
    plt.plot(folds, aucs, 'o-', label='AUC')
    
    plt.axhline(y=results['mean_accuracy'], color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=results['mean_precision'], color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=results['mean_recall'], color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=results['mean_f1'], color='c', linestyle='--', alpha=0.5)
    plt.axhline(y=results['mean_auc'], color='m', linestyle='--', alpha=0.5)
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Model Performance Across Folds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fold_performance.png'))
    plt.close()
    
    # Plot confusion matrix for the last fold
    last_fold = results['fold_results'][-1]
    val_indices = last_fold['val_indices']
    
    # Get predictions for the last fold
    model = results['model']
    scaler = results['scaler']
    X_val = scaler.transform(features[val_indices])
    y_val = pain_classes[val_indices]
    y_pred = model.predict(X_val)
    
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Last Fold)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Plot ROC curve for the last fold
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Last Fold)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and evaluate pain level classification models')
    parser.add_argument('--n_features', type=int, default=5,
                        help='Number of features to select by effect size')
    parser.add_argument('--pain_threshold', type=float, default=5.0,
                        help='Threshold for binary pain classification')
    parser.add_argument('--n_folds', type=int, default=3,
                        help='Number of folds for cross-validation')
    parser.add_argument('--save_top_n', type=int, default=100,
                        help='Number of top features to save by effect size')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path('Syracuse/pain_level_prediction/classification_results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract features and pain levels from all videos
    print("Extracting features and pain levels...")
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    data = extract_features_and_pain_levels(meta_path, feature_dir, pain_threshold=args.pain_threshold)
    
    features = data['features']
    pain_classes = data['pain_classes']
    
    print(f"\nDataset statistics:")
    print(f"Total samples: {len(pain_classes)}")
    print(f"Class distribution:")
    print(f"Low pain (< {args.pain_threshold}): {np.sum(pain_classes == 0)} samples")
    print(f"High pain (>= {args.pain_threshold}): {np.sum(pain_classes == 1)} samples")
    
    # Select features based on effect size
    print(f"\nSelecting {args.n_features} features based on effect size...")
    selected_features, selected_indices, effect_sizes = select_features_by_effect_size(
        features, pain_classes, n_features=args.n_features, save_top_n=args.save_top_n
    )
    
    # Train and evaluate model
    print(f"\nTraining and evaluating model with {args.n_folds}-fold cross-validation...")
    results = train_and_evaluate_model(selected_features, pain_classes, n_folds=args.n_folds)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(results, output_dir, selected_features, pain_classes)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == '__main__':
    main() 