#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run pain classification study using MARLIN features.
This script performs the same study as clip_level_pain_classification.py
but uses the new MarlinPainClassifier class with careful fold setup.

Key features:
- Command-line interface for running pain classification experiments
- Support for multiple classification models and cross-validation strategies
- Customizable class setups (3, 4, or 5 pain classes)
- Augmentation-aware training with controlled augmentation percentages
- Class-weighted sampling to address class imbalance in training data
- Comprehensive results output with performance metrics and visualizations

Usage example:
    python run_pain_classification.py 
        --meta_path "/path/to/meta_with_outcomes_and_classes.xlsx" 
        --marlin_base_dir "/path/to/marlin_base_dir" 
        --output_dir "results" 
        --n_classes 3 
        --fold_strategy "aug_aware"
        --aug_per_video 2

Parameters:
    --meta_path: Path to the metadata Excel file with outcome labels
    --marlin_base_dir: Directory containing pre-extracted MARLIN features
    --output_dir: Directory to save results (subfolders will be created)
    --model_name: Name of the MARLIN model used for feature extraction
    --n_classes: Number of classes for pain classification (3, 4, or 5)
    --n_splits: Number of cross-validation folds
    --fold_strategy: Strategy for creating folds (stratified, video_based, part_based, aug_aware)
    --models: Specific models to evaluate (default: all available models)
    --seed: Random seed for reproducibility
    --aug_per_video: Controls percentage of augmented clips used (1=25%, 2=50%, 3=75%, 4=100%)
                    Class-weighted sampling is applied to prioritize minority classes.

TODO:
- Consider additional evaluation metrics for multi-class imbalanced data (e.g., macro-F1)
- Explore other augmentation techniques to further improve minority class performance
- Implement model selection based on balanced accuracy rather than standard accuracy
"""

import os
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict

from pain_classifier import MarlinPainClassifier

def convert_numpy_types(obj):
    """
    Convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Any Python object that might contain NumPy types
        
    Returns:
        Object with NumPy types converted to native Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif hasattr(obj, 'item'):  # Generic handler for NumPy scalars
        return obj.item()
    else:
        return obj

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run pain classification study using MARLIN features')
    parser.add_argument('--meta_path', type=str, required=True,
                        help='Path to the meta_with_outcomes.xlsx file')
    # parser.add_argument('--feature_dir', type=str, required=True,
    #                     help='Directory containing the feature files')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--marlin_base_dir', type=str, required=True,
                        help='Base directory for MarlinFeatures class (for loading clips)')
    parser.add_argument('--model_name', type=str, default='marlin_vit_small',
                        help='Name of the MARLIN model to use')
    parser.add_argument('--n_classes', type=int, default=3, choices=[3, 4, 5],
                        help='Number of classes for classification (3, 4, or 5)')
    parser.add_argument('--n_splits', type=int, default=3,
                        help='Number of cross-validation splits')
    parser.add_argument('--fold_strategy', type=str, default='aug_aware',
                        choices=['stratified', 'video_based', 'part_based', 'aug_aware'],
                        help='Strategy for creating folds')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to evaluate (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--aug_per_video', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Proportion of augmented clips to use: 1=25%%, 2=50%%, 3=75%%, 4=100%% (default: 1)')
    return parser.parse_args()

def create_video_based_folds(video_names, labels, n_splits, random_state=42):
    """
    Create folds based on video names to ensure videos from the same source
    are not split across train and test sets.
    
    Args:
        video_names: Array of video names
        labels: Array of labels
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        List of (train_idx, test_idx) tuples
    """
    # Get unique video names
    unique_videos = np.unique(video_names)
    
    # Create stratified folds for unique videos
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Get the label for each unique video (use the most common label)
    video_labels = []
    for video in unique_videos:
        video_mask = video_names == video
        video_label = np.bincount(labels[video_mask]).argmax()
        video_labels.append(video_label)
    
    # Create folds for unique videos
    folds = []
    for train_videos_idx, test_videos_idx in skf.split(unique_videos, video_labels):
        train_videos = unique_videos[train_videos_idx]
        test_videos = unique_videos[test_videos_idx]
        
        # Get indices for all clips from these videos
        train_idx = np.array([i for i, v in enumerate(video_names) if v in train_videos])
        test_idx = np.array([i for i, v in enumerate(video_names) if v in test_videos])
        
        folds.append((train_idx, test_idx))
    
    return folds

def create_part_based_folds(video_names, labels, n_splits, random_state=42):
    """
    Create folds based on video parts to ensure parts from the same video
    are not split across train and test sets.
    
    Args:
        video_names: Array of video names
        labels: Array of labels
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        List of (train_idx, test_idx) tuples
    """
    # Extract part information from video names
    # Assuming format like "video_name_part_X.mp4"
    parts = []
    for name in video_names:
        if '_part_' in name:
            parts.append(name.split('_part_')[0])
        else:
            parts.append(name)
    
    parts = np.array(parts)
    
    # Get unique parts
    unique_parts = np.unique(parts)
    
    # Create stratified folds for unique parts
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Get the label for each unique part (use the most common label)
    part_labels = []
    for part in unique_parts:
        part_mask = parts == part
        part_label = np.bincount(labels[part_mask]).argmax()
        part_labels.append(part_label)
    
    # Create folds for unique parts
    folds = []
    for train_parts_idx, test_parts_idx in skf.split(unique_parts, part_labels):
        train_parts = unique_parts[train_parts_idx]
        test_parts = unique_parts[test_parts_idx]
        
        # Get indices for all clips from these parts
        train_idx = np.array([i for i, p in enumerate(parts) if p in train_parts])
        test_idx = np.array([i for i, p in enumerate(parts) if p in test_parts])
        
        folds.append((train_idx, test_idx))
    
    return folds

def plot_results(results, output_dir, model_name, n_classes):
    """Plot classification results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x=range(1, len(results['accuracy']) + 1), y=results['accuracy'])
    plt.axhline(y=results['mean_accuracy'], color='r', linestyle='--', 
                label=f'Mean: {results["mean_accuracy"]:.3f} ± {results["std_accuracy"]:.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - {n_classes}-class Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_{n_classes}class_accuracy.png'))
    plt.close()
    
    # Plot AUC
    plt.figure(figsize=(10, 6))
    sns.barplot(x=range(1, len(results['auc']) + 1), y=results['auc'])
    plt.axhline(y=results['mean_auc'], color='r', linestyle='--', 
                label=f'Mean: {results["mean_auc"]:.3f} ± {results["std_auc"]:.3f}')
    plt.xlabel('Fold')
    plt.ylabel('AUC')
    plt.title(f'{model_name} - {n_classes}-class AUC')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_{n_classes}class_auc.png'))
    plt.close()
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results['true_labels'], results['predictions'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - {n_classes}-class Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f'{model_name}_{n_classes}class_confusion_matrix.png'))
    plt.close()

def main():
    """Main function to run the pain classification study."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Create output directory with subfolder for specific run
    subfolder_name = f"{args.model_name}_{args.n_classes}class_{args.fold_strategy}_aug{args.aug_per_video}"
    output_dir = os.path.join(args.output_dir, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Write README file with experiment details
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Pain Classification Experiment\n\n")
        f.write(f"## Parameters\n")
        f.write(f"- Model: {args.model_name}\n")
        f.write(f"- Number of classes: {args.n_classes}\n")
        f.write(f"- Number of splits: {args.n_splits}\n")
        f.write(f"- Fold strategy: {args.fold_strategy}\n")
        f.write(f"- Augmentation percentage: {args.aug_per_video * 25}% (aug_per_video={args.aug_per_video})\n")
        f.write(f"- Random seed: {args.seed}\n\n")
        
        f.write(f"## Fold Strategy Description\n")
        if args.fold_strategy == 'aug_aware':
            f.write("Augmentation-aware fold strategy:\n")
            f.write("- Folds created using original videos only\n")
            f.write("- Videos with the same ID stay together in a fold\n")
            f.write(f"- When using folds for training, {args.aug_per_video * 25}% of augmented clips are included\n")
            f.write("- Class-weighted sampling is used for augmented clips (more samples from minority classes)\n")
            f.write("- Only original clips are used for testing\n")
        elif args.fold_strategy == 'video_based':
            f.write("Video-based fold strategy:\n")
            f.write("- Ensures videos from the same source are not split across train and test sets\n")
            f.write("- Stratification is done at the video level, using the most common label in each video\n")
        elif args.fold_strategy == 'part_based':
            f.write("Part-based fold strategy:\n")
            f.write("- Ensures parts from the same video are not split across train and test sets\n")
            f.write("- Videos are grouped by their base name before '_part_X'\n")
            f.write("- Stratification is done at the part level, using the most common label in each part\n")
        else:
            f.write("Stratified fold strategy:\n")
            f.write("- Standard stratified K-fold cross-validation\n")
            f.write("- Folds are created to maintain the same class distribution across all splits\n")
    
    # Initialize classifier
    classifier = MarlinPainClassifier(
        meta_path=args.meta_path,
        marlin_base_dir=args.marlin_base_dir,
        model_name=args.model_name
    )
    
    # Load data
    classifier.load_data(include_augmented=True)
    
    # Print fold strategy information
    if args.fold_strategy == 'aug_aware':
        print("\nUsing augmentation-aware fold strategy:")
        print("- Folds created using original videos only")
        print("- Videos with the same ID stay together in a fold")
        print(f"- When using folds for training, {args.aug_per_video * 25}% of augmented clips are included")
        print("- Class-weighted sampling is used for augmented clips (more samples from minority classes)")
        print("- Only original clips are used for testing\n")
    
    # Select models to evaluate
    if args.models:
        models_to_evaluate = {name: model for name, model in classifier.models.items() 
                             if name in args.models}
    else:
        models_to_evaluate = classifier.models
    
    # Run classification for each model
    all_results = {}
    for model_name, model in models_to_evaluate.items():
        print(f"\nEvaluating {model_name} for {args.n_classes}-class classification...")
        
        # Train model with custom fold setup
        results = classifier.train_model(
            model_name=model_name,
            n_classes=args.n_classes,
            n_splits=args.n_splits,
            fold_strategy=args.fold_strategy,
            random_state=args.seed,
            aug_per_video=args.aug_per_video
        )
        
        # Save results
        all_results[model_name] = results
        classifier.save_results(results, os.path.join(output_dir, f'{model_name}_{args.n_classes}class_results.json'))
        
        # Plot results
        plot_results(results, output_dir, model_name, args.n_classes)
        
        # Print summary
        print(f"Mean accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        print(f"Mean AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
    
    # Save all results
    with open(os.path.join(output_dir, f'all_results_{args.n_classes}class.json'), 'w') as f:
        # Convert all NumPy types to native Python types for JSON serialization
        json_safe_results = convert_numpy_types(all_results)
        json.dump(json_safe_results, f, indent=4)
    
    # Create summary table
    summary = pd.DataFrame({
        'Model': list(all_results.keys()),
        'Accuracy': [results['mean_accuracy'] for results in all_results.values()],
        'Accuracy Std': [results['std_accuracy'] for results in all_results.values()],
        'AUC': [results['mean_auc'] for results in all_results.values()],
        'AUC Std': [results['std_auc'] for results in all_results.values()]
    })
    summary = summary.sort_values('Accuracy', ascending=False)
    summary.to_csv(os.path.join(output_dir, f'summary_{args.n_classes}class.csv'), index=False)
    
    # Plot comparison of all models
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Accuracy', data=summary)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Comparison of Models - {args.n_classes}-class Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'model_comparison_{args.n_classes}class.png'))
    plt.close()
    
    print(f"\nResults saved to {output_dir}")
    print("\nSummary of results:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main() 