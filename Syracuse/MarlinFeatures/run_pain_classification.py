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
    --aug_per_video: Controls percentage of augmented clips used (0=None, 1=25%, 2=50%, 3=75%, 4=100%)
                    Class-weighted sampling is applied to prioritize minority classes.
    --binary_task: Specific binary classification task in format "class1,class2" (e.g., "0,4" to classify between class 0 and 4 only)
    --binary_class_set: Which class set to use for binary classification (3, 4, or 5). If not specified, it will be determined automatically.

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
    parser.add_argument('--aug_per_video', type=int, default=1, choices=[0, 1, 2, 3, 4],
                        help='Proportion of augmented clips to use: 0=None, 1=25%%, 2=50%%, 3=75%%, 4=100%% (default: 1)')
    parser.add_argument('--binary_task', type=str, default=None,
                        help='Specific binary classification task in format "class1,class2" (e.g., "0,4" to classify between class 0 and 4 only)')
    parser.add_argument('--binary_class_set', type=int, choices=[3, 4, 5], default=None,
                        help='Which class set to use for binary classification (3, 4, or 5). If not specified, it will be determined automatically.')
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
    
    # Check if this is a binary classification task
    is_binary = isinstance(n_classes, str) and n_classes.startswith('binary_')
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x=range(1, len(results['accuracy']) + 1), y=results['accuracy'])
    plt.axhline(y=results['mean_accuracy'], color='r', linestyle='--', 
                label=f'Mean: {results["mean_accuracy"]:.3f} ± {results["std_accuracy"]:.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - {n_classes} Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_{n_classes}_accuracy.png'))
    plt.close()
    
    # Plot AUC
    plt.figure(figsize=(10, 6))
    sns.barplot(x=range(1, len(results['auc']) + 1), y=results['auc'])
    plt.axhline(y=results['mean_auc'], color='r', linestyle='--', 
                label=f'Mean: {results["mean_auc"]:.3f} ± {results["std_auc"]:.3f}')
    plt.xlabel('Fold')
    plt.ylabel('AUC')
    plt.title(f'{model_name} - {n_classes} AUC')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_{n_classes}_auc.png'))
    plt.close()
    
    # Plot precision, recall, F1 for binary classification
    if is_binary and 'precision' in results:
        # Plot precision
        plt.figure(figsize=(10, 6))
        sns.barplot(x=range(1, len(results['precision']) + 1), y=results['precision'])
        plt.axhline(y=results['mean_precision'], color='r', linestyle='--', 
                    label=f'Mean: {results["mean_precision"]:.3f} ± {results["std_precision"]:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Precision')
        plt.title(f'{model_name} - {n_classes} Precision')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{model_name}_{n_classes}_precision.png'))
        plt.close()
        
        # Plot recall
        plt.figure(figsize=(10, 6))
        sns.barplot(x=range(1, len(results['recall']) + 1), y=results['recall'])
        plt.axhline(y=results['mean_recall'], color='r', linestyle='--', 
                    label=f'Mean: {results["mean_recall"]:.3f} ± {results["std_recall"]:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Recall')
        plt.title(f'{model_name} - {n_classes} Recall')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{model_name}_{n_classes}_recall.png'))
        plt.close()
        
        # Plot F1
        plt.figure(figsize=(10, 6))
        sns.barplot(x=range(1, len(results['f1_score']) + 1), y=results['f1_score'])
        plt.axhline(y=results['mean_f1'], color='r', linestyle='--', 
                    label=f'Mean: {results["mean_f1"]:.3f} ± {results["std_f1"]:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('F1 Score')
        plt.title(f'{model_name} - {n_classes} F1 Score')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{model_name}_{n_classes}_f1.png'))
        plt.close()
        
        # Plot combined metrics
        plt.figure(figsize=(12, 8))
        metrics_df = pd.DataFrame({
            'Fold': np.repeat(range(1, len(results['accuracy']) + 1), 4),
            'Metric': np.tile(['Accuracy', 'Precision', 'Recall', 'F1'], len(results['accuracy'])),
            'Value': np.concatenate([results['accuracy'], results['precision'], results['recall'], results['f1_score']])
        })
        sns.barplot(x='Fold', y='Value', hue='Metric', data=metrics_df)
        plt.title(f'{model_name} - {n_classes} Metrics Comparison')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, f'{model_name}_{n_classes}_metrics_comparison.png'))
        plt.close()
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results['true_labels'], results['predictions'])
    plt.figure(figsize=(10, 8))
    
    # Add labels for binary classification
    if is_binary and 'class_indices' in results:
        class_indices = results['class_indices']
        class_labels = [f'Class {class_indices[0]}', f'Class {class_indices[1]}']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - {n_classes} Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f'{model_name}_{n_classes}_confusion_matrix.png'))
    plt.close()

def main():
    """Main function to run the pain classification study."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Create output directory with subfolder for specific run
    if args.binary_task:
        binary_classes = args.binary_task.split(',')
        if len(binary_classes) != 2:
            raise ValueError("Binary task must specify exactly two classes separated by a comma (e.g., '0,4')")
        
        # Determine class set for folder name
        if args.binary_class_set:
            class_set = args.binary_class_set
            class_set_source = "explicit"
        else:
            max_class = max([int(c) for c in binary_classes])
            if max_class <= 2:
                class_set = 3
            elif max_class <= 3:
                class_set = 4
            else:
                class_set = 5
            class_set_source = "auto"
            
        subfolder_name = f"{args.model_name}_binary{args.binary_task}_c{class_set}_{class_set_source}_{args.fold_strategy}_aug{args.aug_per_video}"
    else:
        subfolder_name = f"{args.model_name}_{args.n_classes}class_{args.fold_strategy}_aug{args.aug_per_video}"
    output_dir = os.path.join(args.output_dir, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Write README file with experiment details
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Pain Classification Experiment\n\n")
        f.write(f"## Parameters\n")
        f.write(f"- Model: {args.model_name}\n")
        
        if args.binary_task:
            f.write(f"- Classification task: Binary classification between classes {args.binary_task.replace(',', ' vs ')}\n")
            if args.binary_class_set:
                f.write(f"- Using {args.binary_class_set}-class label set (explicitly specified)\n")
            else:
                max_class = max([int(c) for c in args.binary_task.split(',')])
                if max_class <= 2:
                    class_set = 3
                elif max_class <= 3:
                    class_set = 4
                else:
                    class_set = 5
                f.write(f"- Using {class_set}-class label set (automatically determined based on class indices)\n")
        else:
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
        if args.binary_task:
            binary_classes = [int(c) for c in args.binary_task.split(',')]
            print(f"\nEvaluating {model_name} for binary classification between classes {binary_classes[0]} and {binary_classes[1]}...")
            
            # Train model with binary task
            results = classifier.train_model_binary(
                model_name=model_name,
                class_indices=binary_classes,
                class_set=args.binary_class_set,
                n_splits=args.n_splits,
                fold_strategy=args.fold_strategy,
                random_state=args.seed,
                aug_per_video=args.aug_per_video
            )
            
            output_file_prefix = f'{model_name}_binary{args.binary_task}'
        else:
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
            
            output_file_prefix = f'{model_name}_{args.n_classes}class'
        
        # Save results
        all_results[model_name] = results
        classifier.save_results(results, os.path.join(output_dir, f'{output_file_prefix}_results.json'))
        
        # Plot results
        if args.binary_task:
            plot_results(results, output_dir, model_name, f"binary_{args.binary_task.replace(',', 'vs')}")
        else:
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
    if args.binary_task:
        summary = pd.DataFrame({
            'Model': list(all_results.keys()),
            'Accuracy': [results['mean_accuracy'] for results in all_results.values()],
            'Accuracy Std': [results['std_accuracy'] for results in all_results.values()],
            'AUC': [results['mean_auc'] for results in all_results.values()],
            'AUC Std': [results['std_auc'] for results in all_results.values()],
            'Precision': [results['mean_precision'] for results in all_results.values()],
            'Precision Std': [results['std_precision'] for results in all_results.values()],
            'Recall': [results['mean_recall'] for results in all_results.values()],
            'Recall Std': [results['std_recall'] for results in all_results.values()],
            'F1 Score': [results['mean_f1'] for results in all_results.values()],
            'F1 Score Std': [results['std_f1'] for results in all_results.values()]
        })
        # Sort by F1 score for binary classification
        summary = summary.sort_values('F1 Score', ascending=False)
        summary.to_csv(os.path.join(output_dir, f'summary_binary{args.binary_task}.csv'), index=False)
        
        # Plot comparison of models by multiple metrics
        if len(all_results) > 1:  # Only if we have multiple models
            metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score']
            plt.figure(figsize=(14, 10))
            
            # Create a comparison dataframe in long format
            comparison_data = []
            for model_name, results in all_results.items():
                for metric in metrics:
                    if metric == 'F1 Score':
                        metric_key = 'mean_f1'  # Use the same key as defined in train_model_binary
                    else:
                        metric_key = f'mean_{metric.lower()}'
                    comparison_data.append({
                        'Model': model_name,
                        'Metric': metric,
                        'Value': results[metric_key]
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Plot
            sns.barplot(x='Model', y='Value', hue='Metric', data=comparison_df)
            plt.title(f'Binary {args.binary_task.replace(",", " vs ")} Classification - Model Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'model_comparison_binary{args.binary_task}.png'))
            plt.close()
    else:
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