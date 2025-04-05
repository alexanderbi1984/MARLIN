import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Syracuse.MicroExprFeatures.micro_expr_dataset import MicroExprDataset

def print_data_statistics(pre_features, post_features, changes, y_binary, y_pain_level, feature_names):
    """Print detailed statistics about the loaded data."""
    print("\n=== Data Statistics ===")
    print("\nDataset Size:")
    print(f"Total samples: {len(y_binary)}")
    print(f"Number of features: {pre_features.shape[1]}")
    print(f"Feature names: {feature_names}")
    
    print("\nFeature Statistics:")
    print("Pre-treatment features:")
    for i, feature in enumerate(feature_names):
        values = pre_features[:, i]
        print(f"{feature:15s}: mean = {values.mean():6.3f}, std = {values.std():6.3f}, range = [{values.min():6.3f}, {values.max():6.3f}]")
    
    print("\nPost-treatment features:")
    for i, feature in enumerate(feature_names):
        values = post_features[:, i]
        print(f"{feature:15s}: mean = {values.mean():6.3f}, std = {values.std():6.3f}, range = [{values.min():6.3f}, {values.max():6.3f}]")
    
    print("\nOutcome Statistics:")
    print(f"Pain reduction:")
    print(f"- Mean: {np.mean(changes):.2f} ± {np.std(changes):.2f}")
    print(f"- Range: [{np.min(changes):.2f}, {np.max(changes):.2f}]")
    print(f"- Distribution of changes:")
    for threshold in range(0, 11, 2):
        count = np.sum(changes >= threshold)
        print(f"  ≥{threshold}: {count} samples ({count/len(changes)*100:.1f}%)")
    
    print(f"\nInitial pain levels:")
    print(f"- Mean: {np.mean(y_pain_level):.2f} ± {np.std(y_pain_level):.2f}")
    print(f"- Range: [{np.min(y_pain_level):.2f}, {np.max(y_pain_level):.2f}]")
    print(f"- Distribution:")
    for level in range(0, 11, 2):
        count = np.sum((y_pain_level >= level) & (y_pain_level < level + 2))
        print(f"  {level}-{level+2}: {count} samples ({count/len(y_pain_level)*100:.1f}%)")

    print("\nBinary Classification Target:")
    print(f"Positive cases (pain reduction ≥4): {np.sum(y_binary)} ({np.mean(y_binary)*100:.1f}%)")
    print(f"Negative cases (pain reduction <4): {np.sum(y_binary == 0)} ({np.mean(y_binary == 0)*100:.1f}%)")

def evaluate_binary_classification(y_true, y_pred):
    """Evaluate binary classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def evaluate_regression(y_true, y_pred):
    """Evaluate regression metrics."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

def train_and_evaluate(X, y, task_type='binary', n_splits=3):
    """Train and evaluate models using k-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    
    print(f"\nShape of input features: {X.shape}")
    print(f"Shape of target: {y.shape}")
    print(f"Target distribution: {np.unique(y, return_counts=True)}" if task_type == 'binary' else f"Target range: [{y.min()}, {y.max()}]")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"\nFold {fold + 1}:")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        if task_type == 'binary':
            print(f"Training distribution: {np.unique(y_train, return_counts=True)}")
            print(f"Validation distribution: {np.unique(y_val, return_counts=True)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        if task_type == 'binary':
            # Use both Logistic Regression and Random Forest
            models = {
                'lr': LogisticRegression(class_weight='balanced', random_state=42),
                'rf': RandomForestClassifier(class_weight='balanced', random_state=42)
            }
            
            fold_metrics = []
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                metrics_dict = evaluate_binary_classification(y_val, y_pred)
                print(f"{name} metrics:", metrics_dict)
                fold_metrics.append(metrics_dict)
            
            # Average metrics across models
            metrics.append({
                k: np.mean([m[k] for m in fold_metrics]) 
                for k in fold_metrics[0].keys()
            })
        else:
            # Use both Linear Regression and Random Forest
            models = {
                'lr': LinearRegression(),
                'rf': RandomForestRegressor(random_state=42)
            }
            
            fold_metrics = []
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                metrics_dict = evaluate_regression(y_val, y_pred)
                print(f"{name} metrics:", metrics_dict)
                fold_metrics.append(metrics_dict)
            
            # Average metrics across models
            metrics.append({
                k: np.mean([m[k] for m in fold_metrics])
                for k in fold_metrics[0].keys()
            })
    
    # Calculate mean and std of metrics
    mean_metrics = {}
    std_metrics = {}
    for metric in metrics[0].keys():
        values = [m[metric] for m in metrics]
        mean_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)
    
    return mean_metrics, std_metrics

def plot_results(results, save_dir='results'):
    """Plot results for all tasks."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot binary classification metrics
    binary_tasks = ['binary_classification', 'pain_reduction_classification']
    for task in binary_tasks:
        if task in results:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            values = [results[task]['mean'][m] for m in metrics]
            errors = [results[task]['std'][m] for m in metrics]
            
            plt.figure(figsize=(10, 6))
            plt.bar(metrics, values, yerr=errors, capsize=5)
            plt.title(f'{task.replace("_", " ").title()} Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{task}_metrics.png'))
            plt.close()
    
    # Plot regression metrics
    regression_tasks = ['pain_level_prediction', 'pain_reduction_prediction']
    for task in regression_tasks:
        if task in results:
            metrics = ['mse', 'rmse', 'r2']
            values = [results[task]['mean'][m] for m in metrics]
            errors = [results[task]['std'][m] for m in metrics]
            
            plt.figure(figsize=(10, 6))
            plt.bar(metrics, values, yerr=errors, capsize=5)
            plt.title(f'{task.replace("_", " ").title()} Metrics')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{task}_metrics.png'))
            plt.close()

def main():
    # Initialize dataset
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    micro_expr_dir = '/Users/hd927/Documents/syracuse_pain_research/MicroExprFeatures'
    
    dataset = MicroExprDataset(meta_path, feature_dir, micro_expr_dir)
    pre_features, post_features, changes = dataset.get_micro_expr_features()
    feature_names = dataset.get_feature_names()
    
    # Calculate feature differences
    feature_diffs = post_features - pre_features
    
    # Prepare targets
    y_binary = (changes >= 4).astype(int)
    y_pain_level = dataset.get_pain_levels()
    y_pain_reduction = changes
    y_percentage_reduction = (changes / y_pain_level) * 100  # Calculate percentage reduction
    
    # Print detailed data statistics
    print_data_statistics(pre_features, post_features, changes, y_binary, y_pain_level, feature_names)
    
    # Store results
    results = {}
    
    # Task 1: Binary Classification (predict pain reduction ≥4)
    print("\nTask 1: Binary Classification")
    mean_metrics, std_metrics = train_and_evaluate(pre_features, y_binary, task_type='binary')
    results['binary_classification'] = {'mean': mean_metrics, 'std': std_metrics}
    print("Final Results:", mean_metrics)
    
    # Task 2: Pain Level Prediction
    print("\nTask 2: Pain Level Prediction")
    mean_metrics, std_metrics = train_and_evaluate(pre_features, y_pain_level, task_type='regression')
    results['pain_level_prediction'] = {'mean': mean_metrics, 'std': std_metrics}
    print("Final Results:", mean_metrics)
    
    # Task 3: Absolute Pain Reduction Prediction
    print("\nTask 3: Absolute Pain Reduction Prediction")
    mean_metrics, std_metrics = train_and_evaluate(feature_diffs, y_pain_reduction, task_type='regression')
    results['pain_reduction_prediction'] = {'mean': mean_metrics, 'std': std_metrics}
    print("Final Results:", mean_metrics)
    
    # Task 4: Percentage Pain Reduction Prediction
    print("\nTask 4: Percentage Pain Reduction Prediction")
    mean_metrics, std_metrics = train_and_evaluate(feature_diffs, y_percentage_reduction, task_type='regression')
    results['percentage_reduction_prediction'] = {'mean': mean_metrics, 'std': std_metrics}
    print("Final Results:", mean_metrics)
    
    # Plot results
    plot_results(results)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'task': list(results.keys()),
        'metrics': [results[task]['mean'] for task in results.keys()],
        'std': [results[task]['std'] for task in results.keys()]
    })
    results_df.to_csv('micro_expr_results.csv', index=False)

if __name__ == "__main__":
    main() 