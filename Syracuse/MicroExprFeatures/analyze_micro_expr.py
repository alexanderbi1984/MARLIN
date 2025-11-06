import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
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
            # Use only Logistic Regression
            model = LogisticRegression(class_weight='balanced', random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            metrics_dict = evaluate_binary_classification(y_val, y_pred)
            print(f"Logistic Regression metrics:", metrics_dict)
            metrics.append(metrics_dict)
        else:
            # Use Linear, Ridge, and Lasso Regression
            models = {
                'Linear': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=1.0)
            }
            
            fold_metrics = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                metrics_dict = evaluate_regression(y_val, y_pred)
                print(f"{name} metrics:", metrics_dict)
                fold_metrics[name] = metrics_dict
            
            metrics.append(fold_metrics)
    
    if task_type == 'binary':
        # Calculate mean and std of metrics for binary classification
        mean_metrics = {}
        std_metrics = {}
        for metric in metrics[0].keys():
            values = [m[metric] for m in metrics]
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
        return mean_metrics, std_metrics
    else:
        # Calculate mean and std of metrics for each regression model
        mean_metrics = {}
        std_metrics = {}
        for model_name in models.keys():
            model_metrics = {}
            for metric in metrics[0][model_name].keys():
                values = [m[model_name][metric] for m in metrics]
                model_metrics[metric] = np.mean(values)
                model_metrics[f"{metric}_std"] = np.std(values)
            mean_metrics[model_name] = model_metrics
        return mean_metrics, None  # No std_metrics needed for regression as they're included in mean_metrics

def plot_results(results, save_dir='results'):
    """Plot results for all tasks."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot binary classification metrics
    binary_tasks = ['binary_classification']
    for task in binary_tasks:
        if task in results:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            values = [results[task]['mean'][m] for m in metrics]
            errors = [results[task]['std'][m] for m in metrics]
            
            plt.figure(figsize=(10, 6))
            plt.bar(metrics, values, yerr=errors, capsize=5)
            plt.title('Binary Classification Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{task}_metrics.png'))
            plt.close()
    
    # Plot regression metrics for each model
    regression_tasks = ['pain_level_prediction', 'pain_reduction_prediction', 'percentage_reduction_prediction']
    for task in regression_tasks:
        if task in results:
            for model_name, metrics in results[task]['mean'].items():
                plt.figure(figsize=(10, 6))
                model_metrics = {k: v for k, v in metrics.items() if not k.endswith('_std')}
                model_errors = {k.replace('_std', ''): v for k, v in metrics.items() if k.endswith('_std')}
                
                plt.bar(model_metrics.keys(), model_metrics.values(), 
                       yerr=list(model_errors.values()), capsize=5)
                plt.title(f'{model_name} Regression Metrics for {task.replace("_", " ").title()}')
                plt.ylabel('Score')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{task}_{model_name.lower()}_metrics.png'))
                plt.close()

def analyze_micro_expr_features():
    """Analyze micro-expression features for pain prediction."""
    # Load data
    data = load_data()
    
    # Print data statistics
    print("\n=== Data Statistics ===\n")
    print("Dataset Size:")
    print(f"Total samples: {len(data)}")
    print(f"Number of features: {len(data.columns)}")
    print(f"Feature names: {list(data.columns)}")
    
    # Print feature statistics
    print("\nFeature Statistics:")
    print("Pre-treatment features:")
    for col in data.columns:
        if col.startswith('logit_'):
            values = data[col].dropna()
            print(f"{col:<12}: mean = {values.mean():6.3f}, std = {values.std():6.3f}, range = [{values.min():6.3f}, {values.max():6.3f}]")
    
    print("\nPost-treatment features:")
    for col in data.columns:
        if col.startswith('post_logit_'):
            values = data[col].dropna()
            print(f"{col:<12}: mean = {values.mean():6.3f}, std = {values.std():6.3f}, range = [{values.min():6.3f}, {values.max():6.3f}]")
    
    # Print outcome statistics
    print("\nOutcome Statistics:")
    print("Pain reduction:")
    changes = data['pain_reduction'].dropna()
    print(f"- Mean: {changes.mean():.2f} ± {changes.std():.2f}")
    print(f"- Range: [{changes.min():.2f}, {changes.max():.2f}]")
    print("- Distribution of changes:")
    for threshold in [0, 2, 4, 6, 8, 10]:
        count = (changes >= threshold).sum()
        print(f"  ≥{threshold}: {count} samples ({count/len(changes)*100:.1f}%)")
    
    print("\nInitial pain levels:")
    initial_pain = data['initial_pain'].dropna()
    print(f"- Mean: {initial_pain.mean():.2f} ± {initial_pain.std():.2f}")
    print(f"- Range: [{initial_pain.min():.2f}, {initial_pain.max():.2f}]")
    print("- Distribution:")
    for i in range(0, 12, 2):
        count = ((initial_pain >= i) & (initial_pain < i+2)).sum()
        print(f"  {i}-{i+2}: {count} samples ({count/len(initial_pain)*100:.1f}%)")
    
    # Prepare features for binary classification
    binary_features = data[['logit_Anger', 'logit_Contempt', 'logit_Disgust', 'logit_Fear',
                           'logit_Happiness', 'logit_Neutral', 'logit_Sadness', 'logit_Surprise']].values
    binary_target = (data['pain_reduction'] >= 4).astype(int).values
    
    # Print binary classification target distribution
    positive_cases = (binary_target == 1).sum()
    print(f"\nBinary Classification Target:")
    print(f"Positive cases (pain reduction ≥4): {positive_cases} ({positive_cases/len(binary_target)*100:.1f}%)")
    print(f"Negative cases (pain reduction <4): {len(binary_target)-positive_cases} ({(len(binary_target)-positive_cases)/len(binary_target)*100:.1f}%)")
    
    # Prepare features for pain level prediction using all videos with valid pain levels
    pain_level_features = data[['logit_Anger', 'logit_Contempt', 'logit_Disgust', 'logit_Fear',
                               'logit_Happiness', 'logit_Neutral', 'logit_Sadness', 'logit_Surprise']].values
    pain_level_target = data['initial_pain'].values
    
    # Prepare features for pain reduction prediction using paired data
    valid_pairs = data.dropna(subset=['pain_reduction'])
    reduction_features = valid_pairs[['logit_Anger', 'logit_Contempt', 'logit_Disgust', 'logit_Fear',
                                    'logit_Happiness', 'logit_Neutral', 'logit_Sadness', 'logit_Surprise']].values
    reduction_target = valid_pairs['pain_reduction'].values
    
    # Prepare features for percentage reduction prediction using paired data
    percentage_reduction_target = valid_pairs['percentage_reduction'].values
    
    # Train and evaluate models
    results = {}
    
    # Task 1: Binary Classification
    print("\nTask 1: Binary Classification")
    mean_metrics, std_metrics = train_and_evaluate(binary_features, binary_target, task_type='binary')
    results['binary_classification'] = {'mean': mean_metrics, 'std': std_metrics}
    
    # Task 2: Pain Level Prediction
    print("\nTask 2: Pain Level Prediction")
    mean_metrics, std_metrics = train_and_evaluate(pain_level_features, pain_level_target, task_type='regression')
    results['pain_level_prediction'] = {'mean': mean_metrics, 'std': std_metrics}
    
    # Task 3: Absolute Pain Reduction Prediction
    print("\nTask 3: Absolute Pain Reduction Prediction")
    mean_metrics, std_metrics = train_and_evaluate(reduction_features, reduction_target, task_type='regression')
    results['pain_reduction_prediction'] = {'mean': mean_metrics, 'std': std_metrics}
    
    # Task 4: Percentage Pain Reduction Prediction
    print("\nTask 4: Percentage Pain Reduction Prediction")
    mean_metrics, std_metrics = train_and_evaluate(reduction_features, percentage_reduction_target, task_type='regression')
    results['percentage_reduction_prediction'] = {'mean': mean_metrics, 'std': std_metrics}
    
    # Plot results
    plot_results(results)
    
    return results

def main():
    # Initialize dataset
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    micro_expr_dir = '/Users/hd927/Documents/syracuse_pain_research/MicroExprFeatures'
    
    dataset = MicroExprDataset(meta_path, feature_dir, micro_expr_dir)
    
    # Get all videos with valid pain levels
    all_features = dataset.get_all_micro_expr_features()  # This should return features for all videos
    all_pain_levels = dataset.get_all_pain_levels()  # This should return pain levels for all videos
    
    # Get paired data for other tasks
    pre_features, post_features, changes = dataset.get_micro_expr_features()
    feature_names = dataset.get_feature_names()
    
    # Calculate feature differences for paired data
    feature_diffs = post_features - pre_features
    
    # Prepare targets
    y_binary = (changes >= 4).astype(int)
    y_pain_reduction = changes
    y_percentage_reduction = (changes / dataset.get_pain_levels()) * 100
    
    # Print detailed data statistics
    print_data_statistics(pre_features, post_features, changes, y_binary, all_pain_levels, feature_names)
    
    # Store results
    results = {}
    
    # Task 1: Binary Classification (predict pain reduction ≥4)
    print("\nTask 1: Binary Classification")
    mean_metrics, std_metrics = train_and_evaluate(feature_diffs, y_binary, task_type='binary')
    results['binary_classification'] = {'mean': mean_metrics, 'std': std_metrics}
    print("Final Results:", mean_metrics)
    
    # Task 2: Pain Level Prediction (using all videos)
    print("\nTask 2: Pain Level Prediction")
    mean_metrics, std_metrics = train_and_evaluate(all_features, all_pain_levels, task_type='regression')
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