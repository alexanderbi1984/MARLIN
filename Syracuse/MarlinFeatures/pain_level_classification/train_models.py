import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_classification_data import PainLevelClassification
from typing import Dict, List, Tuple

# Set random seed for reproducibility
np.random.seed(42)

class ModelTrainer:
    def __init__(self):
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        self.pain_clf = PainLevelClassification()
        self.X, self.y_dict = self.pain_clf.prepare_data()
        
        # Initialize models
        self.models = {
            '3_class': {
                'logistic': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='rbf', probability=True, random_state=42),
                'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            },
            '4_class': {
                'logistic': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='rbf', probability=True, random_state=42),
                'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            },
            '5_class': {
                'logistic': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='rbf', probability=True, random_state=42),
                'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            }
        }
        
        # Initialize scaler and fit on all data
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
    def train_and_evaluate(self, n_splits: int = 3):
        """
        Train and evaluate all models using k-fold cross validation.
        
        Args:
            n_splits: Number of folds for cross validation
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = {}
        
        for problem_type in ['3_class', '4_class', '5_class']:
            print(f"\n=== Training models for {problem_type} problem ===")
            y = self.y_dict[problem_type]
            
            # Initialize results for this problem type
            results[problem_type] = {model_name: {'accuracy': [], 'balanced_accuracy': [], 'f1': [], 'confusion_matrices': []} 
                                   for model_name in self.models[problem_type].keys()}
            
            # Perform k-fold cross validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_scaled)):
                print(f"\nFold {fold + 1}/{n_splits}")
                
                # Split data
                X_train, X_val = self.X_scaled[train_idx], self.X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train and evaluate each model
                for model_name, model in self.models[problem_type].items():
                    print(f"Training {model_name}...")
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_val)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_val, y_pred)
                    balanced_acc = balanced_accuracy_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    conf_matrix = confusion_matrix(y_val, y_pred)
                    
                    # Store results
                    results[problem_type][model_name]['accuracy'].append(accuracy)
                    results[problem_type][model_name]['balanced_accuracy'].append(balanced_acc)
                    results[problem_type][model_name]['f1'].append(f1)
                    results[problem_type][model_name]['confusion_matrices'].append(conf_matrix)
                    
                    print(f"{model_name} Results:")
                    print(f"Accuracy: {accuracy:.4f}")
                    print(f"Balanced Accuracy: {balanced_acc:.4f}")
                    print(f"F1 Score: {f1:.4f}")
        
        # Save results
        self.save_results(results)
        
    def save_results(self, results: Dict):
        """
        Save results to files and create visualizations.
        
        Args:
            results: Dictionary containing results for all models
        """
        # Create results directory if it doesn't exist
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics to CSV and create plots for each problem type
        for problem_type in ['3_class', '4_class', '5_class']:
            # Prepare DataFrame for all models' metrics
            metrics_data = []
            for model_name in results[problem_type].keys():
                model_results = results[problem_type][model_name]
                for fold in range(len(model_results['accuracy'])):
                    metrics_data.append({
                        'model': model_name,
                        'fold': fold + 1,
                        'accuracy': model_results['accuracy'][fold],
                        'balanced_accuracy': model_results['balanced_accuracy'][fold],
                        'f1': model_results['f1'][fold]
                    })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Save detailed metrics
            metrics_df.to_csv(os.path.join(results_dir, f'{problem_type}_metrics.csv'), index=False)
            
            # Plot confusion matrices for each model
            for model_name in results[problem_type].keys():
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f'{problem_type} - {model_name} Confusion Matrices')
                
                for i, conf_matrix in enumerate(results[problem_type][model_name]['confusion_matrices']):
                    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[i])
                    axes[i].set_title(f'Fold {i+1}')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('True')
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'{problem_type}_{model_name}_confusion_matrices.png'))
                plt.close()
            
            # Create boxplot comparing all models
            plt.figure(figsize=(12, 6))
            metrics_melted = pd.melt(metrics_df, id_vars=['model', 'fold'], 
                                   value_vars=['accuracy', 'balanced_accuracy', 'f1'],
                                   var_name='metric', value_name='score')
            
            # Create boxplot
            sns.boxplot(data=metrics_melted, x='model', y='score', hue='metric')
            
            # Add random guess baselines
            n_classes = int(problem_type.split('_')[0])  # Get number of classes from problem_type
            random_guess = 1.0 / n_classes  # Random guess performance
            
            # Add horizontal lines for random guess
            plt.axhline(y=random_guess, color='r', linestyle='--', label=f'Random Guess (1/{n_classes})')
            
            plt.title(f'{problem_type} - Model Performance Comparison')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{problem_type}_model_comparison.png'))
            plt.close()
            
            # Print final results summary
            self._print_final_results(problem_type, metrics_df)
    
    def _print_final_results(self, problem_type: str, metrics_df: pd.DataFrame):
        """
        Print final results summary for a specific problem type.
        
        Args:
            problem_type: Type of classification problem (3_class, 4_class, or 5_class)
            metrics_df: DataFrame containing all metrics for the problem type
        """
        print(f"\n=== Final Results for {problem_type} ===")
        print("-" * 80)
        
        # Calculate mean and std for each model and metric
        summary = metrics_df.groupby('model').agg({
            'accuracy': ['mean', 'std'],
            'balanced_accuracy': ['mean', 'std'],
            'f1': ['mean', 'std']
        }).round(4)
        
        # Print results in a formatted way
        for model in summary.index:
            print(f"\n{model.upper()}:")
            for metric in ['accuracy', 'balanced_accuracy', 'f1']:
                mean_val = summary.loc[model, (metric, 'mean')]
                std_val = summary.loc[model, (metric, 'std')]
                print(f"{metric.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    # Initialize and train models
    trainer = ModelTrainer()
    trainer.train_and_evaluate() 