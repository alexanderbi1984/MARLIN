import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MarlinFeatures.syracuse_dataset import SyracuseDataset

class AUPainLevelClassifier(SyracuseDataset):
    """
    Class for pain level classification using AU features.
    Supports 3-class and 5-class classification.
    """
    def __init__(self, meta_path: str, feature_dir: str, au_features_dir: str):
        """Initialize the classifier with AU features and pain data."""
        super().__init__(meta_path, feature_dir)
        self.au_features_dir = au_features_dir
        self.au_features = self._load_au_features()
        self.scaler = StandardScaler()
        
        # Define class boundaries
        # 3-Class Problem: 0-2 (Low), 3-5 (Medium), 6-10 (High)
        self.class_3_cutoffs = [2.0, 5.0]
        
        # 5-Class Problem: 0-1, 2-3, 4-5, 6-7, 8-10
        self.class_5_cutoffs = [1.0, 3.0, 5.0, 7.0]
        
    def _load_au_features(self) -> dict:
        """Load all processed AU feature files."""
        features = {}
        for file in os.listdir(self.au_features_dir):
            if file.startswith('processed_') and file.endswith('.csv'):
                # Extract IMG_xxxx from filename (e.g., processed_IMG_0003.csv -> IMG_0003)
                video_id = file.split('_')[1] + '_' + file.split('_')[2].split('.')[0]
                df = pd.read_csv(os.path.join(self.au_features_dir, file))
                features[video_id] = df
        return features
    
    def map_score_to_class(self, pain_level: float, cutoffs: List[float]) -> int:
        """
        Map a pain level score to a class based on cutoffs.
        
        Args:
            pain_level: The pain level value (0-10)
            cutoffs: List of upper boundaries for classes
            
        Returns:
            Class label (0 to len(cutoffs))
        """
        for i, cutoff in enumerate(cutoffs):
            if pain_level <= cutoff:
                return i
        return len(cutoffs)
    
    def prepare_features(self, use_pspi: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and class labels for pain level classification.
        
        Args:
            use_pspi: If True, use PSPI scores instead of individual AU features
            
        Returns:
            Tuple containing:
            - features: numpy array of shape (n_samples, n_features)
            - class_3: numpy array of 3-class labels
            - class_5: numpy array of 5-class labels
            - video_ids: list of video IDs
        """
        features_list = []
        pain_levels = []
        video_ids = []
        
        # Select features to use
        if use_pspi:
            print("Using PSPI score as feature")
        else:
            # Top 5 features based on effect size for AU features
            self.selected_features = ['AU12_r', 'AU07_r', 'AU05_r', 'AU01_r', 'AU02_r']
            print(f"Using top 5 AU features: {self.selected_features}")
        
        # Process each video in the meta data with valid pain levels
        valid_videos = 0
        for idx, row in self.meta_df.iterrows():
            if pd.notna(row['pain_level']):
                video_id = row['file_name'].split('.')[0]  # e.g., IMG_0003
                
                if video_id not in self.au_features:
                    continue
                
                # Get AU features for this video
                au_df = self.au_features[video_id]
                
                if use_pspi:
                    # Calculate PSPI score
                    au4 = au_df['AU04_r']
                    au6 = au_df['AU06_r']
                    au7 = au_df['AU07_r']
                    au9 = au_df['AU09_r']
                    au10 = au_df['AU10_r']
                    
                    # PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10)
                    pspi = au4 + np.maximum(au6, au7) + np.maximum(au9, au10)
                    feature_vec = np.mean(pspi)
                    features_list.append([feature_vec])
                else:
                    # Use selected AU features
                    feature_vec = au_df[self.selected_features].mean().values
                    features_list.append(feature_vec)
                
                # Store pain level and video ID
                pain_level = float(row['pain_level'])
                pain_levels.append(pain_level)
                video_ids.append(video_id)
                valid_videos += 1
        
        print(f"Found {valid_videos} videos with valid pain levels and AU features")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        pain_levels = np.array(pain_levels)
        
        # Map pain levels to classes
        class_3 = np.array([self.map_score_to_class(level, self.class_3_cutoffs) for level in pain_levels])
        class_5 = np.array([self.map_score_to_class(level, self.class_5_cutoffs) for level in pain_levels])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Print class distributions
        print("\n=== 3-Class Distribution ===")
        for i in range(3):
            count = np.sum(class_3 == i)
            percentage = (count / len(class_3)) * 100
            print(f"Class {i}: {count} samples ({percentage:.1f}%)")
            
        print("\n=== 5-Class Distribution ===")
        for i in range(5):
            count = np.sum(class_5 == i)
            percentage = (count / len(class_5)) * 100
            print(f"Class {i}: {count} samples ({percentage:.1f}%)")
        
        return X_scaled, class_3, class_5, video_ids
    
    def train_models(self, X: np.ndarray, y_3class: np.ndarray, y_5class: np.ndarray, 
                    n_splits: int = 5, random_state: int = 42) -> Dict:
        """
        Train and evaluate models for 3-class and 5-class pain level classification.
        
        Args:
            X: Feature matrix
            y_3class: 3-class labels
            y_5class: 5-class labels
            n_splits: Number of folds for cross-validation
            random_state: Random seed
            
        Returns:
            Dictionary containing results
        """
        results = {
            '3-class': {},
            '5-class': {}
        }
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state)
        }
        
        # Train and evaluate for 3-class and 5-class
        for problem, y in [('3-class', y_3class), ('5-class', y_5class)]:
            print(f"\n=== {problem} Classification ===")
            
            for model_name, model in models.items():
                print(f"\nEvaluating {model_name}...")
                
                # Create pipeline with standard scaler
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),  # Apply scaling again in each fold
                    ('classifier', model)
                ])
                
                # Initialize metrics
                accuracies = []
                qwks = []  # Quadratic Weighted Kappa
                maes = []  # Mean Absolute Error
                all_y_true = []
                all_y_pred = []
                
                # Perform k-fold cross-validation
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    print(f"Fold {fold+1}/{n_splits}")
                    
                    # Split data
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = pipeline.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Store metrics
                    accuracies.append(accuracy)
                    qwks.append(qwk)
                    maes.append(mae)
                    
                    # Store predictions for confusion matrix
                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)
                    
                    print(f"  Accuracy: {accuracy:.4f}, QWK: {qwk:.4f}, MAE: {mae:.4f}")
                
                # Calculate average metrics
                avg_accuracy = np.mean(accuracies)
                avg_qwk = np.mean(qwks)
                avg_mae = np.mean(maes)
                std_accuracy = np.std(accuracies)
                std_qwk = np.std(qwks)
                std_mae = np.std(maes)
                
                # Calculate confusion matrix
                cm = confusion_matrix(all_y_true, all_y_pred)
                
                # Store results
                results[problem][model_name] = {
                    'accuracy': avg_accuracy,
                    'accuracy_std': std_accuracy,
                    'qwk': avg_qwk,
                    'qwk_std': std_qwk,
                    'mae': avg_mae,
                    'mae_std': std_mae,
                    'confusion_matrix': cm.tolist()
                }
                
                # Print average metrics
                print(f"\nAverage Metrics for {model_name} ({problem}):")
                print(f"  Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
                print(f"  QWK: {avg_qwk:.4f} (±{std_qwk:.4f})")
                print(f"  MAE: {avg_mae:.4f} (±{std_mae:.4f})")
                
                # Print confusion matrix
                print("\nConfusion Matrix:")
                print(cm)
        
        return results
    
    def plot_confusion_matrices(self, results: Dict, output_dir: str = "results"):
        """
        Plot and save confusion matrices.
        
        Args:
            results: Dictionary containing results
            output_dir: Directory to save plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for problem in results.keys():
            for model_name, metrics in results[problem].items():
                cm = np.array(metrics['confusion_matrix'])
                
                # Create figure
                plt.figure(figsize=(8, 6))
                
                # Plot confusion matrix
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=range(cm.shape[1]),
                            yticklabels=range(cm.shape[0]))
                
                plt.title(f'{problem} - {model_name} - Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                
                # Save figure
                output_file = os.path.join(output_dir, f"{problem.replace('-', '_')}_{model_name.replace(' ', '_')}_cm.png")
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()
                
                print(f"Saved confusion matrix to {output_file}")
    
    def save_results(self, results: Dict, output_dir: str = "results", filename: str = "au_pain_classification_results.json"):
        """
        Save results to a JSON file.
        
        Args:
            results: Dictionary containing results
            output_dir: Directory to save results
            filename: Output filename
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to JSON
        output_file = os.path.join(output_dir, filename)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved results to {output_file}")

def main():
    """Main function to run the AU pain level classification analysis."""
    # Set paths
    meta_path = "E:\Pain\syracus\syracus\multimodal_marlin_base\meta_with_outcomes.xlsx"
    feature_dir = "E:\Pain\syracus\syracus\multimodal_marlin_base"
    au_features_dir = "E:\Pain\syracus\syracus\AU_features"
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='AU Pain Level Classification')
    parser.add_argument('--use_pspi', action='store_true', help='Use PSPI score instead of individual AU features')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = AUPainLevelClassifier(meta_path, feature_dir, au_features_dir)
    
    # Prepare features
    X, y_3class, y_5class, video_ids = classifier.prepare_features(use_pspi=args.use_pspi)
    
    # Train and evaluate models
    results = classifier.train_models(X, y_3class, y_5class, n_splits=args.n_splits, random_state=args.random_state)
    
    # Plot confusion matrices
    output_dir = "results/au_pain_classification"
    if args.use_pspi:
        output_dir = "results/pspi_pain_classification"
    classifier.plot_confusion_matrices(results, output_dir=output_dir)
    
    # Save results
    filename = "au_pain_classification_results.json"
    if args.use_pspi:
        filename = "pspi_pain_classification_results.json"
    classifier.save_results(results, output_dir=output_dir, filename=filename)

if __name__ == "__main__":
    main() 