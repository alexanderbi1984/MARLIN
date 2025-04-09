import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV, Lasso
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import json
import pickle

# Import the SyracuseDataset class
from syracuse_dataset import SyracuseDataset

# Suppress warnings
warnings.filterwarnings('ignore')

# Create a LassoClassifier class that wraps Lasso for classification
class LassoClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, max_iter=1000, random_state=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            # For multiclass, use one-vs-rest approach
            self.models = []
            for c in self.classes_:
                model = Lasso(alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_state)
                model.fit(X, (y == c).astype(int))
                self.models.append(model)
        else:
            # For binary classification
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        if len(self.classes_) > 2:
            # For multiclass, predict the class with highest probability
            predictions = np.zeros((X.shape[0], len(self.classes_)))
            for i, model in enumerate(self.models):
                predictions[:, i] = model.predict(X)
            return self.classes_[np.argmax(predictions, axis=1)]
        else:
            # For binary classification
            return (self.model.predict(X) > 0.5).astype(int)
    
    def predict_proba(self, X):
        if len(self.classes_) > 2:
            # For multiclass, return probabilities for each class
            predictions = np.zeros((X.shape[0], len(self.classes_)))
            for i, model in enumerate(self.models):
                predictions[:, i] = model.predict(X)
            # Convert to probabilities using softmax
            exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        else:
            # For binary classification
            pred = self.model.predict(X)
            # Convert to probabilities using sigmoid
            prob = 1 / (1 + np.exp(-pred))
            return np.vstack([1 - prob, prob]).T

class ClipLevelPainClassifier:
    """
    Class for performing clip-level pain level classification using various models.
    """
    
    def __init__(self, meta_path: str, feature_dir: str):
        """
        Initialize the classifier with dataset paths.
        
        Args:
            meta_path: Path to the meta_with_outcomes.xlsx file
            feature_dir: Directory containing the feature files
        """
        self.meta_path = meta_path
        self.feature_dir = feature_dir
        self.dataset = SyracuseDataset(meta_path, feature_dir)
        self.clips = None
        self.X = None
        self.y_3 = None
        self.y_4 = None
        self.y_5 = None
        self.video_names = None
        self.models = self._initialize_models()
        self.fold_setups = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize the classification models.
        
        Returns:
            Dictionary of model names and their instances
        """
        return {
            # Linear models
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Ridge Regression': RidgeClassifier(alpha=1.0, random_state=42),
            'Lasso Classifier': LassoClassifier(alpha=1.0, max_iter=1000, random_state=42),
            'Ridge (alpha=0.1)': RidgeClassifier(alpha=0.1, random_state=42),
            'Ridge (alpha=10)': RidgeClassifier(alpha=10, random_state=42),
            
            # SVM models
            'SVM (linear)': LinearSVC(max_iter=1000, random_state=42),
            'SVM (rbf)': SVC(kernel='rbf', random_state=42),
            'SVM (poly)': SVC(kernel='poly', random_state=42),
            'SVM (sigmoid)': SVC(kernel='sigmoid', random_state=42),
            
            # Neural Network models
            'MLP (small)': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
            'MLP (medium)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'MLP (large)': MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=1000, random_state=42)
        }
    
    def load_data(self) -> None:
        """
        Load all clips and prepare data for classification.
        """
        print("Loading clip data...")
        self.clips = self.dataset.load_all_clips()
        
        # Extract features and labels
        features = []
        pain_levels = []
        class_3 = []
        class_4 = []
        class_5 = []
        video_names = []
        
        for clip in self.clips:
            # Get features (shape: 4, 768)
            features.append(clip['features'])
            
            # Get pain level and class labels
            video_meta = self.dataset.meta_df[self.dataset.meta_df['file_name'] == clip['metadata']['video_name']].iloc[0]
            pain_levels.append(video_meta['pain_level'])
            
            # Convert class labels to integers, handling NaN values
            class_3_val = video_meta['class_3']
            class_4_val = video_meta['class_4']
            class_5_val = video_meta['class_5']
            
            # Replace NaN with -1 (or another value that doesn't conflict with class labels)
            class_3.append(int(class_3_val) if pd.notna(class_3_val) else -1)
            class_4.append(int(class_4_val) if pd.notna(class_4_val) else -1)
            class_5.append(int(class_5_val) if pd.notna(class_5_val) else -1)
            
            # Store video name
            video_names.append(clip['metadata']['video_name'])
        
        # Convert to numpy arrays
        self.X = np.array(features)  # Shape: (n_clips, 4, 768)
        self.y_3 = np.array(class_3, dtype=int)
        self.y_4 = np.array(class_4, dtype=int)
        self.y_5 = np.array(class_5, dtype=int)
        self.video_names = np.array(video_names)
        
        # Reshape features for classification (flatten temporal dimension)
        n_clips = self.X.shape[0]
        self.X = self.X.reshape(n_clips, -1)  # Shape: (n_clips, 4*768)
        
        print(f"Loaded {n_clips} clips with features of shape {self.X.shape}")
        
        # Print class distributions, handling negative values (NaN replacements)
        print(f"Class distribution (3-class): {np.bincount(self.y_3[self.y_3 >= 0])}")
        print(f"Class distribution (4-class): {np.bincount(self.y_4[self.y_4 >= 0])}")
        print(f"Class distribution (5-class): {np.bincount(self.y_5[self.y_5 >= 0])}")
        
        # Print number of clips with valid class labels
        print(f"Clips with valid 3-class labels: {np.sum(self.y_3 >= 0)}/{n_clips}")
        print(f"Clips with valid 4-class labels: {np.sum(self.y_4 >= 0)}/{n_clips}")
        print(f"Clips with valid 5-class labels: {np.sum(self.y_5 >= 0)}/{n_clips}")
    
    def _create_video_stratified_folds(self, video_names, labels, n_splits=3):
        """
        Create folds that maintain class balance while ensuring clips from the same part
        (start or end) of a video stay together.
        
        Args:
            video_names: List of video names for each clip
            labels: Class labels for each clip
            n_splits: Number of folds to create
            
        Returns:
            List of tuples (train_idx, test_idx) for each fold
        """
        # Group clips by video name and part (start/end)
        video_part_groups = {}
        
        # First, get all clips for each video
        unique_videos = np.unique(video_names)
        for video in unique_videos:
            video_mask = video_names == video
            video_indices = np.where(video_mask)[0]
            video_labels = labels[video_mask]
            
            # Skip videos with no valid labels
            valid_label_mask = video_labels >= 0
            if not np.any(valid_label_mask):
                continue
                
            # Split indices into start and end parts
            n_clips = len(video_indices)
            split_point = n_clips // 2
            
            # Add start part clips
            start_indices = video_indices[:split_point]
            start_labels = video_labels[:split_point]
            valid_start_mask = start_labels >= 0
            if np.any(valid_start_mask):
                start_key = f"{video}_start"
                video_part_groups[start_key] = []
                for idx, label in zip(start_indices[valid_start_mask], 
                                    start_labels[valid_start_mask]):
                    video_part_groups[start_key].append((idx, int(label)))
            
            # Add end part clips
            end_indices = video_indices[split_point:]
            end_labels = video_labels[split_point:]
            valid_end_mask = end_labels >= 0
            if np.any(valid_end_mask):
                end_key = f"{video}_end"
                video_part_groups[end_key] = []
                for idx, label in zip(end_indices[valid_end_mask], 
                                    end_labels[valid_end_mask]):
                    video_part_groups[end_key].append((idx, int(label)))
        
        # Convert groups to arrays for stratification
        group_indices = []
        group_labels = []
        
        for video_part, clips in video_part_groups.items():
            if clips:  # Only add if there are clips in this part
                indices, labels = zip(*clips)
                group_indices.append(list(indices))
                # Use the most common label in the group for stratification
                # Ensure all labels are non-negative before using bincount
                valid_labels = np.array(labels)
                group_labels.append(np.bincount(valid_labels).argmax())
        
        # Create stratified folds
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = []
        
        # Convert group_indices and group_labels to numpy arrays
        group_labels = np.array(group_labels)
        
        # Create folds using the video-part groups
        for train_idx, test_idx in skf.split(np.arange(len(group_labels)), group_labels):
            # Convert group indices back to clip indices
            train_clip_indices = []
            test_clip_indices = []
            
            for group_idx in train_idx:
                train_clip_indices.extend(group_indices[group_idx])
                
            for group_idx in test_idx:
                test_clip_indices.extend(group_indices[group_idx])
                
            folds.append((np.array(train_clip_indices), np.array(test_clip_indices)))
            
        return folds
    
    def save_fold_setup(self, problem: str, folds: List[Tuple[np.ndarray, np.ndarray]], output_dir: str = "fold_setups") -> None:
        """
        Save the fold setup to a file for reproducibility.
        
        Args:
            problem: The classification problem (3-class, 4-class, or 5-class)
            folds: List of (train_idx, test_idx) tuples for each fold
            output_dir: Directory to save the fold setup
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a dictionary to store the fold setup
        fold_setup = {
            "problem": problem,
            "n_folds": len(folds),
            "folds": []
        }
        
        # Add each fold to the dictionary
        for i, (train_idx, test_idx) in enumerate(folds):
            # Get the video names for train and test sets
            train_videos = self.video_names[train_idx]
            test_videos = self.video_names[test_idx]
            
            # Get the class labels for train and test sets
            if problem == "3-class":
                train_labels = self.y_3[train_idx]
                test_labels = self.y_3[test_idx]
            elif problem == "4-class":
                train_labels = self.y_4[train_idx]
                test_labels = self.y_4[test_idx]
            else:  # 5-class
                train_labels = self.y_5[train_idx]
                test_labels = self.y_5[test_idx]
            
            # Filter out invalid labels
            train_valid_mask = train_labels >= 0
            test_valid_mask = test_labels >= 0
            
            train_videos = train_videos[train_valid_mask]
            test_videos = test_videos[test_valid_mask]
            train_labels = train_labels[train_valid_mask]
            test_labels = test_labels[test_valid_mask]
            
            # Create a dictionary for this fold
            fold_dict = {
                "fold": i + 1,
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "train_videos": train_videos.tolist(),
                "test_videos": test_videos.tolist(),
                "train_labels": train_labels.tolist(),
                "test_labels": test_labels.tolist(),
                "train_class_distribution": np.bincount(train_labels).tolist(),
                "test_class_distribution": np.bincount(test_labels).tolist()
            }
            
            fold_setup["folds"].append(fold_dict)
        
        # Save the fold setup to a JSON file
        output_file = os.path.join(output_dir, f"{problem.replace('-', '_')}_fold_setup.json")
        with open(output_file, 'w') as f:
            json.dump(fold_setup, f, indent=2)
        
        # Also save the fold indices as a pickle file for easy loading
        pickle_file = os.path.join(output_dir, f"{problem.replace('-', '_')}_fold_indices.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(folds, f)
        
        print(f"Saved fold setup for {problem} to {output_file}")
        print(f"Saved fold indices to {pickle_file}")
        
        # Store the fold setup in the class
        self.fold_setups[problem] = fold_setup
    
    def train_and_evaluate(self, n_splits: int = 3, random_state: int = 42, save_folds: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Train and evaluate all models using k-fold cross-validation.
        
        Args:
            n_splits: Number of folds for cross-validation
            random_state: Random seed for reproducibility
            save_folds: Whether to save the fold setup to a file
            
        Returns:
            Dictionary of results for each model and classification problem
        """
        if self.X is None:
            self.load_data()
        
        # Initialize results dictionary
        results = {
            '3-class': {},
            '4-class': {},
            '5-class': {}
        }
        
        # Define target variables and filter out invalid labels
        targets = {
            '3-class': self.y_3,
            '4-class': self.y_4,
            '5-class': self.y_5
        }
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Initialize results for this model
            for problem in results.keys():
                results[problem][model_name] = {
                    'accuracy': [],
                    'auc': []
                }
            
            # Train and evaluate for each classification problem
            for problem, y in targets.items():
                print(f"  {problem} classification")
                
                # Create video-stratified folds
                folds = self._create_video_stratified_folds(self.video_names, y)
                
                # Save the fold setup if requested
                if save_folds:
                    self.save_fold_setup(problem, folds)
                
                # Perform k-fold cross-validation
                for fold, (train_idx, test_idx) in enumerate(folds):
                    print(f"    Fold {fold+1}/{n_splits}")
                    
                    # Split data
                    X_train, X_test = self.X[train_idx], self.X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Filter out invalid labels
                    train_valid_mask = y_train >= 0
                    test_valid_mask = y_test >= 0
                    
                    X_train = X_train[train_valid_mask]
                    y_train = y_train[train_valid_mask]
                    X_test = X_test[test_valid_mask]
                    y_test = y_test[test_valid_mask]
                    
                    # Skip if no valid data
                    if len(X_train) == 0 or len(X_test) == 0:
                        print(f"      Skipping fold - no valid data")
                        continue
                    
                    # Print class distribution in this fold
                    print(f"      Train class distribution: {np.bincount(y_train)}")
                    print(f"      Test class distribution: {np.bincount(y_test)}")
                    
                    # Create pipeline with standard scaler
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', model)
                    ])
                    
                    # Train model
                    pipeline.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = pipeline.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Calculate AUC (handle multiclass)
                    try:
                        if len(np.unique(y)) > 2:
                            auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr')
                        else:
                            auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
                    except Exception as e:
                        print(f"      Warning: Could not calculate AUC: {str(e)}")
                        auc = np.nan
                    
                    # Store results
                    results[problem][model_name]['accuracy'].append(accuracy)
                    results[problem][model_name]['auc'].append(auc)
                    
                    print(f"      Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        # Calculate average metrics
        for problem in results.keys():
            for model_name in results[problem].keys():
                if results[problem][model_name]['accuracy']:
                    results[problem][model_name]['accuracy'] = np.mean(results[problem][model_name]['accuracy'])
                    results[problem][model_name]['auc'] = np.mean(results[problem][model_name]['auc'])
                else:
                    results[problem][model_name]['accuracy'] = np.nan
                    results[problem][model_name]['auc'] = np.nan
        
        return results
    
    def print_results(self, results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """
        Print the results in a formatted table.
        
        Args:
            results: Dictionary of results from train_and_evaluate
        """
        print("\n" + "="*80)
        print("CLIP-LEVEL PAIN CLASSIFICATION RESULTS")
        print("="*80)
        
        for problem in results.keys():
            print(f"\n{problem.upper()} CLASSIFICATION")
            print("-"*80)
            print(f"{'Model':<30} {'Accuracy':<15} {'AUC':<15}")
            print("-"*80)
            
            # Sort models by accuracy, handling NaN values
            sorted_models = sorted(
                results[problem].items(),
                key=lambda x: x[1]['accuracy'] if not np.isnan(x[1]['accuracy']) else -1,
                reverse=True
            )
            
            for model_name, metrics in sorted_models:
                accuracy_str = f"{metrics['accuracy']:.4f}" if not np.isnan(metrics['accuracy']) else "N/A"
                auc_str = f"{metrics['auc']:.4f}" if not np.isnan(metrics['auc']) else "N/A"
                print(f"{model_name:<30} {accuracy_str:<15} {auc_str:<15}")
            
            print("-"*80)
        
        print("="*80)


def main():
    """
    Main function to run the clip-level pain classification analysis.
    """
    # Define paths
    meta_path = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes_and_classes.xlsx'
    feature_dir = '/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2'
    
    # Create classifier
    classifier = ClipLevelPainClassifier(meta_path, feature_dir)
    
    # Train and evaluate models
    results = classifier.train_and_evaluate(n_splits=3, random_state=42, save_folds=True)
    
    # Print results
    classifier.print_results(results)


if __name__ == "__main__":
    main() 