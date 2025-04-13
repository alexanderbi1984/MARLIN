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
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import json
import pickle
import torch
from pathlib import Path
from marlin_features import MarlinFeatures

"""
Pain Classifier Module

This module provides the MarlinPainClassifier class for pain classification from video clips.
The class handles loading pre-extracted features from a marlin_base_dir, creating stratified
cross-validation folds, and training/evaluating machine learning models.

Key features:
- Loads clip metadata and pre-extracted features from .npy files
- Supports different pain classification schemes (3, 4, or 5 classes)
- Provides multiple fold creation strategies: stratified, video-based, part-based, and augmentation-aware
- Includes built-in classification models and metrics reporting
- Handles training/test split creation with awareness of data augmentation
- Reports detailed training and testing statistics for each fold

Dependencies:
- numpy, pandas, scikit-learn
- marlin_features.MarlinFeatures for feature extraction
"""

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

class MarlinPainClassifier:
    """
    Class for performing pain level classification using MARLIN features and various models.
    """
    
    def __init__(self, meta_path: str, marlin_base_dir: str, model_name: str = "marlin_vit_small"):
        """
        Initialize the classifier with metadata path and MARLIN features directory.
        
        Args:
            meta_path: Path to the meta_with_outcomes.xlsx file
            marlin_base_dir: Base directory containing pre-extracted MARLIN features and clips_json.json
            model_name: Name of the MARLIN model (for reference only, not used for extraction)
        """
        self.meta_path = meta_path
        self.marlin_base_dir = marlin_base_dir
        self.model_name = model_name
        
        # Initialize MarlinFeatures for loading pre-extracted features
        self.marlin_features = MarlinFeatures(marlin_base_dir)
        
        # Initialize data containers
        self.clips = None
        self.X = None  # Features for original clips
        self.X_aug = None  # Features for augmented clips 
        self.y_3 = None
        self.y_4 = None
        self.y_5 = None
        self.video_names = None
        self.video_types = None  # 'original' or 'aug'
        self.video_ids = None  # Store video IDs for mapping between original and augmented
        
        # Initialize models
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
            'SVM (rbf)': SVC(kernel='rbf', random_state=42, probability=True),
            'SVM (poly)': SVC(kernel='poly', random_state=42, probability=True),
            'SVM (sigmoid)': SVC(kernel='sigmoid', random_state=42, probability=True),
            
            # Neural Network models
            'MLP (small)': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
            'MLP (medium)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'MLP (large)': MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=1000, random_state=42)
        }
    
    def get_clip_features(self, clip_filename: str) -> np.ndarray:
        """
        Get pre-extracted features for a clip from the MarlinFeatures class.
        
        Args:
            clip_filename: The filename of the clip
            
        Returns:
            numpy array of features
        """
        clip_data = self.marlin_features.get_clip(clip_filename)
        if clip_data:
            return clip_data['features'].flatten()  # Convert (4, 768) to flat vector
        else:
            raise ValueError(f"No pre-extracted features found for clip: {clip_filename}")
    
    def load_data(self, include_augmented: bool = True) -> None:
        """
        Load all clips and prepare data for classification.
        
        Args:
            include_augmented: Whether to include augmented clips
        """
        print("Loading clip data from pre-extracted features...")
        
        # Load metadata
        self.meta_df = pd.read_excel(self.meta_path)
        
        # Get all video IDs available in MarlinFeatures
        all_video_ids = self.marlin_features.get_all_video_ids()
        print(f"Found {len(all_video_ids)} unique video IDs in MarlinFeatures")
        
        # Extract features and labels
        features_orig = []
        features_aug = []
        class_3_orig = []
        class_3_aug = []
        class_4_orig = []
        class_4_aug = []
        class_5_orig = []
        class_5_aug = []
        video_names_orig = []
        video_names_aug = []
        video_ids_orig = []
        video_ids_aug = []
        
        # Get all clips metadata from MarlinFeatures
        clips_metadata = self.marlin_features.clips_metadata
        
        # Process each clip
        for filename, metadata in clips_metadata.items():
            video_id = metadata['video_id']
            clip_id = metadata['clip_id']
            video_type = metadata['video_type']  # 'original' or 'aug'
            meta_info = metadata['meta_info']
            
            # Skip augmented clips if not requested
            if video_type == 'aug' and not include_augmented:
                continue
            
            # Skip if missing class labels
            try:
                class_3_val = meta_info['class_3']
                class_4_val = meta_info['class_4']
                class_5_val = meta_info['class_5']
            except KeyError:
                continue
            
            # Load pre-extracted features
            try:
                clip_data = self.marlin_features.get_clip(filename)
                if not clip_data:
                    continue
                features = clip_data['features'].flatten()  # Convert (4, 768) to flat vector
            except Exception as e:
                print(f"Error loading clip {filename}: {e}")
                continue
            
            # Add to appropriate lists (original or augmented)
            if video_type == 'aug':
                features_aug.append(features)
                class_3_aug.append(int(class_3_val) if pd.notna(class_3_val) else -1)
                class_4_aug.append(int(class_4_val) if pd.notna(class_4_val) else -1)
                class_5_aug.append(int(class_5_val) if pd.notna(class_5_val) else -1)
                video_names_aug.append(filename)
                video_ids_aug.append(video_id)
            else:
                features_orig.append(features)
                class_3_orig.append(int(class_3_val) if pd.notna(class_3_val) else -1)
                class_4_orig.append(int(class_4_val) if pd.notna(class_4_val) else -1)
                class_5_orig.append(int(class_5_val) if pd.notna(class_5_val) else -1)
                video_names_orig.append(filename)
                video_ids_orig.append(video_id)
        
        # Convert to numpy arrays
        self.X = np.array(features_orig)
        self.X_aug = np.array(features_aug) if features_aug else None
        
        self.y_3 = np.array(class_3_orig, dtype=int)
        self.y_3_aug = np.array(class_3_aug, dtype=int) if class_3_aug else None
        
        self.y_4 = np.array(class_4_orig, dtype=int)
        self.y_4_aug = np.array(class_4_aug, dtype=int) if class_4_aug else None
        
        self.y_5 = np.array(class_5_orig, dtype=int)
        self.y_5_aug = np.array(class_5_aug, dtype=int) if class_5_aug else None
        
        self.video_names = np.array(video_names_orig)
        self.video_names_aug = np.array(video_names_aug) if video_names_aug else None
        
        self.video_ids = np.array(video_ids_orig)
        self.video_ids_aug = np.array(video_ids_aug) if video_ids_aug else None
        
        # Print summary statistics
        print(f"Loaded {len(video_names_orig)} original clips with features of shape {self.X.shape}")
        if include_augmented and self.X_aug is not None:
            print(f"Loaded {len(video_names_aug)} augmented clips with features of shape {self.X_aug.shape}")
        
        # Print class distributions
        print(f"Class distribution (3-class, original): {np.bincount(self.y_3[self.y_3 >= 0])}")
        print(f"Class distribution (4-class, original): {np.bincount(self.y_4[self.y_4 >= 0])}")
        print(f"Class distribution (5-class, original): {np.bincount(self.y_5[self.y_5 >= 0])}")
        
        # Print number of clips with valid class labels
        print(f"Original clips with valid 3-class labels: {np.sum(self.y_3 >= 0)}/{len(video_names_orig)}")
        print(f"Original clips with valid 4-class labels: {np.sum(self.y_4 >= 0)}/{len(video_names_orig)}")
        print(f"Original clips with valid 5-class labels: {np.sum(self.y_5 >= 0)}/{len(video_names_orig)}")
    
    def create_folds(self, y: np.ndarray, n_splits: int = 3, strategy: str = 'stratified', 
                    random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation folds based on the specified strategy.
        
        Args:
            y: Labels for stratification
            n_splits: Number of folds
            strategy: Strategy for creating folds ('stratified', 'video_based', 'part_based', 'aug_aware')
            random_state: Random seed for reproducibility
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        # Remove samples with invalid labels
        valid_mask = y >= 0
        valid_indices = np.where(valid_mask)[0]
        valid_y = y[valid_mask]
        valid_video_names = self.video_names[valid_mask]
        
        if strategy == 'stratified':
            # Standard stratified k-fold
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = []
            for train_idx, test_idx in skf.split(valid_y, valid_y):
                # Map back to original indices
                train_idx_orig = valid_indices[train_idx]
                test_idx_orig = valid_indices[test_idx]
                folds.append((train_idx_orig, test_idx_orig))
            return folds
        
        elif strategy == 'video_based':
            # Create folds based on video names
            return self._create_video_based_folds(valid_video_names, valid_y, valid_indices, n_splits, random_state)
        
        elif strategy == 'part_based':
            # Create folds based on video parts
            return self._create_part_based_folds(valid_video_names, valid_y, valid_indices, n_splits, random_state)
            
        elif strategy == 'aug_aware':
            # Create folds with original videos, but allow augmented videos for training
            return self._create_aug_aware_folds(valid_y, valid_indices, n_splits, random_state)
        
        else:
            raise ValueError(f"Unknown fold strategy: {strategy}")

    def _create_aug_aware_folds(self, labels: np.ndarray, valid_indices: np.ndarray, 
                               n_splits: int, random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create folds based on original videos with augmented videos included in training.
        
        This strategy:
        1. Creates folds using only original videos, keeping videos with the same ID together
        2. For training, includes augmented clips from videos in the training fold
        3. For testing, only uses original clips
        
        Args:
            labels: Array of labels
            valid_indices: Indices of valid samples
            n_splits: Number of folds
            random_state: Random seed
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        valid_video_ids = self.video_ids[valid_indices]
        
        # Get unique video IDs and their most common labels
        unique_video_ids = np.unique(valid_video_ids)
        video_id_labels = []
        
        for vid in unique_video_ids:
            if vid is None:
                continue
            vid_mask = valid_video_ids == vid
            vid_label = np.bincount(labels[vid_mask]).argmax()
            video_id_labels.append((vid, vid_label))
        
        # Filter out None values
        video_id_labels = [(vid, label) for vid, label in video_id_labels if vid is not None]
        
        # Extract video IDs and labels
        filtered_video_ids = np.array([vid for vid, _ in video_id_labels])
        video_labels = np.array([label for _, label in video_id_labels])
        
        # Create stratified folds using video IDs
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        folds = []
        for train_vid_idx, test_vid_idx in skf.split(filtered_video_ids, video_labels):
            # Get video IDs for training and testing
            train_vids = filtered_video_ids[train_vid_idx]
            test_vids = filtered_video_ids[test_vid_idx]
            
            # Get indices for original clips from these videos
            train_idx_orig = np.array([i for i, vid in enumerate(valid_video_ids) 
                                     if vid is not None and vid in train_vids])
            test_idx_orig = np.array([i for i, vid in enumerate(valid_video_ids) 
                                    if vid is not None and vid in test_vids])
            
            # Map back to original indices
            train_idx = valid_indices[train_idx_orig]
            test_idx = valid_indices[test_idx_orig]
            
            # Get indices for augmented clips from training videos (if available)
            aug_train_idx = []
            if self.X_aug is not None and self.video_ids_aug is not None:
                for i, vid in enumerate(self.video_ids_aug):
                    if vid in train_vids:
                        aug_train_idx.append(i)
            
            # Store fold information
            fold_info = {
                'train_idx': train_idx,
                'test_idx': test_idx,
                'aug_train_idx': np.array(aug_train_idx) if aug_train_idx else None
            }
            
            folds.append(fold_info)
        
        return folds
    
    def _create_video_based_folds(self, video_names: np.ndarray, labels: np.ndarray, 
                                 valid_indices: np.ndarray, n_splits: int, random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create folds based on video names to ensure videos from the same source
        are not split across train and test sets.
        
        Args:
            video_names: Array of video names
            labels: Array of labels
            valid_indices: Indices of valid samples
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
            
            # Map back to original indices
            train_idx_orig = valid_indices[train_idx]
            test_idx_orig = valid_indices[test_idx]
            
            folds.append((train_idx_orig, test_idx_orig))
        
        return folds
    
    def _create_part_based_folds(self, video_names: np.ndarray, labels: np.ndarray, 
                                valid_indices: np.ndarray, n_splits: int, random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create folds based on video parts to ensure parts from the same video
        are not split across train and test sets.
        
        Args:
            video_names: Array of video names
            labels: Array of labels
            valid_indices: Indices of valid samples
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
            
            # Map back to original indices
            train_idx_orig = valid_indices[train_idx]
            test_idx_orig = valid_indices[test_idx]
            
            folds.append((train_idx_orig, test_idx_orig))
        
        return folds
    
    def train_model(self, model_name: str, n_classes: int = 3, n_splits: int = 3, 
                   fold_strategy: str = 'stratified', random_state: int = 42,
                   aug_per_video: int = 1) -> Dict[str, Any]:
        """
        Train a model using cross-validation.
        
        Args:
            model_name: Name of the model to train
            n_classes: Number of classes (3, 4, or 5)
            n_splits: Number of cross-validation splits
            fold_strategy: Strategy for creating folds ('stratified', 'video_based', 'part_based', 'aug_aware')
            random_state: Random seed for reproducibility
            aug_per_video: Maximum number of augmented videos to use per original video (1-4, default: 1)
            
        Returns:
            Dictionary containing training results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Select appropriate labels
        if n_classes == 3:
            y = self.y_3
            y_aug = self.y_3_aug
        elif n_classes == 4:
            y = self.y_4
            y_aug = self.y_4_aug
        elif n_classes == 5:
            y = self.y_5
            y_aug = self.y_5_aug
        else:
            raise ValueError(f"Invalid number of classes: {n_classes}")
        
        # Create folds based on the specified strategy
        folds = self.create_folds(y, n_splits, fold_strategy, random_state)
        
        # Initialize results
        results = {
            'accuracy': [],
            'auc': [],
            'predictions': [],
            'true_labels': [],
            'train_sizes': [],
            'test_sizes': []
        }
        
        # Train and evaluate
        model = self.models[model_name]
        
        for fold_idx, fold in enumerate(folds):
            print(f"Training fold {fold_idx+1}/{len(folds)}...")
            
            if fold_strategy == 'aug_aware':
                # For aug_aware strategy, handle augmented clips specially
                train_idx = fold['train_idx']
                test_idx = fold['test_idx']
                aug_train_idx = fold['aug_train_idx']
                
                # Get training and testing data (original clips)
                X_train_orig = self.X[train_idx]
                y_train_orig = y[train_idx]
                X_test = self.X[test_idx]
                y_test = y[test_idx]
                
                # Include augmented clips in training if available, but limit by aug_per_video
                if aug_train_idx is not None and len(aug_train_idx) > 0:
                    # Map from original video ID to its augmented videos
                    vid_to_aug = defaultdict(list)
                    for aug_idx in aug_train_idx:
                        vid_id = self.video_ids_aug[aug_idx]
                        vid_to_aug[vid_id].append(aug_idx)
                    
                    # Select limited number of augmented videos per original
                    limited_aug_idx = []
                    for vid_id, aug_indices in vid_to_aug.items():
                        # Shuffle augmented indices to randomize selection
                        aug_indices_shuffled = list(aug_indices)
                        np.random.shuffle(aug_indices_shuffled)
                        # Take only the specified number of augmented videos
                        limited_aug_idx.extend(aug_indices_shuffled[:aug_per_video])
                    
                    limited_aug_idx = np.array(limited_aug_idx)
                    
                    # Now use the limited augmented indices
                    X_train_aug = self.X_aug[limited_aug_idx]
                    y_train_aug = y_aug[limited_aug_idx]
                    
                    # Combine original and augmented data for training
                    X_train = np.vstack((X_train_orig, X_train_aug))
                    y_train = np.concatenate((y_train_orig, y_train_aug))
                    
                    print(f"  Fold {fold_idx+1} training: {len(X_train_orig)} original clips + {len(X_train_aug)} augmented clips = {len(X_train)} total")
                    print(f"  Limited to max {aug_per_video} augmented clips per original video")
                else:
                    X_train = X_train_orig
                    y_train = y_train_orig
                    print(f"  Fold {fold_idx+1} training: {len(X_train)} original clips (no augmented clips)")
            else:
                # For other strategies, use standard train/test split
                train_idx, test_idx = fold
                X_train = self.X[train_idx]
                y_train = y[train_idx]
                X_test = self.X[test_idx]
                y_test = y[test_idx]
                print(f"  Fold {fold_idx+1} training: {len(X_train)} clips")
            
            print(f"  Fold {fold_idx+1} testing: {len(X_test)} clips")
            
            # Store counts
            results['train_sizes'].append(len(X_train))
            results['test_sizes'].append(len(X_test))
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Check if model has predict_proba method, otherwise create a workaround
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
            else:
                # For models like RidgeClassifier that don't have predict_proba
                # Create a simple one-hot encoding of predictions as probability estimates
                n_samples = len(y_test)
                
                # Get all possible classes (not just those in the current fold)
                if n_classes == 3:
                    all_classes = np.unique(self.y_3[self.y_3 >= 0])
                elif n_classes == 4:
                    all_classes = np.unique(self.y_4[self.y_4 >= 0])
                elif n_classes == 5:
                    all_classes = np.unique(self.y_5[self.y_5 >= 0])
                else:
                    all_classes = np.unique(y)
                
                n_all_classes = len(all_classes)
                y_prob = np.zeros((n_samples, n_all_classes))
                
                # Map predicted class indices to positions in the all_classes array
                for i, pred in enumerate(y_pred):
                    class_idx = np.where(all_classes == pred)[0]
                    if len(class_idx) > 0:
                        y_prob[i, class_idx[0]] = 1.0
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
            except ValueError as e:
                print(f"  Warning: Could not calculate AUC - {str(e)}")
                print(f"  y_test unique classes: {np.unique(y_test)}")
                print(f"  y_prob shape: {y_prob.shape}")
                auc = 0.0  # Default value when AUC cannot be calculated
            
            # Store results
            results['accuracy'].append(accuracy)
            results['auc'].append(auc)
            results['predictions'].extend(y_pred)
            results['true_labels'].extend(y_test)
            
            print(f"  Fold {fold_idx+1} accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        
        # Calculate average metrics
        results['mean_accuracy'] = np.mean(results['accuracy'])
        results['std_accuracy'] = np.std(results['accuracy'])
        results['mean_auc'] = np.mean(results['auc'])
        results['std_auc'] = np.std(results['auc'])
        
        # Print average training/testing sizes
        avg_train = np.mean(results['train_sizes'])
        avg_test = np.mean(results['test_sizes'])
        print(f"\nAverage training set size: {avg_train:.1f} clips")
        print(f"Average testing set size: {avg_test:.1f} clips")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save classification results to a file.
        
        Args:
            results: Dictionary containing classification results
            output_path: Path to save the results
        """
        # Convert NumPy types to native Python types for JSON serialization
        json_safe_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_safe_results[key] = value.tolist()
            elif isinstance(value, list):
                # Convert any NumPy types within lists
                json_safe_results[key] = [
                    item.item() if hasattr(item, 'item') else item 
                    for item in value
                ]
            elif hasattr(value, 'item'):  # For NumPy scalars
                json_safe_results[key] = value.item()
            else:
                json_safe_results[key] = value
                
        with open(output_path, 'w') as f:
            json.dump(json_safe_results, f, indent=4)
    
    def load_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load classification results from a file.
        
        Args:
            input_path: Path to the results file
            
        Returns:
            Dictionary containing classification results
        """
        with open(input_path, 'r') as f:
            return json.load(f) 