"""
Pain Classifier Module

This module provides functionality for pain classification from video clips using pre-extracted features.
It implements robust machine learning approaches for multiclass and binary pain level classification
with cross-validation, class balancing, and performance evaluation.

Key Features:
-----------
- Loads pre-extracted MARLIN features from .npy files for machine learning classification
- Supports multiple pain classification schemes (3, 4, or 5 class configurations)
- Implements various cross-validation fold creation strategies:
  * stratified - balanced class distribution across folds
  * video_based - ensures clips from same video stay together
  * part_based - ensures clips from same video part stay together
  * aug_aware - creates folds with original videos while allowing augmented versions in training
- Provides comprehensive machine learning models including:
  * Linear models (Logistic Regression, Ridge Classifier)
  * SVM models with different kernels (linear, RBF, polynomial, sigmoid)
  * Neural networks (MLPs with various architectures)
  * Custom Lasso regression classifier
- Implements intelligent augmentation handling:
  * Controls augmentation percentage (0%, 25%, 50%, 75%, or 100%)
  * Employs class-weighted sampling for augmented clips to address class imbalance
  * Maintains data integrity by keeping related clips together
- Supports binary classification between any two pain classes
- Provides detailed training statistics and evaluation metrics:
  * Accuracy, AUC, precision, recall, F1 score
  * Per-fold and average performance reporting
  * Training/testing set distribution analysis

Classes:
-------
- LassoClassifier: Custom classifier using Lasso regression, implementing scikit-learn's
  estimator interface with multiclass support via one-vs-rest approach
- MarlinPainClassifier: Main classifier class that handles data loading, fold creation,
  model training, and evaluation for pain classification tasks

Implementation Details:
---------------------
- Uses MarlinFeatures class to load pre-extracted features from a MARLIN base directory
- Intelligently handles class imbalance through weighted sampling of augmented data
- Supports various fold creation strategies to prevent data leakage
- Implements cross-validation with comprehensive performance metrics
- Provides methods for saving and loading classification results

Dependencies:
-----------
- numpy, pandas, scikit-learn for data handling and ML algorithms
- MarlinFeatures class for loading pre-extracted clip features
"""

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
            aug_per_video: Controls the percentage of augmented clips to use in training:
                           0 = No augmentation (original clips only)
                           1 = 25%, 2 = 50%, 3 = 75%, 4 = 100% of available augmented clips
            
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
                if aug_train_idx is not None and len(aug_train_idx) > 0 and aug_per_video > 0:
                    # Calculate percentage based on aug_per_video
                    if aug_per_video < 0 or aug_per_video > 4:
                        print(f"  Warning: aug_per_video must be between 0 and 4. Using aug_per_video=4 (100%)")
                        percentage = 1.0
                    elif aug_per_video == 0:
                        # Skip augmentation completely
                        percentage = 0.0
                    else:
                        percentage = aug_per_video * 0.25
                    
                    # Implement class-weighted sampling to address class imbalance
                    # First, analyze the class distribution in original training set
                    class_counts = np.bincount(y_train_orig[y_train_orig >= 0])
                    print(f"  Original class distribution: {class_counts}")
                    
                    # If all classes have samples, compute inverse frequency weights
                    if np.all(class_counts > 0):
                        # Compute inverse frequency for weighting (more weight to minority classes)
                        class_weights = 1.0 / class_counts
                        class_weights = class_weights / np.sum(class_weights)  # Normalize to sum to 1
                        print(f"  Class weights for sampling: {np.round(class_weights, 3)}")
                        
                        # Group augmented indices by class
                        aug_indices_by_class = {}
                        for i in aug_train_idx:
                            label = y_aug[i]
                            if label >= 0:  # Skip invalid labels
                                if label not in aug_indices_by_class:
                                    aug_indices_by_class[label] = []
                                aug_indices_by_class[label].append(i)
                        
                        # Calculate how many samples to take from each class
                        total_to_select = int(len(aug_train_idx) * percentage)
                        samples_per_class = {}
                        
                        print(f"  Available augmented clips by class:")
                        for label, indices in aug_indices_by_class.items():
                            print(f"    Class {label}: {len(indices)} clips")
                            # Calculate weighted number of samples (at least 1 if available)
                            if label < len(class_weights):
                                weight = class_weights[label]
                                class_samples = max(1, int(total_to_select * weight))
                                # Cap to available samples
                                class_samples = min(class_samples, len(indices))
                                samples_per_class[label] = class_samples
                        
                        # Select samples from each class
                        limited_aug_idx = []
                        print(f"  Selecting augmented clips by class (class-weighted sampling):")
                        for label, count in samples_per_class.items():
                            indices = aug_indices_by_class[label]
                            # Shuffle to randomize selection
                            np.random.shuffle(indices)
                            # Take requested number of samples
                            selected = indices[:count]
                            limited_aug_idx.extend(selected)
                            print(f"    Class {label}: {len(selected)}/{len(indices)} clips (weight={class_weights[label]:.3f})")
                        
                        # Ensure we respect the total percentage limit
                        # If we have too many samples, randomly subsample
                        if len(limited_aug_idx) > total_to_select:
                            np.random.shuffle(limited_aug_idx)
                            limited_aug_idx = limited_aug_idx[:total_to_select]
                            print(f"  Reduced to {total_to_select} clips to match requested percentage")
                        # If we have too few samples, add random samples
                        elif len(limited_aug_idx) < total_to_select and len(aug_train_idx) > 0:
                            remaining = total_to_select - len(limited_aug_idx)
                            # Get indices we haven't already selected
                            remaining_indices = [i for i in aug_train_idx if i not in limited_aug_idx]
                            if remaining_indices:
                                np.random.shuffle(remaining_indices)
                                additional = remaining_indices[:min(remaining, len(remaining_indices))]
                                limited_aug_idx.extend(additional)
                                print(f"  Added {len(additional)} random clips to reach requested percentage")
                        
                        limited_aug_idx = np.array(limited_aug_idx)
                        
                    else:
                        # Fallback to random sampling if not all classes have samples
                        print("  Not all classes have samples in training set. Using random sampling.")
                        # Shuffle all augmented indices to randomize selection
                        all_aug_indices = np.array(aug_train_idx)
                        np.random.shuffle(all_aug_indices)
                        # Select the specified percentage of augmented clips
                        num_to_select = int(len(all_aug_indices) * percentage)
                        limited_aug_idx = all_aug_indices[:num_to_select]
                    
                    # Now use the limited augmented indices
                    X_train_aug = self.X_aug[limited_aug_idx]
                    y_train_aug = y_aug[limited_aug_idx]
                    
                    # Combine original and augmented data for training
                    X_train = np.vstack((X_train_orig, X_train_aug))
                    y_train = np.concatenate((y_train_orig, y_train_aug))
                    
                    # Report on augmentation and class distribution
                    print(f"  Fold {fold_idx+1} training: {len(X_train_orig)} original clips + {len(X_train_aug)} augmented clips = {len(X_train)} total")
                    print(f"  Using {percentage*100:.0f}% of available augmented clips ({len(X_train_aug)}/{len(aug_train_idx)})")
                    
                    # Print class distribution after augmentation
                    aug_class_counts = np.bincount(y_train_aug[y_train_aug >= 0])
                    final_class_counts = np.bincount(y_train[y_train >= 0])
                    print(f"  Augmented clips class distribution: {aug_class_counts}")
                    print(f"  Final training data class distribution: {final_class_counts}")
                else:
                    X_train = X_train_orig
                    y_train = y_train_orig
                    print(f"  Fold {fold_idx+1} training: {len(X_train)} original clips (no augmented clips)")
                    # Print class distribution
                    orig_class_counts = np.bincount(y_train[y_train >= 0])
                    print(f"  Training data class distribution: {orig_class_counts}")
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
    
    def train_model_binary(self, model_name: str, class_indices: List[int], class_set: Optional[int] = None, 
                          n_splits: int = 3, fold_strategy: str = 'stratified', random_state: int = 42,
                          aug_per_video: int = 1) -> Dict[str, Any]:
        """
        Train a model for binary classification between two specific classes.
        
        Args:
            model_name: Name of the model to train
            class_indices: List of two class indices to include in binary classification
            class_set: Which class set to use (3, 4, or 5). If None, will be determined automatically.
            n_splits: Number of cross-validation splits
            fold_strategy: Strategy for creating folds ('stratified', 'video_based', 'part_based', 'aug_aware')
            random_state: Random seed for reproducibility
            aug_per_video: Controls the percentage of augmented clips to use in training:
                           0 = No augmentation (original clips only)
                           1 = 25%, 2 = 50%, 3 = 75%, 4 = 100% of available augmented clips
            
        Returns:
            Dictionary containing training results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        if len(class_indices) != 2:
            raise ValueError(f"Binary classification requires exactly 2 class indices, got {len(class_indices)}")
        
        class1, class2 = class_indices
        print(f"Binary classification between class {class1} and class {class2}")
        
        # Determine which label set to use
        if class_set is not None:
            # Use explicitly specified class set
            if class_set == 3:
                print(f"Using 3-class labels (explicitly specified)")
                y_all = self.y_3
                y_aug_all = self.y_3_aug
            elif class_set == 4:
                print(f"Using 4-class labels (explicitly specified)")
                y_all = self.y_4
                y_aug_all = self.y_4_aug
            elif class_set == 5:
                print(f"Using 5-class labels (explicitly specified)")
                y_all = self.y_5
                y_aug_all = self.y_5_aug
            else:
                raise ValueError(f"Invalid class set: {class_set}. Must be 3, 4, or 5.")
        else:
            # Automatically determine class set based on indices
            max_class = max(class1, class2)
            if max_class <= 2:  # Both classes are in 3-class set
                print("Using 3-class labels (automatically determined)")
                y_all = self.y_3
                y_aug_all = self.y_3_aug
            elif max_class <= 3:  # Both classes are in 4-class set
                print("Using 4-class labels (automatically determined)")
                y_all = self.y_4
                y_aug_all = self.y_4_aug
            else:  # At least one class is in 5-class set
                print("Using 5-class labels (automatically determined)")
                y_all = self.y_5
                y_aug_all = self.y_5_aug
        
        # Filter data to include only the specified classes
        orig_indices = np.where((y_all == class1) | (y_all == class2))[0]
        X = self.X[orig_indices]
        y = y_all[orig_indices]
        
        # Remap classes to 0 and 1 for binary classification
        y_binary = np.zeros_like(y)
        y_binary[y == class2] = 1
        
        # Also get video names and IDs for fold creation
        video_names = self.video_names[orig_indices]
        video_ids = self.video_ids[orig_indices]
        
        # Filter augmented data if available
        if self.X_aug is not None and y_aug_all is not None:
            aug_indices = np.where((y_aug_all == class1) | (y_aug_all == class2))[0]
            X_aug = self.X_aug[aug_indices]
            y_aug = y_aug_all[aug_indices]
            
            # Remap augmented classes to 0 and 1
            y_aug_binary = np.zeros_like(y_aug)
            y_aug_binary[y_aug == class2] = 1
            
            video_names_aug = self.video_names_aug[aug_indices] if self.video_names_aug is not None else None
            video_ids_aug = self.video_ids_aug[aug_indices] if self.video_ids_aug is not None else None
        else:
            X_aug = None
            y_aug_binary = None
            video_names_aug = None
            video_ids_aug = None
        
        # Print class distribution
        class_counts = np.bincount(y_binary)
        print(f"Original class distribution: {class_counts}")
        if y_aug_binary is not None:
            aug_class_counts = np.bincount(y_aug_binary)
            print(f"Augmented class distribution: {aug_class_counts}")
        
        # Create folds based on the specified strategy
        if fold_strategy == 'aug_aware':
            # For aug_aware strategy, we need to create folds based on video IDs
            # and track which augmented samples belong to which video
            folds = []
            
            # Get unique video IDs and their labels
            unique_video_ids = np.unique(video_ids)
            video_id_labels = []
            
            for vid in unique_video_ids:
                if vid is None:
                    continue
                vid_mask = video_ids == vid
                vid_label = np.bincount(y_binary[vid_mask]).argmax()
                video_id_labels.append((vid, vid_label))
            
            # Filter out None values
            video_id_labels = [(vid, label) for vid, label in video_id_labels if vid is not None]
            
            # Extract video IDs and labels
            filtered_video_ids = np.array([vid for vid, _ in video_id_labels])
            video_labels = np.array([label for _, label in video_id_labels])
            
            # Create stratified folds using video IDs
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            for train_vid_idx, test_vid_idx in skf.split(filtered_video_ids, video_labels):
                # Get video IDs for training and testing
                train_vids = filtered_video_ids[train_vid_idx]
                test_vids = filtered_video_ids[test_vid_idx]
                
                # Map video IDs back to indices in the filtered datasets
                train_idx = np.array([i for i, vid in enumerate(video_ids) 
                                   if vid is not None and vid in train_vids])
                test_idx = np.array([i for i, vid in enumerate(video_ids) 
                                  if vid is not None and vid in test_vids])
                
                # Get indices for augmented clips from training videos (if available)
                aug_train_idx = []
                if X_aug is not None and video_ids_aug is not None:
                    for i, vid in enumerate(video_ids_aug):
                        if vid in train_vids:
                            aug_train_idx.append(i)
                
                # Store fold information
                fold_info = {
                    'train_idx': train_idx,
                    'test_idx': test_idx,
                    'aug_train_idx': np.array(aug_train_idx) if aug_train_idx else None
                }
                
                folds.append(fold_info)
        else:
            # For other strategies, use standard cross-validation on binary labels
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = []
            for train_idx, test_idx in skf.split(X, y_binary):
                folds.append((train_idx, test_idx))
        
        # Initialize results
        results = {
            'accuracy': [],
            'auc': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'predictions': [],
            'true_labels': [],
            'train_sizes': [],
            'test_sizes': [],
            'class_indices': class_indices  # Store the original class indices for reference
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
                
                # Get training and testing data
                X_train_orig = X[train_idx]
                y_train_orig = y_binary[train_idx]
                X_test = X[test_idx]
                y_test = y_binary[test_idx]
                
                # Include augmented clips in training if available, but limit by aug_per_video
                if aug_train_idx is not None and len(aug_train_idx) > 0 and X_aug is not None:
                    # Calculate percentage based on aug_per_video
                    if aug_per_video < 0 or aug_per_video > 4:
                        print(f"  Warning: aug_per_video must be between 0 and 4. Using aug_per_video=4 (100%)")
                        percentage = 1.0
                    elif aug_per_video == 0:
                        # Skip augmentation completely
                        percentage = 0.0
                    else:
                        percentage = aug_per_video * 0.25
                    
                    # Implement class-weighted sampling to address class imbalance
                    # First, analyze the class distribution in original training set
                    class_counts = np.bincount(y_train_orig)
                    print(f"  Original class distribution: {class_counts}")
                    
                    # If all classes have samples, compute inverse frequency weights
                    if np.all(class_counts > 0):
                        # Compute inverse frequency for weighting (more weight to minority classes)
                        class_weights = 1.0 / class_counts
                        class_weights = class_weights / np.sum(class_weights)  # Normalize to sum to 1
                        print(f"  Class weights for sampling: {np.round(class_weights, 3)}")
                        
                        # Group augmented indices by class
                        aug_indices_by_class = {}
                        for i in aug_train_idx:
                            label = y_aug_binary[i]
                            if label not in aug_indices_by_class:
                                aug_indices_by_class[label] = []
                            aug_indices_by_class[label].append(i)
                        
                        # Calculate how many samples to take from each class
                        total_to_select = int(len(aug_train_idx) * percentage)
                        samples_per_class = {}
                        
                        print(f"  Available augmented clips by class:")
                        for label, indices in aug_indices_by_class.items():
                            print(f"    Class {label}: {len(indices)} clips")
                            # Calculate weighted number of samples (at least 1 if available)
                            weight = class_weights[label]
                            class_samples = max(1, int(total_to_select * weight))
                            # Cap to available samples
                            class_samples = min(class_samples, len(indices))
                            samples_per_class[label] = class_samples
                        
                        # Select samples from each class
                        limited_aug_idx = []
                        print(f"  Selecting augmented clips by class (class-weighted sampling):")
                        for label, count in samples_per_class.items():
                            indices = aug_indices_by_class[label]
                            # Shuffle to randomize selection
                            np.random.shuffle(indices)
                            # Take requested number of samples
                            selected = indices[:count]
                            limited_aug_idx.extend(selected)
                            print(f"    Class {label}: {len(selected)}/{len(indices)} clips (weight={class_weights[label]:.3f})")
                        
                        # Ensure we respect the total percentage limit
                        if len(limited_aug_idx) > total_to_select:
                            np.random.shuffle(limited_aug_idx)
                            limited_aug_idx = limited_aug_idx[:total_to_select]
                            print(f"  Reduced to {total_to_select} clips to match requested percentage")
                        elif len(limited_aug_idx) < total_to_select and len(aug_train_idx) > 0:
                            remaining = total_to_select - len(limited_aug_idx)
                            remaining_indices = [i for i in aug_train_idx if i not in limited_aug_idx]
                            if remaining_indices:
                                np.random.shuffle(remaining_indices)
                                additional = remaining_indices[:min(remaining, len(remaining_indices))]
                                limited_aug_idx.extend(additional)
                                print(f"  Added {len(additional)} random clips to reach requested percentage")
                        
                        limited_aug_idx = np.array(limited_aug_idx)
                        
                    else:
                        # Fallback to random sampling if not all classes have samples
                        print("  Not all classes have samples in training set. Using random sampling.")
                        all_aug_indices = np.array(aug_train_idx)
                        np.random.shuffle(all_aug_indices)
                        num_to_select = int(len(all_aug_indices) * percentage)
                        limited_aug_idx = all_aug_indices[:num_to_select]
                    
                    # Now use the limited augmented indices
                    X_train_aug = X_aug[limited_aug_idx]
                    y_train_aug = y_aug_binary[limited_aug_idx]
                    
                    # Combine original and augmented data for training
                    X_train = np.vstack((X_train_orig, X_train_aug))
                    y_train = np.concatenate((y_train_orig, y_train_aug))
                    
                    # Report on augmentation and class distribution
                    print(f"  Fold {fold_idx+1} training: {len(X_train_orig)} original clips + {len(X_train_aug)} augmented clips = {len(X_train)} total")
                    print(f"  Using {percentage*100:.0f}% of available augmented clips ({len(X_train_aug)}/{len(aug_train_idx)})")
                    
                    # Print class distribution after augmentation
                    aug_class_counts = np.bincount(y_train_aug)
                    final_class_counts = np.bincount(y_train)
                    print(f"  Augmented clips class distribution: {aug_class_counts}")
                    print(f"  Final training data class distribution: {final_class_counts}")
                else:
                    X_train = X_train_orig
                    y_train = y_train_orig
                    print(f"  Fold {fold_idx+1} training: {len(X_train)} original clips (no augmented clips)")
                    # Print class distribution
                    class_counts = np.bincount(y_train)
                    print(f"  Training data class distribution: {class_counts}")
            else:
                # For other strategies, use standard train/test split
                train_idx, test_idx = fold
                X_train = X[train_idx]
                y_train = y_binary[train_idx]
                X_test = X[test_idx]
                y_test = y_binary[test_idx]
                print(f"  Fold {fold_idx+1} training: {len(X_train)} clips")
                
                # Print class distribution
                class_counts = np.bincount(y_train)
                print(f"  Training data class distribution: {class_counts}")
            
            print(f"  Fold {fold_idx+1} testing: {len(X_test)} clips")
            
            # Store counts
            results['train_sizes'].append(len(X_train))
            results['test_sizes'].append(len(X_test))
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate probability predictions
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]  # Prob of class 1 for binary
            else:
                # For models without predict_proba, use a simple fallback
                y_prob = y_pred
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            
            try:
                auc = roc_auc_score(y_test, y_prob)
            except Exception as e:
                print(f"  Warning: Could not calculate AUC - {str(e)}")
                auc = 0.0
            
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Store results
            results['accuracy'].append(accuracy)
            results['auc'].append(auc)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)
            results['predictions'].extend(y_pred)
            results['true_labels'].extend(y_test)
            
            print(f"  Fold {fold_idx+1} metrics: Acc={accuracy:.3f}, AUC={auc:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
        
        # Calculate average metrics
        results['mean_accuracy'] = np.mean(results['accuracy'])
        results['std_accuracy'] = np.std(results['accuracy'])
        results['mean_auc'] = np.mean(results['auc'])
        results['std_auc'] = np.std(results['auc'])
        results['mean_precision'] = np.mean(results['precision'])
        results['std_precision'] = np.std(results['precision'])
        results['mean_recall'] = np.mean(results['recall'])
        results['std_recall'] = np.std(results['recall'])
        results['mean_f1'] = np.mean(results['f1_score'])
        results['std_f1'] = np.std(results['f1_score'])
        
        # Print average training/testing sizes and metrics
        avg_train = np.mean(results['train_sizes'])
        avg_test = np.mean(results['test_sizes'])
        print(f"\nAverage training set size: {avg_train:.1f} clips")
        print(f"Average testing set size: {avg_test:.1f} clips")
        print(f"Mean accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
        print(f"Mean AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
        print(f"Mean precision: {results['mean_precision']:.3f} ± {results['std_precision']:.3f}")
        print(f"Mean recall: {results['mean_recall']:.3f} ± {results['std_recall']:.3f}")
        print(f"Mean F1 score: {results['mean_f1']:.3f} ± {results['std_f1']:.3f}")
        
        return results 

