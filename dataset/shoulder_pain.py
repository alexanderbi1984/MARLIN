import torch
import numpy as np
import os
import json
from torch.utils.data import Dataset
from typing import List, Dict, Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Define a helper function for mapping VAS score to class index using cutoffs
def map_score_to_class(score: float, cutoffs: List[float]) -> int:
    """Maps a continuous score to a 0-indexed class label based on cutoffs.
    
    Args:
        score: The continuous score (e.g., VAS).
        cutoffs: A sorted list of upper boundaries for classes 0 to N-2. 
                 Example: [1.0, 3.0, 5.0, 7.0] defines 5 classes:
                 - Class 0: score <= 1.0
                 - Class 1: 1.0 < score <= 3.0
                 - Class 2: 3.0 < score <= 5.0
                 - Class 3: 5.0 < score <= 7.0
                 - Class 4: score > 7.0
                 
    Returns:
        The 0-indexed class label.
    """
    class_label = 0
    for cutoff in cutoffs:
        if score > cutoff:
            class_label += 1
        else:
            break
    return class_label

class ShoulderPainLP(Dataset):
    """
    Dataset class for ShoulderPain features with pain level classification.
    Similar structure to BioVid dataset, but focused on pain classification.
    
    Args:
        root_dir (str): Root directory of the ShoulderPain dataset
        feature_dir (str): Directory name containing feature files
        split (str): Dataset split ('train' only for ShoulderPain)
        pain_class_cutoffs (List[float]): List of cutoff points for pain classes
        temporal_reduction (str): Method to reduce temporal dimension ('mean', 'max', 'min', 'none')
        data_ratio (float, optional): Ratio of data to use (0.0-1.0). Default: 1.0
        take_num (int, optional): Number of samples to take. Default: None
        name_list (Optional[List[str]], optional): Explicit list of filenames to load. Default: None
        metadata (Optional[Dict], optional): Preloaded metadata if available. Default: None
    """
    def __init__(
        self,
        root_dir: str,
        feature_dir: str, 
        split: str,
        pain_class_cutoffs: List[float],
        temporal_reduction: str,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None,
        name_list: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        self.root_dir = root_dir
        self.feature_dir = os.path.join(root_dir, feature_dir)
        self.split = split
        self.pain_class_cutoffs = sorted(pain_class_cutoffs)
        self.num_classes = len(self.pain_class_cutoffs) + 1
        self.temporal_reduction = temporal_reduction
        self.data_ratio = data_ratio
        self.take_num = take_num
        self.name_list = name_list
        
        # Only training set is used for ShoulderPain
        if split != 'train':
            raise ValueError("ShoulderPain dataset is only available for training ('train' split)")
        
        # Load metadata
        if metadata is None:
            meta_path = os.path.join(root_dir, 'shoulder_pain_info.json')
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)['clips']
        else:
            self.metadata = metadata['clips'] if 'clips' in metadata else metadata
        
        # Load filenames for the specified split or use provided list
        if self.name_list is None:
            split_file = os.path.join(root_dir, f"{split}.txt")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")
            with open(split_file, 'r') as f:
                # Read relative paths/filenames from train.txt
                self.name_list = [line.strip() for line in f.readlines()]
            print(f"ShoulderPainLP using {split_file} with {len(self.name_list)} items.")

            # Apply data_ratio and take_num if filenames loaded from split file
            if self.data_ratio < 1.0:
                keep_num = int(len(self.name_list) * self.data_ratio)
                self.name_list = self.name_list[:keep_num]
                print(f"  Applying data_ratio {self.data_ratio}, keeping {len(self.name_list)} items.")
            if self.take_num is not None and self.take_num > 0:
                self.name_list = self.name_list[:self.take_num]
                print(f"  Applying take_num {self.take_num}, keeping {len(self.name_list)} items.")

        # Filter name_list to ensure corresponding metadata and feature files exist
        self._validate_and_filter_namelist()
    
    def _validate_and_filter_namelist(self):
        """Filters name_list to include only entries with valid metadata, features, and VAS score."""
        valid_names = []
        missing_meta = 0
        missing_features = 0
        invalid_vas = 0
        original_count = len(self.name_list)

        for name in self.name_list:
            # Metadata uses video filename key (e.g., .mp4)
            # Features use base filename (e.g., .npy)
            base_name = os.path.splitext(os.path.basename(name))[0]
            feature_path = os.path.join(self.feature_dir, f"{base_name}.npy")
            video_key = name # Assume name in train.txt matches the key in metadata

            # Check metadata
            clip_meta = self.metadata.get(video_key)
            if not clip_meta:
                # Try matching without extension if needed (less robust)
                clip_meta = self.metadata.get(base_name)
                if not clip_meta:
                     missing_meta += 1
                     continue # Skip if no metadata

            # Check VAS score in metadata
            vas_str = clip_meta.get('attributes', {}).get('vas')
            if vas_str is None:
                invalid_vas += 1
                continue # Skip if VAS is missing

            try:
                _ = float(vas_str) # Check if convertible to float
            except (ValueError, TypeError):
                invalid_vas += 1
                continue # Skip if VAS is not a valid number

            # Check feature file
            if not os.path.exists(feature_path):
                missing_features += 1
                continue # Skip if feature file doesn't exist

            # If all checks pass, add the original name (from train.txt)
            valid_names.append(name)

        if missing_meta > 0 or missing_features > 0 or invalid_vas > 0:
            print(f"  ShoulderPainLP Validation: Filtered name_list from {original_count} to {len(valid_names)}")
            if missing_meta > 0: print(f"    - Skipped {missing_meta} due to missing metadata.")
            if invalid_vas > 0: print(f"    - Skipped {invalid_vas} due to missing or invalid VAS score.")
            if missing_features > 0: print(f"    - Skipped {missing_features} due to missing feature files.")

        self.name_list = valid_names
        if not self.name_list:
             print("  WARNING: ShoulderPainLP name_list is empty after validation!")
    
    def _reduce_temporal(self, feature):
        """Apply temporal reduction to feature."""
        if self.temporal_reduction == 'mean':
            return feature.mean(axis=0)
        elif self.temporal_reduction == 'max':
            return feature.max(axis=0)
        elif self.temporal_reduction == 'min':
            return feature.min(axis=0)
        elif self.temporal_reduction == 'none':
            return feature
        else:
            raise ValueError(f"Unsupported temporal reduction: {self.temporal_reduction}")
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        video_name = self.name_list[idx]
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        feature_path = os.path.join(self.feature_dir, f"{base_name}.npy")

        # Load features
        try:
            features = np.load(feature_path)
        except Exception as e:
            print(f"Error loading feature file: {feature_path}")
            raise e
            
        # Temporal Reduction
        if self.temporal_reduction == "mean":
            features = np.mean(features, axis=0)
        elif self.temporal_reduction == "max":
            features = np.max(features, axis=0)
        elif self.temporal_reduction == "min":
            features = np.min(features, axis=0)
        elif self.temporal_reduction != "none":
            raise ValueError(f"Unknown temporal reduction type: {self.temporal_reduction}")
        # If 'none', features remain as is (e.g., [T, D])

        features = torch.tensor(features, dtype=torch.float32)

        # Get label from metadata using VAS score and cutoffs
        clip_meta = self.metadata.get(video_name) or self.metadata.get(base_name)
        if not clip_meta:
            raise ValueError(f"Metadata not found for {video_name} during __getitem__.")

        vas_str = clip_meta.get('attributes', {}).get('vas')
        if vas_str is None:
            raise ValueError(f"VAS score missing for {video_name} during __getitem__.")

        try:
            vas_score = float(vas_str)
        except (ValueError, TypeError):
             raise ValueError(f"Invalid VAS score format for {video_name}: {vas_str}")

        # Map score to class label
        class_label = map_score_to_class(vas_score, self.pain_class_cutoffs)
        label = torch.tensor(class_label, dtype=torch.long)

        return features, label

    def get_class_distribution(self):
        """Get distribution of classes."""
        if not self.name_list:
            return {}
        
        class_counts = {}
        for class_idx in range(self.num_classes):
            count = sum(1 for _, label in self if label == class_idx)
            class_counts[class_idx] = count
        
        return class_counts 