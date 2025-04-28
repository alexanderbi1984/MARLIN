import torch
import numpy as np
import os
import json
from torch.utils.data import Dataset
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

class ShoulderPainLP(Dataset):
    """
    Dataset class for ShoulderPain features with pain level classification.
    Similar structure to BioVid dataset, but focused on pain classification.
    
    Args:
        root_dir (str): Root directory of the ShoulderPain dataset
        feature_dir (str): Directory name containing feature files
        split (str): Dataset split ('train' only for ShoulderPain)
        task (str): Task type ('multiclass')
        num_classes (int): Number of pain classes
        temporal_reduction (str): Method to reduce temporal dimension ('mean', 'max', 'min', 'none')
        data_ratio (float, optional): Ratio of data to use (0.0-1.0). Default: 1.0
        take_num (int, optional): Number of samples to take. Default: None
    """
    def __init__(
        self,
        root_dir: str,
        feature_dir: str, 
        split: str,
        task: str,
        num_classes: int,
        temporal_reduction: str,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None
    ):
        self.root_dir = root_dir
        self.feature_dir = os.path.join(root_dir, feature_dir)
        self.split = split
        self.task = task
        self.num_classes = num_classes
        self.temporal_reduction = temporal_reduction
        self.data_ratio = data_ratio
        self.take_num = take_num
        
        # Only training set is used for ShoulderPain
        if split != 'train':
            raise ValueError("ShoulderPain dataset is only available for training ('train' split)")
        
        # Load metadata
        try:
            metadata_path = os.path.join(root_dir, 'shoulder_pain_info.json')
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading ShoulderPain metadata: {e}")
        
        # Load training file list
        try:
            split_file = os.path.join(root_dir, 'train.txt')
            with open(split_file, 'r') as f:
                self.orig_name_list = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise ValueError(f"Error loading ShoulderPain split file: {e}")
        
        # Check if feature directory exists
        if not os.path.exists(self.feature_dir):
            raise ValueError(f"Feature directory not found: {self.feature_dir}")
        
        # Create actual name list with features that exist
        self.name_list = []
        self.label_list = []
        self.vas_values = []
        
        # Define the class binning ranges
        self.vas_bins = [
            (0.0, 1.0),   # Class 0: Pain level 0.0 - 1.0
            (2.0, 3.0),   # Class 1: Pain level 2.0 - 3.0
            (4.0, 5.0),   # Class 2: Pain level 4.0 - 5.0
            (6.0, 7.0),   # Class 3: Pain level 6.0 - 7.0
            (8.0, 10.0)   # Class 4: Pain level 8.0 - 10.0
        ]
        
        # Process each file in the split
        skipped = 0
        for rel_path in self.orig_name_list:
            # Convert video path to feature path (replace .mp4 with .npy)
            feature_filename = os.path.basename(rel_path).replace('.mp4', '.npy')
            feature_path = os.path.join(self.feature_dir, feature_filename)
            
            # Check if feature file exists
            if not os.path.exists(feature_path):
                skipped += 1
                continue
            
            # Get metadata for this file
            if rel_path not in self.metadata.get('clips', {}):
                skipped += 1
                continue
                
            file_metadata = self.metadata['clips'].get(rel_path, {}).get('attributes', {})
            
            # Get VAS score and convert to float
            vas_str = file_metadata.get('vas', '')
            try:
                vas = float(vas_str)
                
                # Determine class based on VAS bins
                pain_class = -1  # Default invalid class
                for class_idx, (min_val, max_val) in enumerate(self.vas_bins):
                    if min_val <= vas <= max_val:
                        pain_class = class_idx
                        break
                
                # Skip samples that don't fit into any bin
                if pain_class == -1:
                    skipped += 1
                    continue
                
                # Store valid sample
                self.name_list.append(feature_path)
                self.label_list.append(pain_class)
                self.vas_values.append(vas)
                
            except (ValueError, TypeError):
                skipped += 1
                continue
        
        # Log dataset statistics
        logger.info(f"ShoulderPain {split} set: {len(self.name_list)} valid samples, {skipped} skipped")
        
        # Sample subset if requested
        if data_ratio < 1.0 or take_num is not None:
            self._subsample()
    
    def _subsample(self):
        """Reduce dataset size based on data_ratio or take_num."""
        n = len(self.name_list)
        if self.take_num is not None and self.take_num < n:
            # Take first N samples
            indices = list(range(self.take_num))
        elif self.data_ratio < 1.0:
            # Take random subset based on ratio
            num_to_keep = max(1, int(n * self.data_ratio))
            indices = np.random.choice(n, num_to_keep, replace=False).tolist()
        else:
            return  # No subsampling needed
        
        # Update lists with selected indices
        self.name_list = [self.name_list[i] for i in indices]
        self.label_list = [self.label_list[i] for i in indices]
        self.vas_values = [self.vas_values[i] for i in indices]
    
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
        feature_path = self.name_list[idx]
        label = self.label_list[idx]
        
        try:
            # Load and preprocess feature
            feature = np.load(feature_path).astype(np.float32)
            feature = self._reduce_temporal(feature)
            feature = torch.from_numpy(feature)
            
            # Convert label to tensor
            label = torch.tensor(label, dtype=torch.long)
            
            return feature, label
        
        except Exception as e:
            logger.error(f"Error loading sample {feature_path}: {e}")
            # Return a dummy sample of correct shape in case of error
            if self.temporal_reduction == 'none':
                # Assuming same shape as Syracuse (4, 768)
                feature = torch.zeros((4, 768), dtype=torch.float32)
            else:
                feature = torch.zeros(768, dtype=torch.float32)
            label = torch.tensor(0, dtype=torch.long)
            return feature, label
            
    def get_class_distribution(self):
        """Get distribution of classes."""
        if not self.label_list:
            return {}
        
        class_counts = {}
        for class_idx in range(self.num_classes):
            count = sum(1 for label in self.label_list if label == class_idx)
            class_counts[class_idx] = count
        
        return class_counts 