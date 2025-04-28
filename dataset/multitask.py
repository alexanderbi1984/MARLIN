import os
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split # Import train_test_split
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
from pytorch_lightning import LightningDataModule
from typing import Optional, Tuple, List, Dict, Any, Union

# Need to import the underlying Dataset classes and DataModule setup logic
# Assuming BioVidLP and SyracuseLP/SyracuseDataModule are accessible
# We might need to refactor SyracuseDataModule setup logic or its Dataset
from dataset.biovid import BioVidLP  # Assuming BioVidLP is the relevant dataset class
from dataset.syracuse import SyracuseLP, SyracuseDataModule # Need SyracuseLP and potentially setup logic
from dataset.shoulder_pain import ShoulderPainLP  # Import the new ShoulderPainLP class


class MultiTaskDataModule(LightningDataModule):
    """
    LightningDataModule to load and combine data from Syracuse (Pain), BioVid (Stimulus),
    and ShoulderPain (Pain) datasets for multi-task learning with transfer learning focus.
    
    This module ensures:
    1. Training set contains Syracuse data (original+augmented), BioVid data, 
       and ShoulderPain data (if provided)
    2. Validation and test sets contain only Syracuse original clips
    3. Optional balancing of data sources and class labels in the training set
    """
    def __init__(
        self,
        # --- Required Non-Default Arguments First ---
        # Syracuse Specific Params
        syracuse_root_dir: str,
        syracuse_feature_dir: str,
        syracuse_marlin_base_dir: str,
        num_pain_classes: int, # Corresponds to Syracuse task
        # BioVid Specific Params
        biovid_root_dir: str,
        biovid_feature_dir: str,
        num_stimulus_classes: int, # Corresponds to BioVid task
        # Common Params
        batch_size: int,
        
        # --- Optional Arguments with Defaults ---
        # ShoulderPain Specific Params (Optional)
        shoulder_pain_root_dir: Optional[str] = None,
        shoulder_pain_feature_dir: Optional[str] = None,
        # Syracuse Split Ratios
        syracuse_val_ratio: float = 0.15, # Ratio for validation set from Syracuse video IDs
        syracuse_test_ratio: float = 0.15, # Ratio for test set from Syracuse video IDs
        # Common Params
        num_workers: int = 0,
        temporal_reduction: str = "mean", # Assuming same reduction for all feature sets
        # Training Set Balancing Options
        balance_sources: bool = False,  # Whether to balance Syracuse vs BioVid in training
        balance_stimulus_classes: bool = False,  # Whether to balance stimulus classes in BioVid portion
        # Data subset params (optional)
        data_ratio: float = 1.0,
        take_train: Optional[int] = None,
    ):
        super().__init__()
        self.syracuse_root_dir = syracuse_root_dir
        self.syracuse_feature_dir = syracuse_feature_dir
        self.syracuse_marlin_base_dir = syracuse_marlin_base_dir
        self.num_pain_classes = num_pain_classes
        self.syracuse_val_ratio = syracuse_val_ratio
        self.syracuse_test_ratio = syracuse_test_ratio

        self.biovid_root_dir = biovid_root_dir
        self.biovid_feature_dir = biovid_feature_dir
        self.num_stimulus_classes = num_stimulus_classes
        
        # ShoulderPain params
        self.shoulder_pain_root_dir = shoulder_pain_root_dir
        self.shoulder_pain_feature_dir = shoulder_pain_feature_dir
        self.use_shoulder_pain = shoulder_pain_root_dir is not None and shoulder_pain_feature_dir is not None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.temporal_reduction = temporal_reduction # Consider separate if needed
        
        # Balancing options
        self.balance_sources = balance_sources
        self.balance_stimulus_classes = balance_stimulus_classes
        
        self.data_ratio = data_ratio # Applied to BioVid train set
        self.take_train = take_train # Applied to BioVid train set

        # Placeholders for the combined datasets
        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[MultiTaskWrapper] = None # Will be only wrapped Syracuse val
        self.test_dataset: Optional[MultiTaskWrapper] = None # Will be only wrapped Syracuse test

        # Save hyperparameters for logging/checkpointing
        self.save_hyperparameters()

    def _balance_biovid_by_class(self, biovid_dataset: BioVidLP, target_count: int) -> List[int]:
        """
        Return indices to sample from BioVid dataset to achieve class balance
        and match the target count as closely as possible.
        
        Args:
            biovid_dataset: The BioVid dataset to balance
            target_count: Approximate number of samples to select
            
        Returns:
            List of indices to include
        """
        # Get all labels from the dataset
        labels = []
        for i in range(len(biovid_dataset)):
            _, label = biovid_dataset[i]
            labels.append(label.item())
        
        # Count samples per class
        class_counts = {}
        for label in labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        # Get indices for each class
        class_indices = {cls: [] for cls in class_counts.keys()}
        for i, label in enumerate(labels):
            class_indices[label].append(i)
        
        # Calculate how many samples to take per class for balance
        num_classes = len(class_counts)
        if num_classes == 0:
            print("Warning: No classes found in BioVid dataset for balancing.")
            return []
        samples_per_class = target_count // num_classes

        # Ensure we don't exceed available samples for any class
        min_samples_in_class = float('inf')
        for count in class_counts.values():
            min_samples_in_class = min(min_samples_in_class, count)
            
        if samples_per_class > min_samples_in_class:
             print(f"Warning: Target samples per class ({samples_per_class}) exceeds minimum samples in a class ({min_samples_in_class}). Adjusting target per class.")
             samples_per_class = min_samples_in_class
        
        if samples_per_class == 0:
             print("Warning: Cannot balance BioVid dataset with 0 samples per class. Returning empty list.")
             return []

        # Collect balanced indices
        balanced_indices = []
        for cls, indices in class_indices.items():
            # Sample with replacement if needed (especially if samples_per_class > len(indices))
            if len(indices) == 0:
                print(f"Warning: No samples for class {cls}. Skipping.")
                continue
            # Sample with replacement always guarantees k samples
            selected = random.choices(indices, k=samples_per_class) 
            balanced_indices.extend(selected)

        # Shuffle the indices
        random.shuffle(balanced_indices)

        # Adjust total count slightly if needed due to floor division
        if len(balanced_indices) > target_count:
            balanced_indices = balanced_indices[:target_count]
        elif len(balanced_indices) < target_count:
            # Optionally add more random samples (could slightly unbalance)
            pass 

        return balanced_indices

    def setup(self, stage: Optional[str] = None):
        """
        Loads data, sets up datasets, and performs splits with balancing options.
        """
        print(f"Setting up MultiTaskDataModule for stage: {stage}...")

        # --- 1. Load Syracuse Metadata --- 
        print("  Loading Syracuse metadata...")
        # Instantiate SyracuseDataModule primarily to use its metadata loading logic
        syracuse_dm_for_meta = SyracuseDataModule(
             root_dir=self.syracuse_root_dir,
             task='multiclass', # Task/num_classes needed for label extraction during meta processing
             num_classes=self.num_pain_classes,
             batch_size=1, # Dummy value
             feature_dir=self.syracuse_feature_dir,
             marlin_base_dir=self.syracuse_marlin_base_dir,
             temporal_reduction=self.temporal_reduction,
             num_workers=0 # Dummy value
        )
        # Call setup to load metadata and populate lists
        try:
            syracuse_dm_for_meta.setup(stage=stage) 
        except Exception as e:
             print(f"Error during SyracuseDataModule setup for metadata: {e}")
             raise

        # Extract necessary info loaded by syracuse_dm_for_meta.setup()
        all_syracuse_metadata = syracuse_dm_for_meta.all_metadata
        original_clips = syracuse_dm_for_meta.original_clips
        augmented_clips = syracuse_dm_for_meta.augmented_clips
        video_id_labels = syracuse_dm_for_meta.video_id_labels # Map {video_id: representative_label}

        if not video_id_labels:
            raise ValueError("Failed to extract video_id_labels from Syracuse metadata.")

        # --- 2. Perform Syracuse Train/Val/Test Split based on Video IDs --- 
        print("  Performing Syracuse train/val/test split based on video IDs...")
        unique_video_ids = list(video_id_labels.keys())
        video_labels_for_stratify = [video_id_labels[vid] for vid in unique_video_ids]

        if len(unique_video_ids) < 2: # Need at least 2 videos to split
            raise ValueError("Not enough unique Syracuse video IDs to perform train/val/test split.")

        # Calculate split sizes
        test_size = self.syracuse_test_ratio
        val_size_rel_to_trainval = self.syracuse_val_ratio / (1.0 - test_size)
        
        # First split: Train+Val vs Test
        try:
            train_val_ids, test_ids = train_test_split(
                unique_video_ids,
                test_size=test_size,
                stratify=video_labels_for_stratify, # Stratify by label if possible
                random_state=42
            )
        except ValueError as e:
             print(f"Warning: Could not stratify Syracuse split ({e}). Splitting without stratification.")
             train_val_ids, test_ids = train_test_split(
                 unique_video_ids, test_size=test_size, random_state=42
             )
        
        # Second split: Train vs Val (from Train+Val set)
        train_ids = []
        val_ids = []
        if train_val_ids:
            try:
                # Need labels corresponding to train_val_ids for stratification
                train_val_labels = [video_id_labels[vid] for vid in train_val_ids]
                if len(set(train_val_labels)) > 1: # Can only stratify with >1 class
                     train_ids, val_ids = train_test_split(
                         train_val_ids,
                         test_size=val_size_rel_to_trainval,
                         stratify=train_val_labels,
                         random_state=42
                     )
                else:
                     print("Warning: Only one class in train/val set for Syracuse. Splitting without stratification.")
                     train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_rel_to_trainval, random_state=42)
            except ValueError as e:
                 print(f"Warning: Could not stratify Syracuse train/val split ({e}). Splitting without stratification.")
                 train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_rel_to_trainval, random_state=42)
        else:
             print("Warning: train_val_ids list is empty after first split.")
             # Handle case where train_val_ids might be empty if test_size is too large

        train_ids_set = set(train_ids)
        val_ids_set = set(val_ids)
        test_ids_set = set(test_ids)

        print(f"    Split Video IDs: Train={len(train_ids_set)}, Val={len(val_ids_set)}, Test={len(test_ids_set)}")
        
        # --- 3. Create Syracuse train/val/test datasets based on the video ID splits ---
        print("  Creating Syracuse datasets...")

        # For each original clip, choose train/val/test based on video_id
        syracuse_train_filenames = []
        syracuse_val_filenames = []
        syracuse_test_filenames = []

        for clip in original_clips:
            vid = clip.get('video_id')
            filename = clip.get('filename')
            if vid in train_ids_set:
                syracuse_train_filenames.append(filename)
            elif vid in val_ids_set:
                syracuse_val_filenames.append(filename)
            elif vid in test_ids_set:
                syracuse_test_filenames.append(filename)
            # else: possible if metadata has videos not in video_id_labels

        # Add augmented clips to training set only (standard practice)
        for clip in augmented_clips:
            vid = clip.get('video_id')
            filename = clip.get('filename')
            if vid in train_ids_set:
                syracuse_train_filenames.append(filename)

        print(f"    Syracuse Train Files: {len(syracuse_train_filenames)}")
        print(f"    Syracuse Val Files: {len(syracuse_val_filenames)}")
        print(f"    Syracuse Test Files: {len(syracuse_test_filenames)}")

        # Create Syracuse datasets for each split
        syracuse_train_set = SyracuseLP(
            self.syracuse_root_dir,
            self.syracuse_feature_dir,
            split='train',
            task='multiclass',
            num_classes=self.num_pain_classes,
            temporal_reduction=self.temporal_reduction,
            name_list=syracuse_train_filenames, # For cross-validation/debugging: explicit list
            metadata=all_syracuse_metadata
        )

        syracuse_val_set = SyracuseLP(
            self.syracuse_root_dir,
            self.syracuse_feature_dir,
            split='val',
            task='multiclass',
            num_classes=self.num_pain_classes,
            temporal_reduction=self.temporal_reduction,
            name_list=syracuse_val_filenames,
            metadata=all_syracuse_metadata
        )

        syracuse_test_set = SyracuseLP(
            self.syracuse_root_dir,
            self.syracuse_feature_dir,
            split='test',
            task='multiclass',
            num_classes=self.num_pain_classes,
            temporal_reduction=self.temporal_reduction,
            name_list=syracuse_test_filenames,
            metadata=all_syracuse_metadata
        )

        # --- 4. Load BioVid Training Dataset ---
        print("  Loading BioVid training dataset...")
        biovid_train_set = BioVidLP(
             self.biovid_root_dir,
             self.biovid_feature_dir,
             split='train', # Only use the training split
             task='multiclass',  
             num_classes=self.num_stimulus_classes,
             temporal_reduction=self.temporal_reduction,
             data_ratio=self.data_ratio, # Sample subset if requested
             take_num=self.take_train # Take limited number if requested
        )
        
        print(f"    BioVid Train Set Size: {len(biovid_train_set)}")
        
        # --- 5. Load ShoulderPain Dataset (if enabled) ---
        shoulder_pain_train_set = None
        if self.use_shoulder_pain:
            print("  Loading ShoulderPain dataset...")
            try:
                shoulder_pain_train_set = ShoulderPainLP(
                    self.shoulder_pain_root_dir,
                    self.shoulder_pain_feature_dir,
                    split='train',  # Only available as training data
                    task='multiclass',
                    num_classes=self.num_pain_classes,  # Same number of classes as Syracuse
                    temporal_reduction=self.temporal_reduction,
                    data_ratio=self.data_ratio,
                    take_num=self.take_train
                )
                print(f"    ShoulderPain Train Set Size: {len(shoulder_pain_train_set)}")
                
                # Print class distribution
                class_distribution = shoulder_pain_train_set.get_class_distribution()
                print(f"    ShoulderPain Class Distribution: {class_distribution}")
            except Exception as e:
                print(f"    Error loading ShoulderPain dataset: {e}")
                print(f"    Continuing without ShoulderPain dataset.")
                self.use_shoulder_pain = False

        # --- 6. Balance BioVid Classes (Stimulus) if requested ---
        if self.balance_stimulus_classes and len(biovid_train_set) > 0:
            print("  Balancing BioVid stimulus classes...")
            # Default target count is just the current set size
            target_count = len(biovid_train_set)
            
            # Get balanced indices
            balanced_indices = self._balance_biovid_by_class(biovid_train_set, target_count)
            if balanced_indices:
                # Create new subset of BioVid with balanced classes
                biovid_train_set = Subset(biovid_train_set, balanced_indices)
                print(f"    Balanced BioVid Train Set Size: {len(biovid_train_set)}")
            else:
                print("    Warning: Could not balance BioVid classes. Using original dataset.")

        # --- 7. Wrap datasets for multi-task format ---
        # Wrap the Syracuse datasets for pain task
        wrapped_syracuse_train = MultiTaskWrapper(syracuse_train_set, 'pain')
        self.val_dataset = MultiTaskWrapper(syracuse_val_set, 'pain') 
        self.test_dataset = MultiTaskWrapper(syracuse_test_set, 'pain')
        
        # Wrap BioVid dataset for stimulus task
        wrapped_biovid_train = MultiTaskWrapper(biovid_train_set, 'stimulus')
        
        # Wrap ShoulderPain dataset for pain task (if available)
        wrapped_shoulder_pain_train = None
        if self.use_shoulder_pain and shoulder_pain_train_set is not None:
            wrapped_shoulder_pain_train = MultiTaskWrapper(shoulder_pain_train_set, 'pain')

        # --- 8. Combine training datasets ---
        # Start with Syracuse and BioVid
        train_datasets = [wrapped_syracuse_train, wrapped_biovid_train]
        
        # Add ShoulderPain if available
        if self.use_shoulder_pain and wrapped_shoulder_pain_train is not None:
            train_datasets.append(wrapped_shoulder_pain_train)
            
        self.train_dataset = ConcatDataset(train_datasets)
        
        # Print dataset statistics
        print("  Final Dataset Sizes:")
        print(f"    Training: {len(self.train_dataset)} samples")
        print(f"      - Syracuse: {len(wrapped_syracuse_train)} samples")
        print(f"      - BioVid: {len(wrapped_biovid_train)} samples")
        if self.use_shoulder_pain and wrapped_shoulder_pain_train is not None:
            print(f"      - ShoulderPain: {len(wrapped_shoulder_pain_train)} samples")
        print(f"    Validation: {len(self.val_dataset)} samples")
        print(f"    Test: {len(self.test_dataset)} samples")

    def train_dataloader(self):
        if not self.train_dataset:
             self.setup(stage='fit')
        # Handle case where train_dataset might be empty after setup issues
        if not self.train_dataset or len(self.train_dataset) == 0:
            print("Warning: Training dataset is empty or not initialized. Returning None.")
            return None 
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        if not self.val_dataset:
             self.setup(stage='fit')
        if not self.val_dataset or len(self.val_dataset) == 0:
            print("Warning: Validation dataset is empty or not initialized. Returning None.")
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        if not self.test_dataset:
             self.setup(stage='test')
        if not self.test_dataset or len(self.test_dataset) == 0:
             print("Warning: Test dataset is empty or not initialized. Returning None.")
             return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

# --- Helper Wrapper Dataset ---
class MultiTaskWrapper(Dataset):
    """
    Wraps an underlying dataset (Syracuse or BioVid) and formats its output
    for the multi-task model, adding a placeholder (-1) for the missing task label.
    Can also wrap an empty list for dummy dataset creation.
    """
    def __init__(self, dataset: Union[Dataset, List], task_type: str):
        self.dataset = dataset
        assert task_type in ['pain', 'stimulus']
        self.task_type = task_type

    def __len__(self) -> int:
        # Return 0 if dataset is an empty list
        if isinstance(self.dataset, list) and not self.dataset:
            return 0
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Handle case where dataset is an empty list
        if isinstance(self.dataset, list) and not self.dataset:
            raise IndexError("Accessing item from an empty wrapped dataset")
            
        # Fetch data from the underlying dataset
        features, label = self.dataset[index]

        if self.task_type == 'pain':
            # Syracuse dataset: provides pain label, stimulus label is missing
            pain_label = label
            stimulus_label = torch.tensor(-1, dtype=torch.long) # Placeholder
        elif self.task_type == 'stimulus':
            # BioVid dataset: provides stimulus label, pain label is missing
            pain_label = torch.tensor(-1, dtype=torch.long) # Placeholder
            stimulus_label = label
        else:
            # Should not happen due to assertion in __init__
            raise ValueError(f"Unknown task_type: {self.task_type}")

        # Ensure labels are tensors (they should be already from BioVidLP/SyracuseLP)
        if not isinstance(pain_label, torch.Tensor):
            pain_label = torch.tensor(pain_label, dtype=torch.long)
        if not isinstance(stimulus_label, torch.Tensor):
             stimulus_label = torch.tensor(stimulus_label, dtype=torch.long)

        return features, pain_label, stimulus_label 