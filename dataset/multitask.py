import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
from pytorch_lightning import LightningDataModule
from typing import Optional, Tuple, List, Dict, Any

# Need to import the underlying Dataset classes and DataModule setup logic
# Assuming BioVidLP and SyracuseLP/SyracuseDataModule are accessible
# We might need to refactor SyracuseDataModule setup logic or its Dataset
from dataset.biovid import BioVidLP  # Assuming BioVidLP is the relevant dataset class
from dataset.syracuse import SyracuseLP, SyracuseDataModule # Need SyracuseLP and potentially setup logic


class MultiTaskDataModule(LightningDataModule):
    """
    LightningDataModule to load and combine data from Syracuse (Pain) and BioVid (Stimulus) datasets
    for multi-task learning with transfer learning focus.
    
    This module ensures:
    1. Training set contains both Syracuse data (original+augmented) and BioVid data
    2. Validation and test sets contain only Syracuse original clips
    3. Optional balancing of data sources and class labels in the training set
    """
    def __init__(
        self,
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
        num_workers: int = 0,
        temporal_reduction: str = "mean", # Assuming same reduction for both feature sets
        
        # Training Set Balancing Options
        balance_sources: bool = False,  # Whether to balance Syracuse vs BioVid in training
        balance_stimulus_classes: bool = False,  # Whether to balance stimulus classes in BioVid portion
        
        # Data subset params (optional)
        data_ratio: float = 1.0,
        take_train: Optional[int] = None,
        take_val: Optional[int] = None, # Keep for SyracuseDM init if needed, but not used for combined val
        take_test: Optional[int] = None, # Keep for SyracuseDM init if needed, but not used for combined test
    ):
        super().__init__()
        self.syracuse_root_dir = syracuse_root_dir
        self.syracuse_feature_dir = syracuse_feature_dir
        self.syracuse_marlin_base_dir = syracuse_marlin_base_dir
        self.num_pain_classes = num_pain_classes

        self.biovid_root_dir = biovid_root_dir
        self.biovid_feature_dir = biovid_feature_dir
        self.num_stimulus_classes = num_stimulus_classes

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.temporal_reduction = temporal_reduction # Consider separate if needed
        
        # Balancing options
        self.balance_sources = balance_sources
        self.balance_stimulus_classes = balance_stimulus_classes
        
        self.data_ratio = data_ratio # Applied to both train sets individually for now
        self.take_train = take_train # Applied to both train sets individually for now
        self.take_val = take_val
        self.take_test = take_test

        # Placeholders for the combined datasets
        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[MultiTaskWrapper] = None # Will be only wrapped Syracuse val
        self.test_dataset: Optional[MultiTaskWrapper] = None # Will be only wrapped Syracuse test

        # Save hyperparameters for logging/checkpointing
        # Ensure all relevant parameters passed to __init__ are captured
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
        samples_per_class = target_count // num_classes
        
        # Ensure we don't exceed available samples for any class
        for cls, count in class_counts.items():
            if count < samples_per_class:
                print(f"Warning: Class {cls} only has {count} samples, less than the target {samples_per_class}")
                samples_per_class = min(samples_per_class, count)
        
        # Collect balanced indices
        balanced_indices = []
        for cls, indices in class_indices.items():
            # Sample with replacement if needed
            if len(indices) < samples_per_class:
                selected = random.choices(indices, k=samples_per_class)
            else:
                selected = random.sample(indices, samples_per_class)
            balanced_indices.extend(selected)
        
        # Shuffle the indices
        random.shuffle(balanced_indices)
        
        # Adjust to match target count if needed
        if len(balanced_indices) > target_count:
            balanced_indices = balanced_indices[:target_count]
        
        return balanced_indices

    def setup(self, stage: Optional[str] = None):
        """
        Loads data, sets up datasets, and performs splits with balancing options.
        """
        print(f"Setting up MultiTaskDataModule for stage: {stage}...")

        # --- 1. Setup Syracuse Data using its video_id-based splitting ---
        print("  Setting up Syracuse dataset...")
        syracuse_dm = SyracuseDataModule(
             root_dir=self.syracuse_root_dir,
             task='multiclass',
             num_classes=self.num_pain_classes,
             batch_size=self.batch_size,
             feature_dir=self.syracuse_feature_dir,
             marlin_base_dir=self.syracuse_marlin_base_dir,
             temporal_reduction=self.temporal_reduction,
             num_workers=self.num_workers
        )
        # Initialize and split the Syracuse data
        syracuse_dm.setup(stage=stage)
        
        # Get the Syracuse datasets - train has original+augmented, val/test have only originals
        syracuse_train_set = syracuse_dm.train_dataset  
        syracuse_val_set = syracuse_dm.val_dataset
        syracuse_test_set = syracuse_dm.test_dataset
        print(f"    Syracuse train/val/test sizes: {len(syracuse_train_set)}/{len(syracuse_val_set)}/{len(syracuse_test_set)}")

        # --- 2. Load BioVid Training Data ---
        print("  Setting up BioVid dataset...")
        biovid_train_set = None
        if stage == 'fit' or stage is None:
            # Use data_ratio and take_train specifically for the BioVid training set
            biovid_train_set = BioVidLP(
                 root_dir=self.biovid_root_dir, feature_dir=self.biovid_feature_dir,
                 split='train', task='multiclass', num_classes=self.num_stimulus_classes,
                 temporal_reduction=self.temporal_reduction,
                 data_ratio=self.data_ratio, # Apply ratio to BioVid train set
                 take_num=self.take_train  # Apply take_num to BioVid train set
            )
            print(f"    BioVid train size: {len(biovid_train_set)}")

        # --- 3. Apply Balancing if Requested ---
        # Wrapper for Syracuse train set
        wrapped_syracuse_train = MultiTaskWrapper(syracuse_train_set, task_type='pain')
        
        # Determine if and how to balance BioVid data
        if self.balance_sources or self.balance_stimulus_classes:
            print("  Applying dataset balancing...")
            
            # Target count based on Syracuse size if balancing sources
            target_biovid_count = len(syracuse_train_set) if self.balance_sources else len(biovid_train_set)
            
            if self.balance_stimulus_classes:
                # Balance by class and match target count
                print(f"    Balancing BioVid stimulus classes (target: ~{target_biovid_count} samples)...")
                balanced_indices = self._balance_biovid_by_class(biovid_train_set, target_biovid_count)
                biovid_train_subset = Subset(biovid_train_set, balanced_indices)
                wrapped_biovid_train = MultiTaskWrapper(biovid_train_subset, task_type='stimulus')
                print(f"    BioVid balanced size: {len(balanced_indices)}")
            else:
                # Just sample to match target count if only source balancing
                if self.balance_sources and len(biovid_train_set) > target_biovid_count:
                    print(f"    Sampling BioVid to match Syracuse size ({target_biovid_count} samples)...")
                    indices = random.sample(range(len(biovid_train_set)), target_biovid_count)
                    biovid_train_subset = Subset(biovid_train_set, indices)
                    wrapped_biovid_train = MultiTaskWrapper(biovid_train_subset, task_type='stimulus')
                    print(f"    BioVid sampled size: {len(indices)}")
                else:
                    # Use full BioVid dataset if no sampling needed
                    wrapped_biovid_train = MultiTaskWrapper(biovid_train_set, task_type='stimulus')
        else:
            # No balancing, use full BioVid training set
            wrapped_biovid_train = MultiTaskWrapper(biovid_train_set, task_type='stimulus')
        
        # --- 4. Build Final Datasets ---
        # Construct training set with both Syracuse and BioVid
        self.train_dataset = ConcatDataset([wrapped_syracuse_train, wrapped_biovid_train])
        
        # Validation and test sets contain ONLY Syracuse data
        self.val_dataset = MultiTaskWrapper(syracuse_val_set, task_type='pain')
        self.test_dataset = MultiTaskWrapper(syracuse_test_set, task_type='pain')
        
        print(f"  Final dataset sizes:")
        print(f"    Train: {len(self.train_dataset)} (Syracuse: {len(wrapped_syracuse_train)}, BioVid: {len(wrapped_biovid_train)})")
        print(f"    Validation: {len(self.val_dataset)} (Syracuse only)")
        print(f"    Test: {len(self.test_dataset)} (Syracuse only)")
        
        print("MultiTaskDataModule setup complete.")

    def train_dataloader(self):
        if not self.train_dataset:
             self.setup(stage='fit')
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
    """
    def __init__(self, dataset: Dataset, task_type: str):
        self.dataset = dataset
        assert task_type in ['pain', 'stimulus']
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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