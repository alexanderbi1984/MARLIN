import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict
from sklearn.model_selection import train_test_split
from collections import Counter

# Import the actual BioVidBase class
from dataset.biovid import BioVidBase
# Import MarlinFeatures
from Syracuse.MarlinFeatures.marlin_features import MarlinFeatures

# Assume BioVidBase is defined elsewhere and provides self.data_root, self.name_list, self.metadata
# Example placeholder for BioVidBase if needed for context
# class BioVidBase(Dataset):  <-- REMOVE THIS BLOCK
#     def __init__(self, root_dir: str, split: str, task: str, num_classes: int, data_ratio: float = 1.0, take_num: Optional[int] = None):
#         self.data_root = root_dir # Example assignment
#         self.split = split
#         self.task = task
#         self.num_classes = num_classes
#         # In a real scenario, self.name_list and self.metadata would be loaded here
#         # For example, loading from a JSON or similar structure expected by the original BioVidBase
#         self.name_list = [] # Placeholder
#         self.metadata = {} # Placeholder: Structure might differ based on BioVidBase implementation
#         print(f"Warning: Using placeholder BioVidBase in {__file__}. Ensure the actual base class loads metadata compatible with MarlinFeatures structure.")

# Define the same helper function here for consistency (or import if moved to utils)
def map_score_to_class(score: float, cutoffs: List[float]) -> int:
    """Maps a continuous score to a 0-indexed class label based on cutoffs.
    
    Args:
        score: The continuous score (e.g., pain_level).
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

class SyracuseLP(BioVidBase):
    """
    Syracuse Lip-Reading Pain Dataset.
    Loads features and derives ordinal pain classes from 'pain_level' scores using configurable cutoffs.
    Inherits from a placeholder BioVidBase for potential structure reuse, but primarily functions independently.
    """

    def __init__(
        self,
        root_dir: str,
        feature_dir: str,
        split: str, # 'train', 'val', or 'test' - used primarily for logging/context
        pain_class_cutoffs: List[float], # List of cutoff points for pain classes
        temporal_reduction: str = "mean",
        name_list: Optional[List[str]] = None, # MUST be provided for specific splits
        metadata: Optional[Dict] = None, # MUST be provided
        # task: str, # No longer needed for label generation
        # num_classes: int, # No longer needed, derived from cutoffs
    ):
        # Simplified super init assuming BioVidBase only needs minimal args
        # super().__init__(root_dir, split, task, num_classes) # Call super if needed, adapt args
        # Manually set necessary attributes if super is bypassed or insufficient
        self.data_root = root_dir # Example of manual setting
        self.feature_dir_name = feature_dir # Store the relative name
        self.feature_dir = os.path.join(root_dir, feature_dir) # Full path
        self.split = split
        self.pain_class_cutoffs = sorted(pain_class_cutoffs)
        self.num_classes = len(self.pain_class_cutoffs) + 1
        self.temporal_reduction = temporal_reduction

        if name_list is None:
            raise ValueError("SyracuseLP requires 'name_list' to be provided for the specific split.")
        if metadata is None:
            raise ValueError("SyracuseLP requires 'metadata' to be provided.")

        self.name_list = name_list
        # Metadata is expected to be the full dictionary loaded from clips_json.json
        self.metadata = metadata

        print(f"SyracuseLP Dataset '{split}' initialized with {len(self.name_list)} clips.")
        if not self.metadata:
             print(f"Warning: Metadata dictionary for {split} dataset is empty!")
        elif not self.name_list:
             print(f"Warning: Name list for {split} dataset is empty!")

        # Validate name_list against metadata and features
        self._validate_namelist()

    def _validate_namelist(self):
        """Filters name_list to ensure metadata, features, and valid pain_level exist."""
        valid_names = []
        missing_meta = 0
        missing_features = 0
        missing_pain_level = 0
        invalid_pain_level = 0
        original_count = len(self.name_list)

        for name in self.name_list:
            # Construct full path to the feature file
            feature_path = os.path.join(self.feature_dir, name)
            clip_meta = self.metadata.get(name)

            # Check 1: Metadata existence
            if clip_meta is None:
                missing_meta += 1
                continue

            # Check 3: Pain level existence in metadata
            pain_level_str = clip_meta.get('meta_info', {}).get('pain_level')
            if pain_level_str is None:
                missing_pain_level += 1
                continue
                
            # Check if convertible to float
            try:
                _ = float(pain_level_str)
            except (ValueError, TypeError):
                invalid_pain_level += 1
                continue

            if not os.path.exists(feature_path):
                missing_features += 1
                continue

            valid_names.append(name)

        if original_count != len(valid_names):
            print(f"  SyracuseLP ({self.split}) Validation: Filtered name_list from {original_count} to {len(valid_names)}")
            if missing_meta > 0: print(f"    - Skipped {missing_meta} due to missing metadata.")
            if missing_features > 0: print(f"    - Skipped {missing_features} due to missing feature files.")
            if missing_pain_level > 0: print(f"    - Skipped {missing_pain_level} due to missing 'pain_level' key.")
            if invalid_pain_level > 0: print(f"    - Skipped {invalid_pain_level} due to invalid 'pain_level' format.")

        self.name_list = valid_names
        if not self.name_list:
             print(f"  WARNING: SyracuseLP ({self.split}) name_list is empty after validation!")

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to ensure shape (4, 768).
        Logic adapted from MarlinFeatures._normalize_features.

        Args:
            features (np.ndarray): Raw features array

        Returns:
            np.ndarray: Normalized features with shape (4, 768)
        """
        target_frames = 4
        feature_dim = 768

        if features is None or features.size == 0:
             print(f"Warning: Input features are None or empty. Returning zeros.")
             return np.zeros((target_frames, feature_dim), dtype=np.float32)

        if features.ndim == 1:
             if features.shape[0] == target_frames * feature_dim:
                 features = features.reshape((target_frames, feature_dim))
             elif features.shape[0] == feature_dim:
                 print(f"Warning: Received 1D features of shape {features.shape}. Repeating to {target_frames} frames.")
                 features = np.tile(features, (target_frames, 1))
             else:
                 print(f"Warning: Received unexpected 1D features shape {features.shape}. Returning zeros.")
                 return np.zeros((target_frames, feature_dim), dtype=features.dtype)
        elif features.ndim != 2:
             print(f"Warning: Input features have unexpected ndim {features.ndim}. Returning zeros.")
             return np.zeros((target_frames, feature_dim), dtype=features.dtype)

        if features.shape[1] != feature_dim:
            print(f"Warning: Feature dimension is {features.shape[1]}, expected {feature_dim}. Returning zeros.")
            return np.zeros((target_frames, feature_dim), dtype=features.dtype)

        n_frames = features.shape[0]
        if n_frames == target_frames:
            return features
        elif n_frames > target_frames:
            indices = np.linspace(0, n_frames - 1, target_frames, dtype=int)
            return features[indices]
        else:
             if n_frames == 0:
                  print(f"Warning: Input features are empty after checks. Returning zeros.")
                  return np.zeros((target_frames, feature_dim), dtype=features.dtype)
             indices = np.linspace(0, n_frames - 1, target_frames)
             interpolated_features = np.zeros((target_frames, feature_dim), dtype=features.dtype)
             frame_indices = np.arange(n_frames)
             for j in range(feature_dim):
                 interpolated_features[:, j] = np.interp(indices, frame_indices, features[:, j])
             return interpolated_features

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index: int):
        if index >= len(self.name_list):
             raise IndexError(f"Index {index} out of bounds for dataset of size {len(self.name_list)}")

        filename = self.name_list[index]
        feature_path = os.path.join(self.feature_dir, filename)

        # --- Load Features ---
        try:
            raw_features = np.load(feature_path)
        except FileNotFoundError:
            print(f"Error: Feature file not found: {feature_path}")
            dummy_features = torch.zeros((4 if self.temporal_reduction == "none" else 1, 768), dtype=torch.float32)
            dummy_label = torch.tensor(-1, dtype=torch.long)
            return dummy_features, dummy_label
        except Exception as e:
            print(f"Error loading feature file {feature_path}: {e}")
            dummy_features = torch.zeros((4 if self.temporal_reduction == "none" else 1, 768), dtype=torch.float32)
            dummy_label = torch.tensor(-1, dtype=torch.long)
            return dummy_features, dummy_label

        normalized_features = self._normalize_features(raw_features)
        x = torch.from_numpy(normalized_features).float()

        if self.temporal_reduction == "mean":
            x = x.mean(dim=0)
        elif self.temporal_reduction == "max":
            x = x.max(dim=0)[0]
        elif self.temporal_reduction == "min":
            x = x.min(dim=0)[0]
        elif self.temporal_reduction == "none":
            pass
        else:
            raise ValueError(f"Unknown temporal reduction strategy: {self.temporal_reduction}")

        # --- Get Label ---
        try:
            clip_meta = self.metadata[filename] # Use getitem since validation should ensure key exists
            meta_info = clip_meta['meta_info']
            pain_level_str = meta_info['pain_level']
            pain_level = float(pain_level_str)

            # Map score to class label using the helper function
            class_label = map_score_to_class(pain_level, self.pain_class_cutoffs)
            label_tensor = torch.tensor(class_label, dtype=torch.long)

        except KeyError as e:
            print(f"Error accessing metadata key for {filename}: {e}. Check metadata structure and task definition.")
            dummy_features = torch.zeros((4 if self.temporal_reduction == "none" else 1, 768), dtype=torch.float32)
            dummy_label = torch.tensor(-1, dtype=torch.long)
            return dummy_features, dummy_label
        except Exception as e:
            print(f"Error processing metadata or label for {filename}: {e}")
            dummy_features = torch.zeros((4 if self.temporal_reduction == "none" else 1, 768), dtype=torch.float32)
            dummy_label = torch.tensor(-1, dtype=torch.long)
            return dummy_features, dummy_label

        # --- Remove DEBUGGING --- 
        # print(f"[DEBUG SyracuseLP] Filename: {filename}, Label Type: {type(y)}, Label Value: {y}")
        # --- END Remove DEBUGGING ---
        return x, label_tensor


class SyracuseDataModule(LightningDataModule):
    """
    DataModule for Syracuse dataset implementing augmentation-aware splitting.
    Validation and Test sets contain only original clips.
    Training set contains original clips + corresponding augmented clips.
    Splits are based on video IDs to prevent data leakage.
    """

    def __init__(self, root_dir: str,          # Root directory of data
        task: str,
        num_classes: int,
        batch_size: int,             # Batch size (still needed for potential dataloaders)
        feature_dir: str,
        marlin_base_dir: str,          # REQUIRED for MarlinFeatures
        pain_class_cutoffs: List[float], # Add cutoffs here
        temporal_reduction: str = "mean",
        num_workers: int = 0
    ):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.feature_dir = feature_dir
        self.temporal_reduction = temporal_reduction
        self.marlin_base_dir = marlin_base_dir # Store the base dir
        self.num_workers = num_workers
        self.pain_class_cutoffs = sorted(pain_class_cutoffs)
        self.num_classes = len(self.pain_class_cutoffs) + 1

        # --- Data containers (will be populated in setup) ---
        self.all_metadata = {}        # All metadata {filename: meta}
        self.original_clips = []    # List of dicts for original clips {filename, video_id, label}
        self.augmented_clips = []   # List of dicts for augmented clips {filename, video_id}
        self.video_id_labels = {}   # Map {video_id: representative_label}
        self.marlin_features = None # MarlinFeatures instance

        # --- Initialize MarlinFeatures --- 
        self.marlin_features = None # Default to None
        try:
            print(f"Initializing MarlinFeatures with base directory: {self.marlin_base_dir}")
            if not os.path.exists(self.marlin_base_dir):
                 print(f"Error: Provided marlin_base_dir does not exist: {self.marlin_base_dir}")
            else:
                 # Check specifically for clips_json.json existence before initializing
                 metadata_json_path = os.path.join(self.marlin_base_dir, 'clips_json.json')
                 if not os.path.exists(metadata_json_path):
                      print(f"Error: clips_json.json not found in marlin_base_dir: {self.marlin_base_dir}")
                 else:
                      self.marlin_features = MarlinFeatures(self.marlin_base_dir)
                      print("MarlinFeatures initialized successfully.")
        except FileNotFoundError as e:
            # This might be redundant if we check above, but keep as fallback
            print(f"Error initializing MarlinFeatures (FileNotFound): {e}")
            print("Please ensure the marlin_base_dir is correct and contains clips_json.json.")
        except Exception as e:
             print(f"An unexpected error occurred during MarlinFeatures initialization: {e}")
             # Optionally re-raise if fatal: raise e

    def _load_and_prepare_metadata(self) -> bool:
        # --- Use initialized MarlinFeatures instance ---
        if self.marlin_features and hasattr(self.marlin_features, 'clips_metadata') and self.marlin_features.clips_metadata:
            self.all_metadata = self.marlin_features.clips_metadata
            print(f"Loaded metadata using initialized MarlinFeatures instance ({len(self.all_metadata)} entries).")
            return True
        else:
            print("Error: MarlinFeatures instance not initialized or has no metadata.")
            return False

    def setup(self, stage: Optional[str] = None):
        print("Setting up SyracuseDataModule...")

        if not self._load_and_prepare_metadata() or not self.all_metadata:
            raise RuntimeError("Failed to load metadata. Cannot proceed with setup.")

        # Clear lists before processing
        self.original_clips = []
        self.augmented_clips = []
        self.video_id_labels = {}

        print("Processing metadata to separate original/augmented clips and get video labels...")
        processed_files = set()
        clips_missing_info = 0
        clips_missing_label = 0

        for filename, meta in self.all_metadata.items():
             if filename in processed_files: continue

             video_id = meta.get('video_id')
             video_type = meta.get('video_type')
             meta_info = meta.get('meta_info')

             if not all([video_id, video_type, meta_info]):
                 clips_missing_info += 1
                 continue

             pain_level_str = meta_info.get('pain_level')
             if pain_level_str is None:
                 clips_missing_label += 1
                 continue

             try:
                pain_level = float(pain_level_str)
             except (ValueError, TypeError):
                 print(f"Warning: Could not convert pain_level '{pain_level_str}' to float for clip {filename}. Skipping.")
                 clips_missing_label += 1
                 continue

             if video_type == 'original':
                 self.original_clips.append({'filename': filename, 'video_id': video_id, 'label': pain_level})
                 if video_id not in self.video_id_labels:
                     self.video_id_labels[video_id] = pain_level
             elif video_type == 'aug':
                 self.augmented_clips.append({'filename': filename, 'video_id': video_id})
             else:
                 print(f"Warning: Unknown video_type '{video_type}' for clip {filename}. Skipping.")

             processed_files.add(filename)

        print(f"Finished processing metadata:")
        print(f"  - Original clips found: {len(self.original_clips)}")
        print(f"  - Augmented clips found: {len(self.augmented_clips)}")
        print(f"  - Unique video IDs found: {len(self.video_id_labels)}")
        if clips_missing_info > 0: print(f"  - Clips skipped (missing info): {clips_missing_info}")
        if clips_missing_label > 0: print(f"  - Clips skipped (missing label): {clips_missing_label}")

        if not self.original_clips:
            raise ValueError("No original clips found after processing metadata. Cannot create splits.")
        if not self.video_id_labels:
             raise ValueError("No unique video IDs found. Cannot create splits.")

        print("SyracuseDataModule setup complete.")

    def train_dataloader(self):
        if not self.train_dataset:
            self.setup(stage='fit')
        if not self.train_dataset:
             raise RuntimeError("Train dataset not initialized after setup.")
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
        if not self.val_dataset:
            raise RuntimeError("Validation dataset not initialized after setup.")
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
        if not self.test_dataset:
             raise RuntimeError("Test dataset not initialized after setup.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
