import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict
from sklearn.model_selection import train_test_split

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


class SyracuseLP(BioVidBase):
    """
    Linear probing dataset class for Syracuse dataset.
    Adapted to use normalization and metadata structure similar to MarlinFeatures.
    """

    def __init__(self, root_dir: str,
        feature_dir: str,
        split: str,
        task: str,
        num_classes: int,
        temporal_reduction: str,
        name_list: List[str],
        metadata: Dict[str, Dict],
        marlin_base_dir: str,          # Added: Path needed for MarlinFeatures
        val_split_ratio: float = 0.15, # Ratio of *videos* for validation
        test_split_ratio: float = 0.15, # Ratio of *videos* for testing
        random_state: int = 42,        # For reproducibility of splits
    ):
        super().__init__(root_dir, split, task, num_classes)
        self.feature_dir = feature_dir
        self.temporal_reduction = temporal_reduction
        self.name_list = name_list
        self.metadata = metadata
        print(f"SyracuseLP Dataset '{split}' initialized with {len(self.name_list)} clips.")
        if not self.metadata:
             print(f"Warning: Metadata dictionary for {split} dataset is empty!")
        elif not self.name_list:
             print(f"Warning: Name list for {split} dataset is empty!")
        self.test_split_ratio = test_split_ratio
        self.random_state = random_state
        self.num_workers = num_workers
        self.marlin_base_dir = marlin_base_dir # Store the base dir

        # --- Data containers (will be populated in setup) ---
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.all_metadata = {} # To store metadata for all clips

        # --- Initialize MarlinFeatures --- 
        try:
            print(f"Initializing MarlinFeatures with base directory: {self.marlin_base_dir}")
            self.marlin_features = MarlinFeatures(self.marlin_base_dir)
            print("MarlinFeatures initialized successfully.")
        except FileNotFoundError as e:
            print(f"Error initializing MarlinFeatures: {e}")
            print("Please ensure the marlin_base_dir is correct and contains clips_json.json.")
            # Decide if this should be a fatal error
            # raise RuntimeError(f"Failed to initialize MarlinFeatures: {e}")
            self.marlin_features = None # Indicate initialization failure
        except Exception as e:
             print(f"An unexpected error occurred during MarlinFeatures initialization: {e}")
             self.marlin_features = None

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
        feat_path = os.path.join(self.data_root, self.feature_dir, filename)

        clip_metadata = self.metadata.get(filename)

        if clip_metadata is None:
             print(f"Error: Metadata not found for clip: {filename} in provided metadata dict.")
             dummy_features = torch.zeros((4 if self.temporal_reduction == "none" else 1, 768), dtype=torch.float32)
             dummy_label = torch.tensor(-1, dtype=torch.long)
             return dummy_features, dummy_label

        try:
            raw_features = np.load(feat_path)
        except FileNotFoundError:
            print(f"Error: Feature file not found: {feat_path}")
            dummy_features = torch.zeros((4 if self.temporal_reduction == "none" else 1, 768), dtype=torch.float32)
            dummy_label = torch.tensor(-1, dtype=torch.long)
            return dummy_features, dummy_label
        except Exception as e:
            print(f"Error loading feature file {feat_path}: {e}")
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

        try:
            meta_info = clip_metadata.get('meta_info')
            if meta_info is None:
                raise KeyError("'meta_info' not found in metadata for clip: " + filename)

            if self.task == "regression":
                y_value = meta_info.get('pain_level')
                if y_value is None: raise KeyError("'pain_level' not found")
                y = torch.tensor(float(y_value), dtype=torch.float32)
            elif self.task == "multiclass":
                class_key = f"class_{self.num_classes}"
                y_value = meta_info.get(class_key)
                if y_value is None: raise KeyError(f"'{class_key}' not found")
                y = torch.tensor(int(float(y_value)), dtype=torch.long)
            elif self.task == "binary":
                outcome = meta_info.get('outcome')
                if outcome is None: raise KeyError("'outcome' not found")
                y_value = 1 if outcome == 'positive' else 0
                y = torch.tensor(y_value, dtype=torch.long)
            else:
                 raise ValueError(f"Unsupported task type: {self.task}")

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

        return x, y


class SyracuseDataModule(LightningDataModule):
    """
    DataModule for Syracuse dataset, similar to BioVidDataModule.
    """

    def __init__(self, root_dir: str,
        task: str,
        num_classes: int,
        batch_size: int,
        feature_dir: str,
        marlin_base_dir: str,          # Moved non-default arg before defaults
        temporal_reduction: str = "mean",
        val_split_ratio: float = 0.15,
        test_split_ratio: float = 0.15,
        random_state: int = 42,
        num_workers: int = 0
    ):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.feature_dir = feature_dir
        self.temporal_reduction = temporal_reduction
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.random_state = random_state
        self.num_workers = num_workers
        self.marlin_base_dir = marlin_base_dir # Store the base dir

        # --- Data containers (will be populated in setup) ---
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.all_metadata = {} # To store metadata for all clips

        # --- Initialize MarlinFeatures --- 
        try:
            print(f"Initializing MarlinFeatures with base directory: {self.marlin_base_dir}")
            self.marlin_features = MarlinFeatures(self.marlin_base_dir)
            print("MarlinFeatures initialized successfully.")
        except FileNotFoundError as e:
            print(f"Error initializing MarlinFeatures: {e}")
            print("Please ensure the marlin_base_dir is correct and contains clips_json.json.")
            # Decide if this should be a fatal error
            # raise RuntimeError(f"Failed to initialize MarlinFeatures: {e}")
            self.marlin_features = None # Indicate initialization failure
        except Exception as e:
             print(f"An unexpected error occurred during MarlinFeatures initialization: {e}")
             self.marlin_features = None

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

        original_clips = []
        augmented_clips = []
        video_id_labels = {}

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

             label_key = 'pain_level' if self.task == 'regression' else f'class_{self.num_classes}'
             label_value = meta_info.get(label_key)

             if label_value is None:
                  fallback_keys = [f'class_{n}' for n in [3, 4, 5] if f'class_{n}' != label_key] + ['pain_level']
                  for key in fallback_keys:
                      label_value = meta_info.get(key)
                      if label_value is not None:
                          break
                  if label_value is None:
                      clips_missing_label += 1
                      continue

             try:
                label = int(float(label_value))
             except (ValueError, TypeError):
                 print(f"Warning: Could not convert label '{label_value}' to int for clip {filename}. Skipping.")
                 clips_missing_label += 1
                 continue

             if video_type == 'original':
                 original_clips.append({'filename': filename, 'video_id': video_id, 'label': label})
                 if video_id not in video_id_labels:
                     video_id_labels[video_id] = label
             elif video_type == 'aug':
                 augmented_clips.append({'filename': filename, 'video_id': video_id})
             else:
                 print(f"Warning: Unknown video_type '{video_type}' for clip {filename}. Skipping.")

             processed_files.add(filename)

        print(f"Finished processing metadata:")
        print(f"  - Original clips found: {len(original_clips)}")
        print(f"  - Augmented clips found: {len(augmented_clips)}")
        print(f"  - Unique video IDs found: {len(video_id_labels)}")
        if clips_missing_info > 0: print(f"  - Clips skipped (missing info): {clips_missing_info}")
        if clips_missing_label > 0: print(f"  - Clips skipped (missing label): {clips_missing_label}")

        if not original_clips:
            raise ValueError("No original clips found after processing metadata. Cannot create splits.")
        if not video_id_labels:
             raise ValueError("No unique video IDs found. Cannot create splits.")

        unique_video_ids = list(video_id_labels.keys())
        video_labels = [video_id_labels[vid] for vid in unique_video_ids]

        n_videos = len(unique_video_ids)
        n_test = int(self.test_split_ratio * n_videos)
        n_val = int(self.val_split_ratio * n_videos)
        n_train = n_videos - n_test - n_val

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
             raise ValueError(f"Split ratios result in 0 videos for one or more sets (Train: {n_train}, Val: {n_val}, Test: {n_test}). Adjust ratios.")

        print(f"Splitting {n_videos} video IDs into Train ({n_train}), Val ({n_val}), Test ({n_test}) sets.")

        try:
            train_val_ids, test_ids = train_test_split(
                unique_video_ids,
                test_size=n_test,
                stratify=[video_id_labels[vid] for vid in unique_video_ids],
                random_state=self.random_state
            )

            val_size_adjusted = n_val / (n_train + n_val)
            train_ids, val_ids = train_test_split(
                train_val_ids,
                test_size=val_size_adjusted,
                stratify=[video_id_labels[vid] for vid in train_val_ids],
                random_state=self.random_state
            )
        except ValueError as e:
             print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
             train_val_ids, test_ids = train_test_split(unique_video_ids, test_size=n_test, random_state=self.random_state)
             val_size_adjusted = n_val / (n_train + n_val)
             train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_adjusted, random_state=self.random_state)

        train_ids_set = set(train_ids)
        val_ids_set = set(val_ids)
        test_ids_set = set(test_ids)

        print(f"Video ID split complete: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

        train_names = []
        val_names = []
        test_names = []

        for clip in original_clips:
            vid = clip['video_id']
            fname = clip['filename']
            if vid in train_ids_set:
                train_names.append(fname)
            elif vid in val_ids_set:
                val_names.append(fname)
            elif vid in test_ids_set:
                test_names.append(fname)

        num_aug_added = 0
        for clip in augmented_clips:
            vid = clip['video_id']
            fname = clip['filename']
            if vid in train_ids_set:
                train_names.append(fname)
                num_aug_added += 1

        print(f"Assigned filenames to splits:")
        print(f"  - Train: {len(train_names)} clips ({num_aug_added} augmented)")
        print(f"  - Val:   {len(val_names)} clips (original only)")
        print(f"  - Test:  {len(test_names)} clips (original only)")

        if not train_names: print("Warning: Training set is empty after assigning filenames.")
        if not val_names: print("Warning: Validation set is empty after assigning filenames.")
        if not test_names: print("Warning: Test set is empty after assigning filenames.")

        print("Instantiating SyracuseLP datasets for each split...")
        common_args = {
            "root_dir": self.root_dir,
            "feature_dir": self.feature_dir,
            "task": self.task,
            "num_classes": self.num_classes,
            "temporal_reduction": self.temporal_reduction,
            "metadata": self.all_metadata
        }

        self.train_dataset = SyracuseLP(split="train", name_list=train_names, **common_args)
        self.val_dataset = SyracuseLP(split="val", name_list=val_names, **common_args)
        self.test_dataset = SyracuseLP(split="test", name_list=test_names, **common_args)

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
