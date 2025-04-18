import os
from abc import ABC, abstractmethod
from itertools import islice
from typing import Optional

import ffmpeg
import numpy as np
import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from marlin_pytorch.util import read_video, padding_video
from util.misc import sample_indexes, read_text, read_json

import random


class BioVidBase(LightningDataModule, ABC):

    def __init__(self, data_root: str, split: str, task: str, num_classes: int, data_ratio: float = 1.0, take_num: int = None):
        super().__init__()
        self.data_root = data_root
        self.split = split
        assert task in ("binary", "multiclass", "regression")
        self.task = task
        self.num_classes = num_classes
        self.take_num = take_num
        #The name_list is populated using the read_text function to read a text file located in data_root (e.g., train.txt, val.txt, or test.txt).
        #The file is expected to contain names of videos, separated by newlines. Empty lines are filtered out.
        self.name_list = list(
            filter(lambda x: x != "", read_text(os.path.join(data_root, f"{self.split}.txt")).split("\n")))
        self.metadata = read_json(os.path.join(data_root, "biovid_info.json"))

        if data_ratio < 1.0:
            self.name_list = self.name_list[:int(len(self.name_list) * data_ratio)]
        if take_num is not None:
            self.name_list = self.name_list[:self.take_num]

        print(f"Dataset {self.split} has {len(self.name_list)} videos")

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    def __len__(self):
        return len(self.name_list)


# for fine-tuning
class BioVidFT(BioVidBase):

    def __init__(self,
        root_dir: str,
        split: str,
        task: str,
        num_classes: int,
        clip_frames: int,
        temporal_sample_rate: int,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None
    ):
        super().__init__(root_dir, split, task, num_classes, data_ratio, take_num)
        #clip_frames: The number of frames to sample from the video.
        self.clip_frames = clip_frames
        #temporal_sample_rate: The rate at which to sample frames from the video.(e.g., 2 means take every 2nd frame)
        self.temporal_sample_rate = temporal_sample_rate

    def __getitem__(self, index: int):
        if self.task == "regression":
            y = self.metadata["clips"][self.name_list[index]]["attributes"]['multiclass']
        else:
            if self.task == "multiclass":
                num_classes = str(self.num_classes)
                # print(self.metadata["clips"][self.name_list[index]]["attributes"][self.task])
                y = self.metadata["clips"][self.name_list[index]]["attributes"][self.task][num_classes]
                # print(f"y value is {y} and data type is {type(y)}")
            else:
                y = self.metadata["clips"][self.name_list[index]]["attributes"][self.task]
        if isinstance(y, str):
            try:
                # Convert to float first, then to int
                y = int(float(y))
            except ValueError:
                print(f"Warning: Could not convert y to int: {y}")
                # y = -1  # Set to a default value or handle error appropriately

        # print(f"the task is {self.task}")
        # print(f"the y in dataloarder is {y}")
        # how to fetch the video and its label may be different from the CelebV-HQ dataset.
        video_path = os.path.join(self.data_root,  self.name_list[index])
        #todo: implement the __getitem__ method, which should return the video and its label.
        #how to fetch the video and its label?
        # print(f"Probing video file: {video_path}")  # Add this line before the probe call

        # probe = ffmpeg.probe(video_path)["streams"][0]
        try:
            probe = ffmpeg.probe(video_path)["streams"][0]
        except ffmpeg._run.Error as e:
            print(f"Error probing video: {video_path}")
            print(f"FFmpeg error: {e.stderr.decode()}")
        n_frames = int(probe["nb_frames"])

        if n_frames <= self.clip_frames:
            try:
                print(f"Reading video: {video_path}")
                print(f"Number of frames: {n_frames}")
                video = read_video(video_path, channel_first=True).video / 255
            except Exception as e:
                print(f"Error reading video: {video_path}")
                print(f"FFmpeg error: {e.stderr.decode()}")

            # video, audio = read_video(video_path, channel_first=True)
            # video = video / 255.0  # Normalize to [0, 1]
            # video = read_video(video_path, channel_first=True).video / 255
            # pad frames to 16
            # Check the shape of the video tensor
            # if video.ndim == 3:  # Shape: (C, H, W)
            #     video = video.unsqueeze(0)  # Add a dimension for T: (1, C, H, W)
            #
            # elif video.ndim != 4:
            #     raise ValueError(f"Unexpected video shape: {video.shape}")
            video = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
            video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
            return video, torch.tensor(y, dtype=torch.long)
        elif n_frames <= self.clip_frames * self.temporal_sample_rate:
            # reset a lower temporal sample rate
            sample_rate = n_frames // self.clip_frames
        else:
            sample_rate = self.temporal_sample_rate
        # sample frames
        video_indexes = sample_indexes(n_frames, self.clip_frames, sample_rate)
        reader = torchvision.io.VideoReader(video_path)
        fps = reader.get_metadata()["video"]["fps"][0]
        reader.seek(video_indexes[0].item() / fps, True)
        frames = []
        for frame in islice(reader, 0, self.clip_frames * sample_rate, sample_rate):
            frames.append(frame["data"])
        video = torch.stack(frames) / 255  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        assert video.shape[1] == self.clip_frames, video_path
        # print(f"y value before returning is {y}")
        # return video, torch.tensor(y, dtype=torch.long).bool()
        return video, torch.tensor(y, dtype=torch.long)


# For linear probing
class BioVidLP(BioVidBase):

    def __init__(self, root_dir: str,
        feature_dir: str,
        split: str,
        task: str,
        num_classes: int,
        temporal_reduction: str,
        data_ratio: float = 1.0,
        take_num: Optional[int] = None
    ):
        super().__init__(root_dir, split, task, num_classes,data_ratio, take_num)
        self.feature_dir = feature_dir
        self.temporal_reduction = temporal_reduction

    def __getitem__(self, index: int):
        try:
            # Get the name without the extension
            name_without_extension = os.path.splitext(self.name_list[index])[0]
            feat_path = os.path.join(self.data_root, self.feature_dir, name_without_extension + ".npy")
        except FileNotFoundError:
            print(f"File not found: {self.name_list[index]}")
            # feat_path = os.path.join(self.data_root, self.feature_dir, self.name_list[index] + ".npy")
        x = torch.from_numpy(np.load(feat_path)).float()
        # print(f"Loaded feature shape: {x.shape}")
        target_clips = 4

        if x.shape[0] > target_clips:
            # If more clips than needed, truncate
            x = x[:target_clips]
        elif x.shape[0] < target_clips:
            # If fewer clips than needed, pad with zeros
            padding = torch.zeros(target_clips - x.shape[0], *x.shape[1:], device=x.device)
            x = torch.cat([x, padding], dim=0)

        if x.size(0) == 0:
            x = torch.zeros(1, 768, dtype=torch.float32)

        if self.temporal_reduction == "mean":
            x = x.mean(dim=0)
        elif self.temporal_reduction == "max":
            x = x.max(dim=0)[0]
        elif self.temporal_reduction == "min":
            x = x.min(dim=0)[0]
        elif self.temporal_reduction == "none":
            # Keep sequence dimension intact
            pass
        else:
            raise ValueError(self.temporal_reduction)

        if self.task == "regression":
            y = self.metadata["clips"][self.name_list[index]]["attributes"]['multiclass']['5']
        else:
            if self.task == "multiclass":
                # clip = self.metadata["clips"][self.name_list[index]]
                # print(f"Clip: {clip}, Type: {type(clip)}")
                #
                # attributes = clip["attributes"]
                # print(f"Attributes: {attributes}, Type: {type(attributes)}")
                #
                # task_value = attributes[self.task]
                # print(f"Task Value: {task_value}, Type: {type(task_value)}")

                num_classes = str(self.num_classes)
                # print(f"type of num_classes is {type(num_classes)}")
                # print(self.metadata["clips"][self.name_list[index]]["attributes"][self.task])
                try:
                    y = self.metadata["clips"][self.name_list[index]]["attributes"][self.task][num_classes]
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Clip: {self.metadata['clips'][self.name_list[index]]}")
                # print(f"y value is {y} and data type is {type(y)}")
            else:
                y = self.metadata["clips"][self.name_list[index]]["attributes"][self.task]
        if isinstance(y, str):
            try:
                # Convert to float first, then to int
                y = int(float(y))
            except ValueError:
                print(f"Warning: Could not convert y to int: {y}")
                # y = -1  # Set to a default value or handle error appropriately


        return x, torch.tensor(y, dtype=torch.long)


class AugmentedBioVidLP(BioVidLP):
    """
    Extends BioVidLP dataset with feature-level augmentations.
    Maintains same interface and functionality as the original class.
    """

    def __init__(self, root_dir: str,
                 feature_dir: str,
                 split: str,
                 task: str,
                 num_classes: int,
                 temporal_reduction: str,
                 data_ratio: float = 1.0,
                 take_num: Optional[int] = None,
                 mixup_alpha: float = 0.2,
                 feature_noise_std: float = 0.01,
                 feature_dropout_p: float = 0.05
                 ):
        super().__init__(root_dir, feature_dir, split, task, num_classes,
                         temporal_reduction, data_ratio, take_num)

        # Only apply augmentations during training
        self.apply_augmentation = split == "train"

        # Augmentation parameters
        self.mixup_alpha = mixup_alpha
        self.feature_noise_std = feature_noise_std
        self.feature_dropout_p = feature_dropout_p

        # Cache for class indices (populated on first use)
        self.class_indices = None

    def _build_class_indices(self):
        """Build a dictionary mapping class labels to sample indices"""
        self.class_indices = {}
        for idx in range(len(self.name_list)):
            name = self.name_list[idx]
            if self.task == "regression":
                y = self.metadata["clips"][name]["attributes"]['multiclass']['5']
            else:
                if self.task == "multiclass":
                    num_classes = str(self.num_classes)
                    y = self.metadata["clips"][name]["attributes"][self.task][num_classes]
                else:
                    y = self.metadata["clips"][name]["attributes"][self.task]

            if isinstance(y, str):
                y = int(float(y))

            if y not in self.class_indices:
                self.class_indices[y] = []
            self.class_indices[y].append(idx)

    def get_same_class_sample(self, idx, y):
        """Get a sample index from the same class"""
        if self.class_indices is None:
            self._build_class_indices()

        same_class_indices = [i for i in self.class_indices[y] if i != idx]
        if same_class_indices:
            return random.choice(same_class_indices)
        return idx  # Fallback to self if no other samples in class

    # def apply_feature_mixup(self, features, idx, y):
    #     """Apply mixup at feature level with another sample from same class"""
    #     # Get a sample from the same class
    #     mix_idx = self.get_same_class_sample(idx, y)
    #
    #     # Load the mix partner's features
    #     mix_name = os.path.splitext(self.name_list[mix_idx])[0]
    #     mix_path = os.path.join(self.data_root, self.feature_dir, mix_name + ".npy")
    #     mix_features = torch.from_numpy(np.load(mix_path)).float()
    #
    #     # Apply temporal reduction to mix partner if needed
    #     if self.temporal_reduction == "mean":
    #         mix_features = mix_features.mean(dim=0)
    #     elif self.temporal_reduction == "max":
    #         mix_features = mix_features.max(dim=0)[0]
    #     elif self.temporal_reduction == "min":
    #         mix_features = mix_features.min(dim=0)[0]
    #     elif self.temporal_reduction != "none":
    #         raise ValueError(f"Unsupported reduction: {self.temporal_reduction}")
    #
    #     # Generate mixup coefficient
    #     lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
    #
    #     # Apply mixup
    #     if features.dim() == mix_features.dim():
    #         mixed = lam * features + (1 - lam) * mix_features
    #         return mixed
    #     else:
    #         # Handle dimension mismatch (should not happen with proper temporal_reduction)
    #         print(f"Warning: Feature dimension mismatch: {features.shape} vs {mix_features.shape}")
    #         return features

    def apply_feature_mixup(self, features, idx, y):
        """Apply mixup at feature level with another sample from same class"""
        # Get a sample from the same class
        mix_idx = self.get_same_class_sample(idx, y)

        # Load the mix partner's features
        mix_name = os.path.splitext(self.name_list[mix_idx])[0]
        mix_path = os.path.join(self.data_root, self.feature_dir, mix_name + ".npy")
        mix_features = torch.from_numpy(np.load(mix_path)).float()

        # Apply temporal reduction if needed
        if self.temporal_reduction == "mean":
            mix_features = mix_features.mean(dim=0)
            features = features.mean(dim=0) if features.dim() > 1 else features
        elif self.temporal_reduction == "max":
            mix_features = mix_features.max(dim=0)[0]
            features = features.max(dim=0)[0] if features.dim() > 1 else features
        elif self.temporal_reduction == "min":
            mix_features = mix_features.min(dim=0)[0]
            features = features.min(dim=0)[0] if features.dim() > 1 else features
        elif self.temporal_reduction == "none":
            # Standardize both sequences to the target length
            target_clips = 4  # Must match what's used in __getitem__

            # Standardize mix_features
            if mix_features.shape[0] > target_clips:
                mix_features = mix_features[:target_clips]
            elif mix_features.shape[0] < target_clips:
                padding = torch.zeros(target_clips - mix_features.shape[0], *mix_features.shape[1:],
                                      dtype=mix_features.dtype)
                mix_features = torch.cat([mix_features, padding], dim=0)

            # Ensure features is also standardized (should already be done in __getitem__, but double-check)
            if features.shape[0] > target_clips:
                features = features[:target_clips]
            elif features.shape[0] < target_clips:
                padding = torch.zeros(target_clips - features.shape[0], *features.shape[1:], dtype=features.dtype)
                features = torch.cat([features, padding], dim=0)

        # Verify shapes match before mixup
        if features.shape != mix_features.shape:
            print(
                f"Warning: Shape mismatch after standardization: features {features.shape}, mix_features {mix_features.shape}")
            print(f"Defaulting to original features")
            return features

        # Generate mixup coefficient
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        # Apply mixup
        mixed = lam * features + (1 - lam) * mix_features
        return mixed

    def apply_feature_noise(self, features):
        """Add Gaussian noise to features"""
        noise = torch.randn_like(features) * self.feature_noise_std
        return features + noise

    def apply_feature_dropout(self, features):
        """Randomly zero out feature dimensions"""
        mask = torch.bernoulli(torch.ones_like(features) * (1 - self.feature_dropout_p))
        return features * mask

    def __getitem__(self, idx):
        """Get a sample with optional feature augmentations"""
        # Load features using parent class implementation
        name_without_extension = os.path.splitext(self.name_list[idx])[0]
        feat_path = os.path.join(self.data_root, self.feature_dir, name_without_extension + ".npy")

        try:
            features = torch.from_numpy(np.load(feat_path)).float()
        except FileNotFoundError:
            print(f"File not found: {feat_path}")
            # Use a small default tensor if file is missing
            features = torch.zeros(1, 768, dtype=torch.float32)

        # Standardize sequence length before augmentation
        target_clips = 4  # Adjust this to your needs

        if features.shape[0] > target_clips:
            # If more clips than needed, truncate
            features = features[:target_clips]
        elif features.shape[0] < target_clips:
            # If fewer clips than needed, pad with zeros
            padding = torch.zeros(target_clips - features.shape[0], *features.shape[1:], device=features.device)
            features = torch.cat([features, padding], dim=0)

        # Apply temporal reduction
        if self.temporal_reduction == "mean":
            features = features.mean(dim=0)
        elif self.temporal_reduction == "max":
            features = features.max(dim=0)[0]
        elif self.temporal_reduction == "min":
            features = features.min(dim=0)[0]
        elif self.temporal_reduction != "none":
            raise ValueError(f"Unsupported reduction: {self.temporal_reduction}")

        # Get label
        if self.task == "regression":
            y = self.metadata["clips"][self.name_list[idx]]["attributes"]['multiclass']['5']
        else:
            if self.task == "multiclass":
                num_classes = str(self.num_classes)
                y = self.metadata["clips"][self.name_list[idx]]["attributes"][self.task][num_classes]
            else:
                y = self.metadata["clips"][self.name_list[idx]]["attributes"][self.task]

        if isinstance(y, str):
            try:
                y = int(float(y))
            except ValueError:
                print(f"Warning: Could not convert y to int: {y}")

        # Apply augmentations during training
        if self.apply_augmentation:
            # 1. Feature mixup (with probability 0.5)
            if random.random() < 0.5:
                features = self.apply_feature_mixup(features, idx, y)

            # 2. Feature noise (with probability 0.3)
            if random.random() < 0.3:
                features = self.apply_feature_noise(features)

            # 3. Feature dropout (with probability 0.2)
            if random.random() < 0.2:
                features = self.apply_feature_dropout(features)

        return features, torch.tensor(y, dtype=torch.long)



class BioVidDataModule(LightningDataModule):

    def __init__(self, root_dir: str,
        load_raw: bool,
        task: str,
        num_classes: int,
        batch_size: int,
        augmentation: bool = False,
        num_workers: int = 0,
        clip_frames: int = None,
        temporal_sample_rate: int = None,
        feature_dir: str = None,
        temporal_reduction: str = "mean",
        data_ratio: float = 1.0,
        take_train: Optional[int] = None,
        take_val: Optional[int] = None,
        take_test: Optional[int] = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.task = task
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate
        self.feature_dir = feature_dir
        self.temporal_reduction = temporal_reduction
        self.load_raw = load_raw
        self.data_ratio = data_ratio
        self.take_train = take_train
        self.take_val = take_val
        self.take_test = take_test
        self.augmentation = augmentation

        if load_raw:
            assert clip_frames is not None
            assert temporal_sample_rate is not None
        else:
            assert feature_dir is not None
            assert temporal_reduction is not None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.load_raw:
            self.train_dataset = BioVidFT(self.root_dir, "train", self.task, self.num_classes, self.clip_frames,
                self.temporal_sample_rate, self.data_ratio, self.take_train)
            self.val_dataset = BioVidFT(self.root_dir, "val", self.task, self.num_classes, self.clip_frames,
                self.temporal_sample_rate, self.data_ratio, self.take_val)
            self.test_dataset = BioVidFT(self.root_dir, "test", self.task, self.num_classes, self.clip_frames,
                self.temporal_sample_rate, 1.0, self.take_test)
        else:
            if self.augmentation:
                self.train_dataset = AugmentedBioVidLP(
                    self.root_dir, self.feature_dir, "train", self.task, self.num_classes,
                    self.temporal_reduction, self.data_ratio, self.take_train,
                    mixup_alpha=0.2,  # Adjust these parameters as needed
                    feature_noise_std=0.01,
                    feature_dropout_p=0.05
                )
            else:
                self.train_dataset = BioVidLP(self.root_dir, self.feature_dir, "train", self.task, self.num_classes,
                    self.temporal_reduction, self.data_ratio, self.take_train)
            self.val_dataset = BioVidLP(self.root_dir, self.feature_dir, "val", self.task, self.num_classes,
                self.temporal_reduction, self.data_ratio, self.take_val)
            self.test_dataset = BioVidLP(self.root_dir, self.feature_dir, "test", self.task, self.num_classes,
                self.temporal_reduction, 1.0, self.take_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
