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


class BioVidDataModule(LightningDataModule):

    def __init__(self, root_dir: str,
        load_raw: bool,
        task: str,
        num_classes: int,
        batch_size: int,
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
