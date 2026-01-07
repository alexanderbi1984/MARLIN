import os
import random
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

class VideoDataset(Dataset):
    """
    Dataset that loads frames from a directory of image files (frames).

    Supports two modes:
    - video-level: each item is one sampled clip (num_frames) from the whole video folder.
    - clip-level: each video is sliced into many sliding-window clips; each clip inherits the video label.
    """
    def __init__(
        self, 
        video_paths: list[str], 
        labels: list[int], 
        num_frames: int = 16, 
        resize: int = 256, 
        crop_size: int = 224, 
        is_train: bool = True,
        transform=None,
        aug_paths_list: list[list[str]] = None,
        aug_ratio: float = 0.0,
        clip_level: bool = False,
        clip_len_frames: int = 150,
        clip_stride_frames: int = 15,
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.resize = resize
        self.crop_size = crop_size
        self.is_train = is_train
        self.transform = transform
        self.aug_paths_list = aug_paths_list if aug_paths_list is not None else [[] for _ in range(len(video_paths))]
        self.aug_ratio = aug_ratio
        self.clip_level = bool(clip_level)
        self.clip_len_frames = max(1, int(clip_len_frames))
        self.clip_stride_frames = max(1, int(clip_stride_frames))

        # Cache per-folder frame file lists to avoid repeated glob/sort cost.
        self._frames_cache: dict[str, list[str]] = {}

        # If clip_level is enabled, pre-materialize (video_idx, clip_start, clip_idx) items.
        self._clip_items: list[tuple[int, int, int]] = []
        if self.clip_level:
            for vid_idx, path in enumerate(self.video_paths):
                files = self._list_frame_files(path)
                total = len(files)
                if total <= 0:
                    starts = [0]
                elif total <= self.clip_len_frames:
                    starts = [0]
                else:
                    max_start = max(0, total - self.clip_len_frames)
                    starts = list(range(0, max_start + 1, self.clip_stride_frames))
                    # Ensure we include the tail clip so the end of video is covered.
                    if starts and starts[-1] != max_start:
                        starts.append(max_start)
                for clip_idx, clip_start in enumerate(starts):
                    self._clip_items.append((vid_idx, int(clip_start), int(clip_idx)))

    def __len__(self):
        if self.clip_level:
            return len(self._clip_items)
        return len(self.video_paths)

    def _list_frame_files(self, path: str) -> list[str]:
        if path in self._frames_cache:
            return self._frames_cache[path]
        if not os.path.isdir(path):
            self._frames_cache[path] = []
            return []
        files = sorted(
            glob.glob(os.path.join(path, "*.png")) +
            glob.glob(os.path.join(path, "*.jpg")) +
            glob.glob(os.path.join(path, "*.jpeg")) +
            glob.glob(os.path.join(path, "*.bmp"))
        )
        self._frames_cache[path] = files
        return files

    def _load_frames_by_indices(self, files: list[str], indices: np.ndarray) -> list[Image.Image]:
        frames: list[Image.Image] = []
        for i in indices:
            ii = int(i)
            ii = max(0, min(ii, len(files) - 1))
            try:
                img = Image.open(files[ii]).convert('RGB')
                frames.append(img)
            except Exception as e:
                print(f"Error loading frame {files[ii]}: {e}")
                frames.append(frames[-1] if frames else Image.new('RGB', (self.resize, self.resize)))
        # Ensure exactly num_frames
        if len(frames) > self.num_frames:
            frames = frames[:self.num_frames]
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.new('RGB', (self.resize, self.resize)))
        return frames

    def _load_video_level_frames(self, path: str) -> list[Image.Image]:
        """
        Loads frames from a directory containing image files.
        Assumes frames are named in a sortable way (e.g., face_00001.png).
        """
        files = self._list_frame_files(path)
        if not files:
            return [Image.new('RGB', (self.resize, self.resize))] * self.num_frames
        total_frames = len(files)
        # Uniform sampling
        if total_frames <= self.num_frames:
            # Pad with last frame if not enough
            indices = list(range(total_frames))
            if total_frames < self.num_frames:
                indices += [total_frames - 1] * (self.num_frames - total_frames)
        else:
            # Uniformly sample indices
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        return self._load_frames_by_indices(files, np.asarray(indices))

    def _load_clip_level_frames(self, path: str, clip_start: int) -> list[Image.Image]:
        files = self._list_frame_files(path)
        if not files:
            return [Image.new('RGB', (self.resize, self.resize))] * self.num_frames
        total = len(files)
        max_start = max(0, total - 1)
        clip_start = max(0, min(int(clip_start), max_start))
        clip_end = min(total, clip_start + self.clip_len_frames)
        if clip_end <= clip_start:
            clip_end = min(total, clip_start + 1)
        # Sample num_frames indices uniformly within [clip_start, clip_end)
        indices = np.linspace(clip_start, clip_end - 1, self.num_frames, dtype=int)
        return self._load_frames_by_indices(files, indices)

    def __getitem__(self, idx):
        if self.clip_level:
            vid_idx, clip_start, clip_idx = self._clip_items[idx]
            original_path = self.video_paths[vid_idx]
            video_base = os.path.basename(original_path)
            path = original_path
            # Apply swap-face augmentation at the *view* level during training.
            if self.is_train and self.aug_ratio > 0 and self.aug_paths_list[vid_idx]:
                if random.random() < self.aug_ratio:
                    aug_path = random.choice(self.aug_paths_list[vid_idx])
                    if os.path.isdir(aug_path):
                        path = aug_path
            label = self.labels[vid_idx]
            frames = self._load_clip_level_frames(path, clip_start=clip_start)
            sample_id = f"{video_base}_clip_{clip_idx:04d}"
        else:
            path = self.video_paths[idx]
            # Apply Augmentation (Swap to augmented view) if training
            if self.is_train and self.aug_ratio > 0 and self.aug_paths_list[idx]:
                if random.random() < self.aug_ratio:
                    # Randomly select one augmented view
                    aug_path = random.choice(self.aug_paths_list[idx])
                    if os.path.isdir(aug_path):
                        path = aug_path
            label = self.labels[idx]
            frames = self._load_video_level_frames(path)
            sample_id = os.path.basename(self.video_paths[idx])
        
        # Deterministic transform for validation, Random for training but consistent across frames
        
        # Deterministic transform for validation
        if not self.is_train:
            frames = [F.resize(img, self.resize) for img in frames]
            frames = [F.center_crop(img, self.crop_size) for img in frames]
        else:
            # Training Augmentation
            # 1. Resize first
            frames = [F.resize(img, self.resize) for img in frames]
            
            # 2. Random Crop (Video Consistent)
            i, j, h, w = torch.randint(0, self.resize - self.crop_size + 1, (2,)).tolist() + [self.crop_size, self.crop_size]
            frames = [F.crop(img, i, j, h, w) for img in frames]
            
            # 3. Random Horizontal Flip (Video Consistent)
            if random.random() > 0.5:
                frames = [F.hflip(img) for img in frames]
                
            # 4. AugMix (Per Spec)
            try:
                from torchvision.transforms import AugMix
                aug = AugMix()
                frames = [aug(img) for img in frames]
            except ImportError:
                pass
            
        # To Tensor and Normalize
        # Standard ImageNet mean/std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        tensors = [F.to_tensor(img) for img in frames]
        tensors = [F.normalize(t, mean, std) for t in tensors]
        
        # Stack: (T, C, H, W)
        video_tensor = torch.stack(tensors, dim=0)
        
        return video_tensor, torch.tensor(label, dtype=torch.long), sample_id
