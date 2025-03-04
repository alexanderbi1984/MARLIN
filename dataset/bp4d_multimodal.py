import os
from math import ceil
from typing import Collection, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from util.misc import sample_indexes


def load_image_safely(file_path, grayscale=False, img_size=224):
    """Safely load an image with fallback to small non-zero values."""
    # Convert to str if Path object
    file_path = str(file_path)

    if os.path.exists(file_path):
        if grayscale:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(file_path)

        if img is not None:
            return img
        else:
            print(f"Warning: Failed to load image {file_path}. Using small values instead.")
    else:
        print(f"Warning: Missing file {file_path}. Using small values instead.")

    # Return array with small non-zero values (e.g., 0.001 after normalization)
    # Using 1 here (before normalization) will result in ~0.004 after dividing by 255
    if grayscale:
        return np.ones((img_size, img_size), dtype=np.uint8) * 1
    else:
        return np.ones((img_size, img_size, 3), dtype=np.uint8) * 1


class BP4DMultiModal(Dataset):
    seg_groups = [
        [2, 4],  # right eye
        [3, 5],  # left eye
        [6],  # nose
        [7, 8, 9],  # mouth
        [10],  # hair
        [1],  # skin
        [0]  # background
    ]

    def __init__(self,
                 root_dir: str,
                 split: str,
                 clip_frames: int,  # T = 16
                 temporal_sample_rate: int,  # 2
                 patch_size: int,  # 16
                 tubelet_size: int,  # 2
                 mask_percentage_target: float = 0.8,  # 0.9
                 mask_strategy: str = "fasking",
                 take_num: Optional[int] = None
                 ):

        """	Arguments:
	•	root_dir: Path to the dataset.
	•	split: Either "train" or "val".
	•	clip_frames: Number of frames per clip.
    .   temporal_sample_rate: The step size between consecutive frames.
	•	patch_size: Size of spatial patches.
	•	tubelet_size: Defines temporal units for masking.
	•	mask_percentage_target: Percentage of patches to keep.
	•	mask_strategy: Specifies different masking strategies (e.g., fasking, tube, random).
	•	take_num: Limits the number of samples. like using only 1000 samples for training.
	"""
        self.img_size = 224
        self.root_dir = Path(root_dir)  # Convert to Path object
        self.clip_frames = clip_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.temporal_sample_rate = temporal_sample_rate
        self.mask_percentage_target = 1 - mask_percentage_target
        self.mask_unit_num = self.img_size // self.patch_size
        assert split in ("train", "val")

        self.patch_masking = True

        if mask_strategy == "fasking":  # Masking components first
            self.face_strategy = True
            self.face_masking_opposite = False
        elif mask_strategy == "fasking_opp":  # Masking background and skin first
            self.face_strategy = True
            self.face_masking_opposite = True
        elif mask_strategy in ("random", "tube", "frame"):  # Masking strategy from VideoMAE
            self.face_strategy = False
            self.face_masking_opposite = None
        else:
            raise ValueError("mask_strategy must be one of 'fasking', 'fasking_opp', 'random', 'tube' and 'frame'")

        self.mask_strategy = mask_strategy
        # Use Path for consistent path handling
        csv_path = self.root_dir / f"{split}_set.csv"
        self.metadata = pd.read_csv(csv_path)
        if take_num:
            self.metadata = self.metadata.iloc[:take_num]

    def __getitem__(self, index):
        """
        Retrieves and processes a multimodal video sample (RGB, Depth, Thermal) with patch-level channel mixing and masking.

        Args:
            index (int): Index of the video sample in the dataset.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                - Mixed multimodal video tensor of shape `(C=3, T, H, W)`, where:
                    - C: Number of mixed channels (3 after dropping 2).
                    - T: Number of frames in the clip.
                    - H, W: Image height and width.
                - Flattened boolean mask tensor indicating visible (`True`) and masked (`False`) regions.
                - Original RGB frames `(3, T, H, W)`, before mixing.
                - Original Depth frames `(1, T, H, W)`, before mixing.
                - Original Thermal frames `(1, T, H, W)`, before mixing.
        """

        # Load metadata and video file paths
        meta = self.metadata.iloc[index]

        # Ensure path is normalized for cross-platform compatibility
        meta_path = Path(str(meta.path).replace('\\', '/'))

        # Get texture directory path and file list
        texture_dir_path = self.root_dir / "Texture_crop_crop_images_DB" / meta_path

        # List all files in the directory
        files = sorted([f.name for f in texture_dir_path.iterdir() if f.is_file()])

        if len(files) < self.clip_frames:
            print(f"Warning: Not enough frames in {meta_path}. Skipping this sample.")

        indexes = self._sample_indexes(len(files))
        assert len(indexes) == self.clip_frames

        # Initialize video tensors
        video = torch.zeros(self.clip_frames, self.img_size, self.img_size, 5,
                            dtype=torch.float32)  # 5-channel (RGB-D-T)
        rgb_frames = torch.zeros(self.clip_frames, self.img_size, self.img_size, 3, dtype=torch.float32)
        depth_frames = torch.zeros(self.clip_frames, self.img_size, self.img_size, 1, dtype=torch.float32)
        thermal_frames = torch.zeros(self.clip_frames, self.img_size, self.img_size, 1, dtype=torch.float32)

        if self.patch_masking:
            masks = torch.zeros(self.clip_frames // self.tubelet_size,
                                self.mask_unit_num, self.mask_unit_num, dtype=torch.float32)
        else:
            masks = torch.zeros(self.clip_frames, self.img_size, self.img_size, dtype=torch.float32)

        if self.face_strategy:
            if self.face_masking_opposite:
                keep_queue = np.concatenate([np.random.permutation(5), [5, 6]])
            else:
                keep_queue = np.concatenate([[6, 5], np.random.permutation(5)])
        else:
            keep_queue = None

        for i in range(self.clip_frames):
            # Construct paths using Path objects for all modalities
            texture_file_path = self.root_dir / "Texture_crop_crop_images_DB" / meta_path / files[indexes[i]]
            depth_file_path = self.root_dir / "Depth_crop_crop_images_DB" / meta_path / files[indexes[i]]
            thermal_file_path = self.root_dir / "Thermal_crop_crop_images_DB" / meta_path / files[indexes[i]]

            # Load images with fallback to small non-zero values
            rgb_img = load_image_safely(
                texture_file_path,
                grayscale=False,
                img_size=self.img_size
            )
            depth_img = load_image_safely(
                depth_file_path,
                grayscale=True,
                img_size=self.img_size
            )
            thermal_img = load_image_safely(
                thermal_file_path,
                grayscale=True,
                img_size=self.img_size
            )

            # Process RGB
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) / 255.0

            # Process depth and thermal (ensuring they're proper shape)
            depth_img = np.expand_dims(depth_img, axis=-1) / 255.0  # Convert (H, W) → (H, W, 1)
            thermal_img = np.expand_dims(thermal_img, axis=-1) / 255.0  # Convert (H, W) → (H, W, 1)

            # Store the original frames before mixing
            rgb_frames[i] = torch.from_numpy(rgb_img)
            depth_frames[i] = torch.from_numpy(depth_img)
            thermal_frames[i] = torch.from_numpy(thermal_img)

            # Stack all modalities along the channel dimension (H, W, 5)
            multimodal_frame = np.concatenate([rgb_img, depth_img, thermal_img], axis=-1)

            # Store in video tensor (T, H, W, 5)
            video[i] = torch.from_numpy(multimodal_frame)

        # Get patch dimensions
        patch_h = self.img_size // self.patch_size  # Number of patches along height
        patch_w = self.img_size // self.patch_size  # Number of patches along width

        # Generate a random patch-wise channel selection map (each patch gets its own channel mix)
        patch_channel_map = np.zeros((patch_h, patch_w, 3), dtype=int)  # Stores selected channels per patch
        for ph in range(patch_h):
            for pw in range(patch_w):
                selected_channels = np.random.choice(5, 3, replace=False)  # Pick 3 of 5 channels
                np.random.shuffle(selected_channels)  # Shuffle the selected channels
                patch_channel_map[ph, pw] = selected_channels  # Assign to this patch

        # Apply the patch-wise channel mixing to the entire video
        mixed_video = torch.zeros(self.clip_frames, self.img_size, self.img_size, 3, dtype=torch.float32)
        for i in range(self.clip_frames):
            for ph in range(patch_h):
                for pw in range(patch_w):
                    # Get pixel locations for this patch
                    h_start, h_end = ph * self.patch_size, (ph + 1) * self.patch_size
                    w_start, w_end = pw * self.patch_size, (pw + 1) * self.patch_size

                    # Apply selected channels to the corresponding patch
                    mixed_video[i, h_start:h_end, w_start:w_end, :] = video[i, h_start:h_end, w_start:w_end,
                                                                      patch_channel_map[ph, pw]]

        # Generate a single spatial mask for all channels
        for i in range(self.clip_frames):
            if i % self.tubelet_size == 0:
                # Use Path for consistent npy file path
                npy_filename = files[indexes[i]].replace(".jpg", ".npy")
                mask = self.gen_mask(meta_path, npy_filename, keep_queue)  # Generate one mask
                masks[i // self.tubelet_size] = mask  # Apply the same mask to all channels

        # Apply masking strategy (tube, frame, random)
        masks = self.apply_mask_strategy(masks).bool()

        # Normalize masking to match target visibility percentage
        masks = self.normalize_mask(masks).bool()

        # Rearrange tensors for proper shape
        mixed_video = rearrange(mixed_video, "t h w c -> c t h w")  # Mixed channels first
        rgb_frames = rearrange(rgb_frames, "t h w c -> c t h w")  # Original RGB
        depth_frames = rearrange(depth_frames, "t h w c -> c t h w")  # Original Depth
        thermal_frames = rearrange(thermal_frames, "t h w c -> c t h w")  # Original Thermal

        return mixed_video, masks.flatten().bool(), rgb_frames, depth_frames, thermal_frames

    def apply_mask_strategy(self, masks):
        """
        Applies the selected masking strategy (tube, frame, or random).

        Args:
            masks (Tensor): Initial mask tensor of shape (T, H, W).

        Returns:
            Tensor: Mask tensor with the selected strategy applied.
        """
        if self.mask_strategy == "tube":
            # Use the first frame's mask for all frames
            first_mask = masks[0].flatten().bool()
            target_visible_num = ceil(len(first_mask) * self.mask_percentage_target)
            visible_indexes = first_mask.nonzero().flatten()
            extra_indexes = np.random.choice(visible_indexes, len(visible_indexes) - target_visible_num, replace=False)
            first_mask[extra_indexes] = False
            # masks = first_mask.repeat(self.clip_frames).view(masks.shape)  # Repeat for all frames
            masks = first_mask.view(1, *masks.shape[1:]).expand(masks.shape)

        elif self.mask_strategy == "frame":
            # Mask whole frames based on visibility percentage
            frame_mask = torch.ones(self.clip_frames)
            target_visible_num = ceil(len(frame_mask) * self.mask_percentage_target)
            visible_indexes = frame_mask.nonzero().flatten()
            extra_indexes = np.random.choice(visible_indexes, len(visible_indexes) - target_visible_num, replace=False)
            frame_mask[extra_indexes] = 0.0
            masks = frame_mask.view(self.clip_frames, 1, 1).expand_as(masks).bool()  # Apply frame-wise mask

        else:
            # Random masking (no structured constraints)
            masks = masks.flatten().bool()

        return masks

    def normalize_mask(self, masks):
        """
        Ensures the mask strictly follows the target visibility percentage.

        Args:
            masks (Tensor): Mask tensor of shape (T, H, W).

        Returns:
            Tensor: Normalized mask tensor.
        """
        target_visible_num = int(len(masks.flatten()) * self.mask_percentage_target)
        visible_indexes = masks.nonzero().flatten()
        extra_indexes = np.random.choice(visible_indexes, len(visible_indexes) - target_visible_num, replace=False)
        masks[extra_indexes] = False  # Set excess visible patches to False (masked)

        return masks

    def gen_mask(self, dir_path, file_name, keep_queue: Collection[int]) -> Tensor:
        # we follow the tube style masking, where the masking is only determined by the first frame
        # 0 -> masked, 1 -> visible

        """
            Generates a spatial mask for a video frame using tube-style masking.

            The mask is determined based on face parsing results from the first frame.
            - `0` represents masked regions.
            - `1` represents visible regions.

            If `face_strategy` is disabled, the function returns a fully visible mask.
            Otherwise, it loads face parsing results and selectively masks/unmasks regions.

            Args:
                dir_path (str): Directory path where the face parsing `.npy` file is stored.
                file_name (str): The filename of the face parsing `.npy` file.
                keep_queue (Collection[int]): Indices of facial regions to keep visible.

            Returns:
                Tensor: A (mask_unit_num, mask_unit_num) tensor where `0` indicates masked areas and `1` indicates visible areas.

            Processing Steps:
            1. **Initialize mask** to all `0`s (fully masked).
            2. If `face_strategy` is disabled, return a fully `1` mask (fully visible).
            3. Load face parsing `.npy` file and extract segmentation data.
            4. If face data exists, iterate over `keep_queue` to unmask selected facial regions.
            5. Apply **max pooling** to expand unmasked regions.
            6. Stop once the unmasked area reaches `mask_percentage_target`.
            7. If no face data exists, return a fully `1` mask.

            Example Usage:
            # mask = gen_mask("subject_01/video_03", "frame_10.npy", keep_queue=[1, 2])
            """
        patch_masking = torch.zeros(self.mask_unit_num, self.mask_unit_num, dtype=torch.float32)
        # if mask randomly, early return
        if not self.face_strategy:
            patch_masking[:] = 1
            return patch_masking

        # Ensure directory path is a Path object
        dir_path = Path(str(dir_path).replace('\\', '/'))

        # load face parsing results using Path for consistent handling
        npy_file = self.root_dir / "Texture_crop_face_parsing_images_DB" / dir_path / file_name

        if npy_file.exists():
            face_parsing = torch.from_numpy(np.load(npy_file))

            if face_parsing.shape[0] > 0:
                terminate = False
                for i in keep_queue:
                    if terminate:
                        break
                    # self.seg_groups[i] holds face component labels (e.g., left eye, mouth, etc.).
                    # 	•	Checks if pixels in face_parsing match comp_value and converts them to float (1 where matched, 0 otherwise).
                    # 	•	Uses max pooling (F.max_pool2d) to enlarge the mask region.
                    # 	•	torch.maximum() ensures that once a pixel is unmasked (1), it stays unmasked.
                    for comp_value in self.seg_groups[i]:
                        #	Convert to Patch-Based Masking Using Max Pooling
                        # •	The max pooling operation (F.max_pool2d) with kernel_size=self.patch_size (e.g., 16) downsamples this absolute mask into the patch grid.
                        # •	If any pixel in a patch is 1 (i.e., belongs to the selected face component), the entire patch is marked as visible (1).
                        patch_masking = torch.maximum(patch_masking, F.max_pool2d((face_parsing == comp_value).float(),
                                                                                  kernel_size=self.patch_size)[0])
                        if patch_masking.mean() >= self.mask_percentage_target:
                            terminate = True
                            break
        else:
            print(f"Warning: Missing face parsing file {npy_file}. Using fully visible mask.")
            patch_masking[:] = 1.

        return patch_masking

    def __len__(self) -> int:
        return len(self.metadata)

    def _sample_indexes(self, num_frames: int) -> Tensor:
        return sample_indexes(num_frames, self.clip_frames, self.temporal_sample_rate)


class BP4DMultiModalDataModule(LightningDataModule):

    def __init__(self,
                 root_dir: str,
                 batch_size: int,
                 clip_frames: int,
                 temporal_sample_rate: int,
                 patch_size: int,
                 tubelet_size: int,
                 mask_percentage_target: float = 0.8,
                 mask_strategy: str = "fasking",
                 num_workers: int = 0,
                 take_train: Optional[int] = None,
                 take_val: Optional[int] = None
                 ):
        super().__init__()
        self.root_dir = Path(root_dir)  # Convert to Path object
        self.batch_size = batch_size
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.mask_percentage_target = mask_percentage_target
        self.mask_strategy = mask_strategy
        self.take_train = take_train
        self.take_val = take_val
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = BP4DMultiModal(
            root_dir=self.root_dir,
            split="train",
            clip_frames=self.clip_frames,
            temporal_sample_rate=self.temporal_sample_rate,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
            mask_percentage_target=self.mask_percentage_target,
            mask_strategy=self.mask_strategy,
            take_num=self.take_train
        )

        self.val_dataset = BP4DMultiModal(
            root_dir=self.root_dir,
            split="val",
            clip_frames=self.clip_frames,
            temporal_sample_rate=self.temporal_sample_rate,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
            mask_percentage_target=self.mask_percentage_target,
            mask_strategy=self.mask_strategy,
            take_num=self.take_val
        )
        # print("The dataset has {} train and {} val samples.".format(len(self.train_dataset), len(self.val_dataset)))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)