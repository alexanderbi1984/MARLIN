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

from util.misc import sample_indexes


class YoutubeFace(Dataset):
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

        ## docstring
        """	Arguments:
	•	root_dir: Path to the dataset.
	•	split: Either “train” or “val”.
	•	clip_frames: Number of frames per clip.
    .   temporal_sample_rate: The step size between consecutive frames.
	•	patch_size: Size of spatial patches.
	•	tubelet_size: Defines temporal units for masking.
	•	mask_percentage_target: Percentage of patches to keep.
	•	mask_strategy: Specifies different masking strategies (e.g., fasking, tube, random).
	•	take_num: Limits the number of samples. like using only 1000 samples for training.
	"""
        self.img_size = 224
        self.root_dir = root_dir
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
        self.metadata = pd.read_csv(os.path.join(root_dir, f"{split}_set.csv"))
        if take_num:
            self.metadata = self.metadata.iloc[:take_num]

    def __getitem__(self, index):
        """
            Retrieves and processes a video sample along with a corresponding masking strategy.

            Args:
                index (int): Index of the video sample in the dataset.

            Returns:
                Tuple[Tensor, Tensor]:
                    - Processed video tensor of shape `(C, T, H, W)`, where:
                        - C: Number of color channels (3 for RGB).
                        - T: Number of frames in the clip.
                        - H, W: Image height and width.
                    - Flattened boolean mask tensor indicating visible (`True`) and masked (`False`) regions.

            Functionality:
            1. **Load Video Frames**:
                - Retrieves metadata for the video sample.
                - Loads image file names from the specified directory.
                - Selects a subset of frames using `_sample_indexes()`.
                - Initializes a zero tensor (`video`) to store processed frames.

            2. **Preprocess Frames**:
                - Reads each image file using OpenCV.
                - Converts images from BGR to RGB.
                - Normalizes pixel values to range [0, 1].
                - Stores the images as PyTorch tensors in the `video` tensor.

            3. **Generate Masking Strategy**:
                - Initializes a mask tensor (`masks`) with either **patch-based masking** (low-resolution)
                  or **full-frame masking** (high-resolution).
                - If `face_strategy` is enabled, determines which face regions should be kept/masked.

            4. **Apply Face-Based Masking**:
                - Calls `gen_mask()` on the first frame of each **tubelet** (a small segment of frames)
                  to determine visible/masked regions.
                - Stores generated masks in the `masks` tensor.

            5. **Apply Masking Strategies**:
                - **"Tube" Strategy**:
                    - Uses only the first frame’s mask.
                    - Randomly reduces the number of visible patches to match `mask_percentage_target`.
                    - Repeats this mask across all tubelets.
                - **"Frame" Strategy**:
                    - Selects frames to be visible/masked at a fixed rate.
                    - Expands this mask to match the shape of the video.
                - **Default**: Uses the original generated mask.

            6. **Normalize Masking for Batch Computation**:
                - Ensures the final mask matches the target visibility percentage.
                - Randomly removes extra visible patches to align with `mask_percentage_target`.

            7. **Return Processed Video and Mask**:
                - Rearranges video tensor to shape `(C, T, H, W)`.
                - Returns the **original video frames** (not masked) and a **1D mask vector**.

            Notes:
            - The mask is not directly applied to the video; it is returned separately for later use.
            - This function is designed for **masked video modeling**, where regions of frames are hidden
              and the model learns to reconstruct or infer missing parts.

            """
        meta = self.metadata.iloc[index]
        files = sorted(os.listdir(os.path.join(self.root_dir, "crop_images_DB", meta.path)))
        indexes = self._sample_indexes(len(files))
        assert len(indexes) == self.clip_frames

        video = torch.zeros(self.clip_frames, self.img_size, self.img_size, 3, dtype=torch.float32)
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

            ## docstring
            """
            	1.	Loads an image from the specified folder.
	            2.	Converts it from BGR to RGB (for compatibility with deep learning models).
	            3.	Normalizes pixel values (scales them to [0, 1]).
	            4.	Converts the image into a PyTorch tensor.
	            5.	Stores it in a pre-allocated video tensor, which contains multiple frames.

            """
            img = cv2.imread(os.path.join(self.root_dir, "crop_images_DB", meta.path, files[indexes[i]]))
            video[i] = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255)
            if i % self.tubelet_size == 0:
                mask = self.gen_mask(meta.path, files[indexes[i]].replace(".jpg", ".npy"), keep_queue)
                masks[i // self.tubelet_size] = mask

        if self.mask_strategy == "tube":
            first_mask = masks[0].flatten().bool()
            target_visible_num = ceil(len(first_mask) * self.mask_percentage_target)
            visible_indexes = first_mask.nonzero().flatten()
            extra_indexes = np.random.choice(visible_indexes, len(visible_indexes) - target_visible_num,
                replace=False)
            first_mask[extra_indexes] = False
            masks = first_mask.repeat(self.clip_frames // self.tubelet_size)
        elif self.mask_strategy == "frame":
            frame_mask = torch.ones(self.clip_frames // self.tubelet_size)
            target_visible_num = ceil(len(frame_mask) * self.mask_percentage_target)
            visible_indexes = frame_mask.nonzero().flatten()
            extra_indexes = np.random.choice(visible_indexes, len(visible_indexes) - target_visible_num,
                replace=False)
            frame_mask[extra_indexes] = 0.0
            masks = rearrange(frame_mask, "t -> t 1 1").expand_as(masks).flatten().bool()

        else:
            masks = masks.flatten().bool()

        # normalize the masking to strictly target percentage for batch computation.
        target_visible_num = int(len(masks) * self.mask_percentage_target)
        visible_indexes = masks.nonzero().flatten()
        extra_indexes = np.random.choice(visible_indexes, len(visible_indexes) - target_visible_num,
            replace=False)
        masks[extra_indexes] = False

        return rearrange(video, "t h w c -> c t h w"), masks.flatten().bool()

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
            # >>> mask = gen_mask("subject_01/video_03", "frame_10.npy", keep_queue=[1, 2])
            """
        patch_masking = torch.zeros(self.mask_unit_num, self.mask_unit_num, dtype=torch.float32)
        # if mask randomly, early return
        if not self.face_strategy:
            patch_masking[:] = 1
            return patch_masking

        # load face parsing results
        npy_file = os.path.join(self.root_dir, "face_parsing_images_DB", dir_path, file_name)
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
            patch_masking[:] = 1.

        return patch_masking

    def __len__(self) -> int:
        return len(self.metadata)

    def _sample_indexes(self, num_frames: int) -> Tensor:
        return sample_indexes(num_frames, self.clip_frames, self.temporal_sample_rate)


class YoutubeFaceDataModule(LightningDataModule):

    def __init__(self,
        root_dir: str,
        batch_size: int,
        clip_frames: int,
        temporal_sample_rate: int,
        patch_size: int,
        tubelet_size: int,
        mask_percentage_target: float = 0.8,
        mask_strategy: str = "face",
        num_workers: int = 0,
        take_train: Optional[int] = None,
        take_val: Optional[int] = None
    ):
        super().__init__()
        self.root_dir = root_dir
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
        self.train_dataset = YoutubeFace(
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

        self.val_dataset = YoutubeFace(
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            pin_memory=True)
