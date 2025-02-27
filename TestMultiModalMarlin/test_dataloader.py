import torch
from torch.utils.data import DataLoader
from dataset.bp4d_multimodal import BP4DMultiModalDataModule

# Create a small test dataset
test_dataset = BP4DMultiModalDataModule(
    root_dir=r"W:\Nan\MARLIN\BP4D+",
    batch_size = 2,
    clip_frames = 16,
    temporal_sample_rate = 2,
    patch_size = 16,
    tubelet_size = 2,

)
test_dataset.setup()

# Create a dataloader with a small batch size
test_loader = test_dataset.val_dataloader()
# Get a single batch and examine its properties
for batch in test_loader:
    mixed_video, mask, rgb_frames, depth_frames, thermal_frames = batch

    # Print shapes and value ranges
    print(f"Mixed video shape: {mixed_video.shape}, range: [{mixed_video.min()}, {mixed_video.max()}]")
    print(f"Mask shape: {mask.shape}, sum: {mask.sum()}")
    print(f"RGB frames shape: {rgb_frames.shape}, range: [{rgb_frames.min()}, {rgb_frames.max()}]")
    print(f"Depth frames shape: {depth_frames.shape}, range: [{depth_frames.min()}, {depth_frames.max()}]")
    print(f"Thermal frames shape: {thermal_frames.shape}, range: [{thermal_frames.min()}, {thermal_frames.max()}]")

    # Only check the first batch
    break