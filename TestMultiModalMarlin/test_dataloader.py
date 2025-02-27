import torch
from torch.utils.data import DataLoader
from dataset.bp4d_multimodal import BP4DMultiModalDataModule

# Create a small test dataset
test_dataset = BP4DMultiModalDataModule(
    meta_file="val_set.csv",
    root_dir="./BP4D+",
)

# Create a dataloader with a small batch size
batch_size = 2
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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