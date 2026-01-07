import numpy as np
import os

file_path = "/data/Nbi/biovid/marlin_rgb_features/120614_w_61-PA4-077_aligned_windows.npy"

if os.path.exists(file_path):
    try:
        data = np.load(file_path)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        
        # Check if it looks like (N, 768) or (N, T, 768)
        if len(data.shape) > 1:
             print(f"Last dim size: {data.shape[-1]}")
    except Exception as e:
        print(f"Error loading file: {e}")
else:
    print(f"File not found: {file_path}")














