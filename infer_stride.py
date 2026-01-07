import pandas as pd
import numpy as np
import os

csv_path = "/data/Nbi/biovid/openface_au/120614_w_61-PA4-077.csv"

if os.path.exists(csv_path):
    try:
        # Load CSV, trimming spaces from headers
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        num_frames = len(df)
        print(f"File: {os.path.basename(csv_path)}")
        print(f"Total Frames: {num_frames}")
        
        # We know from the previous step that the target .npy has 8 windows (instances).
        # Target shape: (8, 768) -> N_windows = 8
        
        # Let's try to infer stride given window_size = 16 (implied from user description of "16 frame pooled")
        # Formula: N_windows = (Num_Frames - Window_Size) // Stride + 1
        # => Stride = (Num_Frames - Window_Size) / (N_windows - 1)
        
        target_windows = 8
        window_size = 16
        
        if target_windows > 1:
            inferred_stride = (num_frames - window_size) / (target_windows - 1)
            print(f"Target Windows: {target_windows}")
            print(f"Assumed Window Size: {window_size}")
            print(f"Inferred Stride: {inferred_stride:.2f}")
            
            # Check for integer stride candidates
            print("\nChecking common strides:")
            for s in [8, 16, 32, 64]:
                n = (num_frames - window_size) // s + 1
                print(f"  Stride {s}: {n} windows")
        else:
            print(f"Target Windows: {target_windows}. Cannot infer stride from single window.")

    except Exception as e:
        print(f"Error reading CSV: {e}")
else:
    print(f"CSV not found: {csv_path}")














