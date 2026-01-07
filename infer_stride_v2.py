import pandas as pd
import numpy as np
import os

# Updated paths from user query
csv_path = "/data/Nbi/Syracuse/syracuse_aug_AU_features/IMG_0083.csv"
npy_path = "/data/Nbi/Syracuse/syracuse_random_features/RGB/IMG_0103_windows.npy"

# Note: The user provided different video IDs for NPY (IMG_0103) and CSV (IMG_0083).
# Ideally we should compare the SAME video to be sure.
# However, let's inspect both to see if we can find a matching pair or just infer from available data.

# Let's try to find if there is an IMG_0103 CSV or an IMG_0083 NPY to make a direct comparison.
base_csv_dir = os.path.dirname(csv_path)
base_npy_dir = os.path.dirname(npy_path)

candidate_csv_103 = os.path.join(base_csv_dir, "IMG_0103.csv")
candidate_npy_083 = os.path.join(base_npy_dir, "IMG_0083_windows.npy")

target_csv = csv_path
target_npy = npy_path

if os.path.exists(candidate_csv_103):
    target_csv = candidate_csv_103
    target_npy = npy_path # Use 103 pair
    print(f"Found matching CSV for IMG_0103. Using pair: IMG_0103")
elif os.path.exists(candidate_npy_083):
    target_csv = csv_path
    target_npy = candidate_npy_083 # Use 083 pair
    print(f"Found matching NPY for IMG_0083. Using pair: IMG_0083")
else:
    print(f"Warning: No matching pair found. Analyzing provided files independently.")
    print(f"NPY: {os.path.basename(npy_path)}")
    print(f"CSV: {os.path.basename(csv_path)}")

# Analyze NPY
npy_windows = 0
if os.path.exists(target_npy):
    try:
        data = np.load(target_npy)
        print(f"\n[NPY Analysis] {os.path.basename(target_npy)}")
        print(f"Shape: {data.shape}")
        npy_windows = data.shape[0]
    except Exception as e:
        print(f"Error reading NPY: {e}")

# Analyze CSV
if os.path.exists(target_csv):
    try:
        df = pd.read_csv(target_csv)
        df.columns = [c.strip() for c in df.columns]
        num_frames = len(df)
        print(f"\n[CSV Analysis] {os.path.basename(target_csv)}")
        print(f"Total Frames: {num_frames}")
        
        if npy_windows > 0:
            # Infer stride
            # Formula: N_windows = (Num_Frames - Window_Size) // Stride + 1
            # Assuming Window_Size = 16 (from context)
            window_size = 16
            
            if npy_windows > 1:
                # Approximate stride
                stride_float = (num_frames - window_size) / (npy_windows - 1)
                print(f"\n[Inference] Assuming Window_Size={window_size}")
                print(f"Target Windows (from NPY): {npy_windows}")
                print(f"Calculated Stride: {stride_float:.4f}")
                
                # Check integer fit
                print("Integer check:")
                for s in [1, 8, 16, 32, 64]:
                    n = (num_frames - window_size) // s + 1
                    print(f"  Stride {s} => {n} windows")
                    if n == npy_windows:
                        print(f"  *** MATCH FOUND: Stride {s} ***")
            else:
                 print(f"NPY has 1 window. Cannot infer stride uniquely.")
                 
    except Exception as e:
        print(f"Error reading CSV: {e}")
else:
    print(f"CSV path not found: {target_csv}")














