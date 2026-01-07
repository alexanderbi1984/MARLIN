import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional

def calculate_pspi(row: pd.Series) -> float:
    """
    Calculate PSPI (Prkachin and Solomon Pain Intensity) score.
    Formula: PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
    
    Note on AU43 (Eyes Closed): OpenFace typically outputs AU45 (Blink).
    We use AU45 as a proxy for AU43 if available.
    All inputs are expected to be intensities (0-5 scale).
    """
    au4 = row.get('AU04_r', 0.0)
    au6 = row.get('AU06_r', 0.0)
    au7 = row.get('AU07_r', 0.0)
    au9 = row.get('AU09_r', 0.0)
    au10 = row.get('AU10_r', 0.0)
    
    # OpenFace usually provides AU45_r (Blink intensity) instead of AU43
    au43_proxy = row.get('AU45_r', 0.0) 
    
    pspi = au4 + max(au6, au7) + max(au9, au10) + au43_proxy
    return pspi

def process_video_csv(
    csv_path: str, 
    window_size: int, 
    stride: int, 
    use_pspi: bool = True,
    use_raw_aus: bool = True,
    min_confidence: float = 0.8
) -> Optional[np.ndarray]:
    
    try:
        # OpenFace headers usually have spaces, let's strip them
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        # Filter by confidence if the column exists
        if 'confidence' in df.columns:
            # We treat low confidence frames as zeros or keep them? 
            # Usually better to keep continuity but maybe zero out features
            # Here we just zero out features for low confidence frames
            mask = df['confidence'] < min_confidence
        else:
            mask = pd.Series([False] * len(df))

        # Identify AU columns (Intensity)
        au_cols = [c for c in df.columns if c.startswith('AU') and c.endswith('_r')]
        au_cols = sorted(au_cols)
        
        features_list = []
        
        # 1. Raw AUs
        if use_raw_aus:
            au_data = df[au_cols].copy()
            au_data.loc[mask, :] = 0.0 # Zero out low confidence
            features_list.append(au_data.values)
            
        # 2. PSPI
        if use_pspi:
            # Calculate PSPI per frame
            pspi_vals = df.apply(calculate_pspi, axis=1).values.reshape(-1, 1)
            pspi_vals[mask] = 0.0
            features_list.append(pspi_vals)
            
        if not features_list:
            return None
            
        # Concatenate features: Shape (N_frames, Feature_Dim)
        # e.g. 17 AUs + 1 PSPI = 18 dim
        combined_features = np.hstack(features_list)
        
        # Sliding Window
        num_frames = combined_features.shape[0]
        windows = []
        
        if num_frames < window_size:
            # Padding if video is shorter than one window
            pad_len = window_size - num_frames
            padded = np.pad(combined_features, ((0, pad_len), (0, 0)), mode='constant')
            windows.append(padded)
        else:
            for start in range(0, num_frames - window_size + 1, stride):
                end = start + window_size
                win = combined_features[start:end]
                windows.append(win)
                
        if not windows:
            return None
            
        # Stack to (N_windows, Window_Size, Feature_Dim)
        return np.stack(windows)

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert OpenFace CSVs to Windowed NPY features for Syracuse MIL")
    parser.add_argument("--csv_root", type=str, required=True, help="Directory containing OpenFace CSV files")
    parser.add_argument("--save_root", type=str, required=True, help="Where to save the .npy files")
    parser.add_argument("--window_size", type=int, default=64, help="Window size in frames (default: 64)")
    parser.add_argument("--stride", type=int, default=32, help="Stride in frames (default: 32)")
    parser.add_argument("--suffix", type=str, default="_au_windows.npy", help="Suffix for output files")
    parser.add_argument("--no_pspi", action="store_true", help="Do not include PSPI feature")
    parser.add_argument("--only_pspi", action="store_true", help="Only use PSPI feature (1 dim)")
    
    args = parser.parse_args()
    
    # Resolve conflicting flags
    use_pspi = True
    use_raw_aus = True
    
    if args.no_pspi:
        use_pspi = False
    if args.only_pspi:
        use_raw_aus = False
        use_pspi = True
        
    print(f"Configuration:")
    print(f"  Input: {args.csv_root}")
    print(f"  Output: {args.save_root}")
    print(f"  Window/Stride: {args.window_size}/{args.stride}")
    print(f"  Include Raw AUs: {use_raw_aus}")
    print(f"  Include PSPI: {use_pspi}")
    
    os.makedirs(args.save_root, exist_ok=True)
    
    # Find CSVs recursively or flat
    csv_files = glob.glob(os.path.join(args.csv_root, "**/*.csv"), recursive=True)
    print(f"Found {len(csv_files)} CSV files.")
    
    success_count = 0
    
    for csv_path in tqdm(csv_files):
        # Infer Video ID from filename
        # Assumption: Filename is like "sub001_trial1.csv" -> video_id = "sub001_trial1"
        filename = os.path.basename(csv_path)
        video_id = os.path.splitext(filename)[0]
        
        # Handle OpenFace specific suffix if present (e.g. some outputs are vid_of_details.csv)
        # For now, we assume the filename IS the video_id needed for matching metadata
        
        windows = process_video_csv(
            csv_path, 
            window_size=args.window_size, 
            stride=args.stride,
            use_pspi=use_pspi,
            use_raw_aus=use_raw_aus
        )
        
        if windows is not None:
            # Save
            out_name = f"{video_id}{args.suffix}"
            out_path = os.path.join(args.save_root, out_name)
            np.save(out_path, windows)
            success_count += 1
            
            # Print dimension of the first successful processing to inform user
            if success_count == 1:
                print(f"\n[INFO] Feature shape for first video: {windows.shape}")
                print(f"       (N_windows, Window_Size, Feature_Dim)")
                print(f"       Use 'input_dim: {windows.shape[2]}' in your YAML config.\n")
                
    print(f"Done. Processed {success_count}/{len(csv_files)} files.")

if __name__ == "__main__":
    main()



