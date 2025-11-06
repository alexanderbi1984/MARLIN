"""
marlin_feature_extract.py

A utility script for extracting features from videos using the MARLIN (Masked Autoencoder 
for facial video Representation LearnINg) model. This script processes video files in bulk,
extracting visual features that can be used for downstream tasks like emotion recognition
or facial expression analysis.

Key Features:
- Extract features from videos using pre-trained MARLIN models
- Support for both standard MARLIN and multimodal MARLIN models
- Batch processing of multiple videos
- Configurable feature extraction parameters
- Error handling to continue processing despite individual video failures
- Save features as NumPy arrays for easy loading in other applications

Functions:
    The script is primarily designed to be run as a standalone program, with the main
    functionality implemented in the __main__ block. It processes all video files in a
    specified directory, extracts features using the MARLIN model, and saves them as
    .npy files.

Command Line Usage:
    python marlin_feature_extract.py --backbone MODEL_NAME --data_dir VIDEO_DIR 
                                    --ckpt CHECKPOINT_PATH --keep_seq --output_dir FEATURE_DIR

Arguments:
    --backbone: Model architecture to use (default: "marlin_vit_base_ytf")
    --data_dir: Directory containing video files (default: "C:\pain\BioVid_224_video")
    --ckpt: Path to model checkpoint (default: "ckpt\multimodal_marlin\last-v1_final.ckpt")
    --keep_seq: Flag to maintain sequential features (default: False)
    --output_dir: Directory to save extracted features (default: same as data_dir)

Model Options:
    1. Standard MARLIN model:
       - Backbone: "marlin_vit_base_ytf"
       - Loaded from online sources
       - Suitable for general facial video feature extraction

    2. Multimodal MARLIN model:
       - Backbone: "multimodal_marlin_base"
       - Loaded from local checkpoint
       - Enhanced for multimodal feature extraction

Feature Extraction Parameters:
    - sample_rate: Controls frame sampling frequency (based on model's tubelet_size)
    - stride: Determines overlap between consecutive feature extractions
    - keep_seq: Option to maintain sequential nature of features
    - crop_face: Option to crop faces from frames (default: False)

Output:
    - Creates a subdirectory named after the model backbone in the data directory
    - Saves each video's features as a .npy file with the same base name as the video
    - Features are stored as NumPy arrays with shape (n_segments, feature_dim)

Dependencies:
    - PyTorch
    - NumPy
    - MARLIN PyTorch implementation
    - tqdm (for progress bars)

Example:
    To extract features from all videos in a directory using the standard MARLIN model:
    python marlin_feature_extract.py --data_dir ./videos --output_dir ./features

Notes:
    - If a video fails to process, an empty feature array is created and processing continues
    - The script automatically creates the output directory if it doesn't exist
    - Features are extracted on GPU if available, otherwise on CPU
    - The script processes videos in sorted order for reproducibility
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from marlin_pytorch import Marlin
from model.marlin_multimodal import MultiModalMarlin
from model.config import resolve_config

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Marlin Feature Extraction")
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--keep_seq", action="store_true")
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    if args.ckpt is None:
        args.ckpt = r"ckpt\multimodal_marlin\last-v1_final.ckpt"
    if args.backbone is None:
        args.backbone = "marlin_vit_base_ytf"
        model = Marlin.from_online(args.backbone)
    else:
        args.backbone = "multimodal_marlin_base"
        model = MultiModalMarlin.from_file(args.backbone, args.ckpt)

    config = resolve_config(args.backbone)
    feat_dir = args.backbone
    if args.keep_seq:
        keep_seq = True
    else:
        keep_seq = False
    model.cuda()
    model.eval()
    if args.data_dir is None:
        args.data_dir = r"C:\pain\BioVid_224_video"
    # raw_video_path = os.path.join(args.data_dir, "cropped")
    raw_video_path = args.data_dir
    all_videos = sorted(list(filter(lambda x: x.endswith((".mp4", ".avi")), os.listdir(raw_video_path))))
    Path(os.path.join(args.data_dir, feat_dir)).mkdir(parents=True, exist_ok=True)
    for video_name in tqdm(all_videos):
        video_path = os.path.join(raw_video_path, video_name)
        # Get the base name without the extension
        base_name = os.path.splitext(video_name)[0]

        # Construct the save path with the new extension .npy
        save_path = os.path.join(args.data_dir, feat_dir, f"{base_name}.npy")
        # save_path = os.path.join(args.data_dir, feat_dir, video_name.replace(".mp4", ".npy"))
        try:
            feat = model.extract_video(
                video_path, crop_face=False,
                sample_rate=config.tubelet_size, stride=config.n_frames,
                keep_seq=keep_seq, reduction="none")

        except Exception as e:
            print(f"Video {video_path} error.", e)
            feat = torch.zeros(0, model.encoder.embed_dim, dtype=torch.float32)
        np.save(save_path, feat.cpu().numpy())
