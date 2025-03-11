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
