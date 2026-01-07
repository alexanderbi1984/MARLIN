import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# Ensure repo root is on sys.path so `model.*` imports work regardless of cwd.
# This file lives at: <repo_root>/preprocess/marlin_feature_extract.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marlin_pytorch import Marlin
from model.marlin_multimodal import MultiModalMarlin
from model.config import resolve_config

def _set_safe_cache_defaults() -> None:
    """
    Avoid writing caches to $HOME by default (repo rule).
    Users can override by setting env vars before running.
    """
    os.environ.setdefault("PIP_CACHE_DIR", "/data/Nbi/.cache/pip")
    os.environ.setdefault("XDG_CACHE_HOME", "/data/Nbi/.cache")
    os.environ.setdefault("TORCH_HOME", "/data/Nbi/.cache/torch")
    os.environ.setdefault("HF_HOME", "/data/Nbi/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/Nbi/.cache/huggingface/transformers")


def _sorted_frame_files(frames_dir: str, exts: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for name in os.listdir(frames_dir):
        if name.lower().endswith(exts):
            paths.append(os.path.join(frames_dir, name))
    paths.sort()
    return paths


def _read_rgb_frame_224(path: str) -> torch.Tensor:
    """
    Returns float tensor in [0,1] with shape (3, 224, 224).
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != 224 or img.shape[1] != 224:
        raise ValueError(f"Expected 224x224 frames, got {img.shape[1]}x{img.shape[0]} for {path}")
    t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
    if t.shape[0] != 3:
        raise ValueError(f"Expected 3-channel RGB frame, got shape {tuple(t.shape)} for {path}")
    return t


def _pad_tail_replicate(frames: list[torch.Tensor], target_len: int) -> list[torch.Tensor]:
    if len(frames) == 0:
        raise ValueError("No frames provided.")
    if len(frames) >= target_len:
        return frames[:target_len]
    last = frames[-1]
    return frames + [last] * (target_len - len(frames))


def _iter_clips_from_frames(
    frame_files: list[str],
    clip_frames: int,
    sample_rate: int,
    stride: int,
    device: torch.device,
) -> "torch.Generator[torch.Tensor, None, None]":
    """
    Mirrors Marlin._load_video behavior for frame sequences.
    Yields clips shaped (1, 3, clip_frames, 224, 224).
    """
    total_frames = len(frame_files)
    if total_frames == 0:
        return

    # Case 1: very short - pad to clip_frames
    if total_frames <= clip_frames:
        frames = [_read_rgb_frame_224(p) for p in frame_files]
        frames = _pad_tail_replicate(frames, clip_frames)
        v = torch.stack(frames, dim=0)  # (T, C, H, W)
        yield v.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        return

    # Case 2: medium - use first clip_frames (note: model ignores sample_rate in this branch)
    if total_frames <= clip_frames * sample_rate:
        to_load = frame_files[:clip_frames]
        frames = [_read_rgb_frame_224(p) for p in to_load]
        frames = _pad_tail_replicate(frames, clip_frames)
        v = torch.stack(frames, dim=0)  # (T, C, H, W)
        yield v.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        return

    # Case 3: long - sliding windows over sampled frames
    sampled = frame_files[::sample_rate]
    if len(sampled) < clip_frames:
        # extremely pathological, but handle gracefully
        frames = [_read_rgb_frame_224(p) for p in sampled]
        frames = _pad_tail_replicate(frames, clip_frames)
        v = torch.stack(frames, dim=0)
        yield v.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        return

    for start in range(0, len(sampled) - clip_frames, stride):
        window = sampled[start:start + clip_frames]
        frames = [_read_rgb_frame_224(p) for p in window]
        v = torch.stack(frames, dim=0)  # (T, C, H, W)
        yield v.permute(1, 0, 2, 3).unsqueeze(0).to(device)


if __name__ == '__main__':
    _set_safe_cache_defaults()
    parser = argparse.ArgumentParser("Marlin Feature Extraction")
    parser.add_argument("--backbone", type=str, default=None,
                        help="If None, uses marlin_vit_base_ytf (downloaded). Otherwise loads multimodal_marlin_base.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing video files (*.mp4/*.avi).")
    parser.add_argument("--frames_root", type=str, default=None,
                        help="Root directory where each subfolder is one video and contains frames (224x224).")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for multimodal model.")
    parser.add_argument("--keep_seq", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save .npy features. Default: <input_root>/<backbone>/")
    parser.add_argument("--frame_exts", type=str, default="jpg,jpeg,png,bmp",
                        help="Comma-separated list of frame extensions to load when using --frames_root.")
    args = parser.parse_args()

    if args.data_dir is None and args.frames_root is None:
        raise ValueError("Please provide either --data_dir (videos) or --frames_root (frame folders).")
    if args.data_dir is not None and args.frames_root is not None:
        raise ValueError("Please provide only one of --data_dir or --frames_root.")

    if args.backbone is None:
        args.backbone = "marlin_vit_base_ytf"
        model = Marlin.from_online(args.backbone)
    else:
        args.backbone = "multimodal_marlin_base"
        if args.ckpt is None:
            raise ValueError("When using multimodal model, please pass --ckpt.")
        model = MultiModalMarlin.from_file(args.backbone, args.ckpt)

    config = resolve_config(args.backbone)
    feat_dir = args.backbone
    if args.keep_seq:
        keep_seq = True
    else:
        keep_seq = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Determine output directory
    input_root = args.data_dir if args.data_dir is not None else args.frames_root
    assert input_root is not None
    out_root = args.output_dir if args.output_dir is not None else os.path.join(input_root, feat_dir)
    Path(out_root).mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        raw_video_path = args.data_dir
        all_videos = sorted(list(filter(lambda x: x.lower().endswith((".mp4", ".avi")), os.listdir(raw_video_path))))
        for video_name in tqdm(all_videos, desc="Extracting (videos)"):
            video_path = os.path.join(raw_video_path, video_name)
            base_name = os.path.splitext(video_name)[0]
            save_path = os.path.join(out_root, f"{base_name}.npy")
            try:
                feat = model.extract_video(
                    video_path, crop_face=False,
                    sample_rate=config.tubelet_size, stride=config.n_frames,
                    keep_seq=keep_seq, reduction="none")
            except Exception as e:
                print(f"Video {video_path} error.", e)
                feat = torch.zeros(0, model.encoder.embed_dim, dtype=torch.float32, device=device)
            np.save(save_path, feat.detach().cpu().numpy())
    else:
        frames_root = args.frames_root
        assert frames_root is not None
        exts = tuple([f".{e.strip().lower()}" for e in args.frame_exts.split(",") if e.strip()])
        video_dirs = sorted([d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))])
        for vid in tqdm(video_dirs, desc="Extracting (frame folders)"):
            vid_dir = os.path.join(frames_root, vid)
            save_path = os.path.join(out_root, f"{vid}.npy")
            try:
                frame_files = _sorted_frame_files(vid_dir, exts)
                if len(frame_files) == 0:
                    raise ValueError(f"No frames found in {vid_dir} with exts={exts}")
                feats = []
                for clip in _iter_clips_from_frames(
                    frame_files=frame_files,
                    clip_frames=config.n_frames,
                    sample_rate=config.tubelet_size,
                    stride=config.n_frames,
                    device=device,
                ):
                    feats.append(model.extract_features(clip, keep_seq=keep_seq))
                feat = torch.cat(feats, dim=0) if len(feats) else torch.zeros(
                    0, model.encoder.embed_dim, dtype=torch.float32, device=device
                )
            except Exception as e:
                print(f"Frames {vid_dir} error.", e)
                feat = torch.zeros(0, model.encoder.embed_dim, dtype=torch.float32, device=device)
            np.save(save_path, feat.detach().cpu().numpy())
