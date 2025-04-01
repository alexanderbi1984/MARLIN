"""
MARLIN Feature Visualization Script

This script provides functionality to visualize and analyze features learned by the MARLIN (Multimodal Action Recognition with Language INtegration) model.
It generates heatmaps and visualizations that show which parts of a video contribute most to specific learned features in the model.

Key Features:
- Visualizes individual features through heatmap overlays on video frames
- Supports multiple feature visualization in a single run
- Generates both static visualizations and comparison videos
- Works with both standard MARLIN and MultiModalMarlin models
- Provides command-line interface for easy usage

Usage:
    python marlin_feature_visualization.py --checkpoint_path <path_to_checkpoint> --video_path <path_to_video> [options]

Required Arguments:
    --checkpoint_path: Path to the model checkpoint file (.ckpt)
    --video_path: Path to the video file to visualize

Optional Arguments:
    --model_name: MARLIN model name (default: multimodal_marlin_base)
    --features: Feature indices to visualize (default: 397 231 490 482 593)
    --output_dir: Directory to save visualizations (default: feature_visualizations)
    --fps: Frame rate for output comparison video (default: 30)
    --skip_comparison_video: Skip creating the feature comparison video
    --use_standard_marlin: Use standard MARLIN instead of MultiModalMarlin

Output:
    - Individual feature visualizations saved as PNG files
    - Optional comparison video showing original frames and feature heatmaps
    - All outputs are saved in the specified output directory

Dependencies:
    - torch
    - numpy
    - matplotlib
    - opencv-python
    - marlin_pytorch
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os
import sys
from tqdm import tqdm
from model.marlin_multimodal import MultiModalMarlin


# Import statement for MARLIN model will be determined at runtime
# based on command-line arguments

class FeatureVisualizer:
    def __init__(self, model, feature_indices, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the feature visualizer.
        
        Args:
            model: The pretrained MARLIN model
            feature_indices: List of feature indices to visualize (e.g., [397, 231, 490, 482, 593])
            device: Device to run the model on
        """
        self.model = model.to(device)
        self.model.eval()
        self.feature_indices = feature_indices
        self.device = device
        
        # Create hook for the encoder
        self.encoder = model.encoder
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        # Register hooks on the last layer of the encoder
        # You might need to adjust this depending on the exact model architecture
        if hasattr(self.encoder, 'blocks') and len(self.encoder.blocks) > 0:
            target_layer = self.encoder.blocks[-1]
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
        else:
            print("Warning: Could not find blocks in encoder. Visualization might not work correctly.")
    
    def visualize_feature(self, video_tensor, feature_idx, output_path=None):
        """
        Visualizes which parts of the video contribute most to a specific feature.
        Processes the video frame by frame to handle arbitrary length videos.
        """
        frames_list = []
        heatmaps_list = []
        
        # Process each frame individually
        num_frames = video_tensor.shape[2]
        for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
            # Extract single frame and pad to clip_frames length
            frame = video_tensor[:, :, frame_idx:frame_idx+1, :, :]  # [1, C, 1, H, W]
            padded_frame = torch.repeat_interleave(frame, self.model.clip_frames, dim=2)  # [1, C, clip_frames, H, W]
            
            # Process the padded frame
            padded_frame = padded_frame.to(self.device)
            padded_frame.requires_grad_()
            
            # Forward pass to get features
            self.model.zero_grad()
            features = self.model.encoder.extract_features(padded_frame, seq_mean_pool=False)
            
            # Set up gradient target
            target = torch.zeros_like(features)
            target[:, :, feature_idx] = 1.0
            
            # Backpropagate
            features.backward(gradient=target)
            
            # Get gradients and activations
            gradients = self.gradients
            activations = self.activations
            
            # Calculate weights and CAM
            weights = torch.mean(gradients, dim=0)
            activations_squeezed = activations.squeeze(0)
            cam = activations_squeezed @ weights.T
            cam = cam[:, feature_idx]
            
            # Reshape CAM to spatial dimensions (take middle frame from padded sequence)
            n_patch_h = padded_frame.shape[3] // self.model.patch_size
            n_patch_w = padded_frame.shape[4] // self.model.patch_size
            cam = cam.reshape(-1, n_patch_h, n_patch_w)  # [-1, H, W]
            cam = cam[cam.shape[0]//2]  # Take middle frame [H, W]
            
            # Normalize and upscale
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam = cam.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            cam = nn.functional.interpolate(
                cam, 
                size=(padded_frame.shape[3], padded_frame.shape[4]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # [H, W]
            
            # Convert to numpy
            frame_np = frame.squeeze(2).permute(0, 2, 3, 1).detach().cpu().numpy()[0]  # [H, W, C]
            cam_np = cam.detach().cpu().numpy()  # [H, W]
            
            # Convert frame to uint8 without renormalizing
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # Create heatmap
            heatmap = (cam_np * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Overlay with adjusted weights to preserve original colors better
            overlay = cv2.addWeighted(frame_np, 0.8, heatmap, 0.2, 0)
            
            frames_list.append(frame_np)
            heatmaps_list.append(overlay)
        
        # Save visualization if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create a grid showing sample frames
            n_frames_to_show = min(16, len(frames_list))
            stride = len(frames_list) // n_frames_to_show
            fig, axs = plt.subplots(2, n_frames_to_show, figsize=(32, 4))
            
            for i in range(n_frames_to_show):
                idx = i * stride
                axs[0, i].imshow(frames_list[idx])
                axs[0, i].set_title(f"Frame {idx}")
                axs[0, i].axis('off')
                
                axs[1, i].imshow(heatmaps_list[idx])
                axs[1, i].set_title(f"Heatmap {idx}")
                axs[1, i].axis('off')
            
            plt.suptitle(f"Feature {feature_idx} Visualization")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        
        return frames_list, heatmaps_list
    
    def visualize_multiple_features(self, video_tensor, output_dir, video_name, clip_name):
        """
        Visualizes multiple features for a video.
        
        Args:
            video_tensor: Input video tensor of shape [1, C, T, H, W]
            output_dir: Directory to save visualizations
            video_name: Name of the input video
            clip_name: Name of the clip
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for feature_idx in self.feature_indices:
            output_path = os.path.join(output_dir, f"{video_name}_{clip_name}_feature_{feature_idx}.png")
            self.visualize_feature(video_tensor, feature_idx, output_path)
            print(f"Saved visualization for feature {feature_idx} to {output_path}")
    
    def create_feature_comparison_video(self, video_tensor, output_path, fps=30):
        """
        Creates a video comparing the original video with heatmaps for each feature.
        Layout: Original video on the left, heatmaps side by side on the right.
        
        Args:
            video_tensor: Input video tensor of shape [1, C, T, H, W]
            output_path: Path to save the output video
            fps: Frames per second for the output video
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get original frames
        frames = video_tensor.detach().cpu().numpy()[0]  # [C, T, H, W]
        frames = np.transpose(frames, (1, 2, 3, 0))  # [T, H, W, C]
        
        # Convert frames to uint8 without renormalizing
        frames = (frames * 255).astype(np.uint8)
        
        # Get heatmaps for each feature
        all_heatmaps = []
        for feature_idx in self.feature_indices:
            _, heatmaps = self.visualize_feature(video_tensor, feature_idx)
            all_heatmaps.append(heatmaps)
        
        # Create a combined visualization
        num_features = len(self.feature_indices)
        num_frames = frames.shape[0]
        
        # Size of each frame
        frame_height, frame_width = frames.shape[1], frames.shape[2]
        
        # Create grid layout: original video on left, heatmaps side by side on right
        grid_height = frame_height
        grid_width = frame_width * (num_features + 1)  # Original + one for each feature
        
        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
        
        # For each frame in the video
        for t in range(num_frames):
            # Create a blank grid
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Add the original frame on the left
            grid[:, :frame_width, :] = frames[t]
            
            # Add heatmaps for each feature side by side
            for i, feature_idx in enumerate(self.feature_indices):
                heatmap = all_heatmaps[i][t]
                start_x = (i + 1) * frame_width
                end_x = start_x + frame_width
                grid[:, start_x:end_x, :] = heatmap
                
                # Add feature label
                cv2.putText(
                    grid, 
                    f"Feature {feature_idx}", 
                    (start_x + 10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )
            
            # Write the frame to the video
            video.write(grid)
        
        # Release the video writer
        video.release()
        print(f"Saved feature comparison video to {output_path}")


def main():
    """Main function to run the feature visualization with command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize important MARLIN features in video.')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='multimodal_marlin_base',
                        help='MARLIN model name (default: multimodal_marlin_base)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint file (.ckpt)')
    
    # Video parameters
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to the video file to visualize')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frame rate for output comparison video (default: 30)')
    
    # Feature parameters
    parser.add_argument('--features', type=int, nargs='+', 
                        default=[397, 231, 490, 482, 593],
                        help='Feature indices to visualize (default: 397 231 490 482 593)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='feature_visualizations',
                        help='Directory to save visualizations (default: feature_visualizations)')
    parser.add_argument('--skip_comparison_video', action='store_true',
                        help='Skip creating the feature comparison video')
    parser.add_argument('--use_standard_marlin', action='store_true',
                        help='Use standard MARLIN instead of MultiModalMarlin')
    
    args = parser.parse_args()
    
    # Extract video name and clip name from the video path
    video_path = Path(args.video_path)
    video_name = video_path.stem.split('_')[0]  # Get the base video name (e.g., 'IMG_0003')
    clip_name = video_path.stem.split('_')[-1]  # Get the clip name (e.g., 'clip_006')
    
    # Print parameters
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Video: {args.video_path}")
    print(f"Features to visualize: {args.features}")
    print(f"Output directory: {args.output_dir}")
    
    # Load model
    print(f"Loading model {args.model_name} from {args.checkpoint_path}")
    if args.use_standard_marlin:
        from marlin_pytorch import Marlin
        model = Marlin.from_file(args.model_name, args.checkpoint_path)
    else:
        model = MultiModalMarlin.from_file(args.model_name, args.checkpoint_path)
    
    # Initialize visualizer
    visualizer = FeatureVisualizer(model, args.features)
    
    # Load video
    from marlin_pytorch.util import read_video, padding_video
    print(f"Loading video from {args.video_path}")
    video = read_video(args.video_path, channel_first=True)  # [T, C, H, W]
    
    # Convert to float and normalize to [0, 1]
    video = video.float() / 255.0
    
    # Convert to the right format without limiting frames
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
    
    # Visualize features
    print("Visualizing features...")
    visualizer.visualize_multiple_features(video, args.output_dir, video_name, clip_name)
    
    # Create comparison video
    if not args.skip_comparison_video:
        comparison_video_path = os.path.join(args.output_dir, f"{video_name}_{clip_name}_feature_comparison.mp4")
        print("Creating feature comparison video...")
        visualizer.create_feature_comparison_video(video, comparison_video_path, fps=args.fps)
    
    print("Done!")


if __name__ == "__main__":
    main()