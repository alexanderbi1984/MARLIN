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
        
        Args:
            video_tensor: Input video tensor of shape [1, C, T, H, W]
            feature_idx: Index of the feature to visualize
            output_path: Path to save visualization, if None will just return the heatmap
            
        Returns:
            Tuple of (frames, heatmaps): Original frames and corresponding heatmaps
        """
        video_tensor = video_tensor.to(self.device)
        video_tensor.requires_grad_()
        
        # Forward pass to get features
        with torch.no_grad():
            # Create a dummy mask (all True = visible)
            batch_size = video_tensor.shape[0]
            height, width = video_tensor.shape[3], video_tensor.shape[4]
            patch_size = self.model.patch_size
            tubelet_size = self.model.tubelet_size
            n_patch_h = height // patch_size
            n_patch_w = width // patch_size
            n_frames = video_tensor.shape[2] // tubelet_size
            
            num_patches = n_patch_h * n_patch_w * n_frames
            mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=self.device)
        
        self.model.zero_grad()
        
        # Get model features with gradients enabled
        features = self.model.encoder.extract_features(video_tensor, seq_mean_pool=True)
        
        # Clear previous gradients
        if video_tensor.grad is not None:
            video_tensor.grad.zero_()
            
        # Set up a hook for backpropagation
        # Create a zero tensor with a 1 at the target feature position
        target = torch.zeros_like(features)
        target[0, feature_idx] = 1.0
        
        # Backpropagate to get gradients
        features.backward(gradient=target)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Calculate weights based on gradients
        weights = torch.mean(gradients, dim=[0, 1])
        
        # Create class activation map
        cam = torch.matmul(activations, weights.unsqueeze(-1)).squeeze(-1)
        
        # Reshape CAM to match temporal structure
        # Assuming activations are [B, N, D] where N is the number of patches
        # We need to reshape to [B, T, H, W]
        cam = cam.reshape(batch_size, n_frames, n_patch_h, n_patch_w)
        
        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Interpolate to original resolution
        cam = cam.unsqueeze(1)  # [B, 1, T, H, W]
        cam = nn.functional.interpolate(
            cam, 
            size=(video_tensor.shape[2], video_tensor.shape[3], video_tensor.shape[4]),
            mode='trilinear',
            align_corners=False
        )
        cam = cam.squeeze(1)  # [B, T, H, W]
        
        # Convert to numpy
        cam = cam.detach().cpu().numpy()[0]  # [T, H, W]
        
        # Get original frames
        frames = video_tensor.detach().cpu().numpy()[0]  # [C, T, H, W]
        frames = np.transpose(frames, (1, 2, 3, 0))  # [T, H, W, C]
        
        # Create heatmaps
        heatmaps = []
        for t in range(frames.shape[0]):
            # Normalize the frame to [0, 255]
            frame = frames[t]
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255
            frame = frame.astype(np.uint8)
            
            # Convert CAM to heatmap
            heatmap = cam[t]
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Overlay heatmap on frame
            overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
            heatmaps.append(overlay)
        
        # Save visualizations if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create a grid of frames and heatmaps
            fig, axs = plt.subplots(2, min(8, frames.shape[0]), figsize=(16, 4))
            for i in range(min(8, frames.shape[0])):
                axs[0, i].imshow(frames[i])
                axs[0, i].set_title(f"Frame {i}")
                axs[0, i].axis('off')
                
                axs[1, i].imshow(heatmaps[i])
                axs[1, i].set_title(f"Heatmap {i}")
                axs[1, i].axis('off')
            
            plt.suptitle(f"Feature {feature_idx} Visualization")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
        return frames, heatmaps
    
    def visualize_multiple_features(self, video_tensor, output_dir):
        """
        Visualizes multiple features for a video.
        
        Args:
            video_tensor: Input video tensor of shape [1, C, T, H, W]
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for feature_idx in self.feature_indices:
            output_path = os.path.join(output_dir, f"feature_{feature_idx}.png")
            self.visualize_feature(video_tensor, feature_idx, output_path)
            print(f"Saved visualization for feature {feature_idx} to {output_path}")
    
    def create_feature_comparison_video(self, video_tensor, output_path, fps=30):
        """
        Creates a video comparing the original video with heatmaps for each feature.
        
        Args:
            video_tensor: Input video tensor of shape [1, C, T, H, W]
            output_path: Path to save the output video
            fps: Frames per second for the output video
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get original frames
        frames = video_tensor.detach().cpu().numpy()[0]  # [C, T, H, W]
        frames = np.transpose(frames, (1, 2, 3, 0))  # [T, H, W, C]
        
        # Normalize frames to [0, 255]
        frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8) * 255
        frames = frames.astype(np.uint8)
        
        # Get heatmaps for each feature
        all_heatmaps = []
        for feature_idx in self.feature_indices:
            _, heatmaps = self.visualize_feature(video_tensor, feature_idx)
            all_heatmaps.append(heatmaps)
        
        # Create a combined visualization
        num_features = len(self.feature_indices)
        num_frames = frames.shape[0]
        
        # Size of each frame in the grid
        frame_height, frame_width = frames.shape[1], frames.shape[2]
        
        # Create grid layout: original video + one row for each feature
        grid_height = frame_height * (num_features + 1)
        grid_width = frame_width
        
        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
        
        # For each frame in the video
        for t in range(num_frames):
            # Create a blank grid
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Add the original frame
            grid[:frame_height, :, :] = frames[t]
            
            # Add heatmaps for each feature
            for i, feature_idx in enumerate(self.feature_indices):
                heatmap = all_heatmaps[i][t]
                start_y = (i + 1) * frame_height
                end_y = start_y + frame_height
                grid[start_y:end_y, :, :] = heatmap
                
                # Add feature label
                cv2.putText(
                    grid, 
                    f"Feature {feature_idx}", 
                    (10, start_y + 20), 
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
    video = read_video(args.video_path, channel_first=True) / 255.0  # [T, C, H, W]
    
    # Make sure the video has the right number of frames
    clip_frames = model.clip_frames
    if video.shape[0] < clip_frames:
        video = padding_video(video, clip_frames, "same")
    elif video.shape[0] > clip_frames:
        video = video[:clip_frames]
    
    # Convert to the right format
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
    
    # Visualize features
    print("Visualizing features...")
    visualizer.visualize_multiple_features(video, args.output_dir)
    
    # Create comparison video
    if not args.skip_comparison_video:
        comparison_video_path = os.path.join(args.output_dir, "feature_comparison.mp4")
        print("Creating feature comparison video...")
        visualizer.create_feature_comparison_video(video, comparison_video_path, fps=args.fps)
    
    print("Done!")


if __name__ == "__main__":
    main()