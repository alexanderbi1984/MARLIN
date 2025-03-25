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
from marlin_pytorch.util import read_video, padding_video


class PrePostFeatureComparison:
    def __init__(self, model, feature_indices, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the feature comparator.
        
        Args:
            model: The pretrained MARLIN model
            feature_indices: List of feature indices to visualize (e.g., [397, 231, 490, 482, 456])
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
        if hasattr(self.encoder, 'blocks') and len(self.encoder.blocks) > 0:
            target_layer = self.encoder.blocks[-1]
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
        else:
            print("Warning: Could not find blocks in encoder. Visualization might not work correctly.")
    
    def get_feature_visualization(self, video_tensor, feature_idx):
        """
        Gets visualization for a specific feature.
        
        Args:
            video_tensor: Input video tensor of shape [1, C, T, H, W]
            feature_idx: Index of the feature to visualize
            
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
            
        # Set up target for backpropagation
        target = torch.zeros_like(features)
        target[0, feature_idx] = 1.0
        
        # Backpropagate
        features.backward(gradient=target)
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Calculate weights
        weights = torch.mean(gradients, dim=[0, 1])
        
        # Create class activation map
        cam = torch.matmul(activations, weights.unsqueeze(-1)).squeeze(-1)
        
        # Reshape CAM to match temporal structure
        cam = cam.reshape(batch_size, n_frames, n_patch_h, n_patch_w)
        
        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Interpolate to original resolution
        cam = cam.unsqueeze(1)
        cam = nn.functional.interpolate(
            cam, 
            size=(video_tensor.shape[2], video_tensor.shape[3], video_tensor.shape[4]),
            mode='trilinear',
            align_corners=False
        )
        cam = cam.squeeze(1)
        
        # Convert to numpy
        cam = cam.detach().cpu().numpy()[0]
        
        # Get original frames
        frames = video_tensor.detach().cpu().numpy()[0]
        frames = np.transpose(frames, (1, 2, 3, 0))
        
        # Create heatmaps
        heatmaps = []
        for t in range(frames.shape[0]):
            frame = frames[t]
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255
            frame = frame.astype(np.uint8)
            
            heatmap = cam[t]
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
            heatmaps.append(overlay)
        
        return frames, heatmaps
    
    def create_comparison(self, pre_video, post_video, output_dir):
        """
        Creates comparison visualizations for pre and post videos.
        
        Args:
            pre_video: Pre-treatment video tensor
            post_video: Post-treatment video tensor
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for feature_idx in self.feature_indices:
            print(f"Processing feature {feature_idx}...")
            
            # Get visualizations for both videos
            pre_frames, pre_heatmaps = self.get_feature_visualization(pre_video, feature_idx)
            post_frames, post_heatmaps = self.get_feature_visualization(post_video, feature_idx)
            
            # Create comparison visualization
            num_frames = min(8, pre_frames.shape[0], post_frames.shape[0])
            fig, axs = plt.subplots(4, num_frames, figsize=(20, 10))
            
            for i in range(num_frames):
                # Pre-treatment frames
                axs[0, i].imshow(pre_frames[i])
                axs[0, i].set_title(f"Pre Frame {i}")
                axs[0, i].axis('off')
                
                axs[1, i].imshow(pre_heatmaps[i])
                axs[1, i].set_title(f"Pre Heatmap {i}")
                axs[1, i].axis('off')
                
                # Post-treatment frames
                axs[2, i].imshow(post_frames[i])
                axs[2, i].set_title(f"Post Frame {i}")
                axs[2, i].axis('off')
                
                axs[3, i].imshow(post_heatmaps[i])
                axs[3, i].set_title(f"Post Heatmap {i}")
                axs[3, i].axis('off')
            
            plt.suptitle(f"Feature {feature_idx} Pre vs Post Comparison")
            plt.tight_layout()
            
            # Save the comparison
            output_path = os.path.join(output_dir, f"feature_{feature_idx}_comparison.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved comparison for feature {feature_idx} to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare pre and post treatment feature visualizations.')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='multimodal_marlin_base',
                        help='MARLIN model name (default: multimodal_marlin_base)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint file (.ckpt)')
    
    # Video parameters
    parser.add_argument('--pre_video', type=str, required=True,
                        help='Path to the pre-treatment video clip')
    parser.add_argument('--post_video', type=str, required=True,
                        help='Path to the post-treatment video clip')
    
    # Feature parameters
    parser.add_argument('--features', type=int, nargs='+', 
                        default=[397, 231, 490, 482, 456],
                        help='Feature indices to visualize (default: 397 231 490 482 456)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='pre_post_comparisons',
                        help='Directory to save comparisons (default: pre_post_comparisons)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model {args.model_name} from {args.checkpoint_path}")
    model = MultiModalMarlin.from_file(args.model_name, args.checkpoint_path)
    
    # Initialize comparator
    comparator = PrePostFeatureComparison(model, args.features)
    
    # Load videos
    print("Loading pre-treatment video...")
    pre_video = read_video(args.pre_video, channel_first=True) / 255.0
    print("Loading post-treatment video...")
    post_video = read_video(args.post_video, channel_first=True) / 255.0
    
    # Process videos to match model's requirements
    clip_frames = model.clip_frames
    
    # Process pre video
    if pre_video.shape[0] < clip_frames:
        pre_video = padding_video(pre_video, clip_frames, "same")
    elif pre_video.shape[0] > clip_frames:
        pre_video = pre_video[:clip_frames]
    pre_video = pre_video.permute(1, 0, 2, 3).unsqueeze(0)
    
    # Process post video
    if post_video.shape[0] < clip_frames:
        post_video = padding_video(post_video, clip_frames, "same")
    elif post_video.shape[0] > clip_frames:
        post_video = post_video[:clip_frames]
    post_video = post_video.permute(1, 0, 2, 3).unsqueeze(0)
    
    # Create comparisons
    print("Creating pre vs post comparisons...")
    comparator.create_comparison(pre_video, post_video, args.output_dir)
    
    print(f"Done! Comparisons saved in {args.output_dir}")


if __name__ == "__main__":
    main() 