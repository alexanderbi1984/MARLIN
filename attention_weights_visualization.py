import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import cv2

from model.attention_classifier import AttentionClassifier
from dataset.biovid import BioVidDataModule
from torch.utils.data import DataLoader


def extract_attention_weights(model, dataloader):
    """
    Extract clip-level attention weights from the model for each sample in the dataloader.

    Note: The Marlin encoder processes videos as sequences of 16-frame clips.
    The attention weights extracted here correspond to these clips, not individual frames.

    Args:
        model: Trained AttentionClassifier model
        dataloader: DataLoader containing samples to visualize

    Returns:
        tuple: (attention_weights, predictions, ground_truth, video_paths)
    """
    model.eval()
    attention_weights_list = []
    predictions_list = []
    ground_truth_list = []
    paths_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting attention weights"):
            # Unpack the batch (features or video, labels, paths)
            if len(batch) == 3:
                x, y, paths = batch
            else:
                x, y = batch
                paths = ["unknown"] * len(x)

            # Forward pass through the model
            if model.model is not None:
                # Get features from backbone, keeping sequence dimension
                features = model.model.extract_features(x.to(model.device), keep_seq=True)
            else:
                features = x.to(model.device)

            # Single-head attention
            if model.num_heads == 1:
                batch_size, seq_len, feature_dim = features.shape

                # Project inputs to query/key space
                queries = model.query(features)
                keys = model.key(features)

                # Compute attention scores
                scores = torch.bmm(queries, keys.transpose(1, 2))
                scores = scores / (model.attention_dim ** 0.5)

                # Apply softmax to get attention weights
                attention_weights = torch.softmax(scores, dim=2)

                # For single-head attention, we're interested in the weights
                # applied to each clip (diagonal of attention matrix)
                # Note: each clip represents 16 frames
                clip_importance = torch.diagonal(attention_weights, dim1=1, dim2=2)

                attention_weights_list.append(clip_importance.cpu().numpy())
            else:
                # For multi-head attention, we need to handle it differently
                # Re-compute attention weights manually since PyTorch's multihead_attention
                # doesn't expose the weights directly
                _, attention_weights = model.attention(features, features, features,
                                                       need_weights=True, average_attn_weights=False)

                # Average across heads
                attention_weights = attention_weights.mean(dim=0)

                # Extract diagonal for clip importance
                # Note: each clip represents 16 frames
                clip_importance = torch.diagonal(attention_weights, dim1=1, dim2=2)

                attention_weights_list.append(clip_importance.cpu().numpy())

            # Get predictions
            outputs = model.fc(model.apply_attention(features))

            if model.task == "binary":
                predictions = torch.sigmoid(outputs) > 0.5
            elif model.task == "multiclass":
                predictions = torch.argmax(outputs, dim=1)
            else:  # regression
                predictions = outputs

            predictions_list.append(predictions.cpu().numpy())
            ground_truth_list.append(y.cpu().numpy())
            paths_list.extend(paths)

    # Concatenate results
    attention_weights = np.vstack(attention_weights_list)
    predictions = np.concatenate(predictions_list)
    ground_truth = np.concatenate(ground_truth_list)

    return attention_weights, predictions, ground_truth, paths_list


def visualize_attention(attention_weights, predictions, ground_truth, video_paths,
                        output_dir, num_samples=5, save_videos=False):
    """
    Visualize clip-level attention weights for selected samples.

    Args:
        attention_weights: Numpy array of attention weights (one per 16-frame clip)
        predictions: Numpy array of model predictions
        ground_truth: Numpy array of ground truth labels
        video_paths: List of paths to the original videos
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        save_videos: Whether to save videos with attention overlay
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a custom colormap: blue (low attention) to red (high attention)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    cmap_name = 'attention_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

    # Randomly select samples to visualize
    indices = np.random.choice(len(attention_weights),
                               min(num_samples, len(attention_weights)),
                               replace=False)

    for i, idx in enumerate(indices):
        # Get data for this sample
        sample_weights = attention_weights[idx]
        prediction = predictions[idx]
        truth = ground_truth[idx]
        video_path = video_paths[idx]

        # Plot attention weights
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(sample_weights, '-o', markersize=6)
        plt.title(f"Sample {i + 1} - Pred: {prediction}, Truth: {truth}")
        plt.xlabel("Clip Index (each clip = 16 frames)")
        plt.ylabel("Attention Weight")
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.imshow([sample_weights], aspect='auto', cmap=cm)
        plt.colorbar(label="Attention Weight")
        plt.yticks([])
        plt.xlabel("Clip Index (each clip = 16 frames)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"attention_sample_{i + 1}.png"), dpi=150)
        plt.close()

        # If requested, create a video with attention overlay
        if save_videos and video_path != "unknown" and os.path.exists(video_path):
            create_attention_video(video_path, sample_weights,
                                   os.path.join(output_dir, f"attention_video_{i + 1}.mp4"))


def create_attention_video(video_path, attention_weights, output_path, fps=30, frames_per_clip=16):
    """
    Create a video with attention weights overlay.

    Args:
        video_path: Path to the original video
        attention_weights: Attention weights for each clip (each clip is 16 frames)
        output_path: Path to save the output video
        fps: Frames per second
        frames_per_clip: Number of frames in each clip (default: 16 for Marlin)
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Normalize attention weights to [0, 1] for visualization

    # Convert clip-level weights to frame-level weights by expanding
    # Each clip-level weight needs to be applied to all frames in that clip
    frame_level_weights = np.repeat(attention_weights, frames_per_clip)

    # If the expanded weights don't match the video frame count, interpolate to fit
    if len(frame_level_weights) != total_frames:
        # Interpolate weights if necessary
        frame_indices = np.linspace(0, len(frame_level_weights) - 1, total_frames)
        frame_level_weights = np.interp(frame_indices,
                                        np.arange(len(frame_level_weights)),
                                        frame_level_weights)

    frame_level_weights = (frame_level_weights - frame_level_weights.min()) / (
                frame_level_weights.max() - frame_level_weights.min())

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get attention weight for this frame
        if frame_idx < len(frame_level_weights):
            weight = frame_level_weights[frame_idx]

            # Create a heatmap overlay
            heatmap = np.zeros((height, width), dtype=np.uint8)
            heatmap = cv2.applyColorMap(
                np.uint8(255 * np.ones((height, width)) * weight),
                cv2.COLORMAP_JET
            )

            # Add overlay to the frame
            overlay = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

            # Add text with attention weight
            cv2.putText(overlay, f"Attention: {weight:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write the frame
            out.write(overlay)

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser("Visualize Attention Weights")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to visualize")
    parser.add_argument("--output_dir", type=str, default="./attention_viz", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--save_videos", action="store_true", help="Save videos with attention overlay")

    args = parser.parse_args()

    # Load the model
    model = AttentionClassifier.load_from_checkpoint(args.model_path)
    model.eval()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get model hyperparameters
    hparams = model.hparams

    # Create data module
    dm = BioVidDataModule(
        root_dir=args.data_path,
        load_raw=hparams.finetune,  # Match what was used for training
        task=hparams.task,
        num_classes=hparams.num_classes,
        batch_size=args.batch_size,
        num_workers=4,
        feature_dir=hparams.backbone if not hparams.finetune else None,
        temporal_reduction="none",  # Don't reduce for attention visualization
        clip_frames=16 if hparams.finetune else None,  # Default for MARLIN
        temporal_sample_rate=2 if hparams.finetune else None
    )

    # Setup data module
    dm.setup()

    # Get the appropriate dataloader
    if args.split == "train":
        dataloader = dm.train_dataloader()
    elif args.split == "val":
        dataloader = dm.val_dataloader()
    else:
        dataloader = dm.test_dataloader()

    # Extract attention weights
    attention_weights, predictions, ground_truth, video_paths = extract_attention_weights(
        model, dataloader
    )

    # Visualize attention weights
    visualize_attention(
        attention_weights, predictions, ground_truth, video_paths,
        args.output_dir, args.num_samples, args.save_videos
    )

    print(f"Attention visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()