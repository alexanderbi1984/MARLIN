import argparse
import torch
import numpy as np
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm.auto import tqdm
from model.classifier import Classifier  # Replace with your model class
from marlin_pytorch.config import resolve_config
from marlin_pytorch.util import read_yaml
from util.seed import Seed
import ffmpeg

def load_model(ckpt):
    """
    Load a trained model from a checkpoint.
    """
    model = Classifier.load_from_checkpoint(ckpt)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_video(video_path, clip_frames, temporal_sample_rate):
    """
    Load and preprocess a single video for inference.
    """
    try:
        probe = ffmpeg.probe(video_path)["streams"][0]
    except ffmpeg._run.Error as e:
        print(f"Error probing video: {video_path}")
        print(f"FFmpeg error: {e.stderr.decode()}")
    n_frames = int(probe["nb_frames"])

    if n_frames <= clip_frames:
        try:
            print(f"Reading video: {video_path}")
            print(f"Number of frames: {n_frames}")
            video = read_video(video_path, channel_first=True).video / 255
        except Exception as e:
            print(f"Error reading video: {video_path}")
            print(f"FFmpeg error: {e.stderr.decode()}")

        # video, audio = read_video(video_path, channel_first=True)
        # video = video / 255.0  # Normalize to [0, 1]
        # video = read_video(video_path, channel_first=True).video / 255
        # pad frames to 16
        # Check the shape of the video tensor
        # if video.ndim == 3:  # Shape: (C, H, W)
        #     video = video.unsqueeze(0)  # Add a dimension for T: (1, C, H, W)
        #
        # elif video.ndim != 4:
        #     raise ValueError(f"Unexpected video shape: {video.shape}")
        video = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        return video, torch.tensor(y, dtype=torch.long)
    elif n_frames <= self.clip_frames * self.temporal_sample_rate:
        # reset a lower temporal sample rate
        sample_rate = n_frames // self.clip_frames
    else:
        sample_rate = self.temporal_sample_rate
    # sample frames
    video_indexes = sample_indexes(n_frames, self.clip_frames, sample_rate)
    reader = torchvision.io.VideoReader(video_path)
    fps = reader.get_metadata()["video"]["fps"][0]
    reader.seek(video_indexes[0].item() / fps, True)
    frames = []
    for frame in islice(reader, 0, self.clip_frames * sample_rate, sample_rate):
        frames.append(frame["data"])
    video = torch.stack(frames) / 255  # (T, C, H, W)
    video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
    assert video.shape[1] == self.clip_frames, video_path
    # print(f"y value before returning is {y}")
    # return video, torch.tensor(y, dtype=torch.long).bool()
    return video, torch.tensor(y, dtype=torch.long)

    return video.unsqueeze(0)  # Add batch dimension: (1, T, C, H, W)

def predict(model, video, task, device="cpu"):
    """
    Perform inference on a single video.
    """
    model.to(device)
    video = video.to(device)
    with torch.no_grad():
        preds = model(video)
        if task == "multiclass" or task == "binary":
            preds = torch.softmax(preds, dim=1)  # Apply softmax for classification tasks
    return preds.cpu().numpy()

def save_predictions(predictions, output_path, task):
    """
    Save predictions to a file.
    """
    if task == "regression":
        np.savetxt(output_path, predictions, delimiter=",")
    else:
        np.save(output_path, predictions)
    print(f"Predictions saved to {output_path}")

def main(args):
    # Load configuration
    config = read_yaml(args.config)

    # Set task-specific parameters
    task = config["task"]
    if config["dataset"] == "celebvhq":
        if task == "appearance":
            num_classes = 40
        elif task == "action":
            num_classes = 35
        else:
            raise ValueError(f"Unknown task: {task}")
    elif config["dataset"] == "biovid":
        if task == "binary" or task == "regression":
            num_classes = 1
        elif task == "multiclass":
            num_classes = 5
        else:
            raise ValueError(f"Unknown task: {task}")
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")

    # Load the model
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        task=task,
        num_classes=num_classes,
        backbone=config["backbone"],
        finetune=config["finetune"],
        marlin_ckpt=args.marlin_ckpt,
        learning_rate=config["learning_rate"],
        use_ddp=args.n_gpus > 1,
    )

    # Preprocess the input video
    backbone_config = resolve_config(config["backbone"]) if config["finetune"] else None
    clip_frames = backbone_config.n_frames if config["finetune"] else None
    temporal_sample_rate = 2 if config["finetune"] else None

    video = preprocess_video(
        video_path=args.video_path,
        clip_frames=clip_frames,
        temporal_sample_rate=temporal_sample_rate,
    )

    # Perform inference
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    predictions = predict(model, video, task=task, device=device)

    # Save or display predictions
    if args.output_path:
        save_predictions(predictions, args.output_path, task=task)
    else:
        print("Predictions:", predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save predictions.")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--marlin_ckpt", type=str, default=None, help="Path to MARLIN checkpoint.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = parser.parse_args()

    # Set random seed for reproducibility
    Seed.set(42)

    main(args)