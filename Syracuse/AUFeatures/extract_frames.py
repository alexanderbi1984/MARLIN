import os
import subprocess
from pathlib import Path

def extract_frames(video_path, output_dir, frame_rate=30):
    """
    Extract frames from a video using ffmpeg.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames
        frame_rate (int): Frame rate for extraction (frames per second)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct ffmpeg command
    # -i: input file
    # -vf fps=30: extract 30 frames per second
    # -q:v 2: high quality (1-31, lower is better)
    # %06d: frame number with 6 digits
    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
    
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={frame_rate}',
        '-q:v', '2',
        output_pattern
    ]
    
    print(f"Extracting frames from {video_path}")
    print(f"Saving frames to {output_dir}")
    print(f"Command: {' '.join(command)}")
    
    try:
        # Run ffmpeg command
        subprocess.run(command, check=True)
        print("Frame extraction completed successfully!")
        
        # Count number of frames extracted
        frame_count = len([f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.jpg')])
        print(f"Extracted {frame_count} frames")
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set paths
    video_path = r"C:\pain\syracus\syracuse_pain_videos\026\IMG_0061.MOV"
    output_dir = os.path.join(os.path.dirname(video_path), "frames61")
    
    # Extract frames
    extract_frames(video_path, output_dir) 