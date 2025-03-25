import os
import subprocess
from pathlib import Path

def extract_frames(video_path, output_dir):
    # Create output directory for this video
    video_name = Path(video_path).stem
    frames_dir = Path(output_dir) / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct ffmpeg command to:
    # 1. Read the video
    # 2. Rotate 90 degrees clockwise (transpose=1)
    # 3. Extract frames at 30 fps
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', 'transpose=1',  # 90 degrees clockwise
        '-r', '30',           # ensure 30 fps output
        '-frame_pts', '1',    # include presentation timestamp
        str(frames_dir / 'frame_%04d.png')  # output pattern
    ]
    
    print(f"Extracting frames from {video_path}...")
    subprocess.run(ffmpeg_cmd)
    print(f"Frames saved to {frames_dir}")
    return frames_dir

def main():
    # Input and output directories
    input_dir = Path(r"C:\pain\syracus\openface_frame_cont")
    output_dir = Path(r"C:\pain\syracus\openface_frame_cont\extracted_frames")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Process each video
    for video_file in input_dir.glob("IMG_*.mp4"):
        extract_frames(video_file, output_dir)

if __name__ == "__main__":
    main() 