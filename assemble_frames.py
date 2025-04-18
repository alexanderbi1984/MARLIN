"""
Video Frame Assembly Script

This script assembles a sequence of BMP image frames into MP4 video files using FFmpeg.
It processes subdirectories containing frame sequences and creates corresponding MP4 videos.

Features:
- Processes multiple subdirectories containing frame sequences
- Supports custom frame rate (default: 30 fps)
- Uses H.264 codec for video compression
- Maintains frame sequence order
- Provides detailed progress and error reporting

Requirements:
- FFmpeg must be installed and accessible in system PATH
- Input frames must be named in the format: frame_det_00_XXXXXX.bmp
- Frames should be numbered sequentially

Usage:
    python assemble_frames.py

Input/Output Structure:
    Input Directory Structure:
        input_directory/
            subfolder_1/
                frame_det_00_000001.bmp
                frame_det_00_000002.bmp
                ...
            subfolder_2/
                frame_det_00_000001.bmp
                frame_det_00_000002.bmp
                ...

    Output Structure:
        input_directory/
            subfolder_1.mp4
            subfolder_2.mp4
            ...

The script will process all subdirectories in the specified input directory,
looking for BMP frame sequences and creating corresponding MP4 videos in the same directory.
Each output video will have the same name as its source subfolder with a .mp4 extension.
"""

import os
import subprocess
from pathlib import Path

def assemble_frames_to_video(input_dir, fps=30):
    """
    Assemble frames within each subfolder into a video using ffmpeg.
    Args:
        input_dir (str): Path to the directory containing frame subfolders
        fps (int): Frames per second for the output video
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return

    # Process each subfolder
    for subfolder in input_path.iterdir():
        if not subfolder.is_dir():
            continue

        print(f"\nProcessing folder: {subfolder}")
        
        # List all files in the directory to debug
        all_files = list(subfolder.glob("frame_det_00_*.bmp"))
        print(f"Total .bmp files found: {len(all_files)}")
        if all_files:
            print("First few files:", [f.name for f in all_files[:5]])
            
            # Create output video path using subfolder name
            output_video = input_path / f"{subfolder.name}.mp4"
            
            # Construct ffmpeg command using the exact filename pattern
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-framerate", str(fps),
                "-i", str(subfolder / "frame_det_00_%06d.bmp"),  # Input pattern matching actual filenames
                "-c:v", "libx264",  # Use H.264 codec
                "-pix_fmt", "yuv420p",  # Pixel format for compatibility
                "-crf", "23",  # Constant Rate Factor (quality)
                str(output_video)
            ]

            try:
                print(f"Processing {subfolder.name}...")
                print(f"Running command: {' '.join(ffmpeg_cmd)}")
                # Sort the files to ensure they're in the correct order
                frame_files = sorted(all_files, key=lambda x: int(x.stem.split('_')[-1]))
                if frame_files:
                    # Check if frame numbers are continuous
                    first_frame = int(frame_files[0].stem.split('_')[-1])
                    last_frame = int(frame_files[-1].stem.split('_')[-1])
                    expected_count = last_frame - first_frame + 1
                    if len(frame_files) != expected_count:
                        print(f"Warning: Frame sequence may have gaps. Found {len(frame_files)} frames, expected {expected_count}")
                    print(f"Frame range: {first_frame} to {last_frame}")
                subprocess.run(ffmpeg_cmd, check=True)
                print(f"Successfully created {output_video}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {subfolder.name}: {e}")
            except Exception as e:
                print(f"Unexpected error processing {subfolder.name}: {e}")
        else:
            print(f"No .bmp files found in {subfolder}")

if __name__ == "__main__":
    input_directory = r"C:\pain\syracus\syracuse_pain_fake_cont"
    assemble_frames_to_video(input_directory) 