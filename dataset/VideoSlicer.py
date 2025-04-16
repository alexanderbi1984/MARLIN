"""
VideoSlicer.py

A utility script for slicing videos into smaller clips with overlapping portions.
This script is designed to process video files in bulk, cutting them into segments
of specified duration with configurable overlap between segments.

Key Features:
- Process single videos or entire directories of videos
- Recursive directory processing (optional)
- Maintains original directory structure in output
- Supports multiple video formats (.mp4, .mov)
- Configurable clip duration and overlap
- High-quality video processing with H.264 codec
- Proper audio handling with AAC codec
- Progress reporting and error handling

Functions:
    slice_video(input_path, output_dir, clip_duration=5, overlap=1):
        Slice a single video into clips of specified duration with overlap.
        
        Args:
            input_path (str): Path to the input video file
            output_dir (str): Directory to save the output clips
            clip_duration (int): Duration of each clip in seconds (default: 5)
            overlap (int): Overlap between consecutive clips in seconds (default: 1)
            
        Returns:
            None
            
        Output:
            Creates multiple video clips in the output directory, named as:
            {original_name}_clip_{number}.mp4

    process_directory(input_dir, output_dir, clip_duration=5, overlap=1, recursive=True):
        Process all video files in a directory (and optionally its subdirectories).
        
        Args:
            input_dir (str): Input directory containing video files
            output_dir (str): Directory to save the output clips
            clip_duration (int): Duration of each clip in seconds (default: 5)
            overlap (int): Overlap between consecutive clips in seconds (default: 1)
            recursive (bool): Whether to process subdirectories recursively (default: True)
            
        Returns:
            None
            
        Output:
            Creates a directory structure matching the input, with sliced video clips
            in each corresponding output directory.

Command Line Usage:
    python VideoSlicer.py --input_dir /path/to/videos --output_dir /path/to/output 
                          --duration 10 --overlap 2 [--no-recursive]

Arguments:
    --input_dir: Directory containing video files (required)
    --output_dir: Directory to save the clips (required)
    --duration: Duration of each clip in seconds (default: 5)
    --overlap: Overlap between consecutive clips in seconds (default: 1)
    --no-recursive: Flag to disable recursive directory processing

Dependencies:
    - ffmpeg: Must be installed and available in system PATH
    - ffprobe: Must be installed and available in system PATH

Example:
    To slice all videos in a directory into 10-second clips with 2-second overlaps:
    python VideoSlicer.py --input_dir ./videos --output_dir ./clips --duration 10 --overlap 2

Notes:
    - Supported video formats: .mp4, .mov (case insensitive)
    - Output videos use H.264 video codec and AAC audio codec
    - Original directory structure is preserved in the output
    - Progress is reported for each video being processed
    - Errors are caught and reported without stopping the entire process
"""

import os
import argparse
import subprocess
from pathlib import Path


def slice_video(input_path, output_dir, clip_duration=5, overlap=1):
    """
    Slice a video into clips of specified duration with overlap.

    Args:
        input_path (str): Path to the input video file
        output_dir (str): Directory to save the output clips
        clip_duration (int): Duration of each clip in seconds
        overlap (int): Overlap between consecutive clips in seconds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get video information using ffprobe
    duration_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ]

    try:
        # Get video duration
        duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())

        # Calculate the number of clips
        step = clip_duration - overlap
        num_clips = max(1, int((duration - overlap) / step))

        # Get the base name of the input file (without extension)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        print(f"Processing '{base_name}' ({duration:.2f}s) into {num_clips} clips...")

        # Create each clip
        for i in range(num_clips):
            start_time = i * step

            # Ensure the final clip doesn't exceed the video duration
            if start_time + clip_duration > duration:
                end_time = duration
            else:
                end_time = start_time + clip_duration

            # Format output filename
            output_filename = f"{base_name}_clip_{i + 1:03d}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            # FFmpeg command to extract the clip
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-i', input_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c:v', 'libx264',  # Use H.264 codec for video
                '-c:a', 'aac',  # Use AAC codec for audio
                '-strict', 'experimental',
                '-b:a', '128k',  # Audio bitrate
                output_path
            ]

            # Execute FFmpeg command
            print(f"  Creating clip {i + 1}/{num_clips}: {start_time:.2f}s to {end_time:.2f}s")
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f"Finished processing '{base_name}': {num_clips} clips created.")

    except subprocess.CalledProcessError as e:
        print(f"Error processing '{input_path}': {e}")
    except Exception as e:
        print(f"Unexpected error processing '{input_path}': {e}")


def process_directory(input_dir, output_dir, clip_duration=5, overlap=1, recursive=True):
    """
    Process all video files in a directory.

    Args:
        input_dir (str): Input directory containing video files
        output_dir (str): Directory to save the output clips
        clip_duration (int): Duration of each clip in seconds
        overlap (int): Overlap between consecutive clips in seconds
        recursive (bool): Whether to process subdirectories recursively
    """
    # Create a Path object for the input directory
    input_path = Path(input_dir)

    # Use a single pattern with case-insensitive matching
    # This avoids counting the same files twice due to case differences
    video_files = []
    
    if recursive:
        # For recursive search, use rglob with a case-insensitive pattern
        for file_path in input_path.rglob("*"):
            if file_path.suffix.lower() in ['.mp4', '.mov']:
                video_files.append(file_path)
    else:
        # For non-recursive search, use glob with a case-insensitive pattern
        for file_path in input_path.glob("*"):
            if file_path.suffix.lower() in ['.mp4', '.mov']:
                video_files.append(file_path)

    if not video_files:
        print(f"No video files found in '{input_dir}'")
        return

    print(f"Found {len(video_files)} video files to process.")

    # Process each video file
    for i, video_file in enumerate(video_files):
        print(f"\nProcessing file {i + 1}/{len(video_files)}: {video_file}")

        # Create a subdirectory in the output directory with the same relative path
        rel_path = video_file.relative_to(input_path).parent
        clip_output_dir = os.path.join(output_dir, rel_path)

        # Slice the video
        slice_video(str(video_file), clip_output_dir, clip_duration, overlap)

    print(f"\nAll {len(video_files)} video files processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice videos into clips with overlap")
    parser.add_argument("--input_dir", help="Input directory containing video files")
    parser.add_argument("--output_dir", help="Output directory for the clips")
    parser.add_argument("--duration", type=int, default=5, help="Duration of each clip in seconds (default: 5)")
    parser.add_argument("--overlap", type=int, default=1,
                        help="Overlap between consecutive clips in seconds (default: 1)")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive",
                        help="Do not process subdirectories recursively")

    args = parser.parse_args()

    # Process the directory
    process_directory(args.input_dir, args.output_dir, args.duration, args.overlap, args.recursive)