import os
import argparse
import subprocess
from pathlib import Path
import re
import shutil


def assemble_frames(input_dir, output_dir, fps=30, clip_duration=5, overlap=1):
    """
    Assemble frames into clips of specified duration with overlap.

    Args:
        input_dir (str): Directory containing the extracted frames
        output_dir (str): Directory to save the output clips
        fps (int): Frames per second of the original video
        clip_duration (int): Duration of each clip in seconds
        overlap (int): Overlap between consecutive clips in seconds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files in the directory
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])

    if not image_files:
        print(f"No image files found in '{input_dir}'")
        return

    # Extract base name from the directory
    base_name = os.path.basename(input_dir)

    # Total number of frames
    total_frames = len(image_files)

    # Calculate frames per clip and overlap in frames
    frames_per_clip = clip_duration * fps
    overlap_frames = overlap * fps
    step_frames = frames_per_clip - overlap_frames

    # Calculate number of clips
    num_clips = max(1, int((total_frames - overlap_frames) / step_frames) +
                    (1 if (total_frames - overlap_frames) % step_frames > 0 else 0))

    print(f"Processing '{base_name}' ({total_frames} frames) into {num_clips} clips...")

    # Process each clip
    for i in range(num_clips):
        # Calculate start and end frame indices
        start_frame = i * step_frames
        end_frame = min(start_frame + frames_per_clip, total_frames)

        # Skip if we don't have enough frames for a meaningful clip
        if end_frame - start_frame < fps:  # At least 1 second of content
            continue

        # Create temp directory for the frames of this clip
        temp_dir = os.path.join(output_dir, f"temp_clip_{i + 1}")
        os.makedirs(temp_dir, exist_ok=True)

        # Copy frames for this clip to temp directory with sequential naming
        for j, frame_idx in enumerate(range(start_frame, end_frame)):
            if frame_idx < total_frames:
                src_file = os.path.join(input_dir, image_files[frame_idx])
                # Ensure proper sequential naming for ffmpeg
                dst_file = os.path.join(temp_dir, f"frame_{j:06d}{os.path.splitext(image_files[frame_idx])[1]}")
                shutil.copy2(src_file, dst_file)

        # Output filename
        output_filename = f"{base_name}_clip_{i + 1:03d}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # Use ffmpeg to assemble frames into a video
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, f"frame_%06d{os.path.splitext(image_files[0])[1]}"),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',  # Quality setting
            output_path
        ]

        print(f"  Creating clip {i + 1}/{num_clips}: frames {start_frame} to {end_frame - 1}")
        subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

    print(f"Finished processing '{base_name}': {num_clips} clips created.")


def process_frame_directories(input_base_dir, output_dir, fps=30, clip_duration=5, overlap=1, recursive=True):
    """
    Process all frame directories in a base directory.

    Args:
        input_base_dir (str): Base directory containing frame directories
        output_dir (str): Directory to save the output clips
        fps (int): Frames per second of the original video
        clip_duration (int): Duration of each clip in seconds
        overlap (int): Overlap between consecutive clips in seconds
        recursive (bool): Whether to process subdirectories recursively
    """
    # Create a Path object for the input directory
    input_path = Path(input_base_dir)

    # Find directories that likely contain frames
    # This assumes frame directories have image files in them
    frame_dirs = []

    if recursive:
        # Check all subdirectories
        for dir_path in input_path.glob('**/*'):
            if dir_path.is_dir():
                # Check if directory contains image files
                has_images = any(
                    f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] for f in dir_path.iterdir() if f.is_file())
                if has_images:
                    frame_dirs.append(dir_path)
    else:
        # Check only immediate subdirectories
        for dir_path in input_path.iterdir():
            if dir_path.is_dir():
                # Check if directory contains image files
                has_images = any(
                    f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] for f in dir_path.iterdir() if f.is_file())
                if has_images:
                    frame_dirs.append(dir_path)

    if not frame_dirs:
        print(f"No frame directories found in '{input_base_dir}'")
        return

    print(f"Found {len(frame_dirs)} frame directories to process.")

    # Process each frame directory
    for i, frame_dir in enumerate(frame_dirs):
        print(f"\nProcessing directory {i + 1}/{len(frame_dirs)}: {frame_dir}")

        # Create a subdirectory in the output directory with the same relative path
        rel_path = frame_dir.relative_to(input_path).parent
        clip_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(clip_output_dir, exist_ok=True)

        # Assemble frames into clips
        assemble_frames(str(frame_dir), clip_output_dir, fps, clip_duration, overlap)

    print(f"\nAll {len(frame_dirs)} frame directories processed successfully.")


def identify_openface_directories(input_base_dir, recursive=True):
    """
    Identify directories that contain OpenFace extracted frames.

    Args:
        input_base_dir (str): Base directory to search
        recursive (bool): Whether to search subdirectories recursively

    Returns:
        list: List of directories that appear to contain OpenFace frames
    """
    input_path = Path(input_base_dir)
    openface_dirs = []

    # Common patterns in OpenFace output directories or files
    openface_patterns = [
        r'.*_aligned',  # Aligned face directories
        r'.*\.csv',  # OpenFace CSV output files
        r'.*_landmarks.*\.txt',  # Landmark files
        r'frame_\d+\.jpg',  # Common frame naming pattern
        r'.*_0+\d+\.jpg'  # Alternative frame naming pattern
    ]

    # Compile the regex patterns
    regex_patterns = [re.compile(pattern) for pattern in openface_patterns]

    def is_openface_dir(dir_path):
        """Check if a directory looks like it contains OpenFace output"""
        # Get all files in the directory
        try:
            files = list(os.listdir(dir_path))

            # If directory is empty, it's not an OpenFace directory
            if not files:
                return False

            # Check for OpenFace patterns in file names
            for file in files:
                for pattern in regex_patterns:
                    if pattern.match(file):
                        return True

            # Check for image files with numeric names (common in OpenFace output)
            image_count = 0
            numeric_image_count = 0

            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    image_count += 1
                    # Check if filename is numeric or has a numeric pattern
                    base_name = os.path.splitext(file)[0]
                    if base_name.isdigit() or re.match(r'.*\d{4,}.*', base_name):
                        numeric_image_count += 1

            # If a significant portion of images have numeric names, it's likely OpenFace output
            if image_count > 10 and numeric_image_count / image_count > 0.8:
                return True

            return False

        except (PermissionError, FileNotFoundError):
            return False

    # Find directories that might contain OpenFace output
    if recursive:
        for dir_path in input_path.glob('**/*'):
            if dir_path.is_dir() and is_openface_dir(dir_path):
                openface_dirs.append(dir_path)
    else:
        for dir_path in input_path.iterdir():
            if dir_path.is_dir() and is_openface_dir(dir_path):
                openface_dirs.append(dir_path)

    return openface_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble frames into video clips with overlap")
    parser.add_argument("--input_dir", help="Input directory containing frame directories")
    parser.add_argument("--output_dir", help="Output directory for the assembled clips")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--duration", type=int, default=5, help="Duration of each clip in seconds (default: 5)")
    parser.add_argument("--overlap", type=int, default=1,
                        help="Overlap between consecutive clips in seconds (default: 1)")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive",
                        help="Do not process subdirectories recursively")
    parser.add_argument("--openface", action="store_true", help="Automatically identify OpenFace output directories")

    args = parser.parse_args()

    if args.openface:
        # Identify OpenFace directories and process them
        print(f"Scanning for OpenFace frame directories in '{args.input_dir}'...")
        openface_dirs = identify_openface_directories(args.input_dir, args.recursive)

        if not openface_dirs:
            print("No OpenFace frame directories found.")
            exit(1)

        print(f"Found {len(openface_dirs)} potential OpenFace frame directories.")

        # Process each OpenFace directory
        for i, frame_dir in enumerate(openface_dirs):
            print(f"\nProcessing OpenFace directory {i + 1}/{len(openface_dirs)}: {frame_dir}")

            # Create a subdirectory in the output directory with the same relative path
            rel_path = frame_dir.relative_to(Path(args.input_dir)).parent
            clip_output_dir = os.path.join(args.output_dir, rel_path)
            os.makedirs(clip_output_dir, exist_ok=True)

            # Assemble frames into clips
            assemble_frames(str(frame_dir), clip_output_dir, args.fps, args.duration, args.overlap)

        print(f"\nAll {len(openface_dirs)} OpenFace frame directories processed successfully.")
    else:
        # Process directories manually
        process_frame_directories(args.input_dir, args.output_dir, args.fps, args.duration, args.overlap,
                                  args.recursive)