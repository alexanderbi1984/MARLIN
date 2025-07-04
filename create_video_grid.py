import os
import subprocess
from pathlib import Path

def create_video_grid(input_dir, output_video, grid_width=2, grid_height=2, speed_factor=0.5):
    """
    Create a video grid from multiple input videos using FFmpeg.
    Args:
        input_dir: Directory containing input videos
        output_video: Path for the output video
        grid_width: Number of columns in the grid (default: 2)
        grid_height: Number of rows in the grid (default: 2)
        speed_factor: Video speed factor (0.5 for half speed)
    """
    # Get list of video files and sort them properly
    video_files = []
    for f in os.listdir(input_dir):
        if f.endswith(('.mp4', '.avi', '.mov')):
            if f == 'original_female_moderate.mp4':
                video_files.append((0, f))  # Original video gets highest priority
            elif f == 'female_mild.mp4':
                video_files.append((1, f))
            elif f == 'female_moderate.mp4':
                video_files.append((2, f))
            elif f == 'female_severe.mp4':
                video_files.append((3, f))
    
    # Sort the videos based on their priority
    video_files.sort(key=lambda x: x[0])
    video_files = [f[1] for f in video_files]
    
    if len(video_files) != grid_width * grid_height:
        print(f"Warning: Expected {grid_width * grid_height} videos, found {len(video_files)}")
        if len(video_files) > grid_width * grid_height:
            video_files = video_files[:grid_width * grid_height]
        else:
            print("Error: Not enough videos for the grid")
            return
    
    # Create FFmpeg filter complex string
    inputs = []
    filter_complex = []
    
    # Add input files
    for i, video in enumerate(video_files):
        inputs.extend(['-i', os.path.join(input_dir, video)])
    
    # Create grid layout
    for i in range(len(video_files)):
        # Scale video, slow down, and add text overlay
        video_name = os.path.splitext(video_files[i])[0]
        # Remove 'female_' prefix from display text
        display_name = video_name.replace('female_', '')
        filter_complex.append(
            f'[{i}:v]scale=640:480,setpts={1/speed_factor}*PTS,drawtext=text=\'{display_name}\':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=h-th-10[v{i}]'
        )
    
    # Create horizontal stacks (rows)
    for row in range(grid_height):
        row_inputs = []
        for col in range(grid_width):
            idx = row * grid_width + col
            row_inputs.append(f'[v{idx}]')
        filter_complex.append(f"{''.join(row_inputs)}hstack=inputs={grid_width}[row{row}]")
    
    # Stack rows vertically
    row_inputs = ''.join(f'[row{i}]' for i in range(grid_height))
    filter_complex.append(f'{row_inputs}vstack=inputs={grid_height}[v]')
    
    # Construct FFmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
    ] + inputs + [
        '-filter_complex', ';'.join(filter_complex),
        '-map', '[v]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-an',  # No audio
        output_video
    ]
    
    print("Running FFmpeg command...")
    print(' '.join(ffmpeg_cmd))
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Successfully created video grid: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video grid: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a video grid from multiple videos')
    parser.add_argument('input_dir', help='Directory containing input videos')
    parser.add_argument('output_video', help='Path for the output video file')
    parser.add_argument('--width', type=int, default=2, help='Number of columns in the grid (default: 2)')
    parser.add_argument('--height', type=int, default=2, help='Number of rows in the grid (default: 2)')
    parser.add_argument('--speed', type=float, default=0.5, help='Video speed factor (default: 0.5 for half speed)')
    
    args = parser.parse_args()
    
    create_video_grid(args.input_dir, args.output_video, args.width, args.height, args.speed) 