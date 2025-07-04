import cv2
import os
import numpy as np
import argparse
from pathlib import Path

def extract_first_frames_and_create_grid(input_videos, output_image, first_frame_path=None, captions=None):
    """
    Extract the first frame from each video and create a grid image with captions.
    
    Args:
        input_videos (list): List of paths to input video files
        output_image (str): Path to save the output grid image
        first_frame_path (str, optional): Path to the first frame image to use instead of extracting from video
        captions (list, optional): List of captions for each frame. If None, default captions will be used.
    """
    frames = []
    
    # Add the provided first frame if available
    if first_frame_path and os.path.exists(first_frame_path):
        print(f"Using provided first frame: {first_frame_path}")
        first_frame = cv2.imread(first_frame_path)
        if first_frame is not None:
            frames.append(first_frame)
        else:
            print(f"Error: Could not read image from {first_frame_path}")
    
    # Extract first frame from each video
    for i, video_path in enumerate(input_videos):
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
        
        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from {video_path}")
            continue
        
        # Store the frame
        frames.append(frame)
        
        # Release video capture
        cap.release()
    
    if not frames:
        print("Error: No frames were extracted")
        return
    
    # Resize frames to same dimensions (use dimensions of first frame)
    target_height, target_width = frames[0].shape[:2]
    for i in range(len(frames)):
        if frames[i].shape[:2] != (target_height, target_width):
            frames[i] = cv2.resize(frames[i], (target_width, target_height))
    
    # Create default captions if not provided
    if captions is None:
        captions = ["Original"]
        captions.extend([f"Augmented ver{i+1}" for i in range(len(frames)-1)])
    
    # Ensure we have a caption for each frame
    captions = captions[:len(frames)]
    while len(captions) < len(frames):
        captions.append(f"Frame {len(captions)+1}")
    
    # Create a row of images
    # Add space for caption text
    caption_height = 40
    grid_height = target_height + caption_height
    grid_width = target_width * len(frames)
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Add frames and captions to the grid
    for i, frame in enumerate(frames):
        # Calculate position
        x_offset = i * target_width
        
        # Add frame
        grid_image[:target_height, x_offset:x_offset+target_width] = frame
        
        # Add caption
        cv2.rectangle(grid_image, 
                     (x_offset, target_height), 
                     (x_offset+target_width, grid_height), 
                     (0, 0, 0), 
                     -1)  # Black background
        
        cv2.putText(grid_image, 
                   captions[i], 
                   (x_offset + 10, target_height + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, 
                   (255, 255, 255), 
                   2, 
                   cv2.LINE_AA)
    
    # Save the grid image
    os.makedirs(os.path.dirname(output_image) if os.path.dirname(output_image) else '.', exist_ok=True)
    cv2.imwrite(output_image, grid_image)
    print(f"Grid image saved to: {output_image}")

if __name__ == "__main__":
    # Direct usage without command line arguments
    
    # Example 1: Using sample videos from test/input_sample with custom captions
    def process_sample_videos():
        sample_dir = Path('test/input_sample')
        if not sample_dir.exists():
            print(f"Sample directory {sample_dir} not found.")
            return
            
        # Get video files
        video_files = list(sample_dir.glob('*.mp4'))[:5]  # Get up to 5 videos
        video_files = [str(f) for f in video_files]
        
        if not video_files:
            print(f"No video files found in {sample_dir}")
            return
            
        # Custom captions for each video
        captions = ["Original", "Augmented ver1", "Augmented ver2", "Augmented ver3", "Augmented ver4"]
        
        # Process videos
        output_image = "sample_frames_grid.jpg"
        extract_first_frames_and_create_grid(video_files, output_image, captions=captions)
    
    # Example 2: Using specific videos with custom captions and a provided first frame
    def process_specific_videos():
        # Path to the first frame image (instead of extracting from video)
        first_frame_path = "E:\\Pain\\syracus\\syracus\\syracuse_pain_clips\\IMG_0003_clip_001_aligned\\frame_det_00_000001.bmp"
        
        # List your specific video paths here (excluding the first video since we're using a provided frame)
        video_paths = [
            "E:\\Pain\\syracus\\syracus\\aug_crop_clip_new\\IMG_0003_1_aligned_clip_001.mp4",
            "E:\\Pain\\syracus\\syracus\\aug_crop_clip_new\\IMG_0003_2_aligned_clip_001.mp4",
            "E:\\Pain\\syracus\\syracus\\aug_crop_clip_new\\IMG_0003_3_aligned_clip_001.mp4",
            "E:\\Pain\\syracus\\syracus\\aug_crop_clip_new\\IMG_0003_4_aligned_clip_001.mp4"
        ]
        
        # Custom captions for each frame
        captions = ["Original", "Augmented ver1", "Augmented ver2", "Augmented ver3", "Augmented ver4"]
        
        # Process videos with provided first frame
        output_image = "custom_frames_grid.jpg"
        extract_first_frames_and_create_grid(video_paths, output_image, first_frame_path, captions)
    
    # Uncomment one of these to run:
    # process_sample_videos()
    process_specific_videos()
    
    # Command line argument handling is still available if needed
    parser = argparse.ArgumentParser(description='Extract first frames from videos and create a grid image')
    parser.add_argument('--input_dir', help='Directory containing input videos')
    parser.add_argument('--output_image', default='first_frames_grid.jpg', help='Path for the output grid image')
    parser.add_argument('--first_frame', help='Path to the first frame image to use instead of extracting from video')
    parser.add_argument('--captions', nargs='+', help='Captions for each frame (optional)')
    
    args = parser.parse_args()
    
    if args.input_dir:
        # Get video files from directory
        video_files = []
        for f in os.listdir(args.input_dir):
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(args.input_dir, f))
        
        if not video_files:
            print(f"No video files found in {args.input_dir}")
        else:
            # Sort video files to ensure consistent order
            video_files.sort()
            extract_first_frames_and_create_grid(video_files, args.output_image, args.first_frame, args.captions) 