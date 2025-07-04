import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video file")
        return
    
    frame_count = 0
    
    while True:
        # Read a frame
        success, frame = video.read()
        
        # Break if no more frames
        if not success:
            break
        
        # Save frame as image
        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Extracted {frame_count} frames...")
    
    # Release video capture
    video.release()
    print(f"\nExtraction complete! {frame_count} frames saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from an AVI video file')
    parser.add_argument('video_path', help='Path to the input AVI video file')
    parser.add_argument('--output_dir', default='extracted_frames', help='Directory to save extracted frames (default: extracted_frames)')
    
    args = parser.parse_args()
    
    extract_frames(args.video_path, args.output_dir) 