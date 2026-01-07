import os
import glob
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_video(video_path, output_root, openface_bin, verbose=False):
    """
    Run OpenFace FeatureExtraction on a single video.
    """
    try:
        # Create a specific output directory for this video or keep flat structure
        # OpenFace creates a directory with the video name by default if not careful,
        # but here we specify -out_dir.
        
        # Derive output filename/dir
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        # We want the CSV to land in output_root/vid_name.csv roughly
        
        # OpenFace command
        # -aus: Extract Action Units
        # -pose: (Optional) Extract Pose
        # -2Dfp: (Optional) 2D landmarks
        # -q: Quiet mode
        cmd = [
            openface_bin,
            "-f", video_path,
            "-out_dir", output_root,
            "-aus",  # We specifically need AUs
            "-2Dfp", # Useful for confidence checks
            "-q"
        ]
        
        if verbose:
            print(f"Running: {' '.join(cmd)}")
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return False, f"Error processing {vid_name}: {result.stderr}"
            
        return True, vid_name
        
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Batch process videos with OpenFace")
    parser.add_argument("--video_root", type=str, required=True, help="Directory containing .mp4 files")
    parser.add_argument("--save_root", type=str, required=True, help="Directory to save OpenFace outputs")
    parser.add_argument("--openface_bin", type=str, default="/usr/local/bin/FeatureExtraction", 
                        help="Path to OpenFace FeatureExtraction executable")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel jobs")
    parser.add_argument("--recursive", action="store_true", help="Search videos recursively")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.openface_bin):
        print(f"Error: OpenFace binary not found at {args.openface_bin}")
        print("Please check the path. Common locations: /usr/local/bin/FeatureExtraction or ./build/bin/FeatureExtraction")
        return

    # Find videos
    pattern = "**/*.mp4" if args.recursive else "*.mp4"
    videos = glob.glob(os.path.join(args.video_root, pattern), recursive=args.recursive)
    
    if not videos:
        print(f"No videos found in {args.video_root}")
        return
        
    print(f"Found {len(videos)} videos. Processing with {args.num_workers} workers...")
    os.makedirs(args.save_root, exist_ok=True)
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_video, v, args.save_root, args.openface_bin): v 
            for v in videos
        }
        
        for future in tqdm(as_completed(futures), total=len(videos)):
            success, msg = future.result()
            if not success:
                print(f"\nFailure: {msg}")

    print("\nOpenFace extraction complete.")

if __name__ == "__main__":
    main()



