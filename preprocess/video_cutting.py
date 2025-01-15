import os
import subprocess
import argparse

def cut_video(input_video, segment_length, overlap, output_dir="output"):
    """
        Cuts a video into segments of specified length with a specified overlap.

        This function uses `ffmpeg` to split a video into smaller segments. Each segment
        has a fixed length, and consecutive segments overlap by a specified duration.
        The segments are saved in the specified output directory.

        Args:
            input_video (str): Path to the input video file.
            segment_length (float): Length of each segment in seconds.
            overlap (float): Overlap duration between consecutive segments in seconds.
            output_dir (str, optional): Directory to save the output segments. Defaults to "output".

        Returns:
            None

        Example:
            To cut a video into 5.5-second segments with a 1-second overlap:
            ```
            cut_video("input.mp4", 5.5, 1.0, "my_output")
            ```
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the duration of the video
    cmd = f"ffmpeg -i {input_video} 2>&1 | grep 'Duration' | cut -d ' ' -f 4 | sed s/,//"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    duration_str = result.stdout.strip()
    h, m, s = map(float, duration_str.split(':'))
    total_duration = h * 3600 + m * 60 + s

    # Calculate the number of segments
    segment_length = float(segment_length)
    overlap = float(overlap)
    step = segment_length - overlap
    num_segments = int((total_duration - overlap) // step)

    # Extract base filename and extension
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    extension = os.path.splitext(input_video)[1]

    # Cut the video into segments
    for i in range(num_segments):
        start_time = i * step
        output_file = os.path.join(output_dir, f"{base_name}_segment_{i+1}{extension}")
        cmd = f"ffmpeg -ss {start_time} -i {input_video} -t {segment_length} -c copy {output_file}"
        subprocess.run(cmd, shell=True)

    print(f"Video cut into {num_segments} segments with {segment_length}s length and {overlap}s overlap.")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Cut a video into segments with specified length and overlap.")
    parser.add_argument("input_video", type=str, help="Path to the input video file.")
    parser.add_argument("segment_length", type=float, help="Length of each segment in seconds.")
    parser.add_argument("overlap", type=float, help="Overlap duration in seconds.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for the segments (default: 'output').")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    cut_video(args.input_video, args.segment_length, args.overlap, args.output_dir)
