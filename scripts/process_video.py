import os
import subprocess
import argparse
import shutil

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video_path,
        os.path.join(output_dir, "%06d.png")
    ], check=True)

def run_segmentation(img_path, output_path, model="td2-psp50", gpu=0):
    subprocess.run([
        "python", "TDNet_VideoSegmentation/Testing/test.py",
        "--img_path", img_path,
        "--output_path", output_path,
        "--gpu", str(gpu),
        "--model", model
    ], check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cleanup", action="store_true", help="Delete frames after segmentation")

    args = parser.parse_args()
    
    print(f"Extracting frames from: {args.video}")
    extract_frames(args.video, args.frames_dir)

    print(f"Running segmentation on frames in: {args.frames_dir}")
    run_segmentation(args.frames_dir, args.output_dir)

    if args.cleanup:
        print(f"Cleaning up extracted frames in: {args.frames_dir}")
        shutil.rmtree(args.frames_dir)

if __name__ == "__main__":
    main()  

# This script extracts frames from a video and runs segmentation on each frame.
# It uses ffmpeg to extract frames and a custom segmentation model for processing.
# The script takes three arguments: the path to the video, the directory to save frames, and the output directory for segmentation results.
# The segmentation model can be specified, and the script defaults to using a GPU for processing.