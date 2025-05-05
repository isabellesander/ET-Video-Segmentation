#!/bin/bash
#SBATCH --job-name=tdnet_process
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

VIDEO=$1 
BASENAME=$(basename "$VIDEO" .mp4)

python scripts/process_video.py \
    --video "$VIDEO" \
    --frames_dir "data/$BASENAME" \
    --output_dir "output/$BASENAME" \
    --cleanup

