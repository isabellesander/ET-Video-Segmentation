#!/bin/bash

VIDEOS_DIR="videos"
FRAMES_DIR="data"
OUTPUT_DIR="output"

for VIDEO_PATH in $VIDEOS_DIR/*.mp4; do
    BASENAME=$(basename "$VIDEO_PATH" .mp4)

    python scripts/process_video.py \
        --video "$VIDEO_PATH" \
        --frames_dir "$FRAMES_DIR/$BASENAME" \
        --output_dir "$OUTPUT_DIR/$BASENAME" \
        --cleanup
done
