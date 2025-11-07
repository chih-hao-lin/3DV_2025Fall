"""
Create side-by-side comparison of two optical flow visualizations.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def create_side_by_side_comparison(video1_path, video2_path, output_path, label1="Original", label2="Scaled"):
    """
    Create side-by-side comparison video.

    Args:
        video1_path: Path to first video
        video2_path: Path to second video
        output_path: Path to save comparison video
        label1: Label for first video
        label2: Label for second video
    """
    print(f"Creating side-by-side comparison...")
    print(f"  Left:  {video1_path} ({label1})")
    print(f"  Right: {video2_path} ({label2})")

    # Open both videos
    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))

    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Could not open one or both videos")

    # Get video properties
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup output
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    padding = 10
    output_width = width * 2 + padding * 3
    output_height = height + padding * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))

    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Create canvas
        canvas = np.ones((output_height, output_width, 3), dtype=np.uint8) * 240

        # Place frames
        y, x = padding, padding
        canvas[y:y+height, x:x+width] = frame1

        x = padding * 2 + width
        canvas[y:y+height, x:x+width] = frame2

        # Add labels
        cv2.putText(canvas, label1, (padding + 10, padding + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(canvas, label2, (padding * 2 + width + 10, padding + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        out.write(canvas)
        frame_count += 1

    cap1.release()
    cap2.release()
    out.release()

    print(f"✓ Created comparison video: {output_path}")
    print(f"  Frames: {frame_count}")


def main():
    parser = argparse.ArgumentParser(description='Create side-by-side flow comparison')
    parser.add_argument('video1', help='First video path')
    parser.add_argument('video2', help='Second video path')
    parser.add_argument('output', help='Output video path')
    parser.add_argument('--label1', default='Original', help='Label for first video')
    parser.add_argument('--label2', default='Scaled', help='Label for second video')

    args = parser.parse_args()

    create_side_by_side_comparison(
        args.video1,
        args.video2,
        args.output,
        args.label1,
        args.label2
    )


if __name__ == '__main__':
    main()
