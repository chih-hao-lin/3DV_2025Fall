"""
Create horizontal 3-way comparison video (no text labels).
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def create_three_way_comparison(video1_path, video2_path, video3_path, output_path):
    """
    Create horizontal 3-way comparison video without text labels.

    Args:
        video1_path: Path to first video (left)
        video2_path: Path to second video (center)
        video3_path: Path to third video (right)
        output_path: Path to save comparison video
    """
    print(f"Creating 3-way horizontal comparison...")
    print(f"  Left:   {video1_path}")
    print(f"  Center: {video2_path}")
    print(f"  Right:  {video3_path}")

    # Open all three videos
    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    cap3 = cv2.VideoCapture(str(video3_path))

    if not cap1.isOpened() or not cap2.isOpened() or not cap3.isOpened():
        raise ValueError("Could not open one or more videos")

    # Get video properties from all three videos
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    width3 = int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH))
    height3 = int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps3 = cap3.get(cv2.CAP_PROP_FPS)
    count3 = int(cap3.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video 1: {width1}x{height1} @ {fps1} fps, {count1} frames")
    print(f"  Video 2: {width2}x{height2} @ {fps2} fps, {count2} frames")
    print(f"  Video 3: {width3}x{height3} @ {fps3} fps, {count3} frames")

    # Use the maximum dimensions as target (or use video 2's dimensions)
    target_height = max(height1, height2, height3)
    target_width = int(target_height * width2 / height2)  # Preserve aspect ratio of center video

    print(f"  Target resolution for all panels: {target_width}x{target_height}")

    # Use optical flow frame count as target (video 2 and 3 should match)
    target_frames = count2
    print(f"  Target frame count: {target_frames} frames")

    # Load all frames from each video
    print(f"  Loading video 1...")
    frames1 = []
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        frames1.append(frame)
    cap1.release()

    print(f"  Loading video 2...")
    frames2 = []
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        frames2.append(frame)
    cap2.release()

    print(f"  Loading video 3...")
    frames3 = []
    while True:
        ret, frame = cap3.read()
        if not ret:
            break
        frames3.append(frame)
    cap3.release()

    # Resample videos to match target frame count
    def resample_frames(frames, target_count):
        """Resample frames using linear interpolation to match target count."""
        if len(frames) == target_count:
            return frames

        print(f"    Resampling from {len(frames)} to {target_count} frames...")
        source_count = len(frames)
        indices = np.linspace(0, source_count - 1, target_count)

        resampled = []
        for idx in indices:
            idx_low = int(np.floor(idx))
            idx_high = int(np.ceil(idx))

            if idx_low == idx_high:
                resampled.append(frames[idx_low])
            else:
                # Linear interpolation between frames
                alpha = idx - idx_low
                frame_low = frames[idx_low].astype(np.float32)
                frame_high = frames[idx_high].astype(np.float32)
                blended = ((1 - alpha) * frame_low + alpha * frame_high).astype(np.uint8)
                resampled.append(blended)

        return resampled

    frames1 = resample_frames(frames1, target_frames)
    frames2 = resample_frames(frames2, target_frames)
    frames3 = resample_frames(frames3, target_frames)

    # Setup output
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    padding = 10
    output_width = target_width * 3 + padding * 4
    output_height = target_height + padding * 2

    # Use target fps (from optical flow videos)
    output_fps = fps2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (output_width, output_height))

    # Write resampled frames
    print(f"  Writing comparison video...")
    for i in range(target_frames):
        frame1 = frames1[i]
        frame2 = frames2[i]
        frame3 = frames3[i]

        # Resize all frames to target resolution
        frame1 = cv2.resize(frame1, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frame2 = cv2.resize(frame2, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frame3 = cv2.resize(frame3, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Create white canvas
        canvas = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255

        # Place three frames horizontally with padding
        y = padding

        # Left video
        x = padding
        canvas[y:y+target_height, x:x+target_width] = frame1

        # Center video
        x = padding * 2 + target_width
        canvas[y:y+target_height, x:x+target_width] = frame2

        # Right video
        x = padding * 3 + target_width * 2
        canvas[y:y+target_height, x:x+target_width] = frame3

        out.write(canvas)

    out.release()

    print(f"✓ Created 3-way comparison video: {output_path}")
    print(f"  Frames: {target_frames}")
    print(f"  Output size: {output_width}x{output_height}")


def main():
    parser = argparse.ArgumentParser(
        description='Create horizontal 3-way video comparison without text labels'
    )
    parser.add_argument('video1', help='First video path (left)')
    parser.add_argument('video2', help='Second video path (center)')
    parser.add_argument('video3', help='Third video path (right)')
    parser.add_argument('output', help='Output video path')

    args = parser.parse_args()

    create_three_way_comparison(
        args.video1,
        args.video2,
        args.video3,
        args.output
    )


if __name__ == '__main__':
    main()
