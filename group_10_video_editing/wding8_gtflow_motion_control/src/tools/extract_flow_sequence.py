"""
extract_flow_sequence.py

Extracts complete optical flow sequences from videos for frequency analysis.
Uses RAFT model via CommonSource.noise_warp integration.

Usage:
    conda activate flow_warp
    python extract_flow_sequence.py <video_path> --output <output_path>
"""

import rp
import numpy as np
import torch
from pathlib import Path
import fire

rp.r._pip_import_autoyes = True
rp.git_import('CommonSource')
import rp.git.CommonSource.noise_warp as nw


def extract_full_flow_sequence(video_path: str,
                              output_path: str = None,
                              fps: int = 30,
                              save_raw: bool = True) -> np.ndarray:
    """
    Extract and save complete optical flow sequence for frequency analysis.
    Returns both forward and backward flows.

    Args:
        video_path: Path to input video file or URL
        output_path: Output path for flow file (default: video_name_flows.npy)
        fps: Frame rate of the video (for frequency analysis)
        save_raw: Whether to save raw flows to disk

    Returns:
        Flow tensor with shape (T-1, 2, H, W)
    """
    print("="*60)
    print("EXTRACTING OPTICAL FLOW SEQUENCE")
    print("="*60)

    # Load and preprocess video
    print(f"\nStep 1: Loading video from {video_path}...")
    video = rp.load_video(video_path)

    # Resize to CogVideoX dimensions
    video = rp.resize_list(video, length=49)
    video = rp.resize_images_to_hold(video, height=480, width=720)
    video = rp.crop_images(video, height=480, width=720, origin='center')
    video = rp.as_numpy_array(video)

    print(f"  Video shape: {video.shape}")

    # Extract flows using existing noise_warp infrastructure
    print("\nStep 2: Extracting optical flow...")
    print("  (This uses RAFT model via CommonSource.noise_warp)")

    # Use the noise warp function to get flows
    output = nw.get_noise_from_video(
        video,
        remove_background=False,
        visualize=False,
        save_files=False,
        noise_channels=16,
        output_folder=None,
        resize_frames=1,  # Keep original resolution for flow
        resize_flow=1,
        downscale_factor=1,
    )

    # Extract flow sequence
    flows = output.numpy_flows  # Shape should be (T-1, 2, H, W)

    print(f"  Extracted flow shape: {flows.shape}")
    print(f"  Flow range: [{flows.min():.2f}, {flows.max():.2f}]")
    print(f"  Mean magnitude: {np.sqrt(flows[:,0]**2 + flows[:,1]**2).mean():.2f}")

    # Determine output path
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"{video_name}_flows.npy"

    # Save flows
    if save_raw:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, flows)
        print(f"\n✓ Saved flow sequence to: {output_path}")

        # Also save metadata
        metadata = {
            'video_path': str(video_path),
            'fps': fps,
            'shape': flows.shape,
            'video_shape': video.shape,
        }
        metadata_path = output_path.with_suffix('.json')
        rp.save_json(metadata, metadata_path)
        print(f"✓ Saved metadata to: {metadata_path}")

    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)

    return flows


def main(video: str, output: str = None, fps: int = 30):
    """
    Main entry point for flow extraction.

    Args:
        video: Path to video file or URL
        output: Output path for flow file (default: video_name_flows.npy)
        fps: Frame rate of the video
    """
    flows = extract_full_flow_sequence(video, output, fps)
    return flows


if __name__ == "__main__":
    fire.Fire(main)
