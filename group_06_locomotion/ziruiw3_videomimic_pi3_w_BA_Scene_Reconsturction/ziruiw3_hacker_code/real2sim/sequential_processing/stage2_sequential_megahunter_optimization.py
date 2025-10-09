# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Sequential MegaHunter Optimization

This script processes multiple videos sequentially through the MegaHunter optimization pipeline.
The key benefit is amortizing JAX compilation time across multiple videos - the model is compiled
once and then reused for all subsequent videos, significantly reducing overall processing time
when building datasets for policy training.

Note: This is NOT parallel processing. Videos are processed one after another to:
1. Maintain stable GPU memory usage
2. Share the compiled JAX model across all videos
3. Provide predictable resource consumption

Example usage:
    python batch_processing/stage2_sequential_megahunter_optimization.py \
        --pattern "jump" \
        --world-reconstruction-base-dir ./demo_data/input_megasam \
        --bbox-base-dir ./demo_data/input_masks \
        --pose2d-base-dir ./demo_data/input_2d_poses \
        --smpl-base-dir ./demo_data/input_3d_meshes \
        --out-base-dir ./demo_data/output_smpl_and_points \
        --use-g1-shape
"""

import os
import sys
import glob
from pathlib import Path
from typing import List, Optional
import tyro
from tqdm import tqdm

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from stage2_optimization.megahunter_optimization import main as megahunter_optimize


def find_matching_videos(
    base_dir: str,
    pattern: str = "",
    start_idx: int = 0,
    end_idx: int = -1
) -> List[str]:
    """Find video directories matching the given pattern."""
    if not os.path.exists(base_dir):
        return []
    
    # Get all subdirectories
    all_dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
    
    # Filter by pattern if provided
    if pattern:
        matching_dirs = [d for d in all_dirs if pattern.lower() in d.lower()]
    else:
        matching_dirs = all_dirs
    
    # Sort for consistent ordering
    matching_dirs = sorted(matching_dirs)
    
    # Apply index slicing
    if end_idx != -1:
        matching_dirs = matching_dirs[start_idx:end_idx]
    else:
        matching_dirs = matching_dirs[start_idx:]
    
    return matching_dirs


def find_world_reconstruction_file(reconstruction_dir: str, video_name: str, is_megasam: bool = True) -> Optional[str]:
    """Find the world reconstruction file for a given video."""
    if is_megasam:
        pattern = f"megasam_reconstruction_results_{video_name}_cam01_*.h5"
    else:
        pattern = f"align3r_reconstruction_results_{video_name}_cam01_*.h5"
    
    files = glob.glob(os.path.join(reconstruction_dir, pattern))
    if files:
        # Return the most recent file if multiple exist
        return max(files, key=os.path.getctime)
    return None


def main(
    pattern: str = "",
    world_reconstruction_base_dir: str = "./demo_data/input_megasam",
    bbox_base_dir: str = "./demo_data/input_masks",
    pose2d_base_dir: str = "./demo_data/input_2d_poses",
    smpl_base_dir: str = "./demo_data/input_3d_meshes",
    contact_base_dir: Optional[str] = None,
    out_base_dir: str = "./demo_data/output_smpl_and_points",
    start_idx: int = 0,
    end_idx: int = -1,
    use_g1_shape: bool = False,
    is_megasam: bool = True,
    multihuman: bool = False,
    top_k: int = 1,
    vis: bool = False,
    skip_existing: bool = True,
):
    """
    Sequential processing of multiple videos through MegaHunter optimization.
    
    This function processes videos one after another, leveraging JAX compilation caching
    to significantly reduce processing time for datasets. The first video will take longer
    due to JAX compilation, but subsequent videos will process much faster.
    
    Args:
        pattern: Pattern to filter video names (e.g., "jump" for all jumping videos)
        world_reconstruction_base_dir: Directory containing world reconstruction files
        bbox_base_dir: Directory containing bounding box data from SAM2
        pose2d_base_dir: Directory containing 2D pose data from ViTPose
        smpl_base_dir: Directory containing SMPL parameters from VIMO
        contact_base_dir: Optional directory containing contact detection from BSTRO
        out_base_dir: Output directory for optimization results
        start_idx: Starting index for video processing
        end_idx: Ending index for video processing (-1 for all)
        use_g1_shape: Use SMPL shape fitted to G1 robot
        is_megasam: Whether using MegaSam (True) or Align3r (False) reconstruction
        multihuman: Enable multi-human processing
        top_k: Number of humans to process (when multihuman=True)
        vis: Enable visualization during optimization
        skip_existing: Skip videos that have already been processed
    """
    
    # Find matching videos from one of the input directories
    video_names = find_matching_videos(bbox_base_dir, pattern, start_idx, end_idx)
    
    if not video_names:
        print(f"No videos found matching pattern '{pattern}' in {bbox_base_dir}")
        return
    
    print(f"Found {len(video_names)} videos to process:")
    for name in video_names:
        print(f"  - {name}")
    
    # Process statistics
    successful = 0
    skipped = 0
    failed = []
    
    # Process each video sequentially (one after another)
    # JAX will compile the optimization code for the first video (~3-5 minutes),
    # then all subsequent videos will run much faster without recompilation!
    for video_name in tqdm(video_names, desc="Processing videos sequentially"):
        # Find world reconstruction file
        world_env_path = find_world_reconstruction_file(
            world_reconstruction_base_dir, video_name, is_megasam
        )
        
        if not world_env_path:
            print(f"\nWarning: No world reconstruction found for {video_name}, skipping...")
            failed.append((video_name, "No world reconstruction file"))
            continue
        
        # Build paths for this video
        bbox_dir = os.path.join(bbox_base_dir, video_name, "cam01", "json_data")
        pose2d_dir = os.path.join(pose2d_base_dir, video_name, "cam01")
        smpl_dir = os.path.join(smpl_base_dir, video_name, "cam01")
        
        # Check if all required directories exist
        missing_dirs = []
        if not os.path.exists(bbox_dir):
            missing_dirs.append(f"bbox: {bbox_dir}")
        if not os.path.exists(pose2d_dir):
            missing_dirs.append(f"pose2d: {pose2d_dir}")
        if not os.path.exists(smpl_dir):
            missing_dirs.append(f"smpl: {smpl_dir}")
        
        if missing_dirs:
            print(f"\nWarning: Missing directories for {video_name}:")
            for d in missing_dirs:
                print(f"  - {d}")
            failed.append((video_name, "Missing input directories"))
            continue
        
        # Check if output already exists
        output_filename = os.path.basename(world_env_path).replace(
            "megasam_reconstruction_results_" if is_megasam else "align3r_reconstruction_results_",
            "megahunter_megasam_reconstruction_results_" if is_megasam else "megahunter_align3r_reconstruction_results_"
        )
        output_path = os.path.join(out_base_dir, output_filename)
        
        if skip_existing and os.path.exists(output_path):
            print(f"\nSkipping {video_name} - output already exists: {output_path}")
            skipped += 1
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {video_name}")
        print(f"{'='*60}")
        print(f"World reconstruction: {world_env_path}")
        print(f"Output will be: {output_path}")
        
        try:
            # Call MegaHunter optimization
            megahunter_optimize(
                world_env_path=world_env_path,
                bbox_dir=bbox_dir,
                pose2d_dir=pose2d_dir,
                smpl_dir=smpl_dir,
                out_dir=out_base_dir,
                use_g1_shape=use_g1_shape,
                multihuman=multihuman,
                top_k=top_k,
                vis=vis,
            )
            successful += 1
            print(f"✓ Successfully processed {video_name}")
            
        except Exception as e:
            print(f"✗ Error processing {video_name}: {str(e)}")
            failed.append((video_name, str(e)))
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("Sequential Processing Summary")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_names)}")
    print(f"Successful: {successful}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed videos:")
        for video_name, error in failed:
            print(f"  - {video_name}: {error}")
    
    print(f"\nNote: JAX compilation occurs only once at the start, making subsequent")
    print(f"videos process much faster. This is ideal for building training datasets.")


if __name__ == "__main__":
    tyro.cli(main)