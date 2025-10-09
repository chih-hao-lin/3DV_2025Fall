# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Sequential Robot Motion Retargeting

This script processes multiple videos sequentially through the robot motion retargeting pipeline.
The key benefit is amortizing JAX compilation time across multiple videos - the retargeting model 
is compiled once and then reused for all subsequent videos, significantly reducing overall 
processing time when building datasets for policy training.

Sequential processing (NOT parallel/batch):
- JAX models are compiled once at startup
- Videos are processed one by one to maintain stable memory usage
- Provides predictable resource consumption

Example usage:
    python sequential_processing/stage4_sequential_robot_motion_retargeting.py \
        --pattern "jump" \
        --postprocessed-base-dir ./demo_data/output_calib_mesh \
        --contact-base-dir ./demo_data/input_contacts
"""

import os
import sys
# Add parent directory to Python path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import tyro
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import time
import functools

import jax
import numpy as onp
import yourdfpy
import pyroki as pk

from stage4_retargeting.robot_motion_retargeting import process_retargeting


def find_matching_directories(base_dir: str, pattern: str) -> List[Path]:
    """Find directories matching the pattern in the base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    # Find all subdirectories containing the pattern
    matching_dirs = []
    for dir_path in base_path.iterdir():
        if dir_path.is_dir() and pattern in dir_path.name:
            matching_dirs.append(dir_path)
    
    return sorted(matching_dirs)


def extract_video_name_from_dir(dir_path: Path) -> str:
    """Extract video name from directory path."""
    # Directory names typically look like:
    # megahunter_megasam_reconstruction_results_video_name_cam01_frame_0_100_subsample_1
    dir_name = dir_path.name
    if "megahunter_megasam_reconstruction_results_" in dir_name:
        video_name = dir_name.split("megahunter_megasam_reconstruction_results_")[1].split("_cam")[0]
    elif "megahunter_align3r_reconstruction_results_" in dir_name:
        video_name = dir_name.split("megahunter_align3r_reconstruction_results_")[1].split("_cam")[0]
    else:
        # Fallback: just use the directory name
        video_name = dir_name.split("_cam")[0] if "_cam" in dir_name else dir_name
    
    return video_name


def extract_subsample_factor(dir_path: Path) -> int:
    """Extract subsample factor from directory name."""
    dir_name = dir_path.name
    if "_subsample_" in dir_name:
        try:
            return int(dir_name.split("_subsample_")[1].split("_")[0])
        except:
            return 1
    return 1


def main(
    postprocessed_base_dir: str,
    pattern: str = "",
    contact_base_dir: Optional[str] = "./demo_data/input_contacts",
    offset_factor: float = 0.0,
    start_idx: int = 0,
    end_idx: int = -1,
    vis: bool = False,
    multihuman: bool = False,
    top_k: int = 1,
    skip_existing: bool = True,
    # Retargeting weights
    local_pose_cost_weight: float = 8.0,
    end_effector_cost_weight: float = 5.0,
    global_pose_cost_weight: float = 2.0,
    self_coll_factor_weight: float = 1.0,
    world_coll_factor_weight: float = 0.1,
    world_coll_margin: float = 0.01,
    limit_cost_factor_weight: float = 1000.0,
    smoothness_cost_factor_weight: float = 10.0,
    foot_skating_cost_weight: float = 10.0,
    ground_contact_cost_weight: float = 1.0,
    hip_yaw_cost_weight: float = 5.0,
    hip_pitch_cost_weight: float = 0.0,
    hip_roll_cost_weight: float = 0.0,
):
    """
    Sequential processing of multiple videos through robot motion retargeting.
    
    This function processes videos one after another, leveraging JAX compilation caching
    to significantly reduce processing time for datasets. The first video will take longer
    due to JAX compilation, but subsequent videos will process much faster.
    
    Args:
        postprocessed_base_dir: Directory containing postprocessed results from stage 3
        pattern: Pattern to filter video names (e.g., "jump" for all jumping videos)
        contact_base_dir: Optional directory containing contact detection from BSTRO
        offset_factor: Offset factor for retargeting
        start_idx: Starting index for video processing
        end_idx: Ending index for video processing (-1 for all)
        vis: Enable visualization during retargeting
        multihuman: Enable multi-human processing
        top_k: Number of humans to process (when multihuman=True)
        skip_existing: Skip videos that have already been processed
        [retargeting weights]: Various weights for the retargeting objective
    """
    
    # Start timing for the entire processing
    start_time = time.time()
    
    # Find matching directories
    matching_dirs = find_matching_directories(postprocessed_base_dir, pattern)
    
    if not matching_dirs:
        print(f"No directories found matching pattern '{pattern}' in {postprocessed_base_dir}")
        return
    
    # Apply index slicing
    if end_idx == -1:
        end_idx = len(matching_dirs)
    matching_dirs = matching_dirs[start_idx:end_idx]
    
    print(f"Found {len(matching_dirs)} directories to process:")
    for dir_path in matching_dirs:
        print(f"  - {dir_path.name}")
    
    # Load robot model once for all videos
    print("\nLoading robot model (this happens only once for all videos)...")
    urdf_path = os.path.join(os.path.dirname(__file__), "../assets/robot_asset/g1/g1_29dof_anneal_23dof_foot.urdf")
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    
    # Create partial function with all the fixed parameters
    process_retargeting_partial = functools.partial(
        process_retargeting,
        urdf=urdf,
        robot=robot,
        robot_coll=robot_coll,
        offset_factor=offset_factor,
        local_pose_cost_weight=local_pose_cost_weight,
        end_effector_cost_weight=end_effector_cost_weight,
        global_pose_cost_weight=global_pose_cost_weight,
        self_coll_factor_weight=self_coll_factor_weight,
        world_coll_factor_weight=world_coll_factor_weight,
        world_coll_margin=world_coll_margin,
        limit_cost_factor_weight=limit_cost_factor_weight,
        smoothness_cost_factor_weight=smoothness_cost_factor_weight,
        foot_skating_cost_weight=foot_skating_cost_weight,
        ground_contact_cost_weight=ground_contact_cost_weight,
        hip_yaw_cost_weight=hip_yaw_cost_weight,
        hip_pitch_cost_weight=hip_pitch_cost_weight,
        hip_roll_cost_weight=hip_roll_cost_weight,
        multihuman=multihuman,
        top_k=top_k,
        vis=vis,
    )
    
    # Process statistics
    successful = 0
    skipped = 0
    failed = []
    
    # Process each directory sequentially
    print("\nStarting sequential retargeting...")
    print("Note: JAX will compile the retargeting code for the first video (~2-3 minutes),")
    print("then all subsequent videos will run much faster without recompilation!\n")
    
    for dir_path in tqdm(matching_dirs, desc="Processing videos sequentially"):
        # Check if output already exists
        output_file = dir_path / "retarget_poses_g1.h5"
        if skip_existing and output_file.exists():
            print(f"\nSkipping {dir_path.name} - output already exists")
            skipped += 1
            continue
        
        # Extract video name and subsample factor
        video_name = extract_video_name_from_dir(dir_path)
        subsample_factor = extract_subsample_factor(dir_path)
        
        # Construct contact directory path if provided
        contact_dir = None
        if contact_base_dir:
            contact_dir = Path(contact_base_dir) / video_name / "cam01"
            if not contact_dir.exists():
                print(f"\nWarning: Contact directory not found for {video_name}: {contact_dir}")
                # Continue anyway - retargeting can work without contacts
        
        print(f"\n{'='*60}")
        print(f"Processing {dir_path.name}")
        print(f"{'='*60}")
        print(f"Video name: {video_name}")
        print(f"Subsample factor: {subsample_factor}")
        if contact_dir and contact_dir.exists():
            print(f"Contact dir: {contact_dir}")
        
        try:
            # Call robot motion retargeting
            process_retargeting_partial(
                src_dir=dir_path,
                contact_dir=contact_dir,
                subsample_factor=subsample_factor,
            )
            successful += 1
            print(f"✓ Successfully processed {dir_path.name}")
            
        except KeyboardInterrupt:
            print(f"\nKeyboardInterrupt: Stopping sequential processing")
            break
            
        except Exception as e:
            print(f"✗ Error processing {dir_path.name}: {str(e)}")
            failed.append((dir_path.name, str(e)))
            continue
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("Sequential Retargeting Summary")
    print(f"{'='*60}")
    print(f"Total directories: {len(matching_dirs)}")
    print(f"Successful: {successful}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed: {len(failed)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    
    if successful > 0:
        avg_time = total_time / successful
        print(f"Average time per video: {avg_time:.2f} seconds")
    
    if failed:
        print("\nFailed videos:")
        for dir_name, error in failed:
            print(f"  - {dir_name}: {error}")
    
    print(f"\nNote: JAX compilation occurs only once at the start, making subsequent videos process much faster. This is ideal for building training datasets.")


if __name__ == "__main__":
    tyro.cli(main)