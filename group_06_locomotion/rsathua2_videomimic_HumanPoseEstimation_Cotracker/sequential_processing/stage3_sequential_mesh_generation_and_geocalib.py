# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Sequential Postprocessing (GeoCalib + Meshification) for Multiple Videos

This script processes multiple videos sequentially through gravity calibration and mesh generation.
The key benefit is that models (especially NKSR if used) are loaded once and reused for all videos,
reducing overall processing time when building datasets.

Sequential processing (NOT parallel/batch):
- Models are loaded once at startup
- Videos are processed one by one to maintain stable memory usage
- Runs in vm1recon environment for GeoCalib and NKSR compatibility

Example usage:
    python sequential_processing/stage3_sequential_mesh_generation_and_geocalib.py \
        --pattern "jump" \
        --hunter-base-dir ./demo_data/output_smpl_and_points \
        --output-base-dir ./demo_data/output_calib_mesh \
        --is-megasam
"""

import os
import sys
# Add parent directory to Python path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import time  # Import time module

def main(
    megahunter_base_dir: str,
    output_base_dir: str,
    pattern: str,
    world_scale_factor: float = 1.0,
    conf_thr: float = 0.0,
    gradient_thr: float = 0.15,
    scale_bbox3d: float = 1.2,
    is_megasam: bool = True,
    meshification_method: str = "nksr",
    no_spf: bool = False,
    multihuman: bool = False,
    gender: str = 'male',
    start_idx: int = 0,
    end_idx: int = -1,
    vis: bool = False,
):
    """
    Sequential processing of multiple videos through postprocessing pipeline.
    
    This function processes videos sequentially to amortize model loading time across
    multiple videos. GeoCalib and meshification models are loaded once and reused,
    making this ideal for building training datasets efficiently.
    
    Args:
        megahunter_base_dir: Directory containing input H5/PKL files from MegaHunter
        output_base_dir: Base directory for outputs
        pattern: Pattern to match video names
        world_scale_factor: Scale factor for world coordinates
        conf_thr: Confidence threshold for filtering points
        gradient_thr: Gradient threshold for filtering points; higher more filtering for the human pointcloud; recommend to be 0.01 when the human pointcloud is noisy, otherwise 0.15 - Hongsuk
        scale_bbox3d: Scale factor for 3D bounding box
        meshification_method: Method to use for meshification; ndc or nksr
        no_spf: If True, skip the spatiotemporal filtering step
        multihuman: Whether to process multiple humans
        gender: Gender for SMPL model
        start_idx: Starting index for video processing
        end_idx: Ending index for video processing (-1 for all)
        vis: Whether to visualize during processing
    """
    # Convert to Path objects
    megahunter_base_dir = Path(megahunter_base_dir)
    output_base_dir = Path(output_base_dir)
    
    # Ensure output base directory exists
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all H5/PKL files in input directory
    h5_files = sorted(list(megahunter_base_dir.glob(f"*{pattern}*.h5")))
    h5_files += sorted(list(megahunter_base_dir.glob(f"*{pattern}*.pkl")))
    if not h5_files:
        print(f"[INFO] No H5 files found in {megahunter_base_dir}")
        return
    print(f"[INFO] Found {len(h5_files)} H5/pkl files to process")
    if end_idx == -1:
        end_idx = len(h5_files)
    h5_files = h5_files[start_idx:end_idx]

    # Start timing for the entire processing
    start_time = time.time()

    # Process each video sequentially (one after another)
    # This maintains stable GPU memory usage while reusing loaded models
    for h5_path in tqdm(h5_files, desc="Processing H5 files sequentially"):
        # Create output directory with same name as PKL (without extension)
        output_dir = output_base_dir / h5_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Import here to avoid module-level import issues
            from stage3_postprocessing.postprocessing_pipeline import run_postprocessing_pipeline
            
            # Start timing for processing each file, set up NKSR network
            start_file_time = time.time()
            
            # Check if the output background mesh already exists
            mesh_path = output_dir / 'background_mesh.obj'
            if mesh_path.exists():
                print(f"[INFO] Skipping {h5_path.name} because background mesh already exists")
                continue
            
            # Process the file using new pipeline
            run_postprocessing_pipeline(
                megahunter_path=str(h5_path),
                out_dir=str(output_dir),
                world_scale_factor=world_scale_factor,
                conf_thr=conf_thr,
                gradient_thr=gradient_thr,
                scale_bbox3d=scale_bbox3d,
                gender=gender,
                is_megasam=is_megasam,
                meshification_method=meshification_method,
                no_spf=no_spf,
                multihuman=multihuman,
                vis=vis,
            )
            
            # End timing for processing each file
            end_file_time = time.time()
            print(f"\033[92mSuccessfully processed {h5_path.name} in {end_file_time - start_file_time:.2f} seconds\033[0m")
            print(f"Outputs saved to {output_dir}")
        
        except KeyboardInterrupt:
            print(f"\nKeyboardInterrupt: Exiting gracefully")
            continue
            
        except Exception as e:
            print(f"\nError processing {h5_path.name}: {str(e)}")
            continue

    # End timing for the entire processing
    end_time = time.time()
    print(f"\033[92mTotal processing time: {end_time - start_time:.2f} seconds for {len(h5_files)} videos\033[0m")


if __name__ == "__main__":
    tyro.cli(main)