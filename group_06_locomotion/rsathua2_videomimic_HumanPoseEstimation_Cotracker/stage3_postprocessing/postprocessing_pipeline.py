# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import os
import tyro
import time
from pathlib import Path
from typing import Optional
import torch
import nksr

# import root directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage3_postprocessing.gravity_calibration import apply_gravity_calibration
from stage3_postprocessing.mesh_generation import generate_mesh_from_calibrated_data


def run_postprocessing_pipeline(
    megahunter_path: str,
    out_dir: str,
    world_scale_factor: float = 1.0,
    conf_thr: float = 5.0,
    gradient_thr: float = 0.15,
    scale_bbox3d: float = 1.2,
    gender: str = 'male',
    is_megasam: bool = False,
    meshification_method: str = "nksr",
    no_spf: bool = False,
    multihuman: bool = False,
    vis: bool = False
) -> None:
    """
    Complete postprocessing pipeline: gravity calibration followed by mesh generation.
    
    Args:
        megahunter_path: Path to input megahunter file
        out_dir: Directory to save outputs
        world_scale_factor: Scale factor for world coordinates
        conf_thr: Confidence threshold for filtering points
        gradient_thr: Gradient threshold for filtering points
        scale_bbox3d: Scale factor for 3D bounding box
        gender: Gender for SMPL model
        is_megasam: Whether using MegaSam reconstruction 
        meshification_method: Method to use for meshification
        no_spf: If True, skip spatiotemporal filtering
        vis: Whether to visualize results
    """
    start_time = time.time()
    
    # Create output directory
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STAGE 3.1: GRAVITY CALIBRATION")
    print("=" * 80)
    
    # Step 1: Apply gravity calibration
    apply_gravity_calibration(
        megahunter_path=megahunter_path,
        out_dir=str(output_dir),
        world_scale_factor=world_scale_factor,
        gender=gender,
        is_megasam=is_megasam,
        multihuman=multihuman,
        vis=vis
    )
    
    print("=" * 80)
    print("STAGE 3.2: MESH GENERATION")
    print("=" * 80)
    
    # Step 2: Generate mesh from calibrated data    
    # Load NKSR network if needed
    nksr_reconstructor = None
    if meshification_method == "nksr":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nksr_reconstructor = nksr.Reconstructor(device)
    
    generate_mesh_from_calibrated_data(
        calibrated_megahunter_dir=str(output_dir),
        conf_thr=conf_thr,
        gradient_thr=gradient_thr,
        scale_bbox3d=scale_bbox3d,
        is_megasam=is_megasam,
        meshification_method=meshification_method,
        nksr_reconstructor=nksr_reconstructor,
        no_spf=no_spf,
        multihuman=multihuman,
        vis=vis
    )
    
    end_time = time.time()
    print("=" * 80)
    print(f"\033[92mComplete postprocessing pipeline finished in {end_time - start_time:.2f} seconds\033[0m")
    print(f"All outputs saved to {output_dir}")
    print("=" * 80)


def main():
    tyro.cli(run_postprocessing_pipeline)


if __name__ == "__main__":
    main()