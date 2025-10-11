# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Stage 3 Postprocessing Module

This module handles the postprocessing pipeline for  VideoMimic Real-to-Sim system:
1. Gravity calibration - aligns world coordinate system with gravity
2. Mesh generation - creates background meshes from calibrated point clouds

The pipeline processes megahunter optimization results and outputs:
- Gravity calibrated keypoints and world rotation
- Background mesh (OBJ format)  
- Filtered point clouds (PLY format)
- Updated megahunter files with calibrated coordinates
"""

from .gravity_calibration import apply_gravity_calibration
from .mesh_generation import generate_mesh_from_calibrated_data
from .postprocessing_pipeline import run_postprocessing_pipeline

__all__ = [
    'apply_gravity_calibration',
    'generate_mesh_from_calibrated_data', 
    'run_postprocessing_pipeline'
]