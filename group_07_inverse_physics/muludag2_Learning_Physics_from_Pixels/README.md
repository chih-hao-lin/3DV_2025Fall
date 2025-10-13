# Learning Physics from Pixels

This project explores how physical parameters (like gravity, elasticity, and material stiffness) can be learned directly from visual observations using differentiable physics and rendering.

## Demos

1. **1D Bouncing Ball** (`inv_pac_nerf_1d.py`)  
   - Learns the bounce coefficient from height data  
   - Uses PyTorch for automatic differentiation  

2. **2D Bouncing Ball** (`inverse_fem_2d.py`, `inverse_fem_2d_smart.py`)  
   - Learns gravity and bounciness from 2D motion  
   - Uses Taichi for fast simulation  
   - Smart version improves optimization and convergence  

3. **2D Deformable Object** (`inverse_mpm_silhouette_2d.py`)  
   - MPM-based soft-body dynamics  
   - Learns material parameters from silhouette loss  
   - Combines differentiable physics and rendering  

## Setup

```bash
conda create -n physics python=3.8
conda activate physics
pip install torch numpy taichi matplotlib opencv-python
# Optional for silhouette demo
pip install pytorch3d
