# Learning Physics from Pixels

This project demonstrates learning physical parameters (gravity, bounciness, material properties) directly from pixel observations using differentiable physics and rendering.

## Demos

1. **1D Bouncing Ball** (`inv_pac_nerf_1d.py`)
   - Learns bounciness coefficient from height observations
   - Uses PyTorch autodiff for gradient computation

2. **2D Bouncing Ball** (`inverse_fem_2d.py`, `inverse_fem_2d_smart.py`)
   - Learns gravity and bounciness from 2D trajectory observations
   - Uses Taichi for fast physics simulation
   - Smart optimization strategies for better convergence

3. **2D Deformable Object** (`inverse_mpm_silhouette_2d.py`)
   - MPM-based simulation of soft body dynamics
   - Learns material parameters from silhouette observations
   - Combines differentiable physics and rendering

4. **2D/3D PyTorch Physics** (`inv_pac_nerf_2d_.py`, `inv_pac_nerf_3d.py`)
   - PyTorch implementation for automatic differentiation
   - Parameter optimization through simulation

## Setup

1. Create a conda environment:
```bash
conda create -n physics python=3.8
conda activate physics
```

2. Install dependencies:
```bash
pip install torch numpy taichi matplotlib opencv-python
```

3. For silhouette demo (optional):
```bash
pip install pytorch3d
```

## Running the Demos

Each script can be run independently:

```bash
# 1D bouncing ball with PyTorch
python inv_pac_nerf_1d.py

# 2D bouncing ball with Taichi
python inverse_fem_2d.py

# Smart optimization version
python inverse_fem_2d_smart.py

# MPM silhouette optimization
python inverse_mpm_silhouette_2d.py

# PyTorch 2D/3D physics
python inv_pac_nerf_2d_.py
python inv_pac_nerf_3d.py
```

## Outputs

Each script generates:
- Real-time visualization
- MP4 video of the optimization process
- Summary plots showing convergence and results

## Code Structure

- `inv_pac_nerf_1d.py`: 1D physics with PyTorch autodiff
- `inverse_fem_2d.py`: Basic 2D physics simulation
- `inverse_fem_2d_smart.py`: Improved optimization strategy
- `inverse_mpm_silhouette_2d.py`: MPM simulation with rendering
- `inv_pac_nerf_2d_.py`: PyTorch 2D implementation
- `inv_pac_nerf_3d.py`: 3D extension of PyTorch physics

## Implementation Details

- Uses Taichi for high-performance physics simulation
- PyTorch for automatic differentiation
- Differentiable rendering for silhouette loss
- Momentum-based optimization for better convergence
- Smart phase-based parameter estimation

## Results

Videos and plots will be saved in:
- `frames/`: Individual animation frames
- `*.mp4`: Final video demonstrations
- `*.png`: Summary plots and visualizations
