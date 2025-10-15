# Learning Physics from Pixels

This demo (or set of experiments) demonstrates the concept of learning physical parameters directly from pixel observations, inspired by PAC-NeRF. The core idea is to treat the derivation of physical properties as a search problem, using an iterative optimization approach to discover the actual properties of observed objects.

## Overview

The project includes three progressive demos that showcase inverse physics problems with increasing complexity. Each demo uses image-based loss and optimization to find physical parameters that match observed behavior.

## Demos

### 1. 1D Bouncing Ball (`inv_pac_nerf_1d.py`)
- Simplest case of inverse physics optimization
- Learns a single parameter: bounciness
- Uses pre-recorded trajectory data
- Demonstrated in `1d_ball_bounce_basic.mp4`
- Results visualized in `summary_plots_1d.png`

### 2. 2D Bouncing Ball (`inv_pac_nerf_2d_.py`)
- Increased complexity with two parameters:
  - Bounciness
  - Gravity
- Simulates ball motion in a 2D box
- Loss function handles both parameters simultaneously
- Converges in approximately 130 iterations
- Results visualized in `summary_plots_2d_smart.png`

### 3. 2D Deformable Mass-Spring System (`2d_spring_mass_inverse.py`)
- Simulates a deformable 'jelly' cube using mass-spring system
- Optimization pipeline:
  1. Makes initial guess for material stiffness (parameter 'k')
  2. Runs physics simulation for jelly deformation under gravity
  3. Renders final shape as 2D silhouette
  4. Calculates loss between rendered and target silhouettes
- Goal is to match green 'learned' silhouette with red 'ground truth'
- Results shown in:
  - `inverse_fem_2d_mesh_final.mp4`
  - `inverse_fem_2d_mesh_final_plot.png`
  - `pacnerf_2d_summary.png`

## Mathematical Concept

The core optimization process in each demo follows an iterative approach where physical parameters are adjusted to minimize the difference between simulated and observed behavior. This is achieved through:
1. Making parameter guesses
2. Running physics simulation
3. Comparing with ground truth
4. Updating parameters based on the loss
5. Repeating until convergence
