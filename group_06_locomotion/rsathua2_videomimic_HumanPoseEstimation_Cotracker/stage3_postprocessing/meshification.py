# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import numpy as np

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial import cKDTree
from matplotlib.path import Path
import copy
import pandas as pd
import open3d as o3d
import torch
import nksr

def convex_hull_mask(mask):
    """
    Returns a mask that corresponds to the convex hull of True pixels in `mask`.
    """
    kernel = np.ones((3,3))
    eroded_mask = binary_erosion(mask, kernel)
    dilated_mask = binary_dilation(eroded_mask, kernel)
    
    # Extract (row, col) coordinates of the True pixels
    points = np.array(np.where(dilated_mask)).T
    if len(points) < 3:
        # Not enough points to form a convex hull
        return dilated_mask
    
    hull = ConvexHull(points)
    hull_path = points[hull.vertices]

    # Create a grid and check if points are inside the convex hull
    grid_x, grid_y = np.meshgrid(range(mask.shape[1]), range(mask.shape[0]))
    grid_points = np.c_[grid_y.ravel(), grid_x.ravel()]
    path = Path(hull_path)
    inside = path.contains_points(grid_points)

    return inside.reshape(mask.shape)


def infill_depth_in_convex_hull_weighted(depth_map, mask, k=4, power=2.0):
    """
    Infill missing pixels in `depth_map` (where `mask` is False) by using a weighted 
    average of the k-nearest valid (hit) neighbors. Distances are weighted according 
    to inverse distance weighting with exponent `power`.
    """
    # Compute the convex hull mask
    hull_mask = convex_hull_mask(mask)
    # Only consider the region inside the hull, where mask was True
    restricted_mask = mask & hull_mask

    # Copy the depth map to avoid modifying in-place
    filled_depth_map = depth_map.copy()

    # Indices where we have valid (hit) data
    hit_indices = np.array(np.where(restricted_mask)).T  # shape: (N_hit, 2) -> (row, col)
    # Corresponding depth values
    hit_depths = depth_map[hit_indices[:, 0], hit_indices[:, 1]]

    # Build a KD-tree for fast k-NN lookup
    kd_tree = cKDTree(hit_indices)

    # Indices where we're missing data but inside the hull
    miss_indices = np.array(np.where(~mask & hull_mask)).T  # shape: (N_miss, 2)

    # For each missing pixel, compute the k-NN among hit pixels
    # Then do inverse-distance weighted interpolation of the hit depths
    for miss_rc in miss_indices:
        dist, nn_idx = kd_tree.query(miss_rc, k=k)

        # If k=1, dist and nn_idx will be scalars, make them arrays
        if k == 1:
            dist = np.array([dist])
            nn_idx = np.array([nn_idx])

        # Inverse distance weights: w_i = 1 / (dist_i^p + epsilon)
        # epsilon is to prevent division by zero
        epsilon = 1e-8
        weights = 1.0 / (dist**power + epsilon)

        # Weighted average of the depths
        weighted_depth = np.sum(hit_depths[nn_idx] * weights) / np.sum(weights)
        filled_depth_map[miss_rc[0], miss_rc[1]] = weighted_depth

    return filled_depth_map, miss_indices

def get_point_cloud_to_fill_holes(mesh: trimesh.Trimesh, resolution: int = 1024):
    """
    Fill holes in the mesh.

    The way we do this is to ray cast from the top of the mesh and get the depth map.
    We then use the convex hull mask to get the points that are "ray misses" on the terrain.
    We then use the infill_depth_in_convex_hull_weighted function to fill in the missing depths.
    We then return the point cloud of the z coordinates of the infilled points.

    Args:
        mesh: The mesh to fill holes in.
        resolution: The resolution of the grid for ray casting.
    """

    # Define parameters for ray casting
    top_z = mesh.bounds[1][2] + 10.0  # Slightly above the top of the mesh
    x_bounds = mesh.bounds[:, 0]
    y_bounds = mesh.bounds[:, 1]

    # Create a grid of points for ray origins
    x_samples = np.linspace(x_bounds[0], x_bounds[1], resolution)
    y_samples = np.linspace(y_bounds[0], y_bounds[1], resolution)
    sample_xs, sample_ys = np.meshgrid(x_samples, y_samples)
    ray_origins = np.column_stack([sample_xs.ravel(), sample_ys.ravel(), np.full(sample_xs.size, top_z)])
    ray_dirs = np.tile([0, 0, -1], (ray_origins.shape[0], 1))  # Rays pointing downwards

    # Cast rays
    locations, index_ray, _ = mesh.ray.intersects_location(ray_origins, ray_dirs)

    # Process results
    depth_map = np.full(ray_origins.shape[0], np.nan)
    mask = np.zeros(ray_origins.shape[0], dtype=bool)

    if len(locations) > 0:
        # Compute world Z coordinates and update depth map
        z_world = locations[:, 2]
        depth_map[index_ray] = z_world
        mask[index_ray] = True

    # Reshape results to 2D for rendering
    depth_map_2d = depth_map.reshape(resolution, resolution)
    mask_2d = mask.reshape(resolution, resolution)

    # Example usage
    # Replace `depth_map_2d` and `mask_2d` with your actual depth map and mask
    infilled_depth_map, infilled_indices = infill_depth_in_convex_hull_weighted(depth_map_2d, mask_2d, k=1500, power=1.5)

    # plt.figure(figsize=(6, 6))
    # plt.imshow(infilled_depth_map, extent=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]), origin='lower', cmap='viridis')
    # plt.colorbar(label="World Z Coordinate")
    # plt.title("Infilled Depth Map (Convex Hull)")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.savefig("infilled_depth_map.png")

    # Generate the point cloud for infilled points
    x_coords = sample_xs[infilled_indices[:, 0], infilled_indices[:, 1]]
    y_coords = sample_ys[infilled_indices[:, 0], infilled_indices[:, 1]]
    z_coords = infilled_depth_map[infilled_indices[:, 0], infilled_indices[:, 1]]

    infilled_pointcloud = np.column_stack((x_coords, y_coords, z_coords))

    return infilled_pointcloud


def two_round_meshify_and_fill_holes(points: np.ndarray, downsample_voxel_size: float | None = 0.1, meshification_method: str = "nksr", nksr_reconstructor = None):
    """
    Meshify the points and fill holes in the mesh.

    NDC Algorithm:
    1. Meshify the points.
    2. Find the point cloud which fills the holes by ray casting from the top of the mesh and interpolating the depths.
    3. Merge the original points with the infilled points.
    4. Meshify the merged points.

    NKSR Algorithm:
    1. Downsample the points using a voxel grid.
    2. Estimate normals for the points.
    3. Feed downsampled points and normals to NKSR algorithm.
    4. Meshify the points.
    """
    if meshification_method == "nksr":
        if downsample_voxel_size is not None:
            print(f"Before voxel downsampling shape: {points.shape}")
            if len(points) > 10000:
                points = downsample_point_cloud(points, downsample_voxel_size, num_point_threshold=50, simple_sampling=True)
            print(f"After voxel downsampling shape: {points.shape}")

        # estimate normals, save points and normals
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        normals = estimate_point_normals(points)
        input_xyz = torch.from_numpy(points).float().to(device)
        input_normal = torch.from_numpy(normals).float().to(device)

        # meshify with NKSR
        if nksr_reconstructor is None:
            nksr_reconstructor = nksr.Reconstructor(device)
        print("First round of NKSR meshification")
        field = nksr_reconstructor.reconstruct(input_xyz, input_normal, detail_level=0.1)
        mesh = field.extract_dual_mesh(mise_iter=1)
        vertices, faces = mesh.v.cpu().numpy(), mesh.f.cpu().numpy()
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # cast rays from the sky down to fill holes and combine with original points
        infilled_pointcloud = get_point_cloud_to_fill_holes(trimesh_mesh, resolution=1024)
        infilled_normals = estimate_point_normals(infilled_pointcloud)
        combined_points = torch.from_numpy(np.concatenate([points, infilled_pointcloud], axis=0)).float().to(device)
        combined_normals = torch.from_numpy(np.concatenate([normals, infilled_normals], axis=0)).float().to(device)

        # meshify the combined points with NKSR
        print("Second round of NKSR meshification")
        field2 = nksr_reconstructor.reconstruct(combined_points, combined_normals, detail_level=0.1)
        mesh2 = field2.extract_dual_mesh(mise_iter=1)
        vertices2, faces2 = mesh2.v.cpu().numpy(), mesh2.f.cpu().numpy()
        mesh2 = trimesh.Trimesh(vertices=vertices2, faces=faces2)

        # vertices, faces = meshify(combined_points.detach().cpu().numpy())
        # mesh2 = trimesh.Trimesh(vertices=vertices, faces=faces)

        return mesh2, combined_points, infilled_pointcloud

    elif meshification_method == "ndc":
        from NDC.ndc_callable import meshify

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # meshify
        vertices, faces = meshify(copy.deepcopy(points))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        infilled_pointcloud = get_point_cloud_to_fill_holes(mesh)
        combined_points = np.concatenate([points, infilled_pointcloud], axis=0)

        vertices, faces = meshify(combined_points)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return mesh, combined_points, infilled_pointcloud

def downsample_point_cloud(points, voxel_size=0.1, num_point_threshold=100, simple_sampling=False):
    """
    Downsample points using a custom voxel grid method (vectorized with Pandas).

    For each voxel:
    - If it contains fewer than 100 points, discard all points in that voxel.
    - If it contains 100 or more points, randomly sample 20 points from it.
    """
    if points.shape[0] == 0:
        return np.empty((0, 3))

    # Calculate voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Create a Pandas DataFrame for easier grouping
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['vx'] = voxel_indices[:, 0]
    df['vy'] = voxel_indices[:, 1]
    df['vz'] = voxel_indices[:, 2]

    # Group by voxel index
    grouped = df.groupby(['vx', 'vy', 'vz'])

    # Filter groups to keep only those with >= 100 points
    filtered_groups = grouped.filter(lambda x: len(x) >= num_point_threshold)

    if filtered_groups.empty:
         return np.empty((0, 3)) # Return empty array if no voxels meet criteria

    if simple_sampling:
        # We group again on the filtered data before sampling
        sampled_df = filtered_groups.groupby(['vx', 'vy', 'vz']).sample(n=20, replace=False, random_state=3301) # Set random_state for reproducibility if needed

        # Return the sampled points as a NumPy array
        return sampled_df[['x', 'y', 'z']].to_numpy()

    else:
        # --- New sampling logic starts here ---
        def process_group(group):
            if len(group) < num_point_threshold: # Should already be filtered, but double-check
                return None

            points_in_group = group[['x', 'y', 'z']].to_numpy()
            centroid = np.mean(points_in_group, axis=0)
            
            # Calculate distances to centroid
            distances = np.linalg.norm(points_in_group - centroid, axis=1)
            
            # Find the number of points for top 20%
            num_top_20_percent = max(1, int(len(points_in_group) * 0.20)) # Ensure at least 1 point
            
            # Get indices of the closest points
            # closest_indices = np.argsort(distances)[:num_top_20_percent]

            # uniformly sample 20 points based on distance to centroid
            chosen_indices = np.argsort(distances)[::len(points_in_group)//20]
            
            # Select the closest points
            selected_points = points_in_group[chosen_indices]
            
            # Calculate mean Z of selected points
            median_z = np.mean(selected_points[:, 2])
            
            # Create a copy to modify
            modified_points = selected_points.copy()
            
            # Set Z coordinate to the median Z
            modified_points[:, 2] = median_z
            
            # Return as DataFrame to easily concatenate later
            return pd.DataFrame(modified_points, columns=['x', 'y', 'z'])

        # Apply the processing function to each group in the filtered data
        # Group again on the filtered data before applying
        processed_dfs = []
        for _, group in filtered_groups.groupby(['vx', 'vy', 'vz']):
            processed_df = process_group(group)
            if processed_df is not None:
                processed_dfs.append(processed_df)


        # Concatenate the results if list is not empty
        if not processed_dfs:
            return np.empty((0, 3))
        else:
            final_sampled_df = pd.concat(processed_dfs, ignore_index=True)
            # Return the sampled points as a NumPy array
            return final_sampled_df[['x', 'y', 'z']].to_numpy()
        # --- New sampling logic ends here ---


def estimate_point_normals(points, radius=0.1, max_nn=30, orient_consistently=True):
    """
    Estimate normals for a set of 3D points.
    
    Args:
        points (np.ndarray): Point cloud as a numpy array of shape (n, 3)
        radius (float): Search radius for normal estimation in metric units (default: 0.1 meters)
        max_nn (int): Maximum number of nearest neighbors to search (default: 30)
        orient_consistently (bool): Whether to orient normals consistently (default: True)
        
    Returns:
        np.ndarray: Estimated normals as a numpy array of shape (n, 3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals and orient them consistently
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, 
            max_nn=max_nn
        )
    )
    if orient_consistently:
        pcd.orient_normals_towards_camera_location()
    
    return np.asarray(pcd.normals)
