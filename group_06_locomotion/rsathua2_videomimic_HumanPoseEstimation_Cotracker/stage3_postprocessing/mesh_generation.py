# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import os
import pickle
import numpy as np
import trimesh
import torch
import cv2
import tyro
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import h5py


# -------------------------------------------------------------------------- #
# HDF5 helpers
# -------------------------------------------------------------------------- #
def load_dict_from_hdf5(h5file: h5py.File, path: str = "/") -> Dict:
    """
    Recursively load a nested dictionary from an HDF5 file.
    """
    result = {}
    for key in h5file[path].keys():
        key_path = f"{path}{key}"
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path + "/")
        else:
            result[key] = h5file[key_path][:]
    # attributes
    for attr_key, attr_value in h5file.attrs.items():
        if attr_key.startswith(path):
            result[attr_key[len(path) :] ] = attr_value
    return result


def get_mesh(bg_points: np.ndarray, bg_colors: np.ndarray, meshification_method: str = "nksr", nksr_reconstructor=None) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """
    Generate mesh from background points using specified meshification method.
    
    Args:
        bg_points: Background points (N, 3)
        bg_colors: Background colors (N, 3)
        meshification_method: Method to use for meshification
        nksr_reconstructor: NKSR reconstructor network
        
    Returns:
        Tuple of (trimesh_mesh, combined_points, infilled_pointcloud)
    """
    from stage3_postprocessing.meshification import two_round_meshify_and_fill_holes
    
    # Start timing for background meshifying
    start_meshifying = time.time()
    
    trimesh_mesh, combined_points, infilled_pointcloud = two_round_meshify_and_fill_holes(
        bg_points, 
        meshification_method=meshification_method, 
        nksr_reconstructor=nksr_reconstructor
    )
    
    # End timing for background meshifying
    end_meshifying = time.time()
    print(f"\033[92mBackground meshifying process time: {end_meshifying - start_meshifying:.2f} seconds\033[0m")
    
    return trimesh_mesh, combined_points, infilled_pointcloud


def filter_points_and_colors(
    calibrated_megahunter_path: str,
    conf_thr: float = 5.0,
    gradient_thr: float = 0.15,
    scale_bbox3d: float = 3.0,
    is_megasam: bool = False,
    no_spf: bool = False,
    multihuman: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter points and colors from calibrated megamegamegamegahunter data.
    
    Args:
        calibrated_megahunter_dir: Path to gravity calibrated megamegamegamegahunter file
        conf_thr: Confidence threshold for filtering points
        gradient_thr: Gradient threshold for filtering points
        scale_bbox3d: Scale factor for 3D bounding box
        is_megasam: Whether using MegaSam reconstruction
        no_spf: If True, skip spatiotemporal filtering
        
    Returns:
        Tuple of (bg_points_filtered, bg_colors_filtered, no_spf_bg_points_filtered, no_spf_bg_colors_filtered)
    """
    if is_megasam:
        conf_thr = 0.00
    
    # Load calibrated data
    if 'h5' in calibrated_megahunter_path:
        print(f"Loading calibrated data from {calibrated_megahunter_path}")
        with h5py.File(calibrated_megahunter_path, 'r') as f:
            world_env_and_human = load_dict_from_hdf5(f)
    else:
        print(f"Loading calibrated data from {calibrated_megahunter_path}")
        with open(calibrated_megahunter_path, 'rb') as f:
            world_env_and_human = pickle.load(f)
    
    world_env = world_env_and_human['our_pred_world_cameras_and_structure']
    human_params_in_world = world_env_and_human['our_pred_humans_smplx_params']
    
    # Support multi-human processing
    if multihuman:
        person_id_list = sorted(list(human_params_in_world.keys()))
        print(f"Processing {len(person_id_list)} humans for mesh generation: {person_id_list}")
    else:
        person_id_list = [list(human_params_in_world.keys())[0]]
        print(f"Processing single person for mesh generation: {person_id_list[0]}")
    
    print("Collecting and filtering points and colors...")
    start_collecting = time.time()
    
    # Initialize lists for point collection
    bg_pt3ds = []
    bg_colors = []
    bg_confs = []
    bg_pc_downsample_factor = 1
    
    no_spf_bg_pt3ds = []
    no_spf_bg_colors = []
    no_spf_bg_pc_downsample_factor = 1
    
    # Morphological kernels
    small_kernel_size = 10
    large_kernel_size = 20
    small_kernel = np.ones((small_kernel_size, small_kernel_size), np.uint8)
    large_kernel = np.ones((large_kernel_size, large_kernel_size), np.uint8)
    
    # Get unique frame names from all persons
    # Each person might appear in different frames
    frame_names_set = set()
    person_frame_mapping = {}  # Maps frame_name -> list of person_ids present
    
    for person_id in person_id_list:
        person_frames = world_env_and_human['person_frame_info_list'][person_id].astype(str)
        person_frames = person_frames.flatten().tolist()
        person_frame_mapping[person_id] = set(person_frames)
        frame_names_set.update(person_frames)
    
    # Convert to sorted list for consistent ordering
    frame_names = sorted(list(frame_names_set))
    num_frames = len(frame_names)
    
    # Create reverse mapping: frame -> persons present in that frame
    frame_to_persons = {}
    for frame_name in frame_names:
        frame_to_persons[frame_name] = []
        for person_id in person_id_list:
            if frame_name in person_frame_mapping[person_id]:
                frame_to_persons[frame_name].append(person_id)
    
    print(f"Total unique frames: {num_frames}")
    print(f"Frame coverage per person: {[(pid, len(frames)) for pid, frames in person_frame_mapping.items()]}")
    
    # Load keypoints for bbox computation
    keypoints_path = Path(calibrated_megahunter_path).parent / 'gravity_calibrated_keypoints.h5'
    with h5py.File(keypoints_path, 'r') as f:
        keypoints_data = load_dict_from_hdf5(f)
    
    # Load joints for all persons
    human_joints_in_world_dict = {}
    for person_id in person_id_list:
        if person_id in keypoints_data['joints']:
            human_joints_in_world_dict[person_id] = keypoints_data['joints'][person_id]  # (T, 45, 3)
        else:
            print(f"Warning: No joints found for person {person_id}")
            human_joints_in_world_dict[person_id] = None
    
    for fn_idx, frame_name in enumerate(frame_names):
        # frame_name is already a string from our sorted list
        pt3d = world_env[frame_name]['pts3d']  # (H, W, 3) or (N, 3)
        if pt3d.ndim == 3:
            pt3d = pt3d.reshape(-1, 3)  # (N, 3)
        
        conf = world_env[frame_name]['conf'].reshape(-1)  # (N,)
        colors = world_env[frame_name]['rgbimg'].reshape(-1, 3)  # (N, 3)
        dynamic_msk = world_env[frame_name]['dynamic_msk'].astype(np.uint8)  # (H, W)
        
        # Get confidence mask
        conf_mask = conf >= conf_thr  # (N,)
        
        # Add gradient mask
        depths = world_env[frame_name]['depths'].copy()  # (H, W)
        dy, dx = np.gradient(depths)  # (H, W), (H, W)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)  # (H, W)
        gradient_mask = gradient_magnitude < gradient_thr  # (H, W)
        gradient_mask = gradient_mask.flatten()  # (N,)
        
        """ Spatial + Temporal filtering """
        # Get 3D bounding box from SMPL joints of persons present in this frame
        all_person_joints = []
        persons_in_frame = frame_to_persons.get(frame_name, [])
        
        for person_id in persons_in_frame:
            if human_joints_in_world_dict[person_id] is not None:
                # Find the frame index for this person
                person_frames = world_env_and_human['person_frame_info_list'][person_id].astype(str)
                person_frame_idx = np.where(person_frames == frame_name)[0]
                if len(person_frame_idx) > 0:
                    person_frame_idx = person_frame_idx[0]
                    curr_smpl_joints = human_joints_in_world_dict[person_id][person_frame_idx]  # (45, 3)
                    all_person_joints.append(curr_smpl_joints)
        
        if len(all_person_joints) > 0:
            # Combine joints from all persons in this frame
            all_joints = np.concatenate(all_person_joints, axis=0)  # (45*num_persons_in_frame, 3)
            min_joints = np.min(all_joints, axis=0)  # (3,)
            max_joints = np.max(all_joints, axis=0)  # (3,)
        else:
            # Fallback to reasonable defaults if no joints found
            min_joints = np.array([-2.0, -2.0, -2.0])
            max_joints = np.array([2.0, 2.0, 2.0])
            
        center_joints = (min_joints + max_joints) / 2  # (3,)
        size_joints = abs(max_joints - min_joints)  # (3,)
        # Clip size with reasonable bounds
        size_joints = np.clip(size_joints * scale_bbox3d, 4.0, 8.0)  # (3,)
        bbox3d_min = (center_joints - size_joints / 2)  # (3,)
        bbox3d_max = (center_joints + size_joints / 2)  # (3,)
        
        # Sample points within 3D bounding box from neighboring frames
        prev_idx = max(0, fn_idx - 5) # neighboring up to 5 frames
        next_idx = min(num_frames - 1, fn_idx + 5) # neighboring up to 5 frames
        
        sampled_pts3d = []
        sampled_colors = []
        sampled_confs = []
        
        for i in range(prev_idx, next_idx + 1):
            neighbor_fname = frame_names[i]  # Already a string
            neighbor_frame_pt3d = world_env[neighbor_fname]['pts3d']  # (H, W, 3) or (N, 3)
            if neighbor_frame_pt3d.ndim == 3:
                neighbor_frame_pt3d = neighbor_frame_pt3d.reshape(-1, 3)  # (N, 3)
                
            neighbor_frame_colors = world_env[neighbor_fname]['rgbimg'].reshape(-1, 3)  # (N, 3)
            neighbor_frame_depths = world_env[neighbor_fname]['depths']  # (H, W)
            neighbor_frame_msk = world_env[neighbor_fname]['dynamic_msk'].astype(np.uint8)  # (H, W)
            neighbor_frame_conf = world_env[neighbor_fname]['conf'].reshape(-1)  # (N,)
            
            # Get confidence mask
            neighbor_frame_conf_mask = neighbor_frame_conf > conf_thr  # (N,)
            
            # Apply gradient filtering 
            neighbor_frame_dy, neighbor_frame_dx = np.gradient(neighbor_frame_depths)  # (H, W), (H, W)
            neighbor_frame_gradient_magnitude = np.sqrt(neighbor_frame_dy**2 + neighbor_frame_dx**2)  # (H, W)
            neighbor_frame_gradient_mask = neighbor_frame_gradient_magnitude < gradient_thr  # (H, W)
            neighbor_frame_gradient_mask = neighbor_frame_gradient_mask.flatten()  # (N,)
            
            large_dilated_msk = cv2.dilate(neighbor_frame_msk, large_kernel, iterations=1).flatten() > 0  # (N,)
            small_dilated_msk = cv2.dilate(neighbor_frame_msk, small_kernel, iterations=1).flatten() > 0  # (N,)
            
            neighbor_frame_non_dynamic_mask = large_dilated_msk & ~small_dilated_msk & neighbor_frame_conf_mask  # (N,)
            neighbor_frame_non_dynamic_mask = neighbor_frame_non_dynamic_mask & neighbor_frame_gradient_mask  # (N,)
            
            # Get mask of points within 3D bounding box
            bbox3d_mask = (neighbor_frame_pt3d >= bbox3d_min) & (neighbor_frame_pt3d <= bbox3d_max)  # (N, 3)
            bbox3d_mask = bbox3d_mask.all(axis=1)  # (N,)
            
            # Sample points within 3D bounding box
            sampled_pts3d.append(neighbor_frame_pt3d[bbox3d_mask & neighbor_frame_non_dynamic_mask])
            sampled_colors.append(neighbor_frame_colors[bbox3d_mask & neighbor_frame_non_dynamic_mask])
            sampled_confs.append(neighbor_frame_conf[bbox3d_mask & neighbor_frame_non_dynamic_mask])
        
        try:
            bg_pt3ds.append(np.concatenate(sampled_pts3d, axis=0)[::bg_pc_downsample_factor, :])
            bg_colors.append(np.concatenate(sampled_colors, axis=0)[::bg_pc_downsample_factor, :])
            bg_confs.append(np.concatenate(sampled_confs, axis=0)[::bg_pc_downsample_factor])
        except:
            # Add empty array if concatenation fails
            bg_pt3ds.append(np.zeros((0, 3)))
            bg_colors.append(np.zeros((0, 3)))
            bg_confs.append(np.zeros((0)))
            
        # Process current frame for no-spf version
        # dynamic_msk is already loaded above, just need to process it
        dynamic_msk_flat = dynamic_msk.flatten() > 0  # (N,)
        dynamic_msk_flat = cv2.dilate(dynamic_msk_flat.astype(np.uint8).reshape(dynamic_msk.shape), large_kernel, iterations=1).flatten() > 0  # (N,)
        bg_mask = ~dynamic_msk_flat & gradient_mask & conf_mask  # (N,)
        
        no_spf_bg_pt3ds.append(pt3d[bg_mask][::no_spf_bg_pc_downsample_factor, :])
        no_spf_bg_colors.append(colors[bg_mask][::no_spf_bg_pc_downsample_factor, :])
    
    bg_points_filtered = np.concatenate(bg_pt3ds, axis=0)  # (M, 3)
    bg_colors_filtered = np.concatenate(bg_colors, axis=0)  # (M, 3)
    no_spf_bg_points_filtered = np.concatenate(no_spf_bg_pt3ds, axis=0)  # (L, 3)
    no_spf_bg_colors_filtered = np.concatenate(no_spf_bg_colors, axis=0)  # (L, 3)
    
    end_collecting = time.time()
    print(f"\033[92mCollecting and filtering points and colors process time: {end_collecting - start_collecting:.2f} seconds\033[0m")
    
    return bg_points_filtered, bg_colors_filtered, no_spf_bg_points_filtered, no_spf_bg_colors_filtered


def generate_mesh_from_calibrated_data(
    calibrated_megahunter_dir: str,
    conf_thr: float = 5.0,
    gradient_thr: float = 0.15,
    scale_bbox3d: float = 3.0,
    is_megasam: bool = False,
    meshification_method: str = "nksr",
    nksr_reconstructor=None,
    no_spf: bool = False,
    multihuman: bool = False,
    vis: bool = False,
) -> None:
    """
    Generate mesh from gravity calibrated megamegamegamegahunter data.
    
    Args:
        calibrated_megahunter_path: Path to gravity calibrated megahunter files
        out_dir: Directory to save outputs
        conf_thr: Confidence threshold for filtering points
        gradient_thr: Gradient threshold for filtering points; higher less filtering for the human pointcloud; recommend to be 0.05 when the human pointcloud is noisy, otherwise 0.15
        scale_bbox3d: Scale factor for 3D bounding box that is used for spatiotemporal filtering. For detailed environments like stairs, set it to smaller values like 1.2. If you are reading this, you might want to look at the clipping values in filter_points_and_colors.
        is_megasam: Whether using MegaSam reconstruction
        meshification_method: Method to use for meshification
        nksr_reconstructor: NKSR reconstructor network
        no_spf: (Experimental)If True, skip spatiotemporal filtering and the process will be x2-3 slower but you will get spatially large mesh. I (Hongsuk) recommend to set it to False for crispy detailed geometry, otherwise True.
        vis: Whether to visualize results
    """
    # Use the same directory for the output
    output_dir = Path(calibrated_megahunter_dir)

    calibrated_megahunter_path = os.path.join(calibrated_megahunter_dir, "gravity_calibrated_megahunter.h5")
    
    # Filter points and colors
    bg_points_filtered, bg_colors_filtered, no_spf_bg_points_filtered, no_spf_bg_colors_filtered = filter_points_and_colors(
        calibrated_megahunter_path=calibrated_megahunter_path,
        conf_thr=conf_thr,
        gradient_thr=gradient_thr,
        scale_bbox3d=scale_bbox3d,
        is_megasam=is_megasam,
        no_spf=no_spf,
        multihuman=multihuman
    )
    
    # Generate mesh
    if not no_spf:
        mesh, _, _ = get_mesh(
            bg_points_filtered, 
            bg_colors_filtered, 
            meshification_method=meshification_method, 
            nksr_reconstructor=nksr_reconstructor
        )
    else:
        mesh, _, _ = get_mesh(
            no_spf_bg_points_filtered, 
            no_spf_bg_colors_filtered, 
            meshification_method=meshification_method, 
            nksr_reconstructor=nksr_reconstructor
        )
    
    # Get the largest mesh component
    mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.faces))
    
    # Save outputs
    print("Saving mesh generation outputs...")
    
    # Save mesh
    mesh_path = output_dir / 'background_mesh.obj'
    mesh.export(mesh_path)
    print(f"Saved mesh to {mesh_path}")
    
    # Save point clouds for visualization
    less_filtered_point_cloud = trimesh.points.PointCloud(
        vertices=no_spf_bg_points_filtered, 
        colors=no_spf_bg_colors_filtered
    )
    more_filtered_point_cloud = trimesh.points.PointCloud(
        vertices=bg_points_filtered, 
        colors=bg_colors_filtered
    )
    
    # Export point clouds to PLY
    less_filtered_point_cloud.export(output_dir / 'background_less_filtered_colored_pointcloud.ply')
    print(f"Saved colored pointcloud to {output_dir / 'background_less_filtered_colored_pointcloud.ply'}")
    
    more_filtered_point_cloud.export(output_dir / 'background_more_filtered_colored_pointcloud.ply')
    print(f"Saved colored pointcloud to {output_dir / 'background_more_filtered_colored_pointcloud.ply'}")
    
    if vis:
        # Add the root directory to the path
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from visualization.mesh_generation_visualization import visualize_mesh_generation
        visualize_mesh_generation(
            mesh, 
            bg_points_filtered if not no_spf else no_spf_bg_points_filtered,
            bg_colors_filtered if not no_spf else no_spf_bg_colors_filtered,
            no_spf
        )


def main():
    tyro.cli(generate_mesh_from_calibrated_data)


if __name__ == "__main__":
    main()