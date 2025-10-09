# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Complete Results Ego-View Visualization

This script provides a comprehensive visualization of the complete pipeline results including:
- Background environment mesh and point clouds
- SMPL human mesh and joints  
- Robot motion (retargeted)
- Ego-view rendering from human head pose
- Camera frustums with RGB images
- Interactive filtering controls

Data structure expected in postprocessed_dir:
├── gravity_calibrated_keypoints.h5      # Human keypoints and world rotation
├── gravity_calibrated_megahunter*.h5    # Updated hunter file with calibrated coordinates  
├── background_mesh.obj                  # Environment mesh
├── background_*_filtered_colored_pointcloud.ply  # Point clouds
├── retarget_poses_g1.h5                # Robot motion data
"""

from __future__ import annotations

import time
import os.path as osp
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import h5py
import numpy as onp
import torch
import trimesh
import tyro
import viser
import yourdfpy
import smplx
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

# import root directory
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import viser_camera_util
from utilities.joint_names import SMPL_KEYPOINTS  
from utilities.egoview_rendering import render_pointcloud_to_rgb_depth


# ============================================================================
# Color Utilities
# ============================================================================

# Load colors from colors.txt file
COLORS_PATH = osp.join(osp.dirname(__file__), 'colors.txt')
if osp.exists(COLORS_PATH):
    with open(COLORS_PATH, 'r') as f:
        COLORS = onp.array([list(map(int, line.strip().split())) for line in f], dtype=onp.uint8)
else:
    # Default color palette if colors.txt doesn't exist
    COLORS = onp.array([
        [155, 0, 0], [0, 255, 0], [0, 0, 155], [255, 255, 0],
        [255, 0, 255], [0, 255, 255], [128, 0, 128], [255, 165, 0]
    ], dtype=onp.uint8)

def get_color(person_id: int) -> Tuple[int, int, int]:
    """Get color for a person ID."""
    color = COLORS[person_id % len(COLORS)]
    return tuple(color.tolist())


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_dict_from_hdf5(h5file: h5py.File, path: str = "/") -> Dict[str, Any]:
    """
    Recursively load a nested dictionary from an HDF5 file.
    
    Args:
        h5file: An open h5py.File object
        path: The current path in the HDF5 file
    
    Returns:
        A nested dictionary with the data
    """
    result = {}
    for key in h5file[path].keys():
        key_path = f"{path}{key}"
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path + "/")
        else:
            data = h5file[key_path][:]
            # Convert to float32 if it's a floating point array
            if data.dtype in [onp.float64, onp.float32] and data.dtype != onp.float32:
                data = data.astype(onp.float32)
            result[key] = data
    
    # Load attributes (scalars stored as attributes)
    for attr_key, attr_value in h5file.attrs.items():
        if attr_key.startswith(path):
            # Convert to float32 if it's a floating point scalar
            if isinstance(attr_value, (onp.float64, float)) and not isinstance(attr_value, onp.float32):
                attr_value = onp.float32(attr_value)
            result[attr_key[len(path):]] = attr_value

    return result



def load_megahunter_data(megahunter_path: Path, person_id: str, device: str) -> Tuple[
    Dict[str, Any],  # world_env
    onp.ndarray,     # smpl_joints3d: (T, 45, 3)
    onp.ndarray,     # smpl_verts: (T, 6890, 3) 
    onp.ndarray,     # smpl_head_transform: (T, 4, 4)
    List[onp.ndarray], # fg_pt3ds: List[T] of (N_fg, 3)
    List[onp.ndarray], # fg_colors: List[T] of (N_fg, 3)
    List[onp.ndarray], # fg_confs: List[T] of (N_fg,)
    List[onp.ndarray], # cam2worlds: List[T] of (4, 4)
    int,             # num_frames
    Optional[onp.ndarray], # left_foot_contact: (T,)
    Optional[onp.ndarray], # right_foot_contact: (T,)
    onp.ndarray,     # smpl_faces: (F, 3)
]:
    """Load and process megahunter data including SMPL mesh and world environment."""
    with h5py.File(megahunter_path, 'r') as f:
        megahunter_data = load_dict_from_hdf5(f)

    world_env = megahunter_data['our_pred_world_cameras_and_structure']
    human_params_in_world = megahunter_data['our_pred_humans_smplx_params']
    
    # Extract SMPL data
    smpl_batch_layer = smplx.create(
        model_path='./assets/body_models', 
        model_type='smpl', 
        gender='male',  # TODO: make configurable
        num_betas=10, 
        batch_size=len(human_params_in_world[person_id]['body_pose'])
    ).to(device)

    num_frames = human_params_in_world[person_id]['body_pose'].shape[0]
    smpl_betas = torch.from_numpy(human_params_in_world[person_id]['betas'].astype(onp.float32)).to(device)
    if smpl_betas.ndim == 1:
        smpl_betas = smpl_betas.repeat(num_frames, 1)

    smpl_output_batch = smpl_batch_layer(
        body_pose=torch.from_numpy(human_params_in_world[person_id]['body_pose'].astype(onp.float32)).to(device), 
        betas=smpl_betas, 
        global_orient=torch.from_numpy(human_params_in_world[person_id]['global_orient'].astype(onp.float32)).to(device), 
        pose2rot=False
    )
    
    smpl_joints = smpl_output_batch['joints']  # (T, 45, 3)
    smpl_root_joint = smpl_joints[:, 0:1, :]  # (T, 1, 3)
    if human_params_in_world[person_id]['root_transl'].ndim == 2:
        smpl_root_transl = human_params_in_world[person_id]['root_transl'][:, None, :]
    else:
        smpl_root_transl = human_params_in_world[person_id]['root_transl']
    smpl_verts = smpl_output_batch['vertices'] - smpl_root_joint + torch.from_numpy(smpl_root_transl).to(device)  # (T, 6890, 3)

    # Get head pose for ego-view rendering
    smpl_joints_transforms = smpl_output_batch['joints_transforms']  # (T, 24, 4, 4)
    smpl_head_transform = smpl_joints_transforms[:, SMPL_KEYPOINTS.index('head'), :, :]  # (T, 4, 4)
    smpl_head_transform[:, :3, 3] = smpl_head_transform[:, :3, 3] - smpl_root_joint[:, 0, :] + torch.from_numpy(smpl_root_transl[:, 0, :]).to(device)

    # Convert to numpy with float32
    smpl_joints3d = (smpl_joints.detach().cpu().numpy() - smpl_root_joint.detach().cpu().numpy() + smpl_root_transl).astype(onp.float32)  # (T, 45, 3)
    smpl_verts = smpl_verts.detach().cpu().numpy().astype(onp.float32)  # (T, 6890, 3)
    smpl_head_transform = smpl_head_transform.detach().cpu().numpy().astype(onp.float32)  # (T, 4, 4)
    
    # Process point clouds and cameras
    fg_pt3ds, fg_colors, fg_confs, cam2worlds = [], [], [], []
    frame_names = megahunter_data['person_frame_info_list'][person_id]
    frame_names = frame_names.astype(str)
    
    for frame_name in frame_names:
        frame_name = frame_name.item()
        pt3d = world_env[frame_name]['pts3d'].reshape(-1, 3).astype(onp.float32)  # (H*W, 3)
        conf = world_env[frame_name]['conf'].reshape(-1).astype(onp.float32)  # (H*W,)
        colors = world_env[frame_name]['rgbimg'].reshape(-1, 3)  # (H*W, 3)
        # Convert colors to uint8 range (0-255) if they're in float range (0-1)
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(onp.uint8)
        else:
            colors = colors.astype(onp.uint8)
        cam2world = world_env[frame_name]['cam2world'].astype(onp.float32)  # (4, 4)
        dynamic_msk = world_env[frame_name]['dynamic_msk'].astype(onp.uint8)  # (H, W)
        
        # Apply gradient filtering first (on full point cloud)
        gradient_mask = onp.ones(pt3d.shape[0], dtype=bool)
        if 'depths' in world_env[frame_name]:
            depths = world_env[frame_name]['depths']
            dy, dx = onp.gradient(depths)
            gradient_magnitude = onp.sqrt(dx**2 + dy**2)
            gradient_mask = gradient_magnitude.flatten() < 0.05  # Default gradient threshold
        
        # Dilate dynamic mask to include human regions
        kernel = onp.ones((20, 20), onp.uint8)
        dynamic_msk = cv2.dilate(dynamic_msk, kernel, iterations=1).flatten() > 0  # (H*W,)
        
        # Combine masks: foreground = dynamic & gradient filter
        fg_mask = dynamic_msk & gradient_mask
        
        fg_pt3ds.append(pt3d[fg_mask].astype(onp.float32))
        fg_colors.append(colors[fg_mask])
        fg_confs.append(conf[fg_mask].astype(onp.float32))
        cam2worlds.append(cam2world.astype(onp.float32))

    # Load contact data (if available)
    left_foot_contact, right_foot_contact = None, None
    
    # Load contact estimation if available
    video_name = megahunter_path.parent.name
    # split results and cam01
    contact_dir = megahunter_path.parent.parent.parent / 'input_contacts' / video_name.split('_results_')[1].split('_cam')[0] / 'cam01'
    
    contact_estimation = None
    if contact_dir.exists():
        contact_estimation = {}
        frame_names = megahunter_data['person_frame_info_list'][person_id].astype(str)
        for frame_name in frame_names:
            frame_name = frame_name.item()
            contact_file = contact_dir / f'{frame_name}.pkl'
            if contact_file.exists():
                import pickle
                with open(contact_file, 'rb') as f:
                    contact_estimation[frame_name] = pickle.load(f)
        
        if contact_estimation:
            # Load SMPL vertex segmentation for contact visualization
            import json
            smpl_vert_seg_path = "./assets/body_models/smpl/smpl_vert_segmentation.json"
            if osp.exists(smpl_vert_seg_path):
                with open(smpl_vert_seg_path, 'r') as f:
                    smpl_vert_seg = json.load(f)
                
                left_foot_vert_ids = onp.array(smpl_vert_seg['leftFoot'], dtype=onp.int32)
                right_foot_vert_ids = onp.array(smpl_vert_seg['rightFoot'], dtype=onp.int32)
                
                # Add vertex IDs to contact information
                for frame_name in contact_estimation.keys():
                    if int(person_id) in contact_estimation[frame_name]:
                        contact_estimation[frame_name][int(person_id)]['left_foot_vert_ids'] = left_foot_vert_ids
                        contact_estimation[frame_name][int(person_id)]['right_foot_vert_ids'] = right_foot_vert_ids
    
    return (world_env, smpl_joints3d, smpl_verts, smpl_head_transform, 
            fg_pt3ds, fg_colors, fg_confs, cam2worlds, num_frames,
            left_foot_contact, right_foot_contact, smpl_batch_layer.faces,
            contact_estimation)


def apply_point_filtering(
    pt3ds: onp.ndarray,     # (N, 3)
    colors: onp.ndarray,    # (N, 3)
    gradient_threshold: float = 0.05
) -> Tuple[onp.ndarray, onp.ndarray]:
    """Apply minimal filtering to point clouds (gradient filtering moved to preprocessing)."""
    return pt3ds, colors


# ============================================================================
# Ego-View Rendering Functions  
# ============================================================================

def render_ego_views(
    smpl_head_transform: onp.ndarray,  # (T, 4, 4)
    bg_pt3ds: onp.ndarray,            # (N, 3)
    bg_colors: onp.ndarray,           # (N, 3)
    num_frames: int,
    img_res: int = 256,
    fov: float = 60.0
) -> Tuple[List[onp.ndarray], List[onp.ndarray]]:
    """Render RGB and depth images from human head pose."""
    ego_view_rgbimg_list = []
    ego_view_depthimg_list = []
    
    # Create camera intrinsics with float32
    f = onp.float32(img_res / (2 * onp.tan(onp.radians(fov) / 2)))
    head_cam_K = onp.array([
        [f, 0, img_res/2],
        [0, f, img_res/2], 
        [0, 0, 1]
    ], dtype=onp.float32)
    
    for t in range(num_frames):
        # Create head camera transform with offset in float32
        transform = onp.eye(4, dtype=onp.float32)
        # 30 degrees around x-axis, 15 degrees around y-axis
        x_rot = R.from_euler('x', 30, degrees=True).as_matrix().astype(onp.float32)
        y_rot = R.from_euler('y', 15, degrees=True).as_matrix().astype(onp.float32)
        rot_180 = onp.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=onp.float32)
        transform[:3, :3] = (x_rot @ y_rot @ rot_180).astype(onp.float32)
        
        smpl_head_cam2world = (smpl_head_transform[t] @ onp.linalg.inv(transform)).astype(onp.float32)
        world2smpl_head_cam = onp.linalg.inv(smpl_head_cam2world).astype(onp.float32)
        
        # Render from head camera
        rgbimg, depthimg = render_pointcloud_to_rgb_depth(
            points=bg_pt3ds,
            colors=bg_colors,
            camera_intrinsics=head_cam_K,
            camera_extrinsics=world2smpl_head_cam,
            image_size=(img_res, img_res),
        )
        
        # Rotate images 180 degrees (pinhole camera)
        rgbimg = cv2.rotate(rgbimg, cv2.ROTATE_180)
        depthimg = cv2.rotate(depthimg, cv2.ROTATE_180)
        
        # Create depth visualization
        depth_vis = onp.clip(depthimg, 0.1, 4.0)
        depth_vis = (depth_vis - 0.1) / (4.0 - 0.1)
        depth_vis = (depth_vis * 255).astype(onp.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        
        ego_view_rgbimg_list.append(rgbimg.astype(onp.uint8))
        ego_view_depthimg_list.append(depth_vis)
    
    return ego_view_rgbimg_list, ego_view_depthimg_list


# ============================================================================
# GUI Utility Functions
# ============================================================================

def create_contact_mesh(
    vertices: onp.ndarray, 
    faces: onp.ndarray,
    contact_info: Dict[str, Any], 
    person_id: int
) -> trimesh.Trimesh:
    """Create mesh with contact information visualization."""
    vertices_color = onp.full_like(vertices, get_color(person_id))
    
    # Highlight contact areas in green
    if contact_info.get('left_foot_contact', False) and 'left_foot_vert_ids' in contact_info:
        left_foot_ids = contact_info['left_foot_vert_ids']
        vertices_color[left_foot_ids] = onp.array([0, 255, 0])
        
    if contact_info.get('right_foot_contact', False) and 'right_foot_vert_ids' in contact_info:
        right_foot_ids = contact_info['right_foot_vert_ids'] 
        vertices_color[right_foot_ids] = onp.array([0, 255, 0])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = vertices_color.astype(onp.uint8)
    return mesh


def add_distance_measurement(server: viser.ViserServer) -> None:
    """Add interactive distance measurement tool."""
    control0 = server.scene.add_transform_controls("/controls/0", position=(1, 0, 0), scale=0.5, visible=False)
    control1 = server.scene.add_transform_controls("/controls/1", position=(-1, 0, 0), scale=0.5, visible=False)
    segments = server.scene.add_line_segments("/controls/line", onp.array([control0.position, control1.position], dtype=onp.float32)[None, :, :], colors=(255, 0, 0), visible=False)

    with server.gui.add_folder("Distance Tool"):
        show_distance_tool = server.gui.add_checkbox("Show Distance Tool", False)
        distance_text = server.gui.add_text("Distance", initial_value="Distance: 0")

    @show_distance_tool.on_update
    def _(_) -> None:
        control0.visible = show_distance_tool.value
        control1.visible = show_distance_tool.value
        segments.visible = show_distance_tool.value

    def update_distance():
        distance = onp.linalg.norm(control0.position - control1.position)
        distance_text.value = f"Distance: {distance:.2f}"
        segments.points = onp.array([control0.position, control1.position], dtype=onp.float32)[None, :, :]

    control0.on_update(lambda _: update_distance())
    control1.on_update(lambda _: update_distance())


# ============================================================================
# Main Visualization Function
# ============================================================================

def main(
    postprocessed_dir: Path,
    robot_name: str = "g1",
    bg_pc_downsample_factor: int = 4,
    is_megasam: bool = True,
    save_ego_view: bool = False,
    no_spf: bool = True,
) -> None:
    """
    Main visualization function for complete pipeline results.
    
    Args:
        postprocessed_dir: Directory containing processed results (gravity_calibrated_*, background_mesh.obj, etc.)
        robot_name: Robot name for loading retargeted poses
        bg_pc_downsample_factor: Downsample factor for background point cloud
        is_megasam: Whether results are from MegaSam or Align3r
        save_ego_view: Whether to save ego-view images to disk
        no_spf: Whether to use less filtered (True) or more filtered (False) point cloud
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load gravity-calibrated keypoints
    keypoints_path = postprocessed_dir / 'gravity_calibrated_keypoints.h5'
    with h5py.File(keypoints_path, 'r') as f:
        keypoints_data = load_dict_from_hdf5(f)
    
    # Get all person IDs
    person_ids = list(keypoints_data['joints'].keys())
    print(f"Found {len(person_ids)} person(s) in the scene: {person_ids}")
    
    # Get num_frames from first person (assuming all have same number of frames)
    first_person_id = person_ids[0]
    num_frames = keypoints_data['joints'][first_person_id].shape[0]
    
    # Load megahunter data for detailed visualization
    megahunter_path = postprocessed_dir / 'gravity_calibrated_megahunter.h5'
    
    # Load additional data if megahunter file exists
    world_env = None
    all_smpl_joints3d = {}  # {person_id: joints}
    all_smpl_verts = {}     # {person_id: vertices}
    all_smpl_head_transform = {}  # {person_id: head_transform}
    fg_pt3ds, fg_colors, fg_confs, cam2worlds = [], [], [], []
    smpl_faces = None
    all_contact_estimation = {}  # {person_id: contact_estimation}
    
    if megahunter_path.exists():
        # Load data for each person
        for person_id in person_ids:
            (world_env_tmp, smpl_joints3d, smpl_verts, smpl_head_transform, 
             fg_pt3ds_tmp, fg_colors_tmp, fg_confs_tmp, cam2worlds_tmp, _, _, _, smpl_faces_tmp,
             contact_estimation) = load_megahunter_data(megahunter_path, person_id, device)
            
            all_smpl_joints3d[person_id] = smpl_joints3d
            all_smpl_verts[person_id] = smpl_verts
            all_smpl_head_transform[person_id] = smpl_head_transform
            if contact_estimation is not None:
                all_contact_estimation[person_id] = contact_estimation
            
            # Use world_env and point clouds from first person only (they should be the same)
            if world_env is None:
                world_env = world_env_tmp
                fg_pt3ds, fg_colors, fg_confs, cam2worlds = fg_pt3ds_tmp, fg_colors_tmp, fg_confs_tmp, cam2worlds_tmp
                smpl_faces = smpl_faces_tmp
                
    # Load background data
    background_mesh = trimesh.load(postprocessed_dir / 'background_mesh.obj')
    
    # Extract background point cloud from world_env if available
    bg_pt3ds, bg_colors = None, None
    if world_env is not None:
        # Collect background points from all frames
        all_bg_pt3ds = []
        all_bg_colors = []
        all_bg_confs = []
        
        for frame_name, cam2world in zip(world_env.keys(), cam2worlds):
            pt3d = world_env[frame_name]['pts3d'].reshape(-1, 3).astype(onp.float32)  # (H*W, 3)
            colors = world_env[frame_name]['rgbimg'].reshape(-1, 3)  # (H*W, 3)
            # Convert colors to uint8 range (0-255) if they're in float range (0-1)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(onp.uint8)
            else:
                colors = colors.astype(onp.uint8)
            conf = world_env[frame_name]['conf'].reshape(-1).astype(onp.float32)  # (H*W,)
            dynamic_msk = world_env[frame_name]['dynamic_msk'].astype(onp.uint8)  # (H, W)
            
            # Dilate dynamic mask to exclude human regions
            kernel = onp.ones((20, 20), onp.uint8)
            dynamic_msk = cv2.dilate(dynamic_msk, kernel, iterations=1).flatten() > 0  # (H*W,)
            
            # Apply gradient filtering if depth data is available
            gradient_mask = onp.ones(pt3d.shape[0], dtype=bool)
            if 'depths' in world_env[frame_name]:
                depths = world_env[frame_name]['depths']
                dy, dx = onp.gradient(depths)
                gradient_magnitude = onp.sqrt(dx**2 + dy**2)
                gradient_mask = gradient_magnitude.flatten() < 0.05  # Use default gradient threshold
            
            # Extract background points (non-dynamic and pass gradient filter)
            bg_mask = ~dynamic_msk & gradient_mask
            all_bg_pt3ds.append(pt3d[bg_mask])
            all_bg_colors.append(colors[bg_mask])
            all_bg_confs.append(conf[bg_mask])
        
        # Concatenate all background points
        bg_pt3ds = onp.concatenate(all_bg_pt3ds, axis=0)[::bg_pc_downsample_factor, :].astype(onp.float32)  # (N, 3)
        bg_colors = onp.concatenate(all_bg_colors, axis=0)[::bg_pc_downsample_factor, :].astype(onp.uint8)  # (N, 3) 
        bg_confs = onp.concatenate(all_bg_confs, axis=0)[::bg_pc_downsample_factor].astype(onp.float32)  # (N,)
    else:
        # Fallback: load from PLY files if world_env is not available
        if no_spf:
            pointcloud_path = postprocessed_dir / 'background_less_filtered_colored_pointcloud.ply'
        else:
            pointcloud_path = postprocessed_dir / 'background_more_filtered_colored_pointcloud.ply'
        
        pointcloud = trimesh.load(pointcloud_path)
        bg_pt3ds = onp.array(pointcloud.vertices, dtype=onp.float32)[::bg_pc_downsample_factor, :]  # (N, 3)
        bg_colors = onp.array(pointcloud.colors, dtype=onp.float32)[::bg_pc_downsample_factor, :3]  # (N, 3)
        bg_confs = onp.ones(bg_pt3ds.shape[0], dtype=onp.float32)  # Default confidence of 1.0
    
    # Load retargeted poses - check for both single and multi-person formats
    retargeted_poses_path = postprocessed_dir / f'retarget_poses_{robot_name}.h5'
    retargeted_poses_multiperson_path = postprocessed_dir / f'retarget_poses_{robot_name}_multiperson.h5'
    
    retargeted_poses = None
    is_multiperson = False
    person_ids = []
    
    # Try loading multi-person file first
    if retargeted_poses_multiperson_path.exists():
        try:
            with h5py.File(retargeted_poses_multiperson_path, 'r') as f:
                retargeted_poses = load_dict_from_hdf5(f)
                is_multiperson = True
                # Get person IDs from the multi-person data
                person_ids = list(retargeted_poses["persons"].keys())
                print(f"Loaded multi-person retargeted poses from {retargeted_poses_multiperson_path}")
                print(f"Multi-person mode: {len(person_ids)} persons detected")
        except Exception as e:
            print(f"Failed to load multi-person file: {e}")
    
    # Fall back to single person file
    if retargeted_poses is None and retargeted_poses_path.exists():
        try:
            with h5py.File(retargeted_poses_path, 'r') as f:
                retargeted_poses = load_dict_from_hdf5(f)
                is_multiperson = False
                print(f"Loaded single-person retargeted poses from {retargeted_poses_path}")
        except Exception as e:
            print(f"Failed to load single-person file: {e}")
    
    # Update num_frames based on retargeting data if needed
    if retargeted_poses is not None:
        if is_multiperson:
            # For multi-person, get max frames across all persons
            retarget_num_frames = max(retargeted_poses["persons"][pid]["joints"].shape[0] for pid in person_ids)
            if retarget_num_frames != num_frames:
                print(f"Warning: Retargeting has {retarget_num_frames} frames, but keypoints have {num_frames} frames. Using min.")
                num_frames = min(num_frames, retarget_num_frames)
        else:
            # Single person format
            retarget_num_frames = retargeted_poses["joints"].shape[0]
            if retarget_num_frames != num_frames:
                print(f"Warning: Retargeting has {retarget_num_frames} frames, but keypoints have {num_frames} frames. Using min.")
                num_frames = min(num_frames, retarget_num_frames)
    
    # Generate ego-view images (use first person's head)
    ego_view_rgbimg_list, ego_view_depthimg_list = [], []
    if all_smpl_head_transform and first_person_id in all_smpl_head_transform:
        print(f"Generating ego-view renderings from person {first_person_id}'s perspective...")
        start_time = time.time()
        ego_view_rgbimg_list, ego_view_depthimg_list = render_ego_views(
            all_smpl_head_transform[first_person_id], bg_pt3ds, bg_colors, num_frames
        )
        print(f"Ego-view rendering completed in {time.time() - start_time:.2f} seconds")
        
        if save_ego_view:
            save_dir = postprocessed_dir / "ego_view"
            save_dir.mkdir(exist_ok=True)
            for t, (rgb, depth) in enumerate(zip(ego_view_rgbimg_list, ego_view_depthimg_list)):
                cv2.imwrite(str(save_dir / f"rgbimg_{t}.png"), rgb[:, :, ::-1])
                cv2.imwrite(str(save_dir / f"depthimg_color_{t}.png"), depth[:, :, ::-1])
            print(f"Saved ego-view images to {save_dir}")
    
    # ========================================================================
    # Start Viser Visualization
    # ========================================================================
    
    server = viser.ViserServer(port=8081)
    
    # Initialize camera follow functions
    stop_camera_follow = None
    resume_camera_follow = None
    
    # ========================================================================
    # GUI Controls
    # ========================================================================
    
    # Playback controls
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames-1, step=1, initial_value=0, disabled=True)
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=30)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("10", "20", "30", "60"))
        
        # Ego-view controls (inside playback folder, right below playback controls)
        if len(ego_view_rgbimg_list) > 0:
            gui_ego_rgb_image = server.gui.add_image(ego_view_rgbimg_list[0], label="Ego RGB View")
            gui_ego_depth_image = server.gui.add_image(ego_view_depthimg_list[0], label="Ego Depth View")
        else:
            gui_ego_rgb_image = None
            gui_ego_depth_image = None

    # Point filtering controls
    with server.gui.add_folder("Point Filtering"):
        gui_gradient_threshold = server.gui.add_slider("Gradient Threshold", min=0.001, max=0.5, step=0.002, initial_value=0.05)
        gui_point_size = server.gui.add_slider("Point Size", min=0.001, max=0.02, step=0.001, initial_value=0.01)
    
    # Camera controls
    with server.gui.add_folder("Camera Controls"):
        gui_frustum_scale = server.gui.add_slider("Frustum Scale", min=0.1, max=0.5, step=0.01, initial_value=0.1)
        if smpl_verts is not None:
            gui_play_camera_to_follow = server.gui.add_checkbox("Play Camera to Follow", initial_value=False)
    
    # Scene element visibility
    with server.gui.add_folder("Scene Elements"):
        gui_show_bg_mesh = server.gui.add_checkbox("Show Background Mesh", False)
        gui_show_bg_pt3ds = server.gui.add_checkbox("Show Background Pointcloud", True)
        if world_env is not None:
            gui_show_fg_pt3ds = server.gui.add_checkbox("Show Human Pointcloud", True)
            gui_show_camera_frustums = server.gui.add_checkbox("Show Camera Frustums", True)
        if retargeted_poses is not None:
            gui_show_robot = server.gui.add_checkbox("Show Robot", True)
    
    # Human visualization
    with server.gui.add_folder("Human Visualization"):
        if smpl_verts is not None:
            gui_show_smpl_mesh = server.gui.add_checkbox("Show SMPL Mesh", True)
            gui_show_smpl_joints = server.gui.add_checkbox("Show SMPL Joints", False)
    
    # ========================================================================
    # Scene Setup
    # ========================================================================
    
    # Add background mesh
    bg_mesh_handle = server.scene.add_mesh_simple(
        "/bg_mesh",
        vertices=background_mesh.vertices,
        faces=background_mesh.faces,
        color=(200, 200, 200),
        opacity=1.0,
        material="standard",
        flat_shading=False,
        side="double",
        visible=gui_show_bg_mesh.value
    )
    
    # Add background point cloud (if available)
    bg_pt3ds_handle = None
    if bg_pt3ds is not None and bg_colors is not None:
        # Use all background points without confidence filtering
        filtered_bg_pt3ds = bg_pt3ds
        filtered_bg_colors = bg_colors
            
        # Ensure colors are in uint8 format (0-255)
        if filtered_bg_colors.dtype != onp.uint8:
            if filtered_bg_colors.max() <= 1.0:
                filtered_bg_colors = (filtered_bg_colors * 255).astype(onp.uint8)
            else:
                filtered_bg_colors = filtered_bg_colors.astype(onp.uint8)
        
        bg_pt3ds_handle = server.scene.add_point_cloud(
            "/bg_pt3ds",
            points=filtered_bg_pt3ds,
            colors=filtered_bg_colors,
            point_size=gui_point_size.value,
            visible=gui_show_bg_pt3ds.value
        )
    
    # Setup robot(s) if available
    robot_frames = {}
    urdf_visers = {}
    if retargeted_poses is not None:
        # Load URDF
        if robot_name == "g1":
            urdf_path = osp.join(osp.dirname(__file__), "../assets/robot_asset/g1/g1_29dof_anneal_23dof_foot.urdf")
            urdf = yourdfpy.URDF.load(urdf_path)
        else:
            raise ValueError(f"Robot {robot_name} not supported")
        
        # Create robot frames and urdf visers for all persons
        if is_multiperson:
            for person_id in person_ids:
                robot_frames[person_id] = server.scene.add_frame(f"/robot_{person_id}", axes_length=0.2, show_axes=False)
                urdf_visers[person_id] = ViserUrdf(
                    server,
                    urdf_or_path=urdf,
                    root_node_name=f"/robot_{person_id}",
                )
        else:
            # Single robot for backward compatibility
            robot_frames["single"] = server.scene.add_frame("/robot", axes_length=0.2, show_axes=False)
            urdf_visers["single"] = ViserUrdf(
                server,
                urdf_or_path=urdf,
                root_node_name="/robot",
            )
    
    # Setup camera frustums
    camera_frustums = []
    if world_env is not None:
        frame_names = [name for name in list(world_env.keys())]
        vfov_rad_list, aspect_list, rgbimg_list, quat_list, trans_list = [], [], [], [], []
        
        for frame_name, cam2world in zip(frame_names, cam2worlds):
            quat = R.from_matrix(cam2world[:3, :3]).as_quat().astype(onp.float32)
            quat = onp.concatenate([quat[3:], quat[:3]])  # xyzw to wxyz
            
            rgbimg = world_env[frame_name]['rgbimg'].astype(onp.uint8)
            rgbimg = rgbimg[::bg_pc_downsample_factor//2, ::bg_pc_downsample_factor//2, :]
            K = world_env[frame_name]['intrinsic'].astype(onp.float32)
            vfov_rad = onp.float32(2 * onp.arctan(K[1, 2] / K[1, 1]))
            aspect = onp.float32(rgbimg.shape[1] / rgbimg.shape[0])
            
            vfov_rad_list.append(vfov_rad)
            aspect_list.append(aspect)
            rgbimg_list.append(rgbimg)
            quat_list.append(quat)
            trans_list.append(cam2world[:3, 3].astype(onp.float32))
    
    # Setup dynamic elements
    human_pts3d_handles = []
    smpl_mesh_handles = []
    smpl_joints_handles = []
    
    # ========================================================================
    # Update Functions
    # ========================================================================
    
    def update_robot_cfg(t: int):
        """Update robot configuration at timestep t."""
        if retargeted_poses is None or not gui_show_robot.value:
            # Hide all robots
            for robot_frame in robot_frames.values():
                robot_frame.visible = False
            for urdf_viser in urdf_visers.values():
                for joint_frame in urdf_viser._joint_frames:
                    joint_frame.visible = False
                for mesh_node in urdf_viser._meshes:
                    mesh_node.visible = False
            return
        
        if is_multiperson:
            # Update multiple robots
            for person_id in person_ids:
                person_data = retargeted_poses["persons"][person_id]
                
                if t >= person_data["joints"].shape[0]:
                    # Hide robot if timestep exceeds this person's data
                    robot_frames[person_id].visible = False
                    for joint_frame in urdf_visers[person_id]._joint_frames:
                        joint_frame.visible = False
                    for mesh_node in urdf_visers[person_id]._meshes:
                        mesh_node.visible = False
                    continue
                
                # Show and update robot
                robot_frames[person_id].visible = True
                for joint_frame in urdf_visers[person_id]._joint_frames:
                    joint_frame.visible = True
                for mesh_node in urdf_visers[person_id]._meshes:
                    mesh_node.visible = True
                
                T_world_robot_xyzw = person_data["root_quat"][t]  # xyzw
                T_world_robot_xyz = person_data["root_pos"][t]    # xyz
                T_world_robot_wxyz = onp.concatenate([T_world_robot_xyzw[3:], T_world_robot_xyzw[:3]])
                
                robot_frames[person_id].wxyz = onp.array(T_world_robot_wxyz, dtype=onp.float32)
                robot_frames[person_id].position = onp.array(T_world_robot_xyz, dtype=onp.float32)
                
                # Update joints
                joints = onp.array(person_data["joints"][t], dtype=onp.float32)
                if len(joints) > 8:  # Check if joint index 8 exists
                    joints[8] = 0.0  # TEMP - same as in retargeting_visualization.py
                urdf_visers[person_id].update_cfg(joints)
        else:
            # Single robot update
            robot_frames["single"].visible = True
            for joint_frame in urdf_visers["single"]._joint_frames:
                joint_frame.visible = True
            for mesh_node in urdf_visers["single"]._meshes:
                mesh_node.visible = True
            
            T_world_robot_xyzw = retargeted_poses["root_quat"][t]  # xyzw
            T_world_robot_xyz = retargeted_poses["root_pos"][t]    # xyz
            T_world_robot_wxyz = onp.concatenate([T_world_robot_xyzw[3:], T_world_robot_xyzw[:3]])
            
            robot_frames["single"].wxyz = onp.array(T_world_robot_wxyz, dtype=onp.float32)
            robot_frames["single"].position = onp.array(T_world_robot_xyz, dtype=onp.float32)
            
            # Update joints
            joints = onp.array(retargeted_poses["joints"][t], dtype=onp.float32)
            if len(joints) > 8:  # Check if joint index 8 exists
                joints[8] = 0.0  # TEMP - same as in retargeting_visualization.py
            urdf_visers["single"].update_cfg(joints)
    
    def update_camera_frustum(t: int):
        """Update camera frustum at timestep t."""
        if not gui_show_camera_frustums.value:
            for frustum in camera_frustums:
                frustum.visible = False
            return
        
        if len(camera_frustums) <= t:
            frustum = server.scene.add_camera_frustum(
                f"/cameras/{t}",
                fov=vfov_rad_list[t],
                aspect=aspect_list[t],
                scale=gui_frustum_scale.value,
                line_width=1.0,
                color=(255, 127, 14),
                wxyz=quat_list[t],
                position=trans_list[t],
                image=rgbimg_list[t],
            )
            camera_frustums.append(frustum)
        else:
            for frustum in camera_frustums:
                frustum.visible = False
                # Update scale for existing frustums
                if hasattr(frustum, 'scale'):
                    frustum.scale = gui_frustum_scale.value
            if 'gui_play_camera_to_follow' in locals() and not gui_play_camera_to_follow.value:
                camera_frustums[t].visible = True
    
    def update_human_pointcloud(t: int):
        """Update human pointcloud at timestep t."""
        if not gui_show_fg_pt3ds.value or t >= len(fg_pt3ds):
            for handle in human_pts3d_handles:
                handle.visible = False
            return
        
        # Use foreground points as-is (filtering already applied during preprocessing)
        pt3ds_filtered, colors_filtered = fg_pt3ds[t], fg_colors[t]
        
        if len(human_pts3d_handles) <= t:
            # Ensure colors are in uint8 format (0-255)
            if colors_filtered.dtype != onp.uint8:
                if colors_filtered.max() <= 1.0:
                    colors_filtered = (colors_filtered * 255).astype(onp.uint8)
                else:
                    colors_filtered = colors_filtered.astype(onp.uint8)
            
            handle = server.scene.add_point_cloud(
                f"/human_pt3ds/{t}",
                points=pt3ds_filtered,
                colors=colors_filtered,
                point_size=gui_point_size.value,
            )
            human_pts3d_handles.append(handle)
        else:
            for handle in human_pts3d_handles:
                handle.visible = False
            human_pts3d_handles[t].visible = True
    
        # Visualize the smpl mesh
    if all_smpl_verts:
        smpl_mesh_handle_list = []  # List of lists, one for each timestep
        def update_smpl_mesh(t: int):
            if len(smpl_mesh_handle_list) <= t:
                # Get frame name for this timestep
                frame_names = list(world_env.keys()) if world_env is not None else []
                current_frame_name = frame_names[t] if t < len(frame_names) else None
                
                timestep_handles = []
                # Create mesh for each person
                for p_idx, (person_id, smpl_verts) in enumerate(all_smpl_verts.items()):
                    # Create mesh with contact visualization if available
                    if (all_contact_estimation.get(person_id) is not None and current_frame_name is not None and 
                        current_frame_name in all_contact_estimation[person_id] and 
                        int(person_id) in all_contact_estimation[person_id][current_frame_name]):
                        
                        vertices = smpl_verts[t]
                        contact_info = all_contact_estimation[person_id][current_frame_name][int(person_id)]
                        
                        # Create mesh with contact visualization
                        mesh = create_contact_mesh(vertices, smpl_faces, contact_info, int(person_id))
                        smpl_mesh_handle = server.scene.add_mesh_trimesh(
                            f"/smpl_mesh/{t}/person_{person_id}", mesh=mesh
                        )
                    else:
                        # Standard mesh without contact visualization
                        smpl_mesh_handle = server.scene.add_mesh_simple(
                            f"/smpl_mesh/{t}/person_{person_id}",
                            vertices=smpl_verts[t],
                            faces=smpl_faces,
                            flat_shading=False,
                            wireframe=False,
                            color=get_color(int(person_id))  # Different color for each person
                        )
                    timestep_handles.append(smpl_mesh_handle)
                
                smpl_mesh_handle_list.append(timestep_handles)
            else:
                # Hide all meshes first
                for timestep_handles in smpl_mesh_handle_list:
                    for smpl_mesh_handle in timestep_handles:
                        smpl_mesh_handle.visible = False
                
                # Show current timestep meshes
                if 'gui_show_smpl_mesh' in locals() and gui_show_smpl_mesh.value:
                    for smpl_mesh_handle in smpl_mesh_handle_list[t]:
                        smpl_mesh_handle.visible = True
    
    # Visualize the joints
    if all_smpl_joints3d:
        smpl_joints3d_handle_list = []  # List of lists, one for each timestep
        def update_smpl_joints(t: int):
            if len(smpl_joints3d_handle_list) <= t:
                timestep_handles = []
                # Create joints for each person
                for person_id, smpl_joints3d in all_smpl_joints3d.items():
                    # Get color for this person
                    person_color = get_color(int(person_id))
                    smpl_joints3d_handle = server.scene.add_point_cloud(
                        f"/smpl_joints3d/{t}/person_{person_id}",
                        points=smpl_joints3d[t],
                        colors=onp.array([person_color] * smpl_joints3d[t].shape[0]),
                        point_size=0.03,
                        point_shape="circle",
                    )
                    timestep_handles.append(smpl_joints3d_handle)
                smpl_joints3d_handle_list.append(timestep_handles)
            else:
                # Hide all joints first
                for timestep_handles in smpl_joints3d_handle_list:
                    for smpl_joints3d_handle in timestep_handles:
                        smpl_joints3d_handle.visible = False
                
                # Show current timestep joints
                if 'gui_show_smpl_joints' in locals() and gui_show_smpl_joints.value:
                    for smpl_joints3d_handle in smpl_joints3d_handle_list[t]:
                        smpl_joints3d_handle.visible = True
    
    def setup_camera_follow():
        """Setup camera to follow SMPL motion."""
        nonlocal stop_camera_follow, resume_camera_follow
        if not all_smpl_verts or len(smpl_mesh_handle_list) < num_frames:
            return
        
        # Hide camera frustums when following
        for frustum in camera_frustums:
            frustum.visible = False
        
        # Use first person's center for camera follow
        first_person_verts = all_smpl_verts[first_person_id]
        target_positions = onp.array([first_person_verts[i].mean(axis=0) for i in range(num_frames)], dtype=onp.float32)
        
        if world_env is not None:
            # Calculate average FOV from camera intrinsics
            fov_degrees_list = []
            for frame_name in world_env.keys():
                K = world_env[frame_name]['intrinsic'].astype(onp.float32)
                vfov_rad = onp.float32(2 * onp.arctan(K[1, 2] / K[1, 1]))
                fov_degrees_list.append(onp.degrees(vfov_rad))
            avg_fov = onp.float32(onp.mean(fov_degrees_list))
        else:
            avg_fov = onp.float32(45.0)
        
        camera_positions = onp.array([cam2worlds[i][:3, 3] for i in range(min(len(cam2worlds), num_frames))], dtype=onp.float32)
        camera_wxyz = onp.array([quat_list[i] for i in range(min(len(quat_list), num_frames))], dtype=onp.float32)
        
        stop_camera_follow, resume_camera_follow = viser_camera_util.setup_camera_follow(
            server=server,
            slider=gui_timestep,
            target_positions=target_positions,
            camera_positions=camera_positions,
            camera_wxyz=camera_wxyz,
            fov=avg_fov
        )
    
    def update_background_pointcloud():
        """Update background pointcloud with current filtering settings."""
        if bg_pt3ds_handle is not None and bg_pt3ds is not None and bg_colors is not None:
            # Use all background points without confidence filtering
            filtered_bg_pt3ds = bg_pt3ds
            filtered_bg_colors = bg_colors
            
            # Ensure colors are in uint8 format (0-255)
            if filtered_bg_colors.dtype != onp.uint8:
                if filtered_bg_colors.max() <= 1.0:
                    filtered_bg_colors = (filtered_bg_colors * 255).astype(onp.uint8)
                else:
                    filtered_bg_colors = filtered_bg_colors.astype(onp.uint8)
            
            # Update the point cloud
            bg_pt3ds_handle.points = filtered_bg_pt3ds
            bg_pt3ds_handle.colors = filtered_bg_colors
            bg_pt3ds_handle.point_size = gui_point_size.value
    
    def update_all_pointclouds_with_gradient_filter():
        """Re-compute all point clouds with current gradient threshold."""
        nonlocal fg_pt3ds, fg_colors, fg_confs, bg_pt3ds, bg_colors
        
        # Reprocess all frames with current gradient threshold
        new_fg_pt3ds, new_fg_colors, new_fg_confs = [], [], []
        new_bg_pt3ds_list, new_bg_colors_list = [], []
        
        frame_names = list(world_env.keys())
        for i, frame_name in enumerate(frame_names):
            pt3d = world_env[frame_name]['pts3d'].reshape(-1, 3).astype(onp.float32)
            conf = world_env[frame_name]['conf'].reshape(-1).astype(onp.float32)
            colors = world_env[frame_name]['rgbimg'].reshape(-1, 3)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(onp.uint8)
            else:
                colors = colors.astype(onp.uint8)
            dynamic_msk = world_env[frame_name]['dynamic_msk'].astype(onp.uint8)
            
            # Apply current gradient filtering
            gradient_mask = onp.ones(pt3d.shape[0], dtype=bool)
            if 'depths' in world_env[frame_name]:
                depths = world_env[frame_name]['depths']
                dy, dx = onp.gradient(depths)
                gradient_magnitude = onp.sqrt(dx**2 + dy**2)
                gradient_mask = gradient_magnitude.flatten() < gui_gradient_threshold.value
            
            # Dilate dynamic mask
            kernel = onp.ones((20, 20), onp.uint8)
            dynamic_msk = cv2.dilate(dynamic_msk, kernel, iterations=1).flatten() > 0
            
            # Separate foreground and background
            fg_mask = dynamic_msk & gradient_mask
            bg_mask = ~dynamic_msk & gradient_mask
            
            new_fg_pt3ds.append(pt3d[fg_mask].astype(onp.float32))
            new_fg_colors.append(colors[fg_mask])
            new_fg_confs.append(conf[fg_mask].astype(onp.float32))
            
            new_bg_pt3ds_list.append(pt3d[bg_mask][::bg_pc_downsample_factor])
            new_bg_colors_list.append(colors[bg_mask][::bg_pc_downsample_factor])
        
        # Update global variables
        fg_pt3ds = new_fg_pt3ds
        fg_colors = new_fg_colors
        fg_confs = new_fg_confs
        
        # Update background point cloud
        if new_bg_pt3ds_list:
            bg_pt3ds = onp.concatenate(new_bg_pt3ds_list, axis=0)
            bg_colors = onp.concatenate(new_bg_colors_list, axis=0)
            
            if bg_pt3ds_handle is not None:
                bg_pt3ds_handle.points = bg_pt3ds
                bg_pt3ds_handle.colors = bg_colors
        
        # Update current foreground point cloud display
        current_timestep = gui_timestep.value
        if current_timestep < len(human_pts3d_handles) and human_pts3d_handles[current_timestep] is not None:
            if current_timestep < len(fg_pt3ds):
                pt3ds_filtered, colors_filtered = fg_pt3ds[current_timestep], fg_colors[current_timestep]
                if colors_filtered.dtype != onp.uint8:
                    if colors_filtered.max() <= 1.0:
                        colors_filtered = (colors_filtered * 255).astype(onp.uint8)
                    else:
                        colors_filtered = colors_filtered.astype(onp.uint8)
                
                human_pts3d_handles[current_timestep].points = pt3ds_filtered
                human_pts3d_handles[current_timestep].colors = colors_filtered
    
    # ========================================================================
    # GUI Callbacks
    # ========================================================================
    
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click  
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)
    
    @gui_show_bg_mesh.on_update
    def _(_) -> None:
        bg_mesh_handle.visible = gui_show_bg_mesh.value
    
    @gui_show_bg_pt3ds.on_update
    def _(_) -> None:
        if bg_pt3ds_handle is not None:
            bg_pt3ds_handle.visible = gui_show_bg_pt3ds.value
    
    @gui_point_size.on_update
    def _(_) -> None:
        update_background_pointcloud()
    
    @gui_gradient_threshold.on_update
    def _(_) -> None:
        # Update filtering for all point clouds
        update_background_pointcloud()
        # Also update foreground point clouds with new gradient threshold
        if world_env is not None:
            update_all_pointclouds_with_gradient_filter()
    
    @gui_frustum_scale.on_update
    def _(_) -> None:
        # Update scale for all camera frustums
        for frustum in camera_frustums:
            if hasattr(frustum, 'scale'):
                frustum.scale = gui_frustum_scale.value
    
    if 'gui_play_camera_to_follow' in locals():
        @gui_play_camera_to_follow.on_update
        def _(_) -> None:
            if stop_camera_follow is not None and resume_camera_follow is not None:
                if gui_play_camera_to_follow.value:
                    resume_camera_follow()
                else:
                    stop_camera_follow()

    # Main update loop
    @gui_timestep.on_update
    def _(_) -> None:
        current_timestep = gui_timestep.value
        
        with server.atomic():
            # Update all dynamic elements
            if retargeted_poses is not None:
                update_robot_cfg(current_timestep)
            
            if world_env is not None:
                update_camera_frustum(current_timestep)
                update_human_pointcloud(current_timestep)
            
            if all_smpl_verts:
                update_smpl_mesh(current_timestep)
            if all_smpl_joints3d:
                update_smpl_joints(current_timestep)
            
            # Update ego-view images
            if gui_ego_rgb_image is not None and current_timestep < len(ego_view_rgbimg_list):
                gui_ego_rgb_image.image = ego_view_rgbimg_list[current_timestep]
                gui_ego_depth_image.image = ego_view_depthimg_list[current_timestep]
            
            # Setup camera follow once all meshes are created
            if len(smpl_mesh_handle_list) == num_frames and stop_camera_follow is None:
                setup_camera_follow()
    
    
    # Add distance measurement tool
    add_distance_measurement(server)
    
    # Add data info
    data_name = postprocessed_dir.name
    if is_megasam and "megahunter_megasam_reconstruction_results_" in data_name:
        data_name = data_name.split("megahunter_megasam_reconstruction_results_")[1]
    elif not is_megasam and "megahunter_align3r_reconstruction_results_" in data_name:
        data_name = data_name.split("megahunter_align3r_reconstruction_results_")[1]
        
    # Start playback loop
    print(f"Starting visualization with {num_frames} frames")
    print(f"Open browser to http://localhost:8081")
    
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)