# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Sequential Visualization with Dropdown Selection

This script provides comprehensive visualization for browsing multiple processed results sequentially.
The key feature is the dropdown menu that allows switching between different results without restarting,
making it ideal for reviewing datasets built with sequential processing.

Features:
- Dropdown selection for switching between different results
- Background environment mesh and point clouds
- SMPL human mesh and joints with multi-person support
- Robot motion (retargeted) with multi-person support
- Ego-view rendering from human head pose
- Camera frustums with adjustable scale
- Interactive point filtering (confidence & gradient thresholds)
- Organized GUI controls

Data structure expected in each result directory:
├── gravity_calibrated_keypoints.h5      # Human keypoints and world rotation
├── gravity_calibrated_megahunter*.h5    # Updated hunter file with calibrated coordinates  
├── background_mesh.obj                  # Environment mesh
├── background_*_filtered_colored_pointcloud.ply  # Point clouds
├── retarget_poses_g1.h5                # Robot motion data
"""

from __future__ import annotations

import time
import os
import os.path as osp
import glob
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

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

# Import root directory
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
            result[key] = h5file[key_path][:]
    
    # Load attributes
    for attr_key, attr_value in h5file.attrs.items():
        if attr_key.startswith(path):
            result[attr_key[len(path):]] = attr_value

    return result

def load_megahunter_data(
    megahunter_path: Path,
    person_id: str,
    device: str = 'cuda'
) -> Tuple[Optional[Dict], Optional[onp.ndarray], Optional[onp.ndarray], 
           Optional[onp.ndarray], List[onp.ndarray], List[onp.ndarray], 
           List[onp.ndarray], List[onp.ndarray], Optional[onp.ndarray], 
           Optional[onp.ndarray], Optional[smplx.SMPL], Any, Optional[Dict]]:
    """Load MegaHunter data for a specific person."""
    
    if not megahunter_path.exists():
        return (None,) * 13
    
    with h5py.File(megahunter_path, 'r') as f:
        megahunter_data = load_dict_from_hdf5(f)
    
    world_env = megahunter_data.get('our_pred_world_cameras_and_structure')
    human_params = megahunter_data.get('our_pred_humans_smplx_params', {})
    
    if person_id not in human_params:
        return (None,) * 13
    
    # Extract SMPL data
    gender = human_params[person_id].get('gender', 'male')
    if isinstance(gender, onp.ndarray):
        gender = gender.item()
    
    smpl_layer = smplx.create(
        model_path='./assets/body_models',
        model_type='smpl',
        gender=gender,
        num_betas=10,
        batch_size=len(human_params[person_id]['body_pose'])
    ).to(device)
    
    # Process SMPL parameters
    smpl_betas = torch.from_numpy(human_params[person_id]['betas']).to(device)
    if smpl_betas.ndim == 1:
        smpl_betas = smpl_betas.repeat(len(human_params[person_id]['body_pose']), 1)
    
    smpl_output = smpl_layer(
        body_pose=torch.from_numpy(human_params[person_id]['body_pose']).to(device),
        betas=smpl_betas,
        global_orient=torch.from_numpy(human_params[person_id]['global_orient']).to(device),
        pose2rot=False
    )
    
    smpl_joints = smpl_output['joints']  # (T, 45, 3)
    smpl_root_joint = smpl_joints[:, 0:1, :]  # (T, 1, 3)
    
    # Handle root translation dimensions properly
    if human_params[person_id]['root_transl'].ndim == 2:
        smpl_root_transl = human_params[person_id]['root_transl'][:, None, :]
    else:
        smpl_root_transl = human_params[person_id]['root_transl']
    
    smpl_verts = smpl_output['vertices'] - smpl_root_joint + torch.from_numpy(smpl_root_transl).to(device)
    
    # Get head pose for ego-view rendering using SMPL joint transforms
    smpl_joints_transforms = smpl_output['joints_transforms']  # (T, 24, 4, 4)
    head_joint_idx = SMPL_KEYPOINTS.index('head')
    smpl_head_transform = smpl_joints_transforms[:, head_joint_idx, :, :]  # (T, 4, 4)
    # Adjust head position with root translation
    smpl_head_transform[:, :3, 3] = smpl_head_transform[:, :3, 3] - smpl_root_joint[:, 0, :] + torch.from_numpy(smpl_root_transl[:, 0, :]).to(device)
    
    # Convert to numpy
    smpl_joints3d = (smpl_joints.detach().cpu().numpy() - smpl_root_joint.detach().cpu().numpy() + smpl_root_transl).astype(onp.float32)
    smpl_verts = smpl_verts.detach().cpu().numpy().astype(onp.float32)
    head_transform = smpl_head_transform.detach().cpu().numpy().astype(onp.float32)
    
    # Process point clouds
    fg_pt3ds, fg_colors, fg_confs, cam2worlds = [], [], [], []
    
    if world_env is not None:
        frame_names = megahunter_data.get('person_frame_info_list', {}).get(person_id, [])
        frame_names = frame_names.astype(str)
        kernel = onp.ones((20, 20), onp.uint8)
        
        for frame_name in frame_names:
            frame_name = frame_name.item() if hasattr(frame_name, 'item') else frame_name
            
            if frame_name in world_env:
                pt3d = world_env[frame_name]['pts3d'].reshape(-1, 3).astype(onp.float32)
                colors = world_env[frame_name]['rgbimg'].reshape(-1, 3)
                if colors.max() <= 1.0:
                    colors = (colors * 255).astype(onp.uint8)
                else:
                    colors = colors.astype(onp.uint8)
                    
                conf = world_env[frame_name]['conf'].reshape(-1).astype(onp.float32)
                cam2world = world_env[frame_name]['cam2world'].astype(onp.float32)
                dynamic_msk = world_env[frame_name]['dynamic_msk'].astype(onp.uint8)
                
                # Apply gradient filtering
                gradient_mask = onp.ones(pt3d.shape[0], dtype=bool)
                if 'depths' in world_env[frame_name]:
                    depths = world_env[frame_name]['depths']
                    dy, dx = onp.gradient(depths)
                    gradient_magnitude = onp.sqrt(dx**2 + dy**2)
                    gradient_mask = gradient_magnitude.flatten() < 0.05
                
                # Extract foreground
                dynamic_msk = cv2.dilate(dynamic_msk, kernel, iterations=1).flatten() > 0
                fg_mask = dynamic_msk & gradient_mask
                
                fg_pt3ds.append(pt3d[fg_mask])
                fg_colors.append(colors[fg_mask])
                fg_confs.append(conf[fg_mask])
                cam2worlds.append(cam2world)
    
    # Load contact estimation
    contact_estimation = None
    left_foot_contact = None
    right_foot_contact = None
    
    # Extract contact directory
    if megahunter_path and world_env:
        video_name = osp.basename(osp.dirname(str(megahunter_path))).split('_cam')[0].split('_results_')[1]
        contact_dir = osp.join(osp.dirname(str(megahunter_path)), '..', '..', 'input_contacts', video_name, 'cam01')
        
        if osp.exists(contact_dir):
            contact_estimation = {}
            for frame_name in world_env.keys():
                contact_file = osp.join(contact_dir, f'{frame_name}.pkl')
                if osp.exists(contact_file):
                    with open(contact_file, 'rb') as f:
                        contact_estimation[frame_name] = pickle.load(f)
            
            # Extract foot contacts for this person
            left_foot_list, right_foot_list = [], []
            for frame_name in world_env.keys():
                if frame_name in contact_estimation and int(person_id) in contact_estimation[frame_name]:
                    left_foot_list.append(contact_estimation[frame_name][int(person_id)]["left_foot_contact"])
                    right_foot_list.append(contact_estimation[frame_name][int(person_id)]["right_foot_contact"])
            
            if left_foot_list:
                left_foot_contact = onp.array(left_foot_list)
                right_foot_contact = onp.array(right_foot_list)
    
    return (world_env, smpl_joints3d, smpl_verts, head_transform, 
            fg_pt3ds, fg_colors, fg_confs, cam2worlds,
            left_foot_contact, right_foot_contact, smpl_layer, smpl_layer.faces,
            contact_estimation)

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
        if t < len(smpl_head_transform):
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
        else:
            # Append blank images if no head transform
            ego_view_rgbimg_list.append(onp.zeros((img_res, img_res, 3), dtype=onp.uint8))
            ego_view_depthimg_list.append(onp.zeros((img_res, img_res, 3), dtype=onp.uint8))
    
    return ego_view_rgbimg_list, ego_view_depthimg_list


# ============================================================================
# Visualization State Class
# ============================================================================

class VisualizationState:
    """Manages the state of visualization elements that need to be cleared between results."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all visualization handles."""
        self.bg_mesh_handle = None
        self.bg_pt3ds_handle = None
        self.camera_frustums = []
        self.human_pts3d_handles = []
        self.smpl_mesh_handles = []
        self.smpl_mesh_handle_list = []  # New structure
        self.smpl_joints_handles = []
        self.robot_frames = {}
        self.urdf_visers = {}
        self.stop_camera_follow = None
        self.resume_camera_follow = None
        self.current_video_id = None  # Track current video
        self.camera_follow_setup_done = False
        
    def clear_scene(self, server: viser.ViserServer):
        """Remove all visualization elements from the scene."""
        # Helper function to safely remove a handle
        def safe_remove(handle):
            try:
                if handle is not None:
                    handle.remove()
            except (KeyError, AttributeError):
                # Handle already removed or doesn't exist
                pass
        
        # Remove background elements
        safe_remove(self.bg_mesh_handle)
        safe_remove(self.bg_pt3ds_handle)
            
        # Remove camera frustums
        for frustum in self.camera_frustums:
            safe_remove(frustum)
            
        # Remove human elements
        for handle in self.human_pts3d_handles:
            safe_remove(handle)
        
        # Handle both old and new smpl mesh structure
        if hasattr(self, 'smpl_mesh_handle_list'):
            for timestep_handles in self.smpl_mesh_handle_list:
                for handle in timestep_handles:
                    safe_remove(handle)
        else:
            for handle in self.smpl_mesh_handles:
                safe_remove(handle)
                
        for handle in self.smpl_joints_handles:
            safe_remove(handle)
            
        # Remove robots
        for frame in self.robot_frames.values():
            safe_remove(frame)
        
        # Remove URDF components
        for urdf_viser in self.urdf_visers.values():
            if hasattr(urdf_viser, '_joint_frames'):
                for joint_frame in urdf_viser._joint_frames:
                    safe_remove(joint_frame)
            if hasattr(urdf_viser, '_meshes'):
                for mesh_node in urdf_viser._meshes:
                    safe_remove(mesh_node)
            
        self.reset()


# ============================================================================
# Visualization Functions
# ============================================================================

def add_distance_measurement(server: viser.ViserServer) -> None:
    """Add distance measurement tool to the scene."""
    control0 = server.scene.add_transform_controls(
        "/controls/0",
        position=(1, 0, 0),
        scale=0.5,
        visible=False,
    )
    control1 = server.scene.add_transform_controls(
        "/controls/1",
        position=(1, 0, 0),
        scale=0.5,
        visible=False,
    )
    segments = server.scene.add_line_segments(
        "/controls/line",
        onp.array([control0.position, control1.position])[None, :, :],
        colors=(255, 0, 0),
        visible=False,
    )

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
        segments.points = onp.array([control0.position, control1.position])[None, :, :]

    control0.on_update(lambda _: update_distance())
    control1.on_update(lambda _: update_distance())


def load_and_visualize_result(
    postprocessed_dir: Path,
    server: viser.ViserServer,
    state: VisualizationState,
    gui_handles: Dict[str, Any],
    robot_name: str = "g1",
    bg_pc_downsample_factor: int = 4,
    no_spf: bool = False,
    save_ego_view: bool = False,
    device: str = 'cuda'
) -> int:
    """
    Load and visualize a single result directory.
    
    Returns:
        Number of frames in the visualization
    """
    # Clear previous visualization
    print(f"Clearing scene for: {postprocessed_dir.name}")
    state.is_loading = True
    state.clear_scene(server)
    
    # Set unique video ID for this load
    state.last_video_id = getattr(state, 'current_video_id', None)
    state.current_video_id = postprocessed_dir.name
    
    # Load gravity calibrated keypoints
    keypoints_path = postprocessed_dir / 'gravity_calibrated_keypoints.h5'
    if not keypoints_path.exists():
        print(f"Warning: {keypoints_path} not found")
        return 1
    
    with h5py.File(keypoints_path, 'r') as f:
        keypoints_data = load_dict_from_hdf5(f)
    print(f"Loaded {len(keypoints_data['joints'])} persons from {keypoints_path}")

    # Get person IDs and determine primary person
    person_ids = list(keypoints_data['joints'].keys())

    first_person_id = person_ids[0] if person_ids else None
    
    if first_person_id is None:
        print("No person data found")
        return 1
    
    num_frames = keypoints_data['joints'][first_person_id].shape[0]
    
    # Load megahunter data for detailed visualization
    megahunter_path = postprocessed_dir / 'gravity_calibrated_megahunter.h5'
    
    # Load additional data if megahunter file exists
    world_env = None
    all_smpl_joints3d = {}
    all_smpl_verts = {}
    all_smpl_head_transform = {}
    fg_pt3ds, fg_colors, fg_confs, cam2worlds = [], [], [], []
    smpl_faces = None
    all_contact_estimation = {}
    
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
            
            # Use world_env and point clouds from first person only
            if world_env is None:
                world_env = world_env_tmp
                fg_pt3ds, fg_colors, fg_confs, cam2worlds = fg_pt3ds_tmp, fg_colors_tmp, fg_confs_tmp, cam2worlds_tmp
                smpl_faces = smpl_faces_tmp
    
    # Load background data
    background_mesh = trimesh.load(postprocessed_dir / 'background_mesh.obj')
    
    # Extract background point cloud
    bg_pt3ds, bg_colors = None, None
    if world_env is not None:
        # Collect background points from all frames
        all_bg_pt3ds = []
        all_bg_colors = []
        all_bg_confs = []
        
        for frame_name, cam2world in zip(world_env.keys(), cam2worlds):
            pt3d = world_env[frame_name]['pts3d'].reshape(-1, 3).astype(onp.float32)
            colors = world_env[frame_name]['rgbimg'].reshape(-1, 3)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(onp.uint8)
            else:
                colors = colors.astype(onp.uint8)
            conf = world_env[frame_name]['conf'].reshape(-1).astype(onp.float32)
            dynamic_msk = world_env[frame_name]['dynamic_msk'].astype(onp.uint8)
            
            # Dilate dynamic mask
            kernel = onp.ones((20, 20), onp.uint8)
            dynamic_msk = cv2.dilate(dynamic_msk, kernel, iterations=1).flatten() > 0
            
            # Apply gradient filtering
            gradient_mask = onp.ones(pt3d.shape[0], dtype=bool)
            if 'depths' in world_env[frame_name]:
                depths = world_env[frame_name]['depths']
                dy, dx = onp.gradient(depths)
                gradient_magnitude = onp.sqrt(dx**2 + dy**2)
                gradient_mask = gradient_magnitude.flatten() < 0.05
            
            # Extract background points
            bg_mask = ~dynamic_msk & gradient_mask
            all_bg_pt3ds.append(pt3d[bg_mask])
            all_bg_colors.append(colors[bg_mask])
            all_bg_confs.append(conf[bg_mask])
        
        # Concatenate all background points
        bg_pt3ds = onp.concatenate(all_bg_pt3ds, axis=0)[::bg_pc_downsample_factor, :].astype(onp.float32)
        bg_colors = onp.concatenate(all_bg_colors, axis=0)[::bg_pc_downsample_factor, :].astype(onp.uint8)
        bg_confs = onp.concatenate(all_bg_confs, axis=0)[::bg_pc_downsample_factor].astype(onp.float32)
    else:
        # Fallback: load from PLY files
        if no_spf:
            pointcloud_path = postprocessed_dir / 'background_less_filtered_colored_pointcloud.ply'
        else:
            pointcloud_path = postprocessed_dir / 'background_more_filtered_colored_pointcloud.ply'
        
        pointcloud = trimesh.load(pointcloud_path)
        bg_pt3ds = onp.array(pointcloud.vertices, dtype=onp.float32)[::bg_pc_downsample_factor, :]
        bg_colors = onp.array(pointcloud.colors, dtype=onp.float32)[::bg_pc_downsample_factor, :3]
        # Convert colors to uint8 range if needed
        if bg_colors.max() <= 1.0:
            bg_colors = (bg_colors * 255).astype(onp.uint8)
        else:
            bg_colors = bg_colors.astype(onp.uint8)
        bg_confs = onp.ones(bg_pt3ds.shape[0], dtype=onp.float32)
    
    # Load retargeted poses - check for both single and multi-person formats
    retargeted_poses_path = postprocessed_dir / f'retarget_poses_{robot_name}.h5'
    retargeted_poses_multiperson_path = postprocessed_dir / f'retarget_poses_{robot_name}_multiperson.h5'
    
    retargeted_poses = None
    is_multiperson = False
    retarget_person_ids = []
    
    # Try loading multi-person file first
    if retargeted_poses_multiperson_path.exists():
        try:
            with h5py.File(retargeted_poses_multiperson_path, 'r') as f:
                retargeted_poses = load_dict_from_hdf5(f)
                is_multiperson = True
                retarget_person_ids = list(retargeted_poses["persons"].keys())
                print(f"Loaded multi-person retargeted poses: {len(retarget_person_ids)} persons")
        except Exception as e:
            print(f"Failed to load multi-person file: {e}")
    
    # Fall back to single person file
    if retargeted_poses is None and retargeted_poses_path.exists():
        try:
            with h5py.File(retargeted_poses_path, 'r') as f:
                retargeted_poses = load_dict_from_hdf5(f)
                is_multiperson = False
                print("Loaded single-person retargeted poses")
        except Exception as e:
            print(f"Failed to load single-person file: {e}")
    
    # Update num_frames based on retargeting data
    if retargeted_poses is not None:
        if is_multiperson:
            retarget_num_frames = max(retargeted_poses["persons"][pid]["joints"].shape[0] for pid in retarget_person_ids)
        else:
            retarget_num_frames = retargeted_poses["joints"].shape[0]
        num_frames = min(num_frames, retarget_num_frames)
    
    # Generate ego-view images
    ego_view_rgbimg_list, ego_view_depthimg_list = [], []
    if all_smpl_head_transform and first_person_id in all_smpl_head_transform and bg_pt3ds is not None and bg_colors is not None:
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
    
    # Update GUI max frames
    gui_handles['timestep'].max = num_frames - 1
    
    # Update ego-view images
    if ego_view_rgbimg_list and 'ego_rgb_image' in gui_handles:
        gui_handles['ego_rgb_image'].image = ego_view_rgbimg_list[0]
        gui_handles['ego_depth_image'].image = ego_view_depthimg_list[0]
    elif 'ego_rgb_image' in gui_handles:
        # Clear ego-view if no images available
        blank_img = onp.zeros((256, 256, 3), dtype=onp.uint8)
        gui_handles['ego_rgb_image'].image = blank_img
        gui_handles['ego_depth_image'].image = blank_img
    
    # ========================================================================
    # Setup Scene Elements
    # ========================================================================
    
    # Add background mesh
    state.bg_mesh_handle = server.scene.add_mesh_simple(
        "/bg_mesh",
        vertices=background_mesh.vertices.astype(onp.float32),
        faces=background_mesh.faces,
        color=(200, 200, 200),
        opacity=1.0,
        material="standard",
        flat_shading=False,
        side="double",
        visible=gui_handles['show_bg_mesh'].value
    )
    
    # Add background pointcloud
    if bg_pt3ds is not None and bg_colors is not None:
        print(f"Adding background point cloud with {len(bg_pt3ds)} points, visible={gui_handles['show_bg_pt3ds'].value}")
        # bg_pt3ds and bg_colors are already filtered during loading
        state.bg_pt3ds_handle = server.scene.add_point_cloud(
            "/bg_pt3ds",
            points=bg_pt3ds,
            colors=bg_colors,
            point_size=gui_handles['point_size'].value,
            visible=gui_handles['show_bg_pt3ds'].value
        )
    
    # Setup robots if available
    if retargeted_poses is not None:
        # Load URDF
        if robot_name == "g1":
            urdf_path = osp.join(osp.dirname(__file__), "../assets/robot_asset/g1/g1_29dof_anneal_23dof_foot.urdf")
            urdf = yourdfpy.URDF.load(urdf_path)
        else:
            raise ValueError(f"Robot {robot_name} not supported")
        
        # Create robot frames and urdf visers
        if is_multiperson:
            for person_id in retarget_person_ids:
                state.robot_frames[person_id] = server.scene.add_frame(
                    f"/robot_{person_id}", axes_length=0.2, show_axes=False
                )
                state.urdf_visers[person_id] = ViserUrdf(
                    server,
                    urdf_or_path=urdf,
                    root_node_name=f"/robot_{person_id}",
                )
        else:
            print(f"Creating single robot for {len(retargeted_poses['joints'])} frames")
            state.robot_frames["single"] = server.scene.add_frame(
                "/robot", axes_length=0.2, show_axes=False
            )
            state.urdf_visers["single"] = ViserUrdf(
                server,
                urdf_or_path=urdf,
                root_node_name="/robot",
            )
    
    # Setup camera frustums
    vfov_rad_list, aspect_list, rgbimg_list, quat_list, trans_list = [], [], [], [], []
    if world_env is not None and cam2worlds is not None:
        frame_names = list(world_env.keys())
        
        # Ensure we don't exceed the number of cam2worlds
        num_camera_frames = min(len(frame_names), len(cam2worlds))
        
        for i in range(num_camera_frames):
            frame_name = frame_names[i]
            cam2world = cam2worlds[i]
            
            quat = R.from_matrix(cam2world[:3, :3]).as_quat().astype(onp.float32)
            quat = onp.concatenate([quat[3:], quat[:3]])  # xyzw to wxyz
            
            rgbimg = world_env[frame_name]['rgbimg']
            # Convert to uint8 if needed
            if rgbimg.dtype != onp.uint8:
                if rgbimg.max() <= 1.0:
                    rgbimg = (rgbimg * 255).astype(onp.uint8)
                else:
                    rgbimg = rgbimg.astype(onp.uint8)
            rgbimg = rgbimg[::bg_pc_downsample_factor//2, ::bg_pc_downsample_factor//2, :]
            K = world_env[frame_name]['intrinsic'].astype(onp.float32)
            vfov_rad = onp.float32(2 * onp.arctan(K[1, 2] / K[1, 1]))
            aspect = onp.float32(rgbimg.shape[1] / rgbimg.shape[0])
            
            vfov_rad_list.append(vfov_rad)
            aspect_list.append(aspect)
            rgbimg_list.append(rgbimg)
            quat_list.append(quat)
            trans_list.append(cam2world[:3, 3].astype(onp.float32))
        
        print(f"Created {len(vfov_rad_list)} camera frustums for {num_frames} total frames")
    
    # ========================================================================
    # Update Functions
    # ========================================================================
    
    # Helper function to safely remove a handle
    def safe_remove(handle):
        try:
            if handle is not None:
                handle.remove()
        except (KeyError, AttributeError):
            pass
    
    def update_robot_cfg(t: int):
        """Update robot configuration at timestep t."""
        if retargeted_poses is None or not gui_handles['show_robot'].value:
            for robot_frame in state.robot_frames.values():
                robot_frame.visible = False
            for urdf_viser in state.urdf_visers.values():
                for joint_frame in urdf_viser._joint_frames:
                    joint_frame.visible = False
                for mesh_node in urdf_viser._meshes:
                    mesh_node.visible = False
            return
        
        if is_multiperson:
            for person_id in retarget_person_ids:
                if person_id not in state.robot_frames or person_id not in state.urdf_visers:
                    continue  # Robot not initialized yet
                    
                person_data = retargeted_poses["persons"][person_id]
                
                if t >= person_data["joints"].shape[0]:
                    state.robot_frames[person_id].visible = False
                    for joint_frame in state.urdf_visers[person_id]._joint_frames:
                        joint_frame.visible = False
                    for mesh_node in state.urdf_visers[person_id]._meshes:
                        mesh_node.visible = False
                    continue
                
                # Show and update robot
                state.robot_frames[person_id].visible = True
                for joint_frame in state.urdf_visers[person_id]._joint_frames:
                    joint_frame.visible = True
                for mesh_node in state.urdf_visers[person_id]._meshes:
                    mesh_node.visible = True
                
                T_world_robot_xyzw = person_data["root_quat"][t]
                T_world_robot_xyz = person_data["root_pos"][t]
                T_world_robot_wxyz = onp.concatenate([T_world_robot_xyzw[3:], T_world_robot_xyzw[:3]])
                
                state.robot_frames[person_id].wxyz = onp.array(T_world_robot_wxyz, dtype=onp.float32)
                state.robot_frames[person_id].position = onp.array(T_world_robot_xyz, dtype=onp.float32)
                
                joints = onp.array(person_data["joints"][t], dtype=onp.float32)
                if len(joints) > 8:
                    joints[8] = 0.0
                state.urdf_visers[person_id].update_cfg(joints)
        else:
            # Single robot update
            if "single" not in state.robot_frames or "single" not in state.urdf_visers:
                return  # Robot not initialized yet
                
            state.robot_frames["single"].visible = True
            for joint_frame in state.urdf_visers["single"]._joint_frames:
                joint_frame.visible = True
            for mesh_node in state.urdf_visers["single"]._meshes:
                mesh_node.visible = True
            
            # Check bounds
            if t >= len(retargeted_poses["root_quat"]):
                return
                
            T_world_robot_xyzw = retargeted_poses["root_quat"][t]
            T_world_robot_xyz = retargeted_poses["root_pos"][t]
            T_world_robot_wxyz = onp.concatenate([T_world_robot_xyzw[3:], T_world_robot_xyzw[:3]])
            
            state.robot_frames["single"].wxyz = onp.array(T_world_robot_wxyz, dtype=onp.float32)
            state.robot_frames["single"].position = onp.array(T_world_robot_xyz, dtype=onp.float32)
            
            joints = onp.array(retargeted_poses["joints"][t], dtype=onp.float32)
            if len(joints) > 8:
                joints[8] = 0.0
            state.urdf_visers["single"].update_cfg(joints)
    
    def update_camera_frustum(t: int):
        """Update camera frustum at timestep t."""
        if not gui_handles['show_camera_frustums'].value or world_env is None:
            for frustum in state.camera_frustums:
                frustum.visible = False
            return
        
        # Check if we have data for this timestep
        if t >= len(vfov_rad_list) or t >= len(rgbimg_list):
            return
        
        # Hide all frustums first
        for frustum in state.camera_frustums:
            frustum.visible = False
            
        if len(state.camera_frustums) <= t:
            # Create new frustum
            frustum = server.scene.add_camera_frustum(
                f"/cameras/{t}",
                fov=vfov_rad_list[t],
                aspect=aspect_list[t],
                scale=gui_handles['frustum_scale'].value,
                line_width=1.0,
                color=(255, 127, 14),
                wxyz=quat_list[t],
                position=trans_list[t],
                image=rgbimg_list[t],
            )
            state.camera_frustums.append(frustum)
        else:
            # Check if this is the same video
            if hasattr(state, 'last_video_id') and state.last_video_id == state.current_video_id:
                # Same video, just update visibility
                pass  # Visibility already handled above
            else:
                # Different video, need to recreate frustum
                frustum = state.camera_frustums[t]
                # Remove old frustum and create new one with updated image
                safe_remove(frustum)
                frustum = server.scene.add_camera_frustum(
                    f"/cameras/{t}",
                    fov=vfov_rad_list[t],
                    aspect=aspect_list[t],
                    scale=gui_handles['frustum_scale'].value,
                    line_width=1.0,
                    color=(255, 127, 14),
                    wxyz=quat_list[t],
                    position=trans_list[t],
                    image=rgbimg_list[t],
                )
                state.camera_frustums[t] = frustum
            
        # Show current frustum if not following camera
        if 'play_camera_to_follow' not in gui_handles or not gui_handles['play_camera_to_follow'].value:
            state.camera_frustums[t].visible = True
    
    def update_human_pointcloud(t: int):
        """Update human pointcloud at timestep t."""
        if not gui_handles['show_fg_pt3ds'].value or t >= len(fg_pt3ds):
            for handle in state.human_pts3d_handles:
                handle.visible = False
            return
        
        # Hide all human point clouds first
        for handle in state.human_pts3d_handles:
            handle.visible = False
            
        pt3ds_filtered, colors_filtered = fg_pt3ds[t], fg_colors[t]
        
        # Ensure colors are in uint8
        if colors_filtered.dtype != onp.uint8:
            if colors_filtered.max() <= 1.0:
                colors_filtered = (colors_filtered * 255).astype(onp.uint8)
            else:
                colors_filtered = colors_filtered.astype(onp.uint8)
        
        if len(state.human_pts3d_handles) <= t:
            # Create new point cloud
            handle = server.scene.add_point_cloud(
                f"/human_pt3ds/{t}",
                points=pt3ds_filtered,
                colors=colors_filtered,
                point_size=gui_handles['point_size'].value,
            )
            state.human_pts3d_handles.append(handle)
        else:
            # Check if this is the same video
            if hasattr(state, 'last_video_id') and state.last_video_id == state.current_video_id:
                # Same video, just update visibility
                pass  # Already handled above
            else:
                # Different video, need to recreate point cloud
                handle = state.human_pts3d_handles[t]
                safe_remove(handle)
                handle = server.scene.add_point_cloud(
                    f"/human_pt3ds/{t}",
                    points=pt3ds_filtered,
                    colors=colors_filtered,
                    point_size=gui_handles['point_size'].value,
                )
                state.human_pts3d_handles[t] = handle
            
        state.human_pts3d_handles[t].visible = True
    
    def update_smpl_mesh(t: int):
        """Update SMPL mesh at timestep t."""
        if not all_smpl_verts:
            return
            
        # Initialize as list of lists if needed
        if not hasattr(state, 'smpl_mesh_handle_list'):
            state.smpl_mesh_handle_list = []
            
        if len(state.smpl_mesh_handle_list) <= t:
            # Create meshes for this timestep
            timestep_handles = []
            for p_idx, (person_id, smpl_verts) in enumerate(all_smpl_verts.items()):
                if t < len(smpl_verts):
                    vertices = smpl_verts[t]
                    color = get_color(p_idx)
                    
                    mesh_handle = server.scene.add_mesh_simple(
                        f"/smpl_mesh/{t}/person_{person_id}",
                        vertices=vertices,
                        faces=smpl_faces,
                        flat_shading=False,
                        wireframe=False,
                        color=color,
                        visible=False,  # Initially hidden
                    )
                    timestep_handles.append(mesh_handle)
            
            state.smpl_mesh_handle_list.append(timestep_handles)
        else:
            # Hide all meshes first
            for timestep_handles in state.smpl_mesh_handle_list:
                for handle in timestep_handles:
                    handle.visible = False
            
            # Check if this is the same video - if so, just update visibility
            if hasattr(state, 'last_video_id') and state.last_video_id == state.current_video_id:
                # Same video, just update visibility
                if gui_handles['show_smpl_mesh'].value and t < len(state.smpl_mesh_handle_list):
                    for handle in state.smpl_mesh_handle_list[t]:
                        handle.visible = True
            else:
                # Different video, need to recreate meshes
                timestep_handles = state.smpl_mesh_handle_list[t]
                new_handles = []
                
                # Remove old meshes and create new ones
                for handle in timestep_handles:
                    safe_remove(handle)
                    
                # Create new meshes for this timestep
                for p_idx, (person_id, smpl_verts) in enumerate(all_smpl_verts.items()):
                    if t < len(smpl_verts):
                        vertices = smpl_verts[t]
                        color = get_color(p_idx)
                        
                        mesh_handle = server.scene.add_mesh_simple(
                            f"/smpl_mesh/{t}/person_{person_id}",
                            vertices=vertices,
                            faces=smpl_faces,
                            flat_shading=False,
                            wireframe=False,
                            color=color,
                            visible=gui_handles['show_smpl_mesh'].value,
                        )
                        new_handles.append(mesh_handle)
                
                state.smpl_mesh_handle_list[t] = new_handles
    
    def update_smpl_joints(t: int):
        """Update SMPL joints at timestep t."""
        if not all_smpl_joints3d:
            return
            
        if len(state.smpl_joints_handles) <= t:
            # Create joints for each person with different colors
            for p_idx, (person_id, smpl_joints) in enumerate(all_smpl_joints3d.items()):
                if t < len(smpl_joints):
                    color = get_color(p_idx)
                    colors = onp.array([color] * smpl_joints[t].shape[0], dtype=onp.uint8)
                    
                    joints_handle = server.scene.add_point_cloud(
                        f"/smpl_joints/{t}/person_{person_id}",
                        points=smpl_joints[t],
                        colors=colors,
                        point_size=0.03,
                        point_shape="circle",
                        visible=gui_handles['show_smpl_joints'].value,
                    )
                    state.smpl_joints_handles.append(joints_handle)
        else:
            # Update visibility
            joints_idx = 0
            for p_idx, (person_id, smpl_joints) in enumerate(all_smpl_joints3d.items()):
                if t < len(smpl_joints) and joints_idx < len(state.smpl_joints_handles):
                    state.smpl_joints_handles[joints_idx].visible = gui_handles['show_smpl_joints'].value
                    joints_idx += 1
    
    def setup_camera_follow():
        """Setup camera to follow the human motion."""
        if state.stop_camera_follow is not None:
            return
        
        for frustum in state.camera_frustums:
            frustum.visible = False
        
        # Get target positions from first person
        if first_person_id in all_smpl_joints3d:
            target_positions = all_smpl_joints3d[first_person_id].mean(axis=1)
        else:
            target_positions = onp.zeros((num_frames, 3))
        
        # Calculate average FOV
        if world_env is not None:
            fov_degrees_list = []
            for frame_name in world_env.keys():
                K = world_env[frame_name]['intrinsic']
                vfov_rad = 2 * onp.arctan(K[1, 2] / K[1, 1])
                vfov_degrees = onp.degrees(vfov_rad)
                fov_degrees_list.append(vfov_degrees)
            avg_fov = onp.mean(fov_degrees_list) if fov_degrees_list else 45.0
        else:
            avg_fov = 45.0
        
        # Set up camera follow
        state.stop_camera_follow, state.resume_camera_follow = viser_camera_util.setup_camera_follow(
            server=server,
            slider=gui_handles['timestep'],
            target_positions=target_positions,
            camera_positions=trans_list if world_env else None,
            camera_wxyz=quat_list if world_env else None,
            fov=avg_fov
        )
    
    def update_all_pointclouds_with_gradient_filter():
        """Re-compute all point clouds with current gradient threshold."""
        nonlocal fg_pt3ds, fg_colors, fg_confs, bg_pt3ds, bg_colors
        
        if world_env is None:
            return
            
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
                gradient_mask = gradient_magnitude.flatten() < gui_handles['gradient_threshold'].value
            
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
            
            if state.bg_pt3ds_handle is not None:
                state.bg_pt3ds_handle.points = bg_pt3ds
                state.bg_pt3ds_handle.colors = bg_colors
        
        # Update current foreground point cloud display
        current_timestep = gui_handles['timestep'].value
        if current_timestep < len(state.human_pts3d_handles) and state.human_pts3d_handles[current_timestep] is not None:
            if current_timestep < len(fg_pt3ds):
                pt3ds_filtered, colors_filtered = fg_pt3ds[current_timestep], fg_colors[current_timestep]
                if colors_filtered.dtype != onp.uint8:
                    if colors_filtered.max() <= 1.0:
                        colors_filtered = (colors_filtered * 255).astype(onp.uint8)
                    else:
                        colors_filtered = colors_filtered.astype(onp.uint8)
                
                state.human_pts3d_handles[current_timestep].points = pt3ds_filtered
                state.human_pts3d_handles[current_timestep].colors = colors_filtered
    
    # Setup main update callback
    @gui_handles['timestep'].on_update
    def _(_) -> None:
        # Skip updates if loading
        if hasattr(state, 'is_loading') and state.is_loading:
            return
            
        current_timestep = gui_handles['timestep'].value
        
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
            if 'ego_rgb_list' in state.update_functions and state.update_functions['ego_rgb_list']:
                ego_rgb_list = state.update_functions['ego_rgb_list']
                ego_depth_list = state.update_functions['ego_depth_list']
                if current_timestep < len(ego_rgb_list):
                    if 'ego_rgb_image' in gui_handles:
                        gui_handles['ego_rgb_image'].image = ego_rgb_list[current_timestep]
                        gui_handles['ego_depth_image'].image = ego_depth_list[current_timestep]
            
            # Setup camera follow once
            if not hasattr(state, 'camera_follow_setup_done') or not state.camera_follow_setup_done:
                if len(state.smpl_mesh_handle_list) >= num_frames or len(state.smpl_joints_handles) >= num_frames:
                    setup_camera_follow()
                    state.camera_follow_setup_done = True
    
    # Store update functions for external access
    state.update_functions = {
        'robot': update_robot_cfg,
        'camera': update_camera_frustum,
        'human_pc': update_human_pointcloud,
        'smpl_mesh': update_smpl_mesh,
        'smpl_joints': update_smpl_joints,
        'setup_camera': setup_camera_follow,
        'bg_pt3ds': bg_pt3ds,
        'bg_colors': bg_colors,
        'bg_confs': bg_confs,
        'ego_rgb_list': ego_view_rgbimg_list,
        'ego_depth_list': ego_view_depthimg_list,
        'update_gradient_filter': update_all_pointclouds_with_gradient_filter,
    }
    
    # Trigger initial update to display first frame
    if num_frames > 0:
        with server.atomic():
            if retargeted_poses is not None:
                update_robot_cfg(0)
            if world_env is not None and len(vfov_rad_list) > 0:
                update_camera_frustum(0)
            if world_env is not None and len(fg_pt3ds) > 0:
                update_human_pointcloud(0)
            if all_smpl_verts:
                update_smpl_mesh(0)
            if all_smpl_joints3d:
                update_smpl_joints(0)
    
    state.is_loading = False
    return num_frames


# ============================================================================
# Main Function
# ============================================================================

def main(
    results_root_dir: str,
    robot_name: str = 'g1',
    pattern: str = 'megahunter_megasam_reconstruction_results',
    bg_pc_downsample_factor: int = 4,
    save_ego_view: bool = False,
):
    """
    Batch visualization with dropdown selection.
    
    Args:
        results_root_dir: Root directory containing multiple result directories
        robot_name: Robot name (e.g., 'g1')
        pattern: Pattern to match result directories
        bg_pc_downsample_factor: Downsampling factor for background point cloud
        save_ego_view: Whether to save ego-view images
    """
    # Find all result directories
    output_dir_list = glob.glob(osp.join(results_root_dir, '*'))
    output_dir_list = [d for d in output_dir_list if osp.isdir(d)]
    output_dir_list = sorted([d for d in output_dir_list if pattern in d])
    
    if not output_dir_list:
        print(f"No directories found matching pattern '{pattern}' in {results_root_dir}")
        return
    
    # Start viser server
    server = viser.ViserServer(port=8081)
    
    # Initialize visualization state
    state = VisualizationState()
    
    # Initialize camera follow functions
    stop_camera_follow = None
    resume_camera_follow = None
    
    # ========================================================================
    # GUI Setup
    # ========================================================================
    
    gui_handles = {}
    
    # Playback controls
    with server.gui.add_folder("Playback"):
        gui_handles['timestep'] = server.gui.add_slider(
            "Timestep", min=0, max=100, step=1, initial_value=0, disabled=True
        )
        gui_handles['next_frame'] = server.gui.add_button("Next Frame", disabled=True)
        gui_handles['prev_frame'] = server.gui.add_button("Prev Frame", disabled=True)
        gui_handles['playing'] = server.gui.add_checkbox("Playing", True)
        gui_handles['framerate'] = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=30)
        gui_handles['framerate_options'] = server.gui.add_button_group("FPS options", ("10", "20", "30", "60"))
        
        # Ego-view controls
        gui_handles['ego_rgb_image'] = server.gui.add_image(
            onp.zeros((480, 640, 3), dtype=onp.uint8), label="Ego RGB View"
        )
        gui_handles['ego_depth_image'] = server.gui.add_image(
            onp.zeros((480, 640, 3), dtype=onp.uint8), label="Ego Depth View"
        )
    
    # Point filtering controls
    with server.gui.add_folder("Point Filtering"):
        gui_handles['gradient_threshold'] = server.gui.add_slider(
            "Gradient Threshold", min=0.001, max=0.5, step=0.002, initial_value=0.05
        )
        gui_handles['point_size'] = server.gui.add_slider(
            "Point Size", min=0.001, max=0.02, step=0.001, initial_value=0.01
        )
    
    # Camera controls
    with server.gui.add_folder("Camera Controls"):
        gui_handles['frustum_scale'] = server.gui.add_slider(
            "Frustum Scale", min=0.1, max=0.5, step=0.01, initial_value=0.1
        )
        gui_handles['play_camera_to_follow'] = server.gui.add_checkbox(
            "Play Camera to Follow", initial_value=False
        )
    
    # Scene element visibility
    with server.gui.add_folder("Scene Elements"):
        gui_handles['show_bg_mesh'] = server.gui.add_checkbox("Show Background Mesh", False)
        gui_handles['show_bg_pt3ds'] = server.gui.add_checkbox("Show Background Pointcloud", True)
        gui_handles['show_fg_pt3ds'] = server.gui.add_checkbox("Show Human Pointcloud", True)
        gui_handles['show_camera_frustums'] = server.gui.add_checkbox("Show Camera Frustums", True)
        gui_handles['show_robot'] = server.gui.add_checkbox("Show Robot", True)
    
    # Human visualization
    with server.gui.add_folder("Human Visualization"):
        gui_handles['show_smpl_mesh'] = server.gui.add_checkbox("Show SMPL Mesh", True)
        gui_handles['show_smpl_joints'] = server.gui.add_checkbox("Show SMPL Joints", False)
    
    # Results selection
    with server.gui.add_folder("Results"):
        dir_name_list = []
        for output_dir in output_dir_list:
            if 'megahunter_megasam_reconstruction_results' in output_dir:
                dir_name_list.append(osp.basename(output_dir).split('megahunter_megasam_reconstruction_results_')[1])
            else:
                dir_name_list.append(osp.basename(output_dir))
        
        gui_handles['dropdown'] = server.gui.add_dropdown(
            "Result",
            options=tuple(str(p) for p in dir_name_list),
            initial_value=str(dir_name_list[0]) if dir_name_list else "",
        )
        gui_handles['next_prev'] = server.gui.add_button_group("", options=("Prev", "Next"))
        gui_handles['current_index'] = server.gui.add_text(
            "Current Index", f"1/{len(dir_name_list)}", disabled=True
        )
    
    # ========================================================================
    # GUI Callbacks
    # ========================================================================
    
    @gui_handles['next_frame'].on_click
    def _(_) -> None:
        max_frames = gui_handles['timestep'].max
        gui_handles['timestep'].value = (gui_handles['timestep'].value + 1) % (max_frames + 1)
    
    @gui_handles['prev_frame'].on_click
    def _(_) -> None:
        max_frames = gui_handles['timestep'].max
        gui_handles['timestep'].value = (gui_handles['timestep'].value - 1) % (max_frames + 1)
    
    @gui_handles['playing'].on_update
    def _(_) -> None:
        gui_handles['timestep'].disabled = gui_handles['playing'].value
        gui_handles['next_frame'].disabled = gui_handles['playing'].value
        gui_handles['prev_frame'].disabled = gui_handles['playing'].value
    
    @gui_handles['framerate_options'].on_click
    def _(_) -> None:
        gui_handles['framerate'].value = int(gui_handles['framerate_options'].value)
    
    @gui_handles['show_bg_mesh'].on_update
    def _(_) -> None:
        if state.bg_mesh_handle is not None:
            state.bg_mesh_handle.visible = gui_handles['show_bg_mesh'].value
    
    @gui_handles['show_bg_pt3ds'].on_update
    def _(_) -> None:
        if state.bg_pt3ds_handle is not None:
            state.bg_pt3ds_handle.visible = gui_handles['show_bg_pt3ds'].value
    
    @gui_handles['point_size'].on_update
    def _(_) -> None:
        if state.bg_pt3ds_handle is not None:
            state.bg_pt3ds_handle.point_size = gui_handles['point_size'].value
    
    @gui_handles['gradient_threshold'].on_update
    def _(_) -> None:
        # Update all point clouds with new gradient threshold
        if 'update_gradient_filter' in state.update_functions:
            state.update_functions['update_gradient_filter']()
    
    @gui_handles['frustum_scale'].on_update
    def _(_) -> None:
        for frustum in state.camera_frustums:
            if hasattr(frustum, 'scale'):
                frustum.scale = gui_handles['frustum_scale'].value
    
    @gui_handles['play_camera_to_follow'].on_update
    def _(_) -> None:
        if state.stop_camera_follow is not None and state.resume_camera_follow is not None:
            if gui_handles['play_camera_to_follow'].value:
                state.resume_camera_follow()
            else:
                state.stop_camera_follow()
    
    @gui_handles['next_prev'].on_click
    def _(_) -> None:
        idx = dir_name_list.index(gui_handles['dropdown'].value)
        if gui_handles['next_prev'].value == "Prev":
            idx = (idx - 1) % len(dir_name_list)
        else:
            idx = (idx + 1) % len(dir_name_list)
        gui_handles['dropdown'].value = str(dir_name_list[idx])
    
    # Add distance measurement tool
    add_distance_measurement(server)
    
    # ========================================================================
    # Main Loop
    # ========================================================================
    
    needs_update = True
    current_result_idx = -1
    num_frames = 1
    is_loading = False
    
    def load_new_result():
        nonlocal needs_update, current_result_idx, num_frames, is_loading
        
        selected_name = gui_handles['dropdown'].value
        new_idx = dir_name_list.index(selected_name)
        
        if new_idx == current_result_idx and not needs_update:
            return
        
        current_result_idx = new_idx
        needs_update = False
        is_loading = True
        
        # Update current index display
        gui_handles['current_index'].value = f"{new_idx + 1}/{len(dir_name_list)}"
        
        # Reset playback and stop updates
        gui_handles['playing'].value = False
        gui_handles['timestep'].value = 0
        gui_handles['timestep'].disabled = True  # Disable during loading
        
        # Load and visualize the new result
        print(f"Loading result: {selected_name}")
        postprocessed_dir = Path(output_dir_list[new_idx])
        
        try:
            num_frames = load_and_visualize_result(
                postprocessed_dir=postprocessed_dir,
                server=server,
                state=state,
                gui_handles=gui_handles,
                robot_name=robot_name,
                bg_pc_downsample_factor=bg_pc_downsample_factor,
                save_ego_view=save_ego_view,
            )
            print(f"Loaded {num_frames} frames")
            is_loading = False
            gui_handles['timestep'].disabled = False  # Re-enable after loading
            
            # Ensure timestep is within bounds
            if gui_handles['timestep'].value >= num_frames:
                gui_handles['timestep'].value = 0
            
            # Start playing
            gui_handles['playing'].value = True
            
        except Exception as e:
            print(f"Error loading result: {e}")
            import traceback
            traceback.print_exc()
            is_loading = False
            gui_handles['timestep'].disabled = False  # Re-enable even on error
    
    @gui_handles['dropdown'].on_update
    def _(_) -> None:
        nonlocal needs_update
        needs_update = True
    
    # Initial load
    load_new_result()
    
    # Main loop
    while True:
        if needs_update:
            load_new_result()
        
        if not is_loading and gui_handles['playing'].value:
            gui_handles['timestep'].value = (gui_handles['timestep'].value + 1) % num_frames
        
        time.sleep(1.0 / gui_handles['framerate'].value)


if __name__ == "__main__":
    tyro.cli(main)