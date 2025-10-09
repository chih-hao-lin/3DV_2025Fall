# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Optimization Results Visualization Tool (Cleaned Version)

This module provides an interactive visualization system for viewing optimization results
from the VideoMimic Real-to-Sim pipeline, including 3D point clouds, human meshes, camera poses,
and contact information with improved type safety and user interface.

Key Features:
- Interactive temporal visualization with playback controls
- Camera frustum visualization with adjustable scale
- Point cloud filtering with confidence and gradient thresholds
- Human mesh and skeleton visualization with contact detection
- Distance measurement tools
- Real-time GUI controls for all parameters
"""

import os
import os.path as osp
import pickle
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

import numpy as np
import cv2
import torch
import h5py
import trimesh
import smplx
import tyro
import viser
from scipy.spatial.transform import Rotation as R


# =============================================================================
# Constants and Configuration
# =============================================================================

# Color palette for visualization
COLORS_PATH = osp.join(osp.dirname(__file__), 'colors.txt')
if osp.exists(COLORS_PATH):
    with open(COLORS_PATH, 'r') as f:
        COLORS = np.array([list(map(int, line.strip().split())) for line in f])
else:
    # Default color palette if colors.txt doesn't exist
    COLORS = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [255, 0, 255], [0, 255, 255], [128, 0, 128], [255, 165, 0]
    ])


# =============================================================================
# Utility Functions
# =============================================================================

def get_color(idx: Union[int, str]) -> np.ndarray:
    """
    Get color from palette by index.
    
    Args:
        idx: Index for color selection (int or string)
        
    Returns:
        RGB color array of shape (3,)
    """
    if isinstance(idx, str):
        idx = int(idx)
    return COLORS[idx % len(COLORS)]


def load_dict_from_hdf5(h5file: h5py.File, path: str = "/") -> Dict[str, Any]:
    """
    Recursively load a nested dictionary from an HDF5 file.
    
    Args:
        h5file: Open h5py.File object
        path: Current path in the HDF5 file
    
    Returns:
        Nested dictionary with the data
    """
    result = {}
    for key in h5file[path].keys():
        key_path = f"{path}{key}"
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path + "/")
        else:
            result[key] = h5file[key_path][:]
    
    # Load attributes (scalars stored as attributes)
    for attr_key, attr_value in h5file.attrs.items():
        if attr_key.startswith(path):
            result[attr_key[len(path):]] = attr_value

    return result



def apply_point_filters(
    points: np.ndarray,
    colors: np.ndarray, 
    confidence: np.ndarray,
    depth_map: np.ndarray,
    conf_threshold: float = 0.0,
    gradient_threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply confidence and gradient filters to point clouds.
    
    Args:
        points: Point cloud coordinates (N, 3)
        colors: Point cloud colors (N, 3) 
        confidence: Confidence values (H, W) or (N,)
        depth_map: Depth map for gradient computation (H, W)
        conf_threshold: Minimum confidence threshold
        gradient_threshold: Maximum gradient magnitude threshold
        
    Returns:
        Tuple of (filtered_points, filtered_colors, mask) where mask is shape (N,)
    """
    # Confidence mask
    if confidence.ndim == 2:
        conf_mask = confidence.flatten() >= conf_threshold
    else:
        conf_mask = confidence >= conf_threshold
    
    # Gradient mask  
    dy, dx = np.gradient(depth_map)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    gradient_mask = gradient_magnitude.flatten() < gradient_threshold
    
    # Combine filters
    final_mask = conf_mask & gradient_mask
    
    return points[final_mask], colors[final_mask], final_mask


# -----------------------------------------------------------------------------
# Additional Camera Utilities
# -----------------------------------------------------------------------------
def get_vfov_and_aspect(intrinsics, img_height, img_width):
    """
    Calculate vertical field of view and aspect ratio from camera intrinsics.
    
    Args:
        intrinsics: Camera intrinsics matrix (3x3)
        img_height: Image height
        img_width: Image width
    
    Returns:
        vfov_rad: Vertical field of view in radians
        aspect: Aspect ratio (width/height)
    """
    if intrinsics is not None and intrinsics.shape == (3, 3):
        fy = intrinsics[1, 1]
        vfov_rad = 2 * np.arctan(img_height / (2 * fy))
    else:
        # Default vertical FOV if no intrinsics
        vfov_rad = np.radians(60)  # 60 degrees default
    
    aspect = img_width / img_height
    return vfov_rad, aspect


# =============================================================================
# Main Visualization Class
# =============================================================================

class OptimizationResultsVisualizer:
    """
    Interactive visualization system for optimization results.
    
    This class manages the 3D visualization of point clouds, human meshes,
    camera poses, and provides interactive controls for temporal playback
    and parameter adjustment.
    """
    
    def __init__(
        self,
        world_env: Dict[str, Any],
        world_scale_factor: float = 1.0,
        init_gradient_thr: float = 0.05,
        bg_pc_downsample_factor: int = 4,
        camera_frustum_scale: float = 0.1,
        apply_rot_180: bool = True,
        server: Optional[viser.ViserServer] = None
    ):
        """
        Initialize the visualizer.
        
        Args:
            world_env: Dictionary containing world environment data
            world_scale_factor: Scale factor for world coordinates  
            init_gradient_thr: Initial gradient threshold for point filtering
            bg_pc_downsample_factor: Downsampling factor for background points
            camera_frustum_scale: Scale factor for camera frustum visualization
            server: Optional viser server instance
        """
        self.world_env = world_env
        self.world_scale_factor = world_scale_factor
        self.bg_pc_downsample_factor = bg_pc_downsample_factor
        self.camera_frustum_scale = camera_frustum_scale
        
        # Apply world scaling to environment data
        self._scale_world_environment()
        
        # Initialize server
        if server is None:
            self.server = viser.ViserServer(port=8090)
        else:
            self.server = server
            
        # Configure scene
        self.server.scene.world_axes.visible = True
        self.server.scene.world_axes.axes_length = 0.25
        self.server.scene.set_up_direction("+y")
        
        # Rotation matrix for coordinate system alignment (180° around x-axis)
        if apply_rot_180:
            self.rot_180 = np.eye(3)
            self.rot_180[1, 1] = -1
            self.rot_180[2, 2] = -1
        else:
            self.rot_180 = np.eye(3)
        
        # Storage for visualization handles
        self.frame_nodes: List[viser.FrameHandle] = []
        self.cam_handles: List[viser.FrameHandle] = []
        self.cam_frustum_handles: List[viser.LineSegmentsHandle] = []
        self.smplx_handles: List[viser.MeshHandle] = []
        self.smpl_joints_handles: List[viser.PointCloudHandle] = []
        self.vitpose_keypoints_handles: List[viser.PointCloudHandle] = []
        self.fg_points_handles: List[viser.PointCloudHandle] = []
        self.bg_points_handle: Optional[viser.PointCloudHandle] = None
        
        # Initialize GUI elements
        self.timesteps = len(self.world_env.keys())
        self._setup_gui(init_gradient_thr)
        
    def _scale_world_environment(self) -> None:
        """Scale world environment coordinates by the scale factor."""
        for img_name in self.world_env.keys():
            self.world_env[img_name]['pts3d'] *= self.world_scale_factor
            self.world_env[img_name]['cam2world'][:3, 3] *= self.world_scale_factor
    
    def _setup_gui(self, init_gradient_thr: float) -> None:
        """Setup GUI controls for the visualization."""
        # Playback controls
        with self.server.gui.add_folder("Playback"):
            self.gui_timestep = self.server.gui.add_slider(
                "Timestep", min=0, max=self.timesteps - 1, step=1, 
                initial_value=0, disabled=True
            )
            self.gui_next_frame = self.server.gui.add_button("Next Frame", disabled=True)
            self.gui_prev_frame = self.server.gui.add_button("Prev Frame", disabled=True)
            self.gui_playing = self.server.gui.add_checkbox("Playing", True)
            self.gui_framerate = self.server.gui.add_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=30
            )
            self.gui_framerate_options = self.server.gui.add_button_group(
                "FPS options", ("10", "20", "30", "60")
            )
        
        # Filtering controls
        with self.server.gui.add_folder("Point Filtering"):
            self.gui_gradient_threshold = self.server.gui.add_number(
                "Gradient Threshold", initial_value=init_gradient_thr, 
                min=0.0, max=0.5, step=0.002
            )
        
        # NEW: Point size controls
        with self.server.gui.add_folder("Visualization"):
            self.gui_fg_point_size = self.server.gui.add_slider(
                "Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.01
            )
            self.gui_bg_point_size = self.server.gui.add_slider(
                "Background Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.005
            )
        
        # Visibility controls
        with self.server.gui.add_folder("Visibility"):
            self.gui_show_fg_points = self.server.gui.add_checkbox("Show FG Points", True)
            self.gui_show_bg_points = self.server.gui.add_checkbox("Show BG Points", True)
            self.gui_show_cameras = self.server.gui.add_checkbox("Show Cam Axes", True)
            self.gui_show_frustums = self.server.gui.add_checkbox("Show Cam Frustums", True)
            self.gui_show_meshes = self.server.gui.add_checkbox("Show SMPL Meshes", True)
            self.gui_show_joints = self.server.gui.add_checkbox("Show SMPL Joints", True)
        
        # Camera frustum controls
        with self.server.gui.add_folder("Camera Frustum"):
            self.gui_frustum_scale = self.server.gui.add_number(
                "Frustum Scale", initial_value=self.camera_frustum_scale,
                min=0.01, max=1.0, step=0.01
            )
        
        # Distance measurement tool
        self._setup_distance_tool()
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_distance_tool(self) -> None:
        """Setup distance measurement tool."""
        self.control0 = self.server.scene.add_transform_controls(
            "/controls/0", position=(0, 0, 0), scale=0.3, visible=False
        )
        self.control1 = self.server.scene.add_transform_controls(
            "/controls/1", position=(1.5, 0, 0), scale=0.3, visible=False
        )
        
        self.gui_show_controls = self.server.gui.add_checkbox("Show Distance Tool", False)
        self.distance_text = self.server.gui.add_text("Distance", initial_value="0.00m")
    
    def _setup_callbacks(self) -> None:
        """Setup all GUI callbacks."""
        # Playback callbacks
        @self.gui_next_frame.on_click
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % self.timesteps

        @self.gui_prev_frame.on_click  
        def _(_) -> None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % self.timesteps

        @self.gui_playing.on_update
        def _(_) -> None:
            self.gui_timestep.disabled = self.gui_playing.value
            self.gui_next_frame.disabled = self.gui_playing.value
            self.gui_prev_frame.disabled = self.gui_playing.value

        @self.gui_framerate_options.on_click
        def _(_) -> None:
            self.gui_framerate.value = int(self.gui_framerate_options.value)

        @self.gui_timestep.on_update
        def _(_) -> None:
            self._update_frame_visibility()
        
        # Filtering callbacks
        self.gui_gradient_threshold.on_update(lambda _: self._update_point_filtering())
        
        # NEW: Point size callbacks
        self.gui_fg_point_size.on_update(lambda _: self._update_fg_point_size())
        self.gui_bg_point_size.on_update(lambda _: self._update_bg_point_size())
        
        # Visibility callbacks
        self.gui_show_fg_points.on_update(lambda _: self._update_fg_visibility())
        self.gui_show_bg_points.on_update(lambda _: self._update_bg_visibility())
        self.gui_show_cameras.on_update(lambda _: self._update_camera_visibility())
        self.gui_show_frustums.on_update(lambda _: self._update_frustum_visibility())
        self.gui_show_meshes.on_update(lambda _: self._update_mesh_visibility())
        self.gui_show_joints.on_update(lambda _: self._update_joints_visibility())
        
        # Camera frustum callbacks
        self.gui_frustum_scale.on_update(lambda _: self._update_frustum_scale())
        
        # Distance tool callbacks
        @self.gui_show_controls.on_update
        def _(_) -> None:
            self.control0.visible = self.gui_show_controls.value
            self.control1.visible = self.gui_show_controls.value
            
        self.control0.on_update(lambda _: self._update_distance())
        self.control1.on_update(lambda _: self._update_distance())
    
    def create_visualization(
        self,
        vitpose_keypoints_3d_in_world: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        smpl_joints_3d_in_world: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        smplx_vertices_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        smplx_faces: Optional[np.ndarray] = None,
        contact_estimation: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    ) -> None:
        """
        Create the main visualization with all components.
        
        Args:
            vitpose_keypoints_3d_in_world: Dict mapping frame->person_id->joints (J, 4)
            smpl_joints_3d_in_world: Dict mapping frame->person_id->joints (J, 4)
            smplx_vertices_dict: Dict mapping frame->person_id->vertices (V, 3)  
            smplx_faces: SMPL-X face indices (F, 3)
            contact_estimation: Dict mapping frame->person_id->contact_info
        """
        bg_points_list = []
        bg_colors_list = []
        
        for t, frame_name in enumerate(self.world_env.keys()):
            # Add frame node
            self.frame_nodes.append(
                self.server.scene.add_frame(f"/t{t}", show_axes=False)
            )
            
            # Process point cloud data
            pts3d = self.world_env[frame_name]['pts3d']
            if pts3d.ndim == 3:
                pts3d = pts3d.reshape(-1, 3)
            
            points = pts3d.copy()
            colors = self.world_env[frame_name]['rgbimg'].reshape(-1, 3)
            
            # Apply dynamic masking if available
            if self.world_env[frame_name].get('dynamic_msk', None) is not None:
                fg_points, fg_colors, bg_points, bg_colors = self._process_dynamic_masking(
                    frame_name, points, colors
                )
                bg_points_list.append(bg_points)
                bg_colors_list.append(bg_colors)
            else:
                fg_points, fg_colors = points, colors
            
            # Create foreground point cloud
            fg_points = fg_points @ self.rot_180
            self.fg_points_handles.append(
                self.server.scene.add_point_cloud(
                    f"/t{t}/pts3d", points=fg_points, colors=fg_colors,
                    point_size=self.gui_fg_point_size.value, visible=True
                )
            )
            
            # Create camera visualization
            self._create_camera_visualization(t, frame_name)
            
            # Create human mesh visualization
            if smplx_vertices_dict is not None:
                self._create_mesh_visualization(
                    t, frame_name, smplx_vertices_dict, smplx_faces, contact_estimation
                )
            
            # Create joint visualization  
            if smpl_joints_3d_in_world is not None:
                self._create_joint_visualization(t, frame_name, smpl_joints_3d_in_world)

            # Create vitpose keypoints (lifted to world) visualization
            if vitpose_keypoints_3d_in_world is not None:
                self._create_vitpose_keypoints_visualization(t, frame_name, vitpose_keypoints_3d_in_world)
        # Create background point cloud
        if bg_points_list:
            self._create_background_pointcloud(bg_points_list, bg_colors_list)
        
        # Set initial frame visibility
        self.prev_timestep = 0
        self._update_frame_visibility()
    
    def _process_dynamic_masking(
        self, frame_name: str, points: np.ndarray, colors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process dynamic masking to separate foreground and background."""
        # Apply initial filters
        conf_mask = self.world_env[frame_name]['conf'].flatten() >= 0
        
        depths = self.world_env[frame_name]['depths']
        dy, dx = np.gradient(depths)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        gradient_mask = gradient_magnitude.flatten() < self.gui_gradient_threshold.value
        
        # Apply dynamic mask with dilation
        dynamic_msk = self.world_env[frame_name]['dynamic_msk'].astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dynamic_msk = cv2.dilate(dynamic_msk, kernel, iterations=1).flatten() > 0
        
        # Separate foreground and background
        fg_mask = dynamic_msk & conf_mask & gradient_mask
        bg_mask = ~dynamic_msk & conf_mask & gradient_mask
        
        fg_points = points[fg_mask]
        fg_colors = colors[fg_mask] 
        bg_points = points[bg_mask][::self.bg_pc_downsample_factor]
        bg_colors = colors[bg_mask][::self.bg_pc_downsample_factor]
        
        return fg_points, fg_colors, bg_points, bg_colors
    
    def _create_camera_visualization(self, t: int, frame_name: str) -> None:
        """Create camera pose and frustum visualization."""
        camera = self.world_env[frame_name]['cam2world'].copy()
        camera[:3, :3] = self.rot_180 @ camera[:3, :3]
        camera[:3, 3] = camera[:3, 3] @ self.rot_180
        
        # Convert rotation matrix to quaternion
        quat = R.from_matrix(camera[:3, :3]).as_quat()
        quat = np.concatenate([quat[3:], quat[:3]])  # xyzw to wxyz
        trans = camera[:3, 3]
        
        # Add camera frame
        cam_handle = self.server.scene.add_frame(
            f"/t{t}/cam", wxyz=quat, position=trans,
            show_axes=True, axes_length=0.15, axes_radius=0.02
        )
        self.cam_handles.append(cam_handle)
        
        # ------------------------------------------------------------------
        # Camera frustum with embedded RGB image
        # ------------------------------------------------------------------
        intrinsics = self.world_env[frame_name].get('K', None)
        rgb_image = self.world_env[frame_name]['rgbimg']
        img_h, img_w = rgb_image.shape[:2]

        vfov_rad, aspect = get_vfov_and_aspect(intrinsics, img_h, img_w)

        try:
            frustum_handle = self.server.scene.add_camera_frustum(
                f"/t{t}/frustum",
                fov=vfov_rad,
                aspect=aspect,
                scale=self.gui_frustum_scale.value,
                color=(255, 255, 255),
                wxyz=quat,
                position=trans,
                image=rgb_image,
            )
            self.cam_frustum_handles.append(frustum_handle)
        except Exception as e:
            print(f"Warning: Could not create camera frustum for frame {t}: {e}")
            self.cam_frustum_handles.append(None)
    
    def _create_mesh_visualization(
        self, 
        t: int, 
        frame_name: str,
        smplx_vertices_dict: Dict[str, Dict[str, np.ndarray]],
        smplx_faces: Optional[np.ndarray],
        contact_estimation: Optional[Dict[str, Dict[str, Dict[str, Any]]]]
    ) -> None:
        """Create human mesh visualization with contact information."""
        if frame_name not in smplx_vertices_dict:
            return
            
        for person_id in smplx_vertices_dict[frame_name].keys():
            vertices = smplx_vertices_dict[frame_name][person_id] * self.world_scale_factor
            vertices = vertices @ self.rot_180
            
            # Handle contact visualization if available
            if (contact_estimation is not None and 
                frame_name in contact_estimation and 
                int(person_id) in contact_estimation[frame_name]):
                
                mesh = self._create_contact_mesh(
                    vertices, smplx_faces, contact_estimation[frame_name][int(person_id)], person_id
                )
                mesh_handle = self.server.scene.add_mesh_trimesh(
                    f"/t{t}/smplx_mesh_{person_id}", mesh=mesh
                )
            else:
                mesh_handle = self.server.scene.add_mesh_simple(
                    f"/t{t}/smplx_mesh_{person_id}", vertices=vertices, faces=smplx_faces,
                    flat_shading=False, wireframe=False, color=get_color(int(person_id))
                )
            
            self.smplx_handles.append(mesh_handle)
    
    def _create_contact_mesh(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray,
        contact_info: Dict[str, Any], 
        person_id: int
    ) -> trimesh.Trimesh:
        """Create mesh with contact information visualization."""
        vertices_color = np.full_like(vertices, get_color(person_id))
        
        # Highlight contact areas in green
        if contact_info.get('left_foot_contact', False):
            left_foot_ids = contact_info['left_foot_vert_ids']
            vertices_color[left_foot_ids] = np.array([0, 255, 0])
            
        if contact_info.get('right_foot_contact', False):
            right_foot_ids = contact_info['right_foot_vert_ids'] 
            vertices_color[right_foot_ids] = np.array([0, 255, 0])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.vertex_colors = vertices_color.astype(np.uint8)
        return mesh
    
    def _create_joint_visualization(
        self, 
        t: int, 
        frame_name: str,
        smpl_joints_3d_in_world: Dict[str, Dict[str, np.ndarray]]
    ) -> None:
        """Create joint visualization with confidence filtering."""
        if frame_name not in smpl_joints_3d_in_world:
            return
            
        for person_id in smpl_joints_3d_in_world[frame_name].keys():
            joints3d = smpl_joints_3d_in_world[frame_name][person_id]  # (J, 3)
            
            if joints3d.shape[0] > 0:
                joints3d = joints3d @ self.rot_180
                
                joint_handle = self.server.scene.add_point_cloud(
                    f"/t{t}/joints3d_{person_id}",
                    points=joints3d,
                    colors=np.full((joints3d.shape[0], 3), [255, 0, 0]),
                    point_size=0.02, point_shape='circle'
                )
                
                self.smpl_joints_handles.append(joint_handle)
    
    def _create_vitpose_keypoints_visualization(
        self,
        t: int,
        frame_name: str,
        vitpose_keypoints_3d_in_world: Dict[str, Dict[str, np.ndarray]],
        threshold_for_keypoints: float = 0.3
    ) -> None:
        """Create vitpose keypoints visualization."""
        if frame_name not in vitpose_keypoints_3d_in_world:
            return
        
        for person_id in vitpose_keypoints_3d_in_world[frame_name].keys():
            keypoints_with_conf = vitpose_keypoints_3d_in_world[frame_name][person_id] # (J, 4)

            if keypoints_with_conf.shape[0] > 0:
                keypoints = keypoints_with_conf[:, :3] @ self.rot_180

                keypoints = keypoints[keypoints_with_conf[:, 3] >= threshold_for_keypoints].copy()
            
                if keypoints.shape[0] > 0:
                    vitpose_keypoints_handle = self.server.scene.add_point_cloud(
                        f"/t{t}/vitpose_keypoints_{person_id}",
                        points=keypoints * self.world_scale_factor,
                        colors=np.array([[125., 125, 0]] * keypoints.shape[0]),
                        point_size=0.03, point_shape='diamond'
                    )
                    self.vitpose_keypoints_handles.append(vitpose_keypoints_handle)
                else:
                    vitpose_keypoints_handle = self.server.scene.add_point_cloud(
                        f"/t{t}/vitpose_keypoints_{person_id}",
                        points=np.zeros((1, 3)) * self.world_scale_factor,
                        colors=np.array([[125., 125, 0]] * 1),
                        point_size=0.03, point_shape='diamond'
                    )
                    self.vitpose_keypoints_handles.append(vitpose_keypoints_handle)

    def _create_background_pointcloud(
        self, bg_points_list: List[np.ndarray], bg_colors_list: List[np.ndarray]
    ) -> None:
        """Create background point cloud from all frames."""
        if bg_points_list:
            bg_points = np.concatenate(bg_points_list)
            bg_colors = np.concatenate(bg_colors_list)
            bg_points = bg_points @ self.rot_180
            
            self.bg_points_handle = self.server.scene.add_point_cloud(
                "/bg/pts3d", points=bg_points, colors=bg_colors, point_size=self.gui_bg_point_size.value
            )
    
    def _update_frame_visibility(self) -> None:
        """Update frame visibility based on current timestep."""
        current_timestep = self.gui_timestep.value
        
        with self.server.atomic():
            # Hide all frames first
            for i, frame_node in enumerate(self.frame_nodes):
                frame_node.visible = (i == current_timestep)
        
        self.server.flush()
    
    def _update_point_filtering(self) -> None:
        """Update point cloud filtering based on GUI parameters."""
        print(f"Updated gradient threshold: {self.gui_gradient_threshold.value}")
        self._apply_point_filters()
    
    def _update_fg_visibility(self) -> None:
        """Toggle foreground points visibility."""
        for handle in self.fg_points_handles:
            handle.visible = self.gui_show_fg_points.value
    
    def _update_bg_visibility(self) -> None:
        """Toggle background points visibility."""
        if self.bg_points_handle:
            self.bg_points_handle.visible = self.gui_show_bg_points.value
    
    def _update_camera_visibility(self) -> None:
        """Toggle camera visibility."""
        for handle in self.cam_handles:
            handle.visible = self.gui_show_cameras.value
    
    def _update_frustum_visibility(self) -> None:
        """Toggle camera frustum visibility."""
        for handle in self.cam_frustum_handles:
            if handle is not None:
                handle.visible = self.gui_show_frustums.value
    
    def _update_mesh_visibility(self) -> None:
        """Toggle mesh visibility."""
        for handle in self.smplx_handles:
            handle.visible = self.gui_show_meshes.value
    
    def _update_joints_visibility(self) -> None:
        """Toggle joints visibility."""
        for handle in self.smpl_joints_handles:
            handle.visible = self.gui_show_joints.value
    
    def _update_frustum_scale(self) -> None:
        """Update camera frustum scale in real-time."""
        self.camera_frustum_scale = self.gui_frustum_scale.value
        for handle in self.cam_frustum_handles:
            if handle is not None:
                handle.scale = self.camera_frustum_scale
    
    def _update_distance(self) -> None:
        """Update distance measurement between control points."""
        distance = np.linalg.norm(self.control0.position - self.control1.position)
        self.distance_text.value = f"{distance:.2f}m"
        
        # Update line between control points
        self.server.scene.add_spline_catmull_rom(
            "/controls/line",
            np.stack([self.control0.position, self.control1.position], axis=0),
            color=(255, 0, 0)
        )
    
    def _update_fg_point_size(self) -> None:
        """Update foreground points size."""
        for handle in self.fg_points_handles:
            handle.point_size = self.gui_fg_point_size.value
    
    def _update_bg_point_size(self) -> None:
        """Update background points size."""
        if self.bg_points_handle is not None:
            self.bg_points_handle.point_size = self.gui_bg_point_size.value
    
    def _apply_point_filters(self) -> None:
        """Re-compute FG / BG masks using the current gradient threshold and refresh point clouds."""
        bg_points_list = []
        bg_colors_list = []

        for t, (frame_name, fg_handle) in enumerate(zip(self.world_env.keys(), self.fg_points_handles)):
            pts3d = self.world_env[frame_name]['pts3d']
            if pts3d.ndim == 3:
                pts3d = pts3d.reshape(-1, 3)
            points = pts3d.copy()
            colors = self.world_env[frame_name]['rgbimg'].reshape(-1, 3)

            # Confidence (if available)
            if 'conf' in self.world_env[frame_name]:
                conf_mask = self.world_env[frame_name]['conf'].flatten() >= 0
            else:
                conf_mask = np.ones(points.shape[0], dtype=bool)

            # Gradient computation (if depth available)
            if 'depths' in self.world_env[frame_name]:
                depths = self.world_env[frame_name]['depths']
                dy, dx = np.gradient(depths)
                gradient_mag = np.sqrt(dx ** 2 + dy ** 2)
                gradient_mask = gradient_mag.flatten() < self.gui_gradient_threshold.value
            else:
                # If depth unavailable, do not filter by gradient
                gradient_mask = np.ones(points.shape[0], dtype=bool)

            # Dynamic mask (foreground / background separation)
            if self.world_env[frame_name].get('dynamic_msk', None) is not None:
                dynamic_msk = self.world_env[frame_name]['dynamic_msk'].astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                dynamic_msk = cv2.dilate(dynamic_msk, kernel, iterations=1).flatten() > 0
                fg_mask = dynamic_msk & conf_mask & gradient_mask
                bg_mask = (~dynamic_msk) & conf_mask & gradient_mask
            else:
                fg_mask = conf_mask & gradient_mask
                bg_mask = (~fg_mask) & conf_mask  # everything else treated as BG

            # Update foreground handle
            fg_points = points[fg_mask] @ self.rot_180
            fg_colors = colors[fg_mask]
            fg_handle.points = fg_points
            fg_handle.colors = fg_colors
            fg_handle.point_size = self.gui_fg_point_size.value

            # Accumulate background
            bg_points = points[bg_mask][::self.bg_pc_downsample_factor]
            bg_colors = colors[bg_mask][::self.bg_pc_downsample_factor]
            if bg_points.shape[0] > 0:
                bg_points_list.append(bg_points @ self.rot_180)
                bg_colors_list.append(bg_colors)

        # Update / create background handle
        if bg_points_list:
            bg_points_all = np.concatenate(bg_points_list)
            bg_colors_all = np.concatenate(bg_colors_list)
            if self.bg_points_handle is None:
                self.bg_points_handle = self.server.scene.add_point_cloud(
                    "/bg/pts3d", points=bg_points_all, colors=bg_colors_all,
                    point_size=self.gui_bg_point_size.value
                )
            else:
                self.bg_points_handle.points = bg_points_all
                self.bg_points_handle.colors = bg_colors_all
                self.bg_points_handle.point_size = self.gui_bg_point_size.value

        self.server.flush()
    
    def run_playback_loop(self) -> None:
        """Run the main playback loop."""
        while True:
            if self.gui_playing.value:
                self.gui_timestep.value = (self.gui_timestep.value + 1) % self.timesteps
            
            time.sleep(1.0 / self.gui_framerate.value)


# =============================================================================
# Main Function
# =============================================================================

def main(
    world_env_path: str,
    bg_pc_downsample_factor: int = 4,
    data_dir_postfix: str = '',
    camera_frustum_scale: float = 0.1,
    apply_rot_180: bool = True
) -> None:
    """
    Main function to load data and create visualization.
    
    Args:
        world_env_path: Path to the HDF5 file containing optimization results
        gender: Gender for SMPL model ('male', 'female', 'neutral')
        bg_pc_downsample_factor: Downsampling factor for background points
        data_dir_postfix: Postfix for data directories
        camera_frustum_scale: Scale factor for camera frustum visualization
    """
    # Load optimization results
    with h5py.File(world_env_path, 'r') as f:
        world_env_and_human = load_dict_from_hdf5(f)

    world_env = world_env_and_human['our_pred_world_cameras_and_structure']
    human_params_in_world = world_env_and_human['our_pred_humans_smplx_params']
    person_frame_info_list = world_env_and_human['person_frame_info_list']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize SMPL models
    smpl_batch_layer_dict = {}
    for person_id in human_params_in_world.keys():
        num_frames = len(human_params_in_world[person_id]['body_pose'])
        smpl_batch_layer_dict[person_id] = smplx.create(
            model_path='./assets/body_models', model_type='smpl', gender='male',
            num_betas=10, batch_size=num_frames
        ).to(device)

    # Generate human meshes and joints
    human_verts_dict = {}
    human_joints_dict = {}
    
    for person_id in human_params_in_world.keys():
        num_frames = human_params_in_world[person_id]['body_pose'].shape[0]
        smpl_betas = torch.from_numpy(human_params_in_world[person_id]['betas']).float().to(device)
        if smpl_betas.ndim == 1:
            smpl_betas = smpl_betas.repeat(num_frames, 1)
    
        smpl_output = smpl_batch_layer_dict[person_id](
            body_pose=torch.from_numpy(human_params_in_world[person_id]['body_pose']).float().to(device),
            betas=smpl_betas,
            global_orient=torch.from_numpy(human_params_in_world[person_id]['global_orient']).float().to(device),
            pose2rot=False
        )
        
        smpl_joints = smpl_output['joints']
        smpl_root_joint = smpl_joints[:, 0:1, :]  # (T, 1, 3)
        root_transl = torch.from_numpy(human_params_in_world[person_id]['root_transl']).float().to(device)
        
        smpl_verts = smpl_output['vertices'] - smpl_root_joint + root_transl
        smpl_joints = smpl_joints - smpl_root_joint + root_transl

        human_verts_dict[person_id] = smpl_verts.detach().cpu().numpy()
        human_joints_dict[person_id] = smpl_joints.detach().cpu().numpy()
    
    # Transform to frame-based dictionaries
    human_verts_by_frame = defaultdict(dict)
    human_joints_by_frame = defaultdict(dict)
    
    for person_id in human_params_in_world.keys():
        person_frame_info_list_per_person = person_frame_info_list[person_id].astype(str)
        for idx, frame_name in enumerate(person_frame_info_list_per_person):
            frame_name = frame_name.item()
            human_verts_by_frame[frame_name][person_id] = human_verts_dict[person_id][idx]
            human_joints_by_frame[frame_name][person_id] = human_joints_dict[person_id][idx]

    # Load contact estimation if available
    # If world_env_path is a gravity calibrated file, use the directory name of that file
    if 'gravity_calibrated' in world_env_path:
        video_name = osp.basename(osp.dirname(world_env_path)).split('_cam')[0].split('_results_')[1]
        contact_dir = osp.join(
            osp.dirname(world_env_path), '..', f'input_contacts{data_dir_postfix}', 
            video_name, 'cam01'
        )
    else:
        video_name = osp.basename(world_env_path).split('_cam')[0].split('_results_')[1]
        contact_dir = osp.join(
            osp.dirname(world_env_path), '..', f'input_contacts{data_dir_postfix}', 
            video_name, 'cam01'
        )
    
    contact_estimation = None
    if osp.exists(contact_dir):
        contact_estimation = {}
        for frame_name in world_env.keys():
            contact_file = osp.join(contact_dir, f'{frame_name}.pkl')
            if osp.exists(contact_file):
                with open(contact_file, 'rb') as f:
                    contact_estimation[frame_name] = pickle.load(f)

        if contact_estimation:
            # Load SMPL vertex segmentation for contact visualization
            smpl_vert_seg_path = "./assets/body_models/smpl/smpl_vert_segmentation.json"
            if osp.exists(smpl_vert_seg_path):
                with open(smpl_vert_seg_path, 'r') as f:
                    smpl_vert_seg = json.load(f)
                
                left_foot_vert_ids = np.array(smpl_vert_seg['leftFoot'], dtype=np.int32)
                right_foot_vert_ids = np.array(smpl_vert_seg['rightFoot'], dtype=np.int32)
                
                # Add vertex IDs to contact information
                for frame_name in contact_estimation.keys():
                    for person_id in human_params_in_world.keys():
                        if int(person_id) in contact_estimation[frame_name]:
                            contact_estimation[frame_name][int(person_id)]['left_foot_vert_ids'] = left_foot_vert_ids
                            contact_estimation[frame_name][int(person_id)]['right_foot_vert_ids'] = right_foot_vert_ids

    # Create and run visualization
    visualizer = OptimizationResultsVisualizer(
        world_env=world_env,
        world_scale_factor=1.0,
        bg_pc_downsample_factor=bg_pc_downsample_factor,
        camera_frustum_scale=camera_frustum_scale,
        apply_rot_180=apply_rot_180
    )
    
    visualizer.create_visualization(
        smpl_joints_3d_in_world=human_joints_by_frame,
        smplx_vertices_dict=human_verts_by_frame,
        smplx_faces=smpl_batch_layer_dict[list(human_params_in_world.keys())[0]].faces,
        contact_estimation=contact_estimation
    )
    
    # Run playback loop
    visualizer.run_playback_loop()

def show_points_and_keypoints(world_env, world_scale_factor=1., keypoints_3d_in_world=None, bg_pc_downsample_factor=4, camera_frustum_scale=0.1):
    # Create and run visualization
    visualizer = OptimizationResultsVisualizer(
        world_env=world_env,
        world_scale_factor=world_scale_factor,
        bg_pc_downsample_factor=bg_pc_downsample_factor,
        camera_frustum_scale=camera_frustum_scale
    )

    visualizer.gui_show_meshes.value = False
    visualizer.gui_show_meshes.disabled = True
    visualizer.gui_show_joints.value = False
    visualizer.gui_show_joints.disabled = True
    
    visualizer.create_visualization(
        vitpose_keypoints_3d_in_world=keypoints_3d_in_world,
    )
    
    # Run playback loop
    visualizer.run_playback_loop()

def show_points_and_keypoints_and_smpl(world_env, world_scale_factor=1., keypoints_3d_in_world=None, smpl_verts_in_world=None, smpl_layer_faces=None, bg_pc_downsample_factor=4, camera_frustum_scale=0.1):
    # Create and run visualization
    visualizer = OptimizationResultsVisualizer(
        world_env=world_env,
        world_scale_factor=world_scale_factor,
        bg_pc_downsample_factor=bg_pc_downsample_factor,
        camera_frustum_scale=camera_frustum_scale
    )

    visualizer.gui_show_joints.value = False
    visualizer.gui_show_joints.disabled = True

    visualizer.create_visualization(
        vitpose_keypoints_3d_in_world=keypoints_3d_in_world,
        smplx_vertices_dict=smpl_verts_in_world,
        smplx_faces=smpl_layer_faces,
    )
    


    # Run playback loop
    visualizer.run_playback_loop()

if __name__ == '__main__':
    tyro.cli(main)