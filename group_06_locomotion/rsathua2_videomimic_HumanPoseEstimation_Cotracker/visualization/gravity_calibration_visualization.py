# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import numpy as np
import os
import time
import viser
import tyro
import h5py
import torch
import smplx
from pathlib import Path
from typing import Dict, Any, List, Tuple


def get_person_color(person_idx: int) -> Tuple[int, int, int]:
    """Get color for a person based on their index."""
    colors = [
        (255, 215, 0),   # Gold
        (0, 255, 127),   # Spring Green
        (255, 105, 180), # Hot Pink
        (65, 105, 225),  # Royal Blue
        (255, 140, 0),   # Dark Orange
        (138, 43, 226),  # Blue Violet
        (50, 205, 50),   # Lime Green
        (220, 20, 60),   # Crimson
    ]
    return colors[person_idx % len(colors)]


def visualize_gravity_calibration(
    keypoints_output: Dict[str, Any],
    human_verts_in_world_dict: Dict[str, np.ndarray],
    smpl_batch_layer_dict: Dict[str, Any],
    person_id_list: List[str],
    num_frames: int
) -> None:
    """
    Visualize gravity calibration results using viser.
    
    Args:
        keypoints_output: Dictionary containing calibrated keypoints and world rotation
        human_verts_in_world_dict: Dictionary of human vertices in world coordinates
        smpl_batch_layer_dict: Dictionary of SMPL models
        person_id_list: List of person IDs for visualization
        num_frames: Number of frames in the sequence
    """
    print("Visualizing gravity calibration results...")
    
    # Start viser server
    server = viser.ViserServer(port=8081)
    server.scene.world_axes.visible = False
    
    # Add playback UI
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=15
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )
    
    # Frame step buttons
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames
    
    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames
    
    # Disable frame controls when playing
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value
    
    # Set framerate when clicking options
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)
    
    # Add visualization controls
    with server.gui.add_folder("Visualization"):
        gui_show_smpl_mesh = server.gui.add_checkbox("Show SMPL Mesh", True)
        gui_show_joints = server.gui.add_checkbox("Show SMPL Joints", True)
        gui_show_gravity_world_axes = server.gui.add_checkbox("Show Gravity World Axes", True)
        gui_show_prev_world_axes = server.gui.add_checkbox("Show Previous World Axes", True)
        gui_show_grid = server.gui.add_checkbox("Show Grid", True)
    
    # Add world coordinate axes
    world_rotation = keypoints_output['world_rotation']  # (3, 3)
    axis_length = 2.0
    
    # Original axes (before calibration) - red, green, blue
    gravity_world_x_handle = server.scene.add_spline_catmull_rom(
        "/world_axes/gravity_world/x",
        positions=np.array([[0, 0, 0], [axis_length, 0, 0]]),
        color=(255, 0, 0),
        line_width=5.0,
        visible=True
    )
    gravity_world_y_handle = server.scene.add_spline_catmull_rom(
        "/world_axes/gravity_world/y", 
        positions=np.array([[0, 0, 0], [0, axis_length, 0]]),
        color=(0, 255, 0),
        line_width=5.0,
        visible=True
    )
    gravity_world_z_handle = server.scene.add_spline_catmull_rom(
        "/world_axes/gravity_world/z",
        positions=np.array([[0, 0, 0], [0, 0, axis_length]]),
        color=(0, 0, 255),
        line_width=5.0,
        visible=True
    )
    
    # Calibrated axes (after rotation) - darker colors
    calibrated_x = world_rotation[:, 0] * axis_length  # (3,)
    calibrated_y = world_rotation[:, 1] * axis_length  # (3,)
    calibrated_z = world_rotation[:, 2] * axis_length  # (3,)
    
    prev_world_x_handle = server.scene.add_spline_catmull_rom(
        "/world_axes/prev_world/x",
        positions=np.array([[0, 0, 0], calibrated_x]),
        color=(128, 0, 0),
        line_width=3.0,
        visible=True
    )
    prev_world_y_handle = server.scene.add_spline_catmull_rom(
        "/world_axes/prev_world/y",
        positions=np.array([[0, 0, 0], calibrated_y]),
        color=(0, 128, 0),
        line_width=3.0,
        visible=True
    )
    prev_world_z_handle = server.scene.add_spline_catmull_rom(
        "/world_axes/prev_world/z",
        positions=np.array([[0, 0, 0], calibrated_z]),
        color=(0, 0, 128),
        line_width=3.0,
        visible=True
    )
    
    # Add SMPL mesh handles for each person
    smpl_mesh_handle_dict: Dict[str, List] = {pid: [] for pid in person_id_list}
    
    # Add joints visualization handles for each person
    joints_handle_dict: Dict[str, List[viser.PointCloudHandle]] = {pid: [] for pid in person_id_list}  # type: ignore
    
    # Get all joints data
    all_joints_data = keypoints_output['joints']  # Dict[person_id, (T, 45, 3)]
    
    # Compute ground plane (2D grid) height using minimum z across all joints frames and all people
    min_z_depth: float = float('inf')
    for pid in person_id_list:
        person_min_z = float(all_joints_data[pid][:, :, 2].min())
        min_z_depth = min(min_z_depth, person_min_z)

    # Add a ground grid on the XY-plane located at min_z_depth
    ground_grid_handle = server.scene.add_grid(
        name="/ground_grid",
        position=(0.0, 0.0, min_z_depth),
        visible=True,
    )
    
    def update_visualization(t: int) -> None:
        # Hide all existing joints and meshes for all people
        for pid in person_id_list:
            for joint_handle in joints_handle_dict[pid]:
                joint_handle.visible = False
            for mesh_handle in smpl_mesh_handle_dict[pid]:
                mesh_handle.visible = False

        # Update for each person
        for person_idx, pid in enumerate(person_id_list):
            # Get color for this person
            person_color = get_person_color(person_idx)
            
            # Update SMPL mesh
            if len(smpl_mesh_handle_dict[pid]) <= t:
                smpl_mesh_handle = server.scene.add_mesh_simple(
                    f"/smpl_mesh/{pid}/{t}",
                    vertices=human_verts_in_world_dict[pid][t],  # (6890, 3)
                    faces=smpl_batch_layer_dict[pid].faces,
                    flat_shading=False,
                    wireframe=False,
                    color=person_color,
                    visible=gui_show_smpl_mesh.value,
                )
                smpl_mesh_handle_dict[pid].append(smpl_mesh_handle)

            if gui_show_smpl_mesh.value:
                smpl_mesh_handle_dict[pid][t].visible = True
            
            # Update joints point cloud
            if len(joints_handle_dict[pid]) <= t:
                joints_pts = all_joints_data[pid][t]  # (J, 3)
                # Use a darker version of person color for joints
                joint_color = tuple(int(c * 0.7) for c in person_color)
                joints_handle = server.scene.add_point_cloud(
                    f"/joints/{pid}/{t}",
                    points=joints_pts,
                    colors=np.full((joints_pts.shape[0], 3), joint_color),
                    point_size=0.02,
                    point_shape='circle',
                    visible=gui_show_joints.value,
                )
                joints_handle_dict[pid].append(joints_handle)
            
            if gui_show_joints.value and t < len(joints_handle_dict[pid]):
                joints_handle_dict[pid][t].visible = True
    
    # Update visualization controls
    @gui_show_smpl_mesh.on_update
    def _(_) -> None:
        current_t = gui_timestep.value
        for pid in person_id_list:
            if current_t < len(smpl_mesh_handle_dict[pid]):
                smpl_mesh_handle_dict[pid][current_t].visible = gui_show_smpl_mesh.value
    
    @gui_show_gravity_world_axes.on_update
    def _(_) -> None:
        gravity_world_x_handle.visible = gui_show_gravity_world_axes.value
        gravity_world_y_handle.visible = gui_show_gravity_world_axes.value
        gravity_world_z_handle.visible = gui_show_gravity_world_axes.value
    
    @gui_show_prev_world_axes.on_update
    def _(_) -> None:
        prev_world_x_handle.visible = gui_show_prev_world_axes.value
        prev_world_y_handle.visible = gui_show_prev_world_axes.value
        prev_world_z_handle.visible = gui_show_prev_world_axes.value
    
    @gui_show_joints.on_update
    def _(_) -> None:
        current_t = gui_timestep.value
        for pid in person_id_list:
            if current_t < len(joints_handle_dict[pid]):
                joints_handle_dict[pid][current_t].visible = gui_show_joints.value
    
    @gui_show_grid.on_update
    def _(_) -> None:
        ground_grid_handle.visible = gui_show_grid.value
    
    # Update scene when timestep changes
    @gui_timestep.on_update
    def _(_) -> None:
        current_timestep = gui_timestep.value
        with server.atomic():
            update_visualization(current_timestep)
    
    # Initialize first frame
    update_visualization(0)
    
    # Main visualization loop
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        
        time.sleep(1.0 / gui_framerate.value)


def load_dict_from_hdf5(h5file: h5py.File, path: str = "/") -> Dict[str, Any]:
    """Load a nested dictionary from an HDF5 file."""
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


def main(
    calib_out_dir: str
) -> None:
    """
    Visualize gravity calibration results.
    
    Args:
        calib_out_dir: Path to gravity calibrated output directory
    """
    # Load keypoints data
    with h5py.File(os.path.join(calib_out_dir, 'gravity_calibrated_keypoints.h5'), 'r') as f:
        keypoints_output = load_dict_from_hdf5(f)
    
    # Load calibrated megahunter data
    calibrated_megahunter_path = os.path.join(calib_out_dir, 'gravity_calibrated_megahunter.h5')
    if 'h5' in calibrated_megahunter_path:
        with h5py.File(calibrated_megahunter_path, 'r') as f:
            world_env_and_human = load_dict_from_hdf5(f)
    else:
        import pickle
        with open(calibrated_megahunter_path, 'rb') as f:
            world_env_and_human = pickle.load(f)
    
    # Extract person info
    person_id_list = list(world_env_and_human['our_pred_humans_smplx_params'].keys())
    print(f"Found {len(person_id_list)} person(s) in the scene: {person_id_list}")
    
    # Get num_frames from first person (assuming all have same number of frames)
    first_person_id = person_id_list[0]
    num_frames = len(world_env_and_human['person_frame_info_list'][first_person_id])
    
    # Create SMPL batch layer dict and human verts for all people
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    smpl_batch_layer_dict = {}
    human_verts_in_world_dict = {}
    
    for person_id in person_id_list:
        # Create SMPL model for this person
        smpl_batch_layer_dict[person_id] = smplx.create(
            model_path='./assets/body_models',
            model_type='smpl',
            gender='male',  # Could be parameterized per person
            num_betas=10,
            batch_size=num_frames
        ).to(device)
        
        # Decode SMPL parameters
        smpl_output_batch = smpl_batch_layer_dict[person_id](
            body_pose=torch.from_numpy(world_env_and_human['our_pred_humans_smplx_params'][person_id]['body_pose']).float().to(device),
            betas=torch.from_numpy(world_env_and_human['our_pred_humans_smplx_params'][person_id]['betas']).float().to(device),
            global_orient=torch.from_numpy(world_env_and_human['our_pred_humans_smplx_params'][person_id]['global_orient']).float().to(device),
            pose2rot=False
        )
        smpl_joints = smpl_output_batch['joints']  # (T, 45, 3)
        smpl_root_joint = smpl_joints[:, 0:1, :]  # (T, 1, 3)
        smpl_verts = smpl_output_batch['vertices'] - smpl_root_joint + torch.from_numpy(world_env_and_human['our_pred_humans_smplx_params'][person_id]['root_transl']).to(device)  # (T, 6890, 3)
        
        human_verts_in_world_dict[person_id] = smpl_verts.detach().cpu().numpy()
    
    # Start visualization
    visualize_gravity_calibration(
        keypoints_output,
        human_verts_in_world_dict,
        smpl_batch_layer_dict,
        person_id_list,
        num_frames
    )


if __name__ == "__main__":
    tyro.cli(main)