# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import pickle
import numpy as np
import tyro
import viser
import time
import cv2
import h5py

from scipy.spatial.transform import Rotation as R


def load_dict_from_hdf5(h5file, path="/"):
    """
    Recursively load a nested dictionary from an HDF5 file.
    
    Args:
        h5file: An open h5py.File object.
        path: The current path in the HDF5 file.
    
    Returns:
        A nested dictionary with the data.
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

def get_scene_summary(world_env):
    """
    Generate a summary of the scene structure.
    
    Args:
        world_env: Dictionary containing world environment data
    
    Returns:
        summary: Dictionary with scene statistics
    """
    num_frames = len(world_env.keys())
    total_points = 0
    avg_confidence = 0
    
    for frame_name in world_env.keys():
        pts3d = world_env[frame_name]['pts3d']
        if pts3d.ndim == 3:
            pts3d = pts3d.reshape(-1, 3)
        
        mask = world_env[frame_name]['msk']
        valid_points = np.sum(mask.flatten())
        total_points += valid_points
        
        conf = world_env[frame_name]['conf']
        avg_confidence += np.mean(conf[mask])
    
    avg_confidence /= num_frames
    avg_points_per_frame = total_points / num_frames
    
    return {
        'num_frames': num_frames,
        'total_points': total_points,
        'avg_points_per_frame': int(avg_points_per_frame),
        'avg_confidence': avg_confidence
    }

def main(world_env_path: str, world_scale_factor: float = 5., conf_thr: float = 1.5, downsample_factor: int = 8):

    if 'pkl' in world_env_path:
        with open(world_env_path, 'rb') as f:
            world_env = pickle.load(f)
    else:
        # Extract data from world_env dictionary
        with h5py.File(world_env_path, 'r') as f:
            world_env = load_dict_from_hdf5(f)
    world_env = world_env['monst3r_ga_output']
    
    for img_name in world_env.keys():
        world_env[img_name]['pts3d'] *= world_scale_factor
        world_env[img_name]['cam2world'][:3, 3] *= world_scale_factor
        # get new mask
        conf = world_env[img_name]['conf']
        world_env[img_name]['msk'] = conf > conf_thr

    # set viser
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.world_axes.axes_length = 0.25
    server.scene.set_up_direction("+y")

    # get rotation matrix of 180 degrees around x axis
    rot_180 = np.eye(3)
    rot_180[1, 1] = -1
    rot_180[2, 2] = -1

    # Get scene summary
    scene_summary = get_scene_summary(world_env)
    
    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    
    # Scene summary panel
    with server.gui.add_folder("Scene Info"):
        server.gui.add_text("Frames", initial_value=str(scene_summary['num_frames']), disabled=True)
        server.gui.add_text("Total Points", initial_value=f"{scene_summary['total_points']:,}", disabled=True)
        server.gui.add_text("Avg Points/Frame", initial_value=f"{scene_summary['avg_points_per_frame']:,}", disabled=True)
        server.gui.add_text("Avg Confidence", initial_value=f"{scene_summary['avg_confidence']:.2f}", disabled=True)
    
    # Visualization controls
    with server.gui.add_folder("Visualization"):
        gui_point_size = server.gui.add_slider(
            "Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.005
        )
        gui_bg_point_size = server.gui.add_slider(
            "Background Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.005
        )
        gui_frustum_scale = server.gui.add_slider(
            "Frustum Scale", min=0.1, max=2.0, step=0.1, initial_value=0.15
        )
        gui_show_cameras = server.gui.add_checkbox("Show Camera Axes", True)
        gui_show_frustums = server.gui.add_checkbox("Show Camera Frustums", True)

    # world_env.pop('frame_0258')
    timesteps = len(world_env.keys())

    frame_nodes: list[viser.FrameHandle] = []
    cam_handles = []
    frustum_handles = []
    point_cloud_handles = []
    bg_points = []
    bg_colors = []
    bg_point_cloud_handle = None
    for t, frame_name in enumerate(world_env.keys()):
        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/t{t}", show_axes=False))

        # Visualize the pointcloud of environment
        pts3d = world_env[frame_name]['pts3d']
        if pts3d.ndim == 3:
            pts3d = pts3d.reshape(-1, 3)
        points = pts3d[world_env[frame_name]['msk'].flatten()]
        colors = world_env[frame_name]['rgbimg'][world_env[frame_name]['msk']].reshape(-1, 3)

        if world_env[frame_name].get('dynamic_msk', None) is not None:
            dynamic_msk = world_env[frame_name]['dynamic_msk'].astype(np.uint8)
            kernel_size = 4
            kernel = np.ones((kernel_size, kernel_size),np.uint8)
            dynamic_msk = cv2.dilate(dynamic_msk, kernel, iterations=1).flatten() > 0
            msk = world_env[frame_name]['msk'].flatten()
            fg_msk = dynamic_msk[msk]
            bg_msk = ~dynamic_msk[msk]
            bg_points.append(points[bg_msk][::downsample_factor])
            bg_colors.append(colors[bg_msk][::downsample_factor])
            points = points[fg_msk]
            colors = colors[fg_msk]

        points = points @ rot_180
        point_handle = server.scene.add_point_cloud(
            f"/t{t}/pts3d",
            points=points,
            colors=colors,
            point_size=gui_point_size.value,
        )
        point_cloud_handles.append(point_handle)

        # Visualize the camera
        camera = world_env[frame_name]['cam2world']
        camera[:3, :3] = rot_180 @ camera[:3, :3] 
        camera[:3, 3] = camera[:3, 3] @ rot_180
        
        # rotation matrix to quaternion
        quat = R.from_matrix(camera[:3, :3]).as_quat()
        # xyzw to wxyz
        quat = np.concatenate([quat[3:], quat[:3]])
        # translation vector
        trans = camera[:3, 3]

        # add camera
        cam_handle = server.scene.add_frame(
            f"/t{t}/cam",
            wxyz=quat,
            position=trans,
            show_axes=True,
            axes_length=0.15,
            axes_radius=0.02,
            origin_radius=0.02
        )
        cam_handles.append(cam_handle)
        
        # add camera frustum with RGB image
        intrinsics = world_env[frame_name].get('K', None)
        rgb_image = world_env[frame_name]['rgbimg']
        img_height, img_width = rgb_image.shape[:2]
        
        # Calculate vertical FOV and aspect ratio
        vfov_rad, aspect = get_vfov_and_aspect(intrinsics, img_height, img_width)
        
        # Add camera frustum with embedded image
        frustum_handle = server.scene.add_camera_frustum(
            f"/t{t}/frustum",
            fov=vfov_rad,
            aspect=aspect,
            scale=gui_frustum_scale.value,
            color=(255, 255, 255),
            wxyz=quat,
            position=trans,
            image=rgb_image,
        )
        frustum_handles.append([frustum_handle])
    
    if len(bg_points) > 0:
        bg_points = np.concatenate(bg_points)
        bg_colors = np.concatenate(bg_colors)
        bg_points = bg_points @ rot_180
        bg_point_cloud_handle = server.scene.add_point_cloud(
            f"/bg/pts3d",
            points=bg_points,
            colors=bg_colors,
            point_size=gui_bg_point_size.value,
        )

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=timesteps - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_play_pause = server.gui.add_button("⏸️ Pause", disabled=False)
        gui_next_frame = server.gui.add_button("⏭️ Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("⏮️ Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Auto Play", True)
        gui_playing.disabled = True
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=20
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS Presets", ("5", "10", "20", "30", "60")
        )

    # Playback control functions
    @gui_play_pause.on_click
    def _(_) -> None:
        gui_playing.value = not gui_playing.value
        gui_play_pause.name = "▶️ Play" if not gui_playing.value else "⏸️ Pause"
    
    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % timesteps

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % timesteps

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value
        gui_play_pause.name = "⏸️ Pause" if gui_playing.value else "▶️ Play"

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)
    
    # Update point sizes
    @gui_point_size.on_update
    def _(_) -> None:
        current_timestep = gui_timestep.value
        if current_timestep < len(point_cloud_handles):
            point_cloud_handles[current_timestep].point_size = gui_point_size.value
    
    @gui_bg_point_size.on_update
    def _(_) -> None:
        if bg_point_cloud_handle is not None:
            bg_point_cloud_handle.point_size = gui_bg_point_size.value
    
    # Toggle camera visibility
    @gui_show_cameras.on_update
    def _(_) -> None:
        for cam_handle in cam_handles:
            cam_handle.visible = gui_show_cameras.value
    
    # Toggle frustum visibility
    @gui_show_frustums.on_update
    def _(_) -> None:
        for frustum_group in frustum_handles:
            for frustum_handle in frustum_group:
                frustum_handle.visible = gui_show_frustums.value
    
    # Update frustum scale
    @gui_frustum_scale.on_update
    def _(_) -> None:
        for frustum_group in frustum_handles:
            for frustum_handle in frustum_group:
                frustum_handle.scale = gui_frustum_scale.value

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!


    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % timesteps

        time.sleep(1.0 / gui_framerate.value)

if __name__ == '__main__':
    tyro.cli(main)
