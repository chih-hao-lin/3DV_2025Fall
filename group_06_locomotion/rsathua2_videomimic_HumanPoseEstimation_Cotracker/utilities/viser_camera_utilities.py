# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import numpy as np
from typing import Optional, Tuple, Union
import viser
import time

def apply_ema(data: np.ndarray, alpha: float) -> np.ndarray:
    """Applies Exponential Moving Average smoothing to a sequence of data points."""
    if not (0 < alpha <= 1.0):
        raise ValueError("EMA alpha must be between 0 and 1")
    if data.ndim == 1:
        data = data[:, None] # Ensure 2D for iteration
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]
    for i in range(1, len(data)):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i-1]
    return smoothed_data

def setup_camera_follow(
    server: viser.ViserServer,
    slider: viser.GuiSliderHandle,
    target_positions: np.ndarray,  # Shape (T,3) - points to look at
    camera_positions: Optional[np.ndarray] = None,  # Shape (T,3) - optional camera positions
    camera_distance: float = 2.0,  # Distance from target (used if camera_positions is None)
    camera_height: float = 1.0,    # Height offset from target (used if camera_positions is None)
    camera_angle: float = -30.0,   # Angle in degrees (used if camera_positions is None)
    up_direction: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    fov: float = 45.0,  # Field of view in degrees
    target_ema_alpha: float = 0.01, # Smoothing factor for target position (0 < alpha <= 1)
    camera_ema_alpha: float = 0.01, # Smoothing factor for camera position (0 < alpha <= 1, only used if camera_positions is not None)
) -> Tuple[callable, callable]:
    """Sets up camera to follow target positions over time with optional EMA smoothing.
    
    Args:
        server: ViserServer instance
        slider: The slider controlling the current timestep.
        target_positions: Array of shape (T,3) containing points to look at.
        camera_positions: Optional array of shape (T,3) containing camera positions.
                        If provided, camera_distance, camera_height, and camera_angle are ignored.
        camera_distance: Distance from camera to target (used if camera_positions is None).
        camera_height: Height offset of camera from target (used if camera_positions is None).
        camera_angle: Camera angle in degrees (used if camera_positions is None).
        up_direction: Up direction for camera orientation.
        fov: Vertical field of view in degrees.
        target_ema_alpha: Smoothing factor for target look-at positions. Lower values mean more smoothing.
        camera_ema_alpha: Smoothing factor for explicit camera positions. Lower values mean more smoothing. Only used if camera_positions is not None.

    Returns:
        A tuple containing (stop_camera_follow_func, resume_camera_follow_func).
    """
    
    # Apply EMA smoothing to target positions
    smoothed_target_positions = apply_ema(target_positions, target_ema_alpha)

    if camera_positions is not None:
        assert camera_positions.shape == target_positions.shape, \
            f"Camera positions shape {camera_positions.shape} must match target positions shape {target_positions.shape}"
        
        # Apply EMA smoothing to camera positions
        smoothed_camera_positions = apply_ema(camera_positions, camera_ema_alpha)

        def update_camera_for_target(client: viser.ClientHandle, t: int):
            client.camera.position = smoothed_camera_positions[t]
            client.camera.look_at = smoothed_target_positions[t]
            client.camera.up_direction = up_direction
            client.camera.fov = np.radians(fov)
    else:
        # Convert angle to radians for automatic camera positioning
        angle_rad = np.radians(camera_angle)
        
        def update_camera_for_target(client: viser.ClientHandle, t: int):
            # Use smoothed target position for calculating relative camera position
            target_pos = smoothed_target_positions[t]
            # Calculate camera position relative to the *smoothed* target
            cam_offset = np.array([
                -camera_distance * np.cos(angle_rad), # x - Adjusted for standard coordinate systems if needed
                camera_height,                        # y or z depending on up_direction
                -camera_distance * np.sin(angle_rad)  # z or y depending on up_direction
            ])
            
            # Adjust offset based on up_direction (simple handling for common cases)
            # This might need refinement for arbitrary up directions
            if tuple(up_direction) == (0.0, 1.0, 0.0): # Y-up
                final_cam_offset = np.array([cam_offset[0], cam_offset[1], cam_offset[2]])
            elif tuple(up_direction) == (0.0, 0.0, 1.0): # Z-up
                 final_cam_offset = np.array([cam_offset[0], cam_offset[2], cam_offset[1]]) # Swap y and z in offset
            else:
                # Fallback for other up directions - might not be perfectly aligned
                # A more robust solution involves calculating rotation based on up_direction
                print(f"Warning: Camera offset calculation might be approximate for up_direction {up_direction}")
                final_cam_offset = cam_offset 

            # Set camera position and look_at
            client.camera.position = target_pos + final_cam_offset
            client.camera.look_at = target_pos
            client.camera.up_direction = up_direction
            client.camera.fov = np.radians(fov)

    # Store the original callback to allow restoring it later
    original_callback = None

    def stop_camera_follow():
        """Stops the camera from following the target by removing the slider callback."""
        nonlocal original_callback
        if original_callback is not None:
            # Check if the callback exists before trying to remove it
            slider.remove_update_callback(original_callback)
            original_callback = None
            print("Camera follow stopped.")
        else:
            print("Camera follow already stopped.")

    def resume_camera_follow():
        """Resumes camera following by restoring the slider callback."""
        nonlocal original_callback
        if original_callback is None:
            @slider.on_update
            def callback(_):
                # Ensure slider value is within bounds
                t = max(0, min(slider.value, len(smoothed_target_positions) - 1))
                for client in server.get_clients().values():
                    update_camera_for_target(client, t)
            original_callback = callback
            print("Camera follow resumed.")
        else:
             print("Camera follow already running.")

    # Start camera following by default
    resume_camera_follow()

    return stop_camera_follow, resume_camera_follow
