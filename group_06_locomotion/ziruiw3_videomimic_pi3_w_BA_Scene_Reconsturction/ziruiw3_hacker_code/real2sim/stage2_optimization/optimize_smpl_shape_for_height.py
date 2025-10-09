# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import tyro
import viser
import json
import os
import os.path as osp
from pathlib import Path
from typing import List, Tuple  

import jax
import jaxlie
import jaxls
import jax_dataclasses as jdc
import jax.numpy as jnp
import numpy as onp
import time
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

# Import from modules with numeric prefixes using importlib
import importlib.util
import sys
import os

# Import SMPL JAX layer
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
smpl_jax_path = os.path.join(project_root, "utilities", "smpl_jax_layer.py")
spec = importlib.util.spec_from_file_location("smpl_jax_layer", smpl_jax_path)
smpl_jax_module = importlib.util.module_from_spec(spec)
# Add to sys.modules before executing to fix jax_dataclasses issue
sys.modules["smpl_jax_layer"] = smpl_jax_module
spec.loader.exec_module(smpl_jax_module)

# Import the classes
SmplModel = smpl_jax_module.SmplModel
# SmplShaped = smpl_jax_module.SmplShaped
# SmplShapedAndPosed = smpl_jax_module.SmplShapedAndPosed
# SmplMesh = smpl_jax_module.SmplMesh


# make_smpl_connection_matrix = retargeting_module.make_smpl_connection_matrix

smpl_joint_names = [
    # "pelvis",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]
left_smpl_joint_indices = jnp.array([
    idx for idx, name in enumerate(smpl_joint_names) if name.startswith("left_")
])
right_smpl_joint_indices = jnp.array([
    idx for idx, name in enumerate(smpl_joint_names) if name.startswith("right_")
])

g1_joint_names = (
    "pelvis_contour_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "head_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "logo_joint",
    "imu_joint",
    "left_palm_joint",
    "left_zero_joint",
    "left_one_joint",
    "left_two_joint",
    "left_three_joint",
    "left_four_joint",
    "left_five_joint",
    "left_six_joint",
    "right_palm_joint",
    "right_zero_joint",
    "right_one_joint",
    "right_two_joint",
    "right_three_joint",
    "right_four_joint",
    "right_five_joint",
    "right_six_joint",
)
h1_joint_names = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "imu_joint",
    "logo_joint",
    "d435_left_imager_joint",
    "d435_rgb_module_joint",
    "mid360_joint",
)

smpl_joint_retarget_indices_to_h1 = []
h1_joint_retarget_indices = []
for smpl_name, h1_name in [
    ("head", "mid360_joint"),
    ("left_elbow", "left_elbow_joint"),
    ("right_elbow", "right_elbow_joint"),
    ("left_shoulder", "left_shoulder_pitch_joint"),
    ("right_shoulder", "right_shoulder_pitch_joint"),
    # ("pelvis", "torso_joint"),
    # ("left_hip", "left_hip_yaw_joint"),
    # ("right_hip", "right_hip_yaw_joint"),
    ("left_hip", "left_hip_pitch_joint"),
    ("right_hip", "right_hip_pitch_joint"),
    ("left_knee", "left_knee_joint"),
    ("right_knee", "right_knee_joint"),
    ("left_ankle", "left_ankle_joint"),
    ("right_ankle", "right_ankle_joint"),

    # ("head", "head_joint"),
    # ("pelvis", "torso_joint"),
    # ("left_hip", "left_hip_pitch_joint"),
    # ("right_hip", "right_hip_pitch_joint"),
    # ("left_elbow", "left_elbow_pitch_joint"),
    # ("right_elbow", "right_elbow_pitch_joint"),
    # ("left_knee", "left_knee_joint"),
    # ("right_knee", "right_knee_joint"),
    # ("left_wrist", "left_palm_joint"),
    # ("right_wrist", "right_palm_joint"),
    # ("left_ankle", "left_ankle_pitch_joint"),
    # ("right_ankle", "right_ankle_pitch_joint"),
    # ("left_shoulder", "left_shoulder_pitch_joint"),
    # ("right_shoulder", "right_shoulder_pitch_joint"),
]:
    smpl_joint_retarget_indices_to_h1.append(smpl_joint_names.index(smpl_name))
    h1_joint_retarget_indices.append(h1_joint_names.index(h1_name))

smpl_joint_retarget_indices_to_g1 = []
g1_joint_retarget_indices = []

for smpl_name, g1_name in [
    # ("pelvis", "torso_joint"),
    ("left_hip", "left_hip_pitch_joint"),
    ("right_hip", "right_hip_pitch_joint"),
    ("left_elbow", "left_elbow_pitch_joint"),
    ("right_elbow", "right_elbow_pitch_joint"),
    ("left_knee", "left_knee_joint"),
    ("right_knee", "right_knee_joint"),
    ("left_wrist", "left_palm_joint"),
    ("right_wrist", "right_palm_joint"),
    ("left_ankle", "left_ankle_roll_joint"),
    ("right_ankle", "right_ankle_roll_joint"),
    # ("left_shoulder", "left_shoulder_pitch_joint"),
    # ("right_shoulder", "right_shoulder_pitch_joint"),
    ("left_shoulder", "left_shoulder_roll_joint"),
    ("right_shoulder", "right_shoulder_roll_joint"),
    # ("head", "head_joint"),
]:
    smpl_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_joint_retarget_indices.append(g1_joint_names.index(g1_name))
    
def save_dict_to_hdf5(h5file, dictionary, path="/"):
    """
    Recursively save a nested dictionary to an HDF5 file.
    
    Args:
        h5file: An open h5py.File object.
        dictionary: The nested dictionary to save.
        path: The current path in the HDF5 file.
    """
    for key, value in dictionary.items():
        key_path = f"{path}{key}"
        if value is None:
            continue
        if isinstance(value, dict):
            # If value is a dictionary, create a group and recurse
            group = h5file.create_group(key_path)
            save_dict_to_hdf5(h5file, value, key_path + "/")
        elif isinstance(value, onp.ndarray):
            h5file.create_dataset(key_path, data=value)
        elif isinstance(value, str):
            # Convert Unicode strings to ASCII strings for HDF5 compatibility
            h5file.attrs[key_path] = value.encode('ascii', 'ignore').decode('ascii')
        elif isinstance(value, (int, float, bytes, list, tuple)):
            h5file.attrs[key_path] = value  # Store scalars as attributes
        else:
            raise TypeError(f"Unsupported data type: {type(value)} for key {key_path}")

class SmplShapeVar(
    jaxls.Var[jnp.ndarray], 
    default_factory=lambda: jnp.zeros(10),
):
    pass

def retargeting_cost(
    var_values: jaxls.VarValues,
    var_smpl_shape: SmplShapeVar,
    smpl_model_jax: SmplModel,
    height: float = 1.8,
) -> jax.Array:
   
    # robot_root_T_world = init_world_T_robot.inverse()

    smpl_shape = var_values[var_smpl_shape]
    smpl_model_with_shapejax = smpl_model_jax.with_shape(smpl_shape)
    default_local_pose_jax = jaxlie.SO3.identity((23,))
    smpl_body_with_pose = smpl_model_with_shapejax.with_pose(
        T_world_root=jaxlie.SE3.identity().wxyz_xyz,
        local_quats=default_local_pose_jax.wxyz,
    )  
    # robot_root_T_smpl_joints  = smpl_body_with_pose.Ts_world_joint
    world_T_smpl_joints = smpl_body_with_pose.Ts_world_joint

    smpl_head_pos = world_T_smpl_joints[..., 4:7][smpl_joint_names.index("head")]
    smpl_left_foot_pos = world_T_smpl_joints[..., 4:7][smpl_joint_names.index("left_foot")]

    adjusted_hieght = height - 0.18 # in meter; 0.1 meter is the distance between the top head and the head joint

    height_residual = jnp.abs((smpl_head_pos[1:2] - smpl_left_foot_pos[1:2]) - adjusted_hieght) * 20

    def smpl_joint_symmetry_cost_func(
        smpl_joints: jnp.ndarray
    ):
        left_smpl_joints_vector = jnp.abs(smpl_joints[left_smpl_joint_indices])
        right_smpl_joints_vector = jnp.abs(smpl_joints[right_smpl_joint_indices])
        residual = left_smpl_joints_vector - right_smpl_joints_vector
        return residual.flatten()
    
    smpl_joint_symmetry_cost = smpl_joint_symmetry_cost_func(world_T_smpl_joints[..., 4:7]) * 1.5

    residual = jnp.concatenate([height_residual, smpl_joint_symmetry_cost])
    
    return residual

def robot_cfg_symmetry_cost(
    var_values: jaxls.VarValues,
    var_robot_cfg: jaxls.Var[jax.Array],
    left_joint_indices: jnp.ndarray,
    right_joint_indices: jnp.ndarray,
) -> jax.Array:
    robot_cfg = var_values[var_robot_cfg]
    left_joint_cfg = robot_cfg[left_joint_indices]
    right_joint_cfg = robot_cfg[right_joint_indices]

    symmetry_residual = jnp.abs(left_joint_cfg - right_joint_cfg)
    return symmetry_residual * 0.1


def smpl_shape_regularization_cost(
    var_values: jaxls.VarValues,
    var_smpl_shape: SmplShapeVar,
    threshold: float = 2.
) -> jax.Array:
    smpl_shape = var_values[var_smpl_shape]

    # penalize if the absoluste value of the shape is greater than 3
    beta_abs = jnp.abs(smpl_shape)
    beta_abs_clipped = jnp.clip(beta_abs - threshold, 0.0)

    return beta_abs_clipped * 0.05


@jdc.jit
def optimize_smpl_shape_for_height(
        smpl_model_jax: SmplModel,
        height: float = 1.8,
):

    var_smpl_shape = SmplShapeVar(0)

    # if robot_name == "h1":
    #     smpl_mask = make_smpl_connection_matrix(urdf_jax, h1_joint_retarget_indices)
    # elif robot_name == "g1":
    #     smpl_mask = make_smpl_connection_matrix(urdf_jax, g1_joint_retarget_indices)
    
    costs = []
    joint_fitting_cost = lambda vals, smpl_shape, smpl_model: retargeting_cost(vals, smpl_shape, smpl_model, height)
    costs.append(
        jaxls.Cost(
                # Wrap the retargeting cost in a lambda to capture robot_name statically.
                compute_residual=joint_fitting_cost,
                args=(
                    var_smpl_shape,
                    smpl_model_jax,
                ),
            )
    )
    
    smpl_shape_reg_cost = lambda vals, shape: smpl_shape_regularization_cost(vals, shape)
    costs.append(
        jaxls.Cost(
            compute_residual=smpl_shape_reg_cost,
            args=(var_smpl_shape,),
        )
    )

    graph = jaxls.LeastSquaresProblem(
        costs=costs,
        variables=[
            var_smpl_shape,
        ]
    ).analyze()
    solved_values = graph.solve(
        initial_vals=jaxls.VarValues.make(
            [
                var_smpl_shape.with_value(jnp.zeros(10)),
            ]
        ),
        linear_solver="dense_cholesky"
        
    )
    jax.block_until_ready(solved_values)

    # Extract the optimized values
    optimized_smpl_shape = solved_values[var_smpl_shape]

    return optimized_smpl_shape

def main(
    height: float = 1.8,
    output_dir: str = "./robot_asset",
    smpl_model_path: str = "./assets/body_models/smpl/SMPL_MALE.pkl",
    vis: bool = False,
):
    
    # Load SMPL model
    smpl_model_jax = SmplModel.load(Path(smpl_model_path))
    
    # Initialize with default shape (zeros)
    current_smpl_shape = jnp.zeros(10)
    current_height = height
    
    # Default poses
    default_global_orient_jax = jaxlie.SE3.identity()
    default_local_pose_jax = jaxlie.SO3.identity((23,))
    
    # Start viser server
    server = viser.ViserServer(port=8081)
    server.scene.world_axes.visible = True
    server.scene.world_axes.axes_length = 0.25
    server.scene.set_up_direction("+y")
    
    # State variables
    is_optimizing = False
    optimization_status = "Ready"
    
    # GUI Elements
    with server.gui.add_folder("SMPL Shape Optimization"):
        # Target height input
        height_input = server.gui.add_slider(
            "Target Height (m)", min=1.0, max=2.2, step=0.01, initial_value=height
        )
        
        # SMPL shape parameter sliders (first 10 principal components)
        shape_sliders = []
        for i in range(10):
            slider = server.gui.add_slider(
                f"Shape β{i}", min=-3.0, max=3.0, step=0.1, initial_value=0.0
            )
            shape_sliders.append(slider)
        
        # Status and measurements
        status_text = server.gui.add_text("Status", initial_value=f"{optimization_status}")
        current_height_text = server.gui.add_text("Approx. Curr. Height", initial_value=f"{height:.3f}m")
        distance_text = server.gui.add_text("Control Distance", initial_value="0.000m")
        
        # Buttons
        optimize_button = server.gui.add_button("Optimize for Height")
        reset_button = server.gui.add_button("Reset to Default")
        save_button = server.gui.add_button("Save Shape")
    
    # Transform controls for height measurement
    control0 = server.scene.add_transform_controls(
        "/controls/foot",
        position=onp.array([0, 0, 0]),
        scale=0.3,
    )
    control1 = server.scene.add_transform_controls(
        "/controls/head",
        position=onp.array([0, height, 0]),
        scale=0.3,
    )
    
    def update_smpl_mesh():
        """Update the SMPL mesh visualization based on current shape parameters."""
        # Get current shape from sliders
        shape_params = jnp.array([slider.value for slider in shape_sliders])
        
        # Create SMPL mesh
        smpl_models_with_shape_jax = smpl_model_jax.with_shape(shape_params)
        shaped_body_with_pose = smpl_models_with_shape_jax.with_pose(
            T_world_root=default_global_orient_jax.wxyz_xyz,
            local_quats=default_local_pose_jax.wxyz,
        )
        smpl_mesh_jax = shaped_body_with_pose.lbs()
        
        # Get mesh data
        smpl_mesh_joints = onp.array(smpl_mesh_jax.posed_model.Ts_world_joint[..., 4:7])
        smpl_mesh_verts = onp.array(smpl_mesh_jax.verts)
        smpl_mesh_faces = onp.array(smpl_mesh_jax.faces)
        
        # Calculate actual height (head to foot distance)
        head_joint_idx = smpl_joint_names.index("head") if "head" in smpl_joint_names else 15  # approximate head joint
        foot_joint_idx = smpl_joint_names.index("left_foot")
        
        head_pos = smpl_mesh_joints[head_joint_idx]
        foot_pos = smpl_mesh_joints[foot_joint_idx]
        actual_height = float(head_pos[1] - foot_pos[1]) + 0.18  # add 0.18 m for head joint offset
        
        # Update height display
        current_height_text.value = f"{actual_height:.3f}m"
        
        # Update control positions
        control0.position = onp.array([0, foot_pos[1], 0])
        control1.position = onp.array([0, head_pos[1] + 0.18, 0])

        # Update mesh visualization
        server.scene.add_mesh_simple(
            "/smpl_mesh",
            vertices=smpl_mesh_verts,
            faces=smpl_mesh_faces,
            flat_shading=False,
            wireframe=False,
            color=(100, 200, 255),
        )
        
        return actual_height, shape_params
    
    def optimize_shape_for_target_height():
        """Optimize SMPL shape for the current target height."""
        nonlocal optimization_status, is_optimizing
        
        if is_optimizing:
            return
            
        is_optimizing = True
        optimization_status = "!!!Optimizing...!!!"
        status_text.value = f"{optimization_status}"
        
        try:
            # Get target height
            target_height = height_input.value
            
            print(f'\033[92mOptimizing SMPL shape for height {target_height:.3f}m...\033[0m')
            
            # Run optimization
            optimized_shape = optimize_smpl_shape_for_height(
                smpl_model_jax, 
                target_height,
            )
            
            # Update sliders with optimized values
            for i, slider in enumerate(shape_sliders):
                slider.value = float(optimized_shape[i])
            
            optimization_status = "Complete!"
            print(f'\033[92mOptimization complete!\033[0m')
            
        except Exception as e:
            optimization_status = f"Failed: {str(e)}"
            print(f'\033[91mOptimization failed: {e}\033[0m')
        
        finally:
            is_optimizing = False
            status_text.value = f"{optimization_status}"
    
    def update_control_distance():
        """Update the distance measurement between control points."""
        distance = onp.linalg.norm(control0.position - control1.position)
        distance_text.value = f"{distance:.3f}m"
        
        # Add visual line between controls
        server.scene.add_spline_catmull_rom(
            "/control_distance_line",
            onp.stack([control0.position, control1.position], axis=0),
            color=(0, 255, 0),
            line_width=2.0,
        )
    
    def on_reset_button(_):
        """Handle reset button click."""
        for slider in shape_sliders:
            slider.value = 0.0
        optimization_status = "Reset to Default"
        status_text.value = f"{optimization_status}"
    
    def on_save_button(_):
        """Handle save button click."""
        try:
            # Get current shape parameters
            shape_params = [slider.value for slider in shape_sliders]
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save shape parameters
            shape_data = {
                "optimized_shape": [shape_params],
                "target_height": height_input.value,
                "timestamp": time.time()
            }
            
            save_path = os.path.join(output_dir, 'optimized_smpl_shape.json')
            with open(save_path, 'w') as f:
                json.dump(shape_data, f, indent=2)
            
            print(f'\033[92mSaved SMPL shape to {save_path}\033[0m')
            optimization_status = f"Saved to {save_path}"
            status_text.value = f"{optimization_status}"
            
        except Exception as e:
            print(f'\033[91mSave failed: {e}\033[0m')
            status_text.value = f"Save Failed - {str(e)}"
    
    # Connect button callbacks
    optimize_button.on_click(lambda _: optimize_shape_for_target_height())
    reset_button.on_click(on_reset_button)
    save_button.on_click(on_save_button)
    
    # Connect slider callbacks for real-time updates
    def on_slider_update(_):
        """Handle all slider updates - just update mesh visualization."""
        update_smpl_mesh()
        update_control_distance()
    
    for slider in shape_sliders:
        slider.on_update(on_slider_update)
    
    # Connect transform control callbacks
    control0.on_update(lambda _: update_control_distance())
    control1.on_update(lambda _: update_control_distance())
    
    # Initial mesh update and distance measurement
    optimize_shape_for_target_height()
    update_smpl_mesh()
    update_control_distance()
    on_save_button(None)
    print("\033[92mInitialization complete!\033[0m")
    
    if vis:
        print('\033[92mViser server started at http://localhost:8081\033[0m')
        print('\033[94mSet target height with the height slider\033[0m')
        print('\033[94mClick "Optimize for Height" to run JAX optimization\033[0m')
        print('\033[94mUse shape parameter sliders for manual fine-tuning in realtime\033[0m')
        print('\033[94mMove control points to measure distances manually\033[0m')
        print('\033[94mClick "Save Shape" to save current parameters\033[0m')    
        # Keep server running
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print('\033[93m\nShutting down server...\033[0m')
    
    server.stop()

if __name__ == "__main__":
    tyro.cli(main)