# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

from __future__ import annotations

import functools
from tqdm import tqdm
from typing import Literal, TypedDict
import time
import os
import os.path as osp
import pickle
import h5py
from pathlib import Path
import glob
import subprocess

import jax
import jaxlie
import jaxls
import numpy as onp
import trimesh
import tyro
import viser
import viser.transforms as vtf
import yourdfpy
import trimesh.ray

from jax import numpy as jnp
from viser.extras import ViserUrdf

import pyroki as pk
import jax_dataclasses as jdc

# import root directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization import viser_camera_util


class RetargetingWeights(TypedDict):
    local_pose_cost_weight: float
    end_effector_cost_weight: float
    global_pose_cost_weight: float
    self_coll_factor_weight: float
    world_coll_factor_weight: float
    limit_cost_factor_weight: float
    smoothness_cost_factor_weight: float
    foot_skating_cost_weight: float
    ground_contact_cost_weight: float
    padding_norm_factor_weight: float
    hip_yaw_cost_weight: float
    hip_pitch_cost_weight: float
    hip_roll_cost_weight: float
    world_coll_margin: float


smpl_joint_names = [
    "pelvis",
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
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

# Note that we now use _links_ instead of joints.
# g1_joint_names = (
#     'pelvis_contour_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_foot_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'right_foot_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'logo_joint', 'head_joint', 'waist_support_joint', 'imu_joint', 'd435_joint', 'mid360_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'left_hand_palm_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 'right_hand_palm_joint'
# )
g1_link_names = (
    'pelvis',
    'pelvis_contour_link',
    'left_hip_pitch_link',
    'left_hip_roll_link',
    'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_pitch_link',
    'left_ankle_roll_link',
    'left_foot_link',
    'right_hip_pitch_link',
    'right_hip_roll_link',
    'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_pitch_link',
    'right_ankle_roll_link',
    'right_foot_link',
    'waist_yaw_link',
    'waist_roll_link',
    'torso_link',
    'logo_link',
    'head_link',
    'waist_support_link',
    'imu_link',
    'd435_link',
    'mid360_link',
    'left_shoulder_pitch_link',
    'left_shoulder_roll_link',
    'left_shoulder_yaw_link',
    'left_elbow_link',
    'left_wrist_roll_link',
    'left_wrist_pitch_link',
    'left_wrist_yaw_link',
    'left_rubber_hand',
    'right_shoulder_pitch_link',
    'right_shoulder_roll_link',
    'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_roll_link',
    'right_wrist_pitch_link',
    'right_wrist_yaw_link',
    'right_rubber_hand'
)

smpl_joint_retarget_indices_to_g1 = []
g1_link_retarget_indices = []

for smpl_name, g1_name in [
    ("pelvis", "pelvis"),
    ("left_hip", "left_hip_roll_link"),
    ("right_hip", "right_hip_roll_link"),
    ("left_elbow", "left_elbow_link"),
    ("right_elbow", "right_elbow_link"),
    ("left_knee", "left_knee_link"),
    ("right_knee", "right_knee_link"),
    ("left_wrist", "left_rubber_hand"),
    ("right_wrist", "right_rubber_hand"),
    ("left_ankle", "left_ankle_pitch_link"),
    ("right_ankle", "right_ankle_pitch_link"),
    ("left_shoulder", "left_shoulder_roll_link"),
    ("right_shoulder", "right_shoulder_roll_link"),
    ("left_foot", "left_foot_link"),
    ("right_foot", "right_foot_link"),
]:
    smpl_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_link_retarget_indices.append(g1_link_names.index(g1_name))

feet_link_pairs_g1 = [
    ("left_foot", "left_foot_link"),
    ("right_foot", "right_foot_link"),
]
ankle_link_pairs_g1 = [
    ("left_ankle", "left_ankle_pitch_link"),
    ("right_ankle", "right_ankle_pitch_link"),
]

smpl_feet_joint_retarget_indices_to_g1 = []
g1_feet_joint_retarget_indices = []

smpl_ankle_joint_retarget_indices_to_g1 = []
g1_ankle_joint_retarget_indices = []

for smpl_name, g1_name in feet_link_pairs_g1:
    smpl_feet_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_feet_joint_retarget_indices.append(g1_link_names.index(g1_name))

for smpl_name, g1_name in ankle_link_pairs_g1:
    smpl_ankle_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_ankle_joint_retarget_indices.append(g1_link_names.index(g1_name))

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
    
    # Get the current group
    if path == "/":
        current_group = h5file
    else:
        current_group = h5file[path]
    
    # Load datasets and groups
    for key in current_group.keys():
        if path == "/":
            key_path = key
        else:
            key_path = f"{path}/{key}"
            
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path)
        else:
            result[key] = h5file[key_path][:]
    
    # Load attributes of the current group
    for attr_key, attr_value in current_group.attrs.items():
        result[attr_key] = attr_value

    return result

def save_dict_to_hdf5(h5file, dictionary, path="/"):
    """
    Recursively save a nested dictionary to an HDF5 file.
    
    Args:
        h5file: An open h5py.File object.
        dictionary: The nested dictionary to save.
        path: The current path in the HDF5 file.
    """
    for key, value in dictionary.items():
        if value is None:
            continue
        if isinstance(value, dict):
            # If value is a dictionary, create a group and recurse
            if path == "/":
                group_path = key
            else:
                group_path = f"{path}/{key}"
            if group_path not in h5file:
                group = h5file.create_group(group_path)
            save_dict_to_hdf5(h5file, value, group_path)
        elif isinstance(value, onp.ndarray):
            if path == "/":
                dataset_path = key
            else:
                dataset_path = f"{path}/{key}"
            h5file.create_dataset(dataset_path, data=value)
        elif isinstance(value, (int, float, str, bytes, list, tuple)):
            # Store scalars as attributes of the parent group
            if path == "/":
                h5file.attrs[key] = value
            else:
                h5file[path].attrs[key] = value
        else:
            raise TypeError(f"Unsupported data type: {type(value)} for key {key}")

def save_all_results(all_person_results, src_dir, urdf, actuated_joint_names, subsample_factor):
    """Save retargeting results for all persons to a single file."""
    if not all_person_results:
        return
        
    # Get joint and link names from URDF
    link_names = list(urdf.scene.geometry.keys())
    link_names = [name.replace(".STL", "") for name in link_names]
    
    # Save combined results
    export_data = {
        "persons": {}
    }
    
    for person_id, results in all_person_results.items():
        num_timesteps = results["num_timesteps"]
        optimized_robot_cfg = results["optimized_robot_cfg"]
        optimized_T_world_root = results["optimized_T_world_root"]
        
        person_data = {
            "joints": onp.array(optimized_robot_cfg[:num_timesteps]),
            "root_quat": onp.zeros((num_timesteps, 4)),
            "root_pos": onp.zeros((num_timesteps, 3)),
            "link_pos": onp.zeros((num_timesteps, len(link_names), 3)),
            "link_quat": onp.zeros((num_timesteps, len(link_names), 4)),
            "contacts": {
                "left_foot": onp.array(results["left_foot_contact"]),
                "right_foot": onp.array(results["right_foot_contact"])
            },
            "fps": 30.0 / subsample_factor
        }
        
        # Fill in root poses
        for t in range(num_timesteps):
            T_world_robot = optimized_T_world_root.wxyz_xyz[t]
            T_world_robot = jaxlie.SE3(wxyz_xyz=T_world_robot)
            
            person_data["root_pos"][t] = onp.array(T_world_robot.wxyz_xyz[4:7])
            wxyz = onp.array(T_world_robot.wxyz_xyz[:4])
            person_data["root_quat"][t] = onp.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])  # xyzw
            
            # Calculate link poses
            for i, link_name in enumerate(link_names):
                T_link_base = urdf.get_transform(link_name, urdf.base_link)
                T_world_robot_mat = onp.array(T_world_robot.as_matrix())
                T_world_current = T_world_robot_mat @ T_link_base
                person_data["link_pos"][t, i] = onp.array(T_world_current[..., :3, 3])
                R = onp.array(T_world_current[..., :3, :3])
                quat = vtf.SO3.from_matrix(R).wxyz
                person_data["link_quat"][t, i] = onp.array([quat[1], quat[2], quat[3], quat[0]])  # xyzw
        
        export_data["persons"][person_id] = person_data
    
    export_data["joint_names"] = actuated_joint_names
    export_data["link_names"] = link_names
    export_data["robot_name"] = "g1"
    
    # Save the data
    save_path = osp.join(src_dir, f"retarget_poses_g1_multiperson.h5")
    with h5py.File(save_path, 'w') as f:
        save_dict_to_hdf5(f, export_data)
    print(f"Saved multi-person retargeting results to {save_path}")

def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
    """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
    delta = delta * jnp.array([1, 1, 1, 0, 0, 0])  # Only update translation.
    return jaxls.SE3Var.retract_fn(transform, delta)

class SmplJointsScaleVarG1(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.ones(
        (len(smpl_joint_retarget_indices_to_g1), len(smpl_joint_retarget_indices_to_g1))
    ),
): ...
smpl_joint_scales_g1 = jnp.ones(
        (len(smpl_joint_retarget_indices_to_g1), len(smpl_joint_retarget_indices_to_g1))
)

def create_conn_tree(robot: pk.Robot, link_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Create a NxN connectivity matrix for N links.
    The matrix is marked Y if there is a direct kinematic chain connection
    between the two links, without bypassing the root link.
    """
    n = len(link_indices)
    conn_matrix = jnp.zeros((n, n))

    joint_indices = [robot.links.parent_joint_indices[link_indices[idx]] for idx in range(n)]

    def is_direct_chain_connection(idx1: int, idx2: int) -> bool:
        """Check if two joints are connected in the kinematic chain without other retargeted joints between"""
        joint1 = joint_indices[idx1]
        joint2 = joint_indices[idx2]

        # Check path from joint2 up to root
        current = joint2
        while current != -1:
            parent = robot.joints.parent_indices[current]
            if parent == joint1:
                return True
            if parent in joint_indices:
                # Hit another retargeted joint before finding joint1
                break
            current = parent

        # Check path from joint1 up to root
        current = joint1
        while current != -1:
            parent = robot.joints.parent_indices[current]
            if parent == joint2:
                return True
            if parent in joint_indices:
                # Hit another retargeted joint before finding joint2
                break
            current = parent

        return False

    # Build symmetric connectivity matrix
    for i in range(n):
        conn_matrix = conn_matrix.at[i, i].set(1.0)  # Self-connection
        for j in range(i + 1, n):
            if is_direct_chain_connection(i, j):
                conn_matrix = conn_matrix.at[i, j].set(1.0)
                conn_matrix = conn_matrix.at[j, i].set(1.0)

    return conn_matrix

@jaxls.Cost.create_factory(name="LocalPoseCost")
def local_pose_cost(
    var_values: jaxls.VarValues,
    var_T_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jax.Array],
    var_smpl_joints_scale: SmplJointsScaleVarG1,
    smpl_mask: jax.Array,
    robot: pk.Robot,
    keypoints: jax.Array, # smpl joints --> keypoints
    local_pose_cost_weight: jax.Array,
) -> jax.Array:
    """Retargeting factor, with a focus on:
    - matching the relative joint/keypoint positions (vectors).
    - and matching the relative angles between the vectors.
    """
    robot_cfg = var_values[var_robot_cfg]
    T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    T_world_root = var_values[var_T_world_root]
    T_world_link = T_world_root @ T_root_link

    smpl_joint_retarget_indices = jnp.array(smpl_joint_retarget_indices_to_g1)
    robot_joint_retarget_indices = jnp.array(g1_link_retarget_indices)
    smpl_pos = keypoints[jnp.array(smpl_joint_retarget_indices)]
    robot_pos = T_world_link.translation()[jnp.array(robot_joint_retarget_indices)]

    # T_world_root = var_values[var_T_world_root]
    # robot_cfg = var_values[var_robot_cfg]
    # T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    # keypoints = T_world_root.inverse() @ keypoints

    # smpl_joint_retarget_indices = jnp.array(smpl_joint_retarget_indices_to_g1)
    # robot_joint_retarget_indices = jnp.array(g1_link_retarget_indices)
    # smpl_pos = keypoints[jnp.array(smpl_joint_retarget_indices)]
    # robot_pos = T_root_link.translation()[jnp.array(robot_joint_retarget_indices)]

    # NxN grid of relative positions.
    delta_smpl = smpl_pos[:, None] - smpl_pos[None, :]
    delta_robot = robot_pos[:, None] - robot_pos[None, :]

    # Vector regularization.
    position_scale = var_values[var_smpl_joints_scale][..., None]
    residual_position_delta = (
        (delta_smpl - delta_robot * position_scale)
        * (1 - jnp.eye(delta_smpl.shape[0])[..., None])
        * smpl_mask[..., None]
    )

    # Vector angle regularization.
    delta_smpl_normalized = delta_smpl # / (jnp.linalg.norm(delta_smpl + 1e-6, axis=-1, keepdims=True) + 1e-6)
    delta_robot_normalized = delta_robot # / (jnp.linalg.norm(delta_robot + 1e-6, axis=-1, keepdims=True) + 1e-6)
    residual_angle_delta = 1 - (delta_smpl_normalized * delta_robot_normalized).sum(axis=-1)

    # delta_smpl_normalized = delta_smpl / (jnp.linalg.norm(delta_smpl + 1e-6, axis=-1, keepdims=True) + 1e-6)
    # delta_robot_normalized = delta_robot / (jnp.linalg.norm(delta_robot + 1e-6, axis=-1, keepdims=True) + 1e-6)
    # residual_angle_delta = jnp.clip(1 - (delta_smpl_normalized * delta_robot_normalized).sum(axis=-1), min=0.1)

    residual_angle_delta = (
        residual_angle_delta
        * (1 - jnp.eye(residual_angle_delta.shape[0]))
        * smpl_mask
    )

    residual = jnp.concatenate(
        [
            residual_position_delta.flatten(),
            residual_angle_delta.flatten(),
        ],
        axis=0,
    ) * local_pose_cost_weight
    return residual

@jaxls.Cost.create_factory(name="GlobalPoseCost")
def global_pose_cost(
    var_values: jaxls.VarValues,
    var_T_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jax.Array],
    var_smpl_joints_scale: SmplJointsScaleVarG1,
    smpl_mask: jax.Array,
    robot: pk.Robot,
    keypoints: jax.Array, # smpl joints --> keypoints
    global_pose_cost_weight: jax.Array,
) -> jax.Array:
    robot_cfg = var_values[var_robot_cfg]
    Ts_root_links = robot.forward_kinematics(cfg=robot_cfg)

    robot_joints = var_values[var_T_world_root] @ jaxlie.SE3(Ts_root_links)

    target_smpl_joint_pos = keypoints[jnp.array(smpl_joint_retarget_indices_to_g1)]
    target_robot_joint_pos = robot_joints.wxyz_xyz[..., 4:7][jnp.array(g1_link_retarget_indices)]

    # center to the feetcenter
    center_between_smpl_feet_joint_pos = target_smpl_joint_pos.mean(axis=0, keepdims=True)
    center_between_robot_feet_joint_pos = target_robot_joint_pos.mean(axis=0, keepdims=True)
    recentered_target_smpl_joint_pos = target_smpl_joint_pos - center_between_smpl_feet_joint_pos
    recentered_target_robot_joint_pos = target_robot_joint_pos - center_between_robot_feet_joint_pos

    global_skeleton_scale = (var_values[var_smpl_joints_scale] * smpl_mask).mean()

    residual = jnp.abs(recentered_target_smpl_joint_pos * global_skeleton_scale - recentered_target_robot_joint_pos).flatten()
    return residual * global_pose_cost_weight

@jaxls.Cost.create_factory(name="EndEffectorCost")
def end_effector_cost(
    var_values: jaxls.VarValues,
    var_T_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jax.Array],
    robot: pk.Robot,
    keypoints: jax.Array, # smpl joints --> keypoints
    end_effector_cost_weight: jax.Array,
) -> jax.Array:
    robot_cfg = var_values[var_robot_cfg]
    Ts_root_links = robot.forward_kinematics(cfg=robot_cfg)

    robot_joints = var_values[var_T_world_root] @ jaxlie.SE3(Ts_root_links)

    target_smpl_feet_joint_pos = keypoints[jnp.array(smpl_feet_joint_retarget_indices_to_g1)]
    target_robot_feet_joint_pos = robot_joints.wxyz_xyz[..., 4:7][jnp.array(g1_feet_joint_retarget_indices)]

    feet_residual = jnp.abs(target_smpl_feet_joint_pos - target_robot_feet_joint_pos).flatten()
    residual = feet_residual

    return residual * end_effector_cost_weight

        
@jaxls.Cost.create_factory(name="FootSkatingCost")
def foot_skating_cost(
    var_values: jaxls.VarValues,
    var_robot_cfg_t0: jaxls.Var[jax.Array],
    var_robot_cfg_t1: jaxls.Var[jax.Array],
    robot: pk.Robot,
    contact_left_foot: jax.Array,
    contact_right_foot: jax.Array,
    foot_skating_cost_weight: jax.Array,   
) -> jax.Array:
    robot_cfg_t0 = var_values[var_robot_cfg_t0]
    robot_cfg_t1 = var_values[var_robot_cfg_t1]
    Ts_root_links_t0 = robot.forward_kinematics(cfg=robot_cfg_t0)
    Ts_root_links_t1 = robot.forward_kinematics(cfg=robot_cfg_t1)

    feet_pos_t0 = Ts_root_links_t0[..., 4:7][jnp.array(g1_feet_joint_retarget_indices)]
    feet_pos_t1 = Ts_root_links_t1[..., 4:7][jnp.array(g1_feet_joint_retarget_indices)]
    ankle_pos_t0 = Ts_root_links_t0[..., 4:7][jnp.array(g1_ankle_joint_retarget_indices)]
    ankle_pos_t1 = Ts_root_links_t1[..., 4:7][jnp.array(g1_ankle_joint_retarget_indices)]

    left_foot_vel = jnp.abs(feet_pos_t1[0] - feet_pos_t0[0]) 
    right_foot_vel = jnp.abs(feet_pos_t1[1] - feet_pos_t0[1]) 

    residual = (left_foot_vel * contact_left_foot).flatten() + (right_foot_vel * contact_right_foot).flatten()

    left_ankle_vel = jnp.abs(ankle_pos_t1[0] - ankle_pos_t0[0]) 
    right_ankle_vel = jnp.abs(ankle_pos_t1[1] - ankle_pos_t0[1]) 
    residual = residual + (left_ankle_vel * contact_left_foot).flatten() + (right_ankle_vel * contact_right_foot).flatten()
    
    return residual * foot_skating_cost_weight



@jaxls.Cost.create_factory(name="GroundContactCost")
def ground_contact_cost(
    var_values: jaxls.VarValues,
    var_T_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jax.Array],
    robot: pk.Robot,
    target_ground_z_for_frame: jax.Array, # Shape (2,) - precomputed Z for left/right
    contact_mask_for_frame: jax.Array, # Shape (2,) - 1 if contact, 0 otherwise
    ground_contact_cost_weight: jax.Array,
) -> jax.Array:
    """
    Penalizes vertical distance between robot feet/ankles and precomputed ground Z,
    only when contact is active.
    """

    robot_cfg = var_values[var_robot_cfg]
    Ts_root_links = robot.forward_kinematics(cfg=robot_cfg)
    robot_joints_world = var_values[var_T_world_root] @ jaxlie.SE3(Ts_root_links)

    # Get the indices of the relevant robot joints (feet or ankles)
    robot_joint_indices = jnp.array(g1_feet_joint_retarget_indices) # Use feet for G1

    # Extract world positions of the relevant robot joints
    target_robot_joint_pos = robot_joints_world.wxyz_xyz[..., 4:7][robot_joint_indices] # (2, 3)

    # Extract Z coordinates
    robot_joint_z = target_robot_joint_pos[..., 2] # (2,)

    # Calculate residual: (robot_z - target_ground_z) * contact_mask
    # We only penalize if the robot foot is above the target ground Z during contact.
    # Penalizing being below might push the foot through the visual mesh.
    # Let's use jnp.maximum(0, robot_joint_z - target_ground_z) to only penalize being above.
    # residual = jnp.maximum(0, robot_joint_z - target_ground_z_for_frame) * contact_mask_for_frame # Shape (2,)
    
    # Simpler: penalize absolute difference, masked by contact
    residual = (robot_joint_z - target_ground_z_for_frame) * contact_mask_for_frame # Shape (2,)


    return residual * ground_contact_cost_weight


@jaxls.Cost.create_factory(name="ScaleRegCost")
def scale_regularization(
    var_values: jaxls.VarValues,
    var_smpl_joints_scale: SmplJointsScaleVarG1,
) -> jax.Array:
    """Regularize the scale of the retargeted joints."""
    # Close to 1.
    res_0 = (var_values[var_smpl_joints_scale] - 1.0).flatten() * 1.0
    # Symmetric.
    res_1 = (
        var_values[var_smpl_joints_scale] - var_values[var_smpl_joints_scale].T
    ).flatten() * 100.0
    # Non-negative.
    res_2 = jnp.clip(-var_values[var_smpl_joints_scale], min=0).flatten() * 100.0
    return jnp.concatenate([res_0, res_1, res_2])

@jaxls.Cost.create_factory(name="HipYawCost")
def hip_yaw_and_pitch_cost(
    var_values: jaxls.VarValues,
    var_robot_cfg: jaxls.Var[jax.Array],
    hip_yaw_cost_weight: jax.Array,
    hip_pitch_cost_weight: jax.Array,
    hip_roll_cost_weight: jax.Array,
) -> jax.Array:
    """Regularize the hip yaw joints to be close to 0."""
    left_hip_pitch_joint_idx = 0
    left_hip_roll_joint_idx = 1
    left_hip_yaw_joint_idx = 2
    right_hip_pitch_joint_idx = 6
    right_hip_roll_joint_idx = 7
    right_hip_yaw_joint_idx = 8

    cfg = var_values[var_robot_cfg]
    residual = jnp.concatenate(
        [
            cfg[..., [left_hip_yaw_joint_idx]] * hip_yaw_cost_weight,
            cfg[..., [right_hip_yaw_joint_idx]] * hip_yaw_cost_weight,
            cfg[..., [left_hip_pitch_joint_idx]] * hip_pitch_cost_weight,
            cfg[..., [right_hip_pitch_joint_idx]] * hip_pitch_cost_weight,
            cfg[..., [left_hip_roll_joint_idx]] * hip_roll_cost_weight,
            cfg[..., [right_hip_roll_joint_idx]] * hip_roll_cost_weight,
        ],
        axis=-1,
    )
    return residual.flatten()

@jaxls.Cost.create_factory(name="RootSmoothnessCost")
def root_smoothness(
    var_values: jaxls.VarValues,
    var_Ts_world_root: jaxls.SE3Var,
    var_Ts_world_root_prev: jaxls.SE3Var,
    root_smoothness_cost_weight: jax.Array,
) -> jax.Array:
    """Smoothness cost for the robot root pose."""
    return (
        var_values[var_Ts_world_root].inverse() @ var_values[var_Ts_world_root_prev]
    ).log().flatten() * root_smoothness_cost_weight


@jdc.jit
def retarget_human_to_robot(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    heightmap: pk.collision.Heightmap,
    valid_timesteps: jax.Array,
    target_keypoints: jax.Array,
    Ts_world_root_smpl: jaxlie.SE3,
    smpl_mask: jax.Array,
    left_foot_contact: jax.Array,
    right_foot_contact: jax.Array,
    target_ground_z: jax.Array,
    weights: RetargetingWeights,
):
    num_timesteps = target_keypoints.shape[0]
    jnp_arange_timesteps = jnp.arange(num_timesteps)

    # Create variables for each timestep
    var_Ts_world_root = jaxls.SE3Var(jnp_arange_timesteps)
    var_joints = robot.joint_var_cls(jnp_arange_timesteps)
    var_smpl_joints_scale = SmplJointsScaleVarG1(0)
    default_smpl_joints_scale = smpl_joint_scales_g1

    costs: list[jaxls.Cost] = []

    costs.extend([
        local_pose_cost(
            var_Ts_world_root,
            var_joints,
            var_smpl_joints_scale,
            jax.tree.map(lambda x: x[None], smpl_mask),
            jax.tree.map(lambda x: x[None], robot),
            jnp.array(target_keypoints),
            weights["local_pose_cost_weight"] * valid_timesteps,
        ),
        end_effector_cost(
            var_Ts_world_root,
            var_joints,
            jax.tree.map(lambda x: x[None], robot),
            jnp.array(target_keypoints),
            weights["end_effector_cost_weight"] * valid_timesteps,
        ),
        global_pose_cost(
            var_Ts_world_root,
            var_joints,
            var_smpl_joints_scale,
            jax.tree.map(lambda x: x[None], smpl_mask),
            jax.tree.map(lambda x: x[None], robot),
            jnp.array(target_keypoints),
            weights["global_pose_cost_weight"] * valid_timesteps,
        ),
    ])

    contact_mask = jnp.stack([left_foot_contact, right_foot_contact]).T # Shape (T, 2)
    target_ground_z = target_ground_z # Shape (T, 2)
    assert contact_mask.shape == (num_timesteps, 2)
    assert target_ground_z.shape == (num_timesteps, 2), (target_ground_z.shape, num_timesteps)

    costs.extend([
        ground_contact_cost(
            var_Ts_world_root,
            var_joints,
            jax.tree.map(lambda x: x[None], robot),
            target_ground_z,
            contact_mask,
            weights["ground_contact_cost_weight"] * valid_timesteps,
        ),
        foot_skating_cost(
            robot.joint_var_cls(jnp.arange(1, num_timesteps)),
            robot.joint_var_cls(jnp.arange(0, num_timesteps-1)),
            jax.tree.map(lambda x: x[None], robot),
            left_foot_contact[1:],
            right_foot_contact[1:],
            weights["foot_skating_cost_weight"] * valid_timesteps[1:],
        ),
    ])

    costs.append(
        scale_regularization(
            var_smpl_joints_scale,
        )
    )

    @jaxls.Cost.create_factory(name="WorldCollisionCost")
    def world_collision_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        coll_weight: jax.Array,
    ) -> jax.Array:
        """
        World collision; we intentionally use a low weight --
        high enough to lift the robot up from the ground, but
        low enough to not interfere with the retargeting.
        """
        Ts_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        coll = robot_coll.at_config(robot, robot_cfg)
        coll = coll.transform(Ts_world_root)

        dist = pk.collision.collide(coll, heightmap)
        act = pk.collision.colldist_from_sdf(dist, activation_dist=weights["world_coll_margin"])
        return act.flatten() * coll_weight

    # Robot sanity costs:
    # - self-collision
    # - joint limits
    # - smoothness
    # - ...
    costs.extend([
        pk.costs.self_collision_cost(
            jax.tree.map(lambda x: x[None], robot),
            jax.tree.map(lambda x: x[None], robot_coll),
            var_joints,
            margin=0.1,
            weight=(
                weights["self_coll_factor_weight"] * weights["padding_norm_factor_weight"]
            ) * valid_timesteps,
        ),
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            weight=(
                weights["limit_cost_factor_weight"] * weights["padding_norm_factor_weight"]
            ) * valid_timesteps,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, num_timesteps)),
            robot.joint_var_cls(jnp.arange(0, num_timesteps-1)),
            weight=(
                weights["smoothness_cost_factor_weight"] * weights["padding_norm_factor_weight"]
            ) * valid_timesteps[1:],
        ),
        # hip_yaw_and_pitch_cost(
        #     var_joints,
        #     weights["hip_yaw_cost_weight"] * weights["padding_norm_factor_weight"] * valid_timesteps,
        #     weights["hip_pitch_cost_weight"] * weights["padding_norm_factor_weight"] * valid_timesteps,
        #     weights["hip_roll_cost_weight"] * weights["padding_norm_factor_weight"] * valid_timesteps,
        # ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, num_timesteps)),
            jaxls.SE3Var(jnp.arange(0, num_timesteps-1)),
            (
                2 * weights["smoothness_cost_factor_weight"]
            ) * (valid_timesteps[1:] * valid_timesteps[:-1]),
        ),
        world_collision_cost(
            var_Ts_world_root,
            var_joints,
            weights['world_coll_factor_weight'] * weights["padding_norm_factor_weight"] * valid_timesteps,
        )
    ])

    # Build and solve graph
    graph = jaxls.LeastSquaresProblem(
        costs=costs,
        variables=[var_Ts_world_root, var_joints, var_smpl_joints_scale],
    ).analyze()

    solved_values = graph.solve(
        initial_vals=jaxls.VarValues.make(
            [
                var_Ts_world_root.with_value(Ts_world_root_smpl),
                var_joints,
                var_smpl_joints_scale.with_value(default_smpl_joints_scale),
            ]
        ),
    )

    # Extract results
    optimized_T_world_root = solved_values[var_Ts_world_root]
    optimized_robot_cfg = solved_values[var_joints]
    optimized_scale = solved_values[var_smpl_joints_scale][0]

    return optimized_scale, optimized_robot_cfg, optimized_T_world_root


def sanitize_joint_angles(joint_angles: jnp.ndarray, joint_limits_upper: jnp.ndarray, joint_limits_lower: jnp.ndarray):
    # joint_angles: (T, N)
    # joint_limits_upper: (N,)
    # joint_limits_lower: (N,)
    # return: (T, N)
    # Reshape to (T,N) if needed
    if len(joint_angles.shape) == 1:
        joint_angles = joint_angles.reshape(1,-1)
        
    # Broadcast limits to match joint angles shape
    joint_limits_upper = jnp.broadcast_to(joint_limits_upper, joint_angles.shape)
    joint_limits_lower = jnp.broadcast_to(joint_limits_lower, joint_angles.shape)

    # Assuming the joint angles are in the range of [-pi, pi]
    # If not, we need to normalize them to be within the range of [-pi, pi]
    joint_angles_mod = (joint_angles + jnp.pi) % (2 * jnp.pi) - jnp.pi

    # And then clip with the limits
    joint_angles_clipped = jnp.clip(joint_angles_mod, joint_limits_lower, joint_limits_upper)

    return joint_angles_clipped


def process_retargeting(
    urdf: yourdfpy.URDF,
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    src_dir: Path,
    contact_dir: Path,
    subsample_factor: int = 1,
    offset_factor: float = 0.00,
    smpl_root_joint_idx: int = 0,
    vis: bool = False,
    multihuman: bool = False,
    top_k: int = 1,
    local_pose_cost_weight: float = 1.0,
    end_effector_cost_weight: float = 5.0,
    global_pose_cost_weight: float = 8.0,
    self_coll_factor_weight: float = 1.0,
    world_coll_factor_weight: float = 1.0,
    world_coll_margin: float = 0.01,
    limit_cost_factor_weight: float = 1000.0,
    smoothness_cost_factor_weight: float = 2.0,
    foot_skating_cost_weight: float = 1.0,
    ground_contact_cost_weight: float = 5.0,
    hip_yaw_cost_weight: float = 50.0,
    hip_pitch_cost_weight: float = 50.0,
    hip_roll_cost_weight: float = 50.0,
) -> None:
    """ Load source data: human keypoints and background mesh. """
    # Extract the frame indices used; input path contains information about the frames used
    # ex) megahunter_align3r_reconstruction_results_IMG_7381-00.00.10.330-00.00.15.516-seg2_cam01_frame_0_178_subsample_2
    src_dir_str = str(src_dir) # Use string for splitting
    start_frame = int(src_dir_str.split("_frame_")[1].split("_subsample_")[0].split("_")[0])
    end_frame = int(src_dir_str.split("_frame_")[1].split("_subsample_")[0].split("_")[1])

    print("--- Background Mesh Loading ---")
    keypoints_path = src_dir / "gravity_calibrated_keypoints.h5"
    bg_mesh_path = src_dir / "background_mesh.obj"
    with h5py.File(keypoints_path, 'r') as f:
        keypoints_output = load_dict_from_hdf5(f)

    world_coll_geoms = [] # Initialize empty list
    if bg_mesh_path.exists():
        background_mesh = trimesh.load(str(bg_mesh_path), force='mesh')
        if not isinstance(background_mesh, trimesh.Trimesh) or len(background_mesh.faces) == 0:
            print(f"\033[93mWarning: Background mesh {bg_mesh_path} could not be loaded or is empty. Ground contact cost will be disabled.\033[0m")
            background_mesh = None
            # Disable ground contact cost if mesh is invalid
            ground_contact_cost_weight = 0.0    
            # Disable world collision cost if mesh is invalid
            world_coll_factor_weight = 0.0
    else:
        print(f"\033[93mWarning: Background mesh file not found at {bg_mesh_path}. Ground contact cost will be disabled.\033[0m")
        background_mesh = None
        # Disable ground contact cost if mesh not found
        ground_contact_cost_weight = 0.0
        # Disable world collision cost if mesh not found
        world_coll_factor_weight = 0.0

    # Extract person IDs to process
    all_person_ids = list(keypoints_output["joints"].keys())
    
    if multihuman:
        # In multi-human mode, process multiple people
        if top_k == 0:
            # Process all people
            person_ids_to_process = all_person_ids
        else:
            # Process top-k people (already sorted by average area from preprocessing)
            person_ids_to_process = all_person_ids[:min(top_k, len(all_person_ids))]
        print(f"Multi-human mode: Processing {len(person_ids_to_process)} out of {len(all_person_ids)} detected persons")
    else:
        # Single person mode - process only the first person
        person_ids_to_process = [all_person_ids[0]]
        print(f"Single person mode: Retargeting person {person_ids_to_process[0]}'s joints to robot.")
    
    # Create heightmap before the person loop
    heightmap = pk.collision.Heightmap.from_trimesh(background_mesh, x_bins=500, y_bins=500)
    
    # Set up retargeting weight helpers before the person loop
    weights: RetargetingWeights = {
        "local_pose_cost_weight": local_pose_cost_weight,
        "end_effector_cost_weight": end_effector_cost_weight,
        "global_pose_cost_weight": global_pose_cost_weight,
        "self_coll_factor_weight": self_coll_factor_weight,
        "world_coll_factor_weight": world_coll_factor_weight,
        "limit_cost_factor_weight": limit_cost_factor_weight,
        "smoothness_cost_factor_weight": smoothness_cost_factor_weight,
        "foot_skating_cost_weight": foot_skating_cost_weight,
        "ground_contact_cost_weight": ground_contact_cost_weight,
        "padding_norm_factor_weight": 1.0,  # Will be updated per person
        "hip_yaw_cost_weight": hip_yaw_cost_weight,
        "hip_pitch_cost_weight": hip_pitch_cost_weight,
        "hip_roll_cost_weight": hip_roll_cost_weight,
        "world_coll_margin": world_coll_margin,
    }
    
    # Start viser server if visualization is enabled
    if vis:
        server = viser.ViserServer(port=8081)
        server.scene.set_up_direction("+y")
        weight_tuner = pk.viewer.WeightTuner(server, weights, max={"limit_cost_factor_weight": 1000.0})
    else:
        server = None
        weight_tuner = None
    
    # Store results for all persons
    all_person_results = {}
    
    # Process each person
    for person_idx, person_id in enumerate(person_ids_to_process):
        print(f"\n--- Processing person {person_id} ({person_idx+1}/{len(person_ids_to_process)}) ---")
        
        target_keypoints = keypoints_output["joints"][person_id]  # Shape: (N, 45, 3)
        target_root_orient = keypoints_output["root_orient"][person_id]  # Shape: (N, 1, 3, 3)
        num_timesteps = target_keypoints.shape[0]
        assert target_keypoints.shape == (num_timesteps, 45, 3)
        assert target_root_orient.shape == (num_timesteps, 1, 3, 3)
        target_keypoints = jnp.array(target_keypoints)
        # Initialize the T_world_root from smpl root joint
        target_root_joint_coord = jnp.array(target_keypoints[:, smpl_root_joint_idx, :]) # (N, 3)
        target_root_joint_orient = jaxlie.SO3.from_matrix(target_root_orient[:, 0, :, :]) # (N, 3, 3)
        target_T_world_root = jaxlie.SE3.from_rotation_and_translation(target_root_joint_orient, target_root_joint_coord) # (N, )

        """ Load contact information. """
        # load the contact estimation
        # src_dir is Path object: PosixPath('demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_release_test_01_cam01_frame_10_90_subsample_3')
        # megahunter_path should be something like: demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_release_test_01_cam01_frame_10_90_subsample_3/gravity_calibrated_megahunter.h5
        megahunter_path = src_dir / "gravity_calibrated_megahunter.h5"
        if osp.exists(contact_dir) and osp.exists(megahunter_path):
            world_env = load_dict_from_hdf5(h5py.File(megahunter_path, 'r'))['our_pred_world_cameras_and_structure']
            contact_estimation = {}
            for frame_name in world_env.keys():
                if osp.exists(osp.join(contact_dir, f'{frame_name}.pkl')):
                    with open(osp.join(contact_dir, f'{frame_name}.pkl'), 'rb') as f:
                        contact_estimation[frame_name] = pickle.load(f) # Dict[person_id, Dict[left_foot_contact, right_foot_contact, frame_contact_vertices]]

            left_foot_contact_list = []
            right_foot_contact_list = []
            try:
                for frame_name in world_env.keys():
                    if int(person_id) in contact_estimation[frame_name].keys():
                        left_foot_contact_list.append(contact_estimation[frame_name][int(person_id)]["left_foot_contact"])
                        right_foot_contact_list.append(contact_estimation[frame_name][int(person_id)]["right_foot_contact"])
                
                left_foot_contact = jnp.array(left_foot_contact_list)
                right_foot_contact = jnp.array(right_foot_contact_list)

                if len(left_foot_contact) > num_timesteps:
                    left_foot_contact = left_foot_contact[:num_timesteps]
                    right_foot_contact = right_foot_contact[:num_timesteps]
                else:
                    left_foot_contact = jnp.pad(left_foot_contact, ((0, num_timesteps - len(left_foot_contact)),), mode="edge")
                    right_foot_contact = jnp.pad(right_foot_contact, ((0, num_timesteps - len(right_foot_contact)),), mode="edge")

                assert left_foot_contact.shape == right_foot_contact.shape == (num_timesteps,)
                
            except Exception as e:
                print(f"\033[93mWarning: No feet contact estimation found for person {person_id} in {src_dir}. Set to zeros.\033[0m")
                left_foot_contact = jnp.zeros((num_timesteps,))
                right_foot_contact = jnp.zeros((num_timesteps,))
                ground_contact_cost_weight = 0.0
                foot_skating_cost_weight = 0.0
        else:
            print(f"[Warning]: No feet contact estimation found for person {person_id}. Set to zeros.")
            left_foot_contact = jnp.zeros((num_timesteps,))
            right_foot_contact = jnp.zeros((num_timesteps,))
            ground_contact_cost_weight = 0.0
            foot_skating_cost_weight = 0.0
        
        # pad the num_timesteps to be mod 0 for 100
        padded_num_timesteps = ((num_timesteps - 1) // 100 + 1) * 100
        # pad target_keypoints, left_foot_contact, right_foot_contact to be the same length
        target_keypoints = jnp.pad(target_keypoints, ((0, padded_num_timesteps - num_timesteps), (0, 0), (0, 0)), mode="edge")
        left_foot_contact = jnp.pad(left_foot_contact, ((0, padded_num_timesteps - num_timesteps),), mode="edge")
        right_foot_contact = jnp.pad(right_foot_contact, ((0, padded_num_timesteps - num_timesteps),), mode="edge")
        valid_timesteps = jnp.ones(num_timesteps)
        valid_timesteps = jnp.pad(valid_timesteps, ((0, padded_num_timesteps - num_timesteps),), mode="constant", constant_values=0)
        org_num_timesteps = num_timesteps
        num_timesteps = padded_num_timesteps
        # valid_timesteps = jnp.ones(num_timesteps)

        """ Load robot urdf, urdf_jax, and define the collision model. """
        # Create the connection matrix
        smpl_mask = create_conn_tree(robot, jnp.array(g1_link_retarget_indices))

        padding_norm_factor_weight = float(valid_timesteps.sum() / num_timesteps)

        # --- Precompute Target Ground Z using Raycasting (Revised Logic) ---
        target_ground_z = onp.zeros((num_timesteps, 2)) # (T, 2) for left/right relevant joint
        if background_mesh is not None and ground_contact_cost_weight > 0:
            print("Precomputing target ground Z using raycasting (triangle center method)...")

            relevant_smpl_indices = smpl_feet_joint_retarget_indices_to_g1
            print("Using SMPL feet for ground projection.")

            assert len(relevant_smpl_indices) == 2, "Expected 2 relevant SMPL indices (left/right)"

            human_joint_positions = onp.array(target_keypoints[:, relevant_smpl_indices, :]) # Use onp for trimesh (T, 2, 3)

            # Prepare ray origins slightly above the joints and flatten
            ray_origins_flat = (human_joint_positions + onp.array([0, 0, 0.1])).reshape(-1, 3) # (T*2, 3)
            num_rays = ray_origins_flat.shape[0]

            # --- Downward Raycast ---
            down_directions = onp.full_like(ray_origins_flat, [0, 0, -1.0]) # (T*2, 3)
            print("Performing downward raycast...")
            index_tri_down = background_mesh.ray.intersects_first(
                ray_origins=ray_origins_flat,
                ray_directions=down_directions
            ) # Shape: (T*2,) Returns -1 on miss

            # Initialize index_tri with downward results
            index_tri = index_tri_down.copy()

            # --- Upward Raycast for Misses ---
            missed_indices_down = onp.where(index_tri == -1)[0]
            if len(missed_indices_down) > 0:
                print(f"Downward raycast missed {len(missed_indices_down)} rays. Performing upward raycast for misses...")
                origins_for_upward = ray_origins_flat[missed_indices_down]
                # Start upward rays slightly below the original joint position
                # origins_for_upward = (human_joint_positions.reshape(-1,3)[missed_indices_down] - onp.array([0,0,0.01]))
                up_directions = onp.full_like(origins_for_upward, [0, 0, 1.0])

                index_tri_up = background_mesh.ray.intersects_first(
                    ray_origins=origins_for_upward,
                    ray_directions=up_directions
                ) # Shape: (len(missed_indices_down),)

                # Update index_tri with upward hits
                valid_up_hits = onp.where(index_tri_up != -1)[0]
                if len(valid_up_hits) > 0:
                     print(f"Upward raycast found hits for {len(valid_up_hits)} previously missed rays.")
                     update_indices = missed_indices_down[valid_up_hits]
                     index_tri[update_indices] = index_tri_up[valid_up_hits]
                else:
                     print("Upward raycast did not find any additional hits.")

            # --- Calculate Triangle Center Z and Handle Final Misses ---
            final_missed_indices = onp.where(index_tri == -1)[0]
            if len(final_missed_indices) > 0:
                 print(f"\033[93mWarning: {len(final_missed_indices)} rays missed in both directions.\033[0m")

            # Calculate fallback Z based on minimum *human* joint height for stability
            fallback_z = onp.min(human_joint_positions[..., 2]) if human_joint_positions.size > 0 else 0.0
            print(f"Using fallback Z: {fallback_z:.3f} for final missed rays.")

            target_ground_z_flat = onp.zeros(num_rays)
            valid_hit_indices = onp.where(index_tri != -1)[0]

            if len(valid_hit_indices) > 0:
                hit_triangle_indices = index_tri[valid_hit_indices]
                # Ensure triangle indices are within bounds
                valid_triangle_indices = hit_triangle_indices < len(background_mesh.triangles)
                if not onp.all(valid_triangle_indices):
                    print(f"\033[91mError: Found invalid triangle indices after raycast! Check mesh integrity.\033[0m")
                     # Handle error appropriately, e.g., use fallback for invalid indices
                    invalid_original_indices = valid_hit_indices[~valid_triangle_indices]
                    index_tri[invalid_original_indices] = -1 # Mark as missed
                    # Recalculate valid hits
                    valid_hit_indices = onp.where(index_tri != -1)[0]
                    hit_triangle_indices = index_tri[valid_hit_indices]


                projected_triangles = background_mesh.triangles[hit_triangle_indices] # (num_valid_hits, 3, 3)
                projected_triangles_center = projected_triangles.mean(axis=1) # (num_valid_hits, 3)
                projected_triangles_center_z = projected_triangles_center[:, 2] # (num_valid_hits,)
                target_ground_z_flat[valid_hit_indices] = projected_triangles_center_z

            # Apply fallback Z to all missed indices
            target_ground_z_flat[index_tri == -1] = fallback_z


            # Reshape back to (T, 2)
            target_ground_z = target_ground_z_flat.reshape(-1, 2) # Reshape to (T, 2)

            print(f"Raycasting complete. Final Misses: {len(final_missed_indices)}")

        else:
            # If no mesh or weight is zero, fill with zeros (or fallback Z)
            fallback_z = onp.min(onp.array(target_keypoints)[..., 2]) if target_keypoints.size > 0 else 0.0
            target_ground_z.fill(fallback_z)
            print("Skipping ground Z precomputation (no mesh or zero weight).")

        target_ground_z = jnp.array(target_ground_z) # Convert to JAX array
        # --- End Precomputation ---
        padding_norm_factor_weight = float(valid_timesteps.sum() / num_timesteps)

        # Update padding_norm_factor_weight in the weights dictionary
        weights["padding_norm_factor_weight"] = padding_norm_factor_weight

        num_timesteps = org_num_timesteps # Restore original timestep count
        optimized_scale, optimized_robot_cfg, optimized_T_world_root = None, None, None

        # add temporal smoothing for T_world_robot and robot_cfg
        from utilities.one_euro_filter import OneEuroFilter
        freq = 30 / subsample_factor
        one_euro_filter_T_world_robot_trans = OneEuroFilter(freq=freq, mincutoff=0.5, beta=2.0, dcutoff=0.5)
        one_euro_filter_robot_cfg = OneEuroFilter(freq=freq, mincutoff=1.0, beta=1.5, dcutoff=1.0)
        T_world_robot_dict = {}
        robot_cfg_dict = {}

        def run_retargeting():
            nonlocal optimized_scale, optimized_robot_cfg, optimized_T_world_root
            print(f"Starting optimization with {org_num_timesteps} timesteps (padded to {padded_num_timesteps}).")

            start_time = time.time()
            _optimized_scale, _optimized_robot_cfg, _optimized_T_world_root = retarget_human_to_robot(
                robot,
                robot_coll,
                heightmap,
                valid_timesteps,
                target_keypoints,
                target_T_world_root,
                smpl_mask,
                left_foot_contact,
                right_foot_contact,
                target_ground_z,
                weight_tuner.get_weights() if weight_tuner else weights,
            )
            jax.block_until_ready((
                _optimized_scale,
                _optimized_robot_cfg,
                _optimized_T_world_root,
            ))
            end_time = time.time()
            print(f"Optimization finished in {end_time - start_time:.2f} seconds.")

            optimized_scale, optimized_robot_cfg, optimized_T_world_root = _optimized_scale, _optimized_robot_cfg, _optimized_T_world_root
            optimized_robot_cfg = sanitize_joint_angles(optimized_robot_cfg, robot.joints.upper_limits, robot.joints.lower_limits)

            T_world_robot_dict.clear()
            robot_cfg_dict.clear()

            # Reset the one euro filters
            one_euro_filter_T_world_robot_trans.x_prev = None
            one_euro_filter_T_world_robot_trans.dx_prev = None
            one_euro_filter_T_world_robot_trans.t_prev = None
            one_euro_filter_robot_cfg.x_prev = None
            one_euro_filter_robot_cfg.dx_prev = None
            one_euro_filter_robot_cfg.t_prev = None

            print("Retargeting complete!")
        
        run_retargeting()
        assert optimized_scale is not None
        assert optimized_robot_cfg is not None
        assert optimized_T_world_root is not None
    
        # Store results for this person
        all_person_results[person_id] = {
            "optimized_scale": optimized_scale,
            "optimized_robot_cfg": optimized_robot_cfg,
            "optimized_T_world_root": optimized_T_world_root,
            "num_timesteps": org_num_timesteps,
            "left_foot_contact": left_foot_contact[:org_num_timesteps],
            "right_foot_contact": right_foot_contact[:org_num_timesteps],
            "target_keypoints": target_keypoints[:org_num_timesteps],
        }
        
        # Per-person visualization during retargeting (if vis is True)
        if vis and server is not None:
            # Show individual robot for this person
            print(f"\n=== Visualizing retargeting results for person {person_id} ===")
            robot_frame_person = server.scene.add_frame(f"/robot_person_{person_id}", axes_length=0.1, axes_radius=0.01)
            urdf_viser_person = ViserUrdf(server, urdf_or_path=urdf, root_node_name=f"/robot_person_{person_id}")
            
            # Show heightmap mesh for context
            if background_mesh is not None:
                heightmap_mesh = heightmap.to_trimesh()
                heightmap_mesh_handle = server.scene.add_mesh_trimesh(
                    f"/heightmap_person_{person_id}",
                    heightmap_mesh,
                    visible=True
                )
            
            # Create person-specific controls
            person_folder = server.gui.add_folder(f"Person {person_id} Controls")
            with person_folder:
                gui_update_person = server.gui.add_button(f"Update Person {person_id}")
                if person_idx < len(person_ids_to_process) - 1:
                    gui_next_person = server.gui.add_button("Continue to Next Person")
                else:
                    gui_finish = server.gui.add_button("Finish and Show All Robots")
                
                # Add playback controls for this person
                gui_person_timestep = server.gui.add_slider(
                    f"Timestep",
                    min=0,
                    max=org_num_timesteps - 1,
                    step=1,
                    initial_value=0,
                )
                gui_person_playing = server.gui.add_checkbox("Playing", True)
                gui_person_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)
            
            # Function to update this person's robot pose
            current_person_joints_handle = None
            def update_person_cfg(t: int):
                T_world_robot = optimized_T_world_root.wxyz_xyz[t]
                T_world_robot = T_world_robot + jnp.array([0, 0, 0, 0, 0, 0, offset_factor])
                T_world_robot = jaxlie.SE3(wxyz_xyz=T_world_robot)
                
                robot_frame_person.wxyz = onp.array(T_world_robot.wxyz_xyz[:4])
                robot_frame_person.position = onp.array(T_world_robot.wxyz_xyz[4:7])
                urdf_viser_person.update_cfg(onp.array(optimized_robot_cfg[t]))
                
                # Show this person's keypoints
                nonlocal current_person_joints_handle
                current_person_joints_handle = server.scene.add_point_cloud(
                    f"/current_person_joints",
                    keypoints_output["joints"][person_id][t],
                    colors=(0, 255, 0),  # Green for current person
                    point_size=0.01,
                )
            
            # Update function for timestep
            @gui_person_timestep.on_update
            def _(_) -> None:
                update_person_cfg(gui_person_timestep.value)
            
            # Re-run retargeting for this person
            @gui_update_person.on_click
            def _(_) -> None:
                nonlocal optimized_scale, optimized_robot_cfg, optimized_T_world_root
                print(f"Re-running retargeting for person {person_id}...")
                gui_person_playing.value = False
                run_retargeting()
                # Update stored results
                all_person_results[person_id] = {
                    "optimized_scale": optimized_scale,
                    "optimized_robot_cfg": optimized_robot_cfg,
                    "optimized_T_world_root": optimized_T_world_root,
                    "num_timesteps": org_num_timesteps,
                    "left_foot_contact": left_foot_contact[:org_num_timesteps],
                    "right_foot_contact": right_foot_contact[:org_num_timesteps],
                    "target_keypoints": target_keypoints[:org_num_timesteps],
                }
                print(f"Updated retargeting for person {person_id}")
            
            # Control for moving to next person or finishing
            proceed_to_next = False
            
            if person_idx < len(person_ids_to_process) - 1:
                @gui_next_person.on_click
                def _(_) -> None:
                    nonlocal proceed_to_next
                    proceed_to_next = True
                    gui_person_playing.value = False
            else:
                @gui_finish.on_click
                def _(_) -> None:
                    nonlocal proceed_to_next
                    proceed_to_next = True
                    gui_person_playing.value = False
            
            # Playback loop for this person
            while not proceed_to_next:
                if gui_person_playing.value:
                    gui_person_timestep.value = (gui_person_timestep.value + 1) % org_num_timesteps
                time.sleep(1.0 / gui_person_fps.value)
            
            # Clean up individual visualization
            robot_frame_person.remove()
            # ViserUrdf doesn't have a remove method, but we can hide its components
            for joint_frame in urdf_viser_person._joint_frames:
                joint_frame.visible = False
            for mesh_node in urdf_viser_person._meshes:
                mesh_node.visible = False
            # Note: point cloud handles are overwritten each frame, so they don't need explicit removal
            if background_mesh is not None:
                heightmap_mesh_handle.remove()
            # Remove the person-specific folder
            person_folder.remove()
            if current_person_joints_handle is not None:
                current_person_joints_handle.remove()

    """ Multi-robot visualization and saving """
    # if not vis or server is None:
    #     server = viser.ViserServer(port=8081)
    #     urdf_viser = ViserUrdf(server, urdf_or_path=urdf)

    #     if multihuman:
    #         # End of person loop
    #         actuated_joint_names = urdf_viser.get_actuated_joint_names()
    #         save_all_results(all_person_results, src_dir, urdf, actuated_joint_names, subsample_factor)

    #     else:
    #         actuated_joint_names = urdf_viser.get_actuated_joint_names()
    #         save_all_results(all_person_results, src_dir, urdf, actuated_joint_names, subsample_factor)

    #     return

    if not vis or server is None:
        # Ugly, but to get the actuated joint names for saving...
        server = viser.ViserServer(port=8081)
        urdf_viser = ViserUrdf(server, urdf_or_path=urdf)
        actuated_joint_names = urdf_viser.get_actuated_joint_names()

    # Create robot frames for all persons
    print("\n=== Final Multi-Robot Visualization ===")
    print(f"Showing {len(all_person_results)} robots together")
    
    robot_frames = {}
    urdf_visers = {}
    for person_id in all_person_results.keys():
        robot_frames[person_id] = server.scene.add_frame(f"/robot_{person_id}", axes_length=0.1, axes_radius=0.01)
        urdf_visers[person_id] = ViserUrdf(server, urdf_or_path=urdf, root_node_name=f"/robot_{person_id}")

    # Add info text and save controls
    with server.gui.add_folder("Final Results"):
        gui_info = server.gui.add_text("Status", f"Showing {len(all_person_results)} robots", disabled=True)
        gui_save_button = server.gui.add_button("Save All Results")

    save_count = 0
    def save_results():
        nonlocal save_count
        
        save_count += 1
        print(f"Saving all robot results...")
        save_all_results(all_person_results, src_dir, urdf, actuated_joint_names, subsample_factor)

    @gui_save_button.on_click
    def _(_) -> None:
        save_results()

    # --- Visualize background mesh and heightmap --- 
    if vis and background_mesh is not None:
        print("Visualizing background mesh.")
        background_mesh_handle = server.scene.add_mesh_simple(
            name="/background",
            vertices=background_mesh.vertices,
            faces=background_mesh.faces,
            color=(200, 200, 200),  # Light gray color
            wireframe=False,
            opacity=1.0,
            material="standard",
            flat_shading=False,
            side="double",  # Render both sides of the mesh
            wxyz=(1.0, 0.0, 0.0, 0.0),  # quaternion (w, x, y, z)
            position=(0.0, 0.0, 0.0),
            visible=False  # Default to not visible
        )
        heightmap_mesh = heightmap.to_trimesh()
        heightmap_mesh_handle = server.scene.add_mesh_trimesh(
            "/heightmap",
            heightmap_mesh,
            visible=True  # Show heightmap by default in final view
        )
    else:
        print("Background mesh not loaded, cannot visualize.")
    # --- End Mesh Visualization --- 

    with server.gui.add_folder("Mesh Visibility"):
        # Color print 
        print("\033[93mHeightmap Mesh is the actual mesh used for contact and collision cost.\033[0m")
        # Add mesh visibility controls
        gui_heightmap_mesh_visible = server.gui.add_checkbox("Show Heightmap Mesh", initial_value=True)
        gui_background_mesh_visible = server.gui.add_checkbox("Show Background Mesh", initial_value=False)

    # Add mesh visibility control callbacks
    @gui_background_mesh_visible.on_update
    def _(_) -> None:
        if background_mesh is not None:
            background_mesh_handle.visible = gui_background_mesh_visible.value

    @gui_heightmap_mesh_visible.on_update  
    def _(_) -> None:
        if background_mesh is not None:
            heightmap_mesh_handle.visible = gui_heightmap_mesh_visible.value

    # Update all robot frames
    def update_cfg(t: int):
        # Update each robot
        for person_id, results in all_person_results.items():
            if t >= results["num_timesteps"]:
                # Hide robot if timestep exceeds this person's data
                robot_frames[person_id].visible = False
                continue
            else:
                robot_frames[person_id].visible = True
            
            optimized_T_world_root = results["optimized_T_world_root"]
            optimized_robot_cfg = results["optimized_robot_cfg"]
            
            T_world_robot = optimized_T_world_root.wxyz_xyz[t]
            T_world_robot = T_world_robot + jnp.array([0, 0, 0, 0, 0, 0, offset_factor])
            T_world_robot = jaxlie.SE3(wxyz_xyz=T_world_robot)
            
            robot_frames[person_id].wxyz = onp.array(T_world_robot.wxyz_xyz[:4])
            robot_frames[person_id].position = onp.array(T_world_robot.wxyz_xyz[4:7])
            
            # Update joints
            urdf_visers[person_id].update_cfg(onp.array(optimized_robot_cfg[t]))
        
        # Visualize all person keypoints
        all_keypoints = []
        all_colors = []
        for idx, person_id in enumerate(all_person_results.keys()):
            if t < keypoints_output["joints"][person_id].shape[0]:
                all_keypoints.append(keypoints_output["joints"][person_id][t])
                # Different color for each person
                color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][idx % 5]
                all_colors.extend([color] * keypoints_output["joints"][person_id][t].shape[0])
        
        if all_keypoints:
            server.scene.add_point_cloud(
                "/joints",
                onp.vstack(all_keypoints),
                colors=onp.array(all_colors),
                point_size=0.01,
            )

    # Step 1: Get joint names and link names (use first robot)
    first_person_id = list(all_person_results.keys())[0]
    actuated_joint_names = urdf_visers[first_person_id].get_actuated_joint_names()
    link_names = list(urdf.scene.geometry.keys())
    # drop .STL
    link_names = [name.replace(".STL", "") for name in link_names]
    actuated_num_joints = len(actuated_joint_names)

    # Get maximum number of timesteps across all persons
    max_timesteps = max(results["num_timesteps"] for results in all_person_results.values())
    
    # Save all results before starting visualization
    if not multihuman:
        # Save single person result in original format for compatibility
        first_person_id = list(all_person_results.keys())[0]
        results = all_person_results[first_person_id]
        
        export_data = {
            "joint_names": actuated_joint_names,
            "joints": onp.array(results["optimized_robot_cfg"][:results["num_timesteps"]]),
            "root_quat": onp.zeros((results["num_timesteps"], 4)),
            "root_pos": onp.zeros((results["num_timesteps"], 3)),
            "link_names": link_names,
            "link_pos": onp.zeros((results["num_timesteps"], len(link_names), 3)),
            "link_quat": onp.zeros((results["num_timesteps"], len(link_names), 4)),
            "contacts": {
                "left_foot": onp.array(results["left_foot_contact"]),
                "right_foot": onp.array(results["right_foot_contact"])
            },
            "fps": 30.0 / subsample_factor
        }
        
        # Fill in transforms
        for t in range(results["num_timesteps"]):
            T_world_robot = results["optimized_T_world_root"].wxyz_xyz[t]
            T_world_robot = jaxlie.SE3(wxyz_xyz=T_world_robot)
            
            export_data["root_pos"][t] = onp.array(T_world_robot.wxyz_xyz[4:7])
            wxyz = onp.array(T_world_robot.wxyz_xyz[:4])
            export_data["root_quat"][t] = onp.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
            
            # Calculate link poses
            for i, link_name in enumerate(link_names):
                T_link_base = urdf.get_transform(link_name, urdf.base_link)
                T_world_robot_mat = onp.array(T_world_robot.as_matrix())
                T_world_current = T_world_robot_mat @ T_link_base
                export_data["link_pos"][t, i] = onp.array(T_world_current[..., :3, 3])
                R = onp.array(T_world_current[..., :3, :3])
                quat = vtf.SO3.from_matrix(R).wxyz
                export_data["link_quat"][t, i] = onp.array([quat[1], quat[2], quat[3], quat[0]])
        
        save_path = osp.join(src_dir, f"retarget_poses_g1.h5")
        with h5py.File(save_path, 'w') as f:
            save_dict_to_hdf5(f, export_data)
        print(f"Saved {save_path}")
    else:
        # Save multi-person results
        save_all_results(all_person_results, src_dir, urdf, actuated_joint_names, subsample_factor)

    if vis:
        # Add playback UI.
        with server.gui.add_folder("Playback"):
            gui_timestep = server.gui.add_slider(
                "Timestep",
                min=0,
                max=max_timesteps - 1,
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

            # Frame step buttons.
            @gui_next_frame.on_click
            def _(_) -> None:
                gui_timestep.value = (gui_timestep.value + 1) % max_timesteps

            @gui_prev_frame.on_click
            def _(_) -> None:
                gui_timestep.value = (gui_timestep.value - 1) % max_timesteps
                
            # Disable frame controls when we're playing.
            @gui_playing.on_update
            def _(_) -> None:
                gui_timestep.disabled = gui_playing.value
                gui_next_frame.disabled = gui_playing.value
                gui_prev_frame.disabled = gui_playing.value

            # Set the framerate when we click one of the options.
            @gui_framerate_options.on_click
            def _(_) -> None:
                gui_framerate.value = int(gui_framerate_options.value)

            # Extract the root positions to follow with camera (use first person)
            first_person_results = all_person_results[first_person_id]
            root_positions = onp.zeros((max_timesteps, 3))
            for t in range(first_person_results["num_timesteps"]):
                T_world_robot = first_person_results["optimized_T_world_root"].wxyz_xyz[t]
                root_positions[t] = T_world_robot[4:7]
            
            # Set up camera to follow the root position
            stop_camera_follow, resume_camera_follow = viser_camera_util.setup_camera_follow(
                server=server,
                slider=gui_timestep,
                target_positions=root_positions,
                camera_distance=3.0,  # Adjust based on your scene scale
                camera_height=2.0,    # Adjust based on your scene scale
                camera_angle=-30.0,   # Look down at 30 degrees
                up_direction=(0.0, 0.0, 1.0),
                fov=45.0
            )
        with server.gui.add_folder("Camera Controls"):
            gui_play_camera_to_follow = server.gui.add_checkbox("Play Camera to Follow", initial_value=False)
            @gui_play_camera_to_follow.on_update
            def _(_) -> None:
                if stop_camera_follow is not None and resume_camera_follow is not None:
                    if gui_play_camera_to_follow.value:
                        resume_camera_follow()
                    else:
                        stop_camera_follow()

            @gui_timestep.on_update
            def _(_) -> None:
                update_cfg(gui_timestep.value)
            
        while True:
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % max_timesteps
            time.sleep(1.0 / gui_framerate.value)


        server.stop()

def main(
    src_dir: Path | None = None,
    contact_dir: Path | None = None,
    data_dir_postfix: str = "",
    pattern: Path | None = None,
    offset_factor: float = 0.0,
    start_idx: int = 0,
    end_idx: int = -1,
    vis: bool = False,
    multihuman: bool = False,
    top_k: int = 1,
    *,
    local_pose_cost_weight: float = 8.0,
    end_effector_cost_weight: float = 5.0,
    global_pose_cost_weight: float = 2.0,
    self_coll_factor_weight: float = 1.0,
    world_coll_factor_weight: float = 0.01,
    world_coll_margin: float = 0.01,
    limit_cost_factor_weight: float = 1000.0,
    smoothness_cost_factor_weight: float = 10.0,
    foot_skating_cost_weight: float = 10.0,
    ground_contact_cost_weight: float = 1.0,
    hip_yaw_cost_weight: float = 5.0,
    hip_pitch_cost_weight: float = 0.0,
    hip_roll_cost_weight: float = 0.0,
) -> None:
    """
    Process robot retargeting from human motion capture data.

    Either:
    - `src_dir` and `contact_dir` are provided, or
    - `pattern` is provided.

    You can also specify the default weights for the retargeting objective.
    
    Args:
        multihuman: Enable multi-human retargeting mode
        top_k: Number of humans to process (0 for all, sorted by average bounding box area)
    """
    urdf_path = osp.join(osp.dirname(__file__), "../assets/robot_asset/g1/g1_29dof_anneal_23dof_foot.urdf")
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    process_retargeting_partial = functools.partial(
        process_retargeting,
        urdf=urdf,
        robot=robot,
        robot_coll=robot_coll,
        offset_factor=offset_factor,
        vis=vis,
        multihuman=multihuman,
        top_k=top_k,
        local_pose_cost_weight=local_pose_cost_weight,
        end_effector_cost_weight=end_effector_cost_weight,
        global_pose_cost_weight=global_pose_cost_weight,
        self_coll_factor_weight=self_coll_factor_weight,
        world_coll_factor_weight=world_coll_factor_weight,
        world_coll_margin=world_coll_margin,
        limit_cost_factor_weight=limit_cost_factor_weight,
        smoothness_cost_factor_weight=smoothness_cost_factor_weight,
        foot_skating_cost_weight=foot_skating_cost_weight,
        ground_contact_cost_weight=ground_contact_cost_weight,
        hip_yaw_cost_weight=hip_yaw_cost_weight,
        hip_pitch_cost_weight=hip_pitch_cost_weight,
        hip_roll_cost_weight=hip_roll_cost_weight,
    )

    if pattern is None:
        assert src_dir and contact_dir, "src_dir and contact_dir must be provided when pattern is not specified."
        subsample_factor = int(str(src_dir).split("_subsample_")[1])
        process_retargeting_partial(
            src_dir=src_dir,
            contact_dir=contact_dir,
            subsample_factor=subsample_factor,
        )

    else:
        assert src_dir is None, "src_dir should be empty when pattern is provided."
        assert contact_dir is None, "contact_dir should be empty when pattern is provided."
        subsample_factor = 1

        # pattern is Path object
        src_dir_list = sorted(glob.glob(f"./demo_data/output_calib_mesh/megahunter*{pattern.name}*"))
        assert len(src_dir_list) > 0, "No source directories found."

        if end_idx == -1:
            end_idx = len(src_dir_list)
        src_dir_list = src_dir_list[start_idx:end_idx]

        for src_dir_str in tqdm(src_dir_list):
            video_dir_basename = src_dir_str.split("megahunter_megasam_reconstruction_results_")[1].split("_cam")[0]
            print("\033[92mProcessing " + video_dir_basename + "\033[0m")
            
            # Construct corresponding contact directory path
            contact_path = Path(f"./demo_data/input_contacts{data_dir_postfix}") / Path(video_dir_basename) / "cam01"
                
            # extract the subsample factor from the src_dir
            subsample_factor = int(src_dir_str.split("_subsample_")[1])
    
            # Check if the output file already exists
            output_file = osp.join(src_dir_str, f"retarget_poses_g1.h5")
            if osp.exists(output_file):
                print(f"Skipping {src_dir_str} because output file already exists")
                continue
            
            try:
                process_retargeting_partial(
                    src_dir=Path(src_dir_str),
                    contact_dir=Path(contact_path),
                    subsample_factor=subsample_factor,
                )
            except KeyboardInterrupt:
                print(f"\033[93mKeyboardInterrupt\033[0m")
                continue
            except Exception as e:
                import pdb; pdb.set_trace()
                print(f"\033[93mError: {e}\033[0m")
                continue



if __name__ == "__main__":
    tyro.cli(main)
