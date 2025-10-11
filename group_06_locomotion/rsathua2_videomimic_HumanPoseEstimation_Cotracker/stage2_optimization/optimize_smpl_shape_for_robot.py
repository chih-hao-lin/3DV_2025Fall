# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import tyro
import viser
import os
import os.path as osp
import time
import json
from pathlib import Path
from typing import List, Tuple  

import jax
import jaxlie
import jaxls
import jaxmp
import jax_dataclasses as jdc
import jax.numpy as jnp
import numpy as onp
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

import smplx
# Import from modules with numeric prefixes using importlib
import importlib.util
import sys

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
    'pelvis_contour_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_foot_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'right_foot_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'logo_joint', 'head_joint', 'waist_support_joint', 'imu_joint', 'd435_joint', 'mid360_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'left_hand_palm_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', 'right_hand_palm_joint'
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
    # ("pelvis", "waist_yaw_joint"),
    ("left_hip", "left_hip_roll_joint"),
    ("right_hip", "right_hip_roll_joint"),
    ("left_elbow", "left_elbow_joint"),
    ("right_elbow", "right_elbow_joint"),
    ("left_knee", "left_knee_joint"),
    ("right_knee", "right_knee_joint"),
    ("left_wrist", "left_hand_palm_joint"),
    ("right_wrist", "right_hand_palm_joint"),
    ("left_ankle", "left_ankle_pitch_joint"),
    ("right_ankle", "right_ankle_pitch_joint"),
    ("left_shoulder", "left_shoulder_roll_joint"),
    ("right_shoulder", "right_shoulder_roll_joint"),
    ("left_foot", "left_foot_joint"),
    ("right_foot", "right_foot_joint"),
    ("head", "mid360_joint"),

]:
    smpl_joint_retarget_indices_to_g1.append(smpl_joint_names.index(smpl_name))
    g1_joint_retarget_indices.append(g1_joint_names.index(g1_name))
    

class SmplShapeVar(
    jaxls.Var[jnp.ndarray], 
    default_factory=lambda: jnp.zeros(10),
):
    pass

def retargeting_cost(
    var_values: jaxls.VarValues,
    # var_T_world_root: jaxls.SE3Var,
    init_world_T_robot: jaxlie.SE3,
    var_robot_cfg: jaxls.Var[jax.Array],
    var_smpl_shape: SmplShapeVar,
    kin: jaxmp.JaxKinTree,
    smpl_model_jax: SmplModel,
    # smpl_mask: jax.Array,
    robot_name: str = "h1",
) -> jax.Array:
    robot_cfg = var_values[var_robot_cfg]
    robot_root_T_robot_joints = kin.forward_kinematics(cfg=robot_cfg)


    world_T_robot_joints = init_world_T_robot @ jaxlie.SE3(robot_root_T_robot_joints)
    world_T_robot_joints = world_T_robot_joints.wxyz_xyz

    # world_T_robot_root = var_values[var_T_world_root]
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

    if robot_name == "h1":
        target_smpl_joint_pos = world_T_smpl_joints[..., 4:7][jnp.array(smpl_joint_retarget_indices_to_h1)]
        target_robot_joint_pos = world_T_robot_joints[..., 4:7][jnp.array(h1_joint_retarget_indices)]
    elif robot_name == "g1":
        target_smpl_joint_pos = world_T_smpl_joints[..., 4:7][jnp.array(smpl_joint_retarget_indices_to_g1)]
        target_robot_joint_pos = world_T_robot_joints[..., 4:7][jnp.array(g1_joint_retarget_indices)]

    # This should be ... a very soft loss to try to align the pointclouds. - CMK
    generally_align_pointclouds = (target_smpl_joint_pos - target_robot_joint_pos).flatten() * 1.

    if robot_name == "g1":
        # manually increase the weight of the head joint position matching loss
        smpl_head_pos = world_T_smpl_joints[..., 4:7][smpl_joint_names.index("head")] + jnp.array([0.0, 0.08, 0.0])
        robot_head_pos = world_T_robot_joints[..., 4:7][g1_joint_names.index("mid360_joint")]
        head_position_delta = (smpl_head_pos - robot_head_pos) * 10.
        generally_align_pointclouds = jnp.concatenate([generally_align_pointclouds, head_position_delta])

        # manually increase the weight of left hip and right hip joint position matching loss
        smpl_left_hip_pos = world_T_smpl_joints[..., 4:7][smpl_joint_names.index("left_hip")]
        smpl_right_hip_pos = world_T_smpl_joints[..., 4:7][smpl_joint_names.index("right_hip")]
        robot_left_hip_pos = world_T_robot_joints[..., 4:7][g1_joint_names.index("left_hip_roll_joint")]
        robot_right_hip_pos = world_T_robot_joints[..., 4:7][g1_joint_names.index("right_hip_roll_joint")]
        left_hip_position_delta = (smpl_left_hip_pos - robot_left_hip_pos) * 5.
        right_hip_position_delta = (smpl_right_hip_pos - robot_right_hip_pos) * 5.
        generally_align_pointclouds = jnp.concatenate([generally_align_pointclouds, left_hip_position_delta, right_hip_position_delta])

    def smpl_joint_symmetry_cost_func(
        smpl_joints: jnp.ndarray
    ):
        left_smpl_joints_vector = jnp.abs(smpl_joints[left_smpl_joint_indices])
        right_smpl_joints_vector = jnp.abs(smpl_joints[right_smpl_joint_indices])
        residual = left_smpl_joints_vector - right_smpl_joints_vector
        return residual.flatten()
    
    smpl_joint_symmetry_cost = smpl_joint_symmetry_cost_func(world_T_smpl_joints[..., 4:7]) * 5.0

    residual = jnp.concatenate([generally_align_pointclouds, smpl_joint_symmetry_cost])
    
    return residual

    # this should be only within the kinematic chain
    # Get the distance betwen the pairs of points. 
    robot_joint_position_delta = target_robot_joint_pos[:, None] - target_robot_joint_pos[None, :]
    smpl_joint_position_delta = target_smpl_joint_pos[:, None] - target_smpl_joint_pos[None, :]
    residual_position_delta = (
        (smpl_joint_position_delta - robot_joint_position_delta)
        * (1 - jnp.eye(robot_joint_position_delta.shape[0])[..., None])
        #   * smpl_mask[..., None]
    )
    residual_position_delta = residual_position_delta.flatten() * 2

    # only within the kinematic chain
    # We want to conserve the relative _angles_ between the joints, using cosine similarity.
    residual_angle_delta = 1 - (robot_joint_position_delta * smpl_joint_position_delta).sum(axis=-1)
    residual_angle_delta = (
        residual_angle_delta * (1 - jnp.eye(residual_angle_delta.shape[0])) #* smpl_mask
    )
    residual_angle_delta = residual_angle_delta.flatten() * 1
    residual = jnp.concatenate([generally_align_pointclouds, residual_position_delta, residual_angle_delta])
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
) -> jax.Array:
    smpl_shape = var_values[var_smpl_shape]

    # penalize if the absoluste value of the shape is greater than 3
    beta_abs = jnp.abs(smpl_shape)
    beta_abs_clipped = jnp.clip(beta_abs - 2.0, 0.0)

    return beta_abs_clipped * 0.1


@jdc.jit
def optimize_smpl_shape_and_robot_pose(
        urdf_jax: jaxmp.JaxKinTree,
        RobotStateVar: jdc.Static[type[jaxls.Var[jax.Array]]],
        smpl_model_jax: SmplModel,
        init_world_T_robot: jaxlie.SE3,
        default_robot_cfg: jax.Array,
        left_joint_indices: jax.Array,
        right_joint_indices: jax.Array,
        robot_name: jdc.Static[str] = "h1",
):
    # var_T_world_root = jaxls.SE3Var(0)
    var_robot_cfg = RobotStateVar(0)

    var_smpl_shape = SmplShapeVar(0)

    # if robot_name == "h1":
    #     smpl_mask = make_smpl_connection_matrix(urdf_jax, h1_joint_retarget_indices)
    # elif robot_name == "g1":
    #     smpl_mask = make_smpl_connection_matrix(urdf_jax, g1_joint_retarget_indices)
    
    costs = []
    joint_fitting_cost = lambda vals, T, cfg, shape, kin, smpl_model: retargeting_cost(vals, T, cfg, shape, kin, smpl_model, robot_name)
    costs.append(
        jaxls.Factor(
                # Wrap the retargeting cost in a lambda to capture robot_name statically.
                compute_residual=joint_fitting_cost,
                args=(
                    # var_T_world_root,
                    init_world_T_robot,
                    var_robot_cfg,
                    var_smpl_shape,
                    urdf_jax,
                    smpl_model_jax,
                ),
            )
    )
    robot_symmetry_cost = lambda vals, cfg, left_joint_indices, right_joint_indices: robot_cfg_symmetry_cost(vals, cfg, left_joint_indices, right_joint_indices)
    costs.append(
        jaxls.Factor(
            compute_residual=robot_symmetry_cost,
            args=(
                var_robot_cfg,
                left_joint_indices,
                right_joint_indices,
            ),
        )
    )
    smpl_shape_reg_cost = lambda vals, shape: smpl_shape_regularization_cost(vals, shape)
    costs.append(
        jaxls.Factor(
            compute_residual=smpl_shape_reg_cost,
            args=(var_smpl_shape,),
        )
    )

    graph = jaxls.FactorGraph.make(
        factors=costs,
        variables=[
            # var_T_world_root,
            var_robot_cfg,
            var_smpl_shape,
        ],
        use_onp=False
    )

    solved_values = graph.solve(
        initial_vals=jaxls.VarValues.make(
            [
                # var_T_world_root.with_value(init_T_world_robot),
                var_robot_cfg.with_value(default_robot_cfg),
                var_smpl_shape.with_value(jnp.zeros(10)),
            ]
        ),
        linear_solver="dense_cholesky"
        
    )
    jax.block_until_ready(solved_values)

    # Extract the optimized values
    optimized_smpl_shape = solved_values[var_smpl_shape]
    optimized_robot_cfg = solved_values[var_robot_cfg]

    return optimized_smpl_shape, optimized_robot_cfg

def main(
    robot_name: str = "g1",
    output_dir: str = "./robot_asset/g1",
    vis: bool = False,
):
    
    num_timesteps = 100
    subsample_factor = 1
    smpl_model_path = osp.join(osp.dirname(__file__), "../assets/body_models/smpl/SMPL_MALE.pkl")
    device = 'cpu'

    # Create Robot Model
    # Load URDF.

    if robot_name == "g1":
        # urdf = load_robot_description("g1_description")
        urdf_path = osp.join(osp.dirname(__file__), "../assets/robot_asset/g1/g1_29dof_anneal_23dof_foot.urdf")
        urdf = yourdfpy.URDF.load(urdf_path)
    
    elif robot_name == "h1":
        raise ValueError("H1 robot is not supported yet.")
        # urdf = load_robot_description("h1_description")
    else:
        raise ValueError(f"Robot name {robot_name} not supported")

    urdf_jax = jaxmp.JaxKinTree.from_urdf(urdf)
    RobotStateVar = jaxmp.RobotFactors.get_var_class(kin=urdf_jax)
    actuated_joint_names = urdf.actuated_joint_names
    left_joint_indices = jnp.array([idx for idx, name in enumerate(actuated_joint_names) if name.startswith("left_")])
    right_joint_indices = jnp.array([idx for idx, name in enumerate(actuated_joint_names) if name.startswith("right_")])
    # check the joint names
    print("Left joint names:", [actuated_joint_names[i] for i in left_joint_indices])
    print("Right joint names:", [actuated_joint_names[i] for i in right_joint_indices])
    print("Number of actuated joints:", len(actuated_joint_names))

    # Create rotation of 90 degrees around Y and Z axes
    rot_y_90 = jaxlie.SO3.from_matrix(onp.array([
        [0, 0, -1],  # -90 deg around Y
        [0, 1, 0],
        [1, 0, 0]
    ]))
    
    rot_z_90 = jaxlie.SO3.from_matrix(onp.array([
        [0, 1, 0],  # -90 deg around Z
        [-1, 0, 0],
        [0, 0, 1]
    ]))
    # Combine rotations
    T_world_robot_SO3 = rot_z_90 @ rot_y_90
    init_T_world_robot = jaxlie.SE3.from_rotation(T_world_robot_SO3)
   
    
    T_world_robot = init_T_world_robot
    default_cfg = (urdf_jax.limits_upper + urdf_jax.limits_lower) / 2

    """
    # Check deafult shape and pose
    # Get kinematic tree in CMK JAX format.
    urdf_jax = jaxmp.JaxKinTree.from_urdf(urdf)
    print(urdf_jax.joint_names)
    print(len(urdf_jax.joint_names))
    print("Number of actuated joints:", urdf_jax.num_actuated_joints)
    # RobotStateVar = jaxmp.RobotFactors.get_var_class(kin=urdf_jax)
    default_cfg = (urdf_jax.limits_upper + urdf_jax.limits_lower) / 2

    # Create SMPL Model
    # Load SMPL jax model
    default_betas_jax = jax.numpy.zeros(10)
    default_local_pose_jax = jaxlie.SO3.identity((23,))
    default_global_orient_jax = jaxlie.SE3.identity()

    smpl_model_jax = SmplModel.load(Path(smpl_model_path))
    smpl_models_with_shape_jax = smpl_model_jax.with_shape(default_betas_jax)
    shaped_body_with_pose = smpl_models_with_shape_jax.with_pose(
        T_world_root=default_global_orient_jax.wxyz_xyz,
        local_quats=default_local_pose_jax.wxyz,
    )  
    smpl_mesh_jax = shaped_body_with_pose.lbs()
    smpl_mesh_verts = onp.array(smpl_mesh_jax.verts)
    smpl_mesh_faces = onp.array(smpl_mesh_jax.faces)
    """


    smpl_model_jax = SmplModel.load(Path(smpl_model_path))
    # color print
    print('\033[92m' + "Optimizing smpl shape and robot pose..." + '\033[0m')
    optimized_smpl_shape, optimized_robot_cfg = optimize_smpl_shape_and_robot_pose(
        urdf_jax, 
        RobotStateVar, 
        smpl_model_jax, 
        init_T_world_robot,
        default_cfg,
        left_joint_indices,
        right_joint_indices,
        robot_name,
    )
    # convert to numpy
    # optimized_smpl_shape = onp.array(optimized_smpl_shape)
    optimized_robot_cfg = onp.array(optimized_robot_cfg)
    # optimized_robot_cfg = default_cfg
    print('\033[92m' + "Optimized smpl shape and robot pose!" + '\033[0m')
    print("\033[92m" + "Optimized smpl shape:" + '\033[0m', optimized_smpl_shape)
    print("\033[92m" + "Optimized robot cfg:" + '\033[0m', optimized_robot_cfg)

    # save h5 file
    optimized_shape = onp.array(optimized_smpl_shape).reshape(1, 10)

    # make output_dir if not exists
    os.makedirs(output_dir, exist_ok=True)
    with open(osp.join(output_dir, 'known_betas.json'), 'w') as f:
        json.dump({"optimized_shape": optimized_shape.tolist()}, f)
    print(f"Saved {osp.join(output_dir, 'known_betas.json')}")


    if vis:
        # Decode smpl mesh
        default_global_orient_jax = jaxlie.SE3.identity()
        default_local_pose_jax = jaxlie.SO3.identity((23,))

        smpl_models_with_shape_jax = smpl_model_jax.with_shape(optimized_smpl_shape)
        shaped_body_with_pose = smpl_models_with_shape_jax.with_pose(
            T_world_root=default_global_orient_jax.wxyz_xyz,
            local_quats=default_local_pose_jax.wxyz,
        )  
        smpl_mesh_jax = shaped_body_with_pose.lbs()
        smpl_mesh_verts = onp.array(smpl_mesh_jax.verts)
        smpl_mesh_faces = onp.array(smpl_mesh_jax.faces)
        smpl_joints = onp.array(shaped_body_with_pose.Ts_world_joint[..., 4:7])


        # Start viser server.
        server = viser.ViserServer(port=8081)

        server.scene.world_axes.visible = True
        server.scene.set_up_direction("+y")

        # ---------------------------------------------------------------------
        # Distance measurement tool (borrowed from optimization_results_visualization_cleaned.py)
        # ---------------------------------------------------------------------
        control0 = server.scene.add_transform_controls(
            "/controls/0", position=(0, 0, 0), scale=0.3, visible=False
        )
        control1 = server.scene.add_transform_controls(
            "/controls/1", position=(1.5, 0, 0), scale=0.3, visible=False
        )

        show_controls = server.gui.add_checkbox("Show Distance Tool", False)
        distance_text = server.gui.add_text("Distance", initial_value="0.00m")

        @show_controls.on_update
        def _(_) -> None:
            control0.visible = show_controls.value
            control1.visible = show_controls.value

        def _update_distance() -> None:
            # Compute Euclidean distance between the two control points.
            distance = onp.linalg.norm(control0.position - control1.position)
            distance_text.value = f"{distance:.2f}m"
            # Draw/refresh a red line segment connecting the two points.
            server.scene.add_spline_catmull_rom(
                "/controls/line",
                onp.stack([control0.position, control1.position], axis=0),
                color=(255, 0, 0),
            )

        # Register callbacks so the distance updates whenever either control moves.
        control0.on_update(lambda _: _update_distance())
        control1.on_update(lambda _: _update_distance())
        
        # ---------------------------------------------------------------------
        # Robot & SMPL visualization setup
        # ---------------------------------------------------------------------
        # This takes either a yourdfpy.URDF object or a path to a .urdf file.
        robot_frame = server.scene.add_frame("/robot", axes_length=0.05)
        urdf_viser = ViserUrdf(
            server,
            urdf_or_path=urdf,
            root_node_name="/robot",
        )


        def update_cfg(t: int):
            # Update robot joints.
            robot_frame.wxyz = onp.array(T_world_robot.wxyz_xyz[:4])
            robot_frame.position = onp.array(T_world_robot.wxyz_xyz[4:7])
        
            urdf_viser.update_cfg(onp.array(optimized_robot_cfg))
            # Update smpl mesh.
            smpl_mesh_handle = server.scene.add_mesh_simple(
                    f"/smpl_mesh",
                    vertices=smpl_mesh_verts,
                    faces=smpl_mesh_faces,
                    flat_shading=False,
                    wireframe=False,
                    color=(124, 250, 250),
                )
            # Update smpl joints.
            smpl_joints_handle = server.scene.add_point_cloud(
                f"/smpl_joints",
                points=smpl_joints,
                colors=(250, 124, 250),
                point_size=0.01,
                point_shape="circle",
            )
                    
        for joint, frame_handle in zip(
            urdf_viser._urdf.joint_map.values(), urdf_viser._joint_frames
        ):
            frame_handle.show_axes = True
            frame_handle.axes_length = 0.1
            frame_handle.axes_radius = 0.01
            frame_handle.origin_radius = 0.01


        # Add a slider to visualize different timesteps
        slider = server.gui.add_slider(
            "Timestep", min=0, max=num_timesteps - 1, step=1, initial_value=0
        )
        playing = server.gui.add_checkbox("Playing", initial_value=True)

        @slider.on_update
        def _(_) -> None:
            update_cfg(slider.value)

        fps = 30.0 / subsample_factor
        while True:
            if playing.value:
                slider.value = (slider.value + 1) % num_timesteps
            time.sleep(1.0 / fps)

if __name__ == "__main__":
    tyro.cli(main)