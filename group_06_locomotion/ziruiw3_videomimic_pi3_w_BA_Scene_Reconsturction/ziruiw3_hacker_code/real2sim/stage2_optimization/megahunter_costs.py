# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
MegaHunter – cost / optimisation module
• JAX Var classes
• All residual / factor functions
• optimise_world_and_humans(...) high-level solver

JAX-based optimization costs and functions for MegaHunter
but lives in a focussed, import-only module.
"""

from __future__ import annotations

# -------------------------------------------------------------------------- #
# Std / third-party
# -------------------------------------------------------------------------- #
from typing import Tuple
import jax
import jax.numpy as jnp
import jaxlie                         # SO3 / SE3
import jaxls                          # Lie-factor-graph optimiser
import jax_dataclasses as jdc
from functools import partial

# -------------------------------------------------------------------------- #
# Project
# -------------------------------------------------------------------------- #
from utilities.smpl_jax_layer import SmplShaped
from stage2_optimization.megahunter_utils import (           # local package import
    smpl_main_body_joint_idx,
    coco_main_body_joint_names,
)

# -------------------------------------------------------------------------- #
# JAX-based optimisation variables
# -------------------------------------------------------------------------- #
class ScaleVar(
    jaxls.Var[jnp.ndarray],
    default_factory=lambda: jnp.array([1.0]),
    retract_fn=lambda val, delta: jnp.abs(val + delta),
    tangent_dim=1,
):
    """Single positive scale for the entire scene."""
    pass


class RootTranslationVar(
    jaxls.Var[jnp.ndarray],
    default_factory=lambda: jnp.zeros(3),
):
    """Residual root translation (per person-frame)."""
    pass


class RootRotationVar(
    jaxls.Var[jnp.ndarray],
    default_factory=lambda: jnp.array([1.0, 0.0, 0.0, 0.0]),
    retract_fn=lambda val, delta: (jaxlie.SO3(val) @ jaxlie.SO3.exp(delta.reshape(3))).wxyz,
    tangent_dim=3,
):
    """Residual root rotation quaternion (per person-frame)."""
    pass


class LocalPoseVar(
    jaxls.Var[jnp.ndarray],
    default_factory=lambda: jnp.array([[1.0, 0.0, 0.0, 0.0]] * 23),
    retract_fn=lambda val, delta: (jaxlie.SO3(val) @ jaxlie.SO3.exp(delta.reshape(23, 3))).wxyz,
    tangent_dim=23 * 3,
):
    """Residual per-joint orientation for the 23 SMPL joints."""
    pass


# -------------------------------------------------------------------------- #
# Residual / factor functions
# (all copied 1-to-1 from the legacy file)
# -------------------------------------------------------------------------- #
def alignment_cost(
    vals: jaxls.VarValues,
    var_scale: ScaleVar,
    var_root_t: RootTranslationVar,
    init_root_t: jnp.ndarray,
    smpl_joints: jnp.ndarray,
    pc_joints: jnp.ndarray,
    joint_conf: jnp.ndarray,
    wt: float = 1.0,
) -> jnp.ndarray:
    scale = vals[var_scale][0]
    root_t = vals[var_root_t]
    residual = (pc_joints * scale - (smpl_joints + init_root_t + root_t)) * joint_conf[:, None]
    return residual.flatten() * wt


def alignment_cost_with_residual_root_rotation(
    vals: jaxls.VarValues,
    var_scale: ScaleVar,
    var_root_t: RootTranslationVar,
    init_root_t: jnp.ndarray,
    var_root_r: RootRotationVar,
    smpl_joints: jnp.ndarray,
    pc_joints: jnp.ndarray,
    joint_conf: jnp.ndarray,
    wt: float = 1.0,
) -> jnp.ndarray:
    scale = vals[var_scale][0]
    root_t = vals[var_root_t]
    root_q = vals[var_root_r]
    T = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(root_q), init_root_t + root_t
    )
    transformed = (T @ jaxlie.SE3.from_translation(smpl_joints)).translation()
    residual = (pc_joints * scale - transformed) * joint_conf[:, None]
    return residual.flatten() * wt


def projected_alignment_cost(
    vals: jaxls.VarValues,
    var_scale: ScaleVar,
    var_root_t: RootTranslationVar,
    init_root_t: jnp.ndarray,
    extrinsic: jnp.ndarray,
    intrinsic: jnp.ndarray,
    smpl_whole: jnp.ndarray,
    pose2d: jnp.ndarray,
    pose2d_conf: jnp.ndarray,
    wt: float = 1.0,
) -> jnp.ndarray:
    scale = vals[var_scale][0]
    root_t = vals[var_root_t]
    pts = smpl_whole + init_root_t + root_t

    world2cam = jnp.linalg.inv(extrinsic)
    R_wc, t_wc = world2cam[:3, :3], world2cam[:3, 3] * scale
    proj_cam = R_wc @ pts.T + t_wc[:, None]        # (3,N)
    proj_img = intrinsic @ proj_cam
    proj_img = (proj_img[:2, :] / (1e-6 + proj_img[2:3, :])).T
    residual = (pose2d - proj_img) * pose2d_conf[:, None]
    return residual.flatten() * wt


def projected_alignment_cost_with_residual_root_rotation(
    vals: jaxls.VarValues,
    var_scale: ScaleVar,
    var_root_t: RootTranslationVar,
    init_root_t: jnp.ndarray,
    var_root_r: RootRotationVar,
    extrinsic: jnp.ndarray,
    intrinsic: jnp.ndarray,
    smpl_whole: jnp.ndarray,
    pose2d: jnp.ndarray,
    pose2d_conf: jnp.ndarray,
    wt: float = 1.0,
) -> jnp.ndarray:
    scale = vals[var_scale][0]
    root_t = vals[var_root_t]
    root_q = vals[var_root_r]

    T = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(root_q), init_root_t + root_t
    )
    pts = (T @ jaxlie.SE3.from_translation(smpl_whole)).translation()

    world2cam = jnp.linalg.inv(extrinsic)
    R_wc, t_wc = world2cam[:3, :3], world2cam[:3, 3] * scale
    proj_cam = R_wc @ pts.T + t_wc[:, None]
    proj_img = intrinsic @ proj_cam
    proj_img = (proj_img[:2, :] / (1e-6 + proj_img[2:3, :])).T
    residual = (pose2d - proj_img) * pose2d_conf[:, None]
    return residual.flatten() * wt


def local_pose_alignment_cost(
    vals: jaxls.VarValues,
    var_scale: ScaleVar,
    var_local_pose: LocalPoseVar,
    init_body_pose: jnp.ndarray,
    var_root_r: RootRotationVar | None,
    init_root_R: jnp.ndarray,
    shaped_body: SmplShaped,
    pc_joints: jnp.ndarray,
    joint_conf: jnp.ndarray,
    wt: float = 1.0,
    use_residual_root_rot: jnp.ndarray = False,
):
    scale = vals[var_scale][0]
    pc = pc_joints * jax.lax.stop_gradient(scale)

    # root-centre pc
    lhip = coco_main_body_joint_names.index("left_hip")
    rhip = coco_main_body_joint_names.index("right_hip")
    pelvis = (pc[lhip] + pc[rhip]) / 2.0
    pc -= pelvis[None, :]

    if isinstance(use_residual_root_rot, jnp.ndarray):
        R_root = jaxlie.SO3(vals[var_root_r]) @ jaxlie.SO3.from_matrix(init_root_R)
    else:
        R_root = jaxlie.SO3.from_matrix(init_root_R)

    local_quats = jaxlie.SO3(vals[var_local_pose]) @ jaxlie.SO3.from_matrix(init_body_pose)

    shaped_with_pose = shaped_body.with_pose(
        T_world_root=jaxlie.SE3.from_rotation(R_root).wxyz_xyz,
        local_quats=local_quats.wxyz,
    )

    new_idx = [i - 1 for i in smpl_main_body_joint_idx]  # model has no pelvis joint
    smpl_joints_world = shaped_with_pose.Ts_world_joint[new_idx, -3:]

    residual = (pc - smpl_joints_world) * joint_conf[:, None]
    return residual.flatten() * wt


def temporal_smoothness_cost(
    vals: jaxls.VarValues,
    var_rt_t: RootTranslationVar,
    var_rt_tp1: RootTranslationVar,
    init_rt_t: jnp.ndarray,
    init_rt_tp1: jnp.ndarray,
    wt: float = 0.1,
):
    delta = (vals[var_rt_tp1] + init_rt_tp1) - (vals[var_rt_t] + init_rt_t)
    return delta * wt


def root_rotation_regularization_cost(vals: jaxls.VarValues, var_rr: RootRotationVar, wt: float = 0.1):
    R = jaxlie.SO3(vals[var_rr]).as_matrix()
    return (R - jnp.eye(3)).flatten() * wt


def local_pose_rotation_regularization_cost(vals: jaxls.VarValues, var_lp: LocalPoseVar, wt: float = 0.1):
    R = jaxlie.SO3(vals[var_lp]).as_matrix()
    return (R - jnp.eye(3)[None, :, :]).flatten() * wt


def root_translation_regularization_cost(
    vals: jaxls.VarValues,
    var_scale: ScaleVar,
    var_rt: RootTranslationVar,
    cam_ext: jax.Array,
    wt: float = 0.1,
):
    scale = vals[var_scale][0]
    root_T = jaxlie.SE3.from_translation(vals[var_rt])
    world2cam = jaxlie.SE3.from_matrix(cam_ext).inverse()
    z = (world2cam @ root_T).translation()[2]
    return jnp.clip(-z, 0.0, 100.0) * wt


# -------------------------------------------------------------------------- #
# High-level optimiser (identical logic, only module refs changed)
# -------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=["num_iterations", "use_residual_root_rot", "use_local_pose"])
def optimize_world_and_humans(
    smpl_models_with_shape: Tuple[SmplShaped],
    smpl_joints: jax.Array,
    pc_joints: jax.Array,
    joint_conf: jax.Array,
    init_root_R: jax.Array,
    init_root_t: jax.Array,
    init_body_pose: jax.Array,
    cam_ext: jax.Array,
    cam_int: jax.Array,
    pose2d: jax.Array,
    pose2d_conf: jax.Array,
    smpl_whole: jax.Array,
    person_frame_mask: jax.Array,
    num_iterations: int = 300,
    root_3d_wt: float = 1.0,
    pose2d_wt: float = 1.0,
    smooth_wt: float = 0.1,
    rot_reg_wt: float = 1.0,
    local_pose_wt: float = 1.0,
    res_rot_smooth_wt: float = 1.0,
    local_pose_smooth_wt: float = 1.0,
    use_residual_root_rot: jdc.Static[bool] = False,
    use_local_pose: jdc.Static[bool] = False
):
    num_p, num_f, _, _ = smpl_joints.shape

    var_scale = ScaleVar(0)  # global scale
    var_rt = [[RootTranslationVar(p * num_f + f) for f in range(num_f)] for p in range(num_p)]
    var_rr = [[RootRotationVar(p * num_f + f) for f in range(num_f)] for p in range(num_p)]
    var_lp = [[LocalPoseVar(p * num_f + f) for f in range(num_f)] for p in range(num_p)]

    costs = []
    for p in range(num_p):
        for f in range(num_f):
            mask = person_frame_mask[p, f]
            if use_residual_root_rot:
                costs.append(
                    jaxls.Cost(
                        alignment_cost_with_residual_root_rotation,
                        (
                            var_scale,
                            var_rt[p][f],
                            init_root_t[p, f],
                            var_rr[p][f],
                            smpl_joints[p, f],
                            pc_joints[p, f],
                            joint_conf[p, f],
                            root_3d_wt * mask,
                        ),
                    )
                )
            else:
                costs.append(
                    jaxls.Cost(
                        alignment_cost,
                        (
                            var_scale,
                            var_rt[p][f],
                            init_root_t[p, f],
                            smpl_joints[p, f],
                            pc_joints[p, f],
                            joint_conf[p, f],
                            root_3d_wt * mask,
                        ),
                    )
                )

            # projection factors
            if use_residual_root_rot:
                costs.append(
                    jaxls.Cost(
                        projected_alignment_cost_with_residual_root_rotation,
                        (
                            var_scale,
                            var_rt[p][f],
                            init_root_t[p, f],
                            var_rr[p][f],
                            cam_ext[f],
                            cam_int[f],
                            smpl_whole[p, f],
                            pose2d[p, f],
                            pose2d_conf[p, f],
                            pose2d_wt * mask,
                        ),
                    )
                )
            else:
                costs.append(
                    jaxls.Cost(
                        projected_alignment_cost,
                        (
                            var_scale,
                            var_rt[p][f],
                            init_root_t[p, f],
                            cam_ext[f],
                            cam_int[f],
                            smpl_whole[p, f],
                            pose2d[p, f],
                            pose2d_conf[p, f],
                            pose2d_wt * mask,
                        ),
                    )
                )

    # temporal smoothness (translation)
    for p in range(num_p):
        for f in range(num_f - 1):
            w = smooth_wt * person_frame_mask[p, f] * person_frame_mask[p, f + 1]
            costs.append(
                jaxls.Cost(
                    temporal_smoothness_cost,
                    (var_rt[p][f], var_rt[p][f + 1], init_root_t[p, f], init_root_t[p, f + 1], w),
                )
            )

    # root rotation regularisation / smoothness
    if use_residual_root_rot:
        for p in range(num_p):
            for f in range(num_f):
                costs.append(
                    jaxls.Cost(root_rotation_regularization_cost, (var_rr[p][f], rot_reg_wt * person_frame_mask[p, f]))
                )
            for f in range(num_f - 1):
                w = res_rot_smooth_wt * person_frame_mask[p, f] * person_frame_mask[p, f + 1]
                costs.append(
                    jaxls.Cost(
                        lambda v, a, b: (jaxlie.SO3(v[a]).inverse() @ jaxlie.SO3(v[b])).log().flatten() * w,
                        (var_rr[p][f], var_rr[p][f + 1]),
                    )
                )

    # local pose alignment / reg / smoothness
    if use_local_pose:
        for p in range(num_p):
            for f in range(num_f):
                flag = jnp.ones(1) if use_residual_root_rot else False
                root_r = var_rr[p][f] if use_residual_root_rot else None
                costs.append(
                    jaxls.Cost(
                        local_pose_alignment_cost,
                        (
                            var_scale,
                            var_lp[p][f],
                            init_body_pose[p, f],
                            root_r,
                            init_root_R[p, f],
                            smpl_models_with_shape[p],
                            pc_joints[p, f],
                            joint_conf[p, f],
                            local_pose_wt * person_frame_mask[p, f],
                            flag,
                        ),
                    )
                )
                costs.append(
                    jaxls.Cost(
                        local_pose_rotation_regularization_cost,
                        (var_lp[p][f], rot_reg_wt * person_frame_mask[p, f]),
                    )
                )
            for f in range(num_f - 1):
                w = local_pose_smooth_wt * person_frame_mask[p, f] * person_frame_mask[p, f + 1]
                costs.append(
                    jaxls.Cost(
                        lambda v, a, b: (jaxlie.SO3(v[a]).inverse() @ jaxlie.SO3(v[b])).log().flatten() * w,
                        (var_lp[p][f], var_lp[p][f + 1]),
                    )
                )

    # build graph & solve
    variables = [var_scale]
    for p in range(num_p):
        for f in range(num_f):
            variables.append(var_rt[p][f])
            if use_residual_root_rot:
                variables.append(var_rr[p][f])
            if use_local_pose:
                variables.append(var_lp[p][f])

    graph = jaxls.LeastSquaresProblem(costs=costs, variables=variables).analyze()
    solved = graph.solve(
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(max_iterations=num_iterations),
        verbose=True,
    )

    scale_opt = solved[var_scale][0]
    rt_res = jnp.array([[solved[var_rt[p][f]] for f in range(num_f)] for p in range(num_p)])
    rr_res = (
        jnp.array([[solved[var_rr[p][f]] for f in range(num_f)] for p in range(num_p)])
        if use_residual_root_rot
        else None
    )
    lp_res = (
        jnp.array([[solved[var_lp[p][f]] for f in range(num_f)] for p in range(num_p)])
        if use_local_pose
        else None
    )
    return scale_opt, rt_res, rr_res, lp_res