# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
MegaHunter – utility module
All helper functions / constants shared by optimisation & runner code.

Nothing in this file performs optimisation; it is import-safe and has no
side-effects beyond standard imports.
"""

from __future__ import annotations

# -------------------------------------------------------------------------- #
# Standard library
# -------------------------------------------------------------------------- #
import os
import os.path as osp
import glob
import pickle
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Literal

# -------------------------------------------------------------------------- #
# Third-party
# -------------------------------------------------------------------------- #
import cv2
from PIL import Image
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree

# -------------------------------------------------------------------------- #
# Project-local (make sure project root is on PYTHONPATH)
# -------------------------------------------------------------------------- #
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in os.sys.path:
    os.sys.path.append(project_root)

# -------------------------------------------------------------------------- #
# Joint-name constants ( imported by cost / runner modules )
# -------------------------------------------------------------------------- #
from utilities.joint_names import (
    COCO_WHOLEBODY_KEYPOINTS,
    ORIGINAL_SMPLX_JOINT_NAMES,
    SMPL_45_KEYPOINTS,
    SMPL_KEYPOINTS,
)

# -------------------------------------------------------------------------- #
# Geometry / joint-set helpers
# -------------------------------------------------------------------------- #
coco_main_body_start_joint_idx = COCO_WHOLEBODY_KEYPOINTS.index("left_shoulder")
coco_main_body_end_joint_idx = COCO_WHOLEBODY_KEYPOINTS.index("right_heel")
coco_main_body_joint_idx = list(
    range(coco_main_body_start_joint_idx, coco_main_body_end_joint_idx + 1)
)
coco_main_body_joint_names = COCO_WHOLEBODY_KEYPOINTS[
    coco_main_body_start_joint_idx : coco_main_body_end_joint_idx + 1
]
smpl_main_body_joint_idx = [
    SMPL_45_KEYPOINTS.index(joint_name) for joint_name in coco_main_body_joint_names
]
assert len(smpl_main_body_joint_idx) == len(coco_main_body_joint_idx)

# COCO skeleton subset used for initial distance-based z-estimate
coco_main_body_skeleton = [
    # torso
    [5 - coco_main_body_start_joint_idx, 6 - coco_main_body_start_joint_idx],
    [5 - coco_main_body_start_joint_idx, 11 - coco_main_body_start_joint_idx],
    [6 - coco_main_body_start_joint_idx, 12 - coco_main_body_start_joint_idx],
    [11 - coco_main_body_start_joint_idx, 12 - coco_main_body_start_joint_idx],
    # left arm
    [5 - coco_main_body_start_joint_idx, 7 - coco_main_body_start_joint_idx],
    [7 - coco_main_body_start_joint_idx, 9 - coco_main_body_start_joint_idx],
    # right arm
    [6 - coco_main_body_start_joint_idx, 8 - coco_main_body_start_joint_idx],
    [8 - coco_main_body_start_joint_idx, 10 - coco_main_body_start_joint_idx],
    # left leg
    [11 - coco_main_body_start_joint_idx, 13 - coco_main_body_start_joint_idx],
    [13 - coco_main_body_start_joint_idx, 15 - coco_main_body_start_joint_idx],
    [15 - coco_main_body_start_joint_idx, 19 - coco_main_body_start_joint_idx],
    # right leg
    [12 - coco_main_body_start_joint_idx, 14 - coco_main_body_start_joint_idx],
    [14 - coco_main_body_start_joint_idx, 16 - coco_main_body_start_joint_idx],
    [16 - coco_main_body_start_joint_idx, 22 - coco_main_body_start_joint_idx],
]

# For 2-D projection loss we currently keep only the main-body joints
smpl_whole_body_joint_idx = smpl_main_body_joint_idx
coco_whole_body_joint_idx = coco_main_body_joint_idx
assert len(smpl_whole_body_joint_idx) == len(coco_whole_body_joint_idx)

# A convenience list used for foot-ground loss (unused in optimisation yet)
coco_foot_joint_idx = [
    COCO_WHOLEBODY_KEYPOINTS.index("right_heel"),
    COCO_WHOLEBODY_KEYPOINTS.index("left_heel"),
    COCO_WHOLEBODY_KEYPOINTS.index("right_big_toe"),
    COCO_WHOLEBODY_KEYPOINTS.index("left_big_toe"),
    COCO_WHOLEBODY_KEYPOINTS.index("right_small_toe"),
    COCO_WHOLEBODY_KEYPOINTS.index("left_small_toe"),
]

# -------------------------------------------------------------------------- #
# HDF5 helpers
# -------------------------------------------------------------------------- #
def load_dict_from_hdf5(h5file: h5py.File, path: str = "/") -> Dict:
    """
    Recursively load a nested dictionary from an HDF5 file.
    """
    result = {}
    for key in h5file[path].keys():
        key_path = f"{path}{key}"
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path + "/")
        else:
            result[key] = h5file[key_path][:]
    # attributes
    for attr_key, attr_value in h5file.attrs.items():
        if attr_key.startswith(path):
            result[attr_key[len(path) :] ] = attr_value
    return result


def save_dict_to_hdf5(h5file: h5py.File, dictionary: Dict, path: str = "/") -> None:
    """
    Recursively save a (potentially nested) dictionary to an HDF5 file.
    """
    for key, value in dictionary.items():
        key_path = f"{path}{key}"
        if value is None:
            continue
        if isinstance(value, dict):
            group = h5file.create_group(key_path)
            save_dict_to_hdf5(h5file, value, key_path + "/")
        elif isinstance(value, np.ndarray):
            h5file.create_dataset(key_path, data=value)
        elif isinstance(value, str):
            h5file.attrs[key_path] = value.encode("ascii", "ignore").decode("ascii")
        elif isinstance(value, (int, float, bytes, list, tuple)):
            h5file.attrs[key_path] = value
        else:
            raise TypeError(f"Unsupported data type: {type(value)} for key {key_path}")


# -------------------------------------------------------------------------- #
# Data-loader helpers (MegaSam → MegaHunter)
# -------------------------------------------------------------------------- #
def get_megahunter_init_data(results: Dict, device: str = "cuda"):
    """
    Convert the nested dictionary stored in `monst3r_ga_output`
    (a.k.a. MegaSam reconstruction result) into per-frame torch tensors.
    """
    pts3d_dict, depths_dict, im_K_dict, im_poses_dict, dynamic_masks_dict = (
        {},
        {},
        {},
        {},
        {},
    )
    frame_names = sorted(list(results.keys()))
    for frame_name in frame_names:
        pts3d_dict[frame_name] = torch.from_numpy(results[frame_name]["pts3d"]).to(
            device
        )
        depths_dict[frame_name] = torch.from_numpy(results[frame_name]["depths"]).to(
            device
        )
        im_K_dict[frame_name] = torch.from_numpy(results[frame_name]["intrinsic"]).to(
            device
        )
        im_poses_dict[frame_name] = torch.from_numpy(
            results[frame_name]["cam2world"]
        ).to(device)
        dynamic_masks_dict[frame_name] = torch.from_numpy(
            results[frame_name]["dynamic_msk"]
        ).to(device)

    return (
        frame_names,
        pts3d_dict,
        depths_dict,
        im_K_dict,
        im_poses_dict,
        dynamic_masks_dict,
    )


def get_smpl_init_data(data_path: str, frame_list: List[int]):
    """Load per-frame SMPL parameter pickles (as saved by VIMO/HMR2)."""
    smpl_params_dict: Dict[int, Dict] = {}
    for frame_idx in frame_list:
        smpl_data_path = osp.join(data_path, f"smpl_params_{frame_idx:05d}.pkl")
        if not osp.exists(smpl_data_path):
            continue
        with open(smpl_data_path, "rb") as f:
            smpl_params_dict[frame_idx] = pickle.load(f)
    return smpl_params_dict


def get_pose2d_init_data(data_path: str, frame_list: List[int]):
    """Load ViTPose 2-D keypoints JSON files."""
    pose2d_params_dict = {}
    for frame_idx in frame_list:
        json_path = osp.join(data_path, f"pose_{frame_idx:05d}.json")
        if not osp.exists(json_path):
            continue
        with open(json_path, "r") as f:
            pose2d = json.load(f)
        # convert string keys to int
        pose2d_params_dict[frame_idx] = {
            int(pid): pose2d[pid] for pid in pose2d
        }
    return pose2d_params_dict


def get_bbox_init_data(data_path: str, frame_list: List[int]):
    """Load SAM2 bbox JSON files created by `gsam2_filter.py`."""
    bbox_params_dict = {}
    for frame_idx in frame_list:
        json_path = osp.join(data_path, f"mask_{frame_idx:05d}.json")
        if not osp.exists(json_path):
            continue
        with open(json_path, "r") as f:
            meta = json.load(f)
        new_bbox_params = {}
        for pid, info in meta["labels"].items():
            xyxy = np.array([info["x1"], info["y1"], info["x2"], info["y2"]])
            if xyxy.sum() == 0:
                continue
            new_bbox_params[int(pid)] = xyxy
        bbox_params_dict[frame_idx] = new_bbox_params
    return bbox_params_dict


def get_mask_init_data(data_path: str, frame_list: List[int]):
    """Load human binary masks (saved as `.npz`)."""
    mask_params_dict = {}
    for frame_idx in frame_list:
        mask_path = osp.join(data_path, f"mask_{frame_idx:05d}.npz")
        if not osp.exists(mask_path):
            continue
        mask_params_dict[frame_idx] = np.load(mask_path)["mask"]
    return mask_params_dict


# -------------------------------------------------------------------------- #
# Numeric helpers
# -------------------------------------------------------------------------- #
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    6-D → rotation-matrix (Zhou et al. CVPR 2019)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation-matrix → 6-D (Zhou et al.)."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


# -------------------------------------------------------------------------- #
# Gaussian & bilateral filters (used for depth map noise reduction)
# -------------------------------------------------------------------------- #
def _gaussian_weight(ksize: int, sigma: float | None = None, device="cuda"):
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    x = (np.arange(ksize, dtype=np.float32) - center).astype(np.float32)
    kernel_1d = np.exp(-(x ** 2) / (2 * sigma**2))
    kernel = torch.from_numpy(kernel_1d[:, None] @ kernel_1d[None, :]).to(device)
    return kernel / kernel.sum()


def gaussian_filter(img: torch.Tensor, ksize: int, sigma: float | None = None, device="cuda"):
    kernel = _gaussian_weight(ksize, sigma, device).view(1, 1, ksize, ksize)
    kernel = kernel.repeat(img.shape[1], 1, 1, 1)
    pad = (ksize - 1) // 2
    return F.conv2d(img, kernel, stride=1, padding=pad, groups=img.shape[1])


def bilateral_filter(
    img: torch.Tensor,
    ksize: int,
    sigma_space: float | None = None,
    sigma_density: float | None = None,
    device="cuda",
):
    if sigma_space is None:
        sigma_space = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigma_density is None:
        sigma_density = sigma_space

    pad = (ksize - 1) // 2
    pad_img = F.pad(img, pad=[pad, pad, pad, pad], mode="reflect")
    pad_patches = pad_img.unfold(2, ksize, 1).unfold(3, ksize, 1)
    diff_density = pad_patches - img.unsqueeze(-1).unsqueeze(-1)
    weight_density = torch.exp(-(diff_density**2) / (2 * sigma_density**2))
    weight_density /= weight_density.sum(dim=(-1, -2), keepdim=True)

    w_space = _gaussian_weight(ksize, sigma_space, device=device)
    w_space = w_space.view(1, 1, ksize, ksize).expand_as(weight_density)
    weight = weight_density * w_space
    return (weight * pad_patches).sum(dim=(-1, -2)) / weight.sum(dim=(-1, -2))


# -------------------------------------------------------------------------- #
# Misc. helpers
# -------------------------------------------------------------------------- #
def draw_2d_keypoints(image: np.ndarray, keypoints: np.ndarray, skeleton=None, color=(255, 0, 0)):
    """
    Debug utility – draw 2-D keypoints (and optional skeleton) onto an RGB image.
    """
    img = image.copy()
    for x, y, conf in keypoints:
        if conf > 0.5:
            cv2.circle(img, (int(x), int(y)), 3, color, -1)
    if skeleton is not None:
        for pt1_idx, pt2_idx in skeleton:
            if (
                pt1_idx < len(keypoints)
                and pt2_idx < len(keypoints)
                and keypoints[pt1_idx][2] > 0.5
                and keypoints[pt2_idx][2] > 0.5
            ):
                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                cv2.line(img, pt1, pt2, color, 2)
    return img


# -------------------------------------------------------------------------- #
# Camera / point-cloud helpers
# -------------------------------------------------------------------------- #
def bilinear_interpolation(
    map_to_be_sampled: torch.Tensor,  # (N,C,H,W)
    pixels: torch.Tensor,            # (N,P,2) – xy in pixel co-ordinates
    device: str,
    mode: str = "bilinear",
):
    """Torch-grid-sample wrapper that returns (N,C,P,1)."""
    N, C, H, W = map_to_be_sampled.shape
    pixels_norm = 2 * pixels / torch.tensor([[W - 1, H - 1]], device=device) - 1
    samples = F.grid_sample(
        map_to_be_sampled,
        pixels_norm.unsqueeze(2),  # (N,P,1,2)
        mode=mode,
        align_corners=True,
    )
    return samples  # (N,C,P,1)


# -------------------------------------------------------------------------- #
# Motion-gap interpolation helper
# -------------------------------------------------------------------------- #
def interpolate_frames(
    valid_mask: np.ndarray,
    smpl_joints: np.ndarray,
    pc_joints: np.ndarray,
    joint_confidences: np.ndarray,
    init_global_orient: np.ndarray,
    init_root_trans: np.ndarray,
    smpl_betas: np.ndarray,
    smpl_body_pose: np.ndarray,
    vitpose_joints: np.ndarray,
    vitpose_confidences: np.ndarray,
    smpl_wholebody_joints: np.ndarray,
    threshold: int = 2,
    get_one_motion: bool = False,
):
    """
    Fill small (< `threshold`) gaps in per-frame data via linear interpolation.
    Larger gaps are zeroed out to exclude them from optimisation.
    """
    num_persons, num_frames = valid_mask.shape
    for p_idx in range(num_persons):
        secured_one_motion = False
        f_idx = 0
        while f_idx < num_frames:
            if valid_mask[p_idx, f_idx] == 0:  # start of a gap
                start = f_idx
                while f_idx < num_frames and valid_mask[p_idx, f_idx] == 0:
                    f_idx += 1
                end = f_idx
                gap_len = end - start
                if gap_len <= threshold:  # small – interpolate
                    for g in range(start, end):
                        l, r = start - 1, end if end < num_frames else num_frames - 1
                        if l >= 0 and r < num_frames:
                            alpha = (g - l) / (r - l)
                            smpl_joints[p_idx, g] = (1 - alpha) * smpl_joints[p_idx, l] + alpha * smpl_joints[p_idx, r]
                            pc_joints[p_idx,   g] = (1 - alpha) * pc_joints[p_idx,   l] + alpha * pc_joints[p_idx,   r]
                            joint_confidences[p_idx, g] = (1 - alpha) * joint_confidences[p_idx, l] + alpha * joint_confidences[p_idx, r]
                            init_global_orient[p_idx, g] = (1 - alpha) * init_global_orient[p_idx, l] + alpha * init_global_orient[p_idx, r]
                            init_root_trans[p_idx, g] = (1 - alpha) * init_root_trans[p_idx, l] + alpha * init_root_trans[p_idx, r]
                            smpl_betas[p_idx, g] = (1 - alpha) * smpl_betas[p_idx, l] + alpha * smpl_betas[p_idx, r]
                            smpl_body_pose[p_idx, g] = (1 - alpha) * smpl_body_pose[p_idx, l] + alpha * smpl_body_pose[p_idx, r]
                            vitpose_joints[p_idx, g] = (1 - alpha) * vitpose_joints[p_idx, l] + alpha * vitpose_joints[p_idx, r]
                            vitpose_confidences[p_idx, g] = (1 - alpha) * vitpose_confidences[p_idx, l] + alpha * vitpose_confidences[p_idx, r]
                            smpl_wholebody_joints[p_idx, g] = (1 - alpha) * smpl_wholebody_joints[p_idx, l] + alpha * smpl_wholebody_joints[p_idx, r]
                            valid_mask[p_idx, g] = 1.0
                else:  # long gap – invalidate
                    for g in range(start, end):
                        valid_mask[p_idx, g] = 0.0
                        joint_confidences[p_idx, g] = 0.0
                        smpl_joints[p_idx, g] = pc_joints[p_idx, g] = 0.0
                        init_global_orient[p_idx, g] = init_root_trans[p_idx, g] = 0.0
                        smpl_betas[p_idx, g] = smpl_body_pose[p_idx, g] = 0.0
                        vitpose_joints[p_idx, g] = vitpose_confidences[p_idx, g] = 0.0
                        smpl_wholebody_joints[p_idx, g] = 0.0

                    if get_one_motion and secured_one_motion:
                        # ignore everything after the first continuous motion
                        valid_mask[p_idx, end:] = 0.0
                        break
                f_idx = end
            else:
                f_idx += 1
                secured_one_motion = True
    return (
        valid_mask,
        smpl_joints,
        pc_joints,
        joint_confidences,
        init_global_orient,
        init_root_trans,
        smpl_betas,
        smpl_body_pose,
        vitpose_joints,
        vitpose_confidences,
        smpl_wholebody_joints,
    )