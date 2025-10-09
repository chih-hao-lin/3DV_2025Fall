# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
MegaHunter – high-level runner
This is the main MegaHunter optimization script for aligning human motion with the world environment.
Behaviour is unchanged; only imports / names were updated.

Example call (exactly the same as before):

    python -m stage2_optimization.megahunter_runner \
        --world-env-path ./demo_data/input_megasam/megasam_reconstruction_results_release_test_01_cam01_frame_10_90_subsample_3.h5 \
        --bbox-dir         ./demo_data/input_masks/release_test_01/cam01/json_data \
        --pose2d-dir       ./demo_data/input_2d_poses/release_test_01/cam01 \
        --smpl-dir         ./demo_data/input_3d_meshes/release_test_01/cam01 \
        --out-dir          ./demo_data/output_hunter_human_and_points
"""

from __future__ import annotations

# -------------------------------------------------------------------------- #
# Std-lib / typing
# -------------------------------------------------------------------------- #
import os
import os.path as osp
import glob
import pickle
import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# -------------------------------------------------------------------------- #
# Third-party
# -------------------------------------------------------------------------- #
import cv2
import h5py
import numpy as np
import torch
import smplx
import tyro
from PIL import Image
from tqdm import tqdm
from scipy.spatial import KDTree

# -------------------------------------------------------------------------- #
# Jax-related imports
# -------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import jaxlie
import jax_dataclasses as jdc

# -------------------------------------------------------------------------- #
# Project-local modules
# -------------------------------------------------------------------------- #
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utilities.smpl_jax_layer import SmplModel
from visualization.optimization_results_visualization import (
    show_points_and_keypoints,
    show_points_and_keypoints_and_smpl,
)

from stage2_optimization.megahunter_utils import (
    # I/O helpers
    load_dict_from_hdf5,
    save_dict_to_hdf5,
    get_megahunter_init_data,
    get_smpl_init_data,
    get_pose2d_init_data,
    get_bbox_init_data,
    get_mask_init_data,
    # maths / filters
    gaussian_filter,
    bilateral_filter,
    bilinear_interpolation,
    interpolate_frames,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    draw_2d_keypoints,
    # joint constants
    coco_main_body_joint_idx,
    coco_main_body_joint_names,
    smpl_main_body_joint_idx,
    coco_main_body_skeleton,
    smpl_whole_body_joint_idx,
)
from stage2_optimization.megahunter_costs import optimize_world_and_humans

# Import robust data loading functions
from stage2_optimization.megahunter_utils_robust import (
    get_smpl_init_data_robust,
    get_pose2d_init_data_robust,
    validate_multi_human_data,
    create_person_frame_mask,
    interpolate_missing_poses,
)

# -------------------------------------------------------------------------- #
# Utility functions that lived in the original runner section
# -------------------------------------------------------------------------- #
def estimate_initial_trans(
    joints3d: torch.Tensor,
    joints2d: torch.Tensor,
    focal: float,
    princpt: torch.Tensor,
    skeleton: List[List[int]],
    r_hip_idx: int,
    l_hip_idx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Approximate human depth from average bone-length ratio (Vickie's trick).
    """
    bone3d, bone2d, conf = [], [], []
    for u, v in skeleton:
        bone3d.append(torch.norm(joints3d[u] - joints3d[v]))
        bone2d.append(torch.norm(joints2d[u, :2] - joints2d[v, :2]))
        conf.append(torch.minimum(joints2d[u, 2], joints2d[v, 2]))
    mean3d = torch.stack(bone3d).mean()
    mean2d = (torch.stack(bone2d) * (torch.stack(conf) > 0)).mean()
    z = focal * (mean3d / mean2d)
    pelvis_2d = (joints2d[r_hip_idx, :2] + joints2d[l_hip_idx, :2]) / 2
    x = (pelvis_2d[0] - princpt[0]) * z / focal
    y = (pelvis_2d[1] - princpt[1]) * z / focal
    return torch.stack([x, y, z])


# -------------------------------------------------------------------------- #
# Main alignment routine (verbatim logic, comments shortened)
# -------------------------------------------------------------------------- #
def run_jax_alignment(
    world_env_path: str,
    bbox_dir: str,
    pose2d_dir: str,
    smpl_dir: str,
    out_dir: str,
    # flags
    get_one_motion: bool = True,
    use_g1_shape: bool = False,
    multihuman: bool = False,
    max_humans: int = 5,
    vis: bool = False,
    use_motion_prob: bool = False,
    num_iterations: int = 300,
    root_3d_trans_alignment_weight: float = 1.0,
    pose2d_alignment_weight: float = 500.0,
    smoothness_weight: float = 0.5,
    rot_reg_weight: float = 1.5,
    local_pose_alignment_weight: float = 0.05,
    res_rot_smooth_weight: float = 1.0,
    local_pose_smooth_weight: float = 1.0,
    gradient_thr: float = 0.01,
    joint3d_conf_threshold: float = 0.3,
    joint2d_conf_threshold: float = 0.5,
    optimize_root_rotation: bool = True,
    optimize_local_pose: bool = True,
    post_temporal_smoothing: bool = False,
    frame_missing_thr: int = 3,
    smpl_model_path: str = "./assets/body_models/smpl/SMPL_MALE.pkl",
    device: str = "cuda"
):
    """
    Main function to optimize the world environment and human positions.
    """

    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load world-environment (MegaSam / Align3r) data
    # ------------------------------------------------------------------ #
    if world_env_path.endswith(".h5"):
        with h5py.File(world_env_path, "r") as f:
            world_env = load_dict_from_hdf5(f)
    else:
        with open(world_env_path, "rb") as f:
            world_env = pickle.load(f)
    video_name = osp.basename(world_env_path).split(".h5")[0]

    monst3r_ga_output = world_env["monst3r_ga_output"]
    (
        frame_names,
        pts3d_dict,
        depths_dict,
        im_K_dict,
        im_poses_dict,
        dynamic_masks_dict,
    ) = get_megahunter_init_data(monst3r_ga_output, device=device)
    frame_list = [int(fn.split("_")[-1]) for fn in frame_names]

    # ------------------------------------------------------------------ #
    # 2. Load human-related detections (SMPL, 2-D pose, masks …)
    # ------------------------------------------------------------------ #
    # Use robust data loading functions for multi-human support
    if multihuman:
        # Load data with robust handling for missing frames
        smpl_params_dict, smpl_missing_data = get_smpl_init_data_robust(
            smpl_dir, frame_list, expected_person_ids=None, 
            fill_missing=True, verbose=True
        )
        pose2d_params_dict, pose2d_missing_data = get_pose2d_init_data_robust(
            pose2d_dir, frame_list, expected_person_ids=None,
            min_confidence=joint2d_conf_threshold, verbose=True
        )
        bbox_params_dict = get_bbox_init_data(bbox_dir, frame_list)
        
        # Log missing data summary
        if smpl_missing_data or pose2d_missing_data:
            print("\nData loading summary:")
            for pid in set(list(smpl_missing_data.keys()) + list(pose2d_missing_data.keys())):
                smpl_missing = len(smpl_missing_data.get(pid, []))
                pose2d_missing = len(pose2d_missing_data.get(pid, []))
                if smpl_missing > 0 or pose2d_missing > 0:
                    print(f"  Person {pid}: {smpl_missing} missing SMPL frames, {pose2d_missing} missing pose2D frames")
    else:
        # Use original functions for single person
        smpl_params_dict = get_smpl_init_data(smpl_dir, frame_list)
        pose2d_params_dict = get_pose2d_init_data(pose2d_dir, frame_list)
        bbox_params_dict = get_bbox_init_data(bbox_dir, frame_list)
    mask_dir = osp.join(bbox_dir, "..", "mask_data")
    mask_params_dict = get_mask_init_data(mask_dir, frame_list)

    # ------------------------------------------------------------------ #
    # 2-a. Optionally override masks with motion-probability union
    # ------------------------------------------------------------------ #
    if use_motion_prob:
        for fn in frame_names:
            motion_prob = monst3r_ga_output[fn]["motion_prob"]
            human_mask = monst3r_ga_output[fn]["dynamic_msk"]
            motion_prob = cv2.resize(motion_prob, human_mask.shape[::-1], interpolation=cv2.INTER_LINEAR)
            motion_mask = motion_prob < motion_prob.mean()
            monst3r_ga_output[fn]["dynamic_msk"] = motion_mask | human_mask

    # ------------------------------------------------------------------ #
    # 2-b. Pick the main person (simple majority vote heuristic)
    # ------------------------------------------------------------------ #
    with open(osp.join(bbox_dir, "..", "meta_data.json"), "r") as f:
        meta = json.load(f)
    if not multihuman:
        candidate_ids = [meta["largest_area_id"]]
        pid = Counter(candidate_ids).most_common(1)[0][0]
        person_ids = [pid]
        print(f"\033[95mSelected person ID: {pid}\033[0m")
    else:
        # Get all instance IDs
        all_person_ids = meta["all_instance_ids"]
        
        # If SAM2 has already calculated average areas and ranking, use it
        if "sorted_by_avg_area" in meta and max_humans > 0:
            sorted_ids = meta["sorted_by_avg_area"]
            person_ids = sorted_ids[:max_humans]
            
            print(f"Total humans detected: {len(all_person_ids)}")
            print(f"Selecting top-{max_humans} humans based on SAM2 area ranking: {person_ids}")
            
            # Print details if available
            if "avg_areas" in meta:
                for pid in person_ids:
                    avg_area = meta["avg_areas"].get(str(pid), 0)
                    frame_count = meta.get("frame_counts", {}).get(str(pid), 0)
                    print(f"  Person {pid}: avg_area={avg_area:.1f}, frames={frame_count}")
        else:
            # Fallback to custom scoring if SAM2 ranking not available
            # Load per-person statistics from SAM2 metadata
            person_stats = {}
            json_files = sorted([f for f in os.listdir(bbox_dir) if f.endswith('.json')])
            
            # Calculate persistence and average area for each person
            for pid in all_person_ids:
                frame_count = 0
                total_area = 0.0
                total_confidence = 0.0
                
                for json_file in json_files:
                    json_path = osp.join(bbox_dir, json_file)
                    with open(json_path, 'r') as f:
                        frame_data = json.load(f)
                    
                    # Check if this person exists in this frame
                    for _, obj_info in frame_data['labels'].items():
                        if int(obj_info['instance_id']) == pid:
                            frame_count += 1
                            x1, y1, x2, y2 = obj_info['x1'], obj_info['y1'], obj_info['x2'], obj_info['y2']
                            area = (x2 - x1) * (y2 - y1)
                            total_area += area
                            total_confidence += obj_info.get('score', 1.0)
                            break
                
                if frame_count > 0:
                    person_stats[pid] = {
                        'persistence': frame_count,
                        'avg_area': total_area / frame_count,
                        'total_area': total_area,
                        'avg_confidence': total_confidence / frame_count,
                        'score': frame_count * 0.5 + (total_area / frame_count) * 0.3 + (total_confidence / frame_count) * 0.2
                    }
            
            # Sort persons by combined score (persistence + area + confidence)
            sorted_persons = sorted(person_stats.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # Select top-k persons
            if max_humans > 0:
                person_ids = [pid for pid, _ in sorted_persons[:max_humans]]
            else:
                person_ids = [pid for pid, _ in sorted_persons]
            
            print(f"Total humans detected: {len(all_person_ids)}")
            print(f"Selecting top-{max_humans} humans based on persistence and visibility:")
            for i, (pid, stats) in enumerate(sorted_persons[:max_humans] if max_humans > 0 else sorted_persons):
                print(f"  Person {pid}: persistence={stats['persistence']} frames, "
                      f"avg_area={stats['avg_area']:.1f}, avg_conf={stats['avg_confidence']:.2f}, "
                      f"score={stats['score']:.2f}")
            
            if max_humans > 0 and len(sorted_persons) > max_humans:
                print(f"Skipping {len(sorted_persons) - max_humans} humans with lower scores")
    
    # Validate multi-human data if enabled
    if multihuman and len(person_ids) > 0:
        print("\nValidating multi-human data consistency...")
        valid_person_ids, valid_frames_per_person = validate_multi_human_data(
            frame_list, person_ids, 
            smpl_params_dict, pose2d_params_dict, bbox_params_dict,
            min_valid_frames_ratio=0.5, verbose=True
        )
        
        if len(valid_person_ids) < len(person_ids):
            print(f"\nWarning: Only {len(valid_person_ids)} out of {len(person_ids)} persons have sufficient valid data")
            person_ids = valid_person_ids
        
        if len(person_ids) == 0:
            raise ValueError("No persons have sufficient valid data for optimization. Please check your input data.")

    # import pdb; pdb.set_trace()
    # ------------------------------------------------------------------ #
    # 3. Pre-process & sample initial human 3-D points from ViTPose
    # In this file, we use two implementations of SMPL model:
    # 1. PyTorch implementation created with smplx api (https://github.com/hongsukchoi/smplx)
    # 2. Jax implementation (smpl_model_jax) implemented in utilities.smpl_jax_layer.py
    # The Jax implementation is used for the optimization.
    # The PyTorch implementation is used for the visualization.
    # The Jax implementation is more efficient and faster.
    # The PyTorch implementation is more flexible and easier to use for Pytorch users.
    # The Jax implementation is more robust and reliable.
    # ------------------------------------------------------------------ #
    smpl_model = smplx.create(
        model_path="./assets/body_models",
        model_type="smpl",
        gender="male",
        num_betas=10,
        batch_size=1,
    ).to(device)

    # ----- a. Gather per-frame human detections into nested dict -------- #
    human_init_per_frame: defaultdict[int, dict] = defaultdict(dict)
    for fidx in frame_list:
        if fidx not in smpl_params_dict or fidx not in pose2d_params_dict or fidx not in bbox_params_dict:
            continue
        smpl_p, pose2d_p, bbox_p = (
            smpl_params_dict[fidx],
            pose2d_params_dict[fidx],
            bbox_params_dict[fidx],
        )
        for pid_ in person_ids:
            try:
                human_init_per_frame[fidx][pid_] = dict(
                    smpl_params=smpl_p[pid_]["smpl_params"],
                    pose2d_params=pose2d_p[pid_],
                    bbox_params=bbox_p[pid_],
                )
            except KeyError:
                continue

    # ----- b.1 Affine transform 2-D detections to MegaSam input size ----- #
    # ----- b.2 Sample 3-D points from the pointcloud using 2-D detections ----- #
    # ----- b.3 Adjust the 3-D points' confidence based on depth gradient ----- #
    # ----- b.4 Adjust the 3-D points' confidence based on occlusion mask ----- #
    # ----- b.5 Adjust the 3-D points' confidence based on KD-tree ----- #
    
    affine = torch.tensor(monst3r_ga_output[frame_names[0]]["affine_matrix"], device=device, dtype=torch.float32)
    affine = torch.cat([affine, torch.tensor([[0, 0, 1]], device=device)], dim=0)
    affine_inv = torch.linalg.inv(affine)

    human_params_per_frame_per_person = defaultdict(dict)
    for fidx, fn in zip(frame_list, frame_names):
        if fidx not in human_init_per_frame:
            continue
        for pid_ in human_init_per_frame[fidx]:
            entry = human_init_per_frame[fidx][pid_]
            pose2d = torch.tensor(
                entry["pose2d_params"]["keypoints"],
                device=device,
                dtype=torch.float32,
            )
            bbox = torch.tensor(entry["bbox_params"], device=device, dtype=torch.float32)
            smpl_param_t = {
                k: torch.tensor(v, device=device, dtype=torch.float32)
                for k, v in entry["smpl_params"].items()
            }

            # Confidence on bbox = mean keypoint conf
            bbox = torch.cat([bbox[:4], pose2d[:, 2].mean()[None]])

            # transform pose2d & bbox
            pose2d_h = pose2d.T
            pose2d_tf = (affine_inv @ pose2d_h).T
            pose2d_tf[:, 2] = pose2d[:, 2]

            bb_tf = (affine_inv @ torch.tensor([[bbox[0], bbox[1], 1], [bbox[2], bbox[3], 1]], dtype=torch.float32, device=device).T).T
            bbox[:4] = bb_tf[:, :2].reshape(-1)

            # sample 3-D points
            pts3d = torch.tensor(pts3d_dict[fn], dtype=torch.float32, device=device).permute(2, 0, 1)[None]  # (1,3,H,W)
            pose2d_main = pose2d_tf[coco_main_body_joint_idx].reshape(1, -1, 3)
            pts3d = gaussian_filter(pts3d, ksize=3, sigma=1.0, device=device)
            sample3d = bilinear_interpolation(pts3d, pose2d_main[:, :, :2], device=device).squeeze().T  # (23,3)

            # confidence re-weighting (depth gradient)
            depth = torch.tensor(
                depths_dict[fn], device=device, dtype=torch.float32
            )
            dy, dx = torch.gradient(depth)
            grad_mag = torch.sqrt(dx**2 + dy**2)
            grad_mag = bilateral_filter(grad_mag[None, None], 3, 1.0, 1.0, device=device).squeeze()
            grad_sampled = bilinear_interpolation(
                grad_mag[None, None], pose2d_main[:, :, :2], device=device, mode="nearest"
            ).squeeze()
            grad_m = grad_sampled.clone()
            grad_m[grad_m > gradient_thr] = 1.0
            conf = (1 - grad_m).unsqueeze(1) * pose2d_main[0, :, 2].unsqueeze(1)

            # outlier rejection (KD-tree)
            points = sample3d.detach().cpu().numpy()
            tree = KDTree(points)
            distances, _ = tree.query(points, k=6) 
            mean_distances = distances[:, 1:].mean(axis=1)
            gradient_filter_passed_mask = (grad_sampled <= gradient_thr).detach().cpu().numpy()
            threshold = mean_distances[gradient_filter_passed_mask].mean() + 0.5 * mean_distances[gradient_filter_passed_mask].std()
            inlier_mask = mean_distances < threshold

            conf *= torch.tensor(inlier_mask, device=device, dtype=torch.float32).unsqueeze(1)

            # occlusion mask (pose inside dynamic mask)
            msk = torch.tensor(
                mask_params_dict[fidx], device=device, dtype=torch.float32
            )
            H, W = msk.shape
            msk_val = bilinear_interpolation(
                msk.reshape(1, 1, H, W).float(), pose2d_main[:, :, :2], device=device, mode="nearest"
            ).squeeze().bool()
            conf *= (~msk_val).unsqueeze(1)

            human_params_per_frame_per_person[fidx][pid_] = dict(
                pose2d=pose2d_tf,
                bbox=bbox,
                smpl_params=smpl_param_t,
                pose3d_world=torch.cat([sample3d, conf], dim=1),
            )

            # ------------------------------------------------------------------ #
            #  Visualise the affine-transformed 2-D detections (optional `--vis`)
            # ------------------------------------------------------------------ #
            if vis:
                vis_output_path = osp.join(out_dir, f"vis_output_{video_name}")
                os.makedirs(vis_output_path, exist_ok=True)

                vis_img_path = osp.join(
                    vis_output_path,
                    f"target_{video_name}_{fidx}_2d_keypoints_bbox.png",
                )

                # reuse previous frame if it exists, otherwise grab the RGB from Monst3r
                if osp.exists(vis_img_path):
                    img = cv2.imread(vis_img_path)
                else:
                    img = (
                        monst3r_ga_output[fn]["rgbimg"][:, :, ::-1]
                    ).astype(np.uint8)  # RGB ➜ BGR for OpenCV

                # draw key-points & bbox
                pose2d_np = pose2d_tf.cpu().numpy()
                bbox_np = bbox.cpu().numpy()

                img = draw_2d_keypoints(img, pose2d_np)
                img = cv2.rectangle(
                    img,
                    (int(bbox_np[0]), int(bbox_np[1])),
                    (int(bbox_np[2]), int(bbox_np[3])),
                    (0, 255, 0),
                    2,
                )
                img = cv2.putText(
                    img,
                    str(pid_),
                    (int(bbox_np[0]), int(bbox_np[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                cv2.imwrite(vis_img_path, img)
    
    if vis:
        print("Visualization of the 2D keypoints and bbox for each person are saved in ", vis_output_path)


    # ------------------------------------------------------------------ #
    # 4. Initialize the human mesh's global translation and rotation in the world coordinate system
    # ------------------------------------------------------------------ #
    # 4-a. Estimate initial root translation using Vickie's method
    # 4-b. Convert the global orientation of each frame each person to the world coordinate system
    # ------------------------------------------------------------------ #
    for frame_idx, frame_name in zip(frame_list, frame_names):
        human_params_per_person_transformed = {}
        human_params_per_person = human_params_per_frame_per_person[frame_idx]
        for person_id, person_info in human_params_per_person.items():
            smpl_params = smpl_params_dict[frame_idx][person_id]['smpl_params']

            with torch.no_grad():
                body_pose = torch.from_numpy(smpl_params['body_pose']).to(device)[None, :, :, :] # (1, 23, 3, 3)
                global_orient = torch.from_numpy(smpl_params['global_orient']).to(device)[None, :, :] # (1, 1, 3, 3)
                betas = torch.from_numpy(smpl_params['betas']).to(device)[None, :] # (1, 10)
            
                smpl_output = smpl_model(body_pose=body_pose, betas=betas, global_orient=global_orient, pose2rot=False)

                smpl_root_joint = smpl_output.joints[0, 0]
                smpl_joints = smpl_output.joints[0] - smpl_root_joint[None, :3]

            vitpose_2d_keypoints = human_params_per_frame_per_person[frame_idx][person_id]['pose2d'] # (J, 3)
            focal_length = im_K_dict[frame_name][0, 0]
            princpt = im_K_dict[frame_name][:2, 2]
            right_hip_idx = coco_main_body_joint_names.index('right_hip')
            left_hip_idx = coco_main_body_joint_names.index('left_hip')
            init_root_trans = estimate_initial_trans(smpl_joints[smpl_main_body_joint_idx], vitpose_2d_keypoints[coco_main_body_joint_idx], focal_length, princpt, coco_main_body_skeleton, right_hip_idx, left_hip_idx)

            # transform the init_root_trans to be in the world coordinate system
            cam2world = im_poses_dict[frame_name]
            init_root_trans_world = cam2world[:3, :3] @ init_root_trans + cam2world[:3, 3]
            smpl_params_dict[frame_idx][person_id]['smpl_params']['init_root_trans'] = init_root_trans_world # (3,)

            # transform the global orient to be in the world coordinate system
            global_orient_cam_rotmat = human_params_per_frame_per_person[frame_idx][person_id]['smpl_params']['global_orient']

            global_orient_world_rotmat = cam2world[:3, :3] @ global_orient_cam_rotmat[0]

            human_params_per_frame_per_person[frame_idx][person_id]['smpl_params']['global_orient'] = global_orient_world_rotmat[None, :, :]

    # ------------------------------------------------------------------ #
    # 5. Visualize the sampled 3D points with the pointcloud from Monst3r
    # ------------------------------------------------------------------ #
    if vis:
        world_env = monst3r_ga_output
        keypoints_3d_in_world = {}
        for frame_name in frame_names:
            frame_idx = int(frame_name.split('_')[-1])
            keypoints_3d_in_world[frame_name] = {}
            for person_id in human_params_per_frame_per_person[frame_idx].keys():
                per_frame_per_person_keypoints_3d_in_world = human_params_per_frame_per_person[frame_idx][person_id]['pose3d_world'][:, :4].cpu().numpy()
                # Only visualize the joints with confidence > 0
                # per_frame_per_person_smpl_joints_3d_in_world = per_frame_per_person_smpl_joints_3d_in_world[per_frame_per_person_smpl_joints_3d_in_world[:, 3] > 0, :3]
                keypoints_3d_in_world[frame_name][person_id] = per_frame_per_person_keypoints_3d_in_world

        try:
            try:
                show_points_and_keypoints(world_env=world_env, world_scale_factor=5., keypoints_3d_in_world=keypoints_3d_in_world)
            except Exception as e:
                print(f"Error in show_points_and_keypoints: {e}")
                import pdb; pdb.set_trace()
        except KeyboardInterrupt:
            print("KeyboardInterrupt, continuing...")
    else:
        keypoints_3d_in_world = None

    # ------------------------------------------------------------------ #
    # 6. Initialize and Optimize the Human and Scene parameters
    # ------------------------------------------------------------------ #
    # 6-a. Extract SMPL and SAM data in a format suitable for optimization
    # 6-b. Optimize the Human and Scene parameters
    # ------------------------------------------------------------------ #
    num_persons = len(person_ids)
    num_frames = len(frame_list)
    num_joints = len(coco_main_body_joint_idx)
    
    # mod num_frames to be 100
    padded_num_frames = ((num_frames - 1)// 100 + 1) * 100
    num_frames = padded_num_frames

    smpl_joints_np = np.zeros((num_persons, num_frames, num_joints, 3))
    megasam_joints_np = np.zeros((num_persons, num_frames, num_joints, 3))
    joint_confidences_np = np.zeros((num_persons, num_frames, num_joints))
    person_frame_mask_np = np.zeros((num_persons, num_frames))
    init_global_orient_np = np.zeros((num_persons, num_frames, 3, 3))
    init_root_trans_np = np.zeros((num_persons, num_frames, 3))
    smpl_betas_np = np.zeros((num_persons, num_frames, 10))
    smpl_body_pose_np = np.zeros((num_persons, num_frames, 23, 3, 3))

    # for projection loss
    camera_extrinsics_np = np.eye(4)[None, :, :].repeat(num_frames, axis=0) #np.zeros((num_frames, 4, 4))
    camera_intrinsics_np = np.eye(3)[None, :, :].repeat(num_frames, axis=0) #np.zeros((num_frames, 3, 3))
    whole_body_num_joints = len(smpl_whole_body_joint_idx) # joints including the feet

    smpl_wholebody_joints_np = np.zeros((num_persons, num_frames, whole_body_num_joints, 3))
    vitpose_joints_np = np.zeros((num_persons, num_frames, whole_body_num_joints, 2))
    vitpose_confidences_np = np.zeros((num_persons, num_frames, whole_body_num_joints))

    if use_g1_shape:
        known_betas_path = osp.join(osp.dirname(__file__), '../assets/robot_asset/g1', 'known_betas.json')
        print("Loading G1 shape from ", known_betas_path)
    else:
        known_betas_path = osp.join(smpl_dir, 'known_betas.json')
        print("Loading shape from ", known_betas_path)
    if osp.exists(known_betas_path):
        with open(known_betas_path, 'r') as f:
            known_betas = json.load(f)
            known_betas = np.array(known_betas['optimized_shape'], dtype=np.float32) # (1, 10)
        print(f"Loaded known betas from {known_betas_path}")
    else:
        known_betas = None

    # ------------------------------------------------------------------ #
    # 6-a. Extract SMPL and SAM data in a format suitable for optimization
    # ------------------------------------------------------------------ #
    # Fill in data arrays
    for p_idx, person_id in enumerate(person_ids):
        for f_idx, frame_idx in enumerate(frame_list):
            # Get 2d poses
            frame_name = frame_names[frame_list.index(frame_idx)]
            camera_extrinsics_np[f_idx] = im_poses_dict[frame_name].cpu().numpy()
            camera_intrinsics_np[f_idx] = im_K_dict[frame_name].cpu().numpy()
            

            if (frame_idx in smpl_params_dict and 
                person_id in smpl_params_dict[frame_idx] and
                frame_idx in human_params_per_frame_per_person and
                person_id in human_params_per_frame_per_person[frame_idx]):
                
                # Get SMPL joints
                smpl_params = smpl_params_dict[frame_idx][person_id]['smpl_params']

                with torch.no_grad():
                    body_pose = torch.from_numpy(smpl_params['body_pose']).to(device)[None, :, :, :] # (1, 23, 3, 3)
                    global_orient = torch.from_numpy(smpl_params['global_orient']).to(device)[None, :, :] # (1, 1, 3, 3)
                    if known_betas is not None:
                        smpl_params['betas'] = known_betas[0]

                    betas = torch.from_numpy(smpl_params['betas']).to(device)[None, :] # (1, 10)
                
                    smpl_output = smpl_model(body_pose=body_pose, betas=betas, global_orient=global_orient, pose2rot=False)

                    smpl_root_joint = smpl_output.joints[0, 0].cpu().numpy()
                    smpl_joints = smpl_output.joints[0].cpu().numpy()
                    # root joint alignment
                    smpl_joints[:, :3] -= smpl_root_joint[None, :3]
                
                pose2d_vitpose_joints = human_params_per_frame_per_person[frame_idx][person_id]['pose2d'][coco_main_body_joint_idx, :2].cpu().numpy()
                pose2d_vitpose_confidences = human_params_per_frame_per_person[frame_idx][person_id]['pose2d'][coco_main_body_joint_idx, 2].cpu().numpy()
                bbox_np = human_params_per_frame_per_person[frame_idx][person_id]['bbox'].cpu().numpy() # xyxy
                bbox_area = (bbox_np[2] - bbox_np[0]) * (bbox_np[3] - bbox_np[1])

                valid_pose2d_confidence = pose2d_vitpose_confidences > joint2d_conf_threshold
                if np.any(valid_pose2d_confidence):
                    vitpose_joints_np[p_idx, f_idx] = pose2d_vitpose_joints
                    pose2d_vitpose_confidences[pose2d_vitpose_confidences <= joint2d_conf_threshold] = 0.0
                    vitpose_confidences_np[p_idx, f_idx] = pose2d_vitpose_confidences / bbox_area # weight by bbox area
                    # person_frame_mask_np[p_idx, f_idx] = 1.0
                    smpl_wholebody_joints_np[p_idx, f_idx] = smpl_joints[smpl_whole_body_joint_idx].copy()
                    init_root_trans_np[p_idx, f_idx] = smpl_params_dict[frame_idx][person_id]['smpl_params']['init_root_trans'].cpu().numpy()

                # Get SAM joints and confidence
                human_data = human_params_per_frame_per_person[frame_idx][person_id]
                pose3d_world = human_data['pose3d_world'].cpu().numpy()
                megasam_joints = pose3d_world[:, :3]
                confidence = pose3d_world[:, 3]
                
                # Filter by confidence threshold
                valid_confidence = confidence > joint3d_conf_threshold
                
                # Only include frame if at least some joints are valid
                if np.any(valid_confidence):
                    smpl_joints_np[p_idx, f_idx] = smpl_joints[smpl_main_body_joint_idx].copy()
                    megasam_joints_np[p_idx, f_idx] = megasam_joints
                    joint_confidences_np[p_idx, f_idx] = confidence
                    person_frame_mask_np[p_idx, f_idx] = 1.0 #person_frame_mask_np[p_idx, f_idx] + 1.0
                    init_global_orient_np[p_idx, f_idx] = global_orient.squeeze(0).detach().cpu().numpy()

                    smpl_betas_np[p_idx, f_idx] = smpl_params['betas']
                    smpl_body_pose_np[p_idx, f_idx] = smpl_params['body_pose']

    # After the loops that populate smpl_joints_np, megasam_joints_np, joint_confidences_np, init_global_orient_np
    valid_mask = person_frame_mask_np.astype(np.float32)  # Create a mask of valid frames
    
    # Check for NaN values before interpolation
    if multihuman:
        print("\nChecking for NaN values in data arrays...")
        nan_issues = []
        for p_idx, person_id in enumerate(person_ids):
            # Check each data array for NaN
            if np.any(np.isnan(smpl_joints_np[p_idx])):
                nan_issues.append(f"Person {person_id}: NaN in smpl_joints")
            if np.any(np.isnan(megasam_joints_np[p_idx])):
                nan_issues.append(f"Person {person_id}: NaN in megasam_joints")
            if np.any(np.isnan(joint_confidences_np[p_idx])):
                nan_issues.append(f"Person {person_id}: NaN in joint_confidences")
            if np.any(np.isnan(init_global_orient_np[p_idx])):
                nan_issues.append(f"Person {person_id}: NaN in init_global_orient")
            if np.any(np.isnan(init_root_trans_np[p_idx])):
                nan_issues.append(f"Person {person_id}: NaN in init_root_trans")
            if np.any(np.isnan(smpl_betas_np[p_idx])):
                nan_issues.append(f"Person {person_id}: NaN in smpl_betas")
            if np.any(np.isnan(smpl_body_pose_np[p_idx])):
                nan_issues.append(f"Person {person_id}: NaN in smpl_body_pose")
        
        if nan_issues:
            print("WARNING: Found NaN values in input data:")
            for issue in nan_issues:
                print(f"  - {issue}")
            print("These will be handled by interpolation...")

    # Call the interpolation function
    valid_mask, smpl_joints_np, megasam_joints_np, joint_confidences_np, init_global_orient_np, init_root_trans_np, smpl_betas_np, smpl_body_pose_np, vitpose_joints_np, vitpose_confidences_np, smpl_wholebody_joints_np = interpolate_frames(
        valid_mask,
        smpl_joints_np,
        megasam_joints_np,
        joint_confidences_np,
        init_global_orient_np,
        init_root_trans_np,
        smpl_betas_np,
        smpl_body_pose_np,
        # for projection loss
        vitpose_joints_np,
        vitpose_confidences_np,
        smpl_wholebody_joints_np,
        # for projection loss
        threshold=frame_missing_thr,
        get_one_motion=get_one_motion  # Set your desired threshold here
    )
    person_frame_mask_np = valid_mask.astype(np.float32)
    
    # If multi-human, create proper person-frame mask
    if multihuman and len(person_ids) > 1:
        print("\nCreating person-frame mask for multi-human optimization...")
        # Use the robust function to create the mask
        if 'valid_frames_per_person' in locals():
            person_frame_mask_np = create_person_frame_mask(
                frame_list[:num_frames], person_ids, valid_frames_per_person
            )
        print(f"Person-frame mask shape: {person_frame_mask_np.shape}")
    
    # If you want to make sure the data is parsed correctly, you can uncomment the following code and add visualization for 2d joints projection.
    # sample_idx = 100
    # vitpose_sample = vitpose_joints_np[0, sample_idx]
    # megasam_joints_sample = megasam_joints_np[0, sample_idx]
    # extrinsics_sample = camera_extrinsics_np[sample_idx]
    # intrinsics_sample = camera_intrinsics_np[sample_idx]
    # world2cam_sample = np.linalg.inv(extrinsics_sample)
    # cam_joints_sample = world2cam_sample @ np.concatenate([megasam_joints_sample, np.ones_like(megasam_joints_sample)[:, :1]], axis=-1).T
    # cam_joints_sample = cam_joints_sample[:3, :].T
    # # projection
    # cam_joints_sample_proj = intrinsics_sample @ cam_joints_sample.T
    # cam_joints_sample_proj = cam_joints_sample_proj[:2, :] / cam_joints_sample_proj[2, :]
    # cam_joints_sample_proj = cam_joints_sample_proj.T
    
    print("Parsed the data...")
    print("Number of frames: ", num_frames)
    print("Number of persons: ", num_persons)
    print("Number of joints: ", num_joints)
    print("Number of wholebody joints: ", whole_body_num_joints)
    print("Start the optimization...")
    # Run the optimization
    person_frame_mask = jnp.array(person_frame_mask_np)
    smpl_joints = jnp.array(smpl_joints_np)
    megasam_joints = jnp.array(megasam_joints_np)
    joint_confidences = jnp.array(joint_confidences_np)
    init_global_orient = jnp.array(init_global_orient_np)
    init_root_trans = jnp.array(init_root_trans_np)
    smpl_body_pose = jnp.array(smpl_body_pose_np) 

    # for projection loss
    vitpose_joints = jnp.array(vitpose_joints_np)
    vitpose_confidences = jnp.array(vitpose_confidences_np)
    extrinsics = jnp.array(camera_extrinsics_np)
    intrinsics = jnp.array(camera_intrinsics_np)
    smpl_wholebody_joints = jnp.array(smpl_wholebody_joints_np)

    # Load SMPL jax model
    smpl_model_jax = SmplModel.load(Path(smpl_model_path))
    smpl_models_with_shape_jax = {}
    smpl_models_beta_mean = {}
    for p_idx, person_id in enumerate(person_ids):
        smpl_model_beta_mean = np.mean(smpl_betas_np[p_idx][valid_mask[p_idx] == 1], axis=0)
        smpl_models_with_shape_jax[person_id] = smpl_model_jax.with_shape(smpl_model_beta_mean)
        smpl_models_beta_mean[person_id] = smpl_model_beta_mean
    smpl_models_with_shape_jax_tuple = tuple(smpl_models_with_shape_jax.values())

    # ------------------------------------------------------------------ #
    # 6-b. Optimize the Human and Scene parameters
    # ------------------------------------------------------------------ #
    # optimize_scale: float scalar
    # optimized_root_residual_translations: (num_persons, num_frames, 3) # xyz in world frame
    # optimized_root_residual_rotations: (num_persons, num_frames, 3, 3)
    # optimized_local_poses: (num_persons, num_frames, 23, 3, 3)
    optimized_scale, optimized_root_residual_translations, optimized_root_residual_rotations, optimized_local_poses = optimize_world_and_humans(
        smpl_models_with_shape_jax_tuple,
        smpl_joints,
        megasam_joints,
        joint_confidences,
        init_global_orient,
        init_root_trans,
        smpl_body_pose,
        # for projection loss
        cam_ext=extrinsics,
        cam_int=intrinsics,
        pose2d=vitpose_joints,
        pose2d_conf=vitpose_confidences,
        smpl_whole=smpl_wholebody_joints,
        # for projection loss
        person_frame_mask=person_frame_mask,
        num_iterations=num_iterations,
        root_3d_wt=root_3d_trans_alignment_weight,
        pose2d_wt=pose2d_alignment_weight,
        smooth_wt=smoothness_weight,
        rot_reg_wt=rot_reg_weight,
        local_pose_wt=local_pose_alignment_weight,
        res_rot_smooth_wt=res_rot_smooth_weight,
        local_pose_smooth_wt=local_pose_smooth_weight,
        use_residual_root_rot=optimize_root_rotation,
        use_local_pose=optimize_local_pose,
    )

    # optimized_scale: scala
    # optimized_root_residual_translations: (num_persons, num_frames, 3)
    # convert to numpy array from jax.Array
    optimized_scale = optimized_scale.item() # to float
    print(f"Optimized scale: {optimized_scale}")

    optimized_root_residual_translations = np.array(optimized_root_residual_translations)
    optimized_root_residual_translations = optimized_root_residual_translations + init_root_trans_np

    if optimized_root_residual_rotations is not None:
        # convert from quaternion to rotation matrix
        optimized_root_rotations = jaxlie.SO3(optimized_root_residual_rotations) @ jaxlie.SO3.from_matrix(init_global_orient)
        optimized_root_rotations = optimized_root_rotations.as_matrix() # (1, num_frames, 3 , 3)
        optimized_root_rotations = np.array(optimized_root_rotations)
    if optimized_local_poses is not None:
        optimized_local_poses = jaxlie.SO3(optimized_local_poses) @ jaxlie.SO3.from_matrix(smpl_body_pose)
        optimized_local_poses = optimized_local_poses.as_matrix() # (num_persons, num_frames, 23, 3, 3)
        optimized_local_poses = np.array(optimized_local_poses)

    # Apply optimized scale and translations
    # Scale environment
    for frame_name in frame_names:
        pts3d_dict[frame_name] *= optimized_scale
        im_poses_dict[frame_name][:3, 3] *= optimized_scale

    # If you want to make sure the data is parsed correctly, you can uncomment the following code and add visualization for 2d joints projection.
    # sample_idx = 100
    # vitpose_sample = vitpose_joints_np[0, sample_idx]
    # smpl_joints_sample = smpl_joints_np[0, sample_idx] + optimized_root_residual_translations[0, sample_idx, None]
    # extrinsics_sample = camera_extrinsics_np[sample_idx]
    # extrinsics_sample[:3, 3] *= optimized_scale
    # intrinsics_sample = camera_intrinsics_np[sample_idx]
    # world2cam_sample = np.linalg.inv(extrinsics_sample)
    # cam_joints_sample = world2cam_sample @ np.concatenate([smpl_joints_sample, np.ones_like(smpl_joints_sample)[:, :1]], axis=-1).T
    # cam_joints_sample = cam_joints_sample[:3, :].T
    # # projection
    # cam_joints_sample_proj = intrinsics_sample @ cam_joints_sample.T
    # cam_joints_sample_proj = cam_joints_sample_proj[:2, :] / cam_joints_sample_proj[2, :]
    # cam_joints_sample_proj = cam_joints_sample_proj.T

    
    # Update human parameters with optimized translations
    for p_idx, person_id in enumerate(person_ids):
        smpl_model_beta_mean = smpl_models_beta_mean[person_id]
        for f_idx, frame_idx in enumerate(frame_list):
            if person_frame_mask_np[p_idx, f_idx]:
                if frame_idx in human_params_per_frame_per_person and \
                   person_id in human_params_per_frame_per_person[frame_idx]:
                    # Update root translation
                    human_params_per_frame_per_person[frame_idx][person_id]['smpl_params']['root_transl'] = \
                        torch.tensor(optimized_root_residual_translations[p_idx, f_idx], device='cuda').unsqueeze(0).float() # (1, 3)
                    
                    # Update beta with mean beta
                    human_params_per_frame_per_person[frame_idx][person_id]['smpl_params']['betas'] = \
                        torch.tensor(smpl_model_beta_mean, device='cuda').float() # (10)
                    

                    if optimized_root_residual_rotations is not None:
                        # Save the optimized residual root rotation
                        new_root_rot = optimized_root_rotations[p_idx, f_idx] # (3, 3)
                        human_params_per_frame_per_person[frame_idx][person_id]['smpl_params']['global_orient'] = \
                            torch.tensor(new_root_rot, device='cuda').unsqueeze(0).float() # (1, 3, 3)

                    if optimized_local_poses is not None:
                        # Save the optimized local pose
                        new_body_pose = optimized_local_poses[p_idx, f_idx] # (23, 3, 3)

                        human_params_per_frame_per_person[frame_idx][person_id]['smpl_params']['body_pose'] = \
                            torch.tensor(new_body_pose, device='cuda').float() # (23, 3, 3)
    
    # Naive One Euro Filter, SO(3) is gaurantted through 6d rot -> SO(3) matrix conversion; TODO: Use SLERP for rotations
    # Caution; assumes all human frames are present or at least consecutive
    # Temporal smoothing
    frame_sample_ratio = int(world_env_path.split('_subsample_')[-1].split('.')[0])
    freq = 30 / frame_sample_ratio

    if post_temporal_smoothing:
        from utilities.one_euro_filter import OneEuroFilter

        for p_idx, person_id in enumerate(person_ids):
            for param_name in ['body_pose', 'global_orient', 'root_transl']:
                forward_one_euro_filter = OneEuroFilter(freq=freq, mincutoff=1.5, beta=0.1, dcutoff=1.0)
                prev_param = None
                forward_param_dict = {}

                for f_idx, frame_idx in enumerate(frame_list):
                    if person_frame_mask_np[p_idx, f_idx]:
                        if frame_idx in human_params_per_frame_per_person and \
                        person_id in human_params_per_frame_per_person[frame_idx]:
                            param = human_params_per_frame_per_person[frame_idx][person_id]['smpl_params'][param_name]

                            if param_name in ['body_pose', 'global_orient']:
                                # convert to 6d rotation representation
                                param = matrix_to_rotation_6d(param[None, ...])[0]

                            if prev_param is None:
                                prev_param = param
                            else:
                                param = forward_one_euro_filter(param)
                                prev_param = param
                            forward_param_dict[frame_idx] = param
                
                backward_one_euro_filter = OneEuroFilter(freq=freq, mincutoff=1.5, beta=0.1, dcutoff=1.0)
                prev_param = None
                backward_param_dict = {}
                for rev_f_idx, frame_idx in enumerate(reversed(frame_list)):
                    if person_frame_mask_np[p_idx, len(frame_list) - 1 - rev_f_idx]:
                        if frame_idx in human_params_per_frame_per_person and \
                        person_id in human_params_per_frame_per_person[frame_idx]:
                            param = human_params_per_frame_per_person[frame_idx][person_id]['smpl_params'][param_name]

                            if param_name in ['body_pose', 'global_orient']:
                                # convert to 6d rotation representation
                                param = matrix_to_rotation_6d(param[None, ...])[0]

                            if prev_param is None:
                                prev_param = param
                            else:
                                param = backward_one_euro_filter(param)
                                prev_param = param
                            backward_param_dict[frame_idx] = param

                # udpate the parameters by averaging the forward and backward parameters
                for frame_idx in forward_param_dict.keys():
                    mean_param = (forward_param_dict[frame_idx] + backward_param_dict[frame_idx]) / 2.

                    if param_name in ['body_pose', 'global_orient']:
                        # convert back to rotation matrix
                        mean_param = rotation_6d_to_matrix(mean_param[None, ...])[0]

                    human_params_per_frame_per_person[frame_idx][person_id]['smpl_params'][param_name] = mean_param

    # Update world_env with scaled points and poses
    for frame_name in frame_names:
        monst3r_ga_output[frame_name]['pts3d'] = pts3d_dict[frame_name].cpu().numpy()
        monst3r_ga_output[frame_name]['cam2world'] = im_poses_dict[frame_name].cpu().numpy()
    
    # Visualize the MegaHunter results
    if vis:    
        world_env = monst3r_ga_output
        smpl_verts_in_world = defaultdict(dict)
        for p_idx, person_id in enumerate(person_ids):
            for f_idx, frame_idx in enumerate(frame_list):
                if person_frame_mask_np[p_idx, f_idx]:
                    if frame_idx in human_params_per_frame_per_person and \
                    person_id in human_params_per_frame_per_person[frame_idx]:
                        
                        root_rot = human_params_per_frame_per_person[frame_idx][person_id]['smpl_params']['global_orient'][None, ...]
                        body_pose = human_params_per_frame_per_person[frame_idx][person_id]['smpl_params']['body_pose'][None, ...]
                        betas = human_params_per_frame_per_person[frame_idx][person_id]['smpl_params']['betas'][None, ...]

                        # Use the rotation in the visualization
                        smpl_output = smpl_model(
                            body_pose=body_pose,
                            betas=betas,
                            global_orient=root_rot,  # Use the optimized root rotation
                            pose2rot=False
                        )
                        smpl_root_joint = smpl_output.joints[0, 0:1, :]
                        
                        try:
                            smpl_verts = smpl_output.vertices - smpl_root_joint + human_params_per_frame_per_person[frame_idx][person_id]['smpl_params']['root_transl']
                            frame_name = frame_names[frame_list.index(frame_idx)]
                            smpl_verts_in_world[frame_name][person_id] = smpl_verts.squeeze(0).detach().cpu().numpy()
                        except:
                            import pdb; pdb.set_trace()

        try:
            try:
                if keypoints_3d_in_world is not None:
                    # Apply the same scale factor as the world environment
                    for frame_name in frame_names:
                        for person_id in person_ids:
                            keypoints_3d_in_world[frame_name][person_id] *= optimized_scale

                show_points_and_keypoints_and_smpl(world_env=world_env, world_scale_factor=1., keypoints_3d_in_world=keypoints_3d_in_world, smpl_verts_in_world=smpl_verts_in_world, smpl_layer_faces=smpl_model.faces)

            except Exception as e:
                print(f"Error in show_points_and_keypoints_and_smpl: {e}")
                import pdb; pdb.set_trace()
        except KeyboardInterrupt:
            print("KeyboardInterrupt for show_points_and_keypoints_and_smpl. Continuing to save the results...")
            pass

    # Save optimized human parameters
    def convert_to_numpy(d):
        for key, value in d.items():
            if isinstance(value, dict):
                convert_to_numpy(value)
            elif isinstance(value, torch.Tensor):
                d[key] = value.detach().cpu().numpy()
    convert_to_numpy(human_params_per_frame_per_person)

    total_output = {}
    total_output['our_pred_world_cameras_and_structure'] = monst3r_ga_output

    num_persons, num_frames = person_frame_mask_np.shape
    
    # Parse the data so that each person has a list of parameters for each frame
    human_params_in_world = defaultdict(lambda: defaultdict(list))
    for p_idx, person_id in enumerate(person_ids):
        valid_frames = []
        for f_idx, frame_idx in enumerate(frame_list):
            if person_frame_mask_np[p_idx, f_idx]:
                if frame_idx in human_params_per_frame_per_person and \
                person_id in human_params_per_frame_per_person[frame_idx]:
                    valid_frames.append(frame_idx)
                
        param_keys = ['body_pose', 'global_orient', 'root_transl', 'betas']
        for frame_idx in valid_frames:
            for param_key in param_keys:
                human_params_in_world[person_id][param_key].append(
                    human_params_per_frame_per_person[frame_idx][person_id]['smpl_params'][param_key])
            
        # convert list of parameters to numpy array
        for param_key in param_keys:
            human_params_in_world[person_id][param_key] = np.array(human_params_in_world[person_id][param_key])

    total_output['our_pred_humans_smplx_params'] = human_params_in_world

    # Parse the person_frame_info_list 
    person_frame_info_list = defaultdict(list)
    for p_idx, person_id in enumerate(person_ids):
        for f_idx, frame_idx in enumerate(frame_list):
            if person_frame_mask_np[p_idx, f_idx]:
                if frame_idx in human_params_per_frame_per_person and \
                person_id in human_params_per_frame_per_person[frame_idx]:
                    frame_name = frame_names[frame_list.index(frame_idx)]
                    person_frame_info_list[person_id].append((frame_name,))
    total_output['person_frame_info_list'] = person_frame_info_list

    # split the extension of the world_env_path 
    output_name = f'megahunter_{video_name}.h5'
    
    # Save results
    output_file = os.path.join(out_dir, f"{output_name}")
    
    # Save updated environment
    with h5py.File(output_file, "w") as h5file:
        save_dict_to_hdf5(h5file, total_output)

    print(f"Results saved to {output_file}")


# -------------------------------------------------------------------------- #
# CLI entrypoint (unchanged)
# -------------------------------------------------------------------------- #
def main(
    world_env_path: str = "",
    sfm_method: str = "megasam",
    data_dir_postfix: str = "",
    bbox_dir: str = "",
    pose2d_dir: str = "",
    smpl_dir: str = "",
    out_dir: str = "./demo_data/output_megahunter_human_and_points",
    use_g1_shape: bool = False,
    get_one_motion: bool = False,
    optimize_root_rotation: bool = True,
    post_temporal_smoothing: bool = True,
    gradient_thr: float = 0.15,
    pattern: str = "",
    start_idx: int = 0,
    end_idx: int = -1,
    multihuman: bool = False,
    top_k: int = 0,  # Renamed from max_humans, 0 means all humans
    vis: bool = False,
):
    if pattern == "":
        run_jax_alignment(
            world_env_path=world_env_path,
            bbox_dir=bbox_dir,
            pose2d_dir=pose2d_dir,
            smpl_dir=smpl_dir,
            out_dir=out_dir,
            get_one_motion=get_one_motion,
            use_g1_shape=use_g1_shape,
            multihuman=multihuman,
            max_humans=top_k,  # Use top_k parameter
            optimize_root_rotation=optimize_root_rotation,
            gradient_thr=gradient_thr,
            post_temporal_smoothing=post_temporal_smoothing,
            vis=vis,
        )
    else: # pattern mode (sequential processing)
        assert world_env_path == '', "world_env_path should be empty when pattern is provided"

        if sfm_method == 'megasam':
            world_env_path_list = sorted(glob.glob(osp.join(f"./demo_data/input_megasam{data_dir_postfix}/", f"*_{pattern}*.h5")))
        elif sfm_method == 'align3r':
            world_env_path_list = sorted(glob.glob(osp.join(f"./demo_data/input_align3r{data_dir_postfix}/", f"*_{pattern}*.h5")))
            world_env_path_list += sorted(glob.glob(osp.join(f"./demo_data/input_align3r{data_dir_postfix}/", f"*_{pattern}*.pkl")))
        else:
            raise ValueError(f"Invalid sfm_method: {sfm_method}")

        if end_idx == -1:
            end_idx = len(world_env_path_list)
        world_env_path_list = world_env_path_list[start_idx:end_idx]

        print(f"Processing {len(world_env_path_list)} videos")
        for world_env_path in tqdm(world_env_path_list):
            video_dir_basename = world_env_path.split(f'{sfm_method}_reconstruction_results_')[1].split('_cam')[0]
            print("\033[92mProcessing " + video_dir_basename + "\033[0m")
            cam_name = 'cam' + world_env_path.split(f'{sfm_method}_reconstruction_results_')[1].split('_cam')[1].split('_frame')[0]
            bbox_dir = osp.join(f"./demo_data/input_masks{data_dir_postfix}", video_dir_basename, cam_name, "json_data")
            pose2d_dir = osp.join(f"./demo_data/input_2d_poses{data_dir_postfix}", video_dir_basename, cam_name)
            smpl_dir = osp.join(f"./demo_data/input_3d_meshes{data_dir_postfix}", video_dir_basename, cam_name)
        
            # # Check if the output file already exists
            # video_name = osp.basename(world_env_path).split('.h5')[0]
            # print(f"Video name: {video_name}")
            # output_name = f'megahunter_{video_name}.h5'
            # output_file = os.path.join(out_dir, f"{output_name}")
            # if os.path.exists(output_file):
            #     print(f"Skipping {world_env_path} because output file already exists")
            #     continue

            try:
                run_jax_alignment(
                    world_env_path=world_env_path,
                    bbox_dir=bbox_dir,
                    pose2d_dir=pose2d_dir,
                    smpl_dir=smpl_dir,
                    out_dir=out_dir,
                    get_one_motion=get_one_motion,
                    use_g1_shape=use_g1_shape,
                    multihuman=multihuman,
                    max_humans=top_k,  # Use top_k parameter
                    optimize_root_rotation=optimize_root_rotation,
                    gradient_thr=gradient_thr,
                    post_temporal_smoothing=post_temporal_smoothing,
                    vis=vis,
                )
            except KeyboardInterrupt:
                print(f"KeyboardInterrupt for {world_env_path}")
                continue
            except Exception as e:
                print(f"Error in run_jax_alignment for {world_env_path}: {e}")
                import pdb; pdb.set_trace()
                print("--------------------------------")
            

if __name__ == "__main__":
    tyro.cli(main)