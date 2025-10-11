# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Robust data loading utilities for multi-human MegaHunter optimization.
These functions handle missing data gracefully and provide detailed logging.
"""

import os
import os.path as osp
import json
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import warnings


def get_smpl_init_data_robust(
    data_path: str, 
    frame_list: List[int], 
    expected_person_ids: Optional[List[int]] = None,
    fill_missing: bool = True,
    verbose: bool = True
) -> Tuple[Dict, Dict[int, Set[int]]]:
    """
    Load SMPL parameters with robust handling for missing data.
    
    Args:
        data_path: Path to SMPL pkl files directory
        frame_list: List of frame indices to load
        expected_person_ids: List of person IDs expected (if None, discovers from data)
        fill_missing: Whether to fill missing frames with interpolation
        verbose: Print detailed logging
        
    Returns:
        smpl_params_dict: Dictionary of SMPL parameters
        missing_data: Dict mapping person_id to set of missing frame indices
    """
    smpl_params_dict = {}
    missing_data = defaultdict(set)
    found_person_ids = set()
    
    # First pass: load all available data
    for frame_idx in frame_list:
        pkl_path = osp.join(data_path, f"smpl_params_{frame_idx:05d}.pkl")
        
        if not osp.exists(pkl_path):
            if verbose:
                print(f"Warning: SMPL file missing for frame {frame_idx}")
            continue
            
        try:
            with open(pkl_path, 'rb') as f:
                frame_data = pickle.load(f)
            
            smpl_params_dict[frame_idx] = frame_data
            found_person_ids.update(frame_data.keys())
            
        except Exception as e:
            warnings.warn(f"Error loading SMPL data for frame {frame_idx}: {e}")
            continue
    
    # Determine which person IDs to process
    if expected_person_ids is None:
        expected_person_ids = sorted(list(found_person_ids))
    
    # Second pass: identify missing data
    for frame_idx in frame_list:
        if frame_idx not in smpl_params_dict:
            for person_id in expected_person_ids:
                missing_data[person_id].add(frame_idx)
        else:
            for person_id in expected_person_ids:
                if person_id not in smpl_params_dict[frame_idx]:
                    missing_data[person_id].add(frame_idx)
                else:
                    # Check for NaN values
                    params = smpl_params_dict[frame_idx][person_id]['smpl_params']
                    for key, val in params.items():
                        if np.any(np.isnan(val)):
                            warnings.warn(f"NaN found in frame {frame_idx}, person {person_id}, param {key}")
                            missing_data[person_id].add(frame_idx)
                            break
    
    if verbose and missing_data:
        print("\nMissing SMPL data summary:")
        for person_id, missing_frames in missing_data.items():
            print(f"  Person {person_id}: {len(missing_frames)} missing frames")
    
    # Fill missing data if requested
    if fill_missing and missing_data:
        smpl_params_dict = fill_missing_smpl_data(
            smpl_params_dict, missing_data, expected_person_ids, frame_list, verbose
        )
    
    return smpl_params_dict, dict(missing_data)


def fill_missing_smpl_data(
    smpl_params_dict: Dict,
    missing_data: Dict[int, Set[int]],
    person_ids: List[int],
    frame_list: List[int],
    verbose: bool = True
) -> Dict:
    """
    Fill missing SMPL data using nearest neighbor interpolation.
    """
    filled_dict = smpl_params_dict.copy()
    
    for person_id in person_ids:
        if person_id not in missing_data or not missing_data[person_id]:
            continue
            
        # Get all valid frames for this person
        valid_frames = []
        for frame_idx in frame_list:
            if (frame_idx in smpl_params_dict and 
                person_id in smpl_params_dict[frame_idx] and
                frame_idx not in missing_data[person_id]):
                valid_frames.append(frame_idx)
        
        if not valid_frames:
            warnings.warn(f"No valid SMPL data found for person {person_id}")
            continue
        
        # Get reference parameters from a valid frame
        ref_frame = valid_frames[0]
        ref_params = smpl_params_dict[ref_frame][person_id]['smpl_params']
        
        # Fill missing frames
        for missing_frame in missing_data[person_id]:
            # Find nearest valid frame
            nearest_frame = min(valid_frames, key=lambda x: abs(x - missing_frame))
            
            # Initialize frame dict if needed
            if missing_frame not in filled_dict:
                filled_dict[missing_frame] = {}
            
            # Copy parameters from nearest frame
            filled_dict[missing_frame][person_id] = {
                'smpl_params': {
                    key: smpl_params_dict[nearest_frame][person_id]['smpl_params'][key].copy()
                    for key in ref_params.keys()
                }
            }
            
            if verbose:
                print(f"Filled frame {missing_frame} for person {person_id} using frame {nearest_frame}")
    
    return filled_dict


def get_pose2d_init_data_robust(
    data_path: str,
    frame_list: List[int],
    expected_person_ids: Optional[List[int]] = None,
    min_confidence: float = 0.3,
    verbose: bool = True
) -> Tuple[Dict, Dict[int, Set[int]]]:
    """
    Load 2D pose data with robust handling for missing data.
    """
    pose2d_params_dict = {}
    missing_data = defaultdict(set)
    found_person_ids = set()
    
    for frame_idx in frame_list:
        json_path = osp.join(data_path, f"pose_{frame_idx:05d}.json")
        
        if not osp.exists(json_path):
            if verbose:
                print(f"Warning: Pose2D file missing for frame {frame_idx}")
            continue
            
        try:
            with open(json_path, 'r') as f:
                frame_data = json.load(f)
            
            # Convert string keys to int
            frame_data_int = {}
            for str_id, data in frame_data.items():
                person_id = int(str_id)
                frame_data_int[person_id] = data
                found_person_ids.add(person_id)
                
                # Check keypoint confidence
                keypoints = np.array(data['keypoints'])
                valid_kpts = keypoints[:, 2] >= min_confidence
                if np.sum(valid_kpts) < 5:  # Need at least 5 valid keypoints
                    warnings.warn(f"Low confidence keypoints in frame {frame_idx}, person {person_id}")
                    missing_data[person_id].add(frame_idx)
            
            pose2d_params_dict[frame_idx] = frame_data_int
            
        except Exception as e:
            warnings.warn(f"Error loading pose2D data for frame {frame_idx}: {e}")
            continue
    
    # Check for missing person data
    if expected_person_ids is None:
        expected_person_ids = sorted(list(found_person_ids))
    
    for frame_idx in frame_list:
        if frame_idx not in pose2d_params_dict:
            for person_id in expected_person_ids:
                missing_data[person_id].add(frame_idx)
        else:
            for person_id in expected_person_ids:
                if person_id not in pose2d_params_dict[frame_idx]:
                    missing_data[person_id].add(frame_idx)
    
    if verbose and missing_data:
        print("\nMissing Pose2D data summary:")
        for person_id, missing_frames in missing_data.items():
            print(f"  Person {person_id}: {len(missing_frames)} missing frames")
    
    return pose2d_params_dict, dict(missing_data)


def validate_multi_human_data(
    frame_list: List[int],
    person_ids: List[int],
    smpl_params_dict: Dict,
    pose2d_params_dict: Dict,
    bbox_params_dict: Dict,
    min_valid_frames_ratio: float = 0.5,
    verbose: bool = True
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Validate multi-human data and filter out persons with insufficient data.
    
    Returns:
        valid_person_ids: List of person IDs with sufficient data
        valid_frames_per_person: Dict mapping person_id to list of valid frame indices
    """
    valid_frames_per_person = defaultdict(list)
    
    for person_id in person_ids:
        for frame_idx in frame_list:
            # Check if all data is available for this person in this frame
            has_smpl = (frame_idx in smpl_params_dict and 
                       person_id in smpl_params_dict[frame_idx])
            has_pose2d = (frame_idx in pose2d_params_dict and 
                         person_id in pose2d_params_dict[frame_idx])
            has_bbox = (frame_idx in bbox_params_dict and 
                       person_id in bbox_params_dict[frame_idx])
            
            if has_smpl and has_pose2d and has_bbox:
                valid_frames_per_person[person_id].append(frame_idx)
    
    # Filter persons with insufficient valid frames
    valid_person_ids = []
    min_frames = int(len(frame_list) * min_valid_frames_ratio)
    
    for person_id in person_ids:
        valid_frames = len(valid_frames_per_person[person_id])
        if valid_frames >= min_frames:
            valid_person_ids.append(person_id)
        elif verbose:
            print(f"Warning: Person {person_id} has only {valid_frames}/{len(frame_list)} valid frames (below {min_valid_frames_ratio:.0%} threshold)")
    
    if verbose:
        print(f"\nData validation summary:")
        print(f"  Total frames: {len(frame_list)}")
        print(f"  Total persons: {len(person_ids)}")
        print(f"  Valid persons: {len(valid_person_ids)}")
        for person_id in valid_person_ids:
            print(f"    Person {person_id}: {len(valid_frames_per_person[person_id])} valid frames")
    
    return valid_person_ids, dict(valid_frames_per_person)


def create_person_frame_mask(
    frame_list: List[int],
    person_ids: List[int],
    valid_frames_per_person: Dict[int, List[int]]
) -> np.ndarray:
    """
    Create a binary mask indicating which person appears in which frame.
    
    Returns:
        mask: (num_persons, num_frames) binary array
    """
    num_persons = len(person_ids)
    num_frames = len(frame_list)
    mask = np.zeros((num_persons, num_frames), dtype=bool)
    
    frame_to_idx = {frame: idx for idx, frame in enumerate(frame_list)}
    
    for p_idx, person_id in enumerate(person_ids):
        valid_frames = valid_frames_per_person.get(person_id, [])
        for frame in valid_frames:
            if frame in frame_to_idx:
                mask[p_idx, frame_to_idx[frame]] = True
    
    return mask


def interpolate_missing_poses(
    poses: np.ndarray,
    valid_mask: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate missing poses for a single person.
    
    Args:
        poses: (num_frames, ...) pose parameters
        valid_mask: (num_frames,) boolean mask of valid frames
        method: Interpolation method ('linear', 'nearest', 'slerp' for rotations)
    
    Returns:
        interpolated_poses: (num_frames, ...) with missing frames filled
    """
    if np.all(~valid_mask):
        warnings.warn("No valid frames to interpolate from")
        return poses
    
    valid_indices = np.where(valid_mask)[0]
    invalid_indices = np.where(~valid_mask)[0]
    
    if len(invalid_indices) == 0:
        return poses
    
    interpolated = poses.copy()
    
    if method == 'nearest':
        for invalid_idx in invalid_indices:
            # Find nearest valid index
            distances = np.abs(valid_indices - invalid_idx)
            nearest_idx = valid_indices[np.argmin(distances)]
            interpolated[invalid_idx] = poses[nearest_idx]
    
    elif method == 'linear':
        # For each invalid index, interpolate between surrounding valid indices
        for invalid_idx in invalid_indices:
            # Find surrounding valid indices
            before_indices = valid_indices[valid_indices < invalid_idx]
            after_indices = valid_indices[valid_indices > invalid_idx]
            
            if len(before_indices) > 0 and len(after_indices) > 0:
                # Interpolate between before and after
                before_idx = before_indices[-1]
                after_idx = after_indices[0]
                
                alpha = (invalid_idx - before_idx) / (after_idx - before_idx)
                interpolated[invalid_idx] = (1 - alpha) * poses[before_idx] + alpha * poses[after_idx]
            
            elif len(before_indices) > 0:
                # Only before indices available, use nearest
                interpolated[invalid_idx] = poses[before_indices[-1]]
            
            elif len(after_indices) > 0:
                # Only after indices available, use nearest
                interpolated[invalid_idx] = poses[after_indices[0]]
    
    return interpolated