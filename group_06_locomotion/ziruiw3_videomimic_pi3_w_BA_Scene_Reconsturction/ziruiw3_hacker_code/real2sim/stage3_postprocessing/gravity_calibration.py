# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import os   
import sys
import pickle
import numpy as np
import torch
import smplx
import tyro
import time
import h5py
from pathlib import Path
from typing import Dict, Any, Tuple

# Add the root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.joint_names import SMPL_45_KEYPOINTS


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


# def save_dict_to_hdf5(h5file: h5py.File, dictionary: Dict, path: str = "/") -> None:
#     """
#     Recursively save a (potentially nested) dictionary to an HDF5 file.
#     """
#     for key, value in dictionary.items():
#         key_path = f"{path}{key}"
#         if value is None:
#             continue
#         if isinstance(value, dict):
#             group = h5file.create_group(key_path)
#             save_dict_to_hdf5(h5file, value, key_path + "/")
#         elif isinstance(value, np.ndarray):
#             h5file.create_dataset(key_path, data=value)
#         elif isinstance(value, str):
#             h5file.attrs[key_path] = value.encode("ascii", "ignore").decode("ascii")
#         elif isinstance(value, (int, float, bytes, list, tuple)):
#             h5file.attrs[key_path] = value
#         else:
#             raise TypeError(f"Unsupported data type: {type(value)} for key {key_path}")
        
def save_dict_to_hdf5(h5file: h5py.File, dictionary: Dict, path: str = "/") -> None:
    """
    Recursively save a (potentially nested) dictionary to an HDF5 file.
    Ensures all numerical data is saved as float32.
    """
    for key, value in dictionary.items():
        key_path = f"{path}{key}"
        if value is None:
            continue
        if isinstance(value, dict):
            group = h5file.create_group(key_path)
            save_dict_to_hdf5(h5file, value, key_path + "/")
        elif isinstance(value, np.ndarray):
            # Convert numerical arrays to float32 if they contain floating point numbers
            if np.issubdtype(value.dtype, np.floating):
                value = value.astype(np.float32)
            if key_path in h5file:
                h5file[key_path][...] = value      # replace data, keep dataset
            else:
                h5file.create_dataset(key_path, data=value)
        elif isinstance(value, str):
            h5file.attrs[key_path] = value.encode("ascii", "ignore").decode("ascii")
        elif isinstance(value, (int, float, bytes, list, tuple)):
            # Convert float values to float32
            if isinstance(value, float):
                value = np.float32(value)
            h5file.attrs[key_path] = value
        else:
            raise TypeError(f"Unsupported data type: {type(value)} for key {key_path}")
        


def get_calibration_roll_pitch(image: np.ndarray, device: str) -> Tuple[float, float]:
    """
    Get roll and pitch calibration from an image using GeoCalib.
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        device: Device for computation
        
    Returns:
        Tuple of (roll_rad, pitch_rad)
    """
    from geocalib.utils import print_calibration
    from geocalib import GeoCalib
    print(f"Running GeoCalib")
    
    # Start timing for geocalib
    start_geocalib = time.time()
    
    model = GeoCalib().to(device)
    input_image = torch.tensor(image, dtype=torch.float32).to(device).permute(2, 0, 1)  # (3, H, W)
    result = model.calibrate(input_image)
    
    # End timing for geocalib
    end_geocalib = time.time()
    print(f"\033[92mGeocalib process time: {end_geocalib - start_geocalib:.2f} seconds\033[0m")
    
    camera, gravity = result["camera"], result["gravity"]
    roll_rad, pitch_rad = gravity.rp.unbind(-1)
    roll_rad = float(roll_rad.item())
    pitch_rad = float(pitch_rad.item())
    print_calibration(result)
    return roll_rad, pitch_rad


def get_world_rotation(world_env: Dict[str, Dict[str, Any]], device: str, is_megasam: bool = True) -> np.ndarray:
    """
    Gets a rotation matrix that aligns the world's z-axis to gravity.
    
    Args:
        world_env: World environment dictionary
        device: Device for computation
        is_megasam: Whether using MegaSam reconstruction
        
    Returns:
        World rotation matrix (3, 3)
    """
    first_frame = next(iter(world_env.values()))['rgbimg']  # (H, W, 3)
    if is_megasam:
        first_frame = first_frame.astype(np.float32) / 255.0
    
    roll, pitch = get_calibration_roll_pitch(first_frame, device)
    
    # Rotation matrix for pitch (around X axis)
    pitch_rotm = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ], dtype=np.float32)  # (3, 3)
    
    # Rotation matrix for roll (around Z axis) 
    roll_rotm = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0], 
        [0, 0, 1]
    ], dtype=np.float32)  # (3, 3)
    
    # Get camera rotation matrix of the first frame
    cam0_rotm = next(iter(world_env.values()))['cam2world'][:3, :3].astype(np.float32)  # (3, 3)
    # Combine rotations (roll then pitch)
    world_rotation = pitch_rotm @ roll_rotm @ cam0_rotm.T  # (3, 3)
    yup_to_zup = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ], dtype=np.float32)  # (3, 3)
    return yup_to_zup @ world_rotation  # (3, 3)


def apply_gravity_calibration(
    megahunter_path: str,
    out_dir: str,
    world_scale_factor: float = 1.0,
    gender: str = 'male',
    is_megasam: bool = True,
    multihuman: bool = False,
    vis: bool = False
) -> None:
    """
    Apply gravity calibration to megahunter optimization results and save calibrated version.
    
    Args:
        megahunter_path: Path to input megahunter file
        out_dir: Directory to save outputs
        world_scale_factor: Scale factor for world coordinates
        gender: Gender for SMPL model
        is_megasam: Whether using MegaSam reconstruction
        multihuman: Whether to process multiple humans
        vis: Whether to visualize results
    """
    # Create output directory
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if 'h5' in megahunter_path:
        print(f"Loading data from {megahunter_path}")
        with h5py.File(megahunter_path, 'r') as f:
            world_env_and_human = load_dict_from_hdf5(f)
    else:
        print(f"Loading data from {megahunter_path}")
        with open(megahunter_path, 'rb') as f:
            world_env_and_human = pickle.load(f)
    
    # close the opened h5 file
    f.close()
    
    world_env = world_env_and_human['our_pred_world_cameras_and_structure']
    human_params_in_world = world_env_and_human['our_pred_humans_smplx_params']
    
    # Support multi-human processing
    if multihuman:
        person_id_list = sorted(list(human_params_in_world.keys()))
        print(f"Processing {len(person_id_list)} humans: {person_id_list}")
    else:
        # Single person - use first person
        person_id = list(human_params_in_world.keys())[0]
        person_id_list = [person_id]
        print(f"Processing single person: {person_id}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get world rotation
    world_rotation = get_world_rotation(world_env, device, is_megasam)  # (3, 3)
    print("Computed world rotation matrix")
    
    # Process SMPL data
    print("Processing SMPL data...")
    smpl_batch_layer_dict = {}
    human_verts_in_world_dict = {}
    smpl_joints3d_in_world_dict = {}
    smpl_global_orient_in_world_dict = {}
    
    for person_id in person_id_list:
        # Create SMPL model
        smpl_batch_layer_dict[person_id] = smplx.create(
            model_path='./assets/body_models',
            model_type='smpl',
            gender=gender,
            num_betas=10,
            batch_size=len(human_params_in_world[person_id]['body_pose'])
        ).to(device)
        
        # Get SMPL parameters
        num_frames = human_params_in_world[person_id]['body_pose'].shape[0]  # (T, 23, 3, 3)
        smpl_betas = torch.from_numpy(human_params_in_world[person_id]['betas']).to(device)  # (10,) or (T, 10)
        if smpl_betas.ndim == 1:
            smpl_betas = smpl_betas.repeat(num_frames, 1)  # (T, 10)
        
        # Get SMPL outputs
        smpl_output_batch = smpl_batch_layer_dict[person_id](
            body_pose=torch.from_numpy(human_params_in_world[person_id]['body_pose']).to(device),  # (T, 23, 3, 3)
            betas=smpl_betas,  # (T, 10)
            global_orient=torch.from_numpy(human_params_in_world[person_id]['global_orient']).to(device),  # (T, 1, 3, 3)
            pose2rot=False
        )
        
        # Process joints and vertices
        smpl_joints = smpl_output_batch['joints']  # (T, 45, 3)
        smpl_root_joint = smpl_joints[:, 0:1, :]  # (T, 1, 3)
        smpl_verts = smpl_output_batch['vertices'] - smpl_root_joint + torch.from_numpy(human_params_in_world[person_id]['root_transl']).to(device)  # (T, 6890, 3)
        
        # Store rotated joints and vertices
        joints = smpl_joints.detach().cpu().numpy() - smpl_root_joint.detach().cpu().numpy() + human_params_in_world[person_id]['root_transl']  # (T, 45, 3)
        joints = (joints @ world_rotation.T * world_scale_factor).astype(np.float32)  # (T, 45, 3)
        smpl_joints3d_in_world_dict[person_id] = joints
        smpl_global_orient_in_world_dict[person_id] = (world_rotation @ human_params_in_world[person_id]['global_orient']).astype(np.float32)  # (3, 3) @ (T, 1, 3, 3) -> (T, 1, 3, 3)
        
        verts = smpl_verts.detach().cpu().numpy() @ world_rotation.T * world_scale_factor  # (T, 6890, 3)
        human_verts_in_world_dict[person_id] = verts.astype(np.float32)
    
    # Save calibrated keypoints
    keypoints_output = {
        'root_orient': smpl_global_orient_in_world_dict,
        'joints': smpl_joints3d_in_world_dict,
        'joint_names': SMPL_45_KEYPOINTS,
        'world_rotation': world_rotation # (gravity_world)T(prev_world)
    }
    
    # Save keypoints
    keypoints_path = output_dir / 'gravity_calibrated_keypoints.h5'
    with h5py.File(keypoints_path, 'w') as f:
        save_dict_to_hdf5(f, keypoints_output)
    print(f"Saved gravity calibrated keypoints to {keypoints_path}")
    
    # Update the original megahunter file with calibrated data
    updated_world_env_and_human = world_env_and_human.copy()
    
    # Apply world rotation to all point clouds in world_env
    print("Applying gravity calibration to world environment...")
    for frame_name in world_env.keys():
        pt3d = world_env[frame_name]['pts3d']  # (H, W, 3) or (N, 3)
        if pt3d.ndim == 3:
            pt3d = pt3d.reshape(-1, 3)  # (N, 3)
        updated_world_env_and_human['our_pred_world_cameras_and_structure'][frame_name]['pts3d'] = (pt3d @ world_rotation.T * world_scale_factor).astype(np.float32).reshape(world_env[frame_name]['pts3d'].shape)
        
        # Update camera poses
        cam2world = world_env[frame_name]['cam2world'].astype(np.float32)  # (4, 4)
        cam2world[:3, :3] = world_rotation @ cam2world[:3, :3]
        cam2world[:3, 3] = (world_rotation @ cam2world[:3, 3] * world_scale_factor).astype(np.float32)
        updated_world_env_and_human['our_pred_world_cameras_and_structure'][frame_name]['cam2world'] = cam2world
    
    # Update human parameters with calibrated values
    for person_id in person_id_list:
        updated_world_env_and_human['our_pred_humans_smplx_params'][person_id]['root_transl'] = smpl_joints3d_in_world_dict[person_id][:, 0:1, :].astype(np.float32)  # (T, 1, 3)
        updated_world_env_and_human['our_pred_humans_smplx_params'][person_id]['global_orient'] = smpl_global_orient_in_world_dict[person_id]  # (T, 1, 3, 3)
    
    # Save updated megahunter file
    calibrated_megahunter_path = output_dir / f"gravity_calibrated_megahunter.h5"

    with h5py.File(calibrated_megahunter_path, 'w') as f:
        save_dict_to_hdf5(f, updated_world_env_and_human)
    print(f"Saved gravity calibrated megahunter data to {calibrated_megahunter_path}")
    
    if vis:
        from visualization.gravity_calibration_visualization import visualize_gravity_calibration

        try:
            visualize_gravity_calibration(
                keypoints_output, 
                human_verts_in_world_dict, 
                smpl_batch_layer_dict, 
                person_id_list, 
                num_frames
            )
        except KeyboardInterrupt:
            print("Gravity calibration visualization interrupted by user. Continuing with the optimization process...")
            pass

def main():
    tyro.cli(apply_gravity_calibration)


if __name__ == "__main__":
    main()