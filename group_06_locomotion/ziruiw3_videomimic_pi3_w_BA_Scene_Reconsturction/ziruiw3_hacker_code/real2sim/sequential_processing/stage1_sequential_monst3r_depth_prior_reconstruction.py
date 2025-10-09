# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import os
import glob
import tyro
import torch
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np

from dust3r.model import AsymmetricCroCo3DStereo_Align3r
from get_world_env_align3r_feb26 import get_reconstructed_scene

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
        elif isinstance(value, np.ndarray):
            h5file.create_dataset(key_path, data=value)
        elif isinstance(value, (int, float, str, bytes, list, tuple)):
            h5file.attrs[key_path] = value  # Store scalars as attributes
        else:
            raise TypeError(f"Unsupported data type: {type(value)} for key {key_path}")
        
def process_video(
    input_dir: str,
    output_dir: str,
    model,
    device: str,
    start_frame: int = 0,
    end_frame: int = -1,
    frame_sample_ratio: int = 1,
    flow_loss: bool = True,
    scene_graph: str = 'swinstride',
    gsam2: bool = False,
    batchify: bool = True,
    empty_cache: bool = False,
):
    """Process a single video directory with Align3r."""
    
    # Get all images
    mono_imgpath_list = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    if len(mono_imgpath_list) == 0:
        mono_imgpath_list = sorted(glob.glob(os.path.join(input_dir, '*.png')))
        
    if len(mono_imgpath_list) == 0:
        print(f"No images found in {input_dir}")
        return

    # Handle end_frame
    if end_frame == -1: # or end_frame > len(mono_imgpath_list):
        end_frame = len(mono_imgpath_list)

    # Subsample frames
    filelist = sorted(mono_imgpath_list)[start_frame:end_frame:frame_sample_ratio]
    # end_frame = start_frame + len(filelist)

    # Set parameters
    silent = False
    image_size = 512
    schedule = 'linear'
    niter = 300
    scenegraph_type = scene_graph
    winsize = 5
    refid = 0

    # Create output directory
    model_name = "align3r"
    video_name = '_'.join(input_dir.split('/')[-2:])
    output_file = os.path.join(
        output_dir, 
        f'{model_name}_reconstruction_results_{video_name}_frame_{start_frame}_{end_frame}_subsample_{frame_sample_ratio}.h5'
    )
    save_folder = os.path.join(
        output_dir,
        f'{model_name}_reconstruction_results_{video_name}_frame_{start_frame}_{end_frame}_subsample_{frame_sample_ratio}'
    )
    os.makedirs(save_folder, exist_ok=True)

    if os.path.exists(output_file):
        print(f"Skipping {video_name} because it already exists")
        return

    print(f"\nProcessing {len(filelist)} frames from {video_name}")
    
    # Run reconstruction
    rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs, affine_matrix_list, output, dynamic_msk = \
        get_reconstructed_scene(
            model, device, silent, image_size, filelist, schedule, niter, 
            scenegraph_type, winsize, refid, flow_loss, save_folder, 
            gsam2, batchify, empty_cache
        )

    if rgbimg is None:
        print(f"Failed to process {video_name}")
        return

    # Save results
    results = {}
    for i, f in enumerate(filelist):
        img_name = os.path.basename(f)[:-4]
        results[img_name] = {
            'rgbimg': rgbimg[i],
            'intrinsic': intrinsics[i],
            'cam2world': cams2world[i],
            'pts3d': pts3d[i],
            'depths': depths[i],
            'msk': msk[i],
            'conf': confs[i],
            'affine_matrix': affine_matrix_list[i],
            'dynamic_msk': dynamic_msk[i]
        }

    total_output = {
        # "monst3r_network_output": output,
        "monst3r_ga_output": results
    }

    # Convert tensors to numpy
    def convert_to_numpy(d):
        for key, value in d.items():
            if isinstance(value, dict):
                convert_to_numpy(value)
            elif isinstance(value, torch.Tensor):
                d[key] = value.numpy()

    convert_to_numpy(total_output)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # with open(output_file, 'wb') as f:
    #     import pickle
    #     pickle.dump(total_output, f)
    # Save to HDF5
    with h5py.File(output_file, "w") as h5file:
        save_dict_to_hdf5(h5file, total_output)

    print(f"Results saved to {output_file}")
    return total_output


def main(
    video_base_dir: str = "./demo_data/input_images",  # Base directory containing video folders
    output_base_dir: str = "./demo_data/input_align3r_apr21",  # Base directory for output
    pattern: str = "",  # Pattern to filter video folders
    input_list_txt: str = "./input_list.txt",  # Path to input list txt file
    model_path: str = './assets/checkpoints/align3r_depthpro.pth',  # Path to model checkpoint
    start_frame: int = 0,  # Start frame index
    end_frame: int = -1,  # End frame index (-1 for all frames)
    frame_sample_ratio: int = 1,  # Frame sampling ratio
    flow_loss: bool = True,  # Whether to use flow loss
    scene_graph: str = 'swinstride',  # Scene graph type
    gsam2: bool = False,  # Whether to use GSam2
    batchify: bool = True,  # Whether to use batching
    empty_cache: bool = False,  # Whether to empty cache between videos
):
    """Process multiple videos with Align3r.
    
    Args:
        video_base_dir: Base directory containing video folders. Each subfolder should contain cam01, cam02, etc.
        output_base_dir: Base directory for output
        pattern: String pattern to filter video folders. Only process folders containing this pattern.
        model_path: Path to model checkpoint
        start_frame: Start frame index
        end_frame: End frame index (-1 for all frames)
        frame_sample_ratio: Frame sampling ratio
        flow_loss: Whether to use flow loss
        scene_graph: Scene graph type
        gsam2: Whether to use GSam2
        batchify: Whether to use batching
        empty_cache: Whether to empty cache between videos
    """
    # Initialize model once
    print("Loading Align3r model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AsymmetricCroCo3DStereo_Align3r.from_pretrained(model_path).to(device)
    
    # Get all video folders and filter by pattern
    if input_list_txt:
        with open(input_list_txt, 'r') as f:
            video_folders = [line.strip() for line in f.readlines()]
    else:
        video_folders = [f for f in os.listdir(video_base_dir) 
                        if os.path.isdir(os.path.join(video_base_dir, f)) 
                        and (not pattern or pattern.lower() in f.lower())]
    
    print(f"Found {len(video_folders)} video folders matching pattern '{pattern}'")
    if len(video_folders) == 0:
        print("No matching folders found!")
        return
        
    print("Processing folders:", video_folders)
    
    for video_name in tqdm(video_folders, desc="Processing videos"):
        video_dir = os.path.join(video_base_dir, video_name)
        cam_folders = [f for f in os.listdir(video_dir) if f.startswith("cam")]
        
        print(f"\nProcessing {video_name} with {len(cam_folders)} cameras")
        
        for cam in tqdm(cam_folders, desc="Processing cameras"):
            input_dir = os.path.join(video_dir, cam)
            output_dir = os.path.join(output_base_dir)
            
            print(f"\nProcessing {video_name}/{cam}")
            try:
                process_video(
                    input_dir,
                    output_dir,
                    model,
                    device,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_sample_ratio=frame_sample_ratio,
                    flow_loss=flow_loss,
                    scene_graph=scene_graph,
                    gsam2=gsam2,
                    batchify=batchify,
                    empty_cache=empty_cache
                )
            except Exception as e:
                print(f"Error processing {video_name}/{cam}: {e}")
                continue

if __name__ == "__main__":
    tyro.cli(main) 