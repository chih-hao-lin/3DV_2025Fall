# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Sequential MegaSam Reconstruction for Multiple Videos

This script processes multiple videos sequentially through MegaSam 3D reconstruction.
The key benefit is that all models (DepthAnything, UniDepth, RAFT, DROID-SLAM) are
loaded once and reused for all videos, significantly reducing overall processing time.

Sequential processing (NOT parallel/batch):
- All reconstruction models are loaded once at startup (~3-5 minutes)
- Videos are processed one by one to maintain stable memory usage
- Total time is dramatically reduced compared to processing videos individually

Memory requirements:
- ~40GB+ VRAM for typical sequences
- Memory usage scales with video length

Example usage:
    python sequential_processing/stage1_sequential_megasam_reconstruction.py \
        --pattern "jump" \
        --video-base-dir ./demo_data/input_images \
        --outdir ./demo_data/input_megasam \
        --gsam2
"""

import os
import sys
import glob
import argparse
import torch
import h5py
import numpy as np
import PIL
from tqdm import tqdm
from pathlib import Path

# Append necessary directories to sys.path
sys.path.append('third_party/megasam-package')
sys.path.append('third_party/megasam-package/Depth-Anything')
sys.path.append('third_party/megasam-package/UniDepth')
sys.path.append("third_party/megasam-package/base/droid_slam")
sys.path.append('third_party/megasam-package/cvd_opt/core')
sys.path.append('third_party/megasam-package/cvd_opt')
sys.path.append("third_party/megasam-package/base/droid_slam")

from raft import RAFT
from droid import Droid
from depth_anything.dpt import DPT_DINOv2
from unidepth.models import UniDepthV2
from camera_tracking_scripts.test_demo import droid_slam_optimize, return_full_reconstruction
from preprocess_flow import prepare_img_data, process_flow
from cvd_opt import cvd_optimize
from run_all import demo_unidepth, demo_depthanything

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

def process_dynamic_masks(dynamic_mask_root, size=512):
    # Load and stack valid masks
    dynamic_msk = np.stack([np.load(f)['mask'] for f in dynamic_mask_root if os.path.exists(f)], axis=0)
    final_msk = []
    for msk in dynamic_msk:
        msk = PIL.Image.fromarray(((msk > 0) * 255).astype(np.uint8))
        msk = crop_img(msk, size)
        final_msk.append(msk)
    final_msk = np.array(final_msk) > 0  # Convert to boolean (True if dynamic)
    return final_msk

def crop_img(img, size=512):
    # Resize image to fit within 'size' while maintaining aspect ratio and aligning to multiple of 8
    h0, w0 = img.size
    if h0 >= w0:
        h1 = size
        w1 = int(w0 * (size / h0))
    else:
        w1 = size
        h1 = int(h0 * (size / w0))

    # Align to a multiple of 8
    w1 = (w1 + 7) // 8 * 8
    h1 = (h1 + 7) // 8 * 8

    # Crop and resize
    img = img.resize((h1, w1), PIL.Image.LANCZOS)
    return img

def preprocess_and_get_transform(file, size=512, square_ok=False):
    img = PIL.Image.open(file)
    original_width, original_height = img.size
    
    # Step 1: Resize
    S = max(img.size)
    if S > size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*size/S)) for x in img.size)
    img_resized = img.resize(new_size, interp)
    
    # Calculate center of the resized image
    cx, cy = img_resized.size[0] // 2, img_resized.size[1] // 2
    
    # Step 2: Crop
    halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
    if not square_ok and new_size[0] == new_size[1]:
        halfh = 3*halfw//4
    
    img_cropped = img_resized.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    
    # Calculate the total transformation
    scale_x = new_size[0] / original_width
    scale_y = new_size[1] / original_height
    
    translate_x = (cx - halfw) / scale_x
    translate_y = (cy - halfh) / scale_y
    
    affine_matrix = np.array([
        [1/scale_x, 0, translate_x],
        [0, 1/scale_y, translate_y]
    ])
    
    return img_cropped, affine_matrix

def project_depth_to_pts3d(depth, intrinsics, cams2world):
    # depth: (N, H, W)
    # intrinsics: (3, 3)
    # cams2world: (N, 4, 4)
    # return: (N, H, W, 3)
    N, H, W = depth.shape
    pts3d_cam = np.zeros((N, H, W, 3))
    pts3d = np.zeros((N, H, W, 3))

    # Create pixel coordinate grid
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Convert to homogeneous coordinates
    ones = np.ones_like(x)
    pixels = np.stack([x, y, ones], axis=-1)  # (H, W, 3)

    # Get camera coordinates using inverse intrinsics
    inv_K = np.linalg.inv(intrinsics)
    rays = np.einsum('ij,hwj->hwi', inv_K, pixels)  # (H, W, 3)
    
    # Scale rays by depth to get 3D points in camera space
    for i in range(N):
        pts3d_cam[i] = rays * depth[i, ..., None]  # (H, W, 3)
        
        # Convert to homogeneous coordinates
        pts_homo = np.concatenate([pts3d_cam[i], np.ones_like(pts3d_cam[i][..., :1])], axis=-1)  # (H, W, 4)
        
        # Transform to world coordinates
        pts3d[i] = np.einsum('ij,hwj->hwi', cams2world[i], pts_homo)[..., :3]  # (H, W, 3)

    return pts3d, pts3d_cam

def process_video(
    input_dir: str,
    output_dir: str,
    models,
    args,
    start_frame: int = 0,
    end_frame: int = -1,
    frame_sample_ratio: int = 1,
    gsam2: bool = False,
):
    """Process a single video directory with MegaSAM."""
    
    # Unpack models
    model_uni, depth_anything, flow_model = models
    
    # Get all images
    img_path_list = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    if len(img_path_list) == 0:
        img_path_list = sorted(glob.glob(os.path.join(input_dir, '*.png')))
        
    if len(img_path_list) == 0:
        print(f"No images found in {input_dir}")
        return

    if len(img_path_list) >= args.max_frames or len(img_path_list) < args.min_frames:
        print(f"Warning: ---- {input_dir} has {len(img_path_list)} frames, which is too many or too few for MegaSAM, skipping")
        return

    # Handle end_frame
    if end_frame == -1:
        end_frame = len(img_path_list)

    # Subsample frames
    img_path_list = img_path_list[start_frame:end_frame:frame_sample_ratio]
    if not img_path_list:
        print(f"No images after subsampling in {input_dir}")
        return
    
    print(f"\nProcessing {len(img_path_list)} frames from {input_dir}")
    
    # Create output file name
    # video_name = '_'.join(input_dir.split('/')[-2:])
    video_name = input_dir.split("/")[-2]
    scene_name = "megasam_reconstruction_results_" + video_name + "_cam01_frame_" + str(start_frame) + "_" + str(end_frame) + "_subsample_" + str(frame_sample_ratio) + ".h5"
    scene_name = f"megasam_reconstruction_results_{video_name}_cam01_frame_{start_frame}_{end_frame}_subsample_{frame_sample_ratio}.h5"
    save_path = os.path.join(output_dir, scene_name)

    # if outputfile exists, skip
    if os.path.exists(save_path):
        print(f"Warning: ---- {save_path} exists, skipping")
        return
    
    # Run the models
    try:
        # Run unidepth
        depth_list_uni, fovs = demo_unidepth(model_uni, img_path_list, args, save=args.save_intermediate)
        
        # Run depth anything
        depth_list_da = demo_depthanything(depth_anything, img_path_list, args, save=args.save_intermediate)
        
        # Process flow
        img_data = prepare_img_data(img_path_list)
        flows_high, flow_masks_high, iijj = process_flow(
            flow_model, img_data, scene_name if args.save_intermediate else None
        )
        
        # Run droid slam
        droid, traj_est, rgb_list, senor_depth_list, motion_prob = droid_slam_optimize(
            img_path_list, depth_list_da, depth_list_uni, fovs, args
        )

        # Get reconstruction
        images, disps, poses, intrinsics, motion_prob = return_full_reconstruction(
            droid, traj_est, rgb_list, senor_depth_list, motion_prob
        )

        # Run cvd optimize
        images, depths, intrinsics, cam_c2w = cvd_optimize(
            images[:, ::-1, ...],
            disps + 1e-6,
            poses,
            intrinsics,
            motion_prob,
            flows_high,
            flow_masks_high,
            iijj,
            output_dir,
            scene_name,
            args.w_grad,
            args.w_normal,
            save=False
        )
        
        # Extract data
        rgbimg = images
        cams2world = cam_c2w
        pts3d, pts3d_cam = project_depth_to_pts3d(depths, intrinsics, cams2world)
        msk = np.ones_like(depths)
        confs = np.ones_like(depths)
        
        # Handle dynamic masks if needed
        if gsam2:
            # Extract dynamic mask files from filelist
            dynamic_mask_root = []
            for f in img_path_list:
                # Extract the base path and frame number
                base_path = f.split('/cam01/')[0]
                # replace input_images with input_masks
                base_path = base_path.replace('input_images', 'input_masks')
                
                # Handle different filename formats
                if 'frame_' in f:
                    # Format: frame_xxxxx.jpg or frame_xxxxx.png
                    frame_num = int(f.split('frame_')[1].split('.')[0])
                else:
                    # Format: xxxxx.jpg or xxxxx.png
                    frame_num = int(os.path.basename(f).split('.')[0])
                
                # Construct the mask path
                mask_path = f"{base_path}/cam01/mask_data/mask_{frame_num:05d}.npz"
                dynamic_mask_root.append(mask_path)

            if len(dynamic_mask_root) > 0:
                dynamic_msk = process_dynamic_masks(dynamic_mask_root)
            else:
                dynamic_msk = np.ones_like(depths)
                print(f"Warning: ---- No dynamic masks found in {input_dir}, setting all masks to 1")
        else:
            dynamic_msk = np.ones_like(depths)
            
        # Get affine matrices
        affine_matrix_list = []
        for i in range(len(rgbimg)):
            _, affine_matrix = preprocess_and_get_transform(img_path_list[i])
            affine_matrix_list.append(affine_matrix)
            
        # Prepare results dictionary
        results = {}
        for i in range(len(rgbimg)):
            img_name = os.path.basename(img_path_list[i])[:-4]
            results[img_name] = {
                "rgbimg": rgbimg[i],                # (512, 288, 3) (H, W, 3)
                "intrinsic": intrinsics,            # (3, 3)
                "cam2world": cams2world[i],         # (4, 4)
                "pts3d": pts3d[i],                  # (512, 288, 3)
                "depths": depths[i],                # (512, 288)
                "msk": msk[i],                      # (512, 288)
                "conf": confs[i],                   # (512, 288)
                "dynamic_msk": dynamic_msk[i],      # (512, 288)
                "affine_matrix": affine_matrix_list[i],
                "motion_prob": motion_prob[i]       # (64, 36)
            }
        
        total_output = {
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

        # Save to HDF5
        with h5py.File(save_path, "w") as h5file:
            save_dict_to_hdf5(h5file, total_output)

        print(f"Results saved to {save_path}")
        return total_output
    
    except Exception as e:
        print(f"Error processing {input_dir}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-base-dir', type=str, help="Base directory containing video folders")
    parser.add_argument('--outdir', type=str, default='./demo_data/input_megasam', help="Base directory for output")
    parser.add_argument('--pattern', type=str, default="", help="Pattern to filter video folders")
    parser.add_argument("--save_intermediate", action="store_true", default=False)
    parser.add_argument("--stride", type=int, default=1, help="Frame sampling stride")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end-frame", type=int, default=-1, help="End frame index (-1 for all frames)")
    parser.add_argument("--gsam2", action="store_true", default=False)

    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=-1, help='End index')
    parser.add_argument('--min_frames', type=int, default=0, help='Minimum number of frames')
    parser.add_argument('--max_frames', type=int, default=1000, help='Maximum number of frames')

    # for unidepth & depthanything
    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--load-from', type=str, default="./assets/checkpoints/depth_anything_vitl14.pth")
    parser.add_argument('--localhub', dest='localhub', action='store_true', default=False)
    
    # for raft
    parser.add_argument('--model', default='./assets/ckpt_raft/raft-things.pth', help='restore checkpoint')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='use mixed precision')
    parser.add_argument('--num_heads',default=1,type=int,help='number of heads in attention and aggregation')
    parser.add_argument('--position_only',default=False,action='store_true',help='only use position-wise attention')
    parser.add_argument('--position_and_content',default=False,action='store_true',help='use position and content-wise attention')
    parser.add_argument('--small', action='store_true', help='use small model')

    # for cvd optimize
    parser.add_argument("--w_grad", type=float, default=2.0, help="w_grad")
    parser.add_argument("--w_normal", type=float, default=5.0, help="w_normal")

    # for droid slam
    parser.add_argument("--weights", default="./assets/checkpoints/megasam_final.pth")
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=2.0)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--scene_name", help="scene_name")
    parser.add_argument("--backend_thresh", type=float, default=16.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--disable_vis", type=bool, default=True)

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Initialize models once - this is the key benefit of sequential processing!
    # Loading all these models takes 3-5 minutes, but we only do it once for all videos
    print("Loading MegaSAM models (this happens only once for all videos)...")
    
    # Load Unidepth model
    print("Loading UniDepthV2 model...")
    model_uni = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2old-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_uni = model_uni.to(device)

    # Load Depth Anything model
    print("Loading Depth Anything model...")
    assert args.encoder in ['vits', 'vitb', 'vitl']
    if args.encoder == 'vits':
        depth_anything = DPT_DINOv2(
            encoder='vits',
            features=64,
            out_channels=[48, 96, 192, 384],
            localhub=args.localhub,
        ).cuda()
    elif args.encoder == 'vitb':
        depth_anything = DPT_DINOv2(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
            localhub=args.localhub,
        ).cuda()
    else:
        depth_anything = DPT_DINOv2(
            encoder='vitl',
            features=256,
            out_channels=[256, 512, 1024, 1024],
            localhub=args.localhub,
        ).cuda()

    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    depth_anything.load_state_dict(
        torch.load(args.load_from, map_location='cpu'), strict=True
    )
    depth_anything.eval()

    # Load RAFT flow model
    print("Loading RAFT flow model...")
    model_raft = torch.nn.DataParallel(RAFT(args))
    model_raft.load_state_dict(torch.load(args.model))
    flow_model = model_raft.module
    flow_model.cuda().eval()
    
    # Pack all models into a tuple
    models = (model_uni, depth_anything, flow_model)
    
    # Get all video folders and filter by pattern
    video_base_dir = args.video_base_dir
    pattern = args.pattern
    
    # Find all video directories
    video_folders = [f for f in sorted(os.listdir(video_base_dir)) 
                    if os.path.isdir(os.path.join(video_base_dir, f)) 
                    and (not pattern or pattern.lower() in f.lower())]
    video_folders = video_folders[args.start_idx:] if args.end_idx == -1 else video_folders[args.start_idx:args.end_idx]
    
    print(f"Found {len(video_folders)} video folders matching pattern '{pattern}'")
    if len(video_folders) == 0:
        print("No matching folders found!")
        return
        
    print("Processing folders:", video_folders)
    
    # Process each video sequentially (one after another)
    # This maintains stable GPU memory usage while reusing all loaded models
    for video_name in tqdm(video_folders, desc="Processing videos sequentially"):
        video_dir = os.path.join(video_base_dir, video_name)
        cam_folders = [f for f in os.listdir(video_dir) if f.startswith("cam")]
        if cam_folders == []: # if the video is not divided into cameras
            cam_folders = [""]
        
        print(f"\nProcessing {video_name} with {len(cam_folders)} cameras")
        
        for cam in tqdm(cam_folders, desc="Processing cameras"):
            input_dir = os.path.join(video_dir, cam) if cam else video_dir
            
            print(f"\nProcessing {video_name}/{cam}")
            try:
                process_video(
                    input_dir,
                    args.outdir,
                    models,
                    args,
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    frame_sample_ratio=args.stride,
                    gsam2=args.gsam2
                )
            except Exception as e:
                print(f"Error processing {video_name}/{cam}: {e}")
                continue

if __name__ == "__main__":
    main()