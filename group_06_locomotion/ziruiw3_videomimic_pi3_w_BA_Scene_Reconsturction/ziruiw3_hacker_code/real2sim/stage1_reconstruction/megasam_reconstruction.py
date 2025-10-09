# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

# CUDA_VISIBLE_DEVICES=1 python run_all.py --img_path DAVIS --iterate --delta

import os
import sys
import glob
import argparse
import torch
import pickle
import torch.nn.functional as F
import numpy as np
import cv2
import imageio
import os.path as osp
from PIL import Image
from timeit import default_timer as timer
from tqdm import tqdm
from torchvision.transforms import Compose
import PIL

# Append necessary directories to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
megasam_package_path = os.path.join(project_root, "third_party/megasam-package")

sys.path.append(megasam_package_path)
sys.path.append(os.path.join(megasam_package_path, 'Depth-Anything'))
sys.path.append(os.path.join(megasam_package_path, 'UniDepth'))
sys.path.append(os.path.join(megasam_package_path, "base/droid_slam"))
sys.path.append(os.path.join(megasam_package_path, 'cvd_opt/core'))
sys.path.append(os.path.join(megasam_package_path, 'cvd_opt'))
sys.path.append(os.path.join(megasam_package_path, "base/droid_slam"))

from raft import RAFT
from droid import Droid
from core.utils.utils import InputPadder
from pathlib import Path  # pylint: disable=g-importing-member
from lietorch import SE3
from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
from unidepth.models import UniDepthV2
from unidepth.utils import colorize, image_grid
from camera_tracking_scripts.test_demo import droid_slam_optimize, return_full_reconstruction
from preprocess_flow import prepare_img_data, process_flow
from cvd_opt import cvd_optimize
from run_all import demo_unidepth, demo_depthanything


LONG_DIM = 640

import h5py

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
    dynamic_msk = np.stack([np.load(f)['mask'] for f in dynamic_mask_root if osp.exists(f)], axis=0)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str)
    parser.add_argument('--out-dir', type=str, default='./demo_data/input_megasam')
    parser.add_argument("--save_intermediate", action="store_true", default=False)
    parser.add_argument("--iterate", action="store_true", default=False)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=512)
    parser.add_argument("--gsam2", action="store_true", default=False)

    # for unidepth & depthanything
    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--load-from', type=str, default="./assets/checkpoints/depth_anything_vitl14.pth")
    # parser.add_argument('--max_size', type=int, required=True)
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
    parser.add_argument(
        "--filter_thresh", type=float, default=2.0
    )  # motion threhold for keyframe
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
    out_dir = args.out_dir
    args.outdir = out_dir
    w_grad = args.w_grad
    w_normal = args.w_normal

    # step 1: prepare unidepth

    # model_uni = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    model_uni = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2old-vitl14")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_uni = model_uni.to(device)


    # step 2: prepare depth anything

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

    # step 3: prepare the flow model

    model_raft = torch.nn.DataParallel(RAFT(args))
    model_raft.load_state_dict(torch.load(args.model))
    print(f'Loaded checkpoint at {args.model}')
    flow_model = model_raft.module
    flow_model.cuda()  # .eval()
    flow_model.eval()


    # run the pipeline

    if args.iterate: # get all the subfolders in the args.video_dir
        folders = sorted([os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if os.path.isdir(os.path.join(args.video_dir, f))])
        # folders = folders[:16]
    else:
        folders = [args.video_dir]
    print(f"Processing {len(folders)} folders")
    for img_path in tqdm(folders):
        # scene_name = img_path.split("/")[-1]
        scene_name = "megasam_reconstruction_results_" + img_path.split("/")[-2] + "_cam01_frame_" + str(args.start_frame) + "_" + str(args.end_frame) + "_subsample_" + str(args.stride) + ".h5"
        save_path = os.path.join(out_dir, scene_name)

        # step1&2&3: Run the demo
        img_path_list = sorted(glob.glob(os.path.join(img_path, "*.jpg")))
        img_path_list += sorted(glob.glob(os.path.join(img_path, "*.png")))
        img_path_list = img_path_list[args.start_frame:args.end_frame:args.stride]

        depth_list_uni, fovs = demo_unidepth(model_uni, img_path_list, args, save=args.save_intermediate)
        depth_list_da = demo_depthanything(depth_anything, img_path_list, args, save=args.save_intermediate)
        img_data = prepare_img_data(img_path_list)
        flows_high, flow_masks_high, iijj = process_flow(flow_model, img_data, scene_name if args.save_intermediate else None)
        
        # step 4: Run the droid slam
        droid, traj_est, rgb_list, senor_depth_list, motion_prob = droid_slam_optimize(
            img_path_list, depth_list_da, depth_list_uni, fovs, args
        )

        images, disps, poses, intrinsics, motion_prob = return_full_reconstruction(
                droid, traj_est, rgb_list, senor_depth_list, motion_prob
            )

        # step 5: Run the cvd optimize
        images, depths, intrinsics, cam_c2w = cvd_optimize(
                images[:, ::-1, ...],
                disps + 1e-6,
                poses,
                intrinsics,
                motion_prob,
                flows_high,
                flow_masks_high,
                iijj,
                out_dir,
                scene_name,
                w_grad,
                w_normal,
                save=False
            )
        
        rgbimg = images
        intrinsics = intrinsics
        cams2world = cam_c2w
        pts3d, pts3d_cam = project_depth_to_pts3d(depths, intrinsics, cams2world)
        depths = depths
        msk = np.ones_like(depths)
        confs = np.ones_like(depths)
        if args.gsam2:
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
                print(f"Warning: ---- No dynamic masks found in {args.video_dir}, setting all masks to 1")
        else:
            dynamic_msk = np.ones_like(depths)
        affine_matrix_list = []
        for i in range(len(rgbimg)):
            _, affine_matrix = preprocess_and_get_transform(img_path_list[i])
            affine_matrix_list.append(affine_matrix)
        results = {}
        for i in range(len(rgbimg)):
            results[img_path_list[i].split("/")[-1][:-4]] = {
                "rgbimg": rgbimg[i],                # (512, 288, 3) (H, W, 3)
                "intrinsic": intrinsics,            # (3, 3)
                "cam2world": cams2world[i],         # (4, 4)
                "pts3d": pts3d[i],                  # (512, 288, 3)
                "depths": depths[i],                # (512, 288)
                "msk": msk[i],                      # (512, 288)
                "conf": confs[i],                   # (512, 288)
                "dynamic_msk": dynamic_msk[i],      # (512, 288)
                "affine_matrix": affine_matrix_list[i],
                "motion_prob": motion_prob[i] # (64, 36)
            }
        total_output = {
            "monst3r_ga_output": results
        }

        # Remove pred_depth from output's view1 and view2
        # Make all the values to be numpy in output
        def convert_to_numpy(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    convert_to_numpy(value)
                elif isinstance(value, torch.Tensor):
                    d[key] = value.numpy()

        convert_to_numpy(total_output)

        if not osp.exists(out_dir):
            os.makedirs(out_dir)

        # Save to an HDF5 file    
        with h5py.File(save_path, "w") as h5file:
            save_dict_to_hdf5(h5file, total_output)

        print(f"Megasam Finished processing {scene_name}, saved to {out_dir}/{scene_name}")
