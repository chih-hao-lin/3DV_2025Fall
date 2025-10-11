# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import os
import os.path as osp
import json
import argparse
import tyro
import pickle
import numpy as np
import cv2
from pathlib import Path
from glob import glob
from tqdm import tqdm
from collections import defaultdict, Counter

from vimo.models import get_hmr_vimo

def process_video(
    img_dir: str,
    mask_dir: str,
    out_dir: str,
    model,
    person_ids: list,
    vis: bool = False,
):
    """Process a single video directory with VIMO."""
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if vis:
        Path(os.path.join(out_dir, 'vis')).mkdir(parents=True, exist_ok=True)

    # load bounding boxes
    bbox_files = sorted(glob(f'{mask_dir}/json_data/mask_*.json'))
    bbox_list = []
    frame_idx_list = []

    for bbox_file in bbox_files:
        frame_idx = int(Path(bbox_file).stem.split('_')[-1])
        with open(bbox_file, 'r') as f:
            bbox_data = json.load(f)
            if not bbox_data['labels']:
                continue
            else:
                labels = bbox_data['labels']
                label_keys = sorted(labels.keys())

                # filter label keys by person ids
                selected_label_keys = [x for x in label_keys if labels[x]['instance_id'] in person_ids]
                label_keys = selected_label_keys

                # get boxes
                boxes = np.array([[labels[str(i)]['x1'], labels[str(i)]['y1'], 
                                 labels[str(i)]['x2'], labels[str(i)]['y2']] for i in label_keys])

                if boxes.sum() == 0:
                    continue

                bbox_list.append(boxes)
                frame_idx_list.append(frame_idx)

    # get img paths
    imgfiles = []
    for frame_idx in frame_idx_list:
        if osp.exists(osp.join(img_dir, f'{frame_idx:05d}.jpg')):
            imgfiles.append(osp.join(img_dir, f'{frame_idx:05d}.jpg'))
        elif osp.exists(osp.join(img_dir, f'frame_{frame_idx:05d}.jpg')):
            imgfiles.append(osp.join(img_dir, f'frame_{frame_idx:05d}.jpg'))
        else:
            print(f"Warning: No image found for frame {frame_idx}")
            continue

    if not imgfiles:
        print(f"No valid images found in {img_dir}")
        return

    assert len(imgfiles) == len(bbox_list)

    img_focal = None
    img_center = None

    bboxes = np.concatenate(bbox_list, axis=0, dtype=np.float32)
    
    print('Estimate HPS ...')
    results = model.inference(imgfiles, bboxes, img_focal=img_focal, img_center=img_center)

    print("Save results to ", out_dir)
    pred_rotmat = results['pred_rotmat']  # (num_frames, 24, 3, 3)
    pred_shape = results['pred_shape']    # (num_frames, 10)

    for f_idx, frame_idx in tqdm(enumerate(frame_idx_list)):
        frame_result_save_path = os.path.join(out_dir, f'smpl_params_{frame_idx:05d}.pkl')
        result_dict = {
            person_ids[0]: {  # Using first person_id as majority_voted_person_id
                'smpl_params': {
                    'global_orient': pred_rotmat[f_idx][0:1, :, :].cpu().numpy(),
                    'body_pose': pred_rotmat[f_idx][1:, :, :].cpu().numpy(),
                    'betas': pred_shape[f_idx].cpu().numpy(),
                }
            }
        }
        with open(frame_result_save_path, 'wb') as f:
            pickle.dump(result_dict, f)

def main(
    video_base_dir: str = "./demo_data/input_images",  # Base directory containing video folders
    output_base_dir: str = "./demo_data/input_3d_meshes",  # Base directory for output
    bbox_base_dir: str = "./demo_data/input_masks",  # Base directory for input bboxes
    pattern: str = "",  # Pattern to filter video folders
    checkpoint: str = './assets/checkpoints/vimo_checkpoint.pth.tar',
    cfg_path: str = './assets/configs/config_vimo.yaml',
    multihuman: bool = False,  # Whether to process multiple humans
    top_k: int = 0,  # Process only top-k largest humans (0 means all)
    vis: bool = False,
    start_idx: int = 0,
    end_idx: int = -1,
):
    """Process multiple videos with VIMO.
    
    Args:
        video_base_dir: Base directory containing video folders. Each subfolder should contain cam01, cam02, etc.
        output_base_dir: Base directory for output meshes
        pattern: String pattern to filter video folders. Only process folders containing this pattern.
        checkpoint: Path to VIMO checkpoint
        cfg_path: Path to VIMO config file
        vis: Whether to visualize results
    """
    
    # Initialize model once
    print("Loading VIMO model...")
    model = get_hmr_vimo(
        checkpoint=checkpoint,
        cfg_path=cfg_path
    )
    
    # Get all video folders and filter by pattern
    video_folders = sorted([f for f in os.listdir(video_base_dir) 
                    if os.path.isdir(os.path.join(video_base_dir, f)) 
                    and (not pattern or pattern.lower() in f.lower())])
    video_folders = sorted(video_folders)
    if end_idx == -1:
        end_idx = len(video_folders)
    video_folders = video_folders[start_idx:end_idx]
    
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
            mask_dir = os.path.join(bbox_base_dir, video_name, cam)
            output_dir = os.path.join(output_base_dir, video_name, cam)
            
            if not os.path.exists(mask_dir):
                print(f"Warning: No mask directory found for {video_name}/{cam}")
                continue

            # Get person IDs from meta data
            meta_data_path = os.path.join(mask_dir, 'meta_data.json')
            if not os.path.exists(meta_data_path):
                print(f"Warning: No meta_data.json found at {meta_data_path}")
                continue

            with open(meta_data_path, 'r') as f:
                meta_data = json.load(f)
            
            # Select person IDs based on multihuman and top_k settings
            if multihuman:
                all_instance_ids = meta_data["all_instance_ids"]
                
                # If top_k is specified, use only the top-k largest humans by average area
                if top_k > 0 and "sorted_by_avg_area" in meta_data:
                    sorted_ids = meta_data["sorted_by_avg_area"]
                    person_ids = sorted_ids[:top_k]
                    print(f"Selecting top-{top_k} person IDs by average area: {person_ids}")
                else:
                    person_ids = all_instance_ids
                    print(f"Selecting all person IDs in the video: {person_ids}")
            else:
                # Do majority voting for person IDs
                person_id_list = [
                    # meta_data.get("most_persistent_id", 1),
                    meta_data.get("largest_area_id", 1),
                    # meta_data.get("highest_confidence_id", 1)
                ]
                person_id_counter = Counter(person_id_list)
                majority_voted_person_id = person_id_counter.most_common(1)[0][0]
                print(f"Majority voted person id: {majority_voted_person_id}")
                person_ids = [majority_voted_person_id]
                
            print(f"\nProcessing {video_name}/{cam}")
            try:
                process_video(
                    input_dir,
                    mask_dir,
                    output_dir,
                    model,
                    person_ids,
                    vis=vis
                )
            except Exception as e:
                print(f"Error processing {video_name}/{cam}: {e}")
                continue

if __name__ == "__main__":
    tyro.cli(main) 