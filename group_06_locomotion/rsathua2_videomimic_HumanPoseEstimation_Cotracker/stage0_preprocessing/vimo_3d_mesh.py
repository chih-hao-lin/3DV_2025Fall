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


from vimo.models import get_hmr_vimo


def main(
    img_dir: str = '../human_in_world/demo_data/input_images/IMG_7415/cam01',
    mask_dir: str = '../human_in_world/demo_data/input_masks/IMG_7415/cam01',
    out_dir: str = '../human_in_world/demo_data/input_3d_meshes/IMG_7415/cam01',
    checkpoint: str = './assets/checkpoints/vimo_checkpoint.pth.tar',
    cfg_path: str = './assets/configs/config_vimo.yaml',
    multihuman: bool = False,
    top_k: int = 0
):
    """
    Get SMPL parameters from VIMO: https://yufu-wang.github.io/tram4d/files/tram.pdf
    Args:
        img_dir: directory of images
        mask_dir: directory of masks
        out_dir: directory of output
    """

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Retrieve the person ids from the meta data in sam2 output using bbox_dir
    # Load the meta data
    with open(osp.join(mask_dir, 'meta_data.json'), 'r') as f:
        meta_data = json.load(f)
    # {"most_persistent_id": 1, "largest_area_id": 1, "highest_confidence_id": 1}
    # person_id_list = [meta_data["most_persistent_id"], meta_data["largest_area_id"], meta_data["highest_confidence_id"]]
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
        # Do the majority voting across the three person ids
        from collections import Counter
        person_id_list = [meta_data["largest_area_id"]]
        person_id_counter = Counter(person_id_list)
        majority_voted_person_id = person_id_counter.most_common(1)[0][0]
        # print(f"Majority voted person id: {majority_voted_person_id} using most persistent, largest area, and highest confidence")
        print(f"Majority voted person id: {majority_voted_person_id} using largest area")
        person_ids = [majority_voted_person_id]

    # load bounding boxes
    bbox_files = sorted(glob(f'{mask_dir}/json_data/mask_*.json'))

    person_data = {person_id: {'frame_indices': [], 'bboxes': []} for person_id in person_ids}
    frame_all_data = {} # has the following data structure:
    # {
    #     frame_idx: {
    #         person_id: {
    #             'smpl_params': {
    #                 'global_orient': numpy array of shape (1, 23, 3, 3),
    #                 'body_pose': numpy array of shape (23, 3, 3),
    #                 'betas': numpy array of shape (10,),
    #             }
    #         }
    #         person_id2: { ... },
    #         ...
    #     }
    # }

    # collect the data for each person
    for bbox_file in bbox_files:
        frame_idx = int(Path(bbox_file).stem.split('_')[-1])
        with open(bbox_file, 'r') as f:
            bbox_data = json.load(f)
            # if value of "labels" key is empty, continue
            if not bbox_data['labels']:
                continue
            else:
                labels = bbox_data['labels']
                # "labels": {"1": {"instance_id": 1, "class_name": "person", "x1": 454, "y1": 399, "x2": 562, "y2": 734, "logit": 0.0}, "2": {"instance_id": 2, "class_name": "person", "x1": 45, "y1": 301, "x2": 205, "y2": 812, "logit": 0.0}}}
                label_keys = sorted(labels.keys())

                 # Collect data for each person
                for person_id in person_ids:
                    key = str(person_id)
                    if key in label_keys:  # This person appears in this frame
                        box = [labels[key]['x1'], labels[key]['y1'], 
                            labels[key]['x2'], labels[key]['y2']]
                        if sum(box) > 0:  # Valid bbox
                            person_data[person_id]['frame_indices'].append(frame_idx)
                            person_data[person_id]['bboxes'].append(box)

    ##### Run HPS (here we use VIMO) #####
    print('Loading VIMO model...')
    model = get_hmr_vimo(
        checkpoint=checkpoint,
        cfg_path=cfg_path)
    
    # Process each person (works for both single and multi-human)
    for person_id in person_ids:
        frame_indices = person_data[person_id]['frame_indices']
        person_bboxes = person_data[person_id]['bboxes']
        
        if not frame_indices:
            print(f"No valid frames found for person {person_id}")
            continue

        # Get image paths for frames where this person appears
        person_imgfiles = []
        valid_indices = []
        valid_bboxes = []

        # Get image paths for frames where this person appears
        person_imgfiles = []
        valid_indices = []
        valid_bboxes = []
        
        for i, frame_idx in enumerate(frame_indices):
            img_path = None
            if osp.exists(osp.join(img_dir, f'{frame_idx:05d}.jpg')):
                img_path = osp.join(img_dir, f'{frame_idx:05d}.jpg')
            elif osp.exists(osp.join(img_dir, f'frame_{frame_idx:05d}.jpg')):
                img_path = osp.join(img_dir, f'frame_{frame_idx:05d}.jpg')
            
            if img_path:
                person_imgfiles.append(img_path)
                valid_indices.append(frame_idx)
                valid_bboxes.append(person_bboxes[i])
            else:
                print(f"Warning: No image found for frame {frame_idx}")
        
        if not person_imgfiles:
            continue

        assert len(person_imgfiles) == len(valid_bboxes)
        bboxes_array = np.array(valid_bboxes, dtype=np.float32)

        # Call inference with matched data for this person
        print(f'Estimating HPS for person {person_id} ({len(person_imgfiles)} frames)...')
        results = model.inference(person_imgfiles, bboxes_array, img_focal=None, img_center=None)

        # Store results for this person
        pred_rotmat = results['pred_rotmat']
        pred_shape = results['pred_shape']
        
        for i, frame_idx in enumerate(valid_indices):
            if frame_idx not in frame_all_data:
                frame_all_data[frame_idx] = {}
            
            frame_all_data[frame_idx][person_id] = {
                'smpl_params': {
                    'global_orient': pred_rotmat[i][0:1, :, :].cpu().numpy(),
                    'body_pose': pred_rotmat[i][1:, :, :].cpu().numpy(),
                    'betas': pred_shape[i].cpu().numpy(),
                }
            }

    # Save results for all frames
    print("Saving results to", out_dir)
    for frame_idx, result_dict in tqdm(frame_all_data.items()):
        frame_result_save_path = os.path.join(out_dir, f'smpl_params_{frame_idx:05d}.pkl')
        with open(frame_result_save_path, 'wb') as f:
            pickle.dump(result_dict, f)


if __name__ == '__main__':
    tyro.cli(main)