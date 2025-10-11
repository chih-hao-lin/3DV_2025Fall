# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Sequential ViTPose 2D Pose Estimation for Multiple Videos

This script processes multiple videos sequentially through ViTPose 2D pose estimation.
The key benefit is that the ViTPose model is loaded once and reused for all videos,
significantly reducing overall processing time when building datasets.

Sequential processing (NOT parallel/batch):
- ViTPose model is loaded once at startup (~1 minute)
- Videos are processed one by one to maintain stable memory usage
- Supports multi-human processing with consistent person tracking

Example usage:
    python sequential_processing/stage0_sequential_vitpose_2d_poses.py \
        --pattern "jump" \
        --video-base-dir ./demo_data/input_images \
        --bbox-base-dir ./demo_data/input_masks \
        --output-base-dir ./demo_data/input_2d_poses \
        --multihuman --top-k 3
"""

import os
import glob
import tyro
import cv2 
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict

# Add path to stage0_preprocessing directory
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
stage0_path = os.path.join(project_root, "stage0_preprocessing")
sys.path.insert(0, stage0_path)

from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result

class ViTPoseModel:
    def __init__(
        self,
        model_config: str,
        model_checkpoint: str,
        device: str = 'cuda:0'
    ):
        self.model = init_pose_model(model_config, model_checkpoint, device)
        self.device = device

    def predict_pose(self, image, bboxes, box_score_threshold=0.5):
        """Predict pose for given image and bounding boxes."""
        # Convert bbox format for mmpose
        person_results = []
        for bbox_dict in bboxes:
            bbox = bbox_dict['bbox']
            if bbox[4] >= box_score_threshold:  # confidence check
                person_results.append({'bbox': bbox})
        
        if not person_results:
            return []
        
        # Run pose estimation
        pose_results, _ = inference_top_down_pose_model(
            self.model,
            image,
            person_results,
            bbox_thr=box_score_threshold,
            format='xyxy'
        )
        
        return pose_results

def custom_draw_pose(image, pose_results, kpt_score_thr=0.3, radius=4, thickness=1):
    """Draw COCO keypoints and skeleton on an image.

    Args:
        image (np.ndarray | None): The image to draw on. If None, a white
            background will be created.
        pose_results (np.ndarray): Pose results for a single person,
            shaped (17, 3). The third column is the score.
        kpt_score_thr (float): Minimum score threshold to draw a keypoint.
        radius (int): Radius of keypoint circles.
        thickness (int): Thickness of skeleton lines.

    Returns:
        np.ndarray: Image with drawn pose.
    """
    if image is None:
        # Create a white background if no image is provided
        height, width = 512, 512  # Default size
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255],
                        [255, 0, 0], [255, 255, 255]])

    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]]

    pose_link_color = palette[[
        0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    ]]
    pose_kpt_color = palette[[
        16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
    ]]

    assert pose_results.shape == (17, 3), \
        f"pose_results should have shape (17, 3), but got {pose_results.shape}"

    img = image.copy()
    # img[:] = 255
    keypoints = pose_results[:, :2].astype(np.int32)
    scores = pose_results[:, 2]

    # Draw skeleton
    for i, (start_idx, end_idx) in enumerate(skeleton):
        if scores[start_idx] > kpt_score_thr and scores[end_idx] > kpt_score_thr:
            start_point = tuple(keypoints[start_idx])
            end_point = tuple(keypoints[end_idx])
            color = tuple(pose_link_color[i].tolist())
            cv2.line(img, start_point, end_point, color, thickness)

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if scores[i] > kpt_score_thr:
            color = tuple(pose_kpt_color[i].tolist())
            cv2.circle(img, (x, y), radius, color, -1) # -1 fills the circle

    return img


def process_video(
    img_dir: str,
    bbox_dir: str,
    output_dir: str,
    model: ViTPoseModel,
    multihuman: bool = False,
    top_k: int = 0,
    vis: bool = False,
    box_score_threshold: float = 0.5,
    kpt_score_threshold: float = 0.3,
    vis_dot_radius: int = 10,
    vis_line_thickness: int = 5,
):
    """Process a single video directory with ViTPose."""
    # Load the GSAM2 output
    gsam2_meta_data_path = os.path.join(bbox_dir, "..", "meta_data.json")
    with open(gsam2_meta_data_path, 'r') as f:
        gsam2_meta_data = json.load(f)
    # {"most_persistent_id": 1, "largest_area_id": 1, "highest_confidence_id": 1}

    # Do the majority voting across the three person ids if multihuman is False, otherwise use all person ids
    if multihuman:
        all_instance_ids = gsam2_meta_data["all_instance_ids"]
        
        # If top_k is specified, use only the top-k largest humans by average area
        if top_k > 0 and "sorted_by_avg_area" in gsam2_meta_data:
            sorted_ids = gsam2_meta_data["sorted_by_avg_area"]
            selected_ids = sorted_ids[:top_k]
            print(f"Selecting top-{top_k} person IDs by average area: {selected_ids}")
        else:
            selected_ids = all_instance_ids
            print(f"Selecting all person IDs in the video: {all_instance_ids}")
    else:
        from collections import Counter
        person_id_list = [
            # gsam2_meta_data["most_persistent_id"], 
            gsam2_meta_data["largest_area_id"], 
            # gsam2_meta_data["highest_confidence_id"]
        ]
        person_id_counter = Counter(person_id_list)
        majority_voted_person_id = person_id_counter.most_common(1)[0][0]
        print(f"Majority voted person id: {majority_voted_person_id} using largest area")
    

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if vis:
        Path(os.path.join(output_dir, 'vis')).mkdir(parents=True, exist_ok=True)

    # Run per image
    img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not img_path_list:
        img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
    print(f"Processing {len(img_path_list)} frames...")
    
    for _, img_path in tqdm(enumerate(img_path_list), total=len(img_path_list)):
        img_idx = int(os.path.splitext(os.path.basename(img_path))[0].split('_')[-1])
        image = cv2.imread(img_path)
        det_result_path = os.path.join(bbox_dir, f'mask_{img_idx:05d}.json')
        
        # Skip if bbox file doesn't exist
        if not os.path.exists(det_result_path):
            print(f"Warning: No bbox file found for frame {img_idx}")
            continue
            
        bboxes = []
        frame_person_ids = []
        
        with open(det_result_path, 'r') as f:
            det_results = json.load(f)
            
            # Skip if no detections
            if len(det_results['labels']) == 0:
                continue
            
            # Get the bbox of the selected person ids
            # If using top-k, first check if area_ranking is available for this frame
            if multihuman and top_k > 0 and "area_ranking" in det_results:
                # Use per-frame area ranking
                frame_selected_ids = det_results['area_ranking'][:top_k]
            else:
                frame_selected_ids = None
                
            for box in det_results['labels'].values():
                instance_id = box['instance_id']
                
                # Filter based on selection criteria
                if not multihuman and instance_id != majority_voted_person_id:
                    continue
                if multihuman:
                    if frame_selected_ids is not None:
                        # Use per-frame ranking if available
                        if instance_id not in frame_selected_ids:
                            continue
                    elif top_k > 0:
                        # Fall back to global ranking
                        if instance_id not in selected_ids:
                            continue
                    # If no top_k specified, include all
                
                if sum([box['x1'], box['y1'], box['x2'], box['y2']]) == 0:
                    continue
                    
                bbox_dict = {'bbox': np.array([box['x1'], box['y1'], box['x2'], box['y2'], 1.0])}
                bboxes.append(bbox_dict)
                frame_person_ids.append(box['instance_id'])

            # Skip if invalid boxes
            bboxes_sum = sum([bbox['bbox'][:4].sum() for bbox in bboxes])
            if bboxes_sum == 0:
                continue

        # Run pose estimation
        out = model.predict_pose(image, bboxes, box_score_threshold)

        # Save results
        save_out = {}
        
        if multihuman:
            for out_idx, person_id in enumerate(frame_person_ids):
                save_out[person_id] = {
                    'bbox': out[out_idx]['bbox'].tolist(),
                    'keypoints': out[out_idx]['keypoints'].tolist(),
                }
        else:
            save_out = {
                majority_voted_person_id: {
                    'bbox': out[0]['bbox'].tolist(),
                    'keypoints': out[0]['keypoints'].tolist(),
                }
            }

        pose_result_path = os.path.join(output_dir, f'pose_{img_idx:05d}.json')
        with open(pose_result_path, 'w') as f:
            json.dump(save_out, f)

        # Visualize if requested
        if vis:
            # vis_out = model.visualize_pose_results(
            #     image, out, 
            #     kpt_score_threshold,
            #     vis_dot_radius, 
            #     vis_line_thickness
            # )
            vis_out = custom_draw_pose(image, out[0]['keypoints'][:17, :], kpt_score_threshold, vis_dot_radius, vis_line_thickness)
            vis_out_path = os.path.join(output_dir, 'vis', f'pose_{img_idx:05d}.jpg')
            cv2.imwrite(vis_out_path, vis_out)

def main(
    video_base_dir: str = "./demo_data/input_images",  # Base directory containing video folders
    output_base_dir: str = "./demo_data/input_2d_poses",  # Base directory for output
    bbox_base_dir: str = "./demo_data/input_masks",  # Base directory for input bboxes
    pattern: str = "",  # Pattern to filter video folders
    model_config: str = './assets/configs/vitpose/ViTPose_huge_wholebody_256x192.py',
    model_checkpoint: str = './assets/checkpoints/vitpose_huge_wholebody.pth',
    multihuman: bool = False,  # Whether to process multiple humans
    top_k: int = 0,  # Process only top-k largest humans (0 means all)
    vis: bool = False,  # Whether to visualize results
    start_idx: int = 0,
    end_idx: int = -1,
):
    """Sequential processing of multiple videos with ViTPose 2D pose estimation.
    
    This function processes videos sequentially to amortize model loading time across
    multiple videos. The ViTPose model is loaded once and reused for all videos,
    making this ideal for building training datasets efficiently.
    
    Args:
        video_base_dir: Base directory containing video folders. Each subfolder should contain cam01, cam02, etc.
        output_base_dir: Base directory for output poses
        bbox_base_dir: Base directory for input bounding boxes from SAM2
        pattern: String pattern to filter video folders. Only process folders containing this pattern.
        model_config: Path to model config file
        model_checkpoint: Path to model checkpoint file
        multihuman: Whether to process multiple humans
        top_k: Process only top-k largest humans (0 means all)
        vis: Whether to visualize results
        start_idx: Starting index for video processing
        end_idx: Ending index for video processing (-1 for all)
    """
    
    # Initialize model once - this is the key benefit of sequential processing!
    # Loading ViTPose takes ~1 minute, but we only do it once for all videos
    print("Loading ViTPose model (this happens only once for all videos)...")
    model = ViTPoseModel(model_config=model_config, model_checkpoint=model_checkpoint)
    
    # Get all video folders and filter by pattern
    video_folders = [f for f in os.listdir(video_base_dir) 
                    if os.path.isdir(os.path.join(video_base_dir, f)) 
                    and (not pattern or pattern.lower() in f.lower())]
    video_folders = sorted(video_folders)
    if end_idx != -1:
        video_folders = video_folders[start_idx:end_idx]
    
    print(f"Found {len(video_folders)} video folders matching pattern '{pattern}'")
    if len(video_folders) == 0:
        print("No matching folders found!")
        return
        
    print("Processing folders:", video_folders)
    
    # Process each video sequentially (one after another)
    # This maintains stable GPU memory usage while reusing the loaded model
    for video_name in tqdm(video_folders, desc="Processing videos sequentially"):
        video_dir = os.path.join(video_base_dir, video_name)
        cam_folders = [f for f in os.listdir(video_dir) if f.startswith("cam")]
        print("\033[92mProcessing " + video_name +  "with " + str(len(cam_folders)) + " cameras\033[0m")
        
        for cam in tqdm(cam_folders, desc="Processing cameras"):
            input_dir = os.path.join(video_dir, cam)
            bbox_dir = os.path.join(bbox_base_dir, video_name, cam, "json_data")
            output_dir = os.path.join(output_base_dir, video_name, cam)
            
            # Skip if bbox directory doesn't exist
            if not os.path.exists(bbox_dir):
                print(f"Warning: No bbox directory found for {video_name}/{cam}")
                continue
                
            print(f"\nProcessing {video_name}/{cam}")
            try:
                process_video(
                    input_dir,
                    bbox_dir,
                    output_dir,
                    model,
                    multihuman=multihuman,
                    top_k=top_k,
                    vis=vis
                )
            except Exception as e:
                print(f"Error processing {video_name}/{cam}: {e}")
                continue

if __name__ == "__main__":
    tyro.cli(main) 