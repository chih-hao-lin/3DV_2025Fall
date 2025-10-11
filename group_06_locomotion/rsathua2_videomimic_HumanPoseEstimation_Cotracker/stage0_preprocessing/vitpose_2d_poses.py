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
import cv2 
import json
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result

os.environ["PYOPENGL_PLATFORM"] = "egl"


class ViTPoseModel:
    def __init__(
            self, 
            model_name: str = 'ViTPose+-G (multi-task train, COCO)',
            model_config: str = None,
            model_checkpoint: str = None,
            device: str = "cuda",
            **kwargs
    ):
        # Get project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        self.MODEL_DICT = {
            'ViTPose+-G (multi-task train, COCO)': {
                'config': os.path.join(project_root, 'assets/configs/vitpose/ViTPose_huge_wholebody_256x192.py'),
                'model': os.path.join(project_root, 'assets/checkpoints/vitpose_huge_wholebody.pth'),
            },
        }
        
        self.device = torch.device(device)
        self.model_name = model_name
        
        # Override with custom config/checkpoint if provided
        if model_config is not None and model_checkpoint is not None:
            self.MODEL_DICT[model_name] = {
                'config': model_config,
                'model': model_checkpoint,
            }

        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic['model']
        model = init_pose_model(dic['config'], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: List[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: List[np.ndarray],
            box_score_threshold: float = 0.5) -> List[Dict[str, np.ndarray]]:
        """
        det_results: a list of Dict[str, np.ndarray] 'bbox': xyxyc
        """
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=det_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: List[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis

def main(video_dir: str='./demo_data/input_images/arthur_tyler_pass_by_nov20/cam01', bbox_dir: str='./demo_data/input_masks/arthur_tyler_pass_by_nov20/cam01/json_data', output_dir: str='./demo_data/input_2d_poses/arthur_tyler_pass_by_nov20/cam01', model_config: str='./assets/configs/vitpose/ViTPose_huge_wholebody_256x192.py', model_checkpoint: str='./assets/checkpoints/vitpose_huge_wholebody.pth', multihuman: bool = False, top_k: int = 0, vis: bool = False):
    # Load the model
    model = ViTPoseModel(model_config=model_config, model_checkpoint=model_checkpoint)

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
        # person_id_list = [gsam2_meta_data["most_persistent_id"], gsam2_meta_data["largest_area_id"], gsam2_meta_data["highest_confidence_id"]]
        person_id_list = [gsam2_meta_data["largest_area_id"]]
        person_id_counter = Counter(person_id_list)
        majority_voted_person_id = person_id_counter.most_common(1)[0][0]
        # print(f"Majority voted person id: {majority_voted_person_id} using most persistent, largest area, and highest confidence")
        print(f"Majority voted person id: {majority_voted_person_id} using largest area")
    # Pose estimation configuration
    box_score_threshold = 0.5
    kpt_score_threshold = 0.3
    vis_dot_radius = 4
    vis_line_thickness = 1


    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run per image
    img_path_list = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
    if img_path_list == []:
        img_path_list = sorted(glob.glob(os.path.join(video_dir, '*.png')))
    for _, img_path in tqdm(enumerate(img_path_list), total=len(img_path_list)):
        img_idx = int(os.path.splitext(os.path.basename(img_path))[0].split('_')[-1])
        image = cv2.imread(img_path)
        det_result_path = os.path.join(bbox_dir, f'mask_{img_idx:05d}.json')
        bboxes = []
        frame_person_ids = []
        # frame_person_ids is a list of person ids for the current frame
        # ex: frame_person_ids = [1, 2] means there are 2 people in the current frame
        with open(det_result_path, 'r') as f:
            det_results = json.load(f)
            # {"mask_name": "mask_00066.npy", "mask_height": 1280, "mask_width": 720, "promote_type": "mask", "labels": {"1": {"instance_id": 1, "class_name": "person", "x1": 501, "y1": 418, "x2": 711, "y2": 765, "logit": 0.0}, "2": {"instance_id": 2, "class_name": "person", "x1": 0, "y1": 300, "x2": 155, "y2": 913, "logit": 0.0}}}
            # 1: {"instance_id": 1, "class_name": "person", "x1": 501, "y1": 418, "x2": 711, "y2": 765, "logit": 0.0, "score": 0.0}
            # 2: {"instance_id": 2, "class_name": "person", "x1": 0, "y1": 300, "x2": 155, "y2": 913, "logit": 0.0, "score": 0.0}
            # If labels is empty, skip the frame
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

                bbox_dict = {'bbox': np.array([box['x1'], box['y1'], box['x2'], box['y2'], box['score']])}
                bboxes.append(bbox_dict)
                frame_person_ids.append(box['instance_id'])

        if len(bboxes) == 0:
            continue

        out = model.predict_pose(image, bboxes, box_score_threshold)

        # out: List[Dict[str, np.ndarray]]; keys: bbox, keypoints. values are numpy arrays
        # convert values to lists
        save_out = {}

        if multihuman:
            for out_idx, person_id in enumerate(frame_person_ids):
                save_out[person_id] = {
                    'bbox': out[out_idx]['bbox'].tolist(), # bbox is a list of 5 elements: [x1, y1, x2, y2, score]
                    'keypoints': out[out_idx]['keypoints'].tolist(), # keypoints is a length 133 list of 3D coordinates [x, y, score]
                }
        else:
            save_out = {
                majority_voted_person_id: {
                    'bbox': out[0]['bbox'].tolist(),
                    'keypoints': out[0]['keypoints'].tolist(),
                }
            }

        # Save the pose results
        pose_result_path = os.path.join(output_dir, f'pose_{img_idx:05d}.json')
        with open(pose_result_path, 'w') as f:
            json.dump(save_out, f)

        if vis:
            vis_out = model.visualize_pose_results(image, out, kpt_score_threshold,
                                            vis_dot_radius, vis_line_thickness)
            vis_out_path = os.path.join(output_dir, 'vis', f'pose_{img_idx:05d}.jpg')
            Path(os.path.join(output_dir, 'vis')).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(vis_out_path, vis_out)

if __name__ == "__main__":
    tyro.cli(main)


