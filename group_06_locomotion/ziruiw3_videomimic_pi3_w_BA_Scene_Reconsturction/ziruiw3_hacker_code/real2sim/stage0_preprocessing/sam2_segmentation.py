# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import os
import sys
import cv2
import time
import tyro
import shutil
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# Add third_party paths to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
grounded_sam2_path = os.path.join(project_root, "third_party", "Grounded-SAM-2")
sys.path.insert(0, grounded_sam2_path)

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from sam2.gdsam2_utils.video_utils import create_video_from_images
from sam2.gdsam2_utils.common_utils import CommonUtils
from sam2.gdsam2_utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy


def main(text: str="person.", video_dir: str="./demo_data/input_images/arthur_tyler_pass_by_nov20/cam01", output_dir: str="./demo_data/input_masks/arthur_tyler_pass_by_nov20/cam01", vis: bool=False, offload_video_to_cpu: bool=False, async_loading_frames: bool=False):
    """Main function for Grounding-SAM-2

    Args:
        text: text queries need to be lowercased + end with a dot
        video_dir: directory of JPEG frames with filenames like `<frame_index>.jpg` // No png files supported by SAM2 but I (Hongsuk) just added some code to copy png to jpg
        output_dir: directory to save the annotated frames and video
        vis: whether to visualize the results
    """
        
    step = 20 # the step to sample frames for Grounding DINO predictor

    
    """
    Step 1: Environment settings and model initialization
    """
    # use bfloat16 for the entire notebook
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sam2_checkpoint = os.path.join(project_root, "assets/ckpt_sam2/sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # don't change this line, it is used in the build_sam2_video_predictor function. it's relative path in sam2 api.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # Load checkpoints
    if sam2_checkpoint is not None:
        checkpoint = torch.load(sam2_checkpoint, map_location="cpu", weights_only=True)
        video_predictor.load_state_dict(checkpoint["model"])
        sam2_image_model.load_state_dict(checkpoint["model"])
    
    # Set device and eval mode
    video_predictor = video_predictor.to(device)
    sam2_image_model = sam2_image_model.to(device)
    video_predictor.eval()
    sam2_image_model.eval()
    
    image_predictor = SAM2ImagePredictor(sam2_image_model)


    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # create the output directory
    CommonUtils.creat_dirs(output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    try:
        new_video_dir = ''
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    except:
        print("Only JPEG frames are supported by SAM2, whose file names should be like <frame_index>.jpg")
        print("Copying the frames to a new directory and renaming them to <frame_index>.jpg...")
        if video_dir[-1] == '/':
            video_dir = video_dir[:-1]
        new_video_dir = os.path.join(os.path.dirname(video_dir), f"new_{os.path.basename(video_dir)}")
        CommonUtils.creat_dirs(new_video_dir)

        frame_names.sort()
        new_frame_names = []
        for _, fn in tqdm(enumerate(frame_names)):
            # extract frame index from the filename
            # frame_name is like this: frame_00000.jpg
            frame_idx = int(os.path.splitext(fn)[0].split('_')[-1])

            # copy the frame to the new directory
            new_frame_name = f"{frame_idx:05d}.jpg"
            shutil.copy(os.path.join(video_dir, fn), os.path.join(new_video_dir, new_frame_name))
            new_frame_names.append(new_frame_name)
        
        video_dir = new_video_dir
        frame_names = new_frame_names
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Get the first frame's image size; height and width
    img_path = os.path.join(video_dir, frame_names[0])
    image = Image.open(img_path)
    width, height = image.size

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=offload_video_to_cpu, async_loading_frames=async_loading_frames)

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
    objects_count = 0

    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
    """
    print("Total frames:", len(frame_names))
    for start_frame_idx in range(0, len(frame_names), step):
    # prompt grounding dino to get the box coordinates on specific frame
        print("start_frame_idx", start_frame_idx)
        # continue
        img_path = os.path.join(video_dir, frame_names[start_frame_idx])
        image = Image.open(img_path)
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

        # run Grounding DINO on the image
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )

        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(np.array(image.convert("RGB")))

        # process the detection results
        input_boxes = results[0]["boxes"] # .cpu().numpy()
        # print("results[0]",results[0])
        OBJECTS = results[0]["labels"]
        if input_boxes.shape[0] != 0:
            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            # convert the mask shape to (n, H, W)
            if masks.ndim == 2:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            """
            Step 3: Register each object's positive points to video predictor
            """

            # If you are using point prompts, we uniformly sample positive points based on the mask
            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS, scores_list=torch.tensor(scores).to(device))
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")


            """
            Step 4: Propagate the video predictor to get the segmentation results for each frame
            """
            objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
            print("objects_count", objects_count)
        else:
            print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
            mask_dict = sam2_masks

        
        if len(mask_dict.labels) == 0:
            mask_dict.mask_height = height
            mask_dict.mask_width = width
            mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list = frame_names[start_frame_idx:start_frame_idx+step])
            print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
            continue
        else: 
            video_predictor.reset_state(inference_state)

            # Hongsuk added
            score_dict = {}
            for object_id, object_info in mask_dict.labels.items():
                score = object_info.score
                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                        inference_state,
                        start_frame_idx,
                        object_id,
                        object_info.mask,
                    )
                # Hongsuk added
                score_dict[out_obj_ids[-1]] = score

            
            video_segments = {}  # output the following {step} frames tracking masks
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                frame_masks = MaskDictionaryModel()
                
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                    object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                    object_info.update_box()
                    object_info.score = score_dict[out_obj_id]
                    frame_masks.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                    frame_masks.mask_name = f"mask_{image_base_name}.npy"
                    frame_masks.mask_height = out_mask.shape[-2]
                    frame_masks.mask_width = out_mask.shape[-1]

                video_segments[out_frame_idx] = frame_masks
                sam2_masks = copy.deepcopy(frame_masks)

            print("video_segments:", len(video_segments))
        """
        Step 5: save the tracking masks and json files
        """
        for frame_idx, frame_masks_info in video_segments.items():
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id

            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

            # Calculate areas and sort objects by area for this frame
            areas_by_id = {}
            for obj_id, obj_info in mask.items():
                x1, y1, x2, y2 = obj_info.x1, obj_info.y1, obj_info.x2, obj_info.y2
                area = (x2 - x1) * (y2 - y1)
                areas_by_id[obj_id] = area
            
            # Sort object IDs by area (largest first)
            sorted_ids = sorted(areas_by_id.keys(), key=lambda x: areas_by_id[x], reverse=True)
            
            json_data = frame_masks_info.to_dict()
            # Add area ranking information
            json_data['area_ranking'] = sorted_ids  # List of object IDs sorted by area (largest first)
            json_data['areas'] = areas_by_id  # Dictionary of {object_id: area}
            
            json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
            with open(json_data_path, "w") as f:
                json.dump(json_data, f)
    
    """
    Step 5.5: Analyze object persistence and accumulated areas
    """
    from collections import defaultdict
    meta_data = {}
    all_ids = set()
    # Dictionaries to store statistics for each object
    frame_counts = defaultdict(int)  # {object_id: number_of_frames}
    total_areas = defaultdict(float)  # {object_id: accumulated_box_areas}
    # Find object with highest accumulated confidence scores
    confidence_scores = defaultdict(float)  # {object_id: accumulated_confidence}

    # Analyze saved JSON files
    json_files = sorted([f for f in os.listdir(json_data_dir) if f.endswith('.json')])
    for json_file in json_files:
        json_path = os.path.join(json_data_dir, json_file)
        with open(json_path, 'r') as f:
            frame_data = json.load(f)
        
        # Process each object in the frame
        for _, obj_info in frame_data['labels'].items():
            obj_id = int(obj_info['instance_id'])  # Convert string ID to in
            all_ids.add(obj_id)
            frame_counts[obj_id] += 1
            
            # Calculate and accumulate bounding box area
            x1, y1, x2, y2 = obj_info['x1'], obj_info['y1'], obj_info['x2'], obj_info['y2']
            area = (x2 - x1) * (y2 - y1)
            total_areas[obj_id] += area

            confidence_scores[obj_id] += obj_info['score']
    # Find object with most appearances
    if frame_counts:
        most_persistent_id = max(frame_counts.items(), key=lambda x: x[1])
        print(f"\nMost persistent object:")
        print(f"Object ID: {most_persistent_id[0]}")
        print(f"Appeared in {most_persistent_id[1]} frames")
        print(f"Total area: {total_areas[most_persistent_id[0]]:.2f}")
        meta_data["most_persistent_id"] = most_persistent_id[0]

    # Find object with largest accumulated area
    if total_areas:
        largest_area_id = max(total_areas.items(), key=lambda x: x[1])
        print(f"\nObject with largest accumulated area:")
        print(f"Object ID: {largest_area_id[0]}")
        print(f"Appeared in {frame_counts[largest_area_id[0]]} frames")
        print(f"Total area: {largest_area_id[1]:.2f}")
        meta_data["largest_area_id"] = largest_area_id[0]
    
    # Find object with highest confidence
    if confidence_scores:
        highest_conf_id = max(confidence_scores.items(), key=lambda x: x[1])
        print(f"\nObject with highest accumulated confidence:")
        print(f"Object ID: {highest_conf_id[0]}")
        print(f"Appeared in {frame_counts[highest_conf_id[0]]} frames") 
        print(f"Total confidence: {highest_conf_id[1]:.2f}")
        print(f"Average confidence: {highest_conf_id[1]/frame_counts[highest_conf_id[0]]:.2f}")
        meta_data["highest_confidence_id"] = highest_conf_id[0]


    # Calculate average areas for ranking
    avg_areas = {}
    for obj_id in all_ids:
        if frame_counts.get(obj_id, 0) > 0:
            avg_areas[obj_id] = total_areas.get(obj_id, 0) / frame_counts[obj_id]
        else:
            avg_areas[obj_id] = 0
    
    # Sort all object IDs by average area
    sorted_by_avg_area = sorted(all_ids, key=lambda x: avg_areas[x], reverse=True)
    
    # save meta_data to json
    meta_data["all_instance_ids"] = sorted(list(all_ids))
    meta_data["sorted_by_avg_area"] = sorted_by_avg_area  # IDs sorted by average area per frame
    meta_data["avg_areas"] = avg_areas  # Dictionary of {object_id: average_area}
    meta_data["frame_counts"] = dict(frame_counts)  # Convert defaultdict to dict
    meta_data["total_areas"] = dict(total_areas)  # Convert defaultdict to dict
    
    meta_data_path = os.path.join(output_dir, "meta_data.json")
    with open(meta_data_path, "w") as f:
        json.dump(meta_data, f)

    # # Print all objects statistics
    # print("\nAll objects statistics:")
    # for obj_id in sorted(frame_counts.keys()):
    #     print(f"Object {obj_id} ({object_classes[obj_id]}):")
    #     print(f"  Frames: {frame_counts[obj_id]}")
    #     print(f"  Total area: {total_areas[obj_id]:.2f}")
    #     print(f"  Average area per frame: {total_areas[obj_id]/frame_counts[obj_id]:.2f}")    
        
    """
    Step 6: Draw the results and save the video
    """
    if vis: 
        frame_rate = 30
        CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)
        output_video_path = os.path.join(output_dir, "sam2_output.mp4")
        create_video_from_images(result_dir, output_video_path, frame_rate=frame_rate)

    if new_video_dir:
        print("Removing the new video directory...")
        shutil.rmtree(new_video_dir)

    # Reload the saved npy masks and save them into npz files and remove the npy files
    mask_data_dir = os.path.join(output_dir, "mask_data")
    for file in os.listdir(mask_data_dir):
        if file.endswith(".npy"):
            mask = np.load(os.path.join(mask_data_dir, file))
            np.savez_compressed(os.path.join(mask_data_dir, file.replace(".npy", ".npz")), mask=mask)
            os.remove(os.path.join(mask_data_dir, file))

if __name__ == "__main__":
    tyro.cli(main)