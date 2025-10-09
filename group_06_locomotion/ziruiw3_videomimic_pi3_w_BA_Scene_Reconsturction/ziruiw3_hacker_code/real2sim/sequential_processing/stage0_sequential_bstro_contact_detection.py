# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

"""
Batch processing for BSTRO contact prediction.
This script processes video (camera) folders:
  • Loads the BSTRO contact prediction model once.
  • Iterates over video folders and then camera subfolders.
  • For each frame, reads the corresponding bounding box file,
    extracts detections, crops the image chunks, batches them,
    runs inference to predict contact probabilities, and
    saves per-frame results as pickle files.
"""

import os
import glob
import json
import pickle
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import tyro
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# ---- Import necessary modules from metro (used in BSTRO model) ----
import metro.modeling.data.config as cfg
cfg.SMPL_FILE = "./assets/body_models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
cfg.JOINT_REGRESSOR_TRAIN_EXTRA = "./assets/body_models/smpl/J_regressor_extra.npy"
cfg.JOINT_REGRESSOR_H36M_correct = "./assets/body_models/smpl/J_regressor_h36m_correct.npy"
cfg.SMPL_sampling_matrix = "./assets/body_models/smpl/mesh_downsampling.npz"

from metro.modeling.bert import BertConfig, METRO
from metro.modeling.bert.modeling_bstro import BSTRO_BodyHSC_Network as BSTRO_Network
from metro.modeling._smpl import SMPL, Mesh
from metro.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from metro.modeling.hrnet.config import config as hrnet_config
from metro.modeling.hrnet.config import update_config as hrnet_update_config
from metro.utils.image_ops import crop
from metro.utils.logger import setup_logger
from metro.utils.comm import synchronize, is_main_process, get_rank
from metro.utils.miscellaneous import mkdir

# ---- Utility function for image pre-processing (copied from get_contact_bstro_feb12.py) ----
def rgb_processing(rgb_img, center, scale, rot, pn, img_res=224):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img, center, scale, [img_res, img_res], rot=rot)
    # Add per-channel pixel noise
    rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
    rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
    rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
    rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
    return rgb_img

# ---- New process_video function modeled after get_batch_smpl_hmr2.py ----
def process_video(
    img_dir: str,
    bbox_dir: str,
    output_dir: str,
    BSTRO_model,
    smpl,
    mesh_sampler,
    device: torch.device,
    batch_size: int = 16,
    feet_contact_ratio_thr: float = 0.25,
    contact_thr: float = 0.98,
):
    """Process a single video (camera) folder for contact prediction."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all images in img_dir (try jpg, then png)
    img_paths = list(Path(img_dir).glob("*.jpg"))
    if len(img_paths) == 0:
        img_paths = list(Path(img_dir).glob("*.png"))
    img_paths.sort()
    
    if len(img_paths) == 0:
        print(f"No images found in {img_dir}")
        return
    
    img_crop_list = []
    meta_info_list = []  # List of tuples (frame_name, person_id)
    
    for img_path in tqdm(img_paths, desc="Processing frames"):
        # Derive frame index from filename (assuming format: something_00001.jpg)
        try:
            frame_idx = int(img_path.stem.split("_")[-1])
        except ValueError:
            frame_idx = None
        if frame_idx is not None:
            bbox_path = Path(bbox_dir) / f"mask_{frame_idx:05d}.json"
        else:
            bbox_path = Path(bbox_dir) / (img_path.stem + ".json")
        
        if not bbox_path.exists():
            print(f"Warning: No bbox file found for frame {img_path.name}")
            continue
        
        with open(bbox_path, "r") as f:
            bbox_data = json.load(f)
        
        # Skip if no detections
        if not bbox_data.get("labels"):
            continue
        
        # Load image and convert to RGB
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process each detection in the bbox file
        for det in bbox_data["labels"].values():
            instance_id = det["instance_id"]
            x1 = int(det["x1"])
            y1 = int(det["y1"])
            x2 = int(det["x2"])
            y2 = int(det["y2"])
            bbox_arr = np.array([x1, y1, x2, y2, 1.0])
            if bbox_arr[:4].sum() == 0:
                continue
            # Ensure coordinates are in bounds
            h_img, w_img = img.shape[:2]
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w_img, x2)
            y2c = min(h_img, y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            img_crop = img[y1c:y2c, x1c:x2c, :]
            img_crop_list.append(img_crop)
            frame_name = img_path.stem  # use the image stem as the frame identifier
            meta_info_list.append((frame_name, instance_id))
    
    if len(img_crop_list) == 0:
        print(f"No valid detections found in {img_dir}")
        return
    
    # Use the first crop to determine center and scale
    h_crop, w_crop = img_crop_list[0].shape[:2]
    center = [w_crop / 2.0, h_crop / 2.0]
    scale = max(h_crop, w_crop) / 200.0
    rot = 0.0
    pn = np.ones(3)
    img_res = 224  # resolution for model input
    
    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    
    images_tensor_list = []
    for crop_img in img_crop_list:
        proc_img = rgb_processing(crop_img, center, scale, rot, pn, img_res=img_res)
        proc_img = torch.from_numpy(proc_img).float()
        proc_img = normalize_img(proc_img)
        proc_img = proc_img.to(device).unsqueeze(0)
        images_tensor_list.append(proc_img)
    images_torch = torch.cat(images_tensor_list, dim=0)  # (N, 3, 224, 224)
    
    # Run batch inference using BSTRO_model
    BSTRO_model.eval()
    pred_contact_list = []
    with torch.no_grad():
        for i in tqdm(range(0, images_torch.shape[0], batch_size), desc="Processing batches"):
            batch_images = images_torch[i : i + batch_size]
            if hasattr(BSTRO_model, "config") and BSTRO_model.config.output_attentions:
                _, _, pred_contact, _, _ = BSTRO_model(batch_images, smpl, mesh_sampler)
            else:
                _, _, pred_contact = BSTRO_model(batch_images, smpl, mesh_sampler)
            pred_contact_list.append(pred_contact)
    pred_contact = torch.cat(pred_contact_list, dim=0)  # (N, 6890, 1)
    
    # Load SMPL segmentation for foot vertices
    smpl_seg_path = "./assets/body_models/smpl/smpl_vert_segmentation.json"
    with open(smpl_seg_path, "r") as f:
        smpl_vert_seg = json.load(f)
    left_foot_vert_ids = np.array(smpl_vert_seg["leftFoot"], dtype=np.int32)
    right_foot_vert_ids = np.array(smpl_vert_seg["rightFoot"], dtype=np.int32)
    
    # Compute foot contact ratios
    pred_contact_left = pred_contact[:, left_foot_vert_ids, :].reshape(pred_contact.shape[0], -1)
    pred_contact_right = pred_contact[:, right_foot_vert_ids, :].reshape(pred_contact.shape[0], -1)
    left_ratio = (pred_contact_left > contact_thr).float().mean(dim=1).cpu().numpy()
    right_ratio = (pred_contact_right > contact_thr).float().mean(dim=1).cpu().numpy()
    
    # Build results dictionary (keyed by frame name, then person id)
    pred_contact_dict = defaultdict(dict)
    for i, (frame_name, person_id) in enumerate(meta_info_list):
        contact_pred = pred_contact[i].cpu().numpy()
        left_contact = left_ratio[i] > feet_contact_ratio_thr
        right_contact = right_ratio[i] > feet_contact_ratio_thr
        pred_contact_dict[frame_name][person_id] = {
            "frame_contact_vertices": contact_pred,
            "left_foot_contact": left_contact,
            "right_foot_contact": right_contact,
        }
    
    print("Saving results...")
    for frame_name in sorted(pred_contact_dict.keys()):
        save_path = os.path.join(output_dir, f"{frame_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(pred_contact_dict[frame_name], f)
        print(f"Saved contact predictions for {frame_name} to {save_path}")

# ---- Args for batch processing. Note these combine video folder settings with the model params ----
@dataclass
class Args:
    video_base_dir: str = "./demo_data/input_images"  # Base dir containing video folders
    bbox_base_dir: str = "./demo_data/input_masks"  # Base directory for input bboxes
    output_base_dir: str = "./demo_data/input_contacts"  # Base directory for output contact files
    pattern: str = ""  # Optionally filter video folders by a pattern
    resume_checkpoint: str = "./assets/checkpoints/hsi_hrnet_3dpw_b32_checkpoint_15.bin"
    batch_size: int = 16
    feet_contact_ratio_thr: float = 0.25
    contact_thr: float = 0.98
    start_idx: int = 0
    end_idx: int = -1
    #########################################################
    # Model architecture parameters
    #########################################################
    arch: str = "hrnet-w64"
    num_hidden_layers: int = 4
    hidden_size: int = -1
    num_attention_heads: int = 4
    intermediate_size: int = -1
    input_feat_dim: str = "2051,512,128"
    hidden_feat_dim: str = "1024,256,128"
    legacy_setting: bool = True
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    model_name_or_path: str = "metro/modeling/bert/bert-base-uncased/"
    config_name: str = "bert-base-uncased"
    #########################################################
    # Others
    #########################################################
    device: str = "cuda"
    output_attentions: bool = False

# ---- Main function that loads the contact model once and processes video folders ----
def main(args: Args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create the output base directory first, before setting up the logger
    mkdir(args.output_base_dir)
    logger = setup_logger("BSTRO", args.output_base_dir, get_rank())
    
    # Build BSTRO model (load once)
    smpl = SMPL().to(device)
    mesh_sampler = Mesh()
    
    trans_encoder = []
    input_feat_dim_list = [int(x) for x in args.input_feat_dim.split(",")]
    hidden_feat_dim_list = [int(x) for x in args.hidden_feat_dim.split(",")]
    output_feat_dim = input_feat_dim_list[1:] + [1]
    
    for i in range(len(output_feat_dim)):
        config = BertConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path
        )
        config.output_attentions = args.output_attentions
        config.hidden_dropout_prob = 0.1
        config.img_feature_dim = input_feat_dim_list[i]
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim_list[i]
        if args.legacy_setting:
            args.intermediate_size = -1
        else:
            args.intermediate_size = int(args.hidden_size * 4)
    
        update_params = ["num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size"]
        for param in update_params:
            arg_val = getattr(args, param)
            config_val = getattr(config, param)
            if arg_val > 0 and arg_val != config_val:
                logger.info(f"Update config parameter {param}: {config_val} -> {arg_val}")
                setattr(config, param, arg_val)
    
        assert config.hidden_size % config.num_attention_heads == 0
        model = METRO(config=config)
        logger.info("Initialized a transformer encoder block.")
        trans_encoder.append(model)
    
    trans_encoder = torch.nn.Sequential(*trans_encoder)
    total_params = sum(p.numel() for p in trans_encoder.parameters())
    logger.info(f"Transformers total parameters: {total_params}")
    
    # Load backbone model based on the chosen architecture
    if args.arch == "hrnet":
        hrnet_yaml = "models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
        hrnet_checkpoint = "models/hrnet/hrnetv2_w40_imagenet_pretrained.pth"
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info("=> Loaded hrnet-v2-w40 model")
    elif args.arch == "hrnet-w64":
        hrnet_yaml = "assets/configs/bstro_hrnet_w64.yaml"
        hrnet_checkpoint = "models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info("=> Loaded hrnet-v2-w64 model")
    else:
        print(f"=> Using pre-trained model '{args.arch}'")
        from torchvision import models
        backbone = models.__dict__[args.arch](pretrained=True)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    
    _bstro_network = BSTRO_Network(args, config, backbone, trans_encoder, mesh_sampler)
    if args.resume_checkpoint and args.resume_checkpoint != "None":
        logger.info(f"Evaluation: Loading from checkpoint {args.resume_checkpoint}")
        state_dict = torch.load(args.resume_checkpoint, map_location="cpu")
        _bstro_network.load_state_dict(state_dict, strict=False)
        del state_dict
    else:
        raise ValueError(f"Invalid checkpoint {args.resume_checkpoint}")
    
    _bstro_network.to(device)
    _bstro_network.eval()
    
    # Process all video folders (each assumed to be a folder inside video_base_dir)
    video_folders = [
        f
        for f in os.listdir(args.video_base_dir)
        if os.path.isdir(os.path.join(args.video_base_dir, f))
        and (not args.pattern or args.pattern.lower() in f.lower())
    ]
    video_folders = sorted(video_folders)
    if args.end_idx != -1:
        video_folders = video_folders[args.start_idx:args.end_idx]
        
    
    logger.info(f"Found {len(video_folders)} video folders matching pattern '{args.pattern}'")
    if len(video_folders) == 0:
        print("No matching video folders found!")
        return
    
    for video_name in video_folders:
        video_dir = os.path.join(args.video_base_dir, video_name)
        cam_folders = [f for f in os.listdir(video_dir) if f.startswith("cam")]
        logger.info(f"Processing video {video_name} with {len(cam_folders)} cameras")
        for cam in cam_folders:
            input_dir = os.path.join(video_dir, cam)
            bbox_dir = os.path.join(args.bbox_base_dir, video_name, cam, "json_data")
            output_dir = os.path.join(args.output_base_dir, video_name, cam)
            if not os.path.exists(bbox_dir):
                print(f"Warning: No bbox directory found for {video_name}/{cam}")
                continue
            logger.info(f"Processing {video_name}/{cam}")
            try:
                process_video(
                    input_dir,
                    bbox_dir,
                    output_dir,
                    _bstro_network,
                    smpl,
                    mesh_sampler,
                    device,
                    batch_size=args.batch_size,
                    feet_contact_ratio_thr=args.feet_contact_ratio_thr,
                    contact_thr=args.contact_thr,
                )
            except Exception as e:
                print(f"Error processing {video_name}/{cam}: {e}")
                continue

if __name__ == "__main__":
    config = tyro.cli(Args)
    main(config)
