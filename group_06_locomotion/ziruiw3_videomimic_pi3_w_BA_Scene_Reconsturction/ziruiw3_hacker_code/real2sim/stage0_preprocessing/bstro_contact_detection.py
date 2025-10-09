"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
You can only use this computer program if you have closed
a license agreement with MPG or you get the right to use the computer
program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and
liable to prosecution.

Copyright©2022 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems and the Max Planck Institute for Biological
Cybernetics. All rights reserved.

Contact: ps-license@tuebingen.mpg.de

"""

# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

from __future__ import absolute_import, division, print_function
import argparse
import os
import glob
import json
import pickle
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import numpy as np
import cv2
import tyro
from dataclasses import dataclass
from collections import defaultdict


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


@dataclass
class Args:
    # Custom arguments
    feet_contact_ratio_thr: float = 0.25
    contact_thr: float = 0.98
    resume_checkpoint: str = './assets/checkpoints/hsi_hrnet_3dpw_b32_checkpoint_15.bin'
    video_dir: str = './demo_data/input_images/IMG_7379-IMG_7379-00.02.20.119-00.02.25.377-seg10/cam01'
    bbox_dir: str = './demo_data/input_masks/IMG_7379-IMG_7379-00.02.20.119-00.02.25.377-seg10/cam01/json_data'
    output_dir: str = './demo_data/input_contacts/IMG_7379-IMG_7379-00.02.20.119-00.02.25.377-seg10/cam01'

    #########################################################
    # Model architectures
    #########################################################
    arch: str = 'hrnet-w64'
    num_hidden_layers: int = 4
    hidden_size: int = -1
    num_attention_heads: int = 4
    intermediate_size: int = -1
    input_feat_dim: str = '2051,512,128'
    hidden_feat_dim: str = '1024,256,128'
    legacy_setting: bool = True
    
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    model_name_or_path: str = 'metro/modeling/bert/bert-base-uncased/'
    config_name: str = "bert-base-uncased"
    #########################################################
    # Others
    #########################################################
    device: str = 'cuda'
    output_attentions: bool = False

def rgb_processing(rgb_img, center, scale, rot, pn, img_res=224):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img, center, scale, [img_res, img_res], rot=rot)
    
    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
    rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
    rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
    # (3,224,224),float,[0,1]
    rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
    return rgb_img

def run_inference(args, BSTRO_model, smpl, mesh_sampler):
    smpl.eval()

    if args.distributed:
        BSTRO_model = torch.nn.parallel.DistributedDataParallel(
            BSTRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    BSTRO_model.eval()

    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    smpl_vert_seg_path = "./assets/body_models/smpl/smpl_vert_segmentation.json"
    with open(smpl_vert_seg_path, 'r') as f:
        smpl_vert_seg = json.load(f)
    left_foot_vert_ids = np.array(smpl_vert_seg['leftFoot'], dtype=np.int32) # 'leftToeBase'
    right_foot_vert_ids = np.array(smpl_vert_seg['rightFoot'], dtype=np.int32)
    contact_thr = args.contact_thr

    with torch.no_grad():
        img_dir = args.video_dir
        bbox_dir = args.bbox_dir
        img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if len(img_path_list) == 0:
            img_path_list = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        bbox_path_list = sorted(glob.glob(os.path.join(bbox_dir, '*.json')))
        img_crop_list = []
        img_name_person_id_tuple_list = []
        for img_path, bbox_path in zip(img_path_list, bbox_path_list):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # read bbox
            with open(bbox_path, 'r') as f:
                det_results = json.load(f)
            
            # Skip if no detections
            if 'labels' not in det_results or len(det_results['labels']) == 0:
                print(f"Warning: No detections in {bbox_path}")
                continue
                
            for box in det_results['labels'].values():
                bbox = np.array([int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2']), 1.0])
                instance_id = box['instance_id']

                # Skip if invalid boxes
                bboxes_sum = sum([bbox[:4].sum()])
                if bboxes_sum == 0:
                    print(f"Warning: Invalid bbox in {bbox_path}")
                    continue
                
                # Check if bbox is valid (width and height > 0)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    print(f"Warning: Invalid bbox dimensions in {bbox_path}: {bbox}")
                    continue
            
                # crop the image with the bbox
                try:
                    img_crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
                    if img_crop.size == 0:
                        print(f"Warning: Empty crop from {img_path} with bbox {bbox}")
                        continue
                        
                    img_crop_list.append(img_crop)
                    # just add the file name, without the directory and extension
                    img_name_person_id_tuple_list.append((os.path.basename(img_path).split('.')[0], instance_id))
                except Exception as e:
                    print(f"Error cropping image {img_path} with bbox {bbox}: {e}")
                    continue

        if len(img_crop_list) == 0:
            print("No valid image crops found. Exiting.")
            return {}
            
        h, w = img_crop_list[0].shape[:2]
        center = [ w / 2., h / 2.]
        scale = max(h, w) / 200.0

        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        images_torch = []
        for img in img_crop_list:
            img = rgb_processing(img, center, sc*scale, rot, pn)
            img = torch.from_numpy(img).float()
            # Store image before normalization to use it in visualization
            images = normalize_img(img)
            images = images.cuda(args.device).unsqueeze(0)
            images_torch.append(images)

        images_torch = torch.cat(images_torch, dim=0)

        # forward-pass
        pred_contact_list = []
        # batchify the images torch with batch size of 16
        for i in range(0, images_torch.shape[0], 16):
            batch_images = images_torch[i:i+16]
            if BSTRO_model.config.output_attentions:
                _, _, pred_contact, hidden_states, att = BSTRO_model(batch_images, smpl, mesh_sampler)
            else:
                _, _, pred_contact = BSTRO_model(batch_images, smpl, mesh_sampler)
            pred_contact_list.append(pred_contact)

        pred_contact = torch.cat(pred_contact_list, dim=0) # (num_images, 6890, 1)
        pred_contact_left_foot = pred_contact[:, left_foot_vert_ids, :].reshape(pred_contact.shape[0], -1)
        pred_contact_right_foot = pred_contact[:, right_foot_vert_ids, :].reshape(pred_contact.shape[0], -1)

        left_foot_contact_ratio = (pred_contact_left_foot > contact_thr).float().mean(dim=1).cpu().numpy()
        right_foot_contact_ratio = (pred_contact_right_foot > contact_thr).float().mean(dim=1).cpu().numpy()

        # Save the pred_contact as a dictionary with the image name as the key and the pred_contact as the value
        pred_contact_dict = defaultdict(dict) # Dict[img_name, Dict[person_id, Dict[left_foot_contact, right_foot_contact]]]
        for i, (img_name, person_id) in enumerate(img_name_person_id_tuple_list):
            frame_pred_contact = pred_contact[i].cpu().numpy()
            left_foot_contact = left_foot_contact_ratio[i] > args.feet_contact_ratio_thr
            right_foot_contact = right_foot_contact_ratio[i] > args.feet_contact_ratio_thr
            pred_contact_dict[img_name][person_id] = {
                'frame_contact_vertices': frame_pred_contact,
                'left_foot_contact': left_foot_contact,
                'right_foot_contact': right_foot_contact
            }

    return pred_contact_dict


def visualize_contact(images,
                    pred_contact, 
                    smpl):
    ref_vert = smpl(torch.zeros((1, 72)).cuda(args.device), torch.zeros((1,10)).cuda(args.device)).squeeze()
    rend_imgs = []
    pred_contact_meshes = []
    batch_size = pred_contact.shape[0]

    import trimesh
    # Do visualization for the first 6 images of the batch

    for i in range(min(batch_size, 50)):
        img = images[i].cpu().numpy()
        # Get predict vertices for the particular example
        contact = pred_contact[i].cpu()
        hit_id = (contact >= 0.5).nonzero()[:,0]

        pred_mesh = trimesh.Trimesh(vertices=ref_vert.detach().cpu().numpy(), faces=smpl.faces.detach().cpu().numpy(), process=False)
        pred_mesh.visual.vertex_colors = (191, 191, 191, 255)
        pred_mesh.visual.vertex_colors[hit_id, :] = (255, 0, 0, 255)
        pred_contact_meshes.append(pred_mesh)

        # Visualize reconstruction and detected pose
        rend_imgs.append(torch.from_numpy(img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs, pred_contact_meshes


def main(args: Args):
    # Moving other arguments here as variables
    arch = args.arch
    num_hidden_layers = args.num_hidden_layers
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    intermediate_size = args.intermediate_size
    input_feat_dim = args.input_feat_dim
    hidden_feat_dim = args.hidden_feat_dim
    legacy_setting = args.legacy_setting
    model_name_or_path = args.model_name_or_path
    config_name = args.config_name
    device = args.device
    output_attentions = args.output_attentions

    global logger
    # Setup CUDA, GPU & distributed training
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = num_gpus > 1
    device = torch.device(device)
    if args.distributed:
        # print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), int(os.environ["NODE_RANK"]), args.num_gpus))
        print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), 0, args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        synchronize()

    mkdir(args.output_dir)
    logger = setup_logger("BSTRO", args.output_dir, get_rank())
    # set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL().to(device)
    mesh_sampler = Mesh()

    # Load model
    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = input_feat_dim[1:] + [1]

    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, METRO
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)

        config.output_attentions = args.output_attentions
        config.hidden_dropout_prob = 0.1
        config.img_feature_dim = input_feat_dim[i] 
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]

        if args.legacy_setting==True:
            # During our paper submission, we were using the original intermediate size, which is 3072 fixed
            # We keep our legacy setting here 
            args.intermediate_size = -1
        else:
            # We have recently tried to use an updated intermediate size, which is 4*hidden-size.
            # But we didn't find significant performance changes on Human3.6M (~36.7 PA-MPJPE)
            args.intermediate_size = int(args.hidden_size*4)

        # update model structure if specified in arguments
        update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

        for idx, param in enumerate(update_params):
            arg_param = getattr(args, param)
            config_param = getattr(config, param)
            if arg_param > 0 and arg_param != config_param:
                logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                setattr(config, param, arg_param)

        # init a transformer encoder and append it to a list
        assert config.hidden_size % config.num_attention_heads == 0
        model = model_class(config=config) 
        logger.info("Init model from scratch.")
        trans_encoder.append(model)

    # init ImageNet pre-trained backbone model
    if args.arch=='hrnet':
        hrnet_yaml = 'assets/configs/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = 'assets/checkpoints/hrnetv2_w40_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loading hrnet-v2-w40 model')
    elif args.arch=='hrnet-w64':
        hrnet_yaml = 'assets/configs/bstro_hrnet_w64.yaml'
        hrnet_checkpoint = 'assets/checkpoints/hrnetv2_w64_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yaml)
        backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
        logger.info('=> loading hrnet-v2-w64 model')
    else:
        print("=> using pre-trained model '{}'".format(args.arch))
        backbone = models.__dict__[args.arch](pretrained=True)
        # remove the last fc layer
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    trans_encoder = torch.nn.Sequential(*trans_encoder)
    total_params = sum(p.numel() for p in trans_encoder.parameters())
    logger.info('Transformers total parameters: {}'.format(total_params))
    backbone_total_params = sum(p.numel() for p in backbone.parameters())
    logger.info('Backbone total parameters: {}'.format(backbone_total_params))

    # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
    _bstro_network = BSTRO_Network(args, config, backbone, trans_encoder, mesh_sampler)

    if args.resume_checkpoint!=None and args.resume_checkpoint!='None':# and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _bstro_network.load_state_dict(state_dict, strict=False)
        del state_dict
    else:
        raise ValueError("Invalid checkpoint {}".format(args.resume_checkpoint))
    
    _bstro_network.to(device)
    pred_contact_dict = run_inference(args, _bstro_network, smpl, mesh_sampler)
    
    # save as the pickle file for each frame
    for img_name, person_id_contact_dict in pred_contact_dict.items():
        with open(os.path.join(args.output_dir, f'{img_name}.pkl'), 'wb') as f:
            pickle.dump(person_id_contact_dict, f)
        print(f"Pred contact saved to {os.path.join(args.output_dir, f'{img_name}.pkl')}")

if __name__ == "__main__":
    config = tyro.cli(Args)
    main(config)
