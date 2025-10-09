import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from lib.utils.eval_utils import *
from lib.utils.rotation_conversions import *
from lib.vis.traj import *
from lib.camera.slam_utils import eval_slam
from sloper4d_loader import SLOPER4D_Dataset

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='results/sloper4d_seq008')
parser.add_argument('--pred_smpl_path', type=str, default='results/sloper4d_seq008/hps_combined_track_0.npy')
parser.add_argument('--gt_pkl_path', type=str, default='demo_data/sloper4d/seq008_running_001/seq008_running_001_labels.pkl')

parser.add_argument('--visualize', action='store_true', help='Visualize trajectories')
parser.add_argument('--grid_size', type=int, default=5, help='Grid size for trajectory visualization')
args = parser.parse_args()
output_dir = args.output_dir

# Create visualization directory
vis_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)

# Load Sloper4D dataset
sloper4d_data = SLOPER4D_Dataset(
    args.gt_pkl_path, 
    device='cpu',  # Use CPU so we don't waste GPU memory
    return_torch=True,  # Get tensors directly
    fix_pts_num=False,  # No need to fix points number for evaluation
    print_info=False,   # Suppress detailed info
    return_smpl=True    # We need SMPL data
)

# SMPL
SMPL_DIR = '../assets/body_models/smpl' # smpl models are in the smpl subdirectory
smpl = SMPL(model_path=SMPL_DIR)
smpls = {g:SMPL(model_path=SMPL_DIR, gender=g) for g in ['neutral', 'male', 'female']}

# Evaluations: world-coordinate SMPL
accumulator = defaultdict(list)
m2mm = 1e3
human_traj = {}
cam_traj = {}
total_invalid = 0

# Process Sloper4D dataset
seq_name = os.path.basename(args.gt_pkl_path).split('_labels.pkl')[0]
print(f"Evaluating sequence: {seq_name}")

# Get ground truth data
gender = sloper4d_data.smpl_gender
# Check if smpl_pose is already a numpy array or a torch tensor
if isinstance(sloper4d_data.smpl_pose, np.ndarray):
    poses_body = sloper4d_data.smpl_pose[:, 3:]
    poses_root = sloper4d_data.smpl_pose[:, :3]
else:
    poses_body = sloper4d_data.smpl_pose[:, 3:].numpy()
    poses_root = sloper4d_data.smpl_pose[:, :3].numpy()

# Check if betas is a list, numpy array, or torch tensor
if isinstance(sloper4d_data.betas, list):
    betas_array = np.array(sloper4d_data.betas)
    betas = np.repeat(betas_array.reshape(1, -1), repeats=sloper4d_data.length, axis=0)
elif isinstance(sloper4d_data.betas, np.ndarray):
    betas = np.repeat(sloper4d_data.betas.reshape(1, -1), repeats=sloper4d_data.length, axis=0)
else:
    betas = np.repeat(sloper4d_data.betas.numpy().reshape(1, -1), repeats=sloper4d_data.length, axis=0)

# Check if global_trans is already a numpy array or a torch tensor
if isinstance(sloper4d_data.global_trans, np.ndarray):
    trans = sloper4d_data.global_trans
else:
    trans = sloper4d_data.global_trans.numpy()

# Get camera information
if isinstance(sloper4d_data.cam_pose[0], np.ndarray):
    cam_pose = np.array(sloper4d_data.cam_pose)
else:
    cam_pose = np.array([pose.numpy() for pose in sloper4d_data.cam_pose])
ext = np.zeros_like(cam_pose)
for i in range(len(cam_pose)):
    # Convert from world2cam to cam2world format needed by the evaluation
    R = cam_pose[i, :3, :3]
    t = cam_pose[i, :3, 3]
    ext[i, :3, :3] = R
    ext[i, :3, 3] = t

# Create valid mask (all frames are considered valid for Sloper4D)
valid = np.ones(sloper4d_data.length, dtype=bool)
total_invalid += (~valid).sum()

# Extract intrinsics
intrinsics = np.zeros((3, 3))
intrinsics[0, 0] = sloper4d_data.cam['intrinsics'][0]  # fx
intrinsics[1, 1] = sloper4d_data.cam['intrinsics'][1]  # fy
intrinsics[0, 2] = sloper4d_data.cam['intrinsics'][2]  # cx
intrinsics[1, 2] = sloper4d_data.cam['intrinsics'][3]  # cy
intrinsics[2, 2] = 1.0

tt = lambda x: torch.from_numpy(x).float()
gt = smpls[gender](body_pose=tt(poses_body), global_orient=tt(poses_root), betas=tt(betas), transl=tt(trans),
                pose2rot=True, default_smpl=True)
gt_vert = gt.vertices
gt_j3d = gt.joints[:,:24] 
gt_ori = axis_angle_to_matrix(tt(poses_root))

# Groundtruth local motion
poses_root_cam = matrix_to_axis_angle(tt(ext[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root)))
gt_cam = smpls[gender](body_pose=tt(poses_body), global_orient=poses_root_cam, betas=tt(betas),
                        pose2rot=True, default_smpl=True)
gt_vert_cam = gt_cam.vertices
gt_j3d_cam = gt_cam.joints[:,:24] 

# PRED
pred_smpl = np.load(args.pred_smpl_path, allow_pickle=True).item()

pred_rotmat = torch.tensor(pred_smpl['pred_rotmat'])    # T, 24, 3, 3
pred_shape = torch.tensor(pred_smpl['pred_shape'])      # T, 10
pred_trans = torch.tensor(pred_smpl['pred_trans'])      # T, 1, 3

mean_shape = pred_shape.mean(dim=0, keepdim=True)
pred_shape = mean_shape.repeat(len(pred_shape), 1)

pred = smpls['neutral'](body_pose=pred_rotmat[:,1:], 
                        global_orient=pred_rotmat[:,[0]], 
                        betas=pred_shape, 
                        transl=pred_trans.squeeze(),
                        pose2rot=False, 
                        default_smpl=True)
pred_vert = pred.vertices
pred_j3d = pred.joints[:, :24]

# subtract the root joint and add pred_transl
# pred_j3d = pred_j3d - pred_j3d[:, [0]]
# pred_j3d = pred_j3d + pred_trans[:,0]

pred_vert_w = pred_vert
pred_j3d_w = pred_j3d
pred_ori_w = pred_rotmat[:,0]

# Make sure the predicted data matches GT size
min_len = min(len(gt_j3d), len(pred_j3d_w))
gt_j3d = gt_j3d[:min_len]
gt_ori = gt_ori[:min_len]
pred_j3d_w = pred_j3d_w[:min_len]
pred_ori_w = pred_ori_w[:min_len]
gt_j3d_cam = gt_j3d_cam[:min_len]
gt_vert_cam = gt_vert_cam[:min_len]
pred_j3d = pred_j3d[:min_len]
pred_vert = pred_vert[:min_len]
valid = valid[:min_len]

# Apply valid mask
gt_j3d = gt_j3d[valid]
gt_ori = gt_ori[valid]
pred_j3d_w = pred_j3d_w[valid]
pred_ori_w = pred_ori_w[valid]
gt_j3d_cam = gt_j3d_cam[valid]
gt_vert_cam = gt_vert_cam[valid]
pred_j3d = pred_j3d[valid]
pred_vert = pred_vert[valid]

# <======= Evaluation on the local motion
pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam = batch_align_by_pelvis(
    [pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam], pelvis_idxs=[1,2]
)
S1_hat = batch_compute_similarity_transform_torch(pred_j3d, gt_j3d_cam)
pa_mpjpe = torch.sqrt(((S1_hat - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
mpjpe = torch.sqrt(((pred_j3d - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
pve = torch.sqrt(((pred_vert - gt_vert_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm

accel = compute_error_accel(joints_pred=pred_j3d.cpu(), joints_gt=gt_j3d_cam.cpu())[1:-1]
accel = accel * (30 ** 2)       # per frame^s to per s^2

accumulator['pa_mpjpe'].append(pa_mpjpe)
accumulator['mpjpe'].append(mpjpe)
accumulator['pve'].append(pve)
accumulator['accel'].append(accel)
# =======>

# <======= Evaluation on the global motion
chunk_length = 100
w_mpjpe, wa_mpjpe = [], []
rte_chunk = []
for start in range(0, valid.sum() - chunk_length, chunk_length):
    end = start + chunk_length
    if start + 2 * chunk_length > valid.sum(): end = valid.sum() - 1
    
    target_j3d = gt_j3d[start:end].clone().cpu()
    pred_j3d = pred_j3d_w[start:end].clone().cpu()
    
    w_j3d = first_align_joints(target_j3d, pred_j3d)
    wa_j3d = global_align_joints(target_j3d, pred_j3d)
    
    w_jpe = compute_jpe(target_j3d, w_j3d)
    wa_jpe = compute_jpe(target_j3d, wa_j3d)
    w_mpjpe.append(w_jpe)
    wa_mpjpe.append(wa_jpe)

    rte_chunk.append(compute_rte(target_j3d[:,0], pred_j3d[:,0]))


# print(f"w_mpjpe: {w_mpjpe}, len: {len(w_mpjpe)}")
# print(f"wa_mpjpe: {wa_mpjpe}, len: {len(wa_mpjpe)}")

w_mpjpe = np.concatenate(w_mpjpe) * m2mm
wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm

# plot w_mpjpe and wa_mpjpe and save as to figures
plt.figure(figsize=(10, 5))
plt.plot(w_mpjpe, label='w_mpjpe')
plt.plot(wa_mpjpe, label='wa_mpjpe')
plt.legend()
plt.savefig(os.path.join(vis_dir, 'mpjpe.png'))
plt.close()

rte_chunk = np.concatenate(rte_chunk) * 1e2

# plot rte_chunk and save as to figures
plt.figure(figsize=(10, 5))
plt.plot(rte_chunk, label='rte_chunk')
plt.legend()
plt.savefig(os.path.join(vis_dir, 'rte_chunk.png'))
plt.close()

# =======>

# <======= Evaluation on the entier global motion
# RTE: root trajectory error
pred_j3d_align = first_align_joints(gt_j3d, pred_j3d_w)
rte_align_first= compute_jpe(gt_j3d[:,[0]], pred_j3d_align[:,[0]])
rte_align_all = compute_rte(gt_j3d[:,0], pred_j3d_w[:,0]) * 1e2 

# ERVE: Ego-centric root velocity error
erve = computer_erve(gt_ori, gt_j3d, pred_ori_w, pred_j3d_w) * m2mm
# =======>

# <======= Record human trajectory
human_traj[seq_name] = {'gt': gt_j3d[:,0], 'pred': pred_j3d_align[:, 0]}
# =======>

accumulator['wa_mpjpe'].append(wa_mpjpe)
accumulator['w_mpjpe'].append(w_mpjpe)
accumulator['rte'].append(rte_align_all)
accumulator['rte_chunk'].append(rte_chunk)
accumulator['erve'].append(erve)
    
for k, v in accumulator.items():
    accumulator[k] = np.concatenate(v).mean()

# Evaluation: Camera motion
results = {}

# Process camera motion for the sequence
# Get GT camera trajectory
cam_r = cam_pose[:,:3,:3].transpose(0,2,1)
cam_t = np.einsum('bij, bj->bi', cam_r, -cam_pose[:, :3, 3])
cam_q = matrix_to_quaternion(torch.from_numpy(cam_r)).numpy()

# Get predicted camera trajectory
camera_matrix = pred_smpl['pred_cams'] # T, 4, 4
pred_traj_t = torch.from_numpy(camera_matrix[:, :3, 3])
pred_traj_q = matrix_to_quaternion(torch.from_numpy(camera_matrix[:, :3, :3]))
pred_traj = torch.concat([pred_traj_t, pred_traj_q], dim=-1).numpy()


# Cut to the same length if needed
min_len = min(len(cam_t), len(pred_traj))
cam_t = cam_t[:min_len]
cam_q = cam_q[:min_len]
pred_traj = pred_traj[:min_len]

stats_slam, _, _ = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=True)
stats_metric, traj_ref, traj_est = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=False)

# Save results
re = {'traj_gt': traj_ref.positions_xyz,
      'traj_est': traj_est.positions_xyz, 
      'traj_gt_q': traj_ref.orientations_quat_wxyz,
      'traj_est_q': traj_est.orientations_quat_wxyz,
      'stats_slam': stats_slam,
      'stats_metric': stats_metric}

results[seq_name] = re

# Store camera trajectories for visualization
cam_traj[seq_name] = {'gt': torch.from_numpy(traj_ref.positions_xyz), 
                      'pred': torch.from_numpy(traj_est.positions_xyz)}

ate = np.mean([re['stats_slam']['mean'] for re in results.values()])
ate_s = np.mean([re['stats_metric']['mean'] for re in results.values()])
accumulator['ate'] = ate
accumulator['ate_s'] = ate_s

# Save evaluation results
for k, v in accumulator.items():
    print(k, accumulator[k])

df = pd.DataFrame(list(accumulator.items()), columns=['Metric', 'Value'])
df.to_excel(f"{output_dir}/evaluation.xlsx", index=False)

# Visualize trajectories if requested
if args.visualize:
    print("Visualizing trajectories...")
    
    # Visualize human trajectories (RTE)
    human_vis_dir = os.path.join(vis_dir, 'human_trajectories')
    os.makedirs(human_vis_dir, exist_ok=True)
    vis_traj(human_traj, human_traj, human_vis_dir, grid=args.grid_size)
    print(f"Human trajectory visualizations saved to {human_vis_dir}")
    
    # Visualize camera trajectories (ATE)
    cam_vis_dir = os.path.join(vis_dir, 'camera_trajectories')
    os.makedirs(cam_vis_dir, exist_ok=True)
    
    # Create a modified version of cam_traj for visualization
    cam_traj_vis = {}
    for seq in cam_traj:
        cam_traj_vis[seq] = {
            'gt': cam_traj[seq]['gt'],
            'pred': cam_traj[seq]['pred']
        }
    
    vis_traj(cam_traj, cam_traj, cam_vis_dir, grid=args.grid_size)
    print(f"Camera trajectory visualizations saved to {cam_vis_dir}")
    
    # Create a combined visualization showing both human and camera trajectories
    combined_vis_dir = os.path.join(vis_dir, 'combined_trajectories')
    os.makedirs(combined_vis_dir, exist_ok=True)
    
    # For each sequence, create a matplotlib figure with both trajectories
    for seq in human_traj:
        plt.figure(figsize=(10, 8))
        
        # Plot human trajectories
        human_gt = human_traj[seq]['gt'].numpy()
        human_pred = human_traj[seq]['pred'].numpy()
        plt.scatter(human_gt[:, 0], human_gt[:, 1], s=10, c='green', alpha=0.7, label='Human GT')
        plt.scatter(human_pred[:, 0], human_pred[:, 1], s=10, c='orange', alpha=0.7, label='Human Pred')
        
        # Plot camera trajectories
        cam_gt = cam_traj[seq]['gt'].numpy()
        cam_pred = cam_traj[seq]['pred'].numpy()
        plt.scatter(cam_gt[:, 0], cam_gt[:, 1], s=10, c='blue', alpha=0.7, label='Camera GT')
        plt.scatter(cam_pred[:, 0], cam_pred[:, 1], s=10, c='red', alpha=0.7, label='Camera Pred')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f'Combined Trajectories - {seq}')
        plt.legend()
        plt.axis('equal')
        plt.savefig(os.path.join(combined_vis_dir, f'{seq}_combined_xy.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"Combined trajectory visualizations saved to {combined_vis_dir}")
