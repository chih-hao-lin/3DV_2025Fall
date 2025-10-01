from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
import pickle
import re

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039, 0.74117647, 0.85882353)

SMPL24 = {
    "pelvis": 0,
    "left_hip": 1,  "right_hip": 2,
    "left_knee": 4, "right_knee": 5,
    "left_ankle": 7,"right_ankle": 8,
    "spine3": 9,    # upper chest
    "neck": 12,     "head": 15,
    "left_shoulder": 16, "right_shoulder": 17,
    "left_elbow": 18,    "right_elbow": 19,
    "left_wrist": 20,    "right_wrist": 21,
}


# ---------- helpers ----------
def coco17_from_smpl_uv(uv24):
    Lhip, Rhip = uv24[SMPL24["left_hip"]], uv24[SMPL24["right_hip"]]
    Lsho, Rsho = uv24[SMPL24["left_shoulder"]], uv24[SMPL24["right_shoulder"]]
    nose  = uv24[SMPL24["head"]]          # proxy
    leye  = 0.6*uv24[SMPL24["head"]] + 0.4*Lsho
    reye  = 0.6*uv24[SMPL24["head"]] + 0.4*Rsho
    lear  = leye
    rear  = reye

    M = {
      "nose": nose, "left_eye": leye, "right_eye": reye,
      "left_ear": lear, "right_ear": rear,
      "left_shoulder": Lsho, "right_shoulder": Rsho,
      "left_elbow": uv24[SMPL24["left_elbow"]],  "right_elbow": uv24[SMPL24["right_elbow"]],
      "left_wrist": uv24[SMPL24["left_wrist"]],  "right_wrist": uv24[SMPL24["right_wrist"]],
      "left_hip": Lhip, "right_hip": Rhip,
      "left_knee": uv24[SMPL24["left_knee"]],    "right_knee": uv24[SMPL24["right_knee"]],
      "left_ankle": uv24[SMPL24["left_ankle"]],  "right_ankle": uv24[SMPL24["right_ankle"]],
    }

    order = ["nose","left_eye","right_eye","left_ear","right_ear",
             "left_shoulder","right_shoulder","left_elbow","right_elbow",
             "left_wrist","right_wrist","left_hip","right_hip",
             "left_knee","right_knee","left_ankle","right_ankle"]

    K17 = np.zeros((17,3), np.float32)
    for i,name in enumerate(order):
        u,v = M[name]
        conf = 0.5 if name in ("nose","left_eye","right_eye","left_ear","right_ear") else 1.0
        K17[i] = [float(u), float(v), conf]
    return K17

def pad_to_wb133(K17):
    K133 = np.zeros((133,3), np.float32)
    K133[:17] = K17
    return K133

def debug_draw_all_joints(img_bgr, uv, radius=4, thickness=2):
    canvas = img_bgr.copy()
    for j,(u,v) in enumerate(uv.astype(int)):
        if 0 <= u < canvas.shape[1] and 0 <= v < canvas.shape[0]:
            cv2.circle(canvas, (u,v), radius, (0,255,0), -1)
            cv2.putText(canvas, str(j), (u+5, v-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), thickness)
    return canvas

def ensure_dirs(base, write_smplx_alias=False):
    """
    Creates the directory structure that the VGGT pipeline expects (ZERO-BASED indices):
      base/json_data         -> mask_{00000}.json
      base/mask_data         -> mask_{00000}.npy
      base/smpl              -> smpl_params_{00000}.pkl
      base/pose2d            -> pose_{00000}.json

    If write_smplx_alias is True, also creates:
      base/smplx             -> smplx_params_{00000}.pkl  (optional alias)
    """
    json_dir = os.path.join(base, "json_data")
    mask_dir = os.path.join(base, "mask_data")
    smpl_dir = os.path.join(base, "smpl")
    pose2d_dir = os.path.join(base, "pose2d")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(smpl_dir, exist_ok=True)
    os.makedirs(pose2d_dir, exist_ok=True)

    smplx_dir = None
    if write_smplx_alias:
        smplx_dir = os.path.join(base, "smplx")
        os.makedirs(smplx_dir, exist_ok=True)
    return json_dir, mask_dir, smpl_dir, pose2d_dir, smplx_dir

def aa_to_rotmat_3x3(aa):
    """Axis-angle (…,3) → (3,3) rotation matrix."""
    aa = np.asarray(aa, dtype=np.float32).reshape(-1)
    if aa.size != 3:
        raise ValueError(f"axis-angle must have 3 numbers, got shape {aa.shape} / size {aa.size}")
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = aa / angle
    x, y, z = axis
    K = np.array([[0, -z, y],
                  [z, 0, -x],
                  [-y, x, 0]], dtype=np.float32)
    I = np.eye(3, dtype=np.float32)
    return (I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)).astype(np.float32)

def smpl_params_from_out(out, n):
    """
    Returns dict with:
      global_orient: (1,3,3) torch.float32
      body_pose:     (23,3,3) torch.float32
      betas:         (10,)    torch.float32
    """
    def npify(x):
        return None if x is None else x.detach().cpu().numpy()

    sp = out.get('pred_smpl_params', {})
    betas            = npify(sp.get('betas'))
    body_pose_raw    = npify(sp.get('body_pose'))
    global_orient_raw= npify(sp.get('global_orient'))

    # fallbacks used by some checkpoints
    if betas is None and 'pred_shape' in out:
        betas = npify(out['pred_shape'])
    if body_pose_raw is None and 'pred_body_pose' in out:
        body_pose_raw = npify(out['pred_body_pose'])
    if global_orient_raw is None and 'pred_global_orient' in out:
        global_orient_raw = npify(out['pred_global_orient'])

    # betas
    betas = np.zeros((10,), np.float32) if betas is None else np.asarray(betas[n]).astype(np.float32).reshape(-1)[:10]

    # --- global orient ---
    if global_orient_raw is None:
        Rg = np.eye(3, dtype=np.float32)[None, ...]
    else:
        go = np.asarray(global_orient_raw[n])
        if go.shape == (3,):                      # axis-angle
            Rg = aa_to_rotmat_3x3(go)[None, ...]
        elif go.shape == (1,3,3):                 # rotmat
            Rg = go.astype(np.float32)
        elif go.shape == (3,3):                   # rotmat without the leading 1
            Rg = go.astype(np.float32)[None, ...]
        else:
            raise ValueError(f"Unexpected global_orient shape {go.shape}")

    # --- body pose (23 joints) ---
    if body_pose_raw is None:
        R_body = np.tile(np.eye(3, dtype=np.float32), (23,1,1))
    else:
        bp = np.asarray(body_pose_raw[n])
        if bp.shape == (23,3):                    # axis-angle
            R_body = np.stack([aa_to_rotmat_3x3(a) for a in bp], axis=0).astype(np.float32)
        elif bp.shape == (23,3,3):                # rotmat
            R_body = bp.astype(np.float32)
        else:
            raise ValueError(f"Unexpected body_pose shape {bp.shape}")

    return {
        'global_orient': torch.from_numpy(Rg),
        'body_pose':     torch.from_numpy(R_body),
        'betas':         torch.from_numpy(betas),
    }


# --- minimal COCO-17 (first 17 of WholeBody 133) mapping we will fill ---
COCO_WB_IDX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}

def project_points_pinhole(X, fx, fy, cx, cy, eps=1e-6):
    """X: (N,3) in camera coords. Returns (N,2)."""
    Z = np.clip(X[:, 2], eps, None)
    u = fx * (X[:, 0] / Z) + cx
    v = fy * (X[:, 1] / Z) + cy
    return np.stack([u, v], axis=-1)

def smpl_to_coco_wb_133(joints2d_px, filled_names):
    """
    Build a 133x3 array for COCO-WholeBody with only a main-body subset filled.
    joints2d_px: dict name->(u,v)
    filled_names: iterable of joint names you filled (conf=1 for those).
    """
    K = np.zeros((133, 3), dtype=np.float32)
    for name in filled_names:
        if name not in COCO_WB_IDX or name not in joints2d_px:
            continue
        idx = COCO_WB_IDX[name]
        u, v = joints2d_px[name]
        K[idx, 0] = float(u)
        K[idx, 1] = float(v)
        K[idx, 2] = 1.0  # confidence
    return K

def main():
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='HMR2 demo + HSfM export')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='../humans/images_single', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='../humans/out', help='Output folder to save results')
    parser.add_argument('--side_view', action='store_true', default=False)
    parser.add_argument('--top_view', action='store_true', default=False)
    parser.add_argument('--full_frame', action='store_true', default=False)
    parser.add_argument('--save_mesh', action='store_true', default=False)
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'])
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'])
    parser.add_argument('--write_smplx_alias', action='store_true',
                        help='Also write smplx/smplx_params_{:05d}.pkl for convenience (optional).')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hmr2
        cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output dirs (VGGT-compatible names)
    os.makedirs(args.out_folder, exist_ok=True)
    json_dir, mask_dir, smpl_dir, pose2d_dir, smplx_dir = ensure_dirs(args.out_folder, args.write_smplx_alias)

    # Collect images
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    img_paths = sorted(img_paths)

    # Iterate over all images (ZERO-BASED indexing to match VGGT)
    for frame_idx, img_path in enumerate(img_paths, start=0):
        img_cv2 = cv2.imread(str(img_path))
        H, W = img_cv2.shape[:2]
        img_fn = os.path.splitext(os.path.basename(img_path))[0]

        # --- KEY CHANGE: strictly zero-based frame index for filenames ---
        fidx = frame_idx + 1  # 0,1,2,... to match VGGT's frame_00000, pose_00000.json, etc.

        # Detect humans
        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)

        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()        # (N,4) xyxy
        scores = det_instances.scores[valid_idx].detach().cpu().numpy()         # (N,)
        has_masks = hasattr(det_instances, "pred_masks")
        masks = det_instances.pred_masks[valid_idx].cpu().numpy().astype(np.uint8) if has_masks else None  # (N,H,W)

        # Build HSfM-style instance mask + bbox JSON
        inst_mask = np.zeros((H, W), dtype=np.uint16)
        labels_json = {
            "mask_name": f"mask_{fidx:05d}.npy",
            "mask_height": H, "mask_width": W,
            "promote_type": "mask", "labels": {}
        }
        # write higher-score instances last to win overlaps
        # order = np.argsort(scores)
        # for k in order:
        #     pid = int(k+1)  # person ids start at 1
        #     x1, y1, x2, y2 = boxes[k].tolist()
        #     labels_json["labels"][str(pid)] = {
        #         "instance_id": pid, "class_name": "person",
        #         "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        #         "logit": float(scores[k])
        #     }
        #     if masks is not None:
        #         inst_mask[masks[k] > 0] = pid
        order = np.argsort(scores)
        boxes = boxes[order]
        if masks is not None:
            masks = masks[order]
        scores = scores[order]
        # now write labels in the SAME order
        for i in range(len(boxes)):
            pid = i + 1                      # 1-based person IDs
            x1, y1, x2, y2 = boxes[i].tolist()
            labels_json["labels"][str(pid)] = {
                "instance_id": pid, "class_name": "person",
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "logit": float(scores[i])
            }
            if masks is not None:
                inst_mask[masks[i] > 0] = pid

        # Save mask + bbox json  (VGGT expects these exact names)
        np.save(os.path.join(mask_dir, f"mask_{fidx:05d}.npy"), inst_mask)
        with open(os.path.join(json_dir, f"mask_{fidx:05d}.json"), "w") as f:
            json.dump(labels_json, f)

        # Run HMR2.0 on all detected humans
        if len(boxes) == 0:
            print(f"[WARN] No people in frame {img_fn}")
            # NOTE: If you want the downstream VGGT script to never KeyError,
            # you can also emit empty pose/smpl files here.
            # with open(os.path.join(pose2d_dir, f"pose_{fidx:05d}.json"), "w") as f:
            #     json.dump({}, f)
            # with open(os.path.join(smpl_dir, f"smpl_params_{fidx:05d}.pkl"), "wb") as f:
            #     pickle.dump({}, f)
            continue

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=max(1, args.batch_size), shuffle=False, num_workers=0)

        all_verts, all_cam_t = [], []
        per_person_cam_t_full = {}   # pid -> cam_t_full (3,)
        frame_smpl = {}              # {pid: {'smpl_params': {...}}}
        frame_focal = None

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            pred_cam = out['pred_cam']                       # weak-persp in crop coords
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            frame_focal = float(scaled_focal_length)  # same for all people in this frame
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render + collect per person
            B = batch['img'].shape[0]
            for n in range(B):
                person_id = int(batch['personid'][n]) + 1  # align with bbox order starting at 1
                input_patch = (batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)).permute(1,2,0).numpy()

                regression_img = renderer(
                    out['pred_vertices'][n].detach().cpu().numpy(),
                    out['pred_cam_t'][n].detach().cpu().numpy(),
                    batch['img'][n],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )
                final_img = np.concatenate([input_patch, regression_img], axis=1)

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        torch.ones_like(batch['img'][n]).cpu(),
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        side_view=True)
                    final_img = np.concatenate([final_img, side_img], axis=1)
                if args.top_view:
                    top_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                       out['pred_cam_t'][n].detach().cpu().numpy(),
                                       torch.ones_like(batch['img'][n]).cpu(),
                                       mesh_base_color=LIGHT_BLUE,
                                       scene_bg_color=(1, 1, 1),
                                       top_view=True)
                    final_img = np.concatenate([final_img, top_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Collect verts and cam_t for full-frame render
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                per_person_cam_t_full[person_id] = cam_t  # cache for pose2d projection

                # Save meshes if requested
                if args.save_mesh:
                    tmesh = renderer.vertices_to_trimesh(verts, cam_t.copy(), LIGHT_BLUE)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

                # ---- HSfM SMPL export (rotation matrices) ----
                # ... inside the loop over n in your dataloader
                fx = fy = float(scaled_focal_length)
                cx, cy = W/2.0, H/2.0

                # ---- HSfM SMPL export (rotation matrices) ----
                smpl_param_dict = smpl_params_from_out(out, n)  # dict of torch tensors

                frame_smpl[person_id] = {
                    'smpl_params': smpl_param_dict,                                    # rotmats + betas
                    'cam_t': torch.from_numpy(pred_cam_t_full[n]).float(),             # (3,)
                    'K': torch.tensor([[fx, 0.0, cx],
                                    [0.0, fy, cy],
                                    [0.0, 0.0, 1.0]], dtype=torch.float32),         # (3,3)
                    'img_wh': (int(W), int(H)),                                        # optional
                }

        # Per-frame SMPL pkl (all persons in this frame) -> exactly what your VGGT code reads
        smpl_pkl_path = os.path.join(smpl_dir, f"smpl_params_{fidx:05d}.pkl")
        with open(smpl_pkl_path, "wb") as f:
            pickle.dump(frame_smpl, f)

        # Optional: also write an SMPL-X alias file if requested (best-effort flatten)
        if args.write_smplx_alias and (smplx_dir is not None):
            smplx_struct = {}
            for pid, rec in frame_smpl.items():
                sp = rec['smpl_params']
                smplx_struct[pid] = {
                    'body_pose':     sp['body_pose'].numpy().astype(np.float32),      # (23,3,3) rotmats
                    'global_orient': sp['global_orient'].numpy().astype(np.float32),  # (1,3,3) rotmat
                    'betas':         sp['betas'].numpy().astype(np.float32),          # (10,)
                    'left_hand_pose': None, 'right_hand_pose': None,
                }
            with open(os.path.join(smplx_dir, f"smplx_params_{fidx:05d}.pkl"), "wb") as f:
                pickle.dump(smplx_struct, f)

        # ---- Pose2D: pose_{XXXXX}.json (COCO WholeBody 133 length, filled subset) ----
        # Camera intrinsics for full image
        fx = fy = float(frame_focal if frame_focal is not None else 5000.0 * max(W, H) / 224.0)
        cx, cy = W / 2.0, H / 2.0

        pose2d_this_frame = {}
        for pid_str, lbl in labels_json["labels"].items():
            pid = int(pid_str)
            if pid not in frame_smpl:
                continue
            P = frame_smpl[pid]["smpl_params"]

            # tensors to current device
            betas = P["betas"].view(1, -1).to(device)                 # (1,10)
            Rg    = P["global_orient"].view(1, 1, 3, 3).to(device)    # (1,1,3,3)
            Rbody = P["body_pose"].view(1, 23, 3, 3).to(device)       # (1,23,3,3)

            # Use the same SMPL module as the renderer/model
            smpl_out = model.smpl(betas=betas, global_orient=Rg, body_pose=Rbody, pose2rot=False)
            J = smpl_out.joints[0].detach().cpu().numpy()  # (J,3), SMPL coords

            cam_t_full = per_person_cam_t_full.get(pid, None)
            if cam_t_full is None:
                x1,y1,x2,y2 = lbl["x1"], lbl["y1"], lbl["x2"], lbl["y2"]
                hpx = max(1.0, float(y2 - y1))
                z_est = 0.55 * fx / hpx
                cam_t_full = np.array([0.0, 0.0, z_est], dtype=np.float32)

            # move to camera coords and project
            J_cam = J + cam_t_full[None, :]
            uv = project_points_pinhole(J_cam, fx, fy, cx, cy)
            dbg = debug_draw_all_joints(img_cv2, uv)
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_joints_debug.png'), dbg)


            # pack subset we trust (COCO-17 body joints)
            # print(uv.shape, J_cam.shape)
            # joints2d_px = {}
            # for name, jidx in SMPL_JOINT_IDX.items():
            #     if jidx < J_cam.shape[0]:
            #         joints2d_px[name] = (uv[jidx, 0], uv[jidx, 1])

            # filled = [
            #     "left_shoulder","right_shoulder","left_elbow","right_elbow",
            #     "left_wrist","right_wrist","left_hip","right_hip",
            #     "left_knee","right_knee","left_ankle","right_ankle"
            # ]
            # filled = [
            #     "nose","left_eye","right_eye","left_ear","right_ear",
            #     "left_shoulder","right_shoulder","left_elbow","right_elbow",
            #     "left_wrist","right_wrist","left_hip","right_hip",
            #     "left_knee","right_knee","left_ankle","right_ankle"
            # ]

            # print(joints2d_px.keys())

            # K133 = smpl_to_coco_wb_133(joints2d_px, filled)
            K17  = coco17_from_smpl_uv(uv)
            K133 = pad_to_wb133(K17)


            pose2d_this_frame[str(pid)] = {
                "keypoints": K133.tolist(),
                "bbox": [float(lbl["x1"]), float(lbl["y1"]), float(lbl["x2"]), float(lbl["y2"]), 1.0]
            }

        with open(os.path.join(pose2d_dir, f"pose_{fidx:05d}.json"), "w") as f:
            json.dump(pose2d_this_frame, f)

        # Optional: full-frame overlay of all people
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1),
                             focal_length=frame_focal if frame_focal is not None else 5000.0 * max(W, H) / 224.0)
            render_res = (int(H), int(W))
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=render_res, **misc_args)
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.png'), 255*input_img_overlay[:, :, ::-1])

        end = time.time()
        print(f"[{img_fn}] wrote "
              f"json_data/mask_{fidx:05d}.json, mask_data/mask_{fidx:05d}.npy, "
              f"smpl/smpl_params_{fidx:05d}.pkl, pose2d/pose_{fidx:05d}.json "
              f"in {end - start:.2f}s")

if __name__ == '__main__':
    main()
