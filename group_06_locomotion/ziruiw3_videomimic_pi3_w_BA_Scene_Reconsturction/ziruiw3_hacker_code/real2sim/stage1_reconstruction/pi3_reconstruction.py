import os
import sys
import glob
import argparse
import numpy as np
import os.path as osp
import torch
import PIL
import h5py
from scipy.optimize import least_squares
import cv2


# Append necessary directories to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
third_party_root = os.path.join(project_root, "third_party")
pi3_package_path = os.path.join(third_party_root, "pi3-package")

sys.path.append(pi3_package_path)


from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
from pi3.utils.geometry import depth_edge


def save_dict_to_hdf5(h5file, dictionary, path="/"):
    for key, value in dictionary.items():
        key_path = f"{path}{key}"
        if value is None:
            continue
        if isinstance(value, dict):
            group = h5file.create_group(key_path)
            save_dict_to_hdf5(h5file, value, key_path + "/")
        elif isinstance(value, np.ndarray):
            h5file.create_dataset(key_path, data=value)
        elif isinstance(value, (int, float, str, bytes, list, tuple)):
            h5file.attrs[key_path] = value
        else:
            raise TypeError(f"Unsupported data type: {type(value)} for key {key_path}")


def preprocess_and_get_transform(file, size=512, square_ok=False):
    img = PIL.Image.open(file)
    original_width, original_height = img.size

    S = max(img.size)
    if S > size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * size / S)) for x in img.size)
    img_resized = img.resize(new_size, interp)

    cx, cy = img_resized.size[0] // 2, img_resized.size[1] // 2

    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if not square_ok and new_size[0] == new_size[1]:
        halfh = 3 * halfw // 4

    _ = img_resized.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    scale_x = new_size[0] / original_width
    scale_y = new_size[1] / original_height

    translate_x = (cx - halfw) / scale_x
    translate_y = (cy - halfh) / scale_y

    affine_matrix = np.array([
        [1 / scale_x, 0, translate_x],
        [0, 1 / scale_y, translate_y],
    ])

    return affine_matrix


def scale_mask(mask_bool, scale):
    h, w = mask_bool.shape[:2]
    img = PIL.Image.fromarray((mask_bool.astype(np.uint8) * 255))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scaled = img.resize((new_w, new_h), PIL.Image.NEAREST)
    # center crop or pad back to (h, w)
    if new_w >= w and new_h >= h:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        scaled = scaled.crop((left, top, left + w, top + h))
    else:
        canvas = PIL.Image.new('L', (w, h), color=0)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        canvas.paste(scaled, (left, top))
        scaled = canvas
    return (np.array(scaled) > 0)


def load_and_resize_mask(mask_path, target_h, target_w, scale=1.0):
    if osp.exists(mask_path):
        arr = np.load(mask_path)['mask']
        img = PIL.Image.fromarray(((arr > 0) * 255).astype(np.uint8))
        img = img.resize((target_w, target_h), PIL.Image.NEAREST)
        mask = (np.array(img) > 0)
        if scale != 1.0:
            mask = scale_mask(mask, scale)
        return mask
    else:
        return np.ones((target_h, target_w), dtype=bool)


def rodrigues_to_matrix(rvec):
    """Convert rotation vector to rotation matrix using Rodrigues formula"""
    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


def matrix_to_rodrigues(R):
    """Convert rotation matrix to rotation vector using Rodrigues formula"""
    trace = np.trace(R)
    if abs(trace - 3) < 1e-6:
        return np.zeros(3)
    elif abs(trace + 1) < 1e-6:
        # Handle the case where trace = -1
        v = np.array([R[0, 2], R[1, 2], R[2, 2]])
        if np.linalg.norm(v) < 1e-6:
            v = np.array([R[0, 1], R[1, 1], R[2, 1]])
        v = v / np.linalg.norm(v)
        return np.pi * v
    else:
        theta = np.arccos((trace - 1) / 2)
        K = (R - R.T) / (2 * np.sin(theta))
        rvec = theta * np.array([K[2, 1], K[0, 2], K[1, 0]])
        return rvec


def bundle_adjustment_residuals(params, points_3d, observations, camera_indices, point_indices, 
                               intrinsics, num_cameras):
    """
    Compute residuals for bundle adjustment optimization.
    
    Args:
        params: Flattened parameters [camera_params..., point_params...]
        points_3d: Initial 3D points (N, 3)
        observations: 2D observations (M, 2)
        camera_indices: Camera index for each observation (M,)
        point_indices: Point index for each observation (M,)
        intrinsics: Camera intrinsic matrix (3, 3)
        num_cameras: Number of cameras
    
    Returns:
        residuals: Reprojection errors (M, 2)
    """
    # Extract camera parameters (6 per camera: 3 for rotation, 3 for translation)
    camera_params = params[:num_cameras * 6].reshape(num_cameras, 6)
    
    # Extract 3D point parameters
    point_params = params[num_cameras * 6:].reshape(-1, 3)
    
    residuals = []
    
    for i in range(len(observations)):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        
        # Get camera pose
        rvec = camera_params[cam_idx, :3]
        tvec = camera_params[cam_idx, 3:]
        
        # Get 3D point
        point_3d = point_params[pt_idx]
        
        # Project 3D point to 2D
        R = rodrigues_to_matrix(rvec)
        point_cam = R @ point_3d + tvec
        
        if point_cam[2] <= 0:  # Point behind camera
            residuals.extend([1000, 1000])  # Large residual
            continue
            
        # Project to image plane
        x_proj = intrinsics[0, 0] * point_cam[0] / point_cam[2] + intrinsics[0, 2]
        y_proj = intrinsics[1, 1] * point_cam[1] / point_cam[2] + intrinsics[1, 2]
        
        # Compute residual
        obs = observations[i]
        residuals.extend([x_proj - obs[0], y_proj - obs[1]])
    
    return np.array(residuals)


def perform_bundle_adjustment(camera_poses, points_3d, confidences, intrinsics, 
                             max_iterations=100, confidence_threshold=0.5):
    """
    Perform bundle adjustment to refine camera poses and 3D points.
    
    Args:
        camera_poses: Initial camera poses (N, 3, 4)
        points_3d: 3D points (N, H, W, 3)
        confidences: Confidence values (N, H, W)
        intrinsics: Camera intrinsic matrix (3, 3)
        max_iterations: Maximum optimization iterations
        confidence_threshold: Minimum confidence for point inclusion
    
    Returns:
        refined_poses: Refined camera poses (N, 3, 4)
    """
    print("Starting bundle adjustment...")
    
    num_cameras = len(camera_poses)
    refined_poses = camera_poses.copy()
    
    # Sample high-confidence points for bundle adjustment
    observations = []
    camera_indices = []
    point_indices = []
    point_3d_list = []
    
    point_counter = 0
    for cam_idx in range(num_cameras):
        conf = confidences[cam_idx]
        points = points_3d[cam_idx]
        
        # Sample points with high confidence
        high_conf_mask = conf > confidence_threshold
        
        if np.sum(high_conf_mask) == 0:
            continue
            
        # Subsample to avoid too many points
        y_coords, x_coords = np.where(high_conf_mask)
        if len(y_coords) > 1000:  # Limit number of points per camera
            indices = np.random.choice(len(y_coords), 1000, replace=False)
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        
        for y, x in zip(y_coords, x_coords):
            point_3d = points[y, x]
            if np.isfinite(point_3d).all() and np.linalg.norm(point_3d) > 0.1:
                # Project to image coordinates
                cam2world = camera_poses[cam_idx]
                world_point = cam2world[:3, :3] @ point_3d + cam2world[:3, 3]
                
                # Project to image plane
                if world_point[2] > 0:
                    x_proj = intrinsics[0, 0] * world_point[0] / world_point[2] + intrinsics[0, 2]
                    y_proj = intrinsics[1, 1] * world_point[1] / world_point[2] + intrinsics[1, 2]
                    
                    observations.append([x_proj, y_proj])
                    camera_indices.append(cam_idx)
                    point_indices.append(point_counter)
                    point_3d_list.append(point_3d)
                    point_counter += 1
    
    if len(observations) < 10:
        print("Not enough observations for bundle adjustment, skipping...")
        return refined_poses
    
    observations = np.array(observations)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    point_3d_list = np.array(point_3d_list)
    
    print(f"Bundle adjustment with {len(observations)} observations, {num_cameras} cameras, {len(point_3d_list)} points")
    
    # Initialize parameters
    initial_params = []
    
    # Camera parameters (rotation vector + translation for each camera)
    for i in range(num_cameras):
        pose = camera_poses[i]
        R = pose[:3, :3]
        t = pose[:3, 3]
        rvec = matrix_to_rodrigues(R)
        initial_params.extend([rvec[0], rvec[1], rvec[2], t[0], t[1], t[2]])
    
    # 3D point parameters
    initial_params.extend(point_3d_list.flatten())
    
    initial_params = np.array(initial_params)
    
    try:
        # Perform optimization
        result = least_squares(
            bundle_adjustment_residuals,
            initial_params,
            args=(point_3d_list, observations, camera_indices, point_indices, intrinsics, num_cameras),
            method='lm',
            max_nfev=max_iterations,
            verbose=1
        )
        
        if result.success:
            print(f"Bundle adjustment converged after {result.nfev} iterations")
            
            # Extract refined camera poses
            refined_params = result.x
            camera_params = refined_params[:num_cameras * 6].reshape(num_cameras, 6)
            
            for i in range(num_cameras):
                rvec = camera_params[i, :3]
                tvec = camera_params[i, 3:]
                R = rodrigues_to_matrix(rvec)
                
                refined_poses[i] = np.hstack([R, tvec.reshape(3, 1)])
        else:
            print("Bundle adjustment did not converge, using original poses")
            
    except Exception as e:
        print(f"Bundle adjustment failed: {e}, using original poses")
    
    return refined_poses


def main():
    parser = argparse.ArgumentParser(description="""
    Pi3 reconstruction with optional bundle adjustment for improved camera pose alignment.
    
    Bundle adjustment refines camera poses by minimizing reprojection error across multiple views.
    Use --bundle_adjustment to enable this feature.
    """)
    parser.add_argument('--video-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='./demo_data/input_pi3')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=512)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_intermediate', action='store_true', default=False)
    parser.add_argument('--gsam2', action='store_true', default=False)
    parser.add_argument('--bundle_adjustment', action='store_true', default=False, 
                        help='Enable bundle adjustment to refine camera poses')
    parser.add_argument('--ba_max_iterations', type=int, default=100,
                        help='Maximum iterations for bundle adjustment')
    parser.add_argument('--ba_confidence_threshold', type=float, default=0.5,
                        help='Minimum confidence threshold for points in bundle adjustment')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Gather input frames
    img_path_list = sorted(glob.glob(os.path.join(args.video_dir, '*.jpg')))
    img_path_list += sorted(glob.glob(os.path.join(args.video_dir, '*.png')))
    img_path_list = img_path_list[args.start_frame:args.end_frame:args.stride]
    if len(img_path_list) == 0:
        raise FileNotFoundError(f"No images found in {args.video_dir}")

    scene_name = "pi3_reconstruction_results_" + args.video_dir.split("/")[-2] + "_cam01_frame_" + str(args.start_frame) + "_" + str(args.end_frame) + "_subsample_" + str(args.stride) + ".h5"
    save_path = os.path.join(args.out_dir, scene_name)

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    # Load images tensor (N,3,H,W)
    interval = 1
    imgs = load_images_as_tensor(args.video_dir, interval=interval).to(device)

    # Infer
    with torch.no_grad():
        dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        with torch.amp.autocast(device_type='cuda', dtype=dtype) if device.type == 'cuda' else torch.autocast(enabled=False):
            res = model(imgs[None])

    # Postprocess confidence and masks
    conf = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    conf = torch.logical_and(conf, non_edge)

    # To numpy and remove batch dim
    def to_numpy(x):
        return x.detach().cpu().numpy().squeeze(0)

    predictions = {
        'points': to_numpy(res['points']),            # (N,H,W,3)
        'local_points': to_numpy(res['local_points']),# (N,H,W,3)
        'conf': to_numpy(conf),               # (N,H,W)
        'camera_poses': to_numpy(res['camera_poses']) # (N,3,4)
    }
    
    # Apply bundle adjustment if requested
    # Bundle adjustment jointly optimizes camera poses and 3D points to minimize reprojection error
    # This helps improve camera pose alignment and overall reconstruction quality
    if args.bundle_adjustment:
        print("Applying bundle adjustment to refine camera poses...")
        # Create a simple intrinsic matrix (assuming square images)
        img_h, img_w = predictions['local_points'].shape[1:3]
        intrinsics = np.array([
            [img_w, 0, img_w/2],
            [0, img_h, img_h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        refined_poses = perform_bundle_adjustment(
            predictions['camera_poses'],
            predictions['local_points'],
            predictions['conf'],
            intrinsics,
            max_iterations=args.ba_max_iterations,
            confidence_threshold=args.ba_confidence_threshold
        )
        predictions['camera_poses'] = refined_poses
        print("Bundle adjustment completed.")

    # Prepare dynamic mask file list if requested
    if args.gsam2:
        dynamic_mask_root = []
        for f in img_path_list:
            base_path = f.split('/cam01/')[0]
            base_path = base_path.replace('input_images', 'input_masks')
            if 'frame_' in f:
                frame_num = int(f.split('frame_')[1].split('.')[0])
            else:
                frame_num = int(os.path.basename(f).split('.')[0])
            mask_path = f"{base_path}/cam01/mask_data/mask_{frame_num:05d}.npz"
            dynamic_mask_root.append(mask_path)
    else:
        dynamic_mask_root = None

    # Build per-frame results similar to other pipelines
    results = {}
    affine_matrix_list = []
    for i, f in enumerate(img_path_list):
        affine_matrix = preprocess_and_get_transform(f)
        affine_matrix_list.append(affine_matrix)

        # Convert image tensor to numpy H,W,3 in [0,255]
        img_np = (imgs[i].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

        # Camera pose
        cam2world = predictions['camera_poses'][i]

        # Depth from local points' z
        depths = predictions['local_points'][i, ..., 2]

        # Masks and conf
        msk = np.ones_like(depths, dtype=np.uint8)
        confs = predictions['conf'][i]

        # Intrinsics unknown -> identity placeholder
        intrinsic = np.eye(3, dtype=np.float32)

        # Select dynamic mask resized to match depths (H, W)
        if dynamic_mask_root is not None and i < len(dynamic_mask_root):
            dyn_m = load_and_resize_mask(dynamic_mask_root[i], depths.shape[0], depths.shape[1], scale=1.5)
        else:
            dyn_m = np.ones_like(depths, dtype=bool)

        results[osp.basename(f)[:-4]] = {
            'rgbimg': img_np,                         # (H, W, 3)
            'intrinsic': intrinsic,                   # (3, 3)
            'cam2world': cam2world,                   # (4, 4)
            'pts3d': predictions['points'][i],        # (H, W, 3)
            'depths': depths,                         # (H, W)
            'msk': msk,                               # (H, W)
            'conf': confs,                            # (H, W)
            'dynamic_msk': dyn_m,                     # (H, W)
            'affine_matrix': affine_matrix            # (2, 3)
        }

    total_output = {
        'monst3r_ga_output': results
    }

    with h5py.File(save_path, "w") as h5file:
        save_dict_to_hdf5(h5file, total_output)

    print(f"Pi3 Finished processing {scene_name}, saved to {args.out_dir}/{scene_name}")


if __name__ == '__main__':
    main()


