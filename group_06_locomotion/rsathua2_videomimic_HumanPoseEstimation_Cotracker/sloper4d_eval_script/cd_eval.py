import numpy as np
import glob
import cv2
import tqdm
import os
import open3d as o3d
import sklearn.neighbors as skln

# ========= Procrustes helper functions (only Umeyama) =========
def umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """
    Compute similarity transform (scale s, rotation R, translation t) between two point sets
    Assumes src[i] corresponds to dst[i].
    src, dst: (N,3) numpy arrays. Returns s, R (3×3), t (3,)
    """
    assert src.shape == dst.shape, "Source and target point clouds must have same shape for Umeyama alignment"

    # Ensure float64 for better precision and range
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c   = src - mu_src, dst - mu_dst # Center the data

    cov            = dst_c.T @ src_c / src.shape[0]
    try:
        U, S, Vt       = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        print("Warning: SVD computation failed. Returning identity transform.")
        return 1.0, np.eye(3), np.zeros(3) # Return invalid but safe default

    R              = U @ Vt
    if np.linalg.det(R) < 0:          # Reflection correction
        Vt[-1, :] *= -1
        R = U @ Vt

    scale = 1.0
    if with_scale:
        # More robust computation of var_src (mean squared norm of centered source points)
        try:
            # Explicitly use float64 for square and mean computation
            src_c_squared_norms = np.sum(np.square(src_c, dtype=np.float64), axis=1)
            var_src = np.mean(src_c_squared_norms)

            # Check if var_src is valid and greater than zero
            if var_src > 1e-10 and np.isfinite(var_src):
                 scale = np.sum(S) / var_src
                 # Check again if scale is finite
                 if not np.isfinite(scale):
                     print(f"Warning: Computed scale is not finite ({scale}), resetting to 1.0. var_src={var_src}, sum(S)={np.sum(S)}")
                     scale = 1.0
            elif not np.isfinite(var_src):
                 print(f"Warning: Computed var_src is not finite ({var_src}), setting scale to 1.0.")
                 scale = 1.0
            else: # var_src near zero
                 print(f"Warning: Computed var_src is near zero ({var_src}), setting scale to 1.0.")
                 scale = 1.0
        except OverflowError:
            print("Warning: Overflow error in var_src computation. Setting scale to 1.0.")
            scale = 1.0 # No scaling if variance computation overflows

    t = mu_dst - scale * R @ mu_src
    return scale, R, t

# =================================================================
# Load valid pixel masks
depth_dir = "/home/junyi42/SLOPER4D/data/seq008_running_001/depth_data"
gt_pcd_path = "/home/junyi42/SLOPER4D/data/seq008_running_001/pcd.pcd"
pred_pts3d_path = "results/sloper4d_seq008/combined_point_cloud.npz"
# depth_dir = "/home/junyi42/SLOPER4D/data/seq007_garden_001/depth_data"
# gt_pcd_path = "/home/junyi42/SLOPER4D/data/seq007_garden_001/pcd.pcd"
# pred_pts3d_path = "results/sloper4d_seq007/combined_point_cloud.npz"
with_scale = False

pred_pts3d = np.load(pred_pts3d_path)['pred_pts3d']
print(f"Loaded predicted point cloud shape: {pred_pts3d.shape}") # (N_frames, H, W, 3)

depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
assert len(depth_paths) == pred_pts3d.shape[0]

all_valid_masks = []
valid_pts3d_list = []
total_png_valid_pixels = 0

print(f"Loading masks and valid points from {len(depth_paths)} depth PNGs...")
for i, depth_path in tqdm.tqdm(enumerate(depth_paths), total=len(depth_paths), desc="Loading data"):
    depth_map_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth_map_uint16 is None:
        print(f"Warning: Could not load image {depth_path}")
        # Handle missing image...
        if i > 0 and all_valid_masks:
            h, w = all_valid_masks[-1].shape
        else:
            try:
                first_depth_map = cv2.imread(depth_paths[0], cv2.IMREAD_UNCHANGED)
                h, w = first_depth_map.shape if first_depth_map is not None else (1080, 1920)
            except:
                 h, w = 1080, 1920
        valid_mask = np.zeros((h, w), dtype=np.uint8)
        all_valid_masks.append(valid_mask)
        valid_pts3d_list.append(np.empty((0, 3), dtype=pred_pts3d.dtype))
        continue

    valid_mask = (depth_map_uint16 != 0).astype(np.uint8)
    all_valid_masks.append(valid_mask)
    total_png_valid_pixels += np.sum(valid_mask)

    pred_pts3d_single = pred_pts3d[i]
    pred_pts3d_single_resized = cv2.resize(pred_pts3d_single, (valid_mask.shape[1], valid_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid_pts3d_list.append(pred_pts3d_single_resized[valid_mask == 1])

print(f"\nMask loading complete.")
print(f"Total valid pixels from PNGs: {total_png_valid_pixels}")

# --- Data Preparation ---
# 1. Concatenate valid predicted points from all frames
pred_points_all = np.concatenate(valid_pts3d_list, axis=0)
print(f"Total predicted points after concatenation: {pred_points_all.shape[0]}")

# 2. Load and extract GT point cloud
gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
gt_points_all = np.asarray(gt_pcd.points)
print(f"Loaded GT point cloud points: {gt_points_all.shape[0]}")

# --- Verify Correspondence ---
assert pred_points_all.shape == gt_points_all.shape, \
    f"Predicted points ({pred_points_all.shape[0]}) and GT points ({gt_points_all.shape[0]}) mismatch! Cannot perform correspondence-based alignment."
print("Predicted and GT point clouds match in size, proceeding with correspondence-based alignment.")

# --- Exact Procrustes Alignment (using full corresponding clouds) ---
print("\nComputing exact Sim(3) alignment (Umeyama) on full corresponding clouds...")
if pred_points_all.shape[0] >= 3: # Umeyama needs at least 3 points
    # Print coordinate ranges for diagnostics
    print(f"  DEBUG: pred_points_all min={np.min(pred_points_all, axis=0)}, max={np.max(pred_points_all, axis=0)}")
    print(f"  DEBUG: gt_points_all min={np.min(gt_points_all, axis=0)}, max={np.max(gt_points_all, axis=0)}")

    s, R, t = umeyama_similarity(pred_points_all, gt_points_all, with_scale=with_scale) # Ensure scaling is enabled

    print(f"Computed alignment parameters: scale={s:.6f}") # Increased precision
    print("Rotation:\n", R)
    print("Translation:", t)

    # Apply transform to *all* predicted points
    pred_points_aligned_all = (s * (R @ pred_points_all.T)).T + t
    print("Applied alignment transform to full predicted cloud.")

    # Optional: Save full cloud before alignment (for comparison)
    # pred_pcd_full_o3d = o3d.geometry.PointCloud()
    # pred_pcd_full_o3d.points = o3d.utility.Vector3dVector(pred_points_all)
    # o3d.io.write_point_cloud("pred_full_unaligned.ply", pred_pcd_full_o3d)

    # Optional: Save full cloud after alignment
    aligned_pcd_full_o3d = o3d.geometry.PointCloud()
    aligned_pcd_full_o3d.points = o3d.utility.Vector3dVector(pred_points_aligned_all)
    o3d.io.write_point_cloud("pred_full_aligned.ply", aligned_pcd_full_o3d)
    print("Saved aligned full cloud as pred_full_aligned.ply")

else:
    print("Too few points (<3) for Umeyama alignment. Skipping alignment and Chamfer computation.")
    # Subsequent steps may be meaningless or error out if alignment fails
    pred_points_aligned_all = pred_points_all # Keep unaligned to allow code to run, but results invalid


# --- Downsampling (for Chamfer computation efficiency) ---
voxel_size = 0.02 # Voxel size in meters, adjust based on scene
print(f"\nDownsampling aligned predicted and GT clouds using voxel size {voxel_size}...")

# Create Open3D point cloud objects (using aligned predicted points)
pred_aligned_pcd_o3d = o3d.geometry.PointCloud()
pred_aligned_pcd_o3d.points = o3d.utility.Vector3dVector(pred_points_aligned_all)
gt_pcd_o3d = o3d.geometry.PointCloud() # GT cloud unchanged
gt_pcd_o3d.points = o3d.utility.Vector3dVector(gt_points_all)

# Perform voxel downsampling
down_pred_aligned_pcd = pred_aligned_pcd_o3d.voxel_down_sample(voxel_size)
down_gt_pcd = gt_pcd_o3d.voxel_down_sample(voxel_size)

pred_points_aligned_down = np.asarray(down_pred_aligned_pcd.points)
gt_points_down = np.asarray(down_gt_pcd.points)
print(f"Aligned predicted points after downsampling: {pred_points_aligned_down.shape[0]}")
print(f"GT points after downsampling: {gt_points_down.shape[0]}")

# Optional: Save downsampled clouds for inspection
o3d.io.write_point_cloud(f"pred_aligned_downsampled.ply", down_pred_aligned_pcd)
o3d.io.write_point_cloud(f"gt_downsampled.ply", down_gt_pcd)
print("Saved downsampled clouds as pred_aligned_downsampled.ply and gt_downsampled.ply")


# --- Chamfer Distance Computation (on downsampled clouds) ---
if pred_points_aligned_down.shape[0] > 0 and gt_points_down.shape[0] > 0:
    print("\nComputing Chamfer Distance (on downsampled clouds)...")
    nn_engine = skln.NearestNeighbors(n_neighbors=1,
                                    algorithm='kd_tree',
                                    n_jobs=-1)

    # D2S (aligned predicted points -> GT points)
    nn_engine.fit(gt_points_down)
    dist_d2s, _ = nn_engine.kneighbors(pred_points_aligned_down,
                                       n_neighbors=1,
                                       return_distance=True)
    accuracy = np.mean(dist_d2s)

    # S2D (GT points -> aligned predicted points)
    nn_engine.fit(pred_points_aligned_down)
    dist_s2d, _ = nn_engine.kneighbors(gt_points_down,
                                       n_neighbors=1,
                                       return_distance=True)
    completeness = np.mean(dist_s2d)

    # Calculate Chamfer distance
    chamfer_dist = (accuracy + completeness) / 2

    print(f"Chamfer Distance Results:")
    print(f"  Accuracy (aligned pred -> gt): {accuracy:.6f}")
    print(f"  Completeness (gt -> aligned pred): {completeness:.6f}")
    print(f"  Chamfer Distance (avg): {chamfer_dist:.6f}")
else:
    print("Downsampled clouds empty or insufficient points, cannot compute Chamfer Distance.")
