import numpy as np
import open3d as o3d
import glob
# use natural sort
import natsort
import torch
import os
import h5py
import argparse # Import argparse

# ========= helper functions =========
def umeyama_similarity(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """
    src, dst: (N,3) numpy arrays. Returns s, R (3×3), t (3,)
    """
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c   = src - mu_src, dst - mu_dst
    cov            = dst_c.T @ src_c / src.shape[0]
    U, S, Vt       = np.linalg.svd(cov)
    R              = U @ Vt
    if np.linalg.det(R) < 0:          # Reflection correction
        U[:, -1] *= -1
        R = U @ Vt
    scale = 1.0
    if with_scale:
        var_src = (src_c ** 2).sum() / src.shape[0]
        scale   = np.trace(np.diag(S)) / var_src
    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def ransac_sim3(src, dst, iters=500, thr=0.01, sample_size=4):
    """
    Basic RANSAC-Procrustes for pts3d -> Sim(3) alignment
    """
    best_inliers, best_par = 0, None
    N = src.shape[0]
    for _ in range(iters):
        idx   = np.random.choice(N, sample_size, replace=False)
        s, R, t = umeyama_similarity(src[idx], dst[idx])
        pred  = (s * (R @ src.T)).T + t
        err   = np.linalg.norm(pred - dst, axis=1)
        n_inl = (err < thr).sum()
        if n_inl > best_inliers:
            best_inliers, best_par = n_inl, (s, R, t)
            if n_inl > 0.9 * N:       # Good enough
                break
    return best_par if best_par is not None else umeyama_similarity(src, dst)
# ====================================

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Concatenate and align HDF5 results for a specific sequence.')
parser.add_argument('--seq_num', type=str, required=True, help='Sequence number (e.g., "007", "008")')
args = parser.parse_args()
seq_num = args.seq_num # Get sequence number from arguments
# --- End Argument Parsing ---


# Construct paths using the sequence number
h5_glob_pattern = f"demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_seq{seq_num}_*"
output_dir = f"results/sloper4d_seq{seq_num}"
output_smpl_path = os.path.join(output_dir, "hps_combined_track_0.npy") # Path for SMPL+Cam data
output_pcd_path = os.path.join(output_dir, "combined_point_cloud.pcd") # Path for PCD file
output_pts3d_npz_path = os.path.join(output_dir, "combined_point_cloud.npz") # Path for NPZ file

h5_paths = natsort.natsorted(glob.glob(h5_glob_pattern))
h5_paths = [path for path in h5_paths if path[-16]!='9'] # Keep this filter if needed
print(f"Processing sequence: {seq_num}")
print("Found HDF5 paths:")
print(h5_paths)

if not h5_paths:
    print(f"Error: No HDF5 files found for sequence {seq_num} with pattern {h5_glob_pattern}")
    exit()

skip_geometry = False
os.makedirs(output_dir, exist_ok=True) # Ensure the directory exists

pred_rotmats = []
pred_shapes = []
pred_transs = []
pred_cams = []
pred_pts3d = []

chunk_size = 300
last_frame_global_orient = None
last_frame_transl = None
last_frame_cams = None
last_frame_pts3d = None

for h5_path in h5_paths:
    with h5py.File(h5_path, 'r') as f:
        # print the keys of the f
        # print(list(f.keys()))
        # ... inside the loop ...
        # Ensure data are numpy arrays for processing

        # --- Check for human parameters and get the first available key ---
        human_params = f.get('our_pred_humans_smplx_params')
        if human_params is None or not list(human_params.keys()):
            print(f"Warning: No human parameters found in {h5_path}. Skipping this file.")
            continue
        
        human_key = list(human_params.keys())[0] # Use the first available human key
        print(human_params.keys())
        print(f"Processing human key '{human_key}' from {h5_path}")
        # ---

        body_pose = np.array(f['our_pred_humans_smplx_params'][human_key]['body_pose'])
        global_orient = np.array(f['our_pred_humans_smplx_params'][human_key]['global_orient'])
        betas = np.array(f['our_pred_humans_smplx_params'][human_key]['betas'])
        root_transl = np.array(f['our_pred_humans_smplx_params'][human_key]['root_transl'])
        keys_geometry = list(f['our_pred_world_cameras_and_structure'].keys())
        cams_chunk = []
        pts3d_chunk = []
        if skip_geometry: keys_geometry = keys_geometry[:1]
        for key in keys_geometry:
            cams_chunk.append(np.array(f['our_pred_world_cameras_and_structure'][key]['cam2world']))
            pts3d_chunk.append(np.array(f['our_pred_world_cameras_and_structure'][key]['pts3d']))
        cams_chunk = np.stack(cams_chunk, axis=0)
        pts3d_chunk = np.stack(pts3d_chunk, axis=0)
        # print(cams_chunk.shape, 'cams_chunk')
        # print(pts3d_chunk.shape, 'pts3d_chunk')
        # (301, 4, 4) cams_chunk
        # (301, 288, 512, 3) pts3d_chunk
        num_frames = body_pose.shape[0]
        print(f"num_frames: {num_frames} for {h5_path}")

        if num_frames == 0:
            print(f"Warning: num_frames is 0 for human key '{human_key}' in {h5_path}. Skipping this file.")
            continue

        if last_frame_global_orient is not None:
            R_prev_last   = last_frame_global_orient            # (3,3)
            R_curr_first  = global_orient[0,0]                  # (3,3)
            transform_R   = R_prev_last @ R_curr_first.T        # Use transpose

            # Rotate all global_orient
            global_orient = (transform_R @ global_orient.reshape(num_frames,3,3)).reshape(num_frames,1,3,3)

            # Synchronously rotate root_transl
            root_transl   = (transform_R @ root_transl.reshape(num_frames,3).T).T.reshape(num_frames,1,3)

        if last_frame_transl is not None:
            offset_t      = last_frame_transl - root_transl[0]
            root_transl  += offset_t       # broadcasting

        # ---------- align cams_chunk ----------
        if last_frame_cams is not None:
            # 1) Rotation
            R_prev = last_frame_cams[:3, :3]
            R_curr = cams_chunk[0, :3, :3]
            transform_R = R_prev @ R_curr.T          # Same approach as global_orient

            cams_chunk[:, :3, :3] = transform_R @ cams_chunk[:, :3, :3]

            # 2) Translation (offset only, no rotation)
            offset_t = last_frame_cams[:3, 3] - cams_chunk[0, :3, 3]
            cams_chunk[:, :3, 3] += offset_t
        # ---------- end cams_chunk ----------


        # ---------- align pts3d_chunk ----------
        if last_frame_pts3d is not None:
            # Flatten & remove NaN, then randomly sample 5k points for speed
            src0 = last_frame_pts3d.reshape(-1, 3)
            dst0 = pts3d_chunk[0].reshape(-1, 3)
            mask = (~np.isnan(src0).any(1)) & (~np.isnan(dst0).any(1))
            src, dst = src0[mask], dst0[mask]
            if src.shape[0] > 5000:
                idx = np.random.choice(src.shape[0], 5000, replace=False)
                src, dst = src[idx], dst[idx]

            if src.shape[0] > 3 and dst.shape[0] > 3: # Need at least 4 points for umeyama
                s, R, t = ransac_sim3(dst, src)          # Align "current first frame" to "last frame"

                # Apply same transformation to entire chunk
                pts_shape = pts3d_chunk.shape            # (T, H, W, 3)
                pts_flat  = pts3d_chunk.reshape(-1, 3).T # 3 × M
                pts_flat  = (s * (R @ pts_flat)).T + t
                pts3d_chunk = pts_flat.reshape(pts_shape)
            else:
                print(f"Warning: Not enough valid points ({src.shape[0]}) for Sim(3) alignment in {h5_path}. Skipping pts3d alignment for this chunk.")

        # ---------- end pts3d_chunk ----------


        pred_rotmat = np.concatenate([global_orient, body_pose], axis=1)
        pred_shape = betas # Already a numpy array
        pred_trans = root_transl # Already a numpy array

        # Append only up to chunk_size frames
        pred_rotmats.append(pred_rotmat[:chunk_size])
        pred_shapes.append(pred_shape[:chunk_size])
        pred_transs.append(pred_trans[:chunk_size])
        pred_cams.append(cams_chunk[:chunk_size])
        pred_pts3d.append(pts3d_chunk[:chunk_size])


        # Store the *aligned* last frame for the next iteration
        last_frame_global_orient = global_orient[chunk_size-1] if chunk_size <= num_frames else global_orient[-1]
        last_frame_transl = root_transl[chunk_size-1] if chunk_size <= num_frames else root_transl[-1]
        last_frame_cams = cams_chunk[chunk_size-1] if chunk_size <= len(cams_chunk) else cams_chunk[-1]
        last_frame_pts3d = pts3d_chunk[chunk_size-1] if chunk_size <= len(pts3d_chunk) else pts3d_chunk[-1]


# Concatenate all chunks
final_pred_rotmat = np.concatenate(pred_rotmats, axis=0)
final_pred_shape = np.concatenate(pred_shapes, axis=0)
final_pred_trans = np.concatenate(pred_transs, axis=0)
final_pred_cams = np.concatenate(pred_cams, axis=0)
final_pred_pts3d = np.concatenate(pred_pts3d, axis=0) # Shape like (Total_Frames, H, W, 3)

print("Final concatenated shapes:")
print(f"  pred_rotmat: {final_pred_rotmat.shape}")
print(f"  pred_shape:  {final_pred_shape.shape}")
print(f"  pred_trans:  {final_pred_trans.shape}")
print(f"  pred_cams:   {final_pred_cams.shape}")
print(f"  pred_pts3d:  {final_pred_pts3d.shape}")

# --- Create the dictionary for SMPL and Cam data (excluding pts3d) ---
smpl_cam_data_to_save = {
    'pred_rotmat': final_pred_rotmat,
    'pred_shape': final_pred_shape,
    'pred_trans': final_pred_trans,
    'pred_cams': final_pred_cams,
}

# --- Save the SMPL and Cam dictionary to a .npy file ---
np.save(output_smpl_path, smpl_cam_data_to_save)
print(f"\nSaved combined SMPL and Cam data to: {output_smpl_path}")

# --- Convert final point cloud to float16 ---
print(f"\nConverting final point cloud to float16...")
try:
    # Check original dtype to avoid unnecessary conversion or errors
    if final_pred_pts3d.dtype != np.float16:
        final_pred_pts3d_f16 = final_pred_pts3d.astype(np.float16)
        print(f"Converted pts3d from {final_pred_pts3d.dtype} to {final_pred_pts3d_f16.dtype}")
    else:
        final_pred_pts3d_f16 = final_pred_pts3d
        print("pts3d is already float16.")
except Exception as e:
    print(f"Error converting point cloud to float16: {e}. Saving original dtype.")
    final_pred_pts3d_f16 = final_pred_pts3d # Fallback to original

# --- Save the float16 point cloud to NPZ (compressed) ---
print(f"\nSaving final point cloud (float16) to NPZ file...")
try:
    np.savez_compressed(output_pts3d_npz_path, pred_pts3d=final_pred_pts3d_f16)
    print(f"Saved combined point cloud (float16) to: {output_pts3d_npz_path}")
except Exception as e:
    print(f"Error saving point cloud to NPZ: {e}")

# --- Save the final point cloud to PCD (Optional, using float32 for compatibility) ---
print(f"\nProcessing final point cloud for PCD saving (using float32)...")
# Reshape to (N, 3) - use the original float32/float64 for PCD
pts3d_flat = final_pred_pts3d.reshape(-1, 3)
# Remove NaN values
pts3d_valid = pts3d_flat[~np.isnan(pts3d_flat).any(axis=1)]

if pts3d_valid.shape[0] > 0:
    # Ensure pts3d_valid is float32 or float64 for Open3D
    if pts3d_valid.dtype not in [np.float32, np.float64]:
        print(f"Converting points for PCD from {pts3d_valid.dtype} to float32")
        pts3d_valid_pcd = pts3d_valid.astype(np.float32)
    else:
        pts3d_valid_pcd = pts3d_valid

    print(f"Saving {pts3d_valid_pcd.shape[0]} valid points (dtype: {pts3d_valid_pcd.dtype}) to PCD...")
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    # Open3D typically expects float64, but Vector3dVector handles conversion
    pcd.points = o3d.utility.Vector3dVector(pts3d_valid_pcd)

    # Save to PCD file
    try:
        o3d.io.write_point_cloud(output_pcd_path, pcd)
        print(f"Saved combined point cloud to: {output_pcd_path}")
    except Exception as e:
        print(f"Error saving point cloud to PCD: {e}")
else:
    print("No valid points found in final_pred_pts3d. Skipping PCD save.")


# --- Verification (Optional) ---
# Load the saved NPY file to check its structure
loaded_data_npy = np.load(output_smpl_path, allow_pickle=True).item()
print("\nVerification - Keys in saved NPY file:", loaded_data_npy.keys())
for key, value in loaded_data_npy.items():
    if isinstance(value, np.ndarray):
        print(f"  Shape of '{key}': {value.shape}")
    else:
        print(f"  Value of '{key}': {value}")

# Optional: Verify NPZ file
try:
    loaded_data_npz = np.load(output_pts3d_npz_path)
    print("\nVerification - Keys in saved NPZ file:", loaded_data_npz.files)
    if 'pred_pts3d' in loaded_data_npz:
        print(f"  Shape of 'pred_pts3d' in NPZ: {loaded_data_npz['pred_pts3d'].shape}")
        print(f"  Dtype of 'pred_pts3d' in NPZ: {loaded_data_npz['pred_pts3d'].dtype}") # Check dtype
    loaded_data_npz.close() # Important to close the file handle
except Exception as e:
    print(f"Could not verify NPZ file: {e}")
