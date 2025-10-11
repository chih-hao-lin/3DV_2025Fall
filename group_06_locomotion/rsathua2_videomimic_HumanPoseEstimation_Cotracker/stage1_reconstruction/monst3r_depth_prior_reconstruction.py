# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import os
import os.path as osp
import glob
import numpy as np
import copy
import pickle
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import tyro
import PIL
import cv2
import torch
import h5py

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


import h5py

def save_dict_to_hdf5(h5file, dictionary, path="/"):
    """
    Recursively save a nested dictionary to an HDF5 file.
    
    Args:
        h5file: An open h5py.File object.
        dictionary: The nested dictionary to save.
        path: The current path in the HDF5 file.
    """
    for key, value in dictionary.items():
        key_path = f"{path}{key}"
        if value is None:
            continue
        if isinstance(value, dict):
            # If value is a dictionary, create a group and recurse
            group = h5file.create_group(key_path)
            save_dict_to_hdf5(h5file, value, key_path + "/")
        elif isinstance(value, np.ndarray):
            h5file.create_dataset(key_path, data=value)
        elif isinstance(value, (int, float, str, bytes, list, tuple)):
            h5file.attrs[key_path] = value  # Store scalars as attributes
        else:
            raise TypeError(f"Unsupported data type: {type(value)} for key {key_path}")


def transform_keypoints(homogeneous_keypoints: np.ndarray, affine_matrix: np.ndarray):
    # Ensure keypoints is a numpy array
    homogeneous_keypoints = np.array(homogeneous_keypoints)

    # Apply the transformation
    transformed_keypoints = np.dot(affine_matrix, homogeneous_keypoints.T).T
    
    # Round to nearest integer for pixel coordinates
    transformed_keypoints = np.round(transformed_keypoints).astype(int)
    
    return transformed_keypoints

def check_affine_matrix(test_img: PIL.Image, original_image: PIL.Image, affine_matrix: np.ndarray):
    assert affine_matrix.shape == (2, 3)

    # get pixels near the center of the image in the new image space
    # Sample 100 pixels near the center of the image
    w, h = test_img.size
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 4  # Use a quarter of the smaller dimension as the radius

    # Generate random offsets within the circular region
    num_samples = 100
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    r = np.random.uniform(0, radius, num_samples)
    
    # Convert polar coordinates to Cartesian
    x_offsets = r * np.cos(theta)
    y_offsets = r * np.sin(theta)
    
    # Add offsets to the center coordinates and ensure they're within image bounds
    sampled_x = np.clip(center_x + x_offsets, 0, w-1).astype(int)
    sampled_y = np.clip(center_y + y_offsets, 0, h-1).astype(int)
    
    # Create homogeneous coordinates
    pixels_near_center = np.column_stack((sampled_x, sampled_y, np.ones(num_samples)))
    
    # draw the pixels on the image and save it
    test_img_pixels = np.asarray(test_img).copy().astype(np.uint8)
    for x, y in zip(sampled_x, sampled_y):
        test_img_pixels = cv2.circle(test_img_pixels, (x, y), 3, (0, 255, 0), -1)
    PIL.Image.fromarray(test_img_pixels).save('test_new_img_pixels.png')

    transformed_keypoints = transform_keypoints(pixels_near_center, affine_matrix)
    # Load the original image
    original_img_array = np.array(original_image)

    # Draw the transformed keypoints on the original image
    for point in transformed_keypoints:
        x, y = point[:2]
        # Ensure the coordinates are within the image bounds
        if 0 <= x < original_image.width and 0 <= y < original_image.height:
            cv2.circle(original_img_array, (int(x), int(y)), int(3*affine_matrix[0,0]), (255, 0, 0), -1)

    # Save the image with drawn keypoints
    PIL.Image.fromarray(original_img_array).save('test_original_img_keypoints.png')

# hard coding to get the affine transform matrix
def preprocess_and_get_transform(file, size=512, square_ok=False):
    img = PIL.Image.open(file)
    original_width, original_height = img.size
    
    # Step 1: Resize
    S = max(img.size)
    if S > size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*size/S)) for x in img.size)
    img_resized = img.resize(new_size, interp)
    
    # Calculate center of the resized image
    cx, cy = img_resized.size[0] // 2, img_resized.size[1] // 2
    
    # Step 2: Crop
    halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
    if not square_ok and new_size[0] == new_size[1]:
        halfh = 3*halfw//4
    
    img_cropped = img_resized.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    
    # Calculate the total transformation
    scale_x = new_size[0] / original_width
    scale_y = new_size[1] / original_height
    
    translate_x = (cx - halfw) / scale_x
    translate_y = (cy - halfh) / scale_y
    
    affine_matrix = np.array([
        [1/scale_x, 0, translate_x],
        [0, 1/scale_y, translate_y]
    ])
    
    return img_cropped, affine_matrix

def get_reconstructed_scene(model, device, silent, image_size, filelist, schedule, niter, scenegraph_type, winsize, refid, flow_loss, save_folder="./tmp_dynamic_mask", gsam2=False, batchify=True, empty_cache=False):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    if gsam2:
        # Extract dynamic mask files from filelist
        dynamic_mask_root = []
        for f in filelist:
            # Extract the base path and frame number
            base_path = f.split('/cam01/')[0]
            # replace input_images with input_masks
            base_path = base_path.replace('input_images', 'input_masks')
            
            # Handle different filename formats
            if 'frame_' in f:
                # Format: frame_xxxxx.jpg or frame_xxxxx.png
                frame_num = f.split('frame_')[1].split('.')[0]
            else:
                # Format: xxxxx.jpg or xxxxx.png
                frame_num = os.path.basename(f).split('.')[0]
            
            # Construct the mask path
            mask_path = f"{base_path}/cam01/mask_data/mask_{int(frame_num):05d}.npz"
            dynamic_mask_root.append(mask_path)
        
        # Filter to only existing mask files
        dynamic_mask_root = [f for f in dynamic_mask_root if osp.exists(f)]
        
        if len(dynamic_mask_root) == 0:
            print("No dynamic masks found for the given video.")
            return None, None, None, None, None, None, None, None, None, None
    else:
        dynamic_mask_root = None
    imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_root)

    # get affine transform matrix list
    affine_matrix_list = []
    for file in filelist:
        img_cropped, affine_matrix = preprocess_and_get_transform(file)
        affine_matrix_list.append(affine_matrix)
        # img_cropped_list.append(img_cropped)

    # CHECK the first image
    # test_img = img_cropped_list[0]
    # org_img = PIL.Image.open(filelist[0])
    # check_affine_matrix(test_img, org_img, affine_matrix_list[0])
    # import pdb; pdb.set_trace()


    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize) #+ "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)
    elif scenegraph_type == "swinstride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "swinstride-cyclic":
        scenegraph_type = "swinstride-" + str(winsize)
    elif scenegraph_type == "swin-cyclic":
        scenegraph_type = "swin-" + str(winsize)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)

    output = inference(pairs, model, device, batch_size=16, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent,
                           shared_focal = True, temporal_smoothing_weight=0.01, translation_weight=0.1,
                            flow_loss_weight=0.01 if flow_loss else 0.0, flow_loss_start_epoch=0.1, use_self_mask=not gsam2, # if not using grounded sam2 mask, use self mask
                            num_total_iter=niter, empty_cache=empty_cache, raft_ckpt_path="assets/ckpt_raft/Tartan-C-T-TSKH-spring540x960-M.pth",
                            sam2_mask_refine=not gsam2, sam2_ckpt_path="assets/ckpt_sam2/sam2.1_hiera_large.pt", batchify=batchify)   #turn on sam2_mask_refine for better mask
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
        print('final loss: ', loss)
    # get optimized values from scene
    # scene = scene_state.sparse_ga
    rgbimg = scene.imgs   # list of N numpy images with shape (H,W,3) , rgb
    intrinsics = to_numpy(scene.get_intrinsics()) # N intrinsics # (N, 3, 3)
    scene.normalize_imposes_to_frame0()
    cams2world = to_numpy(scene.get_im_poses()) # (N,4,4)

    # 3D pointcloud from depthmap, poses and intrinsics
    # pts3d: list of N pointclouds, each shape(H, W,3)
    # confs: list of N confidence scores, each shape(H, W)
    # msk: boolean mask of valid points, shape(H, W)
    pts3d = to_numpy(scene.get_pts3d())
    depths = to_numpy(scene.get_depthmaps())
    msk = to_numpy(scene.get_masks())
    dynamic_msk = to_numpy(scene.save_dynamic_masks(path=save_folder))
    confs = to_numpy([c for c in scene.im_conf])

    return rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs, affine_matrix_list, output, dynamic_msk


def main(model_path: str = './assets/checkpoints/align3r_depthpro.pth', 
            out_dir: str = './demo_data/input_monst3r', 
            video_dir: str = './demo_data/input_images/arthur_jumping_nov20/cam01', 
            start_frame: int = 0, 
            end_frame: int = 20, 
            stride: int = 1, 
            flow_loss: bool = True,
            scene_graph: str = 'swinstride', # this is better in handling longer video ~150 frames in A100 80G # 'oneref' # 'complete'
            gsam2: bool = False,
            batchify: bool = True,
            empty_cache: bool = False,
            ):
    # parameters
    device = 'cuda'
    silent = False
    image_size = 512
    # 
    mono_imgpath_list = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
    # if no image is found, try png
    if len(mono_imgpath_list) == 0:
        mono_imgpath_list = sorted(glob.glob(os.path.join(video_dir, '*.png')))
    
    # subsample the frames but the total number of frames is frame_num
    filelist = sorted(mono_imgpath_list)[start_frame:end_frame:stride]
    # end_frame = start_frame + len(filelist)
    
    schedule = 'linear'
    niter = 300
    scenegraph_type = scene_graph 
    winsize = 5
    refid = 0
    
    # Load your model here
    from dust3r.model import AsymmetricCroCo3DStereo_Align3r
    model =AsymmetricCroCo3DStereo_Align3r.from_pretrained(model_path).to(device)

    model_name = osp.basename(model_path).split('_')[0].lower()
    video_name = '_'.join(video_dir.split('/')[-2:])
    output_file = osp.join(out_dir, f'{model_name}_reconstruction_results_{video_name}_frame_{start_frame}_{end_frame}_subsample_{stride}.h5')
    save_folder = osp.join(out_dir, f'{model_name}_reconstruction_results_{video_name}_frame_{start_frame}_{end_frame}_subsample_{stride}')
    os.makedirs(save_folder, exist_ok=True)
    
    rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs, affine_matrix_list, output, dynamic_msk = \
        get_reconstructed_scene(model, device, silent, image_size, filelist, schedule, niter, scenegraph_type, winsize, refid, flow_loss, save_folder, gsam2, batchify, empty_cache)
    
    # Save the results as a pickle file
    results = {}
    for i, f in enumerate(filelist):
        img_name = osp.basename(f)[:-4]
        results[img_name] = {
            'rgbimg': rgbimg[i],
            'intrinsic': intrinsics[i],
            'cam2world': cams2world[i],
            'pts3d': pts3d[i],
            'depths': depths[i],
            'msk': msk[i],
            'conf': confs[i],
            'affine_matrix': affine_matrix_list[i],
            'dynamic_msk': dynamic_msk[i]
        }

    # Remove pred_depth from output's view1 and view2
    # Make all the values to be numpy in output
    def convert_to_numpy(d):
        for key, value in d.items():
            if isinstance(value, dict):
                convert_to_numpy(value)
            elif isinstance(value, torch.Tensor):
                d[key] = value.numpy()

    convert_to_numpy(output)
    for key, value in output.items():
        if 'view' in key:
            del value['pred_depth']

    total_output = {
        "monst3r_network_output": output,
        "monst3r_ga_output": results
    }
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # Save to an HDF5 file
    with h5py.File(output_file, "w") as h5file:
        save_dict_to_hdf5(h5file, total_output)

    print(f"Results saved to {output_file}")



if __name__ == '__main__':
    tyro.cli(main)