# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def compute_fundamental_matrix(K1, K2, R, t):
    """
    Compute fundamental matrix from camera intrinsics and relative pose.
    
    Args:
        K1: Intrinsic matrix of first camera (3x3)
        K2: Intrinsic matrix of second camera (3x3)
        R: Rotation matrix from first to second camera (3x3)
        t: Translation vector from first to second camera (3x1)
    
    Returns:
        F: Fundamental matrix (3x3)
    """
    # Essential matrix: E = [t]_x * R
    t_cross = np.array([[0, -t[2], t[1]],
                        [t[2], 0, -t[0]],
                        [-t[1], t[0], 0]])
    E = t_cross @ R
    
    # Fundamental matrix: F = K2^(-T) * E * K1^(-1)
    K1_inv = np.linalg.inv(K1)
    K2_inv_T = np.linalg.inv(K2).T
    F = K2_inv_T @ E @ K1_inv
    
    return F

def compute_epipolar_line(F, point):
    """
    Compute epipolar line for a point in the first image.
    
    Args:
        F: Fundamental matrix (3x3)
        point: Point in first image [x, y] (homogeneous coordinates [x, y, 1])
    
    Returns:
        line: Epipolar line in second image [a, b, c] where ax + by + c = 0
    """
    if len(point) == 2:
        point_homo = np.array([point[0], point[1], 1.0])
    else:
        point_homo = point
    
    # l' = F * x
    line = F @ point_homo
    return line

def draw_epipolar_line(img, line, color=(0, 255, 0), thickness=2):
    """
    Draw epipolar line on image.
    
    Args:
        img: Image to draw on
        line: Epipolar line [a, b, c] where ax + by + c = 0
        color: Line color (B, G, R)
        thickness: Line thickness
    
    Returns:
        img: Image with epipolar line drawn
    """
    a, b, c = line
    h, w = img.shape[:2]
    
    # Find intersection points with image boundaries
    points = []
    
    # Intersection with left boundary (x = 0)
    if abs(b) > 1e-6:
        y = -c / b
        if 0 <= y < h:
            points.append((0, int(y)))
    
    # Intersection with right boundary (x = w-1)
    if abs(b) > 1e-6:
        y = -(a * (w-1) + c) / b
        if 0 <= y < h:
            points.append((w-1, int(y)))
    
    # Intersection with top boundary (y = 0)
    if abs(a) > 1e-6:
        x = -c / a
        if 0 <= x < w:
            points.append((int(x), 0))
    
    # Intersection with bottom boundary (y = h-1)
    if abs(a) > 1e-6:
        x = -(b * (h-1) + c) / a
        if 0 <= x < w:
            points.append((int(x), h-1))
    
    # Draw line between intersection points
    if len(points) >= 2:
        cv2.line(img, points[0], points[1], color, thickness)
    
    return img

def get_patch_center(patch_idx, patch_size, img_height, img_width):
    """
    Get the center pixel coordinates of a patch.
    
    Args:
        patch_idx: Patch index
        patch_size: Size of each patch
        img_height: Image height
        img_width: Image width
    
    Returns:
        center: Center coordinates [x, y]
    """
    h_patches = img_height // patch_size
    w_patches = img_width // patch_size
    
    patch_row = patch_idx // w_patches
    patch_col = patch_idx % w_patches
    
    center_x = patch_col * patch_size + patch_size // 2
    center_y = patch_row * patch_size + patch_size // 2
    
    return [center_x, center_y]

parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument("--alpha", type=float, default=0.7, help="Alpha blending factor for attention overlay (default: 0.7)")
parser.add_argument("--colormap", type=str, default="jet", help="Colormap for attention visualization (default: viridis)")
parser.add_argument("--save_individual", action="store_true", help="Save individual attention maps for each view")
parser.add_argument("--comparison", action="store_true", help="Create side-by-side comparison visualizations")

def get_patch_region(center_patch_idx, patch_radius, h_patches, w_patches):
        """
        获取中心patch周围指定半径区域内的所有patch索引
        
        Args:
            center_patch_idx: 中心patch索引
            patch_radius: 选择半径
            h_patches: patch行数
            w_patches: patch列数
            
        Returns:
            selected_patches: 选中的patch索引列表
        """
        if center_patch_idx is None:
            return None
            
        center_row = center_patch_idx // w_patches
        center_col = center_patch_idx % w_patches
        
        selected_patches = []
        for r in range(max(0, center_row - patch_radius), 
                      min(h_patches, center_row + patch_radius + 1)):
            for c in range(max(0, center_col - patch_radius), 
                          min(w_patches, center_col + patch_radius + 1)):
                patch_idx = r * w_patches + c
                selected_patches.append(patch_idx)
        
        return selected_patches

def save_original_images(original_images, args):
    """
    Save original images for reference.
    Args:
        original_images: Original images tensor with shape [B, V, C, H, W]
        args: Command line arguments
    """
    os.makedirs("attention_visualizations", exist_ok=True)
    
    # Convert images to numpy and denormalize if needed
    if original_images.dtype == torch.float32 and original_images.max() <= 1.0:
        # Assuming images are normalized to [0, 1], convert to [0, 255]
        images_np = (original_images.cpu().numpy() * 255).astype(np.uint8)
    else:
        images_np = original_images.cpu().numpy().astype(np.uint8)
    
    num_views = images_np.shape[1]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Original Images', fontsize=16)
    
    for view_idx in range(num_views):
        row = view_idx // 4
        col = view_idx % 4
        original_img = images_np[view_idx].transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
        axes[row, col].imshow(original_img, aspect='equal')
        axes[row, col].set_title(f'View {view_idx}')
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('attention_visualizations/original_images.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Original images saved to attention_visualizations/original_images.png")


def create_comparison_visualization(frame_attn_weights_list, original_images, args, num_views=8, patch_size=16):
    """
    Create side-by-side comparison of original images with attention overlays.
    Args:
        frame_attn_weights_list: List of frame attention weights
        original_images: Original images tensor
        args: Command line arguments
        num_views: Number of views
        patch_size: Size of each patch
    """
    os.makedirs("attention_visualizations", exist_ok=True)
    
    # Convert images to numpy and denormalize if needed
    if original_images.dtype == torch.float32 and original_images.max() <= 1.0:
        images_np = (original_images.cpu().numpy() * 255).astype(np.uint8)
    else:
        images_np = original_images.cpu().numpy().astype(np.uint8)
    
    img_height, img_width = images_np.shape[-2], images_np.shape[-1]
    
    for layer_idx, frame_attn in enumerate(frame_attn_weights_list):
        print(f"Creating comparison visualization for frame attention layer {layer_idx}")
        frame_attn_head = frame_attn[:, 0]
        frame_attn_patches = frame_attn_head[:, 5:, 5:]
        target_patch_idx = 0
        
        # Create a large figure with 2 rows (original + attention) and 4 columns
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Frame Attention Comparison - Layer {layer_idx} - Head 0 - Target Patch {target_patch_idx}', fontsize=16)
        
        for view_idx in range(num_views):
            col = view_idx % 4
            
            # Original image (top row)
            original_img = images_np[view_idx].transpose(1, 2, 0)
            axes[0, col].imshow(original_img, aspect='equal')
            axes[0, col].set_title(f'Original - View {view_idx}')
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])
            
            # Attention overlay (bottom row)
            target_patch_attn = frame_attn_patches[view_idx]
            selected_patch_idx = 0
            target_patch_attn_selected = target_patch_attn[selected_patch_idx]
            target_patch_attn_reshaped = target_patch_attn_selected.reshape(patch_size, patch_size)
            
            # Upsample attention map to image size
            attn_map_upsampled = cv2.resize(target_patch_attn_reshaped, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            
            # Normalize attention map
            attn_min, attn_max = attn_map_upsampled.min(), attn_map_upsampled.max()
            if attn_max - attn_min > 1e-8:
                normalized_attn_map = (attn_map_upsampled - attn_min) / (attn_max - attn_min)
            else:
                normalized_attn_map = attn_map_upsampled
            
            # Create colormap for attention
            cmap = getattr(plt.cm, args.colormap)
            attn_colored = cmap(normalized_attn_map)[:, :, :3]
            
            # Alpha blend attention map with original image
            blended_img = (1 - args.alpha) * original_img / 255.0 + args.alpha * attn_colored
            
            # Display blended image
            im = axes[1, col].imshow(blended_img, aspect='equal')
            axes[1, col].set_title(f'Attention - View {view_idx}')
            axes[1, col].set_xticks([])
            axes[1, col].set_yticks([])
            
            # Add colorbar for attention weights
            norm = plt.Normalize(attn_min, attn_max)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            plt.colorbar(sm, ax=axes[1, col], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(f'attention_visualizations/frame_attention_comparison_layer_{layer_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()


def visualize_frame_attention(frame_attn_weights_list, original_images, args, num_views=8, patch_size=16):
    """
    Visualize frame attention weights overlaid on original images.
    Args:
        frame_attn_weights_list: List of frame attention weights, each with shape [B*V, head_dim, seq_len, seq_len]
        original_images: Original images tensor with shape [B, V, C, H, W]
        args: Command line arguments
        num_views: Number of views (default: 8)
        patch_size: Size of each patch (default: 16)
    """
    os.makedirs("attention_visualizations", exist_ok=True)
    
    # Convert images to numpy and denormalize if needed
    if original_images.dtype == torch.float32 and original_images.max() <= 1.0:
        # Assuming images are normalized to [0, 1], convert to [0, 255]
        images_np = (original_images.cpu().numpy() * 255).astype(np.uint8)
    else:
        images_np = original_images.cpu().numpy().astype(np.uint8)
    
    # Get image dimensions
    img_height, img_width = images_np.shape[-2], images_np.shape[-1]
    
    for layer_idx, frame_attn in enumerate(frame_attn_weights_list):
        print(f"Visualizing frame attention for layer {layer_idx}")
        frame_attn_head = frame_attn[:, 0]   # Shape: [8, 261, 261]
        frame_attn_patches = frame_attn_head[:, 5:, 5:]  # Shape: [8, 256, 256]
        target_patch_idx = 0
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Frame Attention Weights - Layer {layer_idx} - Head 0 - Target Patch {target_patch_idx}', fontsize=16)
        
        for view_idx in range(num_views):
            row = view_idx // 4
            col = view_idx % 4
            
            # Get original image for this view
            print(f"images_np.shape: {images_np.shape}")
            original_img = images_np[view_idx].transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
            
            # Get attention weights for this view
            target_patch_attn = frame_attn_patches[view_idx]
            selected_patch_idx = 0
            target_patch_attn_selected = target_patch_attn[selected_patch_idx]
            target_patch_attn_reshaped = target_patch_attn_selected.reshape(patch_size, patch_size)
            
            # Upsample attention map to image size
            attn_map_upsampled = cv2.resize(target_patch_attn_reshaped, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
            
            # Normalize attention map
            attn_min, attn_max = attn_map_upsampled.min(), attn_map_upsampled.max()
            if attn_max - attn_min > 1e-8:
                normalized_attn_map = (attn_map_upsampled - attn_min) / (attn_max - attn_min)
            else:
                normalized_attn_map = attn_map_upsampled
            
            # Create colormap for attention
            cmap = getattr(plt.cm, args.colormap)
            attn_colored = cmap(normalized_attn_map)[:, :, :3]  # Remove alpha channel
            
            # Alpha blend attention map with original image
            blended_img = (1 - args.alpha) * original_img / 255.0 + args.alpha * attn_colored
            
            # Display blended image
            im = axes[row, col].imshow(blended_img, aspect='equal')
            axes[row, col].set_title(f'View {view_idx}')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            
            # Add colorbar for attention weights
            norm = plt.Normalize(attn_min, attn_max)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            plt.colorbar(sm, ax=axes[row, col], shrink=0.8)
            
            # Save individual attention map if requested
            if args.save_individual:
                plt.figure(figsize=(10, 8))
                plt.imshow(blended_img, aspect='equal')
                plt.title(f'Frame Attention - Layer {layer_idx} - View {view_idx}')
                plt.axis('off')
                plt.colorbar(sm, shrink=0.8)
                plt.tight_layout()
                plt.savefig(f'attention_visualizations/frame_attention_layer_{layer_idx}_view_{view_idx}_individual.png',
                          dpi=300, bbox_inches='tight')
                plt.close()
        
        plt.tight_layout()
        plt.savefig(f'attention_visualizations/frame_attention_layer_{layer_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_global_attention(global_attn_weights_list, original_images, args, extrinsic=None, intrinsic=None, num_views=8, patch_size=16):
    """
    Visualize global attention weights overlaid on original images with epipolar lines.
    Args:
        global_attn_weights_list: List of global attention weights, each with shape [B, head_dim, seq_len, seq_len]
        original_images: Original images tensor with shape [B, V, C, H, W]
        args: Command line arguments
        extrinsic: Camera extrinsic matrices with shape [B, V, 3, 4]
        intrinsic: Camera intrinsic matrices with shape [B, V, 3, 3]
        num_views: Number of views (default: 8)
        patch_size: Size of each patch (default: 16)
    """
    os.makedirs("attention_visualizations", exist_ok=True)
    
    # Convert images to numpy and denormalize if needed
    if original_images.dtype == torch.float32 and original_images.max() <= 1.0:
        # Assuming images are normalized to [0, 1], convert to [0, 255]
        images_np = (original_images.cpu().numpy() * 255).astype(np.uint8)
    else:
        images_np = original_images.cpu().numpy().astype(np.uint8)
    
    # Get image dimensions
    img_height, img_width = images_np.shape[-2], images_np.shape[-1]
    
    # Convert camera parameters to numpy if provided
    if extrinsic is not None:
        extrinsic_np = extrinsic.cpu().numpy()
    if intrinsic is not None:
        intrinsic_np = intrinsic.cpu().numpy()
    
    for layer_idx, global_attn in enumerate(global_attn_weights_list):
        print(f"Visualizing global attention for layer {layer_idx}")
        global_attn_head = global_attn[0, 0] # Shape: [2088, 2088]
        global_attn_patches_reshaped = global_attn_head[5*num_views:, 5*num_views:]
        target_patch_idx = 40
        
        for view_idx in range(num_views):
            target_patch_global_attn = global_attn_patches_reshaped[view_idx*256:(view_idx+1)*256, :]
            selected_patches = get_patch_region(center_patch_idx=target_patch_idx, patch_radius=1, h_patches=16, w_patches=16)
            target_patch_global_attn = target_patch_global_attn[selected_patches].mean(axis=0)
            
            # Create figure with 3 rows: attention visualization, epipolar lines, and combined
            fig, axes = plt.subplots(3, num_views, figsize=(30, 20))
            fig.suptitle(f'Global Attention Weights with Epipolar Lines - Layer {layer_idx}\nHead 0 - Source View {view_idx}', fontsize=16)
            
            for target_view_idx in range(num_views):
                col = target_view_idx % num_views
                
                # Get original image for this target view
                print(f"images_np.shape: {images_np.shape}")
                original_img = images_np[target_view_idx].transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
                
                view_start = target_view_idx * 256
                view_end = (target_view_idx + 1) * 256
                target_view_attn = target_patch_global_attn[view_start:view_end]
                target_view_attn_reshaped = target_view_attn.reshape(patch_size, patch_size)
                
                # Upsample attention map to image size
                attn_map_upsampled = cv2.resize(target_view_attn_reshaped, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
                
                # Normalize attention map
                attn_min, attn_max = attn_map_upsampled.min(), attn_map_upsampled.max()
                if attn_max - attn_min > 1e-8:
                    normalized_attn_map = (attn_map_upsampled - attn_min) / (attn_max - attn_min)
                else:
                    normalized_attn_map = attn_map_upsampled
                
                # Create colormap for attention
                cmap = getattr(plt.cm, args.colormap)
                attn_colored = cmap(normalized_attn_map)[:, :, :3]  # Remove alpha channel
                
                # Alpha blend attention map with original image
                blended_img = (1 - args.alpha) * original_img / 255.0 + args.alpha * attn_colored
                
                # If this is the target view and selected_patches is not None, draw red borders
                if target_view_idx == view_idx and selected_patches is not None:
                    blended_img = original_img / 255.0
                    patch_size_px = img_height // patch_size  # or use patch_size if you know it's 16
                    border_width = 2
                    # Make a copy to avoid modifying the original
                    blended_img_with_boxes = blended_img.copy()
                    for patch_idx in selected_patches:
                        patch_row = patch_idx // patch_size
                        patch_col = patch_idx % patch_size
                        y_start = patch_row * patch_size_px
                        y_end = (patch_row + 1) * patch_size_px
                        x_start = patch_col * patch_size_px
                        x_end = (patch_col + 1) * patch_size_px
                        # Top border
                        blended_img_with_boxes[y_start:y_start+border_width, x_start:x_end, :] = [1, 0, 0]
                        # Bottom border
                        blended_img_with_boxes[y_end-border_width:y_end, x_start:x_end, :] = [1, 0, 0]
                        # Left border
                        blended_img_with_boxes[y_start:y_end, x_start:x_start+border_width, :] = [1, 0, 0]
                        # Right border
                        blended_img_with_boxes[y_start:y_end, x_end-border_width:x_end, :] = [1, 0, 0]
                    blended_img = blended_img_with_boxes
                # Row 0: Original image
                axes[0, col].imshow(original_img/255.0, aspect='equal')
                axes[0, col].set_title(f'Original - View {target_view_idx}')
                axes[0, col].set_xticks([])
                axes[0, col].set_yticks([])
                
                # Row 0: Attention visualization
                im = axes[1, col].imshow(blended_img, aspect='equal')
                axes[1, col].set_title(f'Attention - View {target_view_idx}')
                axes[1, col].set_xticks([])
                axes[1, col].set_yticks([])
                
                # # Add colorbar for attention weights
                # norm = plt.Normalize(attn_min, attn_max)
                # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                # plt.colorbar(sm, ax=axes[0, col], shrink=0.8)
                
                # Row 1: Epipolar lines
                epipolar_img = original_img.copy().astype(np.uint8)
                if extrinsic is not None and intrinsic is not None and target_view_idx != view_idx:
                    # Get camera parameters for source and target views
                    K1 = intrinsic_np[0, view_idx]  # Source camera intrinsics
                    K2 = intrinsic_np[0, target_view_idx]  # Target camera intrinsics
                    
                    # Get relative pose from source to target
                    R1 = extrinsic_np[0, view_idx, :3, :3]  # Source camera rotation
                    t1 = extrinsic_np[0, view_idx, :3, 3]   # Source camera translation
                    R2 = extrinsic_np[0, target_view_idx, :3, :3]  # Target camera rotation
                    t2 = extrinsic_np[0, target_view_idx, :3, 3]   # Target camera translation
                    
                    # Compute relative pose: R = R2 * R1^T, t = t2 - R2 * R1^T * t1
                    R_rel = R2 @ R1.T
                    t_rel = t2 - R2 @ R1.T @ t1
                    
                    # Compute fundamental matrix
                    F = compute_fundamental_matrix(K1, K2, R_rel, t_rel)
                    
                    # Draw epipolar lines for selected patches
                    for patch_idx in selected_patches:
                        patch_center = get_patch_center(patch_idx, patch_size, img_height, img_width)
                        epipolar_line = compute_epipolar_line(F, patch_center)
                        epipolar_img = draw_epipolar_line(epipolar_img, epipolar_line, color=(0, 255, 0), thickness=2)
                    
                    # # Mark the selected patches in source view
                    # for patch_idx in selected_patches:
                    #     patch_center = get_patch_center(patch_idx, patch_size, img_height, img_width)
                    #     cv2.circle(epipolar_img, (patch_center[0], patch_center[1]), 5, (0, 0, 255), -1)
                
                axes[2, col].imshow(epipolar_img, aspect='equal')
                axes[2, col].set_title(f'Epipolar Lines - View {target_view_idx}')
                axes[2, col].set_xticks([])
                axes[2, col].set_yticks([])
            
            plt.tight_layout()
            plt.savefig(f'attention_visualizations/global_attention_layer_{layer_idx}_view{view_idx}_with_epipolar.png',
                      dpi=300, bbox_inches='tight')
            plt.close()

def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT().cuda()

    model.load_state_dict(torch.load("LVSM/model.pt"), strict=True)
    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*"))
    print(f"Found {len(image_names)} images")
    image_names = [
    "test_images/image_8.png", 
    "test_images/image_9.png", 
    "test_images/image_10.png", 
    "test_images/image_11.png", 
    "test_images/image_12.png", 
    "test_images/image_13.png",
    "test_images/image_14.png", 
    "test_images/image_15.png",  
    ]
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    print("Running inference...")    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, patch_start_idx, frame_attn_weights_list, global_attn_weights_list = model.aggregator(images[None])
            pose_enc_list = model.camera_head(aggregated_tokens_list)
            pose_enc = pose_enc_list[-1]  # pose encoding of the last iteration
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            print(f"extrinsic: {extrinsic.shape}")
            c2w = extrinsic[:,:,:,:3].inverse()
            
            print(extrinsic.shape, intrinsic.shape)
            # predictions_base = model_base(images)
            print("len(frame_attn_weights_list)", len(frame_attn_weights_list))
            print("len(global_attn_weights_list)", len(global_attn_weights_list))
            
            # Visualize attention weights
            print("Starting attention visualization...")
            
            # Save original images for reference
            save_original_images(images, args)
            
            # Validate colormap
            try:
                getattr(plt.cm, args.colormap)
            except AttributeError:
                print(f"Warning: Colormap '{args.colormap}' not found, using 'viridis' instead")
                args.colormap = 'viridis'
            
            # Create different types of visualizations based on arguments
            if args.comparison:
                create_comparison_visualization(frame_attn_weights_list[::4], images, args)
            else:
                visualize_frame_attention(frame_attn_weights_list[::4], images, args)
            
            visualize_global_attention(global_attn_weights_list[::4], images, args, extrinsic, intrinsic)
            print("Attention visualization completed!")


if __name__ == "__main__":
    main()