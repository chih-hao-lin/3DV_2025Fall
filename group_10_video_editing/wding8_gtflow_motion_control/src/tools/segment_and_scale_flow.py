"""
segment_and_scale_flow.py

Scale optical flow based on SAM2 video segmentation masks.

Usage:
    # Basic usage
    python src/tools/segment_and_scale_flow.py \
        --flow results/warped_noise/train/flows_dxdy.npy \
        --mask data/videos_sam/train_sam2.mp4 \
        --fg_scale 1.5 \
        --bg_scale 0.3 \
        --output results/warped_noise/train_scaled/flows_dxdy.npy

    # With visualization
    python src/tools/segment_and_scale_flow.py \
        --flow results/warped_noise/train/flows_dxdy.npy \
        --mask data/videos_sam/train_sam2.mp4 \
        --fg_scale 1.5 \
        --bg_scale 0.3 \
        --output results/warped_noise/train_scaled/flows_dxdy.npy \
        --visualize results/visualizations/train_segmented_flow.mp4

Segmentation Format:
    - SAM2 mask video with white pixels = foreground (object of interest)
    - All other colored pixels = background
    - Masks are automatically resized to match flow resolution

Scaling:
    scaled_flow = flow * (fg_mask * fg_scale + bg_mask * bg_scale)

    Examples:
    - Emphasize foreground: --fg_scale 2.0 --bg_scale 0.5
    - Remove background motion: --fg_scale 1.0 --bg_scale 0.0
    - Dampen foreground: --fg_scale 0.3 --bg_scale 1.0
"""

import numpy as np
import cv2
import argparse
from pathlib import Path
import sys

def load_sam2_masks(mask_video_path, target_height, target_width, target_frames=None, threshold=240):
    """
    Load SAM2 segmentation masks from video.

    Args:
        mask_video_path: Path to SAM2 mask video
        target_height: Height to resize masks to (match flow)
        target_width: Width to resize masks to (match flow)
        target_frames: Number of frames to resample to (if None, use original count)
        threshold: Brightness threshold for white detection (default: 240)

    Returns:
        Binary masks (T, H, W) where 1 = foreground, 0 = background
    """
    print(f"Loading SAM2 masks from: {mask_video_path}")

    cap = cv2.VideoCapture(str(mask_video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open mask video: {mask_video_path}")

    masks = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # White pixels (foreground) have high values
        # threshold=240 means pixels >= 240 are considered white/foreground
        fg_mask = (gray >= threshold).astype(np.float32)

        # Resize to match flow resolution
        fg_mask_resized = cv2.resize(
            fg_mask,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )

        # Binarize after resize to avoid interpolation artifacts
        fg_mask_binary = (fg_mask_resized > 0.5).astype(np.float32)

        masks.append(fg_mask_binary)
        frame_count += 1

    cap.release()

    masks = np.array(masks)  # Shape: (T, H, W)

    print(f"  ✓ Loaded {frame_count} mask frames")

    # Resample frames if target_frames is specified
    if target_frames is not None and target_frames != frame_count:
        print(f"  ✓ Resampling from {frame_count} frames to {target_frames} frames...")
        print(f"    (First and last frames will correspond)")

        # Create frame indices for resampling
        # This ensures first (0) and last (frame_count-1) frames map to first (0) and last (target_frames-1)
        source_indices = np.linspace(0, frame_count - 1, target_frames)

        # Interpolate masks at these indices
        resampled_masks = []
        for idx in source_indices:
            # Get integer and fractional parts
            idx_low = int(np.floor(idx))
            idx_high = int(np.ceil(idx))
            alpha = idx - idx_low

            # Linear interpolation between frames
            if idx_low == idx_high:
                # Exact frame match
                resampled_mask = masks[idx_low]
            else:
                # Blend between two frames
                mask_low = masks[idx_low]
                mask_high = masks[idx_high]
                resampled_mask = (1 - alpha) * mask_low + alpha * mask_high
                # Re-binarize after interpolation
                resampled_mask = (resampled_mask > 0.5).astype(np.float32)

            resampled_masks.append(resampled_mask)

        masks = np.array(resampled_masks)
        frame_count = target_frames
        print(f"  ✓ Resampled to {frame_count} frames")
    print(f"  ✓ Resized from video resolution to {target_height}x{target_width}")
    print(f"  ✓ Foreground pixels: {(masks > 0.5).sum() / masks.size * 100:.1f}%")

    return masks


def scale_flow_by_segmentation(flow, fg_masks, fg_scale, bg_scale):
    """
    Scale optical flow based on foreground/background segmentation.

    Args:
        flow: Optical flow (T, 2, H, W)
        fg_masks: Foreground masks (T, H, W) with values 0 or 1
        fg_scale: Scale factor for foreground
        bg_scale: Scale factor for background

    Returns:
        Scaled flow (T, 2, H, W)
    """
    T, C, H, W = flow.shape
    assert C == 2, "Flow must have 2 channels (u, v)"
    assert fg_masks.shape == (T, H, W), "Masks must match flow dimensions"

    print(f"\nScaling flow:")
    print(f"  Foreground scale: {fg_scale}")
    print(f"  Background scale: {bg_scale}")

    # Create background masks (inverse of foreground)
    bg_masks = 1.0 - fg_masks

    # Compute combined scale mask
    # scale_mask[t, h, w] = fg_mask * fg_scale + bg_mask * bg_scale
    scale_masks = fg_masks * fg_scale + bg_masks * bg_scale  # (T, H, W)

    # Broadcast to flow shape: (T, H, W) -> (T, 1, H, W) -> (T, 2, H, W)
    scale_masks_broadcast = scale_masks[:, np.newaxis, :, :]  # (T, 1, H, W)

    # Apply scaling
    scaled_flow = flow * scale_masks_broadcast

    # Calculate statistics
    original_mag = np.sqrt(flow[:, 0]**2 + flow[:, 1]**2).mean()
    scaled_mag = np.sqrt(scaled_flow[:, 0]**2 + scaled_flow[:, 1]**2).mean()

    fg_flow_mag = np.sqrt(flow[:, 0]**2 + flow[:, 1]**2)[fg_masks > 0.5].mean() if (fg_masks > 0.5).any() else 0
    bg_flow_mag = np.sqrt(flow[:, 0]**2 + flow[:, 1]**2)[bg_masks > 0.5].mean() if (bg_masks > 0.5).any() else 0

    fg_scaled_mag = np.sqrt(scaled_flow[:, 0]**2 + scaled_flow[:, 1]**2)[fg_masks > 0.5].mean() if (fg_masks > 0.5).any() else 0
    bg_scaled_mag = np.sqrt(scaled_flow[:, 0]**2 + scaled_flow[:, 1]**2)[bg_masks > 0.5].mean() if (bg_masks > 0.5).any() else 0

    print(f"\n  Results:")
    print(f"    Overall original magnitude: {original_mag:.3f}")
    print(f"    Overall scaled magnitude:   {scaled_mag:.3f}")
    print(f"    Overall change:             {(scaled_mag/original_mag - 1)*100:+.1f}%")
    print(f"\n    Foreground:")
    print(f"      Original: {fg_flow_mag:.3f} → Scaled: {fg_scaled_mag:.3f} ({fg_scale}x)")
    print(f"    Background:")
    print(f"      Original: {bg_flow_mag:.3f} → Scaled: {bg_scaled_mag:.3f} ({bg_scale}x)")

    return scaled_flow


def visualize_segmented_flow(flow, scaled_flow, fg_masks, output_path, fps=12):
    """
    Create visualization comparing original and scaled flow with masks.

    Creates a 2x2 grid:
    - Top-left: Original flow
    - Top-right: Scaled flow
    - Bottom-left: Foreground mask
    - Bottom-right: Scale mask
    """
    from visualize_flow_with_vectors import flow_to_hsv

    print(f"\nCreating visualization...")

    T, _, H, W = flow.shape

    # Grid layout: 2x2
    padding = 10
    panel_h = H
    panel_w = W
    grid_h = 2 * panel_h + 3 * padding
    grid_w = 2 * panel_w + 3 * padding

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_w, grid_h))

    for t in range(T):
        # Create grid background
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

        # Top-left: Original flow
        flow_viz = flow_to_hsv(flow[t], white_background=True)
        y, x = padding, padding
        grid[y:y+panel_h, x:x+panel_w] = flow_viz

        # Add label
        cv2.putText(grid, "Original Flow", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Top-right: Scaled flow
        scaled_viz = flow_to_hsv(scaled_flow[t], white_background=True)
        y, x = padding, padding + panel_w + padding
        grid[y:y+panel_h, x:x+panel_w] = scaled_viz

        cv2.putText(grid, "Scaled Flow", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Bottom-left: Foreground mask
        mask_viz = (fg_masks[t] * 255).astype(np.uint8)
        mask_rgb = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2RGB)
        # Colorize: white = foreground, black = background
        mask_colored = np.zeros_like(mask_rgb)
        mask_colored[:, :, 1] = mask_viz  # Green for foreground
        mask_colored = cv2.addWeighted(mask_rgb, 0.7, mask_colored, 0.3, 0)

        y, x = padding + panel_h + padding, padding
        grid[y:y+panel_h, x:x+panel_w] = mask_colored

        cv2.putText(grid, "Foreground Mask", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Bottom-right: Magnitude comparison
        orig_mag = np.sqrt(flow[t, 0]**2 + flow[t, 1]**2)
        scaled_mag = np.sqrt(scaled_flow[t, 0]**2 + scaled_flow[t, 1]**2)

        # Normalize and colorize
        max_mag = max(orig_mag.max(), scaled_mag.max())
        if max_mag > 0:
            diff = (scaled_mag - orig_mag) / max_mag
        else:
            diff = np.zeros_like(orig_mag)

        # Red = reduced, Green = increased, White = unchanged
        diff_viz = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        diff_viz[:, :, 0] = np.clip((-diff * 255), 0, 255).astype(np.uint8)  # Red (reduced)
        diff_viz[:, :, 1] = np.clip((diff * 255), 0, 255).astype(np.uint8)   # Green (increased)
        diff_viz[:, :, 2] = 128  # Gray baseline

        y, x = padding + panel_h + padding, padding + panel_w + padding
        grid[y:y+panel_h, x:x+panel_w] = diff_viz

        cv2.putText(grid, "Change (R=reduced, G=increased)", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Write frame
        bgr_frame = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()
    print(f"  ✓ Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Scale optical flow based on SAM2 video segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--flow', type=str, required=True,
                       help='Path to optical flow .npy file (T, 2, H, W)')
    parser.add_argument('--mask', type=str, required=True,
                       help='Path to SAM2 mask video (white = foreground)')
    parser.add_argument('--fg_scale', type=float, required=True,
                       help='Scale factor for foreground motion (e.g., 1.5)')
    parser.add_argument('--bg_scale', type=float, required=True,
                       help='Scale factor for background motion (e.g., 0.3)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for scaled flow .npy file')
    parser.add_argument('--visualize', type=str, default=None,
                       help='Optional: Path to save visualization video')
    parser.add_argument('--threshold', type=int, default=240,
                       help='Brightness threshold for white detection (default: 240)')

    args = parser.parse_args()

    print("="*70)
    print("SEGMENTATION-BASED FLOW SCALING")
    print("="*70)

    # Load optical flow
    print(f"\nLoading optical flow from: {args.flow}")
    flow = np.load(args.flow)
    print(f"  ✓ Flow shape: {flow.shape}")
    print(f"  ✓ Mean magnitude: {np.sqrt(flow[:, 0]**2 + flow[:, 1]**2).mean():.3f}")

    T, C, H, W = flow.shape
    assert C == 2, f"Expected flow with 2 channels, got {C}"

    # Load SAM2 masks (will be resampled to match flow frame count)
    print(f"\nTarget: {T} frames at {H}x{W} resolution")
    fg_masks = load_sam2_masks(args.mask, H, W, target_frames=T, threshold=args.threshold)

    # Verify frame counts match (should always match now due to resampling)
    assert fg_masks.shape[0] == T, f"Frame count mismatch after resampling: {fg_masks.shape[0]} != {T}"

    # Scale flow
    scaled_flow = scale_flow_by_segmentation(flow, fg_masks, args.fg_scale, args.bg_scale)

    # Save scaled flow
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(output_path, scaled_flow)
    print(f"\n✓ Saved scaled flow to: {output_path}")

    # Create visualization if requested
    if args.visualize:
        visualize_segmented_flow(flow, scaled_flow, fg_masks, args.visualize)

    print("\n" + "="*70)
    print("SEGMENTATION-BASED FLOW SCALING COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Copy warped noise structure:")
    print(f"     cp -r results/warped_noise/[original]/ results/warped_noise/[scaled]/")
    print(f"  2. Replace flow with scaled version:")
    print(f"     cp {output_path} results/warped_noise/[scaled]/flows_dxdy.npy")
    print(f"  3. Use for video generation as normal")


if __name__ == '__main__':
    main()
