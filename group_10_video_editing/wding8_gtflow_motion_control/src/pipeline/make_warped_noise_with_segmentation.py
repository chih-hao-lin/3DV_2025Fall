"""
make_warped_noise_with_segmentation.py

Integrated pipeline that:
1. Extracts optical flow from video
2. Applies segmentation-based flow scaling
3. Warps noise with the scaled flow

This is the correct approach (Option 2) where segmentation scaling
happens BEFORE warping, not after. This avoids the need to rewarp.

Usage:
    python src/pipeline/make_warped_noise_with_segmentation.py \\
        --video data/videos/train.mp4 \\
        --mask data/videos_sam/train_sam2.mp4 \\
        --fg_scale 1.5 \\
        --bg_scale 0.3 \\
        --output results/warped_noise/train_fg1.5_bg0.3/
"""

import rp
import numpy as np
import cv2
import sys
from pathlib import Path
import argparse

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

rp.pip_import('fire')
rp.git_import('CommonSource')
import rp.git.CommonSource.noise_warp as nw


def load_sam2_masks(mask_video_path, target_height, target_width, threshold=240):
    """
    Load SAM2 segmentation masks from video and resize to target dimensions.

    Args:
        mask_video_path: Path to SAM2 mask video
        target_height: Target height for masks
        target_width: Target width for masks
        threshold: Grayscale threshold for foreground (white pixels)

    Returns:
        fg_masks: Binary masks (T, H, W) where 1 = foreground, 0 = background
    """
    print(f"\nLoading SAM2 masks from: {mask_video_path}")

    cap = cv2.VideoCapture(str(mask_video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open mask video: {mask_video_path}")

    fg_masks = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold: white pixels (>= threshold) are foreground
        fg_mask = (gray >= threshold).astype(np.float32)

        # Resize to target dimensions
        fg_mask_resized = cv2.resize(fg_mask, (target_width, target_height),
                                     interpolation=cv2.INTER_LINEAR)

        # Binarize after resize
        fg_mask_binary = (fg_mask_resized > 0.5).astype(np.float32)

        fg_masks.append(fg_mask_binary)
        frame_count += 1

    cap.release()

    fg_masks = np.array(fg_masks)  # Shape: (T, H, W)

    fg_ratio = fg_masks.mean() * 100
    print(f"  ✓ Loaded {frame_count} mask frames")
    print(f"  ✓ Resized to {target_height}x{target_width}")
    print(f"  ✓ Foreground pixels: {fg_ratio:.1f}%")

    return fg_masks


def scale_flow_by_segmentation(flow, fg_masks, fg_scale, bg_scale):
    """
    Scale optical flow based on foreground/background segmentation.

    Args:
        flow: Optical flow array (T, 2, H, W)
        fg_masks: Foreground masks (T, H, W), values in [0, 1]
        fg_scale: Scale factor for foreground
        bg_scale: Scale factor for background

    Returns:
        scaled_flow: Scaled optical flow (T, 2, H, W)
    """
    T, C, H, W = flow.shape
    assert C == 2, f"Expected 2 flow channels, got {C}"
    assert fg_masks.shape == (T, H, W), f"Mask shape {fg_masks.shape} doesn't match flow shape"

    # Create background masks
    bg_masks = 1.0 - fg_masks

    # Compute scale masks: fg_mask * fg_scale + bg_mask * bg_scale
    scale_masks = fg_masks * fg_scale + bg_masks * bg_scale  # (T, H, W)

    # Broadcast to flow shape: (T, 1, H, W)
    scale_masks_broadcast = scale_masks[:, np.newaxis, :, :]

    # Apply scaling
    scaled_flow = flow * scale_masks_broadcast

    return scaled_flow


def main():
    parser = argparse.ArgumentParser(
        description='Create warped noise with segmentation-based flow scaling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--video', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--mask', type=str, required=True,
                       help='SAM2 segmentation mask video path')
    parser.add_argument('--fg_scale', type=float, required=True,
                       help='Foreground (white) scale factor')
    parser.add_argument('--bg_scale', type=float, required=True,
                       help='Background scale factor')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for warped noise')
    parser.add_argument('--mask_threshold', type=int, default=240,
                       help='Grayscale threshold for foreground (default: 240)')

    args = parser.parse_args()

    output_dir = Path(args.output)
    if output_dir.exists():
        raise RuntimeError(f"Output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("WARPED NOISE WITH SEGMENTATION-BASED FLOW SCALING")
    print("="*70)
    print(f"\nInput video: {args.video}")
    print(f"SAM2 masks: {args.mask}")
    print(f"Foreground scale: {args.fg_scale}")
    print(f"Background scale: {args.bg_scale}")
    print(f"Output: {args.output}")

    # Load and preprocess video
    print("\nLoading video...")
    video = rp.load_video(args.video)
    video = rp.resize_list(video, length=49)  # CogVideoX requires 49 frames
    video = rp.resize_images_to_hold(video, height=480, width=720)
    video = rp.crop_images(video, height=480, width=720, origin='center')
    video = rp.as_numpy_array(video)
    print(f"  ✓ Video shape: {video.shape}")

    # Load and preprocess SAM video (CRITICAL: Use SAME resampling as video!)
    print("\nLoading SAM2 segmentation masks...")
    sam_video = rp.load_video(args.mask)
    print(f"  Original SAM video: {len(sam_video)} frames")
    sam_video = rp.resize_list(sam_video, length=49)  # SAME as video - ensures temporal alignment!
    print(f"  ✓ Resampled SAM video to 49 frames (matches video timeline)")

    # Extract optical flow (WITHOUT warping noise yet)
    print("\nExtracting optical flow...")
    device = rp.select_torch_device(prefer_used=True)
    raft_model = nw.raft.RaftOpticalFlow(device, "large")

    # Parameters from make_warped_noise.py
    FRAME = 2**-1
    FLOW = 2**3
    LATENT = 8

    # Resize for flow extraction
    video_resized = rp.resize_images(video, size=FRAME, interp='area')

    # Calculate flow
    flows = []
    for i in rp.eta(range(len(video_resized) - 1), title='Computing Optical Flow'):
        flow = raft_model(video_resized[i], video_resized[i + 1])
        flows.append(flow)

    # Stack flows: (T, 2, H, W)
    flows_array = np.stack([rp.as_numpy_array(f) for f in flows])
    T, C, H, W = flows_array.shape
    print(f"  ✓ Flow shape: {flows_array.shape}")
    print(f"  ✓ Original mean magnitude: {np.sqrt(flows_array[:, 0]**2 + flows_array[:, 1]**2).mean():.3f}")

    # Process SAM masks to match flow dimensions
    print(f"\nProcessing SAM masks...")
    print(f"  Target: {T} masks at {H}x{W} resolution (to match flow)")

    fg_masks = []
    for i, sam_frame in enumerate(sam_video[:T+1]):  # Process 49 SAM frames
        # Convert to numpy if needed
        sam_frame_np = rp.as_numpy_array(sam_frame)

        # Convert to grayscale
        if len(sam_frame_np.shape) == 3:
            gray = cv2.cvtColor(sam_frame_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = sam_frame_np

        # Threshold: white pixels (>= threshold) are foreground
        fg_mask = (gray >= args.mask_threshold).astype(np.float32)

        # Resize to match flow resolution
        fg_mask_resized = cv2.resize(fg_mask, (W, H), interpolation=cv2.INTER_LINEAR)

        # Binarize after resize
        fg_mask_binary = (fg_mask_resized > 0.5).astype(np.float32)

        fg_masks.append(fg_mask_binary)

    fg_masks = np.array(fg_masks)  # Shape: (49, H, W)

    # Take first 48 masks to match flow count (Flow[t] is between Frame[t] and Frame[t+1])
    fg_masks = fg_masks[:-1]  # Drop last mask, keep first 48

    fg_ratio = fg_masks.mean() * 100
    print(f"  ✓ Processed {len(fg_masks)} masks")
    print(f"  ✓ Resolution: {H}x{W}")
    print(f"  ✓ Foreground pixels: {fg_ratio:.1f}%")
    print(f"  ✓ Temporal alignment: SAM and video resampled identically (88→49 frames)")

    assert len(fg_masks) == T, f"Mask count {len(fg_masks)} doesn't match flow count {T}"

    # Apply segmentation-based scaling to flow
    print(f"\nApplying segmentation-based flow scaling...")
    print(f"  Foreground scale: {args.fg_scale}x")
    print(f"  Background scale: {args.bg_scale}x")

    scaled_flows = scale_flow_by_segmentation(
        flows_array,
        fg_masks,
        args.fg_scale,
        args.bg_scale
    )

    print(f"  ✓ Scaled mean magnitude: {np.sqrt(scaled_flows[:, 0]**2 + scaled_flows[:, 1]**2).mean():.3f}")

    # Save original and scaled flows for reference
    flow_output_dir = output_dir / 'flows'
    flow_output_dir.mkdir(exist_ok=True)

    np.save(flow_output_dir / 'original_flow.npy', flows_array)
    np.save(flow_output_dir / 'scaled_flow.npy', scaled_flows)
    print(f"\n  ✓ Saved flows to {flow_output_dir}/")

    # Now warp noise with the SCALED flow
    print(f"\nWarping noise with scaled flow...")
    print(f"  This uses the SCALED flow, not the original!")

    # Convert scaled flows to dx, dy format for noise warping
    # The get_noise_from_video function expects to calculate flow itself,
    # so we need to use a lower-level approach

    # IMPORTANT: Noise dimensions must be based on VIDEO resolution, not FLOW resolution
    # Flow is downsampled by FRAME factor, but noise is at full video resolution / LATENT
    video_height, video_width = video.shape[1:3]  # Should be (480, 720)
    noise_channels = 16
    noise_height = video_height // LATENT  # 480 // 8 = 60
    noise_width = video_width // LATENT     # 720 // 8 = 90

    print(f"  Video dimensions: {video_height}x{video_width}")
    print(f"  Noise dimensions: {noise_height}x{noise_width} with {noise_channels} channels")

    # Use NoiseWarper to warp noise frame by frame
    import torch
    warper = nw.NoiseWarper(
        noise_channels,
        noise_height,
        noise_width,
        device=device,
        scale_factor=int(FRAME * FLOW)
    )

    # Collect warped noises
    warped_noises = []
    warped_noises.append(warper.noise.cpu().permute(1, 2, 0).numpy())

    for t in rp.eta(range(T), title='Warping Noise'):
        dx = scaled_flows[t, 0]  # Use SCALED flow
        dy = scaled_flows[t, 1]

        warper(dx, dy)

        warped_noise = warper.noise.cpu().permute(1, 2, 0).numpy()
        warped_noises.append(warped_noise)

    warped_noises = np.array(warped_noises)
    print(f"  ✓ Warped noise shape: {warped_noises.shape}")

    # Save warped noises
    np.save(output_dir / 'noises.npy', warped_noises)
    print(f"  ✓ Saved to: {output_dir / 'noises.npy'}")

    # Save flows in the format expected by inference
    np.save(output_dir / 'flows_dxdy.npy', scaled_flows)
    print(f"  ✓ Saved to: {output_dir / 'flows_dxdy.npy'}")

    # Save other required files
    rp.save_image(video[0], str(output_dir / 'first_frame.png'))
    rp.save_video_mp4(video, str(output_dir / 'input.mp4'), framerate=12, video_bitrate='max')

    # Create visualization (match original get_noise_from_video() approach)
    print("\nCreating noise visualization...")
    vis_frames = []
    for t in range(len(warped_noises)):
        noise_frame = warped_noises[t]
        if noise_channels >= 3:
            vis_rgb = noise_frame[:, :, :3]
        else:
            vis_rgb = np.repeat(noise_frame[:, :, :1], 3, axis=2)

        # Apply EXACT original formula: / 4 + 0.5 (not adaptive normalization!)
        vis_rgb = vis_rgb / 4.0 + 0.5

        # Clip to valid range [0, 1]
        vis_rgb = np.clip(vis_rgb, 0, 1)

        vis_frames.append((vis_rgb * 255).astype(np.uint8))

    # Use framerate=30 to match original (not 12!)
    rp.save_video_mp4(vis_frames, str(output_dir / 'noise_video.mp4'), framerate=30, video_bitrate='max')

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nNext step: Use this for video generation:")
    print(f"  python src/pipeline/cut_and_drag_inference.py \\")
    print(f"      --warped_noise_dir {output_dir}/ \\")
    print(f"      --output_path results/generated/output.mp4 \\")
    print(f"      --prompt \"your prompt here\"")


if __name__ == '__main__':
    main()
