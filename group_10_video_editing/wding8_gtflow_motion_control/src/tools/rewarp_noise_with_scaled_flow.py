"""
rewarp_noise_with_scaled_flow.py

Re-warp noise using scaled optical flow from segmentation.

This tool fixes the segmentation-based flow scaling pipeline by:
1. Loading scaled optical flow
2. Generating or loading noise structure
3. Warping noise with the SCALED flow (not original)
4. Saving the re-warped noise

WHY THIS IS NECESSARY:
The inference pipeline loads pre-warped noise from noises.npy.
When we scale optical flow after extraction, the old noises.npy is still
warped with the ORIGINAL flow. This tool re-warps the noise with the
SCALED flow so the motion scaling actually affects the generated video.

Usage:
    python src/tools/rewarp_noise_with_scaled_flow.py \
        --flow results/segmentation_scaled/train_fg1.5_bg0.3/scaled_flow.npy \
        --reference_dir results/warped_noise/train/ \
        --output_dir results/warped_noise/train_seg_scaled/ \
        --visualize
"""

import numpy as np
import torch
import sys
from pathlib import Path
import argparse
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import rp
rp.pip_import('fire')
rp.git_import('CommonSource')
import rp.git.CommonSource.noise_warp as nw


def rewarp_noise_with_flow(
    scaled_flow_path,
    reference_dir,
    output_dir,
    visualize=True,
    device=None
):
    """
    Re-warp noise using scaled optical flow.

    Args:
        scaled_flow_path: Path to scaled flow .npy file (T, 2, H, W)
        reference_dir: Reference warped_noise directory for metadata/structure
        output_dir: Output directory for re-warped noise
        visualize: Whether to create visualizations
        device: Torch device (auto-selected if None)
    """

    print("="*70)
    print("RE-WARPING NOISE WITH SCALED FLOW")
    print("="*70)

    # Setup
    if device is None:
        device = rp.select_torch_device(prefer_used=True)

    reference_dir = Path(reference_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load scaled flow
    print(f"\nLoading scaled flow from: {scaled_flow_path}")
    scaled_flow = np.load(scaled_flow_path)
    T, C, H, W = scaled_flow.shape
    assert C == 2, f"Expected 2 flow channels, got {C}"
    print(f"  Flow shape: {scaled_flow.shape}")
    print(f"  Mean magnitude: {np.sqrt(scaled_flow[:, 0]**2 + scaled_flow[:, 1]**2).mean():.3f}")

    # Load reference flow for comparison
    reference_flow_path = reference_dir / 'flows_dxdy.npy'
    reference_flow = np.load(reference_flow_path)
    print(f"  Reference flow mean magnitude: {np.sqrt(reference_flow[:, 0]**2 + reference_flow[:, 1]**2).mean():.3f}")

    # Check if flows are identical (e.g., fg=1, bg=1 case)
    flows_identical = np.array_equal(scaled_flow, reference_flow)
    if flows_identical:
        print(f"\n  ⚠️  NOTICE: Scaled flow is IDENTICAL to reference flow!")
        print(f"      This means no scaling was applied (e.g., fg=1.0, bg=1.0)")
        print(f"      Copying original warped noise instead of rewarping...")
        print(f"      This preserves the original random noise structure.")

    # Load reference metadata
    print(f"\nLoading reference structure from: {reference_dir}")
    reference_noise_path = reference_dir / 'noises.npy'

    if not reference_noise_path.exists():
        raise FileNotFoundError(f"Reference noise not found: {reference_noise_path}")

    reference_noise = np.load(reference_noise_path)
    print(f"  Reference noise shape: {reference_noise.shape}")

    # Get noise parameters from reference
    noise_T, noise_H, noise_W, noise_C = reference_noise.shape

    # Verify dimensions match
    if noise_T != T:
        print(f"  WARNING: Frame count mismatch (flow: {T}, reference noise: {noise_T})")
        T = min(T, noise_T)
        print(f"  Using first {T} frames")
        scaled_flow = scaled_flow[:T]

    # If flows are identical, just copy the original warped noise
    if flows_identical:
        print(f"\nCopying original warped noise (no rewarping needed)...")
        warped_noise = reference_noise[:T]

        # Save warped noise
        output_noise_path = output_dir / 'noises.npy'
        np.save(output_noise_path, warped_noise)
        print(f"  ✓ Copied to: {output_noise_path}")

        # Save flow for reference
        output_flow_path = output_dir / 'flows_dxdy.npy'
        np.save(output_flow_path, scaled_flow)
        print(f"  ✓ Saved flow to: {output_flow_path}")

        # Copy other necessary files from reference
        print("\nCopying reference files...")
        files_to_copy = ['input.mp4', 'first_frame.png']
        for filename in files_to_copy:
            src = reference_dir / filename
            if src.exists():
                dst = output_dir / filename
                shutil.copy2(src, dst)
                print(f"  ✓ Copied: {filename}")

        # Copy visualization if it exists
        if visualize:
            vis_src = reference_dir / 'noise_video.mp4'
            if vis_src.exists():
                vis_dst = output_dir / 'noise_video.mp4'
                shutil.copy2(vis_src, vis_dst)
                print(f"  ✓ Copied: noise_video.mp4")

        print("\n" + "="*70)
        print("COPY COMPLETE (NO REWARPING NEEDED)")
        print("="*70)
        return  # Early exit

    # Set random seed for reproducibility
    print(f"\nGenerating initial random noise ({noise_H}, {noise_W}, {noise_C})")
    print("  Setting random seed to 42 for reproducibility")

    # IMPORTANT: Set both numpy and torch seeds
    # The NoiseWarper uses torch.randn() internally
    np.random.seed(42)
    torch.manual_seed(42)
    if device != 'cpu':
        torch.cuda.manual_seed(42)

    # Convert to torch for warping
    print("\nWarping noise with scaled flow...")
    torch_noises = []

    # Use NoiseWarper class for proper warping
    # This will create initial random noise using torch.randn()
    # (controlled by the seed we just set above)
    warper = nw.NoiseWarper(
        noise_C,
        noise_H,
        noise_W,
        device=device,
        scale_factor=1  # Flow already at correct resolution
    )

    # Get first frame (initial random noise from warper)
    torch_noises.append(warper.noise.cpu().permute(1, 2, 0))

    # Warp through sequence
    for t in range(1, T):
        dx = scaled_flow[t-1, 0]  # numpy array
        dy = scaled_flow[t-1, 1]  # numpy array

        # Apply warping (NoiseWarper accepts numpy arrays)
        warper(dx, dy)

        # Get warped noise
        warped = warper.noise.cpu().permute(1, 2, 0)
        torch_noises.append(warped)

        if (t+1) % 10 == 0:
            print(f"  Warped {t+1}/{T} frames...")

    # Convert back to numpy
    warped_noise = torch.stack(torch_noises).numpy()
    print(f"\n  Final warped noise shape: {warped_noise.shape}")

    # Save warped noise
    output_noise_path = output_dir / 'noises.npy'
    np.save(output_noise_path, warped_noise)
    print(f"  ✓ Saved to: {output_noise_path}")

    # Save flow for reference
    output_flow_path = output_dir / 'flows_dxdy.npy'
    np.save(output_flow_path, scaled_flow)
    print(f"  ✓ Saved flow to: {output_flow_path}")

    # Copy other necessary files from reference
    print("\nCopying reference files...")
    files_to_copy = ['input.mp4', 'first_frame.png']
    for filename in files_to_copy:
        src = reference_dir / filename
        if src.exists():
            dst = output_dir / filename
            shutil.copy2(src, dst)
            print(f"  ✓ Copied: {filename}")

    # Create visualizations if requested
    if visualize:
        print("\nCreating visualizations...")

        # Create noise visualization video
        try:
            vis_frames = []
            for t in range(T):
                # Normalize noise to [0, 1] for visualization
                noise_frame = warped_noise[t]

                # Use first 3 channels as RGB
                if noise_C >= 3:
                    vis_rgb = noise_frame[:, :, :3]
                else:
                    vis_rgb = np.repeat(noise_frame[:, :, :1], 3, axis=2)

                # Normalize to [0, 1]
                vis_rgb = (vis_rgb - vis_rgb.min()) / (vis_rgb.max() - vis_rgb.min() + 1e-8)
                vis_frames.append((vis_rgb * 255).astype(np.uint8))

            output_vis_path = output_dir / 'noise_video.mp4'
            rp.save_video_mp4(vis_frames, str(output_vis_path), framerate=12, video_bitrate='max')
            print(f"  ✓ Saved noise visualization: {output_vis_path}")

        except Exception as e:
            print(f"  Warning: Could not create visualization: {e}")

    print("\n" + "="*70)
    print("RE-WARPING COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nNext step: Use this warped_noise directory for video generation:")
    print(f"  python src/pipeline/cut_and_drag_inference.py \\")
    print(f"      --warped_noise_dir {output_dir}/ \\")
    print(f"      --output_path results/generated/output.mp4 \\")
    print(f"      --prompt \"your prompt here\"")


def main():
    parser = argparse.ArgumentParser(
        description='Re-warp noise with scaled optical flow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--flow', type=str, required=True,
                       help='Path to scaled flow .npy file')
    parser.add_argument('--reference_dir', type=str, required=True,
                       help='Reference warped_noise directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for re-warped noise')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization video')
    parser.add_argument('--device', type=str, default=None,
                       help='Torch device (default: auto-select)')

    args = parser.parse_args()

    rewarp_noise_with_flow(
        scaled_flow_path=args.flow,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        visualize=args.visualize,
        device=args.device
    )


if __name__ == '__main__':
    main()
