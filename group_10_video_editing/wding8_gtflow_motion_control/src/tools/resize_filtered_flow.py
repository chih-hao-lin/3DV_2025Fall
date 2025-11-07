"""
Resize filtered flow to match target resolution

Usage:
    python resize_filtered_flow.py <input_flow.npy> <target_height> <target_width> [output.npy]
"""
import numpy as np
import cv2
import sys
from pathlib import Path


def resize_flow(flow, target_height, target_width):
    """
    Resize optical flow to target resolution.

    Args:
        flow: (T, 2, H, W) flow array
        target_height: Target height
        target_width: Target width

    Returns:
        Resized flow (T, 2, target_height, target_width)
    """
    T, _, H, W = flow.shape

    if H == target_height and W == target_width:
        print(f"Flow already at target resolution: {H}x{W}")
        return flow

    print(f"Resizing flow: ({T}, 2, {H}, {W}) → ({T}, 2, {target_height}, {target_width})")

    # Calculate scale factors
    scale_h = target_height / H
    scale_w = target_width / W

    print(f"Scale factors: height={scale_h:.3f}, width={scale_w:.3f}")

    # Create output array
    resized = np.zeros((T, 2, target_height, target_width), dtype=flow.dtype)

    # Resize each frame
    for t in range(T):
        # Resize u component (horizontal flow)
        u_frame = flow[t, 0].astype(np.float32)
        resized[t, 0] = cv2.resize(
            u_frame,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        ) * scale_w  # Scale flow values

        # Resize v component (vertical flow)
        v_frame = flow[t, 1].astype(np.float32)
        resized[t, 1] = cv2.resize(
            v_frame,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        ) * scale_h  # Scale flow values

    # Compare magnitudes
    orig_mag = np.sqrt(flow[:, 0]**2 + flow[:, 1]**2).mean()
    resized_mag = np.sqrt(resized[:, 0]**2 + resized[:, 1]**2).mean()

    print(f"\nMagnitude comparison:")
    print(f"  Original: {orig_mag:.3f}")
    print(f"  Resized:  {resized_mag:.3f}")
    print(f"  Ratio:    {resized_mag/orig_mag:.3f}")

    return resized


def main():
    if len(sys.argv) < 4:
        print("Usage: python resize_filtered_flow.py <input_flow.npy> <target_height> <target_width> [output.npy]")
        print("\nExample:")
        print("  python resize_filtered_flow.py dynamite_results/hands_emphasized_flow.npy 240 360")
        sys.exit(1)

    input_path = sys.argv[1]
    target_height = int(sys.argv[2])
    target_width = int(sys.argv[3])

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        # Auto-generate output name
        input_p = Path(input_path)
        output_path = input_p.parent / f"{input_p.stem}_{target_height}x{target_width}.npy"

    print("="*60)
    print("Resizing Filtered Flow")
    print("="*60)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Target: {target_height}x{target_width}")

    # Load flow
    print(f"\nLoading flow...")
    flow = np.load(input_path)
    print(f"✓ Loaded shape: {flow.shape}")

    # Resize
    print(f"\nResizing...")
    resized = resize_flow(flow, target_height, target_width)

    # Save
    print(f"\nSaving...")
    np.save(output_path, resized)
    print(f"✓ Saved: {output_path}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print(f"\nNow you can integrate this flow:")
    print(f"  python integrate_filtered_flow.py integrate {output_path} my_warped_noise/")


if __name__ == '__main__':
    main()
