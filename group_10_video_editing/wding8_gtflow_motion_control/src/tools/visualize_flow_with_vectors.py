"""
Visualize optical flow with HSV color coding + vector arrows

Usage:
    python src/tools/visualize_flow_with_vectors.py <flow.npy> [output_path] [options]

Examples:
    # Basic visualization
    python src/tools/visualize_flow_with_vectors.py results/warped_noise/my_video/flows_dxdy.npy

    # Specify output path
    python src/tools/visualize_flow_with_vectors.py results/warped_noise/my_video/flows_dxdy.npy flow_viz.mp4

    # Custom arrow spacing and scale
    python src/tools/visualize_flow_with_vectors.py flow.npy output.mp4 --arrow_spacing 20 --arrow_scale 2.0

    # Save individual frames as images
    python src/tools/visualize_flow_with_vectors.py flow.npy output_dir/ --save_frames

Visualization:
    - HSV color coding:
      * Hue (color) = Direction of motion
        - Red = Right, Yellow = Down-right, Green = Down
        - Cyan = Left, Blue = Up-left, Magenta = Up
      * Saturation = Full (255)
      * Value (brightness) = Speed/magnitude of motion
    - Arrow vectors:
      * Overlaid on top of color-coded flow
      * Arrow direction = motion direction
      * Arrow length = motion magnitude
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def flow_to_hsv(flow_frame, white_background=True):
    """
    Convert optical flow to HSV color-coded image.

    Args:
        flow_frame: (2, H, W) or (H, W, 2) flow array
        white_background: If True, use white background instead of black

    Returns:
        RGB image (H, W, 3) uint8
    """
    # Handle different input formats
    if flow_frame.shape[0] == 2:
        # (2, H, W) -> (H, W, 2)
        flow_uv = np.transpose(flow_frame, (1, 2, 0))
    else:
        flow_uv = flow_frame

    flow_uv = flow_uv.astype(np.float32)
    h, w = flow_uv.shape[:2]

    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(flow_uv[..., 0], flow_uv[..., 1])

    # Normalize magnitude
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if white_background:
        # Start with white background
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Create HSV visualization for areas with motion
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # Hue: direction
        hsv[..., 1] = 255  # Saturation: full
        hsv[..., 2] = 255  # Value: full brightness for colored areas

        # Convert to RGB
        flow_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Blend based on magnitude (magnitude acts as alpha)
        alpha = mag_norm[:, :, np.newaxis] / 255.0
        rgb = (alpha * flow_colored + (1 - alpha) * rgb).astype(np.uint8)
    else:
        # Original black background
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # Hue: direction
        hsv[..., 1] = 255  # Saturation: full
        hsv[..., 2] = mag_norm  # Value: magnitude (bright = fast, dark = slow)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def draw_flow_vectors(image, flow_frame, arrow_spacing=16, arrow_scale=1.0,
                       color=(255, 255, 255), thickness=1, tip_length=0.3):
    """
    Draw arrow vectors on top of an image to show flow direction.

    Args:
        image: RGB image (H, W, 3) to draw on
        flow_frame: (2, H, W) or (H, W, 2) flow array
        arrow_spacing: Spacing between arrows in pixels
        arrow_scale: Scale factor for arrow length
        color: Arrow color (R, G, B)
        thickness: Arrow line thickness
        tip_length: Arrow tip length relative to arrow length

    Returns:
        Image with arrows drawn (H, W, 3) uint8
    """
    # Handle different input formats
    if flow_frame.shape[0] == 2:
        # (2, H, W) -> (H, W, 2)
        flow_uv = np.transpose(flow_frame, (1, 2, 0))
    else:
        flow_uv = flow_frame

    h, w = flow_uv.shape[:2]
    output = image.copy()

    # Draw arrows on a grid
    for y in range(arrow_spacing // 2, h, arrow_spacing):
        for x in range(arrow_spacing // 2, w, arrow_spacing):
            # Get flow at this point
            fx = flow_uv[y, x, 0] * arrow_scale
            fy = flow_uv[y, x, 1] * arrow_scale

            # Only draw if flow is significant (avoid cluttering with tiny arrows)
            magnitude = np.sqrt(fx**2 + fy**2)
            if magnitude > 0.5:  # Threshold to avoid noise
                # Start and end points
                start_point = (int(x), int(y))
                end_point = (int(x + fx), int(y + fy))

                # Draw arrow
                cv2.arrowedLine(output, start_point, end_point, color,
                               thickness=thickness, tipLength=tip_length)

    return output


def visualize_flow_with_vectors(flow, output_path=None, arrow_spacing=16,
                                arrow_scale=1.0, save_frames=False, fps=12,
                                white_background=True):
    """
    Visualize optical flow with HSV color coding + vector arrows.

    Args:
        flow: (T, 2, H, W) flow array
        output_path: Path to save video/images (optional)
        arrow_spacing: Spacing between arrows in pixels
        arrow_scale: Scale factor for arrow length
        save_frames: If True, save individual frames as images
        fps: Frames per second for video
        white_background: If True, use white background instead of black

    Returns:
        List of RGB frames with visualization
    """
    T = flow.shape[0]
    print(f"Visualizing {T} frames of optical flow...")
    print(f"Background: {'white' if white_background else 'black'}")

    frames = []

    # Choose arrow color based on background
    arrow_color = (0, 0, 0) if white_background else (255, 255, 255)

    for t in range(T):
        flow_frame = flow[t]  # (2, H, W)

        # Create HSV color-coded background
        hsv_image = flow_to_hsv(flow_frame, white_background=white_background)

        # Draw vectors on top
        vis_image = draw_flow_vectors(
            hsv_image,
            flow_frame,
            arrow_spacing=arrow_spacing,
            arrow_scale=arrow_scale,
            color=arrow_color,  # Black arrows on white, white on black
            thickness=1,
            tip_length=0.3
        )

        frames.append(vis_image)

        if save_frames and output_path:
            # Save individual frame
            frame_dir = Path(output_path).parent / (Path(output_path).stem + "_frames")
            frame_dir.mkdir(exist_ok=True, parents=True)
            frame_path = frame_dir / f"frame_{t:04d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    # Save video if output path provided
    if output_path and not save_frames:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        if output_path.suffix in ['.mp4', '.avi', '.mov']:
            # Save as video
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            for frame in frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            out.release()
            print(f"✓ Saved video to: {output_path}")

        elif output_path.suffix == '.png':
            # Save as grid of frames
            n_cols = min(5, T)
            n_rows = (T + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.reshape(n_rows, n_cols)

            for idx, frame in enumerate(frames):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].imshow(frame)
                axes[row, col].set_title(f'Frame {idx}')
                axes[row, col].axis('off')

            # Hide unused subplots
            for idx in range(len(frames), n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].axis('off')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved frame grid to: {output_path}")

    return frames


def create_color_wheel_legend(save_path=None):
    """
    Create a color wheel legend showing what each color means.

    Args:
        save_path: Optional path to save the legend image

    Returns:
        RGB image of color wheel
    """
    size = 512
    center = size // 2
    radius = size // 2 - 20

    # Create image
    hsv = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            dist = np.sqrt(dx**2 + dy**2)

            if dist <= radius:
                # Calculate angle (direction)
                angle = np.arctan2(dy, dx)
                # Calculate magnitude (distance from center)
                mag = dist / radius

                # Map to HSV
                hsv[y, x, 0] = int((angle + np.pi) * 180 / np.pi / 2)  # Hue
                hsv[y, x, 1] = 255  # Saturation
                hsv[y, x, 2] = int(mag * 255)  # Value

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Add labels
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title('Optical Flow Color Wheel\n(Hue = Direction, Brightness = Speed)',
                 fontsize=14, fontweight='bold')

    # Add direction labels
    labels = [
        (center + radius, center, 'Right →', 'left'),
        (center - radius, center, '← Left', 'right'),
        (center, center - radius, '↑ Up', 'center'),
        (center, center + radius, '↓ Down', 'center'),
    ]

    for x, y, text, ha in labels:
        ax.text(x, y, text, fontsize=12, ha=ha, va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved color wheel legend to: {save_path}")

    plt.close()
    return rgb


def main():
    parser = argparse.ArgumentParser(
        description='Visualize optical flow with HSV color coding + vector arrows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('flow_path', type=str,
                       help='Path to flow numpy file (.npy)')
    parser.add_argument('output_path', type=str, nargs='?', default=None,
                       help='Output path for video/image (default: flow_path + _visualization.mp4)')
    parser.add_argument('--arrow_spacing', type=int, default=16,
                       help='Spacing between arrows in pixels (default: 16)')
    parser.add_argument('--arrow_scale', type=float, default=1.0,
                       help='Scale factor for arrow length (default: 1.0)')
    parser.add_argument('--save_frames', action='store_true',
                       help='Save individual frames as images instead of video')
    parser.add_argument('--fps', type=int, default=12,
                       help='Frames per second for video output (default: 12)')
    parser.add_argument('--black_background', action='store_true',
                       help='Use black background instead of white (default: white)')
    parser.add_argument('--create_legend', action='store_true',
                       help='Create a color wheel legend image')

    args = parser.parse_args()

    # Load flow
    print(f"Loading flow from: {args.flow_path}")
    flow = np.load(args.flow_path)
    print(f"Flow shape: {flow.shape}")
    print(f"Flow magnitude range: {np.sqrt(flow[:,0]**2 + flow[:,1]**2).min():.2f} - {np.sqrt(flow[:,0]**2 + flow[:,1]**2).max():.2f}")

    # Set default output path
    if args.output_path is None:
        flow_path = Path(args.flow_path)
        args.output_path = str(flow_path.parent / (flow_path.stem + '_visualization.mp4'))

    # Create visualization
    frames = visualize_flow_with_vectors(
        flow,
        output_path=args.output_path,
        arrow_spacing=args.arrow_spacing,
        arrow_scale=args.arrow_scale,
        save_frames=args.save_frames,
        fps=args.fps,
        white_background=not args.black_background  # Default is white
    )

    # Create color wheel legend if requested
    if args.create_legend:
        legend_path = Path(args.output_path).parent / 'flow_color_wheel_legend.png'
        create_color_wheel_legend(save_path=legend_path)

    print(f"\n✓ Done! Visualized {len(frames)} frames")
    print(f"  Output: {args.output_path}")
    if args.save_frames:
        frame_dir = Path(args.output_path).parent / (Path(args.output_path).stem + "_frames")
        print(f"  Frames: {frame_dir}/")


if __name__ == "__main__":
    main()
