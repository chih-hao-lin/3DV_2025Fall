# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import warp as wp
import numpy as np

@wp.kernel
def project_points(
    points: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    camera_intrinsics: wp.array(dtype=wp.mat33),
    camera_extrinsics: wp.array(dtype=wp.mat44),
    rgb_image: wp.array(dtype=wp.vec3),
    depth_image: wp.array(dtype=float),
    image_size: wp.vec2i,
    point_size: wp.int32
):
    # Get thread index
    tid = wp.tid()

    point_size = 5
    
    # Get point and color
    point = points[tid]
    color = colors[tid]
    
    # Transform point to camera space
    point_cam = wp.transform_point(camera_extrinsics[0], point)
    
    # Project to image space
    fx = camera_intrinsics[0][0][0]
    fy = camera_intrinsics[0][1][1]
    px = camera_intrinsics[0][0][2]
    py = camera_intrinsics[0][1][2]
    
    # Skip if point is behind camera
    if point_cam[2] <= 0.0:
        return
    
    # Project to image space
    # Flip x and y signs to match image coordinates (origin top-left, y down)
    x = (-point_cam[0] * fx / point_cam[2]) + px
    y = (-point_cam[1] * fy / point_cam[2]) + py
    
    # Convert to integer coordinates (center of the splat)
    ix_center = wp.int32(x)
    iy_center = wp.int32(y)

    # print(point_size)
    
    # Calculate neighborhood bounds
    half_size = point_size / 2
    start_x = ix_center - half_size
    start_y = iy_center - half_size
    end_x = start_x + point_size
    end_y = start_y + point_size
    
    # Iterate over the neighborhood
    for iy in range(start_y, end_y):
        for ix in range(start_x, end_x):
            # Check if point is within image bounds
            if ix < 0 or ix >= image_size[0] or iy < 0 or iy >= image_size[1]:
                continue # Skip pixels outside bounds
            
            # Get current depth at pixel
            pixel_idx = iy * image_size[0] + ix
            current_depth = depth_image[pixel_idx]
            
            # Perform depth test: Only update if point is closer OR if the pixel hasn't been written to yet
            if current_depth == 0.0 or point_cam[2] < current_depth:
                # Atomic operation might be needed here in a parallel setting if multiple points map to the same neighborhood,
                # but for simplicity, we'll overwrite. A race condition could occur where a farther point overwrites a closer one
                # if they project to overlapping neighborhoods and are processed concurrently.
                # Using atomicMin on depth_image[pixel_idx] with point_cam[2] would be more robust.
                depth_image[pixel_idx] = point_cam[2]
                rgb_image[pixel_idx] = color

def render_pointcloud_to_rgb_depth(
    points: np.ndarray,  # (N, 3) in world coordinates
    colors: np.ndarray,  # (N, 3) RGB colors
    camera_intrinsics: np.ndarray,  # (3, 3) camera intrinsics
    camera_extrinsics: np.ndarray,  # (4, 4) camera extrinsics
    image_size: tuple = (480, 640),  # (height, width)
    device: str = "cuda" if wp.is_cuda_available() else "cpu",
    point_size: int = 1  # Add point_size parameter with default value
):
    """
    Render point cloud to RGB and depth images using Warp.
    
    Args:
        points: Point cloud coordinates in world frame (N, 3)
        colors: Point cloud colors (N, 3)
        camera_intrinsics: Camera intrinsics matrix (3, 3)
        camera_extrinsics: Camera extrinsics matrix (4, 4)
        image_size: Output image size (height, width)
        device: Device to run the rendering on
        point_size: The side length of the square patch for rendering each point (e.g., 3 for 3x3)
    
    Returns:
        rgb_image: Rendered RGB image (H, W, 3)
        depth_image: Rendered depth image (H, W)
    """
    # Initialize Warp
    wp.init()
    
    # Convert inputs to Warp arrays
    points_wp = wp.array(points, dtype=wp.vec3, device=device)
    colors_wp = wp.array(colors, dtype=wp.vec3, device=device)
    camera_intrinsics_wp = wp.array(camera_intrinsics.reshape(1, 3, 3), dtype=wp.mat33, device=device)
    camera_extrinsics_wp = wp.array(camera_extrinsics.reshape(1, 4, 4), dtype=wp.mat44, device=device)
    
    # Create output arrays
    rgb_image_wp = wp.zeros((image_size[0] * image_size[1],), dtype=wp.vec3, device=device)
    depth_image_wp = wp.zeros((image_size[0] * image_size[1],), dtype=float, device=device)

    # Initialize RGB image with white background (1.0, 1.0, 1.0)
    rgb_image_wp = wp.full((image_size[0] * image_size[1],), wp.vec3(255.0, 255.0, 255.0), device=device)
    
    # Initialize depth image with large values (effectively infinity for depth testing)
    # depth_image_wp = wp.full((image_size[0] * image_size[1],), 1e10, device=device)
    
    # Launch kernel
    wp.launch(
        kernel=project_points,
        dim=len(points),
        inputs=[
            points_wp,
            colors_wp,
            camera_intrinsics_wp,
            camera_extrinsics_wp,
            rgb_image_wp,
            depth_image_wp,
            wp.vec2i(image_size[1], image_size[0]),  # width, height
            wp.int32(point_size)  # Pass point_size to the kernel
        ]
    )
    
    # Convert results to numpy arrays
    rgb_image = rgb_image_wp.numpy().reshape(image_size[0], image_size[1], 3)
    depth_image = depth_image_wp.numpy().reshape(image_size[0], image_size[1])
    
    return rgb_image, depth_image

def test_rendering():
    # Create sample data
    device = "cuda" if wp.is_cuda_available() else "cpu"
    
    # Create a simple point cloud (a cube)
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=np.float32)
    
    # Create colors (red cube)
    colors = np.ones_like(points) * np.array([1.0, 0.0, 0.0])
    
    # Create camera intrinsics (example values)
    camera_intrinsics = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create camera extrinsics (looking at the cube from a distance)
    camera_extrinsics = np.array([
        [1, 0, 0, 2],
        [0, 1, 0, 2],
        [0, 0, 1, 2],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    # Render
    rgb_image, depth_image = render_pointcloud_to_rgb_depth(
        points=points,
        colors=colors,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        image_size=(480, 640),
        device=device,
        point_size=25  # Specify the point size for rendering
    )
    
    # Normalize depth for visualization
    depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    
    return rgb_image, depth_image

if __name__ == "__main__":
    rgb, depth = test_rendering()
    import pdb; pdb.set_trace()
    print(f"RGB image shape: {rgb.shape}")
    print(f"Depth image shape: {depth.shape}")