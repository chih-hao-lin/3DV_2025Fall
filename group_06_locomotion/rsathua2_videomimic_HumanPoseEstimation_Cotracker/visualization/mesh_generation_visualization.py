# Copyright (c) Berkeley VideoMimic team.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# For questions or issues, please contact Hongsuk Choi (redstonepo@gmail.com)

import numpy as np
import time
import viser
import trimesh
import tyro
from pathlib import Path
from typing import Optional, Tuple


def visualize_mesh_generation(
    mesh: trimesh.Trimesh,
    bg_points: np.ndarray,
    bg_colors: np.ndarray,
    no_spf: bool = False
) -> None:
    """
    Visualize mesh generation results using viser.
    
    Args:
        mesh: Generated trimesh mesh
        bg_points: Background points used for mesh generation (N, 3)
        bg_colors: Background colors for points (N, 3)
        no_spf: Whether spatiotemporal filtering was skipped
    """
    print("Visualizing mesh generation results...")
    
    # Start viser server
    server = viser.ViserServer(port=8083)
    
    # Add visualization controls
    with server.gui.add_folder("Visualization"):
        gui_show_mesh = server.gui.add_checkbox("Show Mesh", True)
        gui_show_points = server.gui.add_checkbox("Show Points", True)
        gui_point_size = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=0.01)
    
    # Add mesh
    mesh_handle = server.scene.add_mesh_simple(
        name="background_mesh",
        vertices=mesh.vertices,  # (V, 3)
        faces=mesh.faces,  # (F, 3)
        visible=True,
        color=(180, 130, 255),
        side="double"
    )
    
    
    # Add point cloud
    points_handle = server.scene.add_point_cloud(
        name="background_points",
        points=bg_points,  # (N, 3)
        colors=bg_colors,  # (N, 3)
        point_size=gui_point_size.value,
        point_shape='circle',
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        visible=True
    )
    
    # Add info text
    filtering_type = "No Spatiotemporal Filtering" if no_spf else "With Spatiotemporal Filtering"
    server.gui.add_text("Info", f"Mesh Generation Results\n{filtering_type}\nPoints: {len(bg_points)}\nMesh Vertices: {len(mesh.vertices)}\nMesh Faces: {len(mesh.faces)}")
    
    # Add statistics
    with server.gui.add_folder("Mesh Statistics"):
        server.gui.add_text("Vertices", f"{len(mesh.vertices):,}", disabled=True)
        server.gui.add_text("Faces", f"{len(mesh.faces):,}", disabled=True)
        server.gui.add_text("Edges", f"{len(mesh.edges):,}", disabled=True)
        server.gui.add_text("Is Watertight", f"{mesh.is_watertight}", disabled=True)
        server.gui.add_text("Volume", f"{mesh.volume:.4f}", disabled=True)
        server.gui.add_text("Surface Area", f"{mesh.area:.4f}", disabled=True)
    
    with server.gui.add_folder("Point Cloud Statistics"):
        server.gui.add_text("Total Points", f"{len(bg_points):,}", disabled=True)
        bbox_min = bg_points.min(axis=0)  # (3,)
        bbox_max = bg_points.max(axis=0)  # (3,)
        bbox_size = bbox_max - bbox_min  # (3,)
        server.gui.add_text("Bounding Box", f"Size: [{bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}]", disabled=True)
    
    # Update visualization based on controls
    @gui_show_mesh.on_update
    def _(_) -> None:
        mesh_handle.visible = gui_show_mesh.value
    
    @gui_show_points.on_update
    def _(_) -> None:
        points_handle.visible = gui_show_points.value
    
    @gui_point_size.on_update
    def _(_) -> None:
        # Directly update point size without recreating point cloud
        points_handle.point_size = gui_point_size.value
    
    # Add camera controls info
    with server.gui.add_folder("Camera Controls"):
        server.gui.add_text("Translate", "Left click + drag / wsad", disabled=True)
        server.gui.add_text("Rotate", "Right click + drag", disabled=True)
        server.gui.add_text("Zoom", "Scroll", disabled=True)
    
    print("Mesh generation visualization server running on port 8083")
    print("Use the GUI controls to adjust visualization settings")
    
    # Keep server running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Visualization stopped by user")


def main(
    mesh_path: str,
    points_path: str
) -> None:
    """
    Visualize mesh generation results from file paths.
    
    Args:
        mesh_path: Path to background mesh OBJ file
        points_path: Path to background points PLY file
    """
    # Load mesh
    mesh = trimesh.load(mesh_path)
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    
    # Load point cloud
    point_cloud = trimesh.load(points_path)
    bg_points = point_cloud.vertices  # (N, 3)
    
    # Handle colors
    if hasattr(point_cloud.visual, 'vertex_colors') and point_cloud.visual.vertex_colors is not None:
        bg_colors = point_cloud.visual.vertex_colors[:, :3]  # (N, 3) - drop alpha if present
        if bg_colors.max() > 1.0:
            bg_colors = bg_colors / 255.0  # Convert to 0-1 range if needed
    else:
        # Use default colors if no colors in PLY
        bg_colors = np.ones((len(bg_points), 3)) * 0.5  # Gray default
    
    print(f"Loaded point cloud with {len(bg_points)} points")
    
    # Determine if spatiotemporal filtering was used based on filename
    no_spf = "less_filtered" in Path(points_path).name
    
    # Start visualization
    visualize_mesh_generation(mesh, bg_points, bg_colors, no_spf)


if __name__ == "__main__":
    tyro.cli(main)