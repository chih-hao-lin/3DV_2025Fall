# Commands Guide

This guide provides step-by-step commands for processing human motion videos using the VideoMimic Real-to-Sim pipeline. The complete workflow transforms single-camera videos into robot-ready motion data.

**Environment Requirements**

- **Main processing**: `vm1rs` environment (Python 3.12, CUDA 12.4+)
- **MegaSam + GeoCalib + NKSR meshification**: `vm1recon` environment (Python 3.10, CUDA 11.8)

---

## Quick Start

For a complete pipeline with a single command:
```bash
# Extract frames first
python utilities/extract_frames_from_video.py --video-path {video_name}.{extension} --output-dir ./demo_data/input_images/{video_name}/cam01 --start-frame 0 --end-frame 300
# This will save images with new indexing from 00001.jpg. The files should be named as 00001.jpg, 00002.jpg, ...

# Complete pipeline automation
./process_video.sh <video_name> <start_frame> <end_frame> <subsample_factor> g1 <height of the human>
# Example: ./process_video.sh my_video 0 100 2 g1 1.8
# If height is 0, it will use the SMPL shape parameters fitted to the robot.
# If height is -1, it will use the estimated SMPL shape parameters.
```
<details>
<summary><b>Tips for Capturing Videos</b></summary>

- Moving camera is good. Try to have enough parallex.  

- Try to capture the whole surface of the scene for a complete mesh.  

- Keep human in the center of the scene and make sure the human is not small or big.   

- Avoid occlusion on the human.   

- Avoid textureless surfaces. Ex) indoor white walls, very bright outdoor, etc.   
</details>

<details>
<summary><b>Data Structure for Simulation Ready Outputs</b></summary>  


```
./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1/
├── gravity_calibrated_keypoints.h5                    # Gravity-aligned human keypoints and world rotation
├── gravity_calibrated_megahunter.h5        # Updated megahunter file with gravity-calibrated coordinates
├── background_mesh.obj                                # Environment mesh -> we use this for task environment in Imitation Learning
├── background_less_filtered_colored_pointcloud.ply    # Point cloud with minimal filtering
├── background_more_filtered_colored_pointcloud.ply    # Point cloud with spatiotemporal filtering
├── retarget_poses_g1.h5                              # Robot motion data (after retargeting) -> we use this for reference motion in Imitation Learning 
└── retarget_poses_g1_multiperson.h5                  # Robot motion data (after retargeting) for multiple humans
```
</details>

<details>
<summary><b>Visualize Results (Environment + Human + Robot)</b></summary>

<!-- **Visualize Results (Environment + Human + Robot)** -->
```bash
# Visualize complete results with ego-view rendering and interactive GUI
python visualization/complete_results_egoview_visualization.py \
    --postprocessed-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1 \
    --robot-name g1 \
    --bg-pc-downsample-factor 4 \
    --is-megasam \
    --save-ego-view
```
</details>

<details>
<summary><b>Multi-Human Support</b></summary>

<!-- **Multi-Human Support** -->

The pipeline supports processing multiple humans in a scene. For detailed documentation on multi-human processing, see [**docs/multihuman.md**](./multihuman.md).

**Quick Example:**
```bash
# Enable multi-human processing with top-k selection
python stage0_preprocessing/vitpose_2d_poses.py ... --multihuman --top-k 3
python stage2_optimization/megahunter_optimization.py ... --multihuman --top-k 3
```

**Key Points:**
- By default, only the largest person (by average bounding box area) is processed
- Use `--multihuman --top-k N` to process the N largest humans
- **Memory usage scales with the number of humans**
</details>

---

For step-by-step processing and visualization, continue reading below (recommended).

---


## Stage 0: Preprocessing (Human Detection & Pose Estimation)

### Option A: Automated Preprocessing (Recommended)
```bash
# Runs SAM2, ViTPose, VIMO, and BSTRO in sequence
bash preprocess_human.sh <video_name> <vis_flag>
# Examples:
bash preprocess_human.sh my_video 1    # with visualization, which saves ID of humans, masks/bboxes and 2d poses of humans in the video
bash preprocess_human.sh my_video 0    # without visualization
```

### Option B: Manual Step-by-Step Preprocessing

**Step 0.1: Human Detection and Segmentation (SAM2)**
```bash
# Extract human masks and bounding boxes
python stage0_preprocessing/sam2_segmentation.py \
    --video-dir ./demo_data/input_images/my_video/cam01 \
    --output-dir ./demo_data/input_masks/my_video/cam01 \
    --vis
```

<details>
<summary>Output structure</summary>

```
./demo_data/input_masks/my_video/cam01/
├── json_data/          # Bounding box data
│   ├── mask_00001.json
│   ├── mask_00002.json
│   └── ...
├── mask_data/          # Segmentation masks (H×W arrays with object IDs)
│   ├── mask_00001.npz
│   ├── mask_00002.npz
│   └── ...
├── result/             # Visualization images of ID of humans, masks and bboxes from sam2
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
├── meta_data.json        # Metadata for the sam2 output
└── sam2_output.mp4     # Video with overlaid ID of humans, masks and bboxes from sam2
```

**meta_data.json**: Located in `input_masks/{video}/cam01/`, contains:
  - `largest_area_id`: ID of the largest detected person
  - `all_instance_ids`: List of all detected person IDs
  - `sorted_by_avg_area`: Person IDs sorted by average bounding box area
  - `avg_areas`: Average bounding box area for each person
  - `frame_counts`: Number of frames each person appears in
</details>

**Step 0.2: 2D Pose Estimation (ViTPose)**
> Don't worry about the "The model and loaded state dict do not match exactly" error, unless the visualization looks weird.
```bash
# Extract 2D human poses
python stage0_preprocessing/vitpose_2d_poses.py \
    --video-dir ./demo_data/input_images/my_video/cam01 \
    --bbox-dir ./demo_data/input_masks/my_video/cam01/json_data \
    --output-dir ./demo_data/input_2d_poses/my_video/cam01 \
    --vis
```

**Step 0.3: 3D Human Mesh Estimation (VIMO)**
```bash
# Extract 3D human body meshes
python stage0_preprocessing/vimo_3d_mesh.py \
    --img-dir ./demo_data/input_images/my_video/cam01 \
    --mask-dir ./demo_data/input_masks/my_video/cam01 \
    --out-dir ./demo_data/input_3d_meshes/my_video/cam01 
# I (Hongsuk) didn't implement visualization for this because body mesh rendering is super slow. Just check after the megahunter optimization with viser.
```

**Step 0.4: Contact Detection (BSTRO)**

BSTRO provides contact in human mesh vertex level, and we use this to detect contact between human feet and ground.
```bash
# Detect human-ground contact points
python stage0_preprocessing/bstro_contact_detection.py \
    --video-dir ./demo_data/input_images/my_video/cam01 \
    --bbox-dir ./demo_data/input_masks/my_video/cam01/json_data \
    --output-dir ./demo_data/input_contacts/my_video/cam01 \
    --feet-contact-ratio-thr 0.2 \
    --contact-thr 0.95 
```

### Optional: Hand Pose Estimation and SMPL-X Bodies

<details>
<summary>Deprecated - Not actively maintained after Jan 2025</summary>

**Step 0.5: Hand Mesh Estimation (WiLor)**
```bash
# Requires ViTPose to be run first for hand bounding boxes
python stage0_preprocessing/wilor_hand_poses.py \
    --img-dir ./demo_data/input_images/my_video/cam01 \
    --pose2d-dir ./demo_data/input_2d_poses/my_video/cam01 \
    --output-dir ./demo_data/input_3d_meshes/my_video/cam01
```

**Step 0.6: SMPL + MANO → SMPL-X Conversion**
```bash
python stage0_preprocessing/smpl_to_smplx_conversion.py \
    --smpl-dir ./demo_data/input_3d_meshes/my_video/cam01 \
    --mano-dir ./demo_data/input_3d_meshes/my_video/cam01 \
    --output-dir ./demo_data/input_3d_meshes/my_video/cam01
```
</details>

---

## Stage 1: World Environment Reconstruction

Choose **one** reconstruction method based on your VRAM and accuracy requirements:

### Option A: MegaSam (High Accuracy, Memory Efficient, ~24GB+ VRAM for 300 frames)
```bash
# High-quality reconstruction (requires vm1recon environment)
conda activate vm1recon
python stage1_reconstruction/megasam_reconstruction.py \
    --out-dir ./demo_data/input_megasam \
    --video-dir ./demo_data/input_images/my_video/cam01 \
    --start-frame 0 \
    --end-frame 100 \
    --stride 1 \
    --gsam2
```

**Visualize MegaSam results:**

Note that the world environment reconstruction does not have scale information. That's why we have `world-scale-factor` to have better visualization. And meetgasam does not have confidence output, so set the threshold to 0.0.

```bash
python visualization/environment_only_visualization.py \
    --world-env-path ./demo_data/input_megasam/megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --world-scale-factor 1.5 \
    --conf-thr 0.0
```

### Option B: Depth Prior-conditioned Monst3r, aka, Align3r (Effective for Textureless Scenes, ~80GB+ VRAM for 150 frames)
```bash
# Alternative reconstruction method for textureless scenes
conda activate vm1rs
python stage1_reconstruction/monst3r_depth_prior_reconstruction.py \
    --out-dir ./demo_data/input_align3r \
    --video-dir ./demo_data/input_images/my_video/cam01 \
    --start-frame 0 \
    --end-frame 100 \
    --stride 1 \
    --gsam2
    # Add --no-batchify for even less memory usage (slower)
```

**Visualize Monst3r results:**

Note that the world environment reconstruction does not have accurate scale information despite data-driven depth prior. Normally they are very smaller than the real world. That's why we have `world-scale-factor` to have better visualization. 

```bash
python visualization/environment_only_visualization.py \
    --world-env-path ./demo_data/input_align3r/align3r_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --world-scale-factor 5.0 \
    --conf-thr 1.5
```

---

## Stage 2: MegaHunter Optimization (Human Motion and World Alignment)

### Optional: Optimize SMPL Shape for Specific Height
```bash
# Customize human body shape for specific height
python stage2_optimization/optimize_smpl_shape_for_height.py \
    --height 1.8 \
    --output-dir ./demo_data/input_3d_meshes/my_video/cam01 \
    --vis
```

### Optional: Optimize SMPL Shape for Specific Robot

<details>
<summary>Deprecated - Not actively maintained after JAXMP deprecated in May 2025. </summary>

- This is using [jaxmp](https://github.com/chungmin99/jaxmp), which is an arxived version of [PyRoki](https://github.com/chungmin99/pyroki). Recommended to make new code with PyRoki. - Hongsuk
- You don't have to do this unless you want better fit for the robot or for a custom robot. The SMPL shape for the robot is already saved in this repo.

```bash
# Optimize SMPL shape to match specific robot proportions (G1 or H1)
# Actually, the support for H1 is deprecated. If you’d like, you can find the correspondence between human and humanoid joints yourself.
python stage2_optimization/optimize_smpl_shape_for_robot.py \
    --robot-name g1 \
    --output-dir ./assets/robot_asset/g1/ \
    --vis
```
</details>

### MegaHunter (SfM + SMPL Optimization)

- It’s recommended to use fewer than 300 frames as input. Using more can cause GPU memory to overflow even on an A100 80GB.
- This code pads the number of frames to the next multiple of 100 (e.g., 94 → 100) to enable JAX JIT compilation once and allow repeated execution. 
```bash
conda activate vm1rs
python stage2_optimization/megahunter_optimization.py \
    --world-env-path ./demo_data/input_megasam/megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --bbox-dir ./demo_data/input_masks/my_video/cam01/json_data \
    --pose2d-dir ./demo_data/input_2d_poses/my_video/cam01 \
    --smpl-dir ./demo_data/input_3d_meshes/my_video/cam01 \
    --out-dir ./demo_data/output_smpl_and_points
    # Add --use-g1-shape for G1 robot body proportions. It effectively scales up the world reconstruction results to the size of the robot, instead of the human body size.
    # Add --vis for visualization during optimization; continue with the optimization process by canceling the visualization with ctrl+c
```

**Sequential Processing for Multiple Videos:**
For processing multiple videos efficiently, use the sequential processing script that amortizes JAX compilation time:
```bash
python sequential_processing/stage2_sequential_megahunter_optimization.py --pattern {common substring of the video names} --use-g1-shape
```
See [sequential_processing.md](./sequential_processing.md) for details.

### Visualize Optimization Results
If feet contact exists, this visualization will show the SMPL feet vertices as green. it's just binary contact prediction.
```bash
python visualization/optimization_results_visualization.py \
    --world-env-path ./demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --bg-pc-downsample-factor 4
```

---

## Stage 3: Postprocessing (Gravity Calibration & Mesh Generation)

### Option A: Complete Pipeline (Recommended)
```bash
# Switch to vm1recon environment for GeoCalib and NKSR meshification
conda activate vm1recon
python stage3_postprocessing/postprocessing_pipeline.py \
    --megahunter-path ./demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --out-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1 \
    --conf-thr 0.0 \
    --is-megasam \
    --scale-bbox3d 1.5 \
    --vis
    # Use --gradient-thr 0.15 for cleaner meshes
    # Use --no-spf to meshify without spatiotemporal filtering (slower)
    # Use --meshification-method 'nksr' (default) or 'ndc' (deprecated) for neural meshification
    # Add --multihuman for multi-human scenes
```

### Option B: Step-by-Step Processing

**Step 3.1: Gravity Calibration**

This stage saves gravity calibrated results in two files in the output directory (e.g., `./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1`):
- `gravity_calibrated_keypoints.h5`: Gravity-aligned human keypoints and world rotation. 
- `gravity_calibrated_megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5`: Updated megahunter file with gravity-calibrated coordinates

```bash
# Apply gravity calibration to align world coordinates with gravity
conda activate vm1recon
python stage3_postprocessing/gravity_calibration.py \
    --megahunter-path ./demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --out-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1 \
    --world-scale-factor 1.0 \
    --is-megasam \
    # --multihuman  # Add for multi-human scenes
    # --vis; if set vis, continue with the optimization process by canceling the visualization with ctrl+c
```

**Step 3.2: Mesh Generation**
```bash
# Generate background mesh from gravity-calibrated data
conda activate vm1recon
python stage3_postprocessing/mesh_generation.py \
    --calibrated-megahunter-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1 \
    --conf-thr 0.0 \
    --gradient-thr 0.15 \
    --scale-bbox3d 3.0 \
    --is-megasam \
    --meshification-method nksr \
    --vis
    # Use --no-spf to meshify without spatiotemporal filtering (slower because it meshifies whole scene at once, but you will get more complete environment mesh)
    # Add --multihuman for multi-human scenes
```

### Visualization Commands

**Visualize Gravity Calibration Results**

Simple visualization of gravity-aligned human poses and world axes.

```bash
# Interactive visualization of gravity-aligned human poses and world axes
python visualization/gravity_calibration_visualization.py \
    --calib-out-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1
```

Whole MegaHunter optimization results visualization.

```bash
python visualization/optimization_results_visualization.py \
    --world-env-path ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1/gravity_calibrated_megahunter.h5 \
    --bg-pc-downsample-factor 4
```


**Visualize Mesh Generation Results**
```bash
# Interactive mesh and point cloud viewer with filtering controls
python visualization/mesh_generation_visualization.py \
    --mesh-path ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1/background_mesh.obj \
    --points-path ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1/background_more_filtered_colored_pointcloud.ply
```

## Step 4: Robot Motion Retargeting

The retargeting is very sensitive to the cost weights. You can try different weights to get the best result. If the retargeting looks weird, first try setting the `foot-skating-cost-weight`, `ground-contact-cost-weight`, and `world-coll-factor-weight` to 0.0.

```bash
conda activate vm1rs
python stage4_retargeting/robot_motion_retargeting.py \
    --src-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1/ \
    --contact-dir ./demo_data/input_contacts/my_video/cam01 \
    --vis
```

**Multi-Human Retargeting:**
```bash
# Process multiple humans
python stage4_retargeting/robot_motion_retargeting.py \
    --src-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1/ \
    --contact-dir ./demo_data/input_contacts/my_video/cam01 \
    --multihuman \
    --top-k 3 \ # at this point, you should already have top-k humans from the optimization stage
    --vis
```

<details>
<summary>Retargeting Cost Weight Parameters</summary>
The retargeting optimization uses multiple cost functions to ensure realistic and feasible robot motion. These weights can be adjusted based on your specific requirements:

**Pose Matching Costs:**
- `--local-pose-cost-weight` (default: 8.0): Enforces relative joint position similarity between human and robot. Controls how well the robot maintains the geometric relationships between body parts (e.g., arm length ratios, limb orientations).

- `--global-pose-cost-weight` (default: 2.0): Matches absolute joint positions in world space after centering both skeletons. Ensures overall pose similarity while accounting for size differences.

- `--end-effector-cost-weight` (default: 5.0): Prioritizes accurate foot placement by matching human and robot foot positions exactly. Critical for maintaining ground contact and balance.

**Motion Quality Costs:**
- `--foot-skating-cost-weight` (default: 10.0): Prevents foot sliding when in contact with ground. Enforces that feet remain stationary during contact phases to avoid unrealistic skating motion.

- `--ground-contact-cost-weight` (default: 0.5): Aligns robot feet with the detected ground surface height during contact. Uses raycasting on the background mesh to determine target ground positions.

- `--smoothness-cost-factor-weight` (default: 10.0): Reduces joint velocity changes between consecutive frames. Creates smoother, more natural-looking motion by penalizing abrupt joint angle changes.

**Robot Constraint Costs:**
- `--self-coll-factor-weight` (default: 1.0): Prevents robot self-collision by maintaining minimum distances between body parts. Essential for generating physically feasible poses.

- `--world-coll-factor-weight` (default: 0.1): Avoids collision with the reconstructed environment mesh. Uses a low weight to lift the robot from ground penetration without overly constraining motion.

- `--world-coll-margin` (default: 0.01): Safety margin (in meters) for world collision detection. Larger values create more conservative motion away from obstacles.

- `--limit-cost-factor-weight` (default: 1000.0): Enforces robot joint limits to prevent impossible configurations. High weight ensures generated motion respects hardware constraints.

**Joint-Specific Regularization:**
- `--hip-yaw-cost-weight` (default: 5.0): Regularizes hip yaw joints toward zero to reduce unnecessary torso twisting. Helps maintain stable, natural-looking lower body motion.

- `--hip-pitch-cost-weight` (default: 0.0): Controls hip pitch joint regularization. Set to 0 by default to allow natural forward/backward hip motion.

- `--hip-roll-cost-weight` (default: 0.0): Controls hip roll joint regularization. Set to 0 by default to allow natural side-to-side hip motion.

**Advanced Parameters:**
- `--offset-factor` (default: 0.0): Vertical offset applied to robot root position. Can be used to adjust robot height relative to the ground plane.

</details>

### Visualization Commands

```bash
# Visualize retargeted robot motion with human motion and environment
python visualization/retargeting_visualization.py \
    --postprocessed-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1 \
    --robot-name g1 \
    --bg-pc-downsample-factor 4 \
    --confidence-threshold 0.0
```




