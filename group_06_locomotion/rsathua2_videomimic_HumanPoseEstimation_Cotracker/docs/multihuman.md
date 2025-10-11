# Multi-Human Support Guide

The VideoMimic Real-to-Sim pipeline supports processing multiple humans in a scene. By default, it processes only the largest person (by average bounding box area over time). This guide explains how to enable and use multi-human processing.

## Overview

Multi-human support allows the pipeline to track and reconstruct multiple people simultaneously. Each person is assigned a unique ID that persists throughout the pipeline, and visualization tools display each person with a different color.

## Single Video Processing

### Stage 0: Preprocessing with Multi-Human Support

```bash
# Human segmentation (SAM2) - automatically detects all humans
python stage0_preprocessing/sam2_segmentation.py \
    --video-dir ./demo_data/input_images/my_video/cam01 \
    --output-dir ./demo_data/input_masks/my_video/cam01 \
    --vis

# 2D pose estimation (ViTPose) - process top 3 largest humans
python stage0_preprocessing/vitpose_2d_poses.py \
    --video-dir ./demo_data/input_images/my_video/cam01 \
    --bbox-dir ./demo_data/input_masks/my_video/cam01/json_data \
    --output-dir ./demo_data/input_2d_poses/my_video/cam01 \
    --multihuman \
    --top-k 3 \
    --vis

# 3D mesh estimation (VIMO) - process top 3 largest humans
python stage0_preprocessing/vimo_3d_mesh.py \
    --img-dir ./demo_data/input_images/my_video/cam01 \
    --mask-dir ./demo_data/input_masks/my_video/cam01 \
    --out-dir ./demo_data/input_3d_meshes/my_video/cam01 \
    --multihuman \
    --top-k 3
```

### Stage 2: Optimization with Multi-Human Support

```bash
python stage2_optimization/megahunter_optimization.py \
    --world-env-path ./demo_data/input_megasam/megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --bbox-dir ./demo_data/input_masks/my_video/cam01/json_data \
    --pose2d-dir ./demo_data/input_2d_poses/my_video/cam01 \
    --smpl-dir ./demo_data/input_3d_meshes/my_video/cam01 \
    --out-dir ./demo_data/output_smpl_and_points \
    --multihuman \
    --top-k 3
```

### Stage 3: Postprocessing with Multi-Human Support

```bash
# Gravity calibration and mesh generation
python stage3_postprocessing/postprocessing_pipeline.py \
    --megahunter-path ./demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --out-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1 \
    --multihuman \
    --is-megasam
```

### Stage 4: Robot Retargeting with Multi-Human Support

```bash
python stage4_retargeting/robot_motion_retargeting.py \
    --src-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1 \
    --contact-dir ./demo_data/input_contacts/my_video/cam01 \
    --multihuman \
    --top-k 3 \
    --vis
```



### Visualization

The visualization automatically displays all humans (or top-k humans) with different colors:

```bash
python visualization/optimization_results_visualization.py \
    --world-env-path ./demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --bg-pc-downsample-factor 4
```

## Sequential Processing for Multiple Videos

### Sequential Preprocessing

```bash
# Sequential SAM2 segmentation - loads model once, processes videos one by one
python sequential_processing/stage0_sequential_sam2_segmentation.py \
    --pattern <pattern> \
    --video-base-dir ./demo_data/input_images \
    --output-base-dir ./demo_data/input_masks

# Sequential ViTPose with multi-human support
python sequential_processing/stage0_sequential_vitpose_2d_poses.py \
    --pattern <pattern> \
    --multihuman \
    --top-k 3

# Sequential VIMO with multi-human support
python sequential_processing/stage0_sequential_vimo_3d_mesh.py \
    --pattern <pattern> \
    --multihuman \
    --top-k 3
```

### Sequential Optimization

```bash
# JAX compiles once, then processes all videos sequentially
python sequential_processing/stage2_sequential_megahunter_optimization.py \
    --pattern <pattern> \
    --multihuman \
    --top-k 3
```

### Sequential Postprocessing

```bash
python sequential_processing/stage3_sequential_mesh_generation_and_geocalib.py \
    --pattern <pattern> \
    --hunter-base-dir ./demo_data/output_smpl_and_points \
    --output-base-dir ./demo_data/output_calib_mesh \
    --multihuman \
    --is-megasam
```

**Note:** Sequential processing is ideal for building datasets efficiently. Models are loaded once and reused across all videos, dramatically reducing overall processing time. See [sequential_processing.md](./sequential_processing.md) for details.

## Key Parameters

- **`--multihuman`**: Enable multi-human processing mode
- **`--top-k <N>`**: Process only the N largest humans (by average bounding box area)
  - Set to 0 to process all detected humans
  - Recommended values: 2-5 depending on scene and available memory

## How It Works

1. **Human Detection (SAM2)**
   - Detects all humans in each frame
   - Calculates bounding box area for each person in each frame
   - Computes average area across all frames for each person
   - Saves area rankings in `meta_data.json`

2. **Area-Based Ranking**
   - Humans are sorted by average bounding box area (largest first)
   - This ranking is saved in SAM2 metadata as `sorted_by_avg_area`
   - Per-frame rankings are also saved for frame-specific selection

3. **Consistent Person Selection**
   - ViTPose, VIMO, and MegaHunter use SAM2's area ranking
   - Top-k humans are selected based on this ranking
   - Person IDs remain consistent throughout the pipeline

4. **Multi-Human Optimization**
   - MegaHunter optimizes all selected humans simultaneously
   - Each person's motion is optimized in the shared world coordinate system
   - Interactions between people are preserved

5. **Visualization**
   - Each person is rendered with a different color from a predefined palette
   - Colors are consistent across all visualization tools
   - Contact areas (e.g., feet on ground) are highlighted in green

## Technical Details

### SAM2 Metadata Structure

The `meta_data.json` file contains:
```json
{
  "all_instance_ids": [1, 2, 3],  // All detected person IDs
  "sorted_by_avg_area": [2, 1, 3],  // IDs sorted by average area (largest first)
  "avg_areas": {  // Average bounding box area per person
    "1": 15234.5,
    "2": 18956.2,
    "3": 12456.8
  },
  "frame_counts": {  // Number of frames each person appears in
    "1": 95,
    "2": 100,
    "3": 87
  }
}
```

### Per-Frame Area Rankings

Each frame's JSON file includes:
```json
{
  "area_ranking": [2, 1, 3],  // Person IDs sorted by area in this frame
  "areas": {  // Bounding box areas in this frame
    "1": 14500,
    "2": 19200,
    "3": 12100
  }
}
```

## Performance Considerations

### Memory Usage
- GPU memory usage scales approximately linearly with the number of humans
- For MegaSam reconstruction: ~40GB VRAM for single person, ~50GB for 3 people
- For MegaHunter optimization: ~12GB for single person, ~20GB for 3 people

### Processing Time
- Preprocessing stages (ViTPose, VIMO) process each person independently
- Optimization time increases with more humans due to interaction constraints
- Visualization rendering time increases moderately with more humans

### Recommendations
- Start with `--top-k 2` for two-person interactions
- Use `--top-k 3-5` for group scenes with sufficient GPU memory
- Monitor GPU memory usage and adjust accordingly
- Consider using Align3r instead of MegaSam for lower memory usage

## Stage 4: Robot Retargeting with Multi-Human Support

```bash
# Retarget motion for multiple humans
python stage4_retargeting/robot_motion_retargeting.py \
    --src-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1 \
    --contact-dir ./demo_data/input_contacts/my_video/cam01 \
    --multihuman \
    --top-k 3 \
    --vis
```

When multi-human mode is enabled:
- Each person's motion is retargeted independently to the robot
- Output files are saved as `retarget_poses_g1_person_<id>.h5` for each person
- Visualization shows the first person by default
- All optimization parameters apply to each person's retargeting

## Limitations and Notes

1. **Ego-View Rendering**: Only uses the first person's head pose for ego-view visualization
2. **Camera Following**: The camera follow feature tracks the first person by default
3. **Robot Retargeting**: Supports multi-human retargeting with `--multihuman` flag
4. **Person Tracking**: Relies on consistent detection across frames; may fail with heavy occlusions
5. **Memory Scaling**: Large numbers of people may exceed GPU memory limits

## Robust Data Handling

The multi-human pipeline includes robust data handling to prevent NaN errors and handle missing frames:

### Automatic Features
- **Missing Frame Detection**: Identifies and reports missing data for each person
- **NaN Prevention**: Detects and handles NaN values before optimization
- **Frame Interpolation**: Fills missing frames using nearest neighbor interpolation
- **Person Validation**: Filters out persons with insufficient valid data

### Debug Tools
```bash
# Check data consistency and identify issues
python debug_multihuman_nan.py \
    --bbox-dir ./demo_data/input_masks/my_video/cam01/json_data \
    --pose2d-dir ./demo_data/input_2d_poses/my_video/cam01 \
    --smpl-dir ./demo_data/input_3d_meshes/my_video/cam01 \
    --top-k 3 \
    --test-robust  # Test robust loading functions
```

For detailed information about robust handling, see [stage2_optimization/README_robust_handling.md](../stage2_optimization/README_robust_handling.md).

## Troubleshooting

### Common Issues

1. **"CUDA out of memory" errors**
   - Reduce `--top-k` value
   - Use Align3r instead of MegaSam
   - Increase `--bg-pc-downsample-factor` in visualization

2. **Inconsistent person IDs across frames**
   - Check SAM2 detection quality with `--vis` flag
   - Adjust SAM2 detection thresholds if needed
   - Ensure good lighting and minimal occlusions

3. **Missing persons in later stages**
   - Verify person appears in enough frames (check `frame_counts` in metadata)
   - Check if person's average area is too small
   - Ensure `--multihuman` flag is used consistently

4. **NaN errors in optimization**
   - Run `debug_multihuman_nan.py` to identify issues
   - The robust loading (enabled with `--multihuman`) automatically handles most cases
   - Consider adjusting confidence thresholds or `--frame-missing-thr`

## Examples

### Two-Person Interaction
```bash
# Process a video with two people interacting
./process_video.sh two_person_video 0 200 1 g1 0
# Then add multi-human flags to individual stages:
python stage0_preprocessing/vitpose_2d_poses.py ... --multihuman --top-k 2
python stage0_preprocessing/vimo_3d_mesh.py ... --multihuman --top-k 2
python stage2_optimization/megahunter_optimization.py ... --multihuman --top-k 2
```

### Group Scene (3-5 people)
```bash
# For group scenes, process with higher top-k
python stage0_preprocessing/vitpose_2d_poses.py ... --multihuman --top-k 5
# But maybe optimize fewer for memory constraints
python stage2_optimization/megahunter_optimization.py ... --multihuman --top-k 3
```

### All Detected Humans
```bash
# Process all detected humans (use with caution - high memory usage)
python stage0_preprocessing/vitpose_2d_poses.py ... --multihuman --top-k 0
```