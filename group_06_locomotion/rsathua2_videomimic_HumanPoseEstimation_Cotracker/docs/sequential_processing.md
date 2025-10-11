# Sequential Multi-Video Processing Documentation

This document provides comprehensive guidance for processing multiple videos sequentially through the VideoMimic Real-to-Sim pipeline.

## Overview

**Important:** This is NOT batch/parallel processing. The scripts process videos **sequentially** (one after another) to:
- **Amortize model loading time**: Models and checkpoints are loaded once and reused
- **Share JAX compilation**: JAX compiles optimization code once, making subsequent videos much faster
- **Maintain stable memory usage**: Sequential processing prevents GPU memory overflow
- **Build datasets efficiently**: Ideal for creating training datasets for policy learning

**Key Motivation:** When building datasets for training policies, the overhead of loading models and compiling JAX code for each video individually becomes prohibitive. Sequential processing reduces a 5 to 10 minutes-per-video overhead to just the first video, dramatically speeding up dataset creation.

## Processing Pipeline

The sequential processing system follows the same stages as individual video processing:
- **Stage 0**: Human preprocessing (SAM2, ViTPose, VIMO, BSTRO)
- **Stage 1**: World reconstruction (MegaSam or Align3r)
- **Stage 2**: MegaHunter optimization (now with sequential support!)
- **Stage 3**: Postprocessing (GeoCalib, meshification)
- **Stage 4**: Retargeting (now with sequential support!)

## Directory Structure

```
sequential_processing/
├── stage0_sequential_sam2_segmentation.py      # Sequential SAM2 human segmentation
├── stage0_sequential_vitpose_2d_poses.py       # Sequential ViTPose 2D pose estimation
├── stage0_sequential_vimo_3d_mesh.py           # Sequential VIMO 3D mesh estimation
├── stage0_sequential_bstro_contact_detection.py # Sequential BSTRO contact detection
├── stage1_sequential_megasam_reconstruction.py  # Sequential MegaSam reconstruction
├── stage1_sequential_monst3r_depth_prior_reconstruction.py  # Sequential Align3r reconstruction
├── stage2_sequential_megahunter_optimization.py # Sequential MegaHunter optimization
├── stage3_sequential_mesh_generation_and_geocalib.py # Sequential postprocessing
└── stage4_sequential_robot_motion_retargeting.py # Sequential robot retargeting
```

## Prerequisites

### Environment Setup
Ensure you have the correct conda environments:
- **vm1rs**: For most operations (CUDA 12.4+)
- **vm1recon**: For MegaSam operations (CUDA 11.8)

### Data Structure
Your input data should follow this structure:
```
demo_data/
├── input_images/
│   ├── video_name_1/
│   │   └── cam01/
│   │       ├── 00001.jpg
│   │       ├── 00002.jpg
│   │       └── ...
│   └── video_name_2/
│       └── cam01/
│           └── ...
```

## Stage 0: Human Preprocessing (Sequential)

**What happens:**
- (SAM2/ViTPose/VIMO/BSTRO) models load once at startup
- Each video is processed sequentially
- Model weights stay in GPU memory throughout
- Total time per each substage: ~1 min overhead + 30 sec/video (vs 1.5 min/video if run individually)
- The estimate time is just a rough estimate for guidance. It depends on each cpu/gpu configuration and the number of frames being processed, the current cluster's resource availability, etc.

**Key Parameters:**
- `--pattern`: Filter videos by name pattern (e.g., "jump", "walk")
- `--text`: Detection prompt (default: "person.")
- `--vis`: Enable visualization output
- `--start-idx`/`--end-idx`: Process subset of videos

**Dependencies:** ViTPose/VIMO/BSTRO require SAM2 outputs for bounding boxes.

### SAM2 Human Segmentation

```bash
# Example command
conda activate vm1rs
python sequential_processing/stage0_sequential_sam2_segmentation.py \
    --video-base-dir ./demo_data/input_images \
    --output-base-dir ./demo_data/input_masks \
    --pattern "jump" \
    --start-idx 0 \
    --end-idx 10
```

### ViTPose 2D Pose Estimation

> This stage is not batchified, meaning one image is processed at a time. If users want faster speed, they can implement their own batching logic.

```bash
# Example command
conda activate vm1rs
python sequential_processing/stage0_sequential_vitpose_2d_poses.py \
    --video-base-dir ./demo_data/input_images \
    --output-base-dir ./demo_data/input_2d_poses \
    --bbox-base-dir ./demo_data/input_masks \
    --pattern "jump" \
    --vis # Don't visualize if you want faster speed
```

### VIMO 3D Human Mesh Estimation

> This stage is not batchified, meaning one image is processed at a time. If users want faster speed, they can implement their own batching logic.

```bash
# Example command
conda activate vm1rs
python sequential_processing/stage0_sequential_vimo_3d_mesh.py \
    --video-base-dir ./demo_data/input_images \
    --output-base-dir ./demo_data/input_3d_meshes \
    --bbox-base-dir ./demo_data/input_masks \
    --pattern "jump"
```

### BSTRO Contact Detection

```bash
# Example command
conda activate vm1rs
python sequential_processing/stage0_sequential_bstro_contact_detection.py \
    --video-base-dir ./demo_data/input_images \
    --output-base-dir ./demo_data/input_contacts \
    --bbox-base-dir ./demo_data/input_masks \
    --pattern "jump" \
    --batch-size 16
```

## Stage 1: World Reconstruction (Sequential)

**Performance gains:**
- Model loading: 1 minute (happens once)
- Per video: ~3-4 minutes
- Total for 10 videos with 150 frames per video: ~30 minutes (vs ~40 minutes if run individually)
- The estimate time is just a rough estimate for guidance. It depends on each cpu/gpu configuration and the number of frames being processed, the current cluster's resource availability, etc.

### Option A: MegaSam Reconstruction (Higher Accuracy + More Memory Efficient)

```bash
conda activate vm1recon
python sequential_processing/stage1_sequential_megasam_reconstruction.py \
    --video-base-dir ./demo_data/input_images \
    --outdir ./demo_data/input_megasam \
    --pattern "jump" \
    --gsam2 \
    --start-frame 0 \
    --end-frame -1 \
    --stride 1
```

### Option B: Align3r Reconstruction (Good for texture-less videos)

```bash
conda activate vm1rs
python sequential_processing/stage1_sequential_monst3r_depth_prior_reconstruction.py \
    --video-base-dir ./demo_data/input_images \
    --outdir ./demo_data/input_align3r \
    --pattern "jump"
```

## Stage 2: MegaHunter Optimization (Sequential)

**Performance gains:**
- JAX compilation: 1-2 minutes (happens only for first video!)
- Subsequent videos: Process immediately without compilation; 200ms per video if the video length has been padded to 100 frames and compiled once.
- Total for 10 videos: ~3 minutes (vs ~20 minutes if run individually)

**Key Parameters:**
- `--use-g1-shape`: Use SMPL shape fitted to G1 robot, which effectively scales up the world reconstruction results to the size of the robot, instead of the human body size.
- `--multihuman`: Enable multi-human processing
- `--top-k`: Number of humans to process
- `--skip-existing`: Skip already processed videos

```bash
conda activate vm1rs
python sequential_processing/stage2_sequential_megahunter_optimization.py \
    --pattern "jump" \
    --world-reconstruction-base-dir ./demo_data/input_megasam \
    --bbox-base-dir ./demo_data/input_masks \
    --pose2d-base-dir ./demo_data/input_2d_poses \
    --smpl-base-dir ./demo_data/input_3d_meshes \
    --out-base-dir ./demo_data/output_smpl_and_points \
    --use-g1-shape
```

## Stage 3: Postprocessing

**Key Parameters:**
- `--is-megasam`: Set to `true` for MegaSam inputs, `false` for Align3r
- `--gradient-thr`: Point filtering threshold (0.01 aggresive, 0.15 moderate, 0.3 conservative)
- `--meshification-method`: `nksr` by default
- `--no-spf`: Skip spatiotemporal filtering

```bash
conda activate vm1recon  # Required for GeoCalib and meshification
python sequential_processing/stage3_sequential_mesh_generation_and_geocalib.py \
    --pattern "jump" \
    --megahunter-base-dir ./demo_data/output_smpl_and_points \
    --output-base-dir ./demo_data/output_calib_mesh \
    --is-megasam \
    --gradient-thr 0.15 \
    --meshification-method nksr
```

## Stage 4: Retargeting 

**There will be huge performance gains, since the jit time is huge (~x20 more than the optimization time) and will be amortized across all videos.**

**Key Parameters:**
- `--skip-existing`: Skip already processed videos
- `--multihuman`: Enable multi-human processing
- `--top-k`: Number of humans to process

```bash
conda activate vm1rs
python sequential_processing/stage4_sequential_robot_motion_retargeting.py \
    --pattern "jump" \
    --postprocessed-base-dir ./demo_data/output_calib_mesh \
    --contact-base-dir ./demo_data/input_contacts \
    --skip-existing
```

## Tips for Efficient Dataset Building

1. **Process in batches by video length**: Group similar length videos for better jit time amortization. For example, 87 frames video and 35 frames video will be padded to 100 frames and will reuse the same JAX compilation for MegaHunter and Retargeting.
2. **Use pattern matching**: Process related videos together to maintain context. For example, if you want to process all jumping videos, you can use `--pattern "jump"` to process all jumping videos together.
3. **Skip existing outputs**: Use `--skip-existing` flags where available
4. **Start with smaller sets**: Test your pipeline on 2-3 videos before processing hundreds

## Common Issues and Solutions

### JAX Compilation Takes Too Long for MegaHunter and Retargeting
- **Solution**: This is normal for the first video. Subsequent videos will be much faster.
- **Tip**: Process videos in large batches to maximize the benefit.

### GPU Memory Errors
- **Solution**: Reduce per-video complexity (fewer frames, fewer humans)

### Missing Dependencies Between Stages
- **Solution**: Ensure each stage completes successfully before moving to the next
- **Tip**: Check output directories for expected files before proceeding

## Output Data Management

### File Naming Conventions
Sequential outputs follow the same pattern as individual processing:
- Pattern: `<method>_<video_name>_cam01_frame_<start>_<end>_subsample_<factor>`
- Example: `megahunter_megasam_reconstruction_results_jump_01_cam01_frame_0_100_subsample_1.h5`

### Storage Considerations
- Sequential processing generates the same outputs as individual processing
- Plan for ~500MB-1.5GB per video for all intermediate files (depends on the number of frames, humans, etc.)
- Final outputs in `output_calib_mesh/` are typically ~500-1GB per video (depends on the number of frames, humans, etc.)

## Visualization

### Sequential Visualization with Dropdown Selection

> If you’re reading this, you’re probably serious about building a dataset. This visualization is designed to help you quickly filter out bad outputs without launching Viser for each video. However, the default setup supports all data modalities, which makes it heavy and slow when switching between videos. I (Hongsuk) recommend modifying the code (assuming you’re familiar with Viser) to load only the data modalities you’re interested in — for example, just the mesh or just the robot — so that it supports only the necessary features for your dataset building.

After processing multiple videos, use the sequential visualization tool to review all results:

```bash
python visualization/sequential_visualization.py \
    --results-root-dir ./demo_data/output_calib_mesh \
    --pattern "jump" \
    --bg-pc-downsample-factor 2 \
```

This visualization tool is specifically designed for reviewing datasets built with sequential processing:
- **Dropdown menu**: Switch between different results without restarting
- **Ego-view rendering**: See the scene from the human's perspective
- **Multi-person support**: Automatically handles multi-person scenes
- **Interactive filtering**: Adjust point cloud filters in real-time
- **Organized controls**: GUI controls grouped into logical folders

**Key Parameters:**
- `--results-root-dir`: Directory containing all processed results
- `--pattern`: Filter results by name pattern
- `--bg-pc-downsample-factor`: Point cloud downsampling (higher = faster)
- `--robot-name`: Robot model to visualize (default: g1)
- `--start-idx`/`--end-idx`: Load only a subset of results
- `--confidence-threshold`: Initial confidence threshold for filtering
- `--gradient-threshold`: Initial gradient threshold for filtering