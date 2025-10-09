# Robust Data Handling for Multi-Human MegaHunter Optimization

This module provides robust data loading and validation functions to handle missing frames and data inconsistencies in multi-human scenarios.

## Features

### 1. Robust SMPL Data Loading (`get_smpl_init_data_robust`)
- Handles missing SMPL parameter files gracefully
- Detects and reports NaN values in SMPL parameters
- Fills missing frames using nearest neighbor interpolation
- Provides detailed logging of missing data

### 2. Robust Pose2D Data Loading (`get_pose2d_init_data_robust`)
- Handles missing pose2D JSON files
- Filters out low-confidence keypoints
- Reports frames with insufficient valid keypoints
- Supports per-person confidence thresholds

### 3. Multi-Human Data Validation (`validate_multi_human_data`)
- Validates data consistency across SMPL, pose2D, and bbox data
- Filters out persons with insufficient valid frames
- Configurable minimum valid frame ratio threshold
- Returns only persons with sufficient data for optimization

### 4. Person-Frame Mask Creation (`create_person_frame_mask`)
- Creates binary masks indicating which person appears in which frame
- Essential for multi-human optimization to handle missing persons in frames
- Compatible with JAX optimization pipeline

### 5. Missing Pose Interpolation (`interpolate_missing_poses`)
- Interpolates missing poses using linear or nearest neighbor methods
- Handles edge cases (missing at start/end of sequence)
- Supports different interpolation methods for different data types

## Usage

### In MegaHunter Optimization

The robust functions are automatically used when running with `--multihuman` flag:

```bash
python stage2_optimization/01_megahunter_optimization.py \
    --world-env-path ./path/to/world.h5 \
    --bbox-dir ./path/to/bbox/json_data \
    --pose2d-dir ./path/to/pose2d \
    --smpl-dir ./path/to/smpl \
    --out-dir ./output \
    --multihuman \
    --top-k 3
```

### Debugging NaN Issues

Use the debug script to check data consistency:

```bash
# Basic check
python debug_multihuman_nan.py \
    --bbox-dir ./path/to/bbox/json_data \
    --pose2d-dir ./path/to/pose2d \
    --smpl-dir ./path/to/smpl \
    --top-k 3

# Test robust loading
python debug_multihuman_nan.py \
    --bbox-dir ./path/to/bbox/json_data \
    --pose2d-dir ./path/to/pose2d \
    --smpl-dir ./path/to/smpl \
    --top-k 3 \
    --test-robust
```

## Key Benefits

1. **Prevents NaN Errors**: Detects and handles NaN values before they propagate to optimization
2. **Handles Missing Frames**: Automatically fills missing frames with interpolation
3. **Filters Invalid Data**: Removes persons with insufficient valid data
4. **Detailed Logging**: Provides clear information about data issues
5. **Backward Compatible**: Works seamlessly with existing single-person pipeline

## Implementation Details

### Missing Frame Handling
- Missing frames are detected when files don't exist or contain NaN values
- Interpolation uses nearest valid frame data
- Persons with >50% missing frames are filtered out by default

### NaN Detection
- Checks all SMPL parameters (betas, body_pose, global_orient, trans)
- Checks pose2D keypoints and confidences
- Reports exact location of NaN values for debugging

### Data Validation
- Ensures all required data (SMPL, pose2D, bbox) exists for each person-frame
- Creates consistent person-frame masks for optimization
- Validates data shapes and types before optimization

## Troubleshooting

### Common Issues and Solutions

1. **"No valid SMPL data found for person X"**
   - Check if VIMO successfully processed this person
   - Verify person ID consistency across pipeline stages
   - Consider lowering `min_valid_frames_ratio`

2. **"NaN found in frame X, person Y, param Z"**
   - Re-run VIMO for affected frames
   - Check if person is partially occluded in those frames
   - Enable interpolation with `fill_missing=True`

3. **"Only X out of Y persons have sufficient valid data"**
   - Normal behavior when some persons appear briefly
   - Adjust `--top-k` to process fewer persons
   - Check SAM2 detection quality

## Parameters

### Key Parameters for Robust Handling
- `min_confidence`: Minimum confidence for pose2D keypoints (default: 0.3)
- `min_valid_frames_ratio`: Minimum ratio of valid frames per person (default: 0.5)
- `fill_missing`: Whether to interpolate missing frames (default: True)
- `frame_missing_thr`: Maximum consecutive missing frames to interpolate (default: 3)

### Optimization Parameters for Multi-Human
- `--multihuman`: Enable multi-human processing
- `--top-k`: Number of largest humans to process
- `--joint2d-conf-threshold`: Confidence threshold for 2D joints
- `--joint3d-conf-threshold`: Confidence threshold for 3D joints
- `--frame-missing-thr`: Missing frame interpolation threshold