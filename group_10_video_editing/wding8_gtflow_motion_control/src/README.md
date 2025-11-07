# src/ - Source Code

Technical documentation for the Go-with-the-Flow pipeline implementation.

## Structure

```
src/
├── pipeline/          # Main pipeline scripts
│   ├── make_warped_noise.py                        # Standard flow-based noise warping
│   ├── make_warped_noise_with_segmentation.py      # Segmentation-scaled flow warping
│   ├── cut_and_drag_inference.py                   # CogVideoX video generation
│   └── cut_and_drag_gui.py                         # GUI for inference
│
└── tools/             # Utility scripts
    ├── extract_flow_sequence.py          # Standalone RAFT flow extraction
    ├── segment_and_scale_flow.py         # Standalone segmentation scaling
    ├── visualize_flow_with_vectors.py    # Flow visualization with arrows
    ├── create_three_way_comparison.py    # Side-by-side video comparison
    └── rewarp_noise_with_scaled_flow.py  # DEPRECATED: Post-warp scaling approach
```

## Technical Deep Dive: Segmentation-Based Flow Scaling

### The Critical Discovery: Motion is Encoded in Warped Noise

The key insight that enabled segmentation-based motion control:

**In latent diffusion models, motion is not stored in the flow field alone - it's encoded in the structure of the warped noise.**

#### Why This Matters

**Naive Approach (DOES NOT WORK)**:
```python
# 1. Extract flow and warp noise (standard pipeline)
flow = extract_flow(video)
noise = warp_noise(gaussian_noise, flow)

# 2. Scale flow after the fact
scaled_flow = scale_flow_with_segmentation(flow, masks, fg_scale, bg_scale)

# 3. Use old noise + new flow → NO EFFECT!
generate_video(noise, scaled_flow, prompt)  # Motion unchanged!
```

The diffusion model doesn't actually use the flow during generation - it uses the pre-warped noise structure. The flow is only used to CREATE the warped noise.

**Correct Approach (WORKS)**:
```python
# 1. Extract flow
flow = extract_flow(video)

# 2. Scale flow BEFORE warping
scaled_flow = scale_flow_with_segmentation(flow, masks, fg_scale, bg_scale)

# 3. Warp noise with scaled flow
noise = warp_noise(gaussian_noise, scaled_flow)

# 4. Generate video → Motion scaling works!
generate_video(noise, flow_unused, prompt)  # Uses warped noise structure
```

### Architecture Decision: Why Integrated Pipeline?

We considered two approaches:

#### Option 1: Post-Warp Re-scaling (DEPRECATED)
```python
# 1. Create warped noise (standard pipeline)
python make_warped_noise.py video.mp4 output/

# 2. Scale the flow
python segment_and_scale_flow.py \
    output/flows_dxdy.npy scaled_flow.npy \
    --mask sam2.mp4 --fg_scale 1.5 --bg_scale 1.0

# 3. Re-warp noise with scaled flow
python rewarp_noise_with_scaled_flow.py \
    output/ scaled_flow.npy
```

**Problem**: Re-warping uses a different random seed for the initial Gaussian noise, causing completely different results even with identical flow. The random noise structure is critical to the generation.

#### Option 2: Integrated Pipeline (CORRECT)
```python
# Single pass: scale flow then warp
python make_warped_noise_with_segmentation.py \
    video.mp4 output/ \
    --mask sam2.mp4 --fg_scale 1.5 --bg_scale 1.0
```

**Advantages**:
- Same random seed for noise generation → consistent results
- Flow scaling happens before warping → correct motion encoding
- Single pass → faster and cleaner

### Temporal Alignment: The Subtle But Critical Bug

#### The Problem

Initial implementation had different resampling strategies for video and SAM masks:

```python
# Video processing
video = load_video(path)  # 88 frames
video = resize_list(video, length=49)  # Resample to 49 frames
flow = extract_flow(video)  # 48 frames (49 → 48)

# SAM processing (WRONG!)
sam_masks = load_video(sam_path)  # 88 frames
sam_masks = resize_list(sam_masks, length=48)  # Directly to 48 frames
```

**Why this breaks**:
- Video resampling: 88 → 49 frames using indices `[0.0, 1.81, 3.63, ..., 87.0]`
- SAM resampling: 88 → 48 frames using indices `[0.0, 1.84, 3.67, ..., 87.0]`
- Different timelines → Mask[t] doesn't correspond to VideoFrame[t]
- Result: Scaling affects wrong regions (background gets scaled when foreground should be)

#### The Solution

Resample video and SAM identically, THEN extract flow:

```python
# Video processing
video = load_video(path)  # 88 frames
video = resize_list(video, length=49)  # 49 frames @ indices [0, 1.81, ..., 87]

# SAM processing (CORRECT!)
sam_masks = load_video(sam_path)  # 88 frames
sam_masks = resize_list(sam_masks, length=49)  # 49 frames @ SAME indices [0, 1.81, ..., 87]

# Flow extraction
flow = extract_flow(video)  # 48 frames (motion between frames 0→1, 1→2, ..., 47→48)

# Take first 48 SAM masks
sam_masks = sam_masks[:-1]  # Drop last mask → 48 masks
```

**Now the correspondence is correct**:
- `Flow[t]` = motion between `VideoFrame[t]` and `VideoFrame[t+1]`
- `Mask[t]` = segmentation for `VideoFrame[t]`
- Both use the same temporal sampling from the original video

### Flow Scaling Formula

The segmentation-based scaling applies per-pixel weights:

```python
# Convert SAM masks to foreground masks (white pixels = foreground)
fg_mask = (sam_grayscale >= threshold).astype(float)  # Shape: (T, H, W)
bg_mask = 1.0 - fg_mask

# Scale flow per-pixel
scaled_flow = flow * (fg_mask * fg_scale + bg_mask * bg_scale)
```

**Breakdown**:
- Foreground pixels: `flow * (1.0 * fg_scale + 0.0 * bg_scale) = flow * fg_scale`
- Background pixels: `flow * (0.0 * fg_scale + 1.0 * bg_scale) = flow * bg_scale`
- Edge pixels: Smooth blending due to mask interpolation during resizing

### Resolution and Dimension Handling

CogVideoX operates in latent space at 1/8 resolution:

```python
# Video dimensions
video_height, video_width = 480, 720  # Original video
flow_height, flow_width = 240, 360    # Optical flow (downsampled for speed)

# Latent dimensions (for noise)
LATENT_SCALE = 8
noise_height = video_height // LATENT_SCALE  # 480 // 8 = 60
noise_width = video_width // LATENT_SCALE     # 720 // 8 = 90
noise_shape = (49, 16, noise_height, noise_width)  # (T, C, H, W)
```

**Critical bug we fixed**: Initial implementation used flow resolution instead of video resolution for noise dimensions, causing incompatible shapes.

### Mask Processing Pipeline

```python
# 1. Load SAM video frame
sam_frame = load_frame(sam_video, idx)  # RGB, arbitrary size

# 2. Convert to grayscale
gray = cv2.cvtColor(sam_frame, cv2.COLOR_RGB2GRAY)

# 3. Threshold to binary mask (white = foreground)
fg_mask = (gray >= threshold).astype(float)  # 255 → 1.0, others → 0.0

# 4. Resize to match flow resolution
fg_mask_resized = cv2.resize(fg_mask, (flow_width, flow_height),
                              interpolation=cv2.INTER_LINEAR)

# 5. Re-binarize after interpolation
fg_mask_binary = (fg_mask_resized > 0.5).astype(float)
```

**Why re-binarize**: Linear interpolation during resizing creates intermediate values (0.0-1.0). We re-threshold to maintain clean binary masks while preserving smooth edges.

### NoiseWarper Integration

The pipeline uses `rp.git.CommonSource.noise_warp.NoiseWarper`:

```python
from rp.git.CommonSource.noise_warp import NoiseWarper

# Initialize with scaled flow
warper = NoiseWarper(
    flows=scaled_flow,  # (T, 2, H, W) - scaled by segmentation
    noise_shape=(T+1, C, noise_H, noise_W),
    device=device
)

# Warp Gaussian noise
gaussian_noise = torch.randn(noise_shape, device=device)
warped_noise = warper.warp(gaussian_noise)
```

**Key detail**: `noise_shape[0] = T+1` because noise needs an extra frame (49 frames of noise → 48 frames of flow).

### Visualization Normalization

For consistent visualization across different pipeline runs:

```python
# Convert flow (dx, dy) to RGB visualization
flow_rgb = flow_to_rgb(flow)  # Shape: (T, H, W, 3), range: [-∞, ∞]

# CRITICAL: Use fixed normalization formula (not adaptive)
flow_rgb_normalized = flow_rgb / 4.0 + 0.5  # Assumes most flow in [-2, 2] range
flow_rgb_clipped = np.clip(flow_rgb_normalized, 0, 1)
```

**Why fixed normalization**: Adaptive normalization (min-max per frame) makes different videos incomparable. Fixed formula maintains consistent magnitude representation across runs.

### File Structure Output

```
results/warped_noise/my_video_fg1.5_bg1.0/
├── video.mp4                # Preprocessed 49-frame video (480x720)
├── first_frame.png          # First frame reference
├── noises.npy              # Warped noise (49, 16, 60, 90)
├── flows_dxdy.npy          # Scaled flow used for warping (48, 2, 240, 360)
├── noise_vis.mp4           # Noise visualization (optional)
└── flows/                  # Debug outputs
    ├── original_flow.npy   # Original flow before scaling (48, 2, 240, 360)
    └── scaled_flow.npy     # Scaled flow (same as flows_dxdy.npy)
```

## Development Notes

### Adding New Motion Control Features

When extending the pipeline with new motion control methods:

1. **Always apply modifications BEFORE noise warping** - motion is encoded in warped noise structure
2. **Maintain temporal correspondence** - ensure all videos/masks use identical frame sampling
3. **Use fixed random seeds** - for reproducibility and comparison
4. **Validate with fg=1.0, bg=1.0** - should produce identical results to standard pipeline

### Testing Temporal Alignment

To verify temporal alignment is correct:

```python
# Generate with fg=1.0, bg=1.0 (should be identical to standard pipeline)
python make_warped_noise_with_segmentation.py video.mp4 output_seg/ \
    --mask sam2.mp4 --fg_scale 1.0 --bg_scale 1.0

python make_warped_noise.py video.mp4 output_std/

# Compare warped noise (should be nearly identical, allowing for numerical precision)
import numpy as np
noise_seg = np.load('output_seg/noises.npy')
noise_std = np.load('output_std/noises.npy')
print(f"Max difference: {np.abs(noise_seg - noise_std).max()}")  # Should be < 1e-5
```

### Common Pitfalls

1. **Resampling masks independently from video** → Temporal misalignment
2. **Scaling flow after warping** → No effect on generated motion
3. **Using adaptive normalization** → Inconsistent visualizations
4. **Wrong noise dimensions** → Shape mismatch errors
5. **Different random seeds** → Non-reproducible results

## Dependencies

### Conda Environments

**flow_warp** (optical flow & preprocessing):
```bash
conda activate flow_warp
# Uses: RAFT, OpenCV, NumPy, rp.git.CommonSource
# For: make_warped_noise*.py, tools/*
```

**flow** (video diffusion):
```bash
conda activate flow
# Uses: CogVideoX, transformers, diffusers
# For: cut_and_drag_inference.py, cut_and_drag_gui.py
```

### Import Conventions

```python
# CommonSource modules (legacy motion transfer code)
import rp
rp.git_import('CommonSource')
import rp.git.CommonSource.noise_warp as nw

# Standard imports
import numpy as np
import cv2
import torch
```

## Performance Characteristics

### make_warped_noise_with_segmentation.py
- **Time**: 1-3 minutes for 49 frames (GPU)
- **GPU Memory**: ~2-4GB (RAFT optical flow)
- **Output Size**: ~8-15MB (warped noise + flows + video)

### cut_and_drag_inference.py
- **Time**: 5-15 minutes (depends on steps and GPU)
- **GPU Memory**: ~8-16GB (CogVideoX-5B)
- **Output Size**: ~2-5MB (final video)

## References

For usage instructions and examples, see the main [README.md](../README.md).
