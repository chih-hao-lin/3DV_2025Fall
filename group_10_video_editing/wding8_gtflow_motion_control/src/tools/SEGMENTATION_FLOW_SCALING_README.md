# Segmentation-Based Flow Scaling

Scale different regions of optical flow based on SAM2 video segmentation masks.

## Overview

This tool allows you to apply different motion scales to foreground vs background regions, enabling precise spatial control over motion transfer.

**Key Use Cases:**
- Emphasize subject motion while dampening background
- Remove background camera shake while keeping subject motion
- Amplify subtle foreground motion
- Create dynamic motion effects with spatial control

## Quick Start

```bash
conda activate flow_warp

# Basic usage: Emphasize foreground, dampen background
python src/tools/segment_and_scale_flow.py \
    --flow results/warped_noise/train/flows_dxdy.npy \
    --mask data/videos_sam/train_sam2.mp4 \
    --fg_scale 1.5 \
    --bg_scale 0.3 \
    --output results/fft_analysis/train_scaled/scaled_flow.npy

# With visualization
python src/tools/segment_and_scale_flow.py \
    --flow results/warped_noise/train/flows_dxdy.npy \
    --mask data/videos_sam/train_sam2.mp4 \
    --fg_scale 1.5 \
    --bg_scale 0.3 \
    --output results/fft_analysis/train_scaled/scaled_flow.npy \
    --visualize results/visualizations/train_segmented_flow.mp4
```

## SAM2 Mask Format

**Expected format:**
- Video file with segmentation visualization
- **White pixels** (RGB 255,255,255 or grayscale ≥240) = **Foreground** (object of interest)
- **Any other color** = **Background**

**How to get SAM2 masks:**
1. Use SAM2 model to segment your video
2. Export visualization with foreground in white
3. Save as video file (mp4, avi, etc.)

## Parameters

### Required

- `--flow PATH` - Optical flow .npy file (T, 2, H, W)
- `--mask PATH` - SAM2 mask video (white = foreground)
- `--fg_scale FLOAT` - Scale factor for foreground motion
- `--bg_scale FLOAT` - Scale factor for background motion
- `--output PATH` - Output path for scaled flow

### Optional

- `--visualize PATH` - Save visualization video showing segmentation and scaling effects
- `--threshold INT` - Brightness threshold for white detection (default: 240)

## Scaling Strategy

The tool applies per-pixel scaling based on segmentation:

```
scaled_flow = flow * (fg_mask * fg_scale + bg_mask * bg_scale)
```

Where:
- `fg_mask`: 1 where pixel is foreground, 0 otherwise
- `bg_mask`: 1 where pixel is background, 0 otherwise
- Each pixel gets exactly one of the two scales applied

## Usage Examples

### Example 1: Emphasize Subject, Dampen Background

**Goal:** Make subject motion 50% stronger, reduce background to 30%

```bash
python src/tools/segment_and_scale_flow.py \
    --flow results/warped_noise/dance/flows_dxdy.npy \
    --mask data/videos_sam/dance_sam2.mp4 \
    --fg_scale 1.5 \
    --bg_scale 0.3 \
    --output results/fft_analysis/dance_scaled/scaled_flow.npy
```

Result:
- Dancer motion amplified by 50%
- Background motion reduced to 30%
- Creates focus on subject

### Example 2: Remove Background Motion Completely

**Goal:** Keep only foreground motion, freeze background

```bash
python src/tools/segment_and_scale_flow.py \
    --flow results/warped_noise/person/flows_dxdy.npy \
    --mask data/videos_sam/person_sam2.mp4 \
    --fg_scale 1.0 \
    --bg_scale 0.0 \
    --output results/fft_analysis/person_fg_only/scaled_flow.npy
```

Result:
- Foreground motion unchanged
- Background motion completely removed
- Like green-screen effect for motion

### Example 3: Amplify Subtle Foreground Motion

**Goal:** Make subtle subject motion more visible

```bash
python src/tools/segment_and_scale_flow.py \
    --flow results/warped_noise/subtle/flows_dxdy.npy \
    --mask data/videos_sam/subtle_sam2.mp4 \
    --fg_scale 3.0 \
    --bg_scale 1.0 \
    --output results/fft_analysis/subtle_amplified/scaled_flow.npy
```

Result:
- Foreground motion tripled
- Background motion unchanged
- Highlights subtle movements

### Example 4: Reverse Effect - Dampen Foreground

**Goal:** Keep background motion, reduce subject motion

```bash
python src/tools/segment_and_scale_flow.py \
    --flow results/warped_noise/video/flows_dxdy.npy \
    --mask data/videos_sam/video_sam2.mp4 \
    --fg_scale 0.2 \
    --bg_scale 1.0 \
    --output results/fft_analysis/video_bg_emphasis/scaled_flow.npy
```

Result:
- Foreground motion reduced to 20%
- Background motion unchanged
- Unusual but creative effect

## Complete Workflow

### Step 1: Extract Original Flow

```bash
python src/pipeline/make_warped_noise.py \
    data/videos/train.mp4 \
    results/warped_noise/train/
```

Creates: `results/warped_noise/train/flows_dxdy.npy`

### Step 2: Get SAM2 Segmentation

Use SAM2 to segment your video and save mask as video with white foreground.

Place in: `data/videos_sam/train_sam2.mp4`

### Step 3: Scale Flow by Segmentation

```bash
python src/tools/segment_and_scale_flow.py \
    --flow results/warped_noise/train/flows_dxdy.npy \
    --mask data/videos_sam/train_sam2.mp4 \
    --fg_scale 1.5 \
    --bg_scale 0.3 \
    --output results/fft_analysis/train_scaled/scaled_flow.npy \
    --visualize results/visualizations/train_scaled.mp4
```

### Step 4: Integrate into Pipeline

```bash
# Copy warped noise structure
cp -r results/warped_noise/train/ results/warped_noise/train_scaled/

# Replace flow with scaled version
cp results/fft_analysis/train_scaled/scaled_flow.npy \
   results/warped_noise/train_scaled/flows_dxdy.npy
```

### Step 5: Generate Video

```bash
python src/pipeline/cut_and_drag_inference.py \
    --warped_noise_dir results/warped_noise/train_scaled/ \
    --output_path results/generated/train_scaled.mp4 \
    --prompt "your prompt here"
```

## Visualization Output

When using `--visualize`, creates a 2x2 grid showing:

**Top-left:** Original flow (HSV color-coded)
- Shows motion before scaling

**Top-right:** Scaled flow (HSV color-coded)
- Shows motion after scaling
- Compare to see effects

**Bottom-left:** Foreground mask
- White = foreground (will be scaled by fg_scale)
- Black = background (will be scaled by bg_scale)

**Bottom-right:** Change visualization
- Red = motion reduced
- Green = motion increased
- Gray = unchanged

## Tips & Best Practices

### Choosing Scale Values

**For natural results:**
- Keep scales between 0.5-2.0
- Avoid extreme values (>3.0 or <0.1) unless intentional

**Common combinations:**
```
# Subtle emphasis: 1.2x foreground, 0.8x background
--fg_scale 1.2 --bg_scale 0.8

# Strong emphasis: 2.0x foreground, 0.3x background
--fg_scale 2.0 --bg_scale 0.3

# Clean foreground: 1.0x foreground, 0.0x background
--fg_scale 1.0 --bg_scale 0.0
```

### SAM2 Mask Quality

**Good masks:**
- Clean separation between foreground and background
- Consistent segmentation across frames
- White foreground clearly visible

**If masks are poor:**
- Adjust `--threshold` parameter (try 200-250)
- Re-run SAM2 with better prompts
- Use temporal smoothing in SAM2

### Frame Count Mismatch

If flow and mask have different frame counts:
- Tool automatically uses minimum frame count
- Warning will be printed
- Truncates longer sequence to match shorter

**To fix:**
- Ensure source video matches mask video
- Re-extract flow if needed
- Check video frame rates match

### Visualization for Debugging

Always use `--visualize` the first time to verify:
- Mask is correctly loaded (bottom-left panel)
- Segmentation matches your intent
- Scaling effects look correct (bottom-right panel)

## Technical Details

### Flow Format

Expects optical flow in numpy format:
- Shape: `(T, 2, H, W)`
- T = number of frames
- 2 = flow channels (horizontal, vertical)
- H, W = spatial dimensions

### Mask Processing

1. Load mask video frame-by-frame
2. Convert to grayscale
3. Threshold to binary (≥240 = foreground)
4. Resize to match flow resolution
5. Apply per-frame scaling

### Resolution Handling

Masks are automatically resized to match flow:
- Uses bilinear interpolation
- Re-binarizes after resize to avoid artifacts
- Preserves segmentation accuracy

## Integration with Other Tools

### Compare with FFT Filtering

**Segmentation scaling:**
- ✓ Spatial control (foreground vs background)
- ✓ Semantic understanding
- ✗ No frequency control

**FFT filtering:**
- ✗ No spatial control
- ✓ Temporal frequency control (slow vs fast)
- ✗ No semantic understanding

**Can combine both:**
1. Apply FFT filtering first (remove shake)
2. Then apply segmentation scaling (emphasize foreground)

### Visualize Results

Use existing visualization tools:

```bash
# Visualize scaled flow
python src/tools/visualize_flow_with_vectors.py \
    results/fft_analysis/train_scaled/scaled_flow.npy \
    results/visualizations/train_scaled_flow.mp4 \
    --arrow_spacing 20
```

## Troubleshooting

### "Could not open mask video"
- Check mask file path is correct
- Verify file is valid video format
- Try absolute path instead of relative

### "Frame count mismatch"
- Normal if flow and mask have different lengths
- Tool uses minimum frame count
- To fix: ensure source videos match

### "Foreground pixels: 0.0%"
- Mask might not have white pixels
- Try adjusting `--threshold` (lower value)
- Check mask visualization in video player

### Scaling has no visible effect
- Check if fg_scale and bg_scale are different
- Verify mask is loading correctly (use `--visualize`)
- Original flow might be very low magnitude

## Example Results

### Train Video (Example from Testing)

**Original flow:**
- Overall magnitude: 2.428
- Foreground: 1.067
- Background: 2.559

**After scaling (fg=1.5, bg=0.3):**
- Overall magnitude: 0.841 (-65.4%)
- Foreground: 1.602 (+50%)
- Background: 0.768 (-70%)

**Interpretation:**
- Train (foreground) motion amplified as requested
- Background motion heavily dampened
- Overall motion reduced because background was most of image
- Clean separation between subject and background motion

## See Also

- `visualize_flow_with_vectors.py` - Visualize optical flow
- `frequency_motion_transfer.py` - FFT-based frequency filtering
- Archive: `archive/spatiotemporal_degradation/` - Old per-pixel control system
