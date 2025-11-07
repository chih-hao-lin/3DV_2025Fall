# Optical Flow Visualization with Vectors

Visualize optical flow with HSV color coding + directional arrows.

## Quick Start

```bash
conda activate flow_warp

# Basic visualization
python src/tools/visualize_flow_with_vectors.py \
    results/warped_noise/my_video/flows_dxdy.npy

# Custom output path
python src/tools/visualize_flow_with_vectors.py \
    results/warped_noise/my_video/flows_dxdy.npy \
    results/visualizations/flow_viz.mp4

# Adjust arrow density and scale
python src/tools/visualize_flow_with_vectors.py \
    results/warped_noise/my_video/flows_dxdy.npy \
    --arrow_spacing 20 \
    --arrow_scale 2.0

# Save individual frames instead of video
python src/tools/visualize_flow_with_vectors.py \
    results/warped_noise/my_video/flows_dxdy.npy \
    output_dir/ \
    --save_frames

# Create color wheel legend
python src/tools/visualize_flow_with_vectors.py \
    results/warped_noise/my_video/flows_dxdy.npy \
    --create_legend
```

## Visualization Explanation

### HSV Color Coding (Background)

**Colors represent direction:**
- Red → Right
- Yellow → Down-right
- Green → Down
- Cyan → Left
- Blue → Up-left
- Magenta → Up

**Brightness represents speed:**
- Bright = Fast motion
- Dark = Slow motion

### Arrow Vectors (Overlay)

White arrows overlaid on the color-coded flow:
- **Arrow direction** = Motion direction
- **Arrow length** = Motion magnitude
- Arrows are sampled on a grid (spacing controlled by `--arrow_spacing`)

## Parameters

```
--arrow_spacing INT     Spacing between arrows in pixels (default: 16)
                       Lower = more arrows, higher = fewer arrows

--arrow_scale FLOAT    Scale factor for arrow length (default: 1.0)
                       Higher = longer arrows

--save_frames          Save individual PNG frames instead of video

--fps INT              Frames per second for video (default: 12)

--create_legend        Create a color wheel legend image
```

## Examples

### Example 1: Visualize Dynamite Flow

```bash
python src/tools/visualize_flow_with_vectors.py \
    results/warped_noise/dynamite/flows_dxdy.npy \
    results/visualizations/dynamite_flow.mp4 \
    --create_legend
```

Output:
- `dynamite_flow.mp4` - Video with color-coded flow + arrows
- `flow_color_wheel_legend.png` - Color wheel explaining the colors

### Example 2: Dense Arrows for Detailed View

```bash
python src/tools/visualize_flow_with_vectors.py \
    results/fft_analysis/dynamite/hands_emphasized_flow_240x360.npy \
    hands_emphasized_viz.mp4 \
    --arrow_spacing 10 \
    --arrow_scale 1.5
```

Smaller spacing (10px) + larger arrows (1.5x) = more detailed view

### Example 3: Compare Original vs Filtered

```bash
# Visualize original flow
python src/tools/visualize_flow_with_vectors.py \
    results/fft_analysis/dynamite/original_flow.npy \
    original_viz.mp4

# Visualize filtered flow
python src/tools/visualize_flow_with_vectors.py \
    results/fft_analysis/dynamite/hands_emphasized_flow.npy \
    hands_emphasized_viz.mp4
```

Compare the two videos side-by-side to see FFT filtering effect!

### Example 4: Save Frames for Detailed Analysis

```bash
python src/tools/visualize_flow_with_vectors.py \
    results/warped_noise/my_video/flows_dxdy.npy \
    analysis/ \
    --save_frames \
    --arrow_spacing 12
```

Saves individual frames to `analysis_frames/frame_0000.png`, etc.

## Use Cases

1. **Debugging motion extraction:** Check if RAFT captured the motion correctly
2. **Comparing FFT presets:** Visualize how different frequency filters affect motion
3. **Understanding flow patterns:** See exactly where and how things are moving
4. **Creating documentation:** Generate visuals for papers/presentations

## Output Files

**Video output (.mp4):**
- Displays the flow with color coding + arrows
- Can be viewed directly or used in presentations

**Frame output (--save_frames):**
- Individual PNG files for each frame
- Useful for detailed analysis or creating custom visualizations

**Legend (--create_legend):**
- Color wheel showing what each color means
- Useful for explaining visualizations to others

## Tips

- **For high-resolution flows:** Use larger `--arrow_spacing` (20-30) to avoid clutter
- **For low-resolution flows:** Use smaller `--arrow_spacing` (8-12) for detail
- **For fast motion:** Increase `--arrow_scale` (1.5-2.0) to make arrows more visible
- **For slow motion:** Keep default `--arrow_scale` (1.0) or decrease slightly

## Integration with Pipeline

You can visualize flows at different stages:

```bash
# 1. After creating warped noise
python src/tools/visualize_flow_with_vectors.py \
    results/warped_noise/my_video/flows_dxdy.npy \
    flow_original.mp4

# 2. After FFT filtering
python src/tools/visualize_flow_with_vectors.py \
    results/fft_analysis/my_video/hands_emphasized_flow_240x360.npy \
    flow_filtered.mp4

# 3. Compare them!
```

This helps you understand what the FFT filtering is doing to your motion!
