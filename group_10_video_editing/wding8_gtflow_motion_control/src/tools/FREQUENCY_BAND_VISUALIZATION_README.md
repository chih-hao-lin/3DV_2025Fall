# Frequency Band Visualization

Visualize individual frequency bands from FFT decomposition to understand what motion each band captures.

## Quick Start

```bash
conda activate flow_warp

# Create grid showing all bands together
python src/tools/visualize_frequency_bands.py \
    results/fft_analysis/beat_it_bands/ \
    --output results/visualizations/bands_grid.mp4

# Create separate videos for each band
python src/tools/visualize_frequency_bands.py \
    results/fft_analysis/beat_it_bands/ \
    --output results/visualizations/bands/ \
    --separate
```

## What Are Frequency Bands?

When we apply FFT (Fast Fourier Transform) to optical flow, we decompose motion into **frequency bands** based on how fast the motion changes over time.

### Beat It Example (12 fps, 4 bands)

For the beat_it video at 12 fps, we get 4 frequency bands:

**Band 0: 0.10-0.28 Hz (Slow drift)**
- Period: 3.6-10 seconds
- Contains: Very slow camera drift, gradual panning
- Motion type: MICRO_MOTION (magnitude: 1.03)

**Band 1: 0.28-0.77 Hz (Medium motion)**
- Period: 1.3-3.6 seconds
- Contains: Body swaying, slow gestures
- Motion type: MICRO_MOTION (magnitude: 1.36)

**Band 2: 0.77-2.16 Hz (Fast gestures)**
- Period: 0.46-1.3 seconds
- Contains: Hand movements, head motion, walking
- Motion type: CAMERA_SHAKE (magnitude: 1.53)

**Band 3: 2.16-6.00 Hz (Very fast/shake)**
- Period: 0.17-0.46 seconds
- Contains: Camera shake, jitter, rapid movements
- Motion type: CAMERA_SHAKE (magnitude: 1.24)

## Understanding the Visualization

### Grid View (Default)
Shows all bands in a 2x3 or 2x2 grid:
- **Top-left**: Original flow (all frequencies combined)
- **Other panels**: Individual frequency bands

Each panel shows:
- HSV color coding (hue = direction, brightness = magnitude)
- Band number and frequency range
- Average magnitude for that band

### Separate Videos (--separate)
Creates individual video files for each band:
- `band_0_0.10-0.28Hz.mp4` - Band 0
- `band_1_0.28-0.77Hz.mp4` - Band 1
- `band_2_0.77-2.16Hz.mp4` - Band 2
- `band_3_2.16-6.00Hz.mp4` - Band 3

## Use Cases

### 1. Understanding Motion Content
See exactly what type of motion each frequency band captures:
```bash
python src/tools/visualize_frequency_bands.py \
    results/fft_analysis/my_video/ \
    --separate
```

Then watch each band video to see:
- Band 0: Is there camera drift?
- Band 1: What slow motions are present?
- Band 2: What hand/body gestures occur?
- Band 3: How much shake/jitter is there?

### 2. Debugging FFT Filtering
When using FFT presets (hands_emphasized, ultra_smooth, etc.), visualize what you're keeping vs removing:

```bash
# Decompose original flow
python src/tools/frequency_motion_transfer.py \
    --flow_path results/warped_noise/my_video/flows_dxdy.npy \
    --output results/fft_analysis/my_video_bands/ \
    --analyze_only

# Visualize all bands
python src/tools/visualize_frequency_bands.py \
    results/fft_analysis/my_video_bands/ \
    --output results/visualizations/my_video_bands_grid.mp4
```

Compare this with your filtered flow to see what frequencies you removed.

### 3. Choosing FFT Preset
Look at the band analysis to decide which preset to use:

```bash
# Analyze motion
python src/tools/frequency_motion_transfer.py \
    --flow_path results/warped_noise/my_video/flows_dxdy.npy \
    --output results/fft_analysis/my_video_bands/ \
    --analyze_only
```

The output shows classification for each band:
- If Band 2-3 are "CAMERA_SHAKE" → Use `ultra_smooth` or `remove_shake`
- If Band 1-2 are "BODY_MOVEMENT" → Use `hands_emphasized`
- If Band 0-1 are "MICRO_MOTION" → They're safe to keep

### 4. Creating Custom FFT Presets
Visualize bands to design custom filtering:

```bash
# See what each band contains
python src/tools/visualize_frequency_bands.py \
    results/fft_analysis/my_video_bands/ --separate

# Then create custom weights based on what you see
# Example: Keep band 0-1, reduce band 2, remove band 3
python src/tools/frequency_motion_transfer.py \
    --flow_path results/warped_noise/my_video/flows_dxdy.npy \
    --custom_weights 1.0,1.0,0.3,0.0
```

## Parameters

```
--output PATH          Output path for video/directory
                       Grid mode: path to .mp4 file
                       Separate mode: path to directory

--separate             Create separate video for each band
                       instead of grid

--fps INT              Frames per second (default: 12)

--black_background     Use black background instead of white

--no_original          Don't include original flow in grid
```

## Examples

### Example 1: Quick Grid View
```bash
python src/tools/visualize_frequency_bands.py \
    results/fft_analysis/beat_it_bands/
```

Creates `results/fft_analysis/beat_it_bands/bands_grid.mp4`

### Example 2: Separate Band Videos
```bash
python src/tools/visualize_frequency_bands.py \
    results/fft_analysis/beat_it_bands/ \
    --output results/visualizations/beat_it_bands/ \
    --separate
```

Creates:
- `results/visualizations/beat_it_bands/band_0_0.10-0.28Hz.mp4`
- `results/visualizations/beat_it_bands/band_1_0.28-0.77Hz.mp4`
- `results/visualizations/beat_it_bands/band_2_0.77-2.16Hz.mp4`
- `results/visualizations/beat_it_bands/band_3_2.16-6.00Hz.mp4`

### Example 3: Black Background
```bash
python src/tools/visualize_frequency_bands.py \
    results/fft_analysis/beat_it_bands/ \
    --black_background
```

### Example 4: Grid Without Original
```bash
python src/tools/visualize_frequency_bands.py \
    results/fft_analysis/beat_it_bands/ \
    --no_original
```

Shows only the 4 bands in a 2x2 grid.

## Complete Workflow

Here's the full workflow from video to band visualization:

```bash
# 1. Create warped noise (extracts flow)
conda activate flow_warp
python src/pipeline/make_warped_noise.py \
    data/videos/my_video.mp4 \
    results/warped_noise/my_video/

# 2. Decompose flow into frequency bands
python src/tools/frequency_motion_transfer.py \
    --flow_path results/warped_noise/my_video/flows_dxdy.npy \
    --output results/fft_analysis/my_video_bands/ \
    --fps 12 \
    --num_bands 4 \
    --analyze_only

# 3. Visualize all bands
python src/tools/visualize_frequency_bands.py \
    results/fft_analysis/my_video_bands/ \
    --output results/visualizations/my_video_bands_grid.mp4

# 4. Watch the video to understand motion composition!
```

## Files Required

The script expects these files in the input directory:
- `band_0_flow.npy` - Band 0 flow
- `band_1_flow.npy` - Band 1 flow
- `band_2_flow.npy` - Band 2 flow
- `band_3_flow.npy` - Band 3 flow (if 4 bands)
- `band_info.json` - Band metadata (frequency ranges, fps)
- `original_flow.npy` - Original flow for comparison (optional)

These are automatically created by `frequency_motion_transfer.py`.

## Tips

- **Higher FPS = Higher Max Frequency**: At 30fps, Band 3 goes up to 15Hz. At 12fps, it only goes to 6Hz.
- **More Bands = Finer Control**: Use 6-8 bands for very precise frequency control.
- **Watch in Slow Motion**: Some bands have subtle motion - watch frame by frame.
- **Compare with Original**: Always look at the original flow panel to see how bands combine.

## Integration with FFT Presets

The FFT presets work by weighting these bands:

| Preset | Band 0 | Band 1 | Band 2 | Band 3 | Effect |
|--------|--------|--------|--------|--------|--------|
| `ultra_smooth` | 1.0 | 0.8 | 0.3 | 0.0 | Keep slow drift, reduce fast motion |
| `hands_emphasized` | 1.0 | 1.0 | 1.5 | 0.2 | Boost hand gestures (Band 2) |
| `remove_shake` | 1.0 | 1.0 | 1.0 | 0.0 | Remove high-freq shake |
| `extreme_hands` | 0.3 | 1.0 | 2.0 | 0.0 | Extreme hand emphasis |

Visualizing bands helps you understand what these presets are actually doing to your motion!
