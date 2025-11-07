# Go-with-the-Flow: Segmentation-Based Motion Control

## Hacker: Wenzhou Ding

Extension to Go-with-the-Flow enabling independent foreground and background motion scaling using SAM2 segmentation masks.
## Setup

```bash
pip install -r requirements.txt
pip install -r requirements_local.txt
```

## Usage

### Standard Workflow (No Segmentation)

```bash
# 1. Extract optical flow and create warped noise
python src/pipeline/make_warped_noise.py \
    data/videos/my_video.mp4 \
    results/warped_noise/my_video/

# 2. Generate video with motion transfer
python src/pipeline/cut_and_drag_inference.py \
    results/warped_noise/my_video/ \
    results/generated/output.mp4 \
    --prompt "A dancing flame" \
    --degradation 0.0 \
    --num_inference_steps 50
```

### Segmentation-Based Motion Control

**Step 1: Generate SAM2 Masks**
- Use [Meta's SAM2 Demo](https://sam2.metademolab.com/demo)
- Export video with masks (white = foreground, other colors = background)
- Save to `data/videos_sam/my_video_sam2.mp4`

**Step 2: Create Warped Noise with Segmentation Scaling**
```bash
python src/pipeline/make_warped_noise_with_segmentation.py \
    data/videos/my_video.mp4 \
    results/warped_noise/my_video_fg1.5_bg1.0/ \
    --mask data/videos_sam/my_video_sam2.mp4 \
    --fg_scale 1.5 \
    --bg_scale 1.0
```

**Step 3: Generate Video**
```bash
python src/pipeline/cut_and_drag_inference.py \
    results/warped_noise/my_video_fg1.5_bg1.0/ \
    results/generated/my_video_scaled.mp4 \
    --prompt "Your prompt here" \
    --degradation 0.0 \
    --num_inference_steps 50
```

### Visualization Tools

**Visualize optical flow with motion vectors:**
```bash
python src/tools/visualize_flow_with_vectors.py \
    results/warped_noise/my_video_fg1.5_bg1.0/flows_dxdy.npy \
    results/visualizations/flow_vectors.mp4 \
    --arrow_spacing 20
```

**Create 3-way comparison (SAM | Original Flow | Scaled Flow):**
```bash
python src/tools/create_three_way_comparison.py \
    data/videos_sam/my_video_sam2_vis.mp4 \
    results/visualizations/original_flow.mp4 \
    results/visualizations/scaled_flow.mp4 \
    results/visualizations/comparison.mp4
```

## Parameters

### Segmentation Scaling
- `--fg_scale`: Foreground motion multiplier (e.g., 1.5 = 50% faster)
- `--bg_scale`: Background motion multiplier (e.g., 0.3 = 70% slower, 1.0 = unchanged)
- `--mask_threshold`: Foreground detection threshold (default: 200, range: 0-255)

### Video Generation
- `--prompt`: Text description for generation
- `--degradation`: Motion strength (0.0 = full motion, 1.0 = no motion)
- `--num_inference_steps`: Diffusion steps (higher = better quality, slower)
- `--model_name`: Model selection
  - T2V: `T2V5B_blendnorm_i25000_DATASET_nearest_lora_weights`, `T2V2B_RDeg_i30000_lora_weights`
  - I2V: `I2V5B_final_i38800_nearest_lora_weights` (default)

## Key Technical Details

### Motion Encoding
Motion in diffusion models is encoded in the **warped noise structure**, not the flow field. Therefore:
- Scaling flow AFTER warping has no effect
- Must apply segmentation scaling BEFORE warping noise
- Use the integrated pipeline (`make_warped_noise_with_segmentation.py`)

### Temporal Alignment
SAM masks and video frames must share identical temporal timelines:
1. Both video and SAM are resampled to 49 frames using the same indices
2. Optical flow (48 frames) is extracted from the 49-frame video
3. First 48 SAM masks are used to match flow frame count

Failure to align properly causes incorrect motion scaling (background affected when scaling foreground).

## Technical Documentation

See [src/README.md](src/README.md) for detailed implementation documentation including:
- Architecture decisions and design rationale
- Temporal alignment bug and solution
- Flow scaling formulas and resolution handling
- Common pitfalls and testing procedures

## Acknowledgements

- Original Go-with-the-Flow: [Project Page](https://eyeline-labs.github.io/Go-with-the-Flow/) | [Paper](https://arxiv.org/abs/2501.08331)
- [Meta's SAM2](https://sam2.metademolab.com/demo) for segmentation

