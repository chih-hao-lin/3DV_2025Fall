# Attention Visualization for VGGT

This script provides enhanced attention visualization for the VGGT model, with the ability to upsample attention weights to image size and overlay them with alpha blending.

## Features

- **Upsampled Attention Maps**: Attention weights are upsampled from patch size (16x16) to full image resolution
- **Alpha Blending**: Attention maps are overlaid on original images with configurable alpha blending
- **Multiple Visualization Types**:
  - Standard attention overlay visualization
  - Side-by-side comparison (original vs. attention overlay)
  - Individual attention maps for each view
- **Configurable Parameters**: Alpha blending factor, colormap, and other visualization options
- **Original Image Reference**: Saves original images for easy comparison

## Usage

### Basic Usage

```bash
python attn_viz.py --image_folder /path/to/images
```

### Advanced Usage

```bash
python attn_viz.py \
    --image_folder /path/to/images \
    --alpha 0.8 \
    --colormap plasma \
    --comparison \
    --save_individual
```

### Command Line Arguments

- `--image_folder`: Path to folder containing images (default: "examples/kitchen/images/")
- `--alpha`: Alpha blending factor for attention overlay (default: 0.7)
- `--colormap`: Colormap for attention visualization (default: "viridis")
- `--comparison`: Create side-by-side comparison visualizations
- `--save_individual`: Save individual attention maps for each view
- `--use_point_map`: Use point map instead of depth-based points
- `--background_mode`: Run the viser server in background mode
- `--port`: Port number for the viser server (default: 8080)

### Available Colormaps

Common matplotlib colormaps include:
- `viridis` (default)
- `plasma`
- `inferno`
- `magma`
- `hot`
- `cool`
- `spring`
- `summer`
- `autumn`
- `winter`

## Output

The script generates several types of visualizations in the `attention_visualizations/` folder:

1. **Original Images**: `original_images.png` - Reference images for comparison
2. **Frame Attention**: 
   - `frame_attention_layer_X.png` - Standard attention overlay
   - `frame_attention_comparison_layer_X.png` - Side-by-side comparison (if `--comparison` is used)
   - `frame_attention_layer_X_view_Y_individual.png` - Individual views (if `--save_individual` is used)
3. **Global Attention**:
   - `global_attention_layer_X_viewY.png` - Global attention visualization
   - `global_attention_layer_X_source_Y_target_Z_individual.png` - Individual global attention maps

## Technical Details

### Attention Weight Processing

1. **Extraction**: Attention weights are extracted from the model's attention layers
2. **Reshaping**: Patch-based attention weights are reshaped to 16x16 patches
3. **Upsampling**: Attention maps are upsampled to full image resolution using bilinear interpolation
4. **Normalization**: Attention weights are normalized to [0, 1] range
5. **Colormapping**: Normalized weights are mapped to colors using the specified colormap
6. **Alpha Blending**: Colored attention maps are blended with original images using the formula:
   ```
   blended = (1 - alpha) * original + alpha * attention_colored
   ```

### Image Processing

- Images are automatically denormalized if they are in [0, 1] range
- RGB channels are properly handled for visualization
- Aspect ratios are preserved during visualization

## Testing

Run the test script to verify the visualization functionality:

```bash
python test_attn_viz.py
```

This will create dummy attention weights and images to test all visualization functions.

## Requirements

- torch
- numpy
- matplotlib
- opencv-python (cv2)
- tqdm
- viser (for 3D visualization)
- seaborn

## Example Output

The script generates high-quality visualizations showing:
- How attention weights are distributed across image patches
- Which regions of the image the model focuses on during processing
- Cross-view attention patterns in multi-view scenarios
- Layer-wise attention evolution through the model

This enhanced visualization helps understand the model's attention mechanisms and interpretability. 