# Go-with-the-Flow Setup Documentation

**Date:** November 3, 2025
**Environment:** Ubuntu with conda, NVIDIA RTX 4090 Laptop (16GB VRAM)

---

## Current Working Configuration

### ✅ ACTIVE SETUP: Two Separate Environments (November 3, 2025)

#### Environment 1: `flow_diffusion` - For Video Generation
- **Python:** 3.10
- **PyTorch:** 2.1.0+cu118
- **TorchVision:** 0.16.0+cu118
- **Transformers:** 4.44.0
- **Diffusers:** 0.35.2
- **NumPy:** 1.26.4
- **OpenCV:** 4.11.0.86 (opencv-contrib-python)
- **SciPy:** 1.15.3
- **scikit-image:** 0.25.2
- **Additional:** timm, scipy, scikit-image, addict, yapf, ipython, einops
- **Isolation:** PYTHONNOUSERSITE=1
- **Dummy Module:** source.stable_diffusion created

#### Environment 2: `flow_warp` - For GUI Script
- **Python:** 3.10
- **PyTorch:** 2.0.1+cu118
- **TorchVision:** 0.15.2+cu118
- **Transformers:** 4.36.2
- **Diffusers:** 0.30.3
- **NumPy:** 1.26.4
- **OpenCV:** 4.11.0.86 (opencv-contrib-python)
- **SciPy:** 1.15.3
- **scikit-image:** 0.25.2
- **Additional:** timm, scipy, scikit-image, addict, yapf, ipython, einops
- **Isolation:** PYTHONNOUSERSITE=1
- **Dummy Module:** source.stable_diffusion created

### Legacy Environment: `flow` (Deprecated)
The original `flow` environment has been superseded by the two specialized environments above.
- Use `flow_diffusion` for video generation
- Use `flow_warp` for GUI operations

---

## Critical Fixes Applied

### 1. Environment Isolation
**Problem:** Conda environment was sharing packages with system Python and ROS.

**Solutions:**
- Set `PYTHONNOUSERSITE=1` for flow environment:
  ```bash
  conda activate flow
  conda env config vars set PYTHONNOUSERSITE=1
  conda deactivate
  conda activate flow
  ```
- Commented out ROS in `~/.bashrc` (line 119):
  ```bash
  # source /opt/ros/humble/setup.bash  # Commented out to prevent interference
  ```
- Removed torch packages from system Python (`~/.local/lib/python.10/site-packages/`)

### 2. Requirements.txt Fix
**File:** `/home/wding/Desktop/Go-with-the-Flow/requirements.txt`

**Change:**
```diff
- --index-url https://download.pytorch.org/whl/cu118
+ --extra-index-url https://download.pytorch.org/whl/cu118
```

**Reason:** `--index-url` only searches PyTorch index, missing packages like `diffusers`. `--extra-index-url` searches both PyPI and PyTorch.

### 3. Code Fix: cut_and_drag_gui.py
**Problem:** PyTorch import order conflicted with lazy loader.

**Fix:** Moved torch/einops imports to top of file (lines 4-5):
```python
from rp import *

# Import torch/einops BEFORE git_import to prevent lazy loader conflicts
import torch
import einops

import matplotlib.pyplot as plt
# ... rest of imports
git_import('CommonSource')
```

**Also fixed:** Line 329-331, replaced `destructure()` with direct attribute access:
```python
# Replaced destructure to avoid lazy loader issues
frames = animation_output.frames
transformed_polygons = animation_output.transformed_polygons
```

### 4. Code Fix: cut_and_drag_inference.py
**Problem:** CUDA OOM during VAE decoding phase.

**Fix:** Added memory optimizations (lines 100-103, 336-338):
```python
# Enable VAE tiling and slicing to reduce memory usage during decoding
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
print("\tVAE TILING AND SLICING ENABLED")

# Before inference:
torch.cuda.empty_cache()
```

### 5. Dummy Module Creation
**Problem:** `rp` package imports missing `source.stable_diffusion` module.

**Solution:** Created dummy module in site-packages:
```bash
# Location: /home/wding/miniconda3/envs/flow/lib/python.10/site-packages/source/
```

Files created:
- `__init__.py` (empty)
- `stable_diffusion.py` (with dummy `_get_stable_diffusion_singleton()`)

---

## Known Issues & Limitations

### PyTorch Version Conflict

**Current Status (PyTorch 2.1.0):**
- ✅ Inference script (`cut_and_drag_inference.py`) works perfectly
- ❌ GUI script (`cut_and_drag_gui.py`) may have issues with `noise_warp` module loading

**PyTorch 2.0.1 vs 2.1.0:**

| Feature | PyTorch 2.0.1 | PyTorch 2.1.0 |
|---------|---------------|---------------|
| GUI script `nw.regaussianize()` | ✅ Works | ❌ Broken |
| Inference script | ⚠️ Warnings | ✅ Works |
| Transformers compatibility | ⚠️ "PyTorch not found" | ✅ Full support |
| Diffusers compatibility | Requires 0.30.3 | Works with 0.35.2 |

**Root Cause:** PyTorch 2.1+ has stricter C++ library initialization that conflicts with `rp` package's lazy loading mechanism, preventing `noise_warp` module from loading its functions.

### ✅ Solution Implemented: Two Separate Environments

Two isolated conda environments have been created and tested:

1. **`flow_warp`** - For GUI script (PyTorch 2.0.1) ✅ WORKING
2. **`flow_diffusion`** - For inference script (PyTorch 2.1.0) ✅ WORKING

---

## Usage Instructions

### Two-Environment Setup (CURRENT SETUP - RECOMMENDED)

As of November 3, 2025, two separate isolated environments have been created:

**Environment 1: flow_diffusion (PyTorch 2.1.0) - For Video Generation**
```bash
conda activate flow_diffusion
python cut_and_drag_inference.py noise_warp_output_folder \
    --prompt "A cat waving claws" \
    --output_mp4_path "output.mp4" \
    --device "cuda" \
    --num_inference_steps 30
```

**Environment 2: flow_warp (PyTorch 2.0.1) - For GUI**
```bash
conda activate flow_warp
python cut_and_drag_gui.py
```

### Legacy: Two-Environment Setup Instructions (ALREADY COMPLETED)

**Environment 1: flow_gui (PyTorch 2.0.1)**
```bash
conda create -n flow_gui python=3.10 -y
conda activate flow_gui

# Install requirements_local.txt first, then requirements.txt (with fixed --extra-index-url)
pip install -r requirements_local.txt
pip install -r requirements.txt

# Downgrade to PyTorch 2.0.1
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Downgrade diffusers for compatibility
pip install diffusers==0.30.3 transformers==4.36.2

# Install additional dependencies
pip install scipy scikit-image addict yapf ipython timm

# Set environment isolation
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate flow_gui

# Create dummy source module
python -c "
import os, sys
site_packages = os.path.join(sys.prefix, 'lib/python.10/site-packages')
source_path = os.path.join(site_packages, 'source')
os.makedirs(source_path, exist_ok=True)
with open(os.path.join(source_path, '__init__.py'), 'w') as f: f.write('')
with open(os.path.join(source_path, 'stable_diffusion.py'), 'w') as f:
    f.write('import torch\nclass DummySD:\n    device=\"cpu\"\n_singleton=DummySD()\ndef _get_stable_diffusion_singleton():\n    return _singleton')
"

# Usage
python cut_and_drag_gui.py
```

**Environment 2: flow_video (PyTorch 2.1.0)**
```bash
conda create -n flow_video python=3.10 -y
conda activate flow_video

# Install requirements
pip install -r requirements.txt  # Uses PyTorch 2.1+ by default

# Install additional dependencies (less needed for inference only)
pip install scipy scikit-image addict yapf ipython timm

# Set environment isolation
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate flow_video

# Usage
python cut_and_drag_inference.py noise_warp_output_folder \
    --prompt "Your prompt" \
    --output_mp4_path "output.mp4" \
    --device "cuda" \
    --num_inference_steps 30
```

---

## Memory Optimization Details

### VAE Tiling & Slicing

**Why Needed:**
- CogVideoX VAE decoding requires significant VRAM
- Without optimization: ~18GB+ VRAM needed
- With tiling/slicing: ~14GB VRAM needed (fits in 16GB GPU)

**Trade-offs:**
- **Quality:** Virtually identical (no visible difference)
- **Speed:** ~10-30% slower during decoding phase
- **Necessity:** Required for 16GB GPUs, optional for 24GB+

**Technical Details:**
- **Tiling:** Processes spatial dimensions (H×W) in overlapping tiles
- **Slicing:** Processes frames sequentially instead of parallel batches
- **Implementation:** Built into Diffusers library, very reliable

---

## Troubleshooting

### "CUDA out of memory" Error
**Symptom:** Error during VAE decoding (after 100% diffusion progress)

**Solutions:**
1. VAE tiling/slicing already enabled ✅
2. Try `--low_vram` flag for CPU offloading
3. Reduce inference steps (though 30 is already reasonable)
4. Close other GPU applications

### "module 'torch' has no attribute 'float8_e4m3fn'"
**Symptom:** Error when importing diffusers

**Cause:** PyTorch version too old for diffusers version

**Solution:** Either:
- Upgrade PyTorch: `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118`
- Downgrade diffusers: `pip install diffusers==0.30.3`

### "AttributeError: module 'rp.git.CommonSource.noise_warp' has no attribute 'regaussianize'"
**Symptom:** GUI script can't find noise_warp functions

**Cause:** PyTorch 2.1+ incompatible with lazy loader

**Solution:** Use PyTorch 2.0.1:
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### "ModuleNotFoundError: No module named 'scipy/skimage/addict/yapf/IPython'"
**Solution:** Install missing package:
```bash
pip install scipy scikit-image addict yapf ipython
```

---

## File Modifications Summary

### Modified Files
1. `/home/wding/Desktop/Go-with-the-Flow/requirements.txt`
   - Changed `--index-url` to `--extra-index-url`

2. `/home/wding/Desktop/Go-with-the-Flow/cut_and_drag_gui.py`
   - Lines 4-5: Added `import torch` and `import einops` before `git_import`
   - Lines 329-331: Replaced `destructure()` with direct attribute access

3. `/home/wding/Desktop/Go-with-the-Flow/cut_and_drag_inference.py`
   - Lines 100-103: Added VAE tiling and slicing
   - Lines 336-338: Added CUDA cache clearing before inference

### System Configuration Changes
1. `~/.bashrc` line 119: Commented out ROS setup
2. Conda environment variable: `PYTHONNOUSERSITE=1` set for flow environment
3. Dummy module created: `/home/wding/miniconda3/envs/flow/lib/python.10/site-packages/source/`

---

## Package Version Lock File

For reproducibility, here are the exact working versions:

```
# Core ML
torch==2.1.0+cu118
torchvision==0.16.0+cu118
transformers==4.44.0
diffusers==0.35.2

# Scientific Computing
numpy==1.26.4
scipy==1.15.3
scikit-image==0.25.2

# Vision
opencv-contrib-python==4.11.0.86

# Additional
timm==1.0.21
addict==2.4.0
yapf==0.43.0
ipython==8.37.0
einops==0.8.1
easydict==1.13

# Already in requirements
rp==0.1.1348
```

---

## Testing Checklist

### GUI Script Test (PyTorch 2.0.1 required)
```bash
conda activate flow_gui  # or flow with PyTorch 2.0.1
python -c "
from rp import *
import torch
import einops
git_import('CommonSource')
import rp.git.CommonSource.noise_warp as nw
test_noise = torch.randn(4, 32, 32)
result, counts = nw.regaussianize(test_noise)
print(f'✓ regaussianize works: {test_noise.shape} -> {result.shape}')
"
```

### Inference Script Test (PyTorch 2.1.0)
```bash
conda activate flow_video  # or flow with PyTorch 2.1.0
python -c "
from diffusers import CogVideoXImageToVideoPipeline
from transformers import T5EncoderModel
import torch
print('✓ All imports successful')
print(f'PyTorch: {torch.__version__}')
"
```

---

## Future Improvements

1. **Create separate conda environments** for GUI and inference scripts
2. **Add requirements-gui.txt** and **requirements-video.txt** for each environment
3. **Add memory profiling** to monitor VRAM usage during generation
4. **Automate environment setup** with a shell script
5. **Add progress bars** for VAE decoding phase (currently happens after 100%)

---

## References

- [Go-with-the-Flow GitHub](https://github.com/YOUR_REPO_HERE)
- [CogVideoX Documentation](https://huggingface.co/THUDM/CogVideoX-5b-I2V)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)

---

**Last Updated:** November 3, 2025
**Maintained By:** Environment setup documented during debugging session
