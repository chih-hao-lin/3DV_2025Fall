# Setup Guide

This guide walks you through setting up the VideoMimic Real-to-Sim pipeline for transforming human motion videos into robot-ready motion data.

## Overview

The pipeline requires two separate conda environments due to dependency conflicts:

| Environment | Name | Python | CUDA | Purpose |
|------------|------|--------|------|---------|
| **Main** | `vm1rs` | 3.12 | 12.4+ | Human preprocessing, optimization, retargeting |
| **Reconstruction** | `vm1reocn` | 3.10 | 11.8 | MegaSam, NKSR meshification, GeoCalib |

> **Why two environments?** MegaSam requires xformers ≤0.0.27 (due to deprecated NyquistAttention) which only compiles with CUDA 11.8. NKSR is also tied to CUDA 11.8.

## Test Environment / Prerequisites

- Ubuntu 24.04
- Conda package manager
- ~10GB free disk space for models and data
- NVIDIA GPU with CUDA support (A5000, A6000, A6000 ADA, A100 40GB, A100 80GB)

## Installation

### 1. Main Environment (`vm1rs`)

Create and activate the main environment:

```bash
conda create -n vm1rs python=3.12
conda activate vm1rs
```

Install other dependencies:

```bash
pip install -r requirements.txt
```

#### Human Detection & Pose Estimation

```bash
# 1. Grounded-SAM-2 (bounding boxes and segmentation)
cd third_party/
git clone https://github.com/hongsukchoi/Grounded-SAM-2.git
cd Grounded-SAM-2
export CUDA_HOME=/usr/local/cuda-12.4  # Adjust to your CUDA version
pip install -e .                        # Segment Anything 2
pip install --no-build-isolation -e grounding_dino  # Grounding DINO
pip install transformers
cd ../..

# 2. ViTPose (2D pose estimation)
pip install -U openmim
pip install --upgrade setuptools
mim install mmcv==1.3.9  # If error, try: pip install setuptools --upgrade
cd third_party/
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
cd ../..

# 3. VIMO (3D human mesh - primary method)
pip install git+https://github.com/hongsukchoi/VIMO.git

# 4D Humans (deprecated)
# pip install git+https://github.com/hongsukchoi/4D-Humans.git

# 4. BSTRO (contact detection)
cd third_party/
git clone --recursive https://github.com/hongsukchoi/bstro.git
cd bstro
python setup.py build develop
cd ../..
```

<details>
<summary>Troubleshooting: g++-11 errors</summary>

If you encounter g++-11 related errors:

```bash
# Install g++-11
sudo apt update
sudo apt install g++-11

# Set environment variables
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# Retry the installation
pip install --no-build-isolation -e grounding_dino
```
</details>

#### MegaHunter + PyRoki 

```bash
# Second order optimization for MegaHunter and PyRoki
pip install -U "jax[cuda12]"
pip install "git+https://github.com/brentyi/jaxls.git"
# PyRoki for robot motion retargeting
git clone https://github.com/chungmin99/pyroki.git
cd pyroki
# pyroki might have updated some variable names; git checkout 70b30a56b1e1ea83fb4c2cac8fe2c63a0624b9ce 
pip install -e .
cd ../..
```

#### Core Dependencies

```bash
# PyTorch (avoid 2.6 - it's unstable)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Viser for visualization
cd third_party/
git clone https://github.com/nerfstudio-project/viser
cd viser
pip install -e .
cd ../..
```


#### Optional: World Reconstruction (Align3r)

```bash
# Monst3r/Align3r (skip if only using MegaSam)
cd third_party/
git clone https://github.com/Junyi42/monst3r-depth-package.git
cd monst3r-depth-package
pip install -e .
cd ../..
pip install git+https://github.com/Junyi42/croco_package.git
```

#### Optional: Neural Meshification (NDC)

```bash
# NDC (skip if only using NKSR)
pip install trimesh h5py cython opencv-python
cd third_party/NDC
python setup.py build_ext --inplace
cd ../..
```

#### Optional: Hand Pose Estimation

```bash
# WiLor (3D hand mesh)
pip install git+https://github.com/warmshao/WiLoR-mini
```

### 2. Reconstruction Environment (`vm1reocn`)

This environment handles MegaSam reconstruction, NKSR meshification, and GeoCalib operations.

```bash
cd third_party/
git clone --recursive https://github.com/Junyi42/megasam-package
cd megasam-package

# Create environment from yaml
conda env create -f environment.yml

conda activate vm1recon

# other dependencies
pip install -r requirements.txt

# Additional dependencies
# cuda 11.8 is required
export CUDA_HOME=/usr/local/cuda-11.8

# Install g++-11 if not already installed
# sudo apt update
# sudo apt install g++-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
pip install torch-scatter==2.1.2

# Install specific xformers version (required for MegaSam)
wget https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
conda install xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
rm xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2

# Compile DROID-SLAM components
cd base

python setup.py install
cd ../..

# NKSR for fast meshification
conda install -c pyg -c nvidia -c conda-forge pytorch-lightning=1.9.4 tensorboard pybind11 pyg rich pandas omegaconf
pip install -f https://pycg.huangjh.tech/packages/index.html python-pycg[full]==0.5.2 randomname pykdtree plyfile flatten-dict pyntcloud
pip install nksr -f https://nksr.huangjh.tech/whl/torch-2.0.0+cu118.html
pip install trimesh tyro h5py rtree
cd ..

# GeoCalib for gravity calibration
git clone https://github.com/hongsukchoi/GeoCalib.git third_party/GeoCalib
cd third_party/GeoCalib
pip install -e .
cd ../..
```

## Environment Quick Reference

Always activate the correct environment before running commands:

```bash
# Most operations
conda activate vm1rs

# For MegaSam reconstruction and postprocessing
conda activate vm1reocn
```

See [commands.md](./commands.md) for detailed usage instructions.

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   - Ensure CUDA 12.4+ is available for `vm1rs`
   - Ensure CUDA 11.8 is available for `vm1reocn`

2. **Memory Errors**
   - MegaSam: Requires ~24GB+ GPU memory for 300 frames
   - Align3r: Requires ~80GB+ GPU memory for 150 frames
   - Reduce `--end-frame` or use `--stride` to process fewer frames

3. **Import Errors**
   - Verify you're in the correct conda environment
   - Check that all installation steps completed without errors

### Getting Help

- Check existing issues on GitHub
- Include error messages and system info (GPU, CUDA, etc.) when reporting issues
