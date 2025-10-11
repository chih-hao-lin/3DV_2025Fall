#!/bin/bash
# setup_vm1rs.sh
# Script to set up the vm1rs environment and install dependencies

set -e  # Exit on any error

echo "=== Creating conda environment ==="
conda create -n vm1rs python=3.11

echo "=== Activating environment ==="
# Conda activation inside scripts needs 'source'
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vm1rs

echo "=== Installing base requirements ==="
pip install -r requirements.txt

echo "=== Installing Grounded-SAM-2 ==="
cd third_party/Grounded-SAM-2
export CUDA_HOME=/usr/local/cuda-12
pip install -e .
pip install --no-build-isolation -e grounding_dino
pip install transformers

cd ../..

echo "=== Installing MMCV via OpenMIM ==="
pip install -U openmim
pip install --upgrade setuptools
mim install mmcv==1.3.9

echo "=== Installing ViTPose ==="
cd third_party/ViTPose
pip install -v -e .

cd ../..

echo "=== Installing VIMO ==="
pip install git+https://github.com/hongsukchoi/VIMO.git

echo "=== Installing BSTRO ==="
cd third_party/bstro
python setup.py build develop

cd ../..

echo "✅ Setup complete! Environment 'vm1rs' is ready."

