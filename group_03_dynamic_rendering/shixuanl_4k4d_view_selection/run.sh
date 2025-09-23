#!/bin/bash

#SBATCH --account=bdaj-delta-gpu
#SBATCH --job-name=run
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --partition=gpuA40x4
#SBATCH --ntasks-per-node=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --export=ALL

# python job to run
# conda init bash


evc-tests -c configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml,configs/specs/eval.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml,configs/specs/video.yaml
