# GNS reproduction on 2D fluid

## Hacker: Yi-Ruei Liu (yrliu2@illinois.edu)

## Setup
```
# The implementation is tested on Ubuntu 24.04 with RTX 4090 GPU.
conda create --name GNS python=3.10
conda activate GNS
pip install -r requirements.txt
```

## How to run?
+ Collecting rollouts
```
bash collect_rollout.sh
```
+ Train the model
```
# It takes about 1 day to train 2M steps
bash train.sh
```
+ Evaluation (visualization)
```
# Change the ckpt path in eval.sh in advance
bash eval.sh
```

## Acknowledgement
+ The simulation and visualization of this project is based on [Taichi](https://github.com/taichi-dev/taichi/). Thank for their work.