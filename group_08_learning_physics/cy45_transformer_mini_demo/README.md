# Transformer Mini Demo of Water Drop

## Hacker: Chengyu Yang (cy45@illinois.edu)

## Setup
```
# The implementation is tested on Ubuntu 22.04 with NVIDIA RTX 6000 Ada.
# Customize env path in the file before running it.
conda env create -f env_server.yml
conda activate transformer_scratch
```

## How to run?
+ Check sample data and simulation of the single trajectory, which will be used as ground truth.
+ Train the model.
```
python particle_trainer.py
```
+ Evaluation (visualization).
```
# Change the pth path to check exsisting results.
python particle_evaluator.py
```

## Acknowledgement
+ The ground truth data come from [learning_to_simulate](https://github.com/sigma-pi/deepmind-research/tree/master/learning_to_simulate) and are converted from tfrecord to npz files. Thanks for their work.