# Energy Restrain Experiments (xuejunz2_energy_restrain)

This folder contains code and visualizations for energy-constraint experiments. To reproduce the baseline simulator and rollouts we rely on the official "Learning to Simulate" (ICML 2020) codebase. Please follow the steps below.

## 0) Clone Learning to Simulate (L2S)

Clone (from the parent directory where you keep repos):
```bash
git clone https://github.com/google-deepmind/learning_to_simulate.git
```

## 1) (Optional) TACC environment
If using TACC, load CUDA/cuDNN modules first:
```bash
module load cuda/10.0
module load cudnn/7.6.2
```

## 2) Install dependencies
From the parent directory (containing the repo):
```bash
pip install -r learning_to_simulate/requirements.txt
```

## 3) Example: Train a model and display a trajectory (WaterRamps)
Download dataset:
```bash
mkdir -p /tmp/datasets
bash ./learning_to_simulate/download_dataset.sh WaterRamps /tmp/datasets
```

Train a model:
```bash
mkdir -p /tmp/models
python -m learning_to_simulate.train \
    --data_path=/tmp/datasets/WaterRamps \
    --model_path=/tmp/models/WaterRamps
```

Generate test rollouts:
```bash
mkdir -p /tmp/rollouts
python -m learning_to_simulate.train \
    --mode="eval_rollout" \
    --data_path=/tmp/datasets/WaterRamps \
    --model_path=/tmp/models/WaterRamps \
    --output_path=/tmp/rollouts/WaterRamps
```

Plot a rollout trajectory:
```bash
python -m learning_to_simulate.render_rollout \
    --rollout_path=/tmp/rollouts/WaterRamps/rollout_test_0.pkl
```

---

## This folder (energy restrain) scripts
- `train.py` / `train_plus.py`: baseline/extended training
- `energy_correction_global.py`: global energy correction and summary plots.
- `kinetic_energy.py`: compute/plot kinetic energy curves.
- `speed_calibration.py`: speed calibration utilities.
- `error_boundary.py`: error boundary statistics and visualization.
- `heat_map.py`: heatmap generation.
- Figures: `kinetic.png`, `distance.png`, `speed.png`, `heatmap.png`; overlays as `*_overlay.gif`.