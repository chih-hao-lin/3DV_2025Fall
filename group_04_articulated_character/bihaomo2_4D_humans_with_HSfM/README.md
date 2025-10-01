# Volume Fusion 2D

## Author
Bihao Mo (bihaomo2@illinois.edu)

## Set up
- Conda is recommended to manage Python packages
- You might need to set up 2 conda environments for 4D-Humans and HSfM. Check their GitHub for more info.

## How to run
- Check for the input variables inside the code. The variables are not provided down below. It is a sample pipeline on how to make the code works.

```bash
cd 4D-Humans
conda activate 4D-humans
python new.py

cd ../HSfM
conda activate hsfm
python get_pose2d_vitpose_for_hsfm.py
python align_world_env_and_smpl_hsfm_optim.py
python vis_viser_hsfm.py
