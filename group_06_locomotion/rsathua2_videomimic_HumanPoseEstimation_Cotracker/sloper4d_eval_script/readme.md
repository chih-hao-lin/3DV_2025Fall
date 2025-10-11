# SLOPER4D Evaluation Script

This directory contains evaluation scripts for the SLOPER4D dataset. Follow the steps below to process the dataset and run evaluations.

## Prerequisites

Make sure you have the required environment and dependencies installed before proceeding. Download the Sloper4D dataset from [here](http://www.lidarhumanmotion.net/data-sloper4d/) and put it in the `demo_data` directory.

## Step 1: Copy Dataset

First, copy the SLOPER4D sequences and rename the files:

```bash
# in the root directory
bash sloper4d_eval_script/copy_sloper4d_sequences.sh
```

```bash
# in the root directory
python sloper4d_eval_script/rename_files.py
```

## Step 2: Preprocess Human Data

Preprocess all human data using the following script:

```bash
bash sloper4d_eval_script/preprocess_human_all.sh
```

## Step 3: Preprocess MegaSAM

Run the distributed MegaSAM preprocessing:

```bash
conda activate mega_sam
bash sloper4d_eval_script/preprocess_megasam_distributed.sh
```

## Step 4: Run Our Method

### Hunt3r Processing
Run Hunt3r based on the chunk data:

```bash
bash sloper4d_eval_script/process_hunt3r.sh
```

### Concatenate Results
Concatenate the results to get the final output:

```bash
python sloper4d_eval_script/cat_our_results.py --seq_num 007
python sloper4d_eval_script/cat_our_results.py --seq_num 008
```

## Step 5: Run Evaluation

Navigate to the evaluation script directory and run the evaluation:

```bash
cd sloper4d_eval_script/
```

### Evaluate Our Method
```bash
python run_eval_sloper4d.py --pred_smpl_path ../results/sloper4d_seq008/hps_combined_track_0.npy --output_dir ../results/sloper4d_seq008 --gt_pkl_path ../demo_data/sloper4d/seq008_running_001/seq008_running_001_labels.pkl

python run_eval_sloper4d.py --pred_smpl_path ../results/sloper4d_seq007/hps_combined_track_0.npy --output_dir ../results/sloper4d_seq007 --gt_pkl_path ../demo_data/sloper4d/seq007_garden_001/seq007_garden_001_labels.pkl
```

### Run CD Evaluation
```bash
python cd_eval.py
```

## Notes

- Make sure to adjust the sequence number (`--seq_num`) and paths according to your specific setup
- The evaluation results will be generated in the specified output directories