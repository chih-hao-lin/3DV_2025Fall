#!/bin/bash

# Add signal handling
trap 'echo "Received interrupt signal, terminating all processes..."; pkill -P $$; exit 1' INT TERM

# Run seq007 on GPUs 0,1
bash sloper4d_eval_script/preprocess_megasam_multi_gpu.sh seq007_garden_001_imgs "0,1" &

# Run seq008 on GPUs 2,3
bash sloper4d_eval_script/preprocess_megasam_multi_gpu.sh seq008_running_001_imgs "2,3" &

# Wait for all background tasks to complete
wait

echo "All processing complete" 