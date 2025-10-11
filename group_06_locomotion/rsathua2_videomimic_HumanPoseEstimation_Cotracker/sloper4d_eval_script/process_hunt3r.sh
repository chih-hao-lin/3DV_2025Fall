#!/bin/bash

for h5_file in ./demo_data/input_megasam/megasam_reconstruction_results_seq007_garden_001_imgs_cam01_frame_*.h5 \
                 ./demo_data/input_megasam/megasam_reconstruction_results_seq008_running_001_imgs_cam01_frame_*.h5; do
    
    if [ ! -f "$h5_file" ]; then
        echo "No files found for pattern $h5_file"
        continue
    fi
    
    filename=$(basename "$h5_file")
    temp_name=${filename#megasam_reconstruction_results_}
    sequence_name=${temp_name%_cam01_frame_*}

    echo "Processing $h5_file of $sequence_name"

    python stage2_optimization/01_megahunter_optimization.py \
        --world-env-path "$h5_file" \
        --bbox-dir "./demo_data/input_masks/${sequence_name}/cam01/json_data" \
        --pose2d-dir "./demo_data/input_2d_poses/${sequence_name}/cam01" \
        --smpl-dir "./demo_data/input_3d_meshes/${sequence_name}/cam01" \
        --out-dir ./demo_data/output_smpl_and_points
done