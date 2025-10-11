#!/bin/bash

# Add signal handling to terminate all child processes on Ctrl+C
trap 'echo "Received interrupt signal, terminating all processes..."; kill $(jobs -p) 2>/dev/null; exit 1' INT TERM

VIDEO_NAME=$1
GPU_LIST=$2  # Comma-separated list of GPU IDs, e.g., "0,1,2"

BASE_DIR="demo_data"
video_dir="${BASE_DIR}/input_images/${VIDEO_NAME}/cam01"

# Count the number of frames (include both regular files and symbolic links)
num_frames=$(find ${video_dir} \( -type f -o -type l \) \( -name "*.jpg" -o -name "*.png" \) | wc -l)
echo "Number of frames: $num_frames in ${video_dir}"

# If no image files found, output directory contents for debugging
if [ $num_frames -eq 0 ]; then
    echo "No image files found. Directory contents:"
    ls -la ${video_dir}
    exit 1
fi

# Convert comma-separated GPU list to array
IFS=',' read -ra GPU_ARRAY <<< "$GPU_LIST"
num_gpus=${#GPU_ARRAY[@]}

# Calculate total number of 100-frame blocks
# Calculate total number of 300-frame blocks
total_blocks=$(( (num_frames + 299) / 300 ))
blocks_per_gpu=$(( (total_blocks + num_gpus - 1) / num_gpus ))

echo "Processing $num_frames frames ($total_blocks blocks) using $num_gpus GPUs with $blocks_per_gpu blocks per GPU"

# Process blocks in parallel on different GPUs
for ((i=0; i<num_gpus; i++)); do
    gpu_id=${GPU_ARRAY[$i]}
    
    # Calculate block range for this GPU
    start_block=$((i * blocks_per_gpu))
    end_block=$((start_block + blocks_per_gpu))
    
    # Ensure end_block doesn't exceed total blocks
    if [ $end_block -gt $total_blocks ]; then
        end_block=$total_blocks
    fi
    
    # Skip if this GPU has no blocks
    if [ $start_block -ge $total_blocks ]; then
        continue
    fi
    
    # echo "GPU $gpu_id processing blocks $start_block to $((end_block-1)) (frames $((start_block*100)) to $((end_block*100-1)))"
    echo "GPU $gpu_id processing blocks $start_block to $((end_block-1)) (frames $((start_block*300)) to $((end_block*300-1)))"
    
    # Process each block SEQUENTIALLY on this GPU
    (
        for ((block=start_block; block<end_block; block++)); do
            # block_start=$((block * 100))
            # block_end=$((block_start + 100))
            block_start=$((block * 300))
            block_end=$((block_start + 300 + 1))
            
            # Ensure block_end doesn't exceed total frames
            if [ $block_end -gt $num_frames ]; then
                block_end=$num_frames
            fi
            
            echo "GPU $gpu_id processing block $block (frames $block_start to $((block_end-1)))"
            
            # Run the processing command with the specific GPU - wait for it to complete
            CUDA_VISIBLE_DEVICES=$gpu_id python stage1_reconstruction/00_megasam_reconstruction.py \
                --video-dir "${BASE_DIR}/input_images/${VIDEO_NAME}/cam01" \
                --out-dir "${BASE_DIR}/input_megasam" \
                --start-frame "$block_start" \
                --end-frame "$block_end" \
                --stride 1 \
                --gsam2
            
            # Check if the command succeeded
            if [ $? -ne 0 ]; then
                echo "Error processing block $block (frames $block_start to $((block_end-1))) on GPU $gpu_id"
                exit 1
            fi
            
            echo "GPU $gpu_id completed block $block (frames $block_start to $((block_end-1)))"
        done
    ) &  # Run the entire sequence for this GPU in background
    
    # Add a small delay to prevent all processes from starting simultaneously
    sleep 2
done

# Wait for all background tasks to complete
wait

echo "Processing complete for ${VIDEO_NAME}" 