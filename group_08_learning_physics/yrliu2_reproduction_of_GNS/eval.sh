set -e

vis_output_dir=./eval_results/tmp
vis_index=11 # id of rollout to visualize in the testing set (max size = 30)
ckpt_path=/path/to/your/checkpoint.pth  # path to the checkpoint to evaluate
test_data_root=./rollouts_test

# just for visualization, so limit the batch size to 100
python train.py \
    --test_only \
    --test_vis_index $vis_index \
    --test_max_batch 100 \
    --test_checkpoint $ckpt_path \
    --test_root $test_data_root \
    --output_dir $vis_output_dir \

