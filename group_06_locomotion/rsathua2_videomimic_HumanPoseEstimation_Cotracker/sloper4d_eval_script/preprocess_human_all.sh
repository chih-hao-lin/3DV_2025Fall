# using cuda:0 for seq007_garden_001_imgs and cuda:1 for seq008_running_001_imgs
CUDA_VISIBLE_DEVICES=0 bash preprocess_human.sh seq007_garden_001_imgs 0 &
CUDA_VISIBLE_DEVICES=1 bash preprocess_human.sh seq008_running_001_imgs 0

# wait for all background tasks to complete
wait