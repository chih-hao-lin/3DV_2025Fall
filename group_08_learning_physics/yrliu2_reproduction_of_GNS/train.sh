set -e

date=$(date +%Y%m%d_%H%M%S)
output_dir=./outputs/$date

# 500k steps is sufficient to get decent results
# more steps might improve performance further

# n_step=500000
# log_every=200
# val_every=10000
# vis_every=20000
# save_every=50000

n_step=2000000
log_every=500
val_every=50000
vis_every=100000
save_every=200000

python train.py \
    --train_root ./rollouts_train \
    --val_root ./rollouts_val \
    --test_root ./rollouts_test \
    --output_dir $output_dir \
    --max_steps $n_step \
    --log_every $log_every \
    --val_every $val_every \
    --vis_every $vis_every \
    --save_every $save_every 
