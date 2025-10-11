set -e

n_particles=1900
n_frames=750
n_rollouts=1000
seed=0
out_folder=./rollouts_train

python collect_rollout.py \
    --n_rollouts $n_rollouts \
    --start_seed $seed \
    --out_folder $out_folder \
    --n_particles $n_particles \
    --n_frames $n_frames 

n_particles=1900
n_frames=750
n_rollouts=30
seed=4758
out_folder=./rollouts_val

python collect_rollout.py \
    --n_rollouts $n_rollouts \
    --start_seed $seed \
    --out_folder $out_folder \
    --n_particles $n_particles \
    --n_frames $n_frames 


n_particles=1900
n_frames=750
n_rollouts=30
seed=29046270
out_folder=./rollouts_test

python collect_rollout.py \
    --n_rollouts $n_rollouts \
    --start_seed $seed \
    --out_folder $out_folder \
    --n_particles $n_particles \
    --n_frames $n_frames 
