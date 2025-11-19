# Proj: 3D Consistency Refinement

---

## 1. Environment Setup

   ```bash
   conda create -n self_forcing python=3.10 -y
   conda activate self_forcing
   pip install -r requirements.txt
   pip install flash-attn --no-build-isolation
   python setup.py develop
   huggingface-cli login
   ```

---

## 2. Checkpoints & Assets

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False \
  --local-dir wan_models/Wan2.1-T2V-1.3B

huggingface-cli download gdhe17/Self-Forcing \
  checkpoints/self_forcing_dmd.pt --local-dir checkpoints

huggingface-cli download gdhe17/Self-Forcing \
  checkpoints/ode_init.pt --local-dir checkpoints
```

---

## 3. Data Preparation
We curated ~60 short product-turntable clips (`video_gt/*.mp4`) plus text assets (`video_gt_assets/<stem>/prompt_0.txt`). Use the helper to align, split, and encode them into LMDB latents.

```bash
python scripts/prepare_real_dataset.py \
  --video-dir video_gt \
  --asset-dir video_gt_assets \
  --output-root data/real_ft \
  --test-ratio 0.2 \
  --num-workers 4 \
  --use-gpu-workers
```

Outputs:
- `data/real_ft/lmdb/` – Wan-VAE latents for training.
- `prompts/real_ft_train_prompts.txt` – text file referenced by our fine-tuning config.
- `prompts/real_ft_test_prompts.txt` + `prompts/real_ft_test_manifest.tsv` – prompts + metadata for the held-out evaluation split.
- `test_videos/` – copies of the held-out `.mp4` files for qualitative checks.

If you already have a pre-defined evaluation subset, pass it via `--test-input-dir` and set `--test-ratio 0`.

---

## 4. Running Things

```bash
torchrun --nproc_per_node=4 \
  train.py \
  --config_path configs/self_forcing_dmd_subset.yaml \
  --logdir logs/self_forcing_dmd_subset \
  --disable-wandb
```
