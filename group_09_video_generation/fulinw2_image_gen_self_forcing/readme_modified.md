### Hacker Task: Self-forcing for Image Generation

#### 1. Env setup

```
conda create -n image_sf python=3.10 -y
conda activate image_sf
pip install -r requirements.txt
# runwayml/stable-diffusion-v1-5 is automatically downloaded with the train script, no hugging-face cli download is required.
```

#### 2. Checkpoints/Results

- The base model is automatically downloaded
- The fine-tuned checkpoints with LoRA_rank = 8 for MSE (baseline TF method), perceptual (baseline TF method + perceptual loss) and SF method is stored under ``lora_mse``, ``lora_perceptual``, ``lora_sf`` respectively. 
- For simplicity, we only provide the final used model.

#### 3. Dataset and Evaluation Metrics

The dataset for fine-tuning is conducted with dataset collected from lexica. This site contains cyberpunk-style cat images and corresponding prompts.

The dataset is stored under ``cyberpunk_cat_data``.

Evaluation Metrics: CLIP score openai/clip-vit-large-patch14 to measure the similarity between image and its caption. 

#### 4. Core scripts usage

(1) Base model LoRA fine-tune

TF:

```
python train.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --instance_data_dir="cyberpunk_cat_data" --output_dir="lora_mse" --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=1 --learning_rate=1e-4 --max_train_steps=500 --lora_rank=4 --mixed_precision="fp16" --loss_type="mse" --seed=42
```

Perceptual:

```
python train.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --instance_data_dir="cyberpunk_cat_data" --output_dir="lora_perceptual" --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=1 --learning_rate=1e-4 --max_train_steps=500 --lora_rank=4 --mixed_precision="fp16" --loss_type="perceptual" --lambda_perceptual 0.1 --seed=42
```

SF:

```
python train_sf.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  --instance_data_dir="cyberpunk_cat_data" --output_dir="lora_perceptual" --resolution=512 --train_batch_size=4 --gradient_accumulation_steps=1 --learning_rate=1e-4 --max_train_steps=500 --lora_rank=4 --mixed_precision="fp16" --loss_type="sf_gan" --lambda_perceptual 0.1 --sf_steps 4 --lambda_gan 0.3 --sf_noise_std 0.1 --d_lr 5e-5 --d_r1_gamma 0 --d_iters 1 --seed=42
```

(2) LoRA checkpoint inference (the image results are generated in this step)

TF:

```
python infer.py   --base_model_path="runwayml/stable-diffusion-v1-5"   --lora_path="./lora_mse/epoch_167" --output_dir="results/mse"   --lora_epoch=167   --seed=1234   --prompts     "close up portrait of a cat with visible cybernetic eye implant, intricate mechanical details, studio lighting"     "photo of a sleek robotic cat, polished chrome finish, glowing blue optical sensors, dark background"     "a cat with metallic paws and reinforced spine, detailed micro-circuitry visible, standing on a metal grating"     "cyborg cat face, half biological fur, half smooth metal plating, intricate wiring, sharp focus"     "illustration of a cat wearing powered exoskeleton armor, detailed joints and hydraulics, futuristic design"     "a cat sitting on a wet neon-lit street, reflections in puddles, cyberpunk city alley background"     "photo of a cat looking out a window at a dense futuristic cityscape with towering neon skyscrapers"     "a cat perched on a rooftop antenna, vast neon city below, dramatic night lighting"     "cat walking through a crowded market in a cyberpunk city, holographic advertisements in the background"     "a cat hiding under a glowing neon sign, steam rising from vents, dark alleyway"     "cinematic shot of a robotic cat with glowing red eyes, stalking through a neon-drenched Chinatown street"     "highly detailed digital painting of a cyborg cat wearing a futuristic visor, intricate wiring visible, bokeh background"     "photorealistic image of a cat with subtle mechanical enhancements on its ears and tail, sitting by a neon bar sign"     "a small robotic kitten exploring a cluttered electronics workshop, glowing LED components scattered around"     "concept art of a heavily augmented cybernetic cat, large mechanical limbs, menacing pose, dramatic lighting"     "robotic cat, neon lights"     "cyberpunk cat, detailed fur and metal"     "cat with glowing eyes, futuristic city"     "photo, cyborg feline, neon glow" \
```

Perceptual:

```
python infer.py   --base_model_path="runwayml/stable-diffusion-v1-5"   --lora_path="./lora_perceptual/epoch_167" --output_dir="results/percetual"   --lora_epoch=167   --seed=1234   --prompts xxxxxx (same as above)
```

SF:

```
python infer.py   --base_model_path="runwayml/stable-diffusion-v1-5"   --lora_path="./lora_sf/epoch_167" --output_dir="results/sf"   --lora_epoch=167   --seed=1234   --prompts xxxxxx (same as above)
```

(3) Result evaluation

```
python eval_clip.py --image_dir ./results/mse --prompts_file prompt2.txt
python eval_clip.py --image_dir ./results/perceptual --prompts_file prompt2.txt
python eval_clip.py --image_dir ./results/sf --prompts_file prompt2.txt
```

