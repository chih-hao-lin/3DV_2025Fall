# train_lora_sd_no_accel_v_final.py
# Hacker Demo V-SD: (No Accelerate Version - Corrected)

import argparse
import os
import math
import logging
from pathlib import Path
import random
import numpy as np # Import numpy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path

import diffusers
import transformers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from PIL import Image
# [!] Use torch.amp for mixed precision
from torch.amp import autocast, GradScaler 



# --- Perceptual Loss Implementation ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features
        self.features = nn.Sequential(*list(features[:17])).eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        if x.min() < -0.1: x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return x

    def forward(self, input, target):
        # Move mean and std to the correct device
        self.mean = self.mean.to(input.device)
        self.std = self.std.to(input.device)

        if input.shape[1] == 1: input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1: target = target.repeat(1, 3, 1, 1)
        input = self.normalize(input); target = self.normalize(target)
        if self.resize and input.shape[2:] != (224, 224):
            input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        # Move features model to the correct device if needed (should happen on init)
        self.features = self.features.to(input.device)
        input_features = self.features(input); target_features = self.features(target)
        loss = F.l1_loss(input_features, target_features)
        return loss

# --- Dataset Definition ---
class DreamBoothDataset(Dataset):
    def __init__(
        self, instance_data_root, tokenizer, size=512, center_crop_cat6=True,
    ):
        self.size = size; self.center_crop_cat6 = center_crop_cat6; self.tokenizer = tokenizer
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists(): raise ValueError("Instance data root doesn't exist.")
        self.instance_images_path = [p for p in Path(instance_data_root).iterdir() if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        self.num_instance_images = len(self.instance_images_path); self._length = self.num_instance_images
        self.to_tensor = transforms.ToTensor(); self.normalize = transforms.Normalize([0.5], [0.5])
        self.resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.crop_cat6 = transforms.CenterCrop(size) if center_crop_cat6 else transforms.RandomCrop(size)

    def __len__(self): return self._length

    def __getitem__(self, index):
        example = {}
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        filename = instance_image_path.name
        try: instance_image = Image.open(instance_image_path).convert("RGB")
        except Exception as e: raise e
        resized_image = self.resize(instance_image)
        if filename == "cat6.png": cropped_image = self.crop_cat6(resized_image)
        else:
            try: cropped_image = TF.crop(resized_image, top=0, left=0, height=self.size, width=self.size)
            except Exception as crop_error:
                 print(f"Warning: Top crop failed for {filename}. Using center crop. Error: {crop_error}")
                 cropped_image = transforms.CenterCrop(self.size)(resized_image)
        example["instance_images"] = self.normalize(self.to_tensor(cropped_image))
        txt_path = instance_image_path.with_suffix(".txt")
        if txt_path.exists():
             with open(txt_path, 'r') as f: instance_prompt = f.read().strip()
        else: instance_prompt = "a photo in ccat style"
        example["instance_prompt_ids"] = self.tokenizer(instance_prompt, truncation=True, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt",).input_ids
        return example

# --- Main Training Function ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- [!] Utility: Diffusion Schedule Value Extractor ---
def extract(a, t, x_shape):
    """
    Extracts values from a 1-D tensor 'a' based on indices in 't',
    and reshapes them to match the dimensions of 'x_shape' for broadcasting.
    """
    batch_size = t.shape[0]
    # Ensure 'a' is on the same device as 't' before gathering
    out = a.to(t.device).gather(0, t) 
    # Reshape to (batch_size, 1, 1, ...) to match x_shape except batch dim
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available(): device = torch.device("mps") # Uncomment for Mac Metal
    else: device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # --- Load Models ---
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # --- Move models to device ---
    vae.to(device); text_encoder.to(device); unet.to(device)
    vae.requires_grad_(False); text_encoder.requires_grad_(False)

    # --- Add LoRA Adapters ---
    unet_lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    logger.info("Added LoRA adapters to UNet.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # --- Optimizer ---
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = optim.AdamW(
        lora_layers, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    # --- Load Dataset ---
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir, tokenizer=tokenizer, size=args.resolution,
        # center_crop_cat6=args.cat6_center_crop # Assuming default True is okay
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=4, pin_memory=torch.cuda.is_available()
    )

    # --- Noise Scheduler & Schedule Values ---
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # [!] Get schedule values needed later
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # --- Perceptual Loss ---
    perceptual_loss_fn = None
    if args.loss_type == "perceptual":
        perceptual_loss_fn = VGGPerceptualLoss().to(device)
        perceptual_loss_fn.requires_grad_(False)
        logger.info("Using Perceptual Loss (SF Analogy).")
    else:
        logger.info("Using standard MSE Loss (TF/DF Baseline).")

    # --- Scheduler ---
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # --- Precision Setup ---
    if args.mixed_precision == "fp16" and torch.cuda.is_available():
        weight_dtype = torch.float16
        compute_dtype = torch.float16 # Use float16 for autocast
        scaler = torch.amp.GradScaler('cuda') # Use updated scaler
        logger.info("Using mixed precision: fp16")
    elif args.mixed_precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        weight_dtype = torch.bfloat16
        compute_dtype = torch.bfloat16 # Use bfloat16 for autocast
        # BF16 typically doesn't need a scaler
        scaler = torch.amp.GradScaler('cuda', enabled=False) # Scaler created but disabled
        logger.info("Using mixed precision: bf16 (no GradScaler needed)")
    else:
        weight_dtype = torch.float32
        compute_dtype = torch.float32 # Use float32 for autocast
        scaler = torch.amp.GradScaler('cuda', enabled=False) # Scaler created but disabled
        logger.info("Using precision: float32")
        args.mixed_precision = "no"

    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    # --- Training Loop Setup ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps))
    progress_bar.set_description("Steps")

    # --- Actual Training Loop ---
    for epoch in range(args.num_train_epochs):
        unet.train()
        epoch_train_loss = 0.0
        epoch_train_loss_diff = 0.0
        epoch_train_loss_perc = 0.0

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True) # Zero grad at the start of accumulation

            # --- Forward pass with autocast ---
            # [!] Use updated autocast call with device_type and dtype
            with torch.amp.autocast(device_type=device.type, dtype=compute_dtype, enabled=(args.mixed_precision != "no")):
                pixel_values = batch["pixel_values"].to(device) # Keep pixel values in float32 for VAE initially
                input_ids = batch["input_ids"].to(device)

                # VAE Encode needs float32 input usually
                latents = vae.encode(pixel_values.to(torch.float32)).latent_dist.sample() 
                latents = latents * vae.config.scaling_factor # Now latents are likely float32
                latents = latents.to(dtype=weight_dtype) # Cast latents to working precision

                noise = torch.randn_like(latents) # Noise should match latent precision
                bsz = latents.shape[0]

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                # add_noise expects latents and noise to have same dtype
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Ensure encoder_hidden_states has the compute_dtype for the UNet
                encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=compute_dtype)

                # Predict noise - UNet expects compute_dtype inputs
                model_pred = unet(noisy_latents.to(dtype=compute_dtype), timesteps, encoder_hidden_states).sample

                # Calculate Loss
                if noise_scheduler.config.prediction_type == "epsilon": target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction": target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else: raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Calculate losses in float32 for stability
                loss_diff = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss_diff
                current_loss_diff = loss_diff.detach().item()

                current_loss_perc = 0.0
                if args.loss_type == "perceptual" and perceptual_loss_fn is not None:
                    # Calculate x0 for perceptual loss
                    # Use the globally defined/accessed schedule values and the extract function
                    sqrt_alpha_prod_t = extract(sqrt_alphas_cumprod, timesteps, noisy_latents.shape)
                    sqrt_one_minus_alpha_prod_t = extract(sqrt_one_minus_alphas_cumprod, timesteps, noisy_latents.shape)
                    
                    # Ensure calculation happens in float32 before VAE decode
                    pred_original_sample_latents = (noisy_latents.float() - sqrt_one_minus_alpha_prod_t * model_pred.float()) / sqrt_alpha_prod_t
                    
                    # VAE decode needs float32
                    pred_original_sample = vae.decode(pred_original_sample_latents / vae.config.scaling_factor, return_dict=False)[0]
                    target_image = batch["pixel_values"].to(device, dtype=torch.float32)

                    loss_perc = perceptual_loss_fn(pred_original_sample, target_image) # Already float32
                    loss = loss_diff + args.lambda_perceptual * loss_perc
                    current_loss_perc = loss_perc.detach().item()
                # End of autocast block

            # Scale loss
            loss_scaled = scaler.scale(loss / args.gradient_accumulation_steps)
            loss_scaled.backward() # Backward pass on scaled loss

            # Accumulate step losses for logging average
            epoch_train_loss += loss.detach().item() / args.gradient_accumulation_steps
            epoch_train_loss_diff += current_loss_diff / args.gradient_accumulation_steps
            epoch_train_loss_perc += current_loss_perc / args.gradient_accumulation_steps

            # Perform optimizer step after accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                params_to_clip = [p for p in unet.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                # optimizer.zero_grad was done at the start

                progress_bar.update(1)
                global_step += 1

                # Logging
                if global_step % 20 == 0: # Log less frequently
                    avg_epoch_loss = epoch_train_loss / (step + 1) # Avg loss so far this epoch
                    log_dict = {"loss": avg_epoch_loss}
                    if args.loss_type == "perceptual":
                        log_dict["loss_diff"] = epoch_train_loss_diff / (step + 1)
                        log_dict["loss_perc"] = epoch_train_loss_perc / (step + 1)
                    logger.info(f"Epoch: {epoch+1} Step: {global_step} - {log_dict}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        # --- End of Epoch ---
        # Manual saving

        # [!] Manual saving using PEFT's method
        # --- End of Epoch ---
        # [!!!] V-FINAL CORRECTED SAVING LOGIC [!!!]
        if (epoch + 1) % 5 == 0 or epoch == args.num_train_epochs - 1:
            epoch_num_str = str(epoch+1).zfill(4)
            epoch_save_dir = os.path.join(args.output_dir, f"epoch_{epoch_num_str}")
            os.makedirs(epoch_save_dir, exist_ok=True)

            logger.info(f"Saving LoRA adapter for epoch {epoch+1}...")

            # --- 1. Get ONLY LoRA Weights ---
            try:
                # Use PEFT utility to extract only adapter weights
                unet_lora_state_dict = get_peft_model_state_dict(unet)
                
                # Check if dict is empty (might happen if PEFT state is wrong)
                if not unet_lora_state_dict:
                     logger.error(f"Epoch {epoch+1}: get_peft_model_state_dict returned an empty dictionary! Cannot save LoRA weights.")
                     continue # Skip saving for this epoch

                lora_weights_filename = os.path.join(epoch_save_dir, "adapter_model.safetensors")
                
                # --- 2. Save LoRA Weights using safetensors ---
                from safetensors.torch import save_file # Ensure import
                save_file(unet_lora_state_dict, lora_weights_filename)

            except Exception as e:
                 logger.error(f"Error getting or saving LoRA state dict for epoch {epoch+1}: {e}")
                 continue # Skip saving

            # --- 3. Save the CORRECT adapter_config.json ---
            try:
                 # Option A: Try saving config via PEFT model attribute (if available)
                 if hasattr(unet, "peft_config") and unet.peft_config.get("default"): # Check if 'default' adapter exists
                     unet.peft_config["default"].save_pretrained(epoch_save_dir)
                     logger.info(f"LoRA adapter config saved via peft_config for epoch {epoch+1} at {epoch_save_dir}")
                 # Option B: Manually create if Option A fails (as fallback)
                 elif not os.path.exists(os.path.join(epoch_save_dir, "adapter_config.json")):
                     logger.warning(f"Epoch {epoch+1}: Could not save config via peft_config. Saving manually.")
                     adapter_config = {
                        "base_model_name_or_path": args.pretrained_model_name_or_path,
                        "bias": "none", "fan_in_fan_out": False, "inference_mode": True,
                        "init_lora_weights": "gaussian", # Should match LoraConfig
                        "lora_alpha": args.lora_rank, # Assuming alpha == rank
                        "lora_dropout": 0.0,
                        "modules_to_save": None, "peft_type": "LORA",
                        "r": args.lora_rank, "revision": None,
                        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"], # Should match LoraConfig
                        "task_type": None
                     }
                     config_filename = os.path.join(epoch_save_dir, "adapter_config.json")
                     import json
                     with open(config_filename, 'w') as f:
                         json.dump(adapter_config, f, indent=2)
                     logger.info(f"Manually saved adapter_config.json for epoch {epoch+1}")
                 else:
                      logger.info(f"adapter_config.json already exists for epoch {epoch+1} (likely saved by option A).")


            except Exception as config_e:
                 logger.error(f"Error saving adapter_config.json for epoch {epoch+1}: {config_e}")

            # --- 4. Generate Sample Image (Keep this) ---
            try:
                generate_sample_image(epoch, args, tokenizer, text_encoder, vae, unet, device, weight_dtype)
            except Exception as gen_e:
                 logger.error(f"Error generating sample image for epoch {epoch+1}: {gen_e}")


        if global_step >= args.max_train_steps:
             break # Exit outer loop too

# --- Helper for DataLoader ---
def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float() # Keep as float32 for VAE encode initially
    batch = {"input_ids": input_ids, "pixel_values": pixel_values,}
    return batch


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (Copy all arguments from the previous version) ...
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--instance_data_dir", type=str, default="cyberpunk_cat_data")
    parser.add_argument("--output_dir", type=str, default="lora_output_noaccel")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "perceptual"])
    parser.add_argument("--lambda_perceptual", type=float, default=0.1)
    # --- End arguments ---
    args = parser.parse_args()

    if args.max_train_steps is None and args.num_train_epochs is None:
        args.num_train_epochs = 100
        print(f"Defaulting to {args.num_train_epochs} epochs.")
    elif args.max_train_steps is not None:
         print(f"max_train_steps is set to {args.max_train_steps}. num_train_epochs will be calculated.")
    else:
         print(f"num_train_epochs is set to {args.num_train_epochs}. max_train_steps will be calculated.")
    main(args)