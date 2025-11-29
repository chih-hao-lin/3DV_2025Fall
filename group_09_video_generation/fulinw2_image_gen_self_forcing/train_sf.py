# train_lora_sd_sf_minimal.py
# Hacker Demo V-SD: (No Accelerate Version - Corrected + Minimal Self-Forcing GAN)

import argparse
import os
import math
import logging
from pathlib import Path
import random
import numpy as np

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

import diffusers
import transformers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
# [!] Use torch.amp for mixed precision
from torch.amp import autocast, GradScaler


# -----------------------------
# Perceptual Loss (optional)
# -----------------------------
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
        if x.min() < -0.1:
            x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return x

    def forward(self, input, target):
        self.mean = self.mean.to(input.device)
        self.std = self.std.to(input.device)
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        input = self.normalize(input)
        target = self.normalize(target)
        if self.resize and input.shape[2:] != (224, 224):
            input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)
        self.features = self.features.to(input.device)
        input_features = self.features(input)
        target_features = self.features(target)
        loss = F.l1_loss(input_features, target_features)
        return loss


# -----------------------------
# Minimal Discriminator for SF-GAN
# -----------------------------
class TinyDiscriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        def block(ci, co, k=4, s=2, p=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, kernel_size=k, stride=s, padding=p),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.net = nn.Sequential(
            block(in_ch, 64),    # 512 -> 256
            block(64, 128),      # 256 -> 128
            block(128, 256),     # 128 -> 64
            block(256, 512),     # 64  -> 32
            block(512, 512),     # 32  -> 16
        )
        self.head = nn.Linear(512, 1)

    def forward(self, x):
        # x in [-1,1], shape Bx3xHxW
        h = self.net(x)
        h = F.adaptive_avg_pool2d(h, 1).view(x.size(0), -1)
        logit = self.head(h)
        return logit


# -----------------------------
# Dataset
# -----------------------------
class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, tokenizer, size=512, center_crop_cat6=True):
        self.size = size
        self.center_crop_cat6 = center_crop_cat6
        self.tokenizer = tokenizer
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance data root doesn't exist.")
        self.instance_images_path = [
            p for p in Path(instance_data_root).iterdir()
            if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']
        ]
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.crop_cat6 = transforms.CenterCrop(size) if center_crop_cat6 else transforms.RandomCrop(size)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        filename = instance_image_path.name
        try:
            instance_image = Image.open(instance_image_path).convert("RGB")
        except Exception as e:
            raise e
        resized_image = self.resize(instance_image)
        if filename == "cat6.png":
            cropped_image = self.crop_cat6(resized_image)
        else:
            try:
                cropped_image = TF.crop(resized_image, top=0, left=0, height=self.size, width=self.size)
            except Exception as crop_error:
                print(f"Warning: Top crop failed for {filename}. Using center crop. Error: {crop_error}")
                cropped_image = transforms.CenterCrop(self.size)(resized_image)
        example["instance_images"] = self.normalize(self.to_tensor(cropped_image))
        txt_path = instance_image_path.with_suffix(".txt")
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                instance_prompt = f.read().strip()
        else:
            instance_prompt = "a photo in ccat style"
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return example


# -----------------------------
# Utils
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Diffusion schedule value extractor
def extract(a, t, x_shape):
    """
    Extract values from 1-D tensor 'a' by indices 't', reshape for broadcasting to 'x_shape'.
    """
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def generate_sample_image(epoch, args, tokenizer, text_encoder, vae, unet, device, weight_dtype):
    """Minimal stub to avoid errors during periodic saving.
    You can implement preview generation later if needed.
    """
    return


# -----------------------------
# Main
# -----------------------------

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # --- Load Models ---
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # --- Move models to device ---
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # --- Add LoRA Adapters ---
    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    logger.info("Added LoRA adapters to UNet.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # --- Optimizer ---
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = optim.AdamW(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- Load Dataset ---
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
    )

    # --- Noise Scheduler & Precomputed schedule values ---
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # --- Self-Forcing: ultra-short rollout scheduler ---
    sf_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    sf_scheduler.set_timesteps(args.sf_steps)
    sf_timesteps = sf_scheduler.timesteps.to(device)   # ✅ 用 device，而不是 latents.device

    # --- Perceptual Loss ---
    perceptual_loss_fn = None
    if args.loss_type == "perceptual":
        perceptual_loss_fn = VGGPerceptualLoss().to(device)
        perceptual_loss_fn.requires_grad_(False)
        logger.info("Using Perceptual Loss (extra regularizer).")
    elif args.loss_type == "sf_gan":
        logger.info("Using MSE + Minimal SF-GAN (short self-rollout + GAN).")
    else:
        logger.info("Using standard MSE Loss (TF/DF baseline).")

    # --- Scheduler ---
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # --- Precision Setup ---
    if args.mixed_precision == "fp16" and torch.cuda.is_available():
        weight_dtype = torch.float16
        compute_dtype = torch.float16
        scaler = torch.amp.GradScaler('cuda')
        logger.info("Using mixed precision: fp16")
    elif args.mixed_precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        weight_dtype = torch.bfloat16
        compute_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        logger.info("Using mixed precision: bf16 (no GradScaler needed)")
    else:
        weight_dtype = torch.float32
        compute_dtype = torch.float32
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        logger.info("Using precision: float32")
        args.mixed_precision = "no"

    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    # --- Discriminator (for SF-GAN) ---
    D = None
    optimizer_d = None
    if args.loss_type == "sf_gan":
        D = TinyDiscriminator(in_ch=3).to(device)
        optimizer_d = optim.AdamW(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
        logger.info("SF-GAN enabled: TinyDiscriminator initialized.")

    if args.mixed_precision != "no" and torch.cuda.is_available():
        scaler_d = torch.amp.GradScaler('cuda')
    else:
        scaler_d = torch.amp.GradScaler('cuda', enabled=False)

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
            optimizer.zero_grad(set_to_none=True)

            # --- Forward pass with autocast ---
            with torch.amp.autocast(device_type=device.type, dtype=compute_dtype, enabled=(args.mixed_precision != "no")):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)

                # VAE encode (float32), then cast latents to weight/computation dtype
                latents = vae.encode(pixel_values.to(torch.float32)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(input_ids)[0].to(device=device, dtype=compute_dtype)

                # Predict noise
                model_pred = unet(noisy_latents.to(dtype=compute_dtype), timesteps, encoder_hidden_states).sample

                # Diffusion training target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # MSE (float32 for stability)
                loss_diff = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss_diff
                current_loss_diff = loss_diff.detach().item()
                current_loss_perc = 0.0

                # Optional perceptual reg on x0
                if args.loss_type == "perceptual" and perceptual_loss_fn is not None:
                    sqrt_alpha_prod_t = extract(sqrt_alphas_cumprod, timesteps, noisy_latents.shape)
                    sqrt_one_minus_alpha_prod_t = extract(sqrt_one_minus_alphas_cumprod, timesteps, noisy_latents.shape)
                    pred_original_sample_latents = (noisy_latents.float() - sqrt_one_minus_alpha_prod_t * model_pred.float()) / sqrt_alpha_prod_t
                    pred_original_sample = vae.decode(pred_original_sample_latents / vae.config.scaling_factor, return_dict=False)[0]
                    target_image = batch["pixel_values"].to(device, dtype=torch.float32)
                    loss_perc = perceptual_loss_fn(pred_original_sample, target_image)
                    loss = loss + args.lambda_perceptual * loss_perc
                    current_loss_perc = loss_perc.detach().item()

                # ======== SF-GAN branch: self-rollout + distribution matching ========
                if args.loss_type == "sf_gan" and D is not None:
                    # (a) ultra-short self-rollout from noise; only last step keeps grad
                    bsz = latents.shape[0]
                    rollout_latent = torch.randn_like(latents).to(dtype=compute_dtype)

                    with torch.no_grad():
                        for t_ in sf_scheduler.timesteps[:-1]:
                            t_b = t_.repeat(bsz).to(rollout_latent.device)  # ✅ 移到 CUDA
                            e_t = unet(rollout_latent, t_b, encoder_hidden_states).sample
                            rollout_latent = sf_scheduler.step(e_t, t_, rollout_latent).prev_sample  # 这里的 t_ 传给 scheduler 无所谓设备

                    t_last = sf_scheduler.timesteps[-1]
                    t_b = t_last.repeat(bsz).to(rollout_latent.device)  # ✅ 移到 CUDA
                    e_t = unet(rollout_latent, t_b, encoder_hidden_states).sample

                    # 这里也要用 CUDA 的 t_b 来计算 sqrt_*，并确保结果在同一 device
                    sqrt_alpha_prod_t = extract(sqrt_alphas_cumprod, t_b, rollout_latent.shape).to(rollout_latent.device)
                    sqrt_one_minus_alpha_prod_t = extract(sqrt_one_minus_alphas_cumprod, t_b, rollout_latent.shape).to(rollout_latent.device)

                    pred_x0_latent = (rollout_latent.float() - sqrt_one_minus_alpha_prod_t * e_t.float()) / sqrt_alpha_prod_t
                    pred_x0 = vae.decode((pred_x0_latent / vae.config.scaling_factor).to(torch.float32), return_dict=False)[0]  # [-1,1]

                    # (b) shared pixel noise to both real and fake (cheap approximation of common noising)
                    real_img = batch["pixel_values"].to(device, dtype=torch.float32)
                    noise_pix = torch.randn_like(real_img) * args.sf_noise_std
                    real_noised = torch.clamp(real_img + noise_pix, -1.0, 1.0)
                    fake_noised = torch.clamp(pred_x0 + noise_pix, -1.0, 1.0)

                    # (c) Discriminator updates (non-saturating GAN + optional R1)
                    for _ in range(args.d_iters):
                        optimizer_d.zero_grad(set_to_none=True)
                        with torch.amp.autocast(device_type=device.type, dtype=compute_dtype, enabled=(args.mixed_precision != "no")):
                            d_real = D(real_noised.requires_grad_(args.d_r1_gamma > 0))
                            d_fake = D(fake_noised.detach())
                            d_loss = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
                        if args.d_r1_gamma > 0:
                            with torch.enable_grad():
                                grad_real = torch.autograd.grad(
                                    outputs=d_real.sum(), inputs=real_noised,
                                    create_graph=True, retain_graph=True, only_inputs=True
                                )[0]
                            r1_penalty = 0.5 * args.d_r1_gamma * grad_real.pow(2).view(bsz, -1).sum(dim=1).mean()
                            d_loss = d_loss + r1_penalty
                        d_loss_scaled = scaler_d.scale(d_loss / args.gradient_accumulation_steps)
                        d_loss_scaled.backward()

                    # (d) Generator GAN loss
                    with torch.amp.autocast(device_type=device.type, dtype=compute_dtype, enabled=(args.mixed_precision != "no")):
                        g_logit = D(fake_noised)
                        g_loss_gan = F.softplus(-g_logit).mean()
                    loss = loss + args.lambda_gan * g_loss_gan

            # Scale loss and backprop for generator (UNet-LoRA)
            loss_scaled = scaler.scale(loss / args.gradient_accumulation_steps)
            loss_scaled.backward()

            # Accumulate for logging
            epoch_train_loss += loss.detach().item() / args.gradient_accumulation_steps
            epoch_train_loss_diff += current_loss_diff / args.gradient_accumulation_steps
            epoch_train_loss_perc += current_loss_perc / args.gradient_accumulation_steps

            # Optimizer steps at accumulation boundary
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # G step
                scaler.unscale_(optimizer)
                params_to_clip = [p for p in unet.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                # D step (no lr scheduler for simplicity)
                if args.loss_type == "sf_gan" and optimizer_d is not None:
                    scaler_d.unscale_(optimizer_d)
                    scaler_d.step(optimizer_d)
                    scaler_d.update()

                progress_bar.update(1)
                global_step += 1

                # Logging
                if global_step % 20 == 0:
                    avg_epoch_loss = epoch_train_loss / (step + 1)
                    log_dict = {"loss": avg_epoch_loss}
                    if args.loss_type in ["perceptual", "sf_gan"]:
                        log_dict["loss_diff"] = epoch_train_loss_diff / (step + 1)
                        log_dict["loss_perc"] = epoch_train_loss_perc / (step + 1)
                    if args.loss_type == "sf_gan":
                        log_dict["lambda_gan"] = args.lambda_gan
                        log_dict["sf_steps"] = args.sf_steps
                    logger.info(f"Epoch: {epoch+1} Step: {global_step} - {log_dict}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # --- End of Epoch ---
        # Saving LoRA adapter every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == args.num_train_epochs - 1:
            epoch_num_str = str(epoch + 1).zfill(4)
            epoch_save_dir = os.path.join(args.output_dir, f"epoch_{epoch_num_str}")
            os.makedirs(epoch_save_dir, exist_ok=True)

            logger.info(f"Saving LoRA adapter for epoch {epoch+1}...")

            # 1) Only LoRA weights
            try:
                unet_lora_state_dict = get_peft_model_state_dict(unet)
                if not unet_lora_state_dict:
                    logger.error(f"Epoch {epoch+1}: get_peft_model_state_dict returned empty dict! Skip saving.")
                else:
                    from safetensors.torch import save_file
                    lora_weights_filename = os.path.join(epoch_save_dir, "adapter_model.safetensors")
                    save_file(unet_lora_state_dict, lora_weights_filename)
            except Exception as e:
                logger.error(f"Error saving LoRA state dict for epoch {epoch+1}: {e}")

            # 2) Save adapter_config.json
            try:
                if hasattr(unet, "peft_config") and unet.peft_config.get("default"):
                    unet.peft_config["default"].save_pretrained(epoch_save_dir)
                    logger.info(f"LoRA adapter config saved via peft_config for epoch {epoch+1}.")
                else:
                    import json
                    adapter_config = {
                        "base_model_name_or_path": args.pretrained_model_name_or_path,
                        "bias": "none",
                        "fan_in_fan_out": False,
                        "inference_mode": True,
                        "init_lora_weights": "gaussian",
                        "lora_alpha": args.lora_rank,
                        "lora_dropout": 0.0,
                        "modules_to_save": None,
                        "peft_type": "LORA",
                        "r": args.lora_rank,
                        "revision": None,
                        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                        "task_type": None,
                    }
                    with open(os.path.join(epoch_save_dir, "adapter_config.json"), 'w') as f:
                        json.dump(adapter_config, f, indent=2)
                    logger.info(f"Manually saved adapter_config.json for epoch {epoch+1}.")
            except Exception as config_e:
                logger.error(f"Error saving adapter_config.json for epoch {epoch+1}: {config_e}")

            # 3) Optional preview
            try:
                generate_sample_image(epoch, args, tokenizer, text_encoder, vae, unet, device, weight_dtype)
            except Exception as gen_e:
                logger.error(f"Error generating sample image for epoch {epoch+1}: {gen_e}")

        if global_step >= args.max_train_steps:
            break


# -----------------------------
# Dataloader Collate
# -----------------------------

def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    batch = {"input_ids": input_ids, "pixel_values": pixel_values}
    return batch


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--instance_data_dir", type=str, default="cyberpunk_cat_data")
    parser.add_argument("--output_dir", type=str, default="lora_sf_noaccel")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Loss type & weights
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "perceptual", "sf_gan"])
    parser.add_argument("--lambda_perceptual", type=float, default=0.1)

    # SF-GAN minimal knobs
    parser.add_argument("--sf_steps", type=int, default=4, help="self-rollout inference steps")
    parser.add_argument("--lambda_gan", type=float, default=0.3, help="G loss weight")
    parser.add_argument("--sf_noise_std", type=float, default=0.1, help="shared pixel noise std for D")
    parser.add_argument("--d_lr", type=float, default=5e-5)
    parser.add_argument("--d_r1_gamma", type=float, default=0.0, help="R1 GP on reals (0 to disable)")
    parser.add_argument("--d_iters", type=int, default=1)

    args = parser.parse_args()

    if args.max_train_steps is None and args.num_train_epochs is None:
        args.num_train_epochs = 100
        print(f"Defaulting to {args.num_train_epochs} epochs.")
    elif args.max_train_steps is not None:
        print(f"max_train_steps is set to {args.max_train_steps}. num_train_epochs will be calculated.")
    else:
        print(f"num_train_epochs is set to {args.num_train_epochs}. max_train_steps will be calculated.")

    main(args)
