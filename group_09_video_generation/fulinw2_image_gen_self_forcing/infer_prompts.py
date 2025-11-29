# infer_lora.py
# Loads a base Stable Diffusion model and a trained LoRA adapter,
# then generates images based on text prompts.

import argparse
import os
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

def assert_lora_active(pipe):
    # 先判断走的是 diffusers LoRA 还是 PEFT LoRA
    used_peft = hasattr(pipe.unet, "peft_config") and pipe.unet.peft_config.get("default") is not None
    if not used_peft:
        # diffusers 路线：attn_processors 里应是 LoRAAttnProcessor*
        any_lora = any("Lora" in p.__class__.__name__ for p in pipe.unet.attn_processors.values())
        if not any_lora:
            raise RuntimeError("LoRA not active: UNet attn_processors do not include LoRA processors.")
    else:
        # PEFT 路线：统计 lora_* 权重范数
        total = 0.0
        for n, p in pipe.unet.named_parameters():
            if "lora_" in n and p is not None:
                total += float(p.detach().abs().mean())
        if total == 0.0:
            raise RuntimeError("LoRA not active: PEFT lora_* params all zero/absent.")
    print("[check] LoRA appears active.")


def main(args):
    # --- 0. Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16 # Use float16 for GPU inference
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32 # CPU requires float32
    print(f"Using device: {device}, dtype: {torch_dtype}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a consistent generator for reproducibility if needed
    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
        print(f"Using seed: {args.seed}")
    else:
        print("Using random seed.")


    # --- 1. Load Base Stable Diffusion Pipeline ---
    print(f"Loading base model: {args.base_model_path}...")
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.base_model_path,
            torch_dtype=torch_dtype,
            safety_checker=None # Disable safety checker if desired
        )
        pipeline.to(device)
        print("Base model loaded.")
    except Exception as e:
        print(f"Error loading base model: {e}"); exit()

    # --- 2. Load LoRA Weights ---
    lora_dir = Path(args.lora_path)
    print(f"\nAttempting to load LoRA weights from directory: {lora_dir}")

    # Check required files exist (adapter_config.json and adapter_model file)
    config_file = lora_dir / "adapter_config.json"
    weights_file_st = lora_dir / "adapter_model.safetensors"
    weights_file_bin = lora_dir / "adapter_model.bin" # Fallback

    if not config_file.exists(): print(f"ERROR: adapter_config.json not found in {lora_dir}"); exit()
    weights_path = None
    if weights_file_st.exists(): weights_path = weights_file_st; print("Found adapter_model.safetensors")
    elif weights_file_bin.exists(): weights_path = weights_file_bin; print("Found adapter_model.bin")
    else: print(f"ERROR: Neither adapter_model.safetensors nor adapter_model.bin found in {lora_dir}"); exit()

    try:
        # --- Load using the standard diffusers/PEFT method ---
        # It expects the directory containing config and weights
        pipeline.load_lora_weights(
             lora_dir,
             weight_name=weights_path.name # Explicitly provide filename found
        )
        print(f"Successfully loaded LoRA weights from {weights_path.name}")
        assert_lora_active(pipeline)
        # --- Optional: Fuse LoRA for potential speedup (cannot unload after) ---
        if args.fuse_lora:
            pipeline.fuse_lora()
            print("Fused LoRA weights into UNet.")

        # --- Optional: Set adapter explicitly (usually not needed if loaded correctly) ---
        # pipeline.unet.set_adapter('default')

    except Exception as e:
        print(f"ERROR during pipeline.load_lora_weights: {e}")
        print("LoRA weights might be incompatible, corrupted, or PEFT/Diffusers versions may mismatch.")
        exit() # Exit if LoRA fails to load

    # --- 3. Generation ---
    print(f"\nGenerating {len(args.prompts) * args.num_images_per_prompt} images...")
    start_time = time.time()
    
    image_count = 0
    for i, prompt in enumerate(args.prompts):
        print(f"  Prompt {i+1}/{len(args.prompts)}: '{prompt}'")
        try:
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch_dtype, enabled=(torch_dtype==torch.float16)):
                images = pipeline(
                    prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=args.num_images_per_prompt,
                    generator=generator,
                ).images

            # Save generated images
            for j, img in enumerate(images):
                image_count += 1
                # Create a safe filename
                safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:50]
                # Include LoRA name/epoch in filename for clarity
                lora_name = lora_dir.name # e.g., "epoch_0167"
                filename = output_dir / f"{lora_name}_prompt_{i+1}_{safe_prompt}_seed_{args.seed}_{j+1}.png"
                img.save(filename)
                print(f"    Saved: {filename}")

        except Exception as e:
            print(f"    Error generating image for prompt '{prompt}': {e}")
            continue # Continue to the next prompt

    end_time = time.time()
    print(f"\nGeneration complete. Saved {image_count} images in {end_time - start_time:.2f} seconds.")

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion with specified LoRA weights.")

    parser.add_argument("--base_model_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Base Stable Diffusion model ID.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the DIRECTORY containing adapter_config.json and adapter_model.safetensors/bin.")
    parser.add_argument("--output_dir", type=str, default="generated_lora_images", help="Directory to save generated images.")
    parser.add_argument("--prompts", type=str, nargs='+', required=True, help="One or more text prompts.")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, noisy, deformed, text, words, bad anatomy", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation (default: random).")
    parser.add_argument("--num_steps", type=int, default=30, help="Number of diffusion inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (CFG).")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images to generate for each prompt.")
    parser.add_argument("--fuse_lora", action="store_true", help="Fuse LoRA weights into the UNet for potentially faster inference (cannot unload).")

    args = parser.parse_args()
    main(args)