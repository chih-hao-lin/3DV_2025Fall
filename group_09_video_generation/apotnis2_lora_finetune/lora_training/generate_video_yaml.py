import time, yaml, torch
from pathlib import Path
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

# ---- Load prompts (YAML is just a list of strings) ----
with open("prompts_test.yaml", "r") as f:
    prompts = yaml.safe_load(f)
if not isinstance(prompts, list) or not prompts:
    raise ValueError("prompts_test.yaml must be a YAML list of prompt strings.")

# ---- Fixed settings ----
NUM_FRAMES = 181
WIDTH, HEIGHT = 680, 384
SEED = 42
FPS = 15
QUALITY = 5
TILED = True

NEG = (
    "Vivid tones, overexposed, static, unclear details, subtitles, stylized, artwork, painting, still image, "
    "overall gray, worst quality, low quality, JPEG compression artifacts, ugly, defective, extra fingers, "
    "poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, motionless image, "
    "cluttered background, three legs, crowded background, walking backward."
)

# ---- Output directory ----
out_dir = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
out_dir.mkdir(parents=True, exist_ok=True)

# ---- Load model once ----
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[
            "models/Wan-AI/Wan2.2-TI2V-1.3B/diffusion_pytorch_model-00001-of-00003.safetensors",
            "models/Wan-AI/Wan2.2-TI2V-1.3B/diffusion_pytorch_model-00002-of-00003.safetensors",
            "models/Wan-AI/Wan2.2-TI2V-1.3B/diffusion_pytorch_model-00003-of-00003.safetensors",
        ]),
        ModelConfig(path="models/Wan-AI/Wan2.2-TI2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="models/Wan-AI/Wan2.2-TI2V-1.3B/Wan2.2_VAE.pth"),
    ]
)

# Optional: LoRA (comment out if not needed)
pipe.load_lora(pipe.dit, "models/train/Wan2.2-TI2V-1.3B_lora/epoch-5.safetensors", alpha=1)
# pipe.enable_vram_management() # disabled for faster inference in A100

# ---- Generate videos from prompts ----

if __name__ == "__main__":
    for i, prompt in enumerate(prompts, start=1):
        if not isinstance(prompt, str) or not prompt.strip():
            print(f"[SKIP] Item #{i} is not a non-empty string.")
            continue

        torch.manual_seed(SEED)
        print(f"[{i}/{len(prompts)}] Generating…")
        video = pipe(
            prompt=prompt.strip(),
            negative_prompt=NEG,
            num_frames=NUM_FRAMES,
            width=WIDTH,
            height=HEIGHT,
            seed=SEED,
            tiled=TILED,
        )

        out_path = out_dir / f"vid_{i:02d}.mp4"
        save_video(video, str(out_path), fps=FPS, quality=QUALITY)
        print(f"  saved -> {out_path}")

    print(f"\nDone. Outputs in: {out_dir}")