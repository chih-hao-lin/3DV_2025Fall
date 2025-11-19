#!/usr/bin/env python
"""
Utility to convert real video assets into WAN latents and split train/test sets.

Steps performed:
1. Scan the provided video + prompt directories and align samples by stem.
2. Take a deterministic split (default 10%) as the held-out test set.
3. Copy test videos into a workspace folder and emit their prompts as a text file.
4. Encode the remaining training videos to VAE latents (shape: 21x16x60x104) and
   store them directly inside an LMDB dataset that matches the repository loaders.

Example:
    python scripts/prepare_real_dataset.py \
        --video-dir ~/video_gt \
        --asset-dir ~/video_gt_assets \
        --output-root data/real_ft \
        --test-ratio 0.1
"""
import argparse
import math
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import lmdb
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.lmdb import store_arrays_to_lmdb
from utils.wan_wrapper import WanVAEWrapper


_WORKER_VAE = None
_WORKER_DEVICE = None
_NUM_TARGET_FRAMES = 21


def list_video_stems(video_dir: Path) -> List[str]:
    return sorted(p.stem for p in video_dir.glob("*.mp4"))


def load_primary_prompt(asset_dir: Path, stem: str) -> str:
    prompt_path = asset_dir / stem / "prompt_0.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def load_video_frames(video_path: Path) -> np.ndarray:
    reader = imageio.get_reader(video_path.as_posix())
    frames = [frame for frame in reader]
    reader.close()
    if not frames:
        raise ValueError(f"No frames found in {video_path}")
    return np.stack(frames, axis=0)


def normalize_frames(
    frames: np.ndarray,
    num_target_frames: int = 21,
    target_hw: Tuple[int, int] = (480, 832),
) -> torch.Tensor:
    """Convert raw frames into (num_frames, C, H, W) tensor in [-1, 1]."""
    indices = np.linspace(0, len(frames) - 1, num_target_frames).astype(int)
    sampled = torch.from_numpy(frames[indices]).float() / 255.0  # (F, H, W, C)
    sampled = sampled.permute(0, 3, 1, 2)  # (F, C, H, W)
    sampled = F.interpolate(
        sampled,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    sampled = sampled * 2.0 - 1.0  # [-1, 1]
    return sampled  # (F, C, H, W)


def _ensure_vae(device: torch.device) -> WanVAEWrapper:
    vae = WanVAEWrapper().to(device)
    vae.eval()
    return vae


def encode_to_latents(
    frames: torch.Tensor,
    vae: WanVAEWrapper,
    device: torch.device,
) -> torch.Tensor:
    """Encode per-frame to keep 1 latent per video frame."""
    latents: List[torch.Tensor] = []
    for frame in frames:
        frame_btchw = frame.unsqueeze(0).unsqueeze(2).to(device=device)
        latent = vae.encode_to_latent(frame_btchw)
        latents.append(latent.cpu())
    return torch.cat(latents, dim=1)  # (1, num_frames, 16, 60, 104)


def write_test_prompts(prompts: Iterable[Tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for _, prompt in prompts:
            f.write(prompt.replace("\n", " ").strip())
            f.write("\n")


def prepare_lmdb(
    output_dir: Path,
    samples: Iterator[Tuple[np.ndarray, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env_path = output_dir / "shard_0"
    env_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(env_path),
        map_size=int(5e11),  # ~500 GB upper bound
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
    )
    latents_shape_tail = None
    counter = 0
    with env.begin(write=True) as txn:
        pass  # ensure environment created
    env.sync()

    for latent_np, prompt in samples:
        if latents_shape_tail is None:
            latents_shape_tail = latent_np.shape[1:]
        store_arrays_to_lmdb(
            env,
            {
                "latents": latent_np,
                "prompts": np.array([prompt]),
            },
            start_index=counter,
        )
        counter += 1

    if latents_shape_tail is None:
        raise RuntimeError("No training samples were encoded; LMDB is empty.")

    latents_shape = (counter, *latents_shape_tail)
    prompts_shape = (counter,)
    with env.begin(write=True) as txn:
        txn.put(
            b"latents_shape",
            " ".join(map(str, latents_shape)).encode(),
        )
        txn.put(
            b"prompts_shape",
            " ".join(map(str, prompts_shape)).encode(),
        )
    env.sync()
    env.close()


def _worker_init(use_gpu: bool, num_target_frames: int):
    global _WORKER_DEVICE, _WORKER_VAE, _NUM_TARGET_FRAMES
    _WORKER_DEVICE = torch.device(
        "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    )
    _WORKER_VAE = _ensure_vae(_WORKER_DEVICE)
    _NUM_TARGET_FRAMES = num_target_frames


def _encode_worker(args):
    stem, video_path, prompt_text = args
    frames = load_video_frames(video_path)
    frames_tensor = normalize_frames(frames, num_target_frames=_NUM_TARGET_FRAMES)
    latents = encode_to_latents(frames_tensor, _WORKER_VAE, _WORKER_DEVICE)
    return stem, latents.half().numpy(), prompt_text


def main():
    parser = argparse.ArgumentParser(description="Prepare real video dataset.")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/real_ft"),
        help="Root for generated artefacts (LMDB, manifests, etc.).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of videos reserved for the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    parser.add_argument(
        "--test-prompt-file",
        type=Path,
        default=Path("prompts/real_ft_test_prompts.txt"),
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=Path("prompts/real_ft_test_manifest.tsv"),
    )
    parser.add_argument(
        "--test-video-dir",
        type=Path,
        default=Path("test_videos"),
        help="Directory to store held-out test videos.",
    )
    parser.add_argument(
        "--test-input-dir",
        type=Path,
        default=None,
        help="Optional directory containing pre-selected test videos.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on number of training videos to process.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers for VAE encoding (each loads its own VAE).",
    )
    parser.add_argument(
        "--use-gpu-workers",
        action="store_true",
        help="Let workers try to place VAE on GPU if available.",
    )
    parser.add_argument(
        "--num-target-frames",
        type=int,
        default=21,
        help="Number of frames to sample per video before VAE encoding.",
    )
    args = parser.parse_args()

    video_dir = args.video_dir.expanduser().resolve()
    asset_dir = args.asset_dir.expanduser().resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stems = list_video_stems(video_dir)
    if not stems:
        raise RuntimeError(f"No mp4 videos found in {video_dir}")

    if args.test_input_dir:
        test_input_dir = args.test_input_dir.expanduser().resolve()
        test_stems = set(list_video_stems(test_input_dir))
        train_stems = [stem for stem in stems if stem not in test_stems]
    else:
        if args.test_ratio > 0:
            rng = random.Random(args.seed)
            test_size = max(1, int(math.ceil(len(stems) * args.test_ratio)))
            test_stems = set(rng.sample(stems, test_size))
        else:
            test_stems = set()
        train_stems = [stem for stem in stems if stem not in test_stems]

    print(f"Discovered {len(stems)} videos in training directory.")
    print(f"Assigning {len(train_stems)} to train and {len(test_stems)} to test.")

    if args.max_train_samples is not None:
        train_stems = train_stems[: args.max_train_samples]

    train_samples: List[Tuple[str, Path, str]] = []

    test_prompt_records: List[Tuple[str, Path]] = []
    test_prompts: List[Tuple[str, str]] = []
    test_video_dir = (Path("Self-Forcing") / args.test_video_dir).resolve()
    test_video_dir.mkdir(parents=True, exist_ok=True)

    for stem in tqdm(train_stems, desc="Processing train videos"):
        video_path = video_dir / f"{stem}.mp4"
        prompt_text = load_primary_prompt(asset_dir, stem)
        train_samples.append((stem, video_path, prompt_text))

    if test_stems:
        test_input_dir = (
            args.test_input_dir.expanduser().resolve()
            if args.test_input_dir
            else video_dir
        )
        for stem in tqdm(sorted(test_stems), desc="Preparing test videos"):
            source_path = test_input_dir / f"{stem}.mp4"
            if not source_path.exists():
                raise FileNotFoundError(f"Test video not found: {source_path}")
            destination = test_video_dir / f"{stem}.mp4"
            if not destination.exists():
                shutil.copy2(source_path, destination)
            asset_prompt_path = asset_dir / stem / "prompt_0.txt"
            test_prompt_records.append((stem, asset_prompt_path.resolve()))
            prompt_text = load_primary_prompt(asset_dir, stem)
            test_prompts.append((stem, prompt_text))

    def iter_training_samples() -> Iterator[Tuple[np.ndarray, str]]:
        if not train_samples:
            return iter(())

        if args.num_workers <= 1:
            device = torch.device(
                "cuda" if args.use_gpu_workers and torch.cuda.is_available() else "cpu"
            )
            vae = _ensure_vae(device)

            def generator():
                for _, video_path, prompt_text in tqdm(
                    train_samples, desc="Encoding train videos", unit="video"
                ):
                    frames = load_video_frames(video_path)
                    frames_tensor = normalize_frames(
                        frames, num_target_frames=args.num_target_frames
                    )
                    latents = encode_to_latents(frames_tensor, vae, device)
                    yield latents.half().numpy(), prompt_text

            return generator()

        def parallel_generator():
            with ProcessPoolExecutor(
                max_workers=args.num_workers,
                initializer=_worker_init,
                initargs=(args.use_gpu_workers, args.num_target_frames),
            ) as executor:
                futures = {
                    executor.submit(_encode_worker, sample): sample[0]
                    for sample in train_samples
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Encoding train videos",
                    unit="video",
                ):
                    stem, latent_np, prompt_text = future.result()
                    yield latent_np, prompt_text

        return parallel_generator()

    samples_iter = iter_training_samples()

    # Persist training latents into LMDB.
    lmdb_dir = output_root / "lmdb"
    prepare_lmdb(lmdb_dir, samples_iter)

    # Emit simple manifest for reproducibility.
    manifest_path = output_root / "train_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("stem\tprompt_path\n")
        for stem, _, _ in train_samples:
            prompt_path = asset_dir / stem / "prompt_0.txt"
            f.write(f"{stem}\t{prompt_path.resolve()}\n")

    # Write test prompts.
    test_prompt_file = (Path("Self-Forcing") / args.test_prompt_file).resolve()
    test_prompt_file.parent.mkdir(parents=True, exist_ok=True)
    write_test_prompts(test_prompts, test_prompt_file)

    test_manifest = (Path("Self-Forcing") / args.test_manifest).resolve()
    test_manifest.parent.mkdir(parents=True, exist_ok=True)
    with test_manifest.open("w", encoding="utf-8") as f:
        f.write("stem\tprompt_path\tvideo_path\n")
        for stem, prompt_path in test_prompt_records:
            video_path = test_video_dir / f"{stem}.mp4"
            f.write(f"{stem}\t{prompt_path}\t{video_path}\n")

    print(f"Training LMDB written to: {lmdb_dir}")
    print(f"Test prompts stored at: {test_prompt_file}")
    print(f"Held-out videos copied to: {test_video_dir}")


if __name__ == "__main__":
    main()
