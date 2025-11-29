# eval_clip_direct_final.py

import argparse
import os
import torch
import transformers
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re # Keep re import

# --- Helper Functions (load_image, calculate_clip_similarity remain the same) ---
def load_image(image_path):
    try: img = Image.open(image_path).convert("RGB"); return img
    except Exception as e: print(f"Warning: Could not load image {image_path}. Skipping. Error: {e}"); return None

def calculate_clip_similarity(model, processor, images_pil, texts, device):
    try:
        inputs = processor(text=texts, images=images_pil, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
            cosine_sim = torch.diag(torch.matmul(image_embeds, text_embeds.t())).cpu().numpy()
            return cosine_sim
    except Exception as e: print(f"Error calculating CLIP similarity: {e}"); return None

# --- Main Logic ---
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {device}")
    try: # --- Load Prompts ---
        with open(args.prompts_file, 'r') as f: prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    except FileNotFoundError: print(f"Error: Prompts file not found at {args.prompts_file}"); return
    # --- Find Images ---
    image_dir = Path(args.image_dir); image_paths_unsorted = [p for p in image_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']]
    if not image_paths_unsorted: print(f"Error: No images found in {image_dir}"); return
    print(f"Found {len(image_paths_unsorted)} images in {image_dir}")

    # --- [!!!] V-FINAL FIX: Match Prompts and Images using '_pX_' pattern [!!!] ---
    images_to_eval_paths = []; prompts_to_eval = []; original_indices = []
    # New pattern looks for '_p' followed by digits, then '_'
    pattern = re.compile(r'_p(\d+)_') # <-- Corrected pattern
    image_map = {}

    print("Matching images to prompts based on '_pX_' pattern in filename...")
    for img_path in image_paths_unsorted:
        match = pattern.search(img_path.name) # Use search to find pattern anywhere
        if match:
            try:
                prompt_index_1based = int(match.group(1)) # Group 1 contains the digits
                prompt_index_0based = prompt_index_1based - 1
                if 0 <= prompt_index_0based < len(prompts):
                    if prompt_index_1based not in image_map: image_map[prompt_index_1based] = []
                    image_map[prompt_index_1based].append(img_path)
                else: print(f"Warning: Image {img_path.name} has out-of-bounds index {prompt_index_1based}.")
            except ValueError: print(f"Warning: Could not convert index for {img_path.name}")
        else:
            # Add a check for the old pattern just in case filenames are mixed
            old_pattern = re.compile(r'prompt_(\d+)_')
            old_match = old_pattern.search(img_path.name)
            if old_match:
                 print(f"Warning: Found old filename pattern 'prompt_X_' in {img_path.name}. Trying to match...")
                 try: # Duplicate matching logic for old pattern
                     prompt_index_1based = int(old_match.group(1))
                     prompt_index_0based = prompt_index_1based - 1
                     if 0 <= prompt_index_0based < len(prompts):
                         if prompt_index_1based not in image_map: image_map[prompt_index_1based] = []
                         image_map[prompt_index_1based].append(img_path)
                     else: print(f"Warning: Image {img_path.name} (old pattern) has out-of-bounds index {prompt_index_1based}.")
                 except ValueError: print(f"Warning: Could not convert index for {img_path.name} (old pattern)")
            else:
                 print(f"Warning: Could not find suitable prompt pattern ('_pX_' or 'prompt_X_') in {img_path.name}")


    # Sort by prompt index (1-based) and build final lists
    for i in sorted(image_map.keys()):
        img_path = image_map[i][0]; prompt_index_0based = i - 1
        images_to_eval_paths.append(str(img_path))
        prompts_to_eval.append(prompts[prompt_index_0based])
        original_indices.append(i)

    if len(prompts_to_eval) == 0: print("Error: No prompt-image pairs matched."); return
    print(f"Successfully matched {len(prompts_to_eval)} prompt-image pairs.")
    if len(prompts_to_eval) != len(prompts): print(f"Warning: Only matched {len(prompts_to_eval)}/{len(prompts)} prompts.")
    # --- [!!!] End V-FINAL FIX ---

    # --- Load CLIP Model ---
    model_name = args.model_name; print(f"Loading CLIP model and processor: {model_name}...")
    try:
        model_kwargs = {}; device_type = 'cuda' if device=='cuda' else 'cpu' # Needed for torch_dtype logic
        if device_type == "cuda" and args.use_float16: model_kwargs["torch_dtype"] = torch.float16; print("Using float16 for CLIP model.")
        model = CLIPModel.from_pretrained(model_name, **model_kwargs).to(device)
        processor = CLIPProcessor.from_pretrained(model_name); model.eval(); print("CLIP model and processor loaded.")
    except Exception as e: print(f"Error loading CLIP model/processor: {e}"); return

    # --- Process in Batches ---
    all_scores = []; scored_pairs = []; batch_size = args.batch_size
    print(f"Processing images in batches of {batch_size}...")
    for i in tqdm(range(0, len(images_to_eval_paths), batch_size), desc="Calculating CLIP Scores"):
        # ... (Batch processing loop remains the same) ...
        batch_image_paths = images_to_eval_paths[i:i+batch_size]; batch_prompts = prompts_to_eval[i:i+batch_size]; batch_original_indices = original_indices[i:i+batch_size]
        batch_images_pil = [load_image(p) for p in batch_image_paths]
        valid_indices_in_batch = [idx for idx, img in enumerate(batch_images_pil) if img is not None]
        if not valid_indices_in_batch: continue
        batch_images_pil_valid = [batch_images_pil[idx] for idx in valid_indices_in_batch]; batch_prompts_valid = [batch_prompts[idx] for idx in valid_indices_in_batch]; batch_original_indices_valid = [batch_original_indices[idx] for idx in valid_indices_in_batch]
        if not batch_images_pil_valid: continue
        scores = calculate_clip_similarity(model, processor, batch_images_pil_valid, batch_prompts_valid, device)
        if scores is not None:
            all_scores.extend(scores)
            for idx_in_batch, score in enumerate(scores): original_prompt_idx = batch_original_indices_valid[idx_in_batch]; image_filename = Path(batch_image_paths[valid_indices_in_batch[idx_in_batch]]).name; scored_pairs.append( (original_prompt_idx, score, image_filename) )
        else: print(f"Warning: Failed to calculate scores for batch starting at index {i}")

    # --- Calculate and Print Results ---
    if not all_scores: print("\nError: No scores were calculated."); return
    average_score = np.mean(all_scores) * 100; median_score = np.median(all_scores) * 100; std_dev = np.std(all_scores) * 100
    print("\n--- Results ---"); print(f"Directory: {args.image_dir}"); print(f"Number of Valid Pairs Evaluated: {len(all_scores)}")
    print(f"Average CLIP Score (x100): {average_score:.4f}"); print(f"Median CLIP Score (x100): {median_score:.4f}"); print(f"Std Dev CLIP Score (x100): {std_dev:.4f}")
    print("\n--- Individual Scores (Cosine Similarity * 100) ---"); scored_pairs.sort(key=lambda x: x[0])
    for original_idx, score, filename in scored_pairs: print(f"  Prompt {original_idx}: {score * 100:.4f}  ({filename})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CLIP Score directly using Transformers and print individual scores.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_float16", action="store_true")
    args = parser.parse_args()
    main(args)