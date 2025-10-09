#!/usr/bin/env python3
import os, shutil, re
from tqdm import tqdm

# ────────── 6 sequences to process ──────────
sequences = [
    "seq007_garden_001",
    "seq008_running_001",
]

base_dir = "demo_data/input_images"

# Supported extensions
exts = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}

float_re = re.compile(r"[-+]?\d*\.\d+|\d+")   # Extract the first number from filename (can include decimal point)

def get_float(name: str) -> float:
    """
    Extract the first numeric segment from filename (without extension) and convert to float.
    Raises exception if no number is found.
    """
    m = float_re.search(name)
    if not m:
        raise ValueError(f"cannot find numeric value in {name}")
    return float(m.group(0))

for seq in sequences:
    cam_dir = os.path.join(base_dir, f"{seq}_imgs", "cam01")
    if not os.path.isdir(cam_dir):
        print(f"[warn] {cam_dir} not found, skip.")
        continue

    print(f"Processing {seq} ...")

    # Collect frame files
    frame_files = [
        f for f in os.listdir(cam_dir)
        if os.path.splitext(f)[1] in exts and os.path.isfile(os.path.join(cam_dir, f))
    ]
    if not frame_files:
        print(f"  [warn] no images in {cam_dir}, skip.")
        continue

    # Sort by float values
    try:
        frame_files.sort(key=lambda f: get_float(os.path.splitext(f)[0]))
    except Exception as e:
        print(f"  [error] cannot sort in {cam_dir}: {e}")
        continue

    # Rename to temporary directory to avoid overwriting
    tmp_dir = os.path.join(cam_dir, "_tmp_renaming")
    os.makedirs(tmp_dir, exist_ok=True)

    for idx, fname in enumerate(tqdm(frame_files, desc="  renaming", ncols=70)):
        ext = os.path.splitext(fname)[1].lower()
        new_fname = f"{idx:05d}{ext}"
        shutil.move(
            os.path.join(cam_dir, fname),
            os.path.join(tmp_dir, new_fname)
        )

    # Move back to original directory and delete temporary directory
    for fname in os.listdir(tmp_dir):
        shutil.move(os.path.join(tmp_dir, fname), os.path.join(cam_dir, fname))
    os.rmdir(tmp_dir)

    print(f"  Completed {seq}: {len(frame_files)} frames renamed\n")

print("All sequences processed!")
