#!/usr/bin/env python3
"""
Script to fix the metadata for RealEstate10K dataset.
The issue is that the metadata was built incorrectly - it's looking for videos in the wrong directory structure.
"""

import torch
from pathlib import Path
from torchvision.datasets.video_utils import _VideoTimestampsDataset, _collate_fn
from tqdm import tqdm

def fix_metadata():
    # Paths
    data_dir = Path("/shared/nas/data/m1/jiateng5/GeometryForcing/data/real-estate-10k")
    metadata_dir = data_dir / "metadata"
    
    # For training split, videos are in train/train_256/
    train_video_dir = data_dir / "train" / "train_256"
    
    print(f"Looking for videos in: {train_video_dir}")
    print(f"Directory exists: {train_video_dir.exists()}")
    
    if not train_video_dir.exists():
        print("ERROR: Video directory does not exist!")
        return
    
    # Find all video files
    video_paths = sorted(list(train_video_dir.glob("*.mp4")), key=str)
    print(f"Found {len(video_paths)} video files")
    
    if len(video_paths) == 0:
        print("ERROR: No video files found!")
        return
    
    print(f"First video: {video_paths[0]}")
    
    # Build metadata using the same process as the original code
    dl = torch.utils.data.DataLoader(
        _VideoTimestampsDataset(video_paths),
        batch_size=16,
        num_workers=8,  # Reduced to avoid issues
        collate_fn=_collate_fn,
    )
    
    video_pts = []
    video_fps = []
    
    print("Building metadata...")
    with tqdm(total=len(dl), desc="Building metadata for training") as pbar:
        for batch in dl:
            pbar.update(1)
            batch_pts, batch_fps = list(zip(*batch))
            batch_pts = [
                torch.as_tensor(pts, dtype=torch.long) for pts in batch_pts
            ]
            video_pts.extend(batch_pts)
            video_fps.extend(batch_fps)
    
    # Create metadata
    metadata = {
        "video_paths": video_paths,
        "video_pts": video_pts,
        "video_fps": video_fps,
    }
    
    # Save metadata
    output_path = metadata_dir / "training.pt"
    print(f"Saving metadata to: {output_path}")
    torch.save(metadata, output_path)
    
    print(f"Successfully saved metadata with {len(video_paths)} videos")
    
    # Verify the saved metadata
    print("Verifying saved metadata...")
    loaded_metadata = torch.load(output_path, weights_only=False)
    print(f"Loaded metadata keys: {list(loaded_metadata.keys())}")
    print(f"Number of videos: {len(loaded_metadata['video_paths'])}")
    print(f"Number of video_pts: {len(loaded_metadata['video_pts'])}")
    print(f"Number of video_fps: {len(loaded_metadata['video_fps'])}")

if __name__ == "__main__":
    fix_metadata()
