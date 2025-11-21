#!/usr/bin/env python3
"""
Utility to check and fix collisions in a scene JSON file.

This script uses the same collision detection logic as the original Holodeck code.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import compress_json
import numpy as np

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from ai2holodeck.generation.utils import get_bbox_dims
from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever


def get_bounding_box_3d(placement: Dict[str, Any], database: Dict) -> Dict[str, List[float]]:
    """
    Get 3D bounding box for a placement (same logic as SmallObjectGenerator.get_bounding_box).
    
    Args:
        placement: Object placement dict with 'assetId' and 'position'
        database: Object database from ObjathorRetriever
    
    Returns:
        Dict with 'min' and 'max' keys, each containing [x, y, z] coordinates in cm
    """
    asset_id = placement["assetId"]
    dimensions = get_bbox_dims(database[asset_id])
    size = (dimensions["x"] * 100, dimensions["y"] * 100, dimensions["z"] * 100)
    position = placement["position"]
    
    # Account for rotation (simplified - assumes rotation only affects x/z, not y)
    rotation_y = placement.get("rotation", {}).get("y", 0)
    
    # Swap x and z dimensions if rotated 90 or 270 degrees
    if rotation_y in [90, 270]:
        size = (size[2], size[1], size[0])
    
    box = {
        "min": [
            position["x"] * 100 - size[0] / 2,
            position["y"] * 100 - size[1] / 2,
            position["z"] * 100 - size[2] / 2,
        ],
        "max": [
            position["x"] * 100 + size[0] / 2,
            position["y"] * 100 + size[1] / 2,
            position["z"] * 100 + size[2] / 2,
        ],
    }
    return box


def get_box_volume(box: Dict[str, List[float]]) -> float:
    """
    Calculate the volume of a 3D bounding box.
    
    Args:
        box: Dict with 'min' and 'max' keys, each containing [x, y, z] coordinates
    
    Returns:
        Volume in cm³
    """
    width = box["max"][0] - box["min"][0]
    height = box["max"][1] - box["min"][1]
    depth = box["max"][2] - box["min"][2]
    return width * height * depth


def intersect_3d(box1: Dict[str, List[float]], box2: Dict[str, List[float]], threshold_ratio: float = 0.0) -> Tuple[bool, float, float]:
    """
    Check if two 3D bounding boxes intersect, with optional threshold based on smaller object's volume.
    
    Args:
        box1, box2: Dicts with 'min' and 'max' keys, each containing [x, y, z] coordinates
        threshold_ratio: Maximum allowed overlap as a ratio of the smaller object's volume (0.0-1.0).
                        Default 0.0 means any overlap is detected. E.g., 0.05 means 5% of smaller object.
    
    Returns:
        Tuple of (intersects: bool, overlap_volume: float, threshold_volume: float)
    """
    # Check if boxes overlap in all dimensions
    overlaps = True
    for i in range(3):
        if box1["max"][i] < box2["min"][i] or box1["min"][i] > box2["max"][i]:
            overlaps = False
            break
    
    if not overlaps:
        return False, 0.0, 0.0
    
    # Calculate overlap volume
    overlap_x = max(0, min(box1["max"][0], box2["max"][0]) - max(box1["min"][0], box2["min"][0]))
    overlap_y = max(0, min(box1["max"][1], box2["max"][1]) - max(box1["min"][1], box2["min"][1]))
    overlap_z = max(0, min(box1["max"][2], box2["max"][2]) - max(box1["min"][2], box2["min"][2]))
    overlap_volume = overlap_x * overlap_y * overlap_z
    
    # Calculate volumes of both boxes
    volume1 = get_box_volume(box1)
    volume2 = get_box_volume(box2)
    smaller_volume = min(volume1, volume2)
    
    # Calculate threshold based on smaller object's volume
    threshold_volume = smaller_volume * threshold_ratio if threshold_ratio > 0 else 0.0
    
    # If threshold is set, ignore small overlaps relative to smaller object
    if threshold_ratio > 0 and overlap_volume <= threshold_volume:
        return False, overlap_volume, threshold_volume
    
    return True, overlap_volume, threshold_volume


def check_collisions(
    scene: Dict[str, Any],
    database: Dict,
    check_floor_objects: bool = True,
    check_wall_objects: bool = True,
    check_small_objects: bool = True,
    overlap_threshold_ratio: float = 0.0,
) -> List[Tuple[str, str, str, float, float]]:
    """
    Check for collisions in a scene.
    
    Args:
        scene: Scene dictionary
        database: Object database from ObjathorRetriever
        check_floor_objects: Whether to check floor objects
        check_wall_objects: Whether to check wall objects
        check_small_objects: Whether to check small objects
        overlap_threshold_ratio: Maximum allowed overlap as a ratio of the smaller object's volume (0.0-1.0).
                                Default 0.0 means any overlap is detected. E.g., 0.05 means 5% of smaller object.
    
    Returns:
        List of tuples (object1_id, object2_id, collision_type, overlap_volume, threshold_volume) for each collision
    """
    collisions = []
    
    # Collect all objects to check
    all_objects = []
    
    if check_floor_objects:
        for obj in scene.get("floor_objects", []):
            if obj.get("kinematic", True):  # Only check static objects
                all_objects.append(("floor", obj))
    
    if check_wall_objects:
        for obj in scene.get("wall_objects", []):
            if obj.get("kinematic", True):
                all_objects.append(("wall", obj))
    
    if check_small_objects:
        for obj in scene.get("small_objects", []):
            if obj.get("kinematic", True):
                all_objects.append(("small", obj))
    
    # Also check combined objects list if it exists
    if "objects" in scene:
        for obj in scene["objects"]:
            if obj.get("kinematic", True):
                # Skip if already added
                obj_id = obj.get("id", "")
                if not any(existing_obj.get("id") == obj_id for _, existing_obj in all_objects):
                    all_objects.append(("combined", obj))
    
    # Check pairwise collisions
    for i, (type1, obj1) in enumerate(all_objects):
        try:
            box1 = get_bounding_box_3d(obj1, database)
        except Exception as e:
            print(f"Warning: Could not get bounding box for {obj1.get('id', 'unknown')}: {e}")
            continue
        
        for j, (type2, obj2) in enumerate(all_objects[i + 1:], start=i + 1):
            try:
                box2 = get_bounding_box_3d(obj2, database)
            except Exception as e:
                print(f"Warning: Could not get bounding box for {obj2.get('id', 'unknown')}: {e}")
                continue
            
            intersects, overlap_volume, threshold_volume = intersect_3d(box1, box2, overlap_threshold_ratio)
            if intersects:
                collisions.append((
                    obj1.get("id", "unknown"),
                    obj2.get("id", "unknown"),
                    f"{type1}-{type2}",
                    overlap_volume,
                    threshold_volume
                ))
    
    return collisions


def fix_collisions_by_removal(
    scene: Dict[str, Any],
    collisions: List[Tuple[str, str, str, float, float]],
    database: Dict,
    strategy: str = "remove_smaller"
) -> Dict[str, Any]:
    """
    Fix collisions by removing objects (same logic as SmallObjectGenerator.check_collision).
    
    Args:
        scene: Scene dictionary
        collisions: List of collision tuples
        database: Object database
        strategy: "remove_smaller" (remove smaller objects) or "remove_all" (remove all colliding)
    
    Returns:
        Modified scene dictionary
    """
    if not collisions:
        return scene
    
    # Create a deep copy to avoid modifying original
    scene = copy.deepcopy(scene)
    
    # Collect all colliding object IDs
    colliding_ids = set()
    for obj1_id, obj2_id, _, _, _ in collisions:
        colliding_ids.add(obj1_id)
        colliding_ids.add(obj2_id)
    
    if strategy == "remove_smaller":
        # Sort by size (smaller first) - same as original code
        id2assetId = {}
        for obj_list_name in ["floor_objects", "wall_objects", "small_objects", "objects"]:
            for obj in scene.get(obj_list_name, []):
                if obj.get("id") in colliding_ids:
                    id2assetId[obj.get("id")] = obj.get("assetId")
        
        # Sort by area (x * z)
        colliding_ids_sorted = sorted(
            colliding_ids,
            key=lambda x: get_bbox_dims(database.get(id2assetId.get(x, {}), {})).get("x", 0)
            * get_bbox_dims(database.get(id2assetId.get(x, {}), {})).get("z", 0),
        )
        
        # Remove objects one by one until no collisions remain
        remove_ids = set()
        remaining_collisions = collisions.copy()
        
        for obj_id in colliding_ids_sorted:
            remove_ids.add(obj_id)
            remaining_collisions = [
                (o1, o2, t, v, th) for o1, o2, t, v, th in remaining_collisions
                if o1 not in remove_ids and o2 not in remove_ids
            ]
            if not remaining_collisions:
                break
    else:  # remove_all
        remove_ids = colliding_ids
    
    # Remove objects from scene
    for obj_list_name in ["floor_objects", "wall_objects", "small_objects", "objects"]:
        if obj_list_name in scene:
            scene[obj_list_name] = [
                obj for obj in scene[obj_list_name]
                if obj.get("id") not in remove_ids
            ]
    
    print(f"Removed {len(remove_ids)} objects to fix collisions:")
    for obj_id in remove_ids:
        print(f"  - {obj_id}")
    
    return scene


def main():
    parser = argparse.ArgumentParser(
        description="Check and fix collisions in a scene JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check for collisions
  python check_and_fix_collisions.py --scene_json scene.json --check_only
  
  # Fix collisions by removing smaller objects
  python check_and_fix_collisions.py --scene_json scene.json --output_json scene_fixed.json
  
  # Fix collisions by removing all colliding objects
  python check_and_fix_collisions.py --scene_json scene.json --strategy remove_all
        """,
    )
    
    parser.add_argument(
        "--scene_json",
        type=str,
        required=True,
        help="Path to the scene JSON file",
    )
    
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save the fixed scene JSON (if not specified, overwrites input)",
    )
    
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only check for collisions, don't fix them",
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["remove_smaller", "remove_all"],
        default="remove_smaller",
        help="Strategy for fixing collisions (default: remove_smaller)",
    )
    
    parser.add_argument(
        "--overlap_threshold_ratio",
        type=float,
        default=0.05,
        help="Maximum allowed overlap as a ratio of the smaller object's volume (0.0-1.0). Objects with overlap <= threshold_ratio * smaller_volume will be ignored. Default: 0.05 (5% of smaller object)",
    )
    
    parser.add_argument(
        "--objaverse_asset_dir",
        type=str,
        default=None,
        help="Directory containing Objaverse assets (uses default if not specified)",
    )
    
    args = parser.parse_args()
    
    # Load scene
    print(f"Loading scene from: {args.scene_json}")
    try:
        scene = compress_json.load(args.scene_json)
    except Exception as e:
        print(f"Error loading scene JSON: {e}")
        sys.exit(1)
    
    # Initialize object retriever to get database
    from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
    from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
    import torch
    import open_clip
    from sentence_transformers import SentenceTransformer
    
    objaverse_asset_dir = args.objaverse_asset_dir or OBJATHOR_ASSETS_DIR
    
    print("Loading object database (this may take a moment)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )
    clip_model = clip_model.to(device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    sbert_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    
    retriever = ObjathorRetriever(
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        sbert_model=sbert_model,
        retrieval_threshold=28,
    )
    database = retriever.database
    
    # Check for collisions
    print(f"\nChecking for collisions (overlap threshold: {args.overlap_threshold_ratio*100:.1f}% of smaller object)...")
    collisions = check_collisions(scene, database, overlap_threshold_ratio=args.overlap_threshold_ratio)
    
    if not collisions:
        print("✓ No collisions found!")
        return
    
    print(f"\nFound {len(collisions)} collision(s):")
    for obj1_id, obj2_id, collision_type, overlap_volume, threshold_volume in collisions:
        print(f"  - {obj1_id} <-> {obj2_id} ({collision_type}, overlap: {overlap_volume:.1f} cm³, threshold: {threshold_volume:.1f} cm³)")
    
    if args.check_only:
        print("\nCheck-only mode: not fixing collisions.")
        return
    
    # Fix collisions
    print(f"\nFixing collisions using strategy: {args.strategy}")
    fixed_scene = fix_collisions_by_removal(scene, collisions, database, args.strategy)
    
    # Save fixed scene
    output_path = args.output_json or args.scene_json
    print(f"\nSaving fixed scene to: {output_path}")
    compress_json.dump(
        fixed_scene,
        output_path,
        json_kwargs=dict(indent=4)
    )
    print("✓ Done!")


if __name__ == "__main__":
    main()

