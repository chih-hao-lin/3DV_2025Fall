#!/usr/bin/env python3
"""
Simplify scene JSON files by removing rendering-unnecessary fields.

This script removes intermediate generation data (like selected_objects, 
object_selection_plan, raw_*_plan, etc.) while keeping only what's needed
for rendering (objects, rooms, walls, doors, windows, proceduralParameters, metadata).

Usage:
    python simplify_scene_json.py <input_json> [--output <output_json>] [--backup]
"""

import os
import sys
import argparse
import compress_json
from typing import Dict, Any


# Fields required for rendering
RENDERING_REQUIRED_FIELDS = {
    "objects",           # All objects (floor + wall + small + ceiling)
    "rooms",            # Room definitions
    "walls",            # Wall definitions
    "doors",            # Door definitions
    "windows",          # Window definitions
    "proceduralParameters",  # Lights, skybox, materials
    "metadata",         # Agent poses, schema, etc.
    "query",            # Original query (optional but useful)
    "wall_height",      # Wall height (used in rendering)
}

# Fields that are redundant (already in objects)
REDUNDANT_FIELDS = {
    "floor_objects",   # Already merged into objects
    "wall_objects",     # Already merged into objects
    "small_objects",    # Already merged into objects
    "ceiling_objects", # Already merged into objects (if exists)
}

# Fields that are intermediate generation data (not needed for rendering)
INTERMEDIATE_FIELDS = {
    "selected_objects",        # Layout design step
    "object_selection_plan",   # Layout design step
    "raw_floor_plan",          # LLM intermediate output
    "raw_doorway_plan",        # LLM intermediate output
    "raw_window_plan",         # LLM intermediate output
    "raw_ceiling_plan",        # LLM intermediate output (if exists)
    "room_pairs",              # Intermediate calculation
    "open_room_pairs",         # Intermediate calculation
    "open_walls",             # Intermediate calculation
    "receptacle2small_objects", # Intermediate calculation
}


def simplify_scene(scene: Dict[str, Any], keep_intermediate: bool = False) -> Dict[str, Any]:
    """
    Simplify scene JSON by removing rendering-unnecessary fields.
    
    Args:
        scene: Original scene dictionary
        keep_intermediate: If True, keep intermediate fields (for debugging)
    
    Returns:
        Simplified scene dictionary
    """
    simplified = {}
    
    # Keep required rendering fields
    for field in RENDERING_REQUIRED_FIELDS:
        if field in scene:
            simplified[field] = scene[field]
    
    # Optionally keep intermediate fields
    if keep_intermediate:
        for field in INTERMEDIATE_FIELDS:
            if field in scene:
                simplified[field] = scene[field]
    
    # Always remove redundant fields (they're already in objects)
    # (Don't add them to simplified)
    
    return simplified


def verify_simplified_scene(scene: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Verify that simplified scene has all required fields for rendering.
    
    Returns:
        (is_valid, missing_fields)
    """
    missing = []
    for field in RENDERING_REQUIRED_FIELDS:
        if field not in scene:
            missing.append(field)
    
    return len(missing) == 0, missing


def main():
    parser = argparse.ArgumentParser(
        description="Simplify scene JSON by removing rendering-unnecessary fields"
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to input scene JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output JSON file (default: <input>_simple.json)",
    )
    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        help="Create backup of original file",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate generation fields (for debugging)",
    )
    parser.add_argument(
        "--in-place", "-i",
        action="store_true",
        help="Modify file in place (overwrite original)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify simplified scene has all required fields (default: True)",
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input_json):
        print(f"ERROR: Input file not found: {args.input_json}")
        sys.exit(1)
    
    # Load scene
    print(f"Loading scene from: {args.input_json}")
    try:
        scene = compress_json.load(args.input_json)
    except Exception as e:
        print(f"ERROR: Failed to load scene: {e}")
        sys.exit(1)
    
    # Show original size
    original_keys = set(scene.keys())
    print(f"Original scene has {len(original_keys)} top-level fields:")
    print(f"  {sorted(original_keys)}")
    
    # Simplify
    print("\nSimplifying scene...")
    simplified = simplify_scene(scene, keep_intermediate=args.keep_intermediate)
    
    # Show simplified size
    simplified_keys = set(simplified.keys())
    removed_keys = original_keys - simplified_keys
    print(f"Simplified scene has {len(simplified_keys)} top-level fields:")
    print(f"  {sorted(simplified_keys)}")
    print(f"\nRemoved {len(removed_keys)} fields:")
    for key in sorted(removed_keys):
        print(f"  - {key}")
    
    # Verify
    if args.verify:
        is_valid, missing = verify_simplified_scene(simplified)
        if not is_valid:
            print(f"\nWARNING: Simplified scene is missing required fields: {missing}")
            print("This scene may not render correctly!")
        else:
            print("\n✓ Simplified scene has all required fields for rendering")
    
    # Determine output path
    if args.in_place:
        output_path = args.input_json
    elif args.output:
        output_path = args.output
    else:
        # Default: same directory as input, with _simple suffix
        input_dir = os.path.dirname(args.input_json)
        input_basename = os.path.basename(args.input_json)
        base, ext = os.path.splitext(input_basename)
        output_basename = f"{base}_simple{ext}"
        output_path = os.path.join(input_dir, output_basename) if input_dir else output_basename
    
    # Create backup if requested
    if args.backup and not args.in_place:
        backup_path = f"{args.input_json}.backup"
        print(f"\nCreating backup: {backup_path}")
        try:
            import shutil
            shutil.copy2(args.input_json, backup_path)
        except Exception as e:
            print(f"WARNING: Failed to create backup: {e}")
    
    # Save simplified scene
    print(f"\nSaving simplified scene to: {output_path}")
    try:
        compress_json.dump(
            simplified,
            output_path,
            json_kwargs=dict(indent=4),
        )
        
        # Show file size comparison
        original_size = os.path.getsize(args.input_json)
        simplified_size = os.path.getsize(output_path)
        reduction = (1 - simplified_size / original_size) * 100
        
        print(f"\nFile size comparison:")
        print(f"  Original:  {original_size:,} bytes")
        print(f"  Simplified: {simplified_size:,} bytes")
        print(f"  Reduction:  {reduction:.1f}%")
        print(f"\n✓ Successfully simplified scene!")
        
    except Exception as e:
        print(f"ERROR: Failed to save simplified scene: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

