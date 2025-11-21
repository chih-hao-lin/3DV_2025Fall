#!/usr/bin/env python3
"""
Script to render an image from a modified scene JSON file.

Usage:
    python render_from_json.py --scene_json path/to/scene.json --output_image path/to/output.png
    python render_from_json.py --scene_json path/to/scene.json  # saves to same directory as input
"""

import argparse
import os
import sys
from pathlib import Path

import compress_json
from PIL import Image

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from ai2holodeck.generation.utils import get_top_down_frame
from check_and_fix_collisions import check_collisions, fix_collisions_by_removal


def render_scene_from_json(
    scene_json_path: str,
    output_image_path: str = None,
    objaverse_asset_dir: str = None,
    width: int = 1024,
    height: int = 1024,
    fix_collisions: bool = False,
    overlap_threshold_ratio: float = 0.05,
    collision_strategy: str = "remove_smaller",
) -> Image.Image:
    """
    Load a scene JSON file and render a top-down view image.
    
    Args:
        scene_json_path: Path to the scene JSON file (can be compress_json format)
        output_image_path: Path to save the output image. If None, saves next to JSON file.
        objaverse_asset_dir: Directory containing Objaverse assets. If None, uses default.
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        PIL Image object
    """
    # Load scene JSON
    print(f"Loading scene from: {scene_json_path}")
    try:
        scene = compress_json.load(scene_json_path)
    except Exception as e:
        print(f"Error loading scene JSON: {e}")
        raise
    
    # Fix collisions if requested
    if fix_collisions:
        print("\n" + "="*60)
        print("Checking and fixing collisions...")
        print("="*60)
        
        # Initialize object retriever to get database
        import torch
        import open_clip
        from sentence_transformers import SentenceTransformer
        from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading object database (device: {device})...")
        
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
        print(f"Checking for collisions (overlap threshold: {overlap_threshold_ratio*100:.1f}% of smaller object)...")
        collisions = check_collisions(scene, database, overlap_threshold_ratio=overlap_threshold_ratio)
        
        if not collisions:
            print("✓ No collisions found!")
        else:
            print(f"\nFound {len(collisions)} collision(s):")
            for obj1_id, obj2_id, collision_type, overlap_volume, threshold_volume in collisions:
                print(f"  - {obj1_id} <-> {obj2_id} ({collision_type}, overlap: {overlap_volume:.1f} cm³, threshold: {threshold_volume:.1f} cm³)")
            
            # Fix collisions
            print(f"\nFixing collisions using strategy: {collision_strategy}")
            scene = fix_collisions_by_removal(scene, collisions, database, strategy=collision_strategy)
            print("✓ Collisions fixed!")
        print("="*60 + "\n")
    
    # Use default asset directory if not provided
    if objaverse_asset_dir is None:
        objaverse_asset_dir = OBJATHOR_ASSETS_DIR
    
    if not os.path.exists(objaverse_asset_dir):
        print(f"Warning: Objaverse asset directory not found: {objaverse_asset_dir}")
        print("Please specify --objaverse_asset_dir or set OBJATHOR_ASSETS_DIR")
    
    # Render the image
    print(f"Rendering top-down view (this may take a moment)...")
    try:
        image = get_top_down_frame(
            scene=scene,
            objaverse_asset_dir=objaverse_asset_dir,
            width=width,
            height=height,
        )
    except Exception as e:
        print(f"Error rendering scene: {e}")
        raise
    
    # Determine output path
    if output_image_path is None:
        # Save next to the JSON file with .png extension
        json_path = Path(scene_json_path)
        output_image_path = json_path.parent / f"{json_path.stem}.png"
    
    # Save the image
    print(f"Saving image to: {output_image_path}")
    image.save(output_image_path)
    print(f"✓ Image saved successfully!")
    
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Render a top-down view image from a scene JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render and save to default location (next to JSON file)
  python render_from_json.py --scene_json data/scenes/a_living_room/a_living_room.json
  
  # Render and save to specific location
  python render_from_json.py --scene_json scene.json --output_image rendered.png
  
  # Render with custom dimensions
  python render_from_json.py --scene_json scene.json --width 2048 --height 2048
  
  # Render with collision fixing
  python render_from_json.py --scene_json scene.json --fix_collisions
  
  # Render with custom collision threshold
  python render_from_json.py --scene_json scene.json --fix_collisions --overlap_threshold_ratio 0.1
        """,
    )
    
    parser.add_argument(
        "--scene_json",
        type=str,
        required=True,
        help="Path to the scene JSON file (supports compress_json format)",
    )
    
    parser.add_argument(
        "--output_image",
        type=str,
        default=None,
        help="Path to save the output image. If not specified, saves next to JSON file with .png extension",
    )
    
    parser.add_argument(
        "--objaverse_asset_dir",
        type=str,
        default=None,
        help=f"Directory containing Objaverse assets (default: {OBJATHOR_ASSETS_DIR})",
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (default: 1024)",
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels (default: 1024)",
    )
    
    parser.add_argument(
        "--fix_collisions",
        action="store_true",
        help="Check and fix collisions before rendering (removes colliding objects)",
    )
    
    parser.add_argument(
        "--overlap_threshold_ratio",
        type=float,
        default=0.05,
        help="Maximum allowed overlap as a ratio of the smaller object's volume (0.0-1.0). Default: 0.05 (5%%)",
    )
    
    parser.add_argument(
        "--collision_strategy",
        type=str,
        default="remove_smaller",
        choices=["remove_smaller", "remove_all"],
        help="Strategy for fixing collisions: 'remove_smaller' (default) removes smaller objects, 'remove_all' removes all colliding objects",
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.scene_json):
        print(f"Error: Scene JSON file not found: {args.scene_json}")
        sys.exit(1)
    
    # Render the scene
    try:
        render_scene_from_json(
            scene_json_path=args.scene_json,
            output_image_path=args.output_image,
            objaverse_asset_dir=args.objaverse_asset_dir,
            width=args.width,
            height=args.height,
            fix_collisions=args.fix_collisions,
            overlap_threshold_ratio=args.overlap_threshold_ratio,
            collision_strategy=args.collision_strategy,
        )
    except Exception as e:
        print(f"Failed to render scene: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



