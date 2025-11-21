#!/usr/bin/env python3
"""
High-level interface for revising layout design and regenerating scenes.

Instead of modifying the long JSON file directly, you can:
1. Load an existing scene
2. Modify the high-level layout design (selected_objects)
3. Regenerate the scene from that point with constraints preserved
"""

import os
import sys
import compress_json

sys.path.insert(0, os.path.dirname(__file__))

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR
from ai2holodeck.generation.holodeck import Holodeck
from ai2holodeck.generation.utils import get_annotations


def revise_layout_and_regenerate(
    scene_json_path: str,
    modifications: dict,
    save_dir: str = "./data/scenes",
    use_constraint: bool = True,
):
    """
    High-level function to revise layout design and regenerate scene.
    
    Args:
        scene_json_path: Path to existing scene JSON file
        modifications: Dictionary of high-level modifications to apply:
            - "remove_objects": List of object names/patterns to remove
            - "modify_object_positions": Dict mapping object_name to new position/rotation
                Example: {
                    "sofa-0": {"position": {"x": 2.0, "y": 0.4, "z": 3.0}, "rotation": {"y": 180}},
                    "coffee_table-0": {"position": {"x": 2.1, "y": 0.25, "z": 2.8}}
                }
            - "modify_constraints": Dict mapping room_type to custom constraint plan
                Example: {
                    "living room": "sofa-0 | middle\ncoffee_table-0 | middle | near, sofa-0"
                }
            - "add_objects": List of objects to add. Each item should be a dict with:
                - "room_type": str (e.g., "living room")
                - "location": str ("floor" or "wall")
                - "object_name": str (e.g., "lamp-0")
                - "description": str (e.g., "a table lamp")
                Example: [{
                    "room_type": "living room",
                    "location": "floor",
                    "object_name": "lamp-0",
                    "description": "a modern table lamp"
                }]
            - "modify_rooms": Room-specific modifications (not implemented yet)
        save_dir: Directory to save regenerated scene
        use_constraint: Whether to use constraints (recommended: True)
    
    Returns:
        Tuple of (revised_scene_dict, save_directory_path)
    """
    # Initialize Holodeck
    holodeck = Holodeck(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_org=os.environ.get("OPENAI_ORG"),
        objaverse_asset_dir=OBJATHOR_ASSETS_DIR,
        single_room=False,
    )
    
    # Load existing scene
    print(f"Loading scene from: {scene_json_path}")
    scene = compress_json.load(scene_json_path)
    
    # Check if scene has layout design (selected_objects)
    if "selected_objects" not in scene:
        print("ERROR: Scene does not have 'selected_objects'. Cannot revise layout design.")
        print("This scene was likely generated without the layout design step.")
        return None, None
    
    print(f"Scene contains rooms: {list(scene.get('selected_objects', {}).keys())}")
    
    # Apply high-level modifications
    print("\nApplying modifications...")
    
    # Remove objects
    if "remove_objects" in modifications:
        removed_count = 0
        for room_type, room_objects in scene["selected_objects"].items():
            # Remove from floor objects
            if "floor" in room_objects:
                original_count = len(room_objects["floor"])
                room_objects["floor"] = [
                    obj for obj in room_objects["floor"]
                    if not any(
                        pattern.lower() in str(obj[0]).lower() 
                        for pattern in modifications["remove_objects"]
                    )
                ]
                removed_count += original_count - len(room_objects["floor"])
            
            # Remove from wall objects
            if "wall" in room_objects:
                original_count = len(room_objects["wall"])
                room_objects["wall"] = [
                    obj for obj in room_objects["wall"]
                    if not any(
                        pattern.lower() in str(obj[0]).lower()
                        for pattern in modifications["remove_objects"]
                    )
                ]
                removed_count += original_count - len(room_objects["wall"])
        
        print(f"  Removed {removed_count} objects matching patterns: {modifications['remove_objects']}")
    
    # Modify object positions/rotations (low-level, but preserves constraint system)
    if "modify_object_positions" in modifications:
        print("  Modifying object positions/rotations...")
        if "floor_objects" not in scene:
            print("  WARNING: floor_objects not found. Will be regenerated from selected_objects.")
        else:
            modified_count = 0
            for obj in scene["floor_objects"]:
                object_name = obj.get("object_name", "")
                if object_name in modifications["modify_object_positions"]:
                    mod = modifications["modify_object_positions"][object_name]
                    if "position" in mod:
                        obj["position"].update(mod["position"])
                        print(f"    Updated position for {object_name}: {obj['position']}")
                    if "rotation" in mod:
                        obj["rotation"].update(mod["rotation"])
                        print(f"    Updated rotation for {object_name}: {obj['rotation']}")
                    modified_count += 1
            print(f"  Modified {modified_count} objects' positions/rotations")
            # Note: This modifies existing floor_objects directly
            # If you want to regenerate with constraints, clear floor_objects instead
    
    # Modify constraints (high-level approach)
    if "modify_constraints" in modifications:
        print("  Modifying constraints...")
        # Store custom constraint plans in scene
        # These will be used by floor_object_generator if we modify it to check for existing constraints
        if "custom_constraint_plans" not in scene:
            scene["custom_constraint_plans"] = {}
        scene["custom_constraint_plans"].update(modifications["modify_constraints"])
        print(f"  Set custom constraints for rooms: {list(modifications['modify_constraints'].keys())}")
        # Note: This requires modifying floor_object_generator to use custom constraints
        # For now, this is a placeholder for future implementation
    
    # Add objects (using description-based retrieval, similar to object_selector)
    if "add_objects" in modifications:
        print("  Adding objects...")
        added_count = 0
        
        # Get room information for size checking
        room2size = {}
        for room in scene.get("rooms", []):
            room_type = room["roomType"]
            room2size[room_type] = holodeck.object_selector.get_room_size(
                room, scene.get("wall_height", 3.0)
            )
        
        for obj_spec in modifications["add_objects"]:
            room_type = obj_spec.get("room_type")
            location = obj_spec.get("location", "floor")  # "floor" or "wall"
            object_name = obj_spec.get("object_name")
            description = obj_spec.get("description")
            
            if not all([room_type, location, object_name, description]):
                print(f"  WARNING: Skipping incomplete object spec: {obj_spec}")
                continue
            
            if room_type not in scene["selected_objects"]:
                print(f"  WARNING: Room type '{room_type}' not found in scene. Skipping.")
                continue
            
            if location not in ["floor", "wall"]:
                print(f"  WARNING: Invalid location '{location}'. Must be 'floor' or 'wall'. Skipping.")
                continue
            
            print(f"  -> Adding {object_name} ({description}) to {room_type} ({location})...")
            
            # Retrieve similar objects using description (same as object_selector)
            query = f"a 3D model of {object_name}, {description}"
            threshold = (
                holodeck.object_selector.similarity_threshold_floor
                if location == "floor"
                else holodeck.object_selector.similarity_threshold_wall
            )
            
            candidates = holodeck.object_retriever.retrieve([query], threshold)
            print(f"    Found {len(candidates)} candidates, filtering...")
            
            if len(candidates) == 0:
                print(f"    ERROR: No candidates found for {object_name}. Skipping.")
                continue
            
            # Filter candidates based on location (floor or wall)
            database = holodeck.object_retriever.database
            filtered_candidates = []
            for candidate in candidates:
                asset_id = candidate[0]
                annotation = get_annotations(database[asset_id])
                
                # Check location requirement
                if location == "floor":
                    if not annotation.get("onFloor", False):
                        continue
                    if annotation.get("onCeiling", False):
                        continue
                elif location == "wall":
                    if not annotation.get("onWall", False):
                        continue
                
                # Ignore doors, windows, frames
                category = annotation.get("category", "").lower()
                if any(k in category for k in ["door", "window", "frame"]):
                    continue
                
                filtered_candidates.append(candidate)
            
            if len(filtered_candidates) == 0:
                print(f"    ERROR: No valid candidates after filtering for {object_name}. Skipping.")
                continue
            
            # Check object size if room size is available
            if room_type in room2size:
                filtered_candidates = holodeck.object_selector.check_object_size(
                    filtered_candidates, room2size[room_type]
                )
                if len(filtered_candidates) == 0:
                    print(f"    ERROR: All candidates too large for room. Skipping.")
                    continue
            
            # Select the best candidate (first one after filtering)
            selected_asset_id = filtered_candidates[0][0]
            print(f"    Selected asset: {selected_asset_id}")
            
            # Add to selected_objects
            if location not in scene["selected_objects"][room_type]:
                scene["selected_objects"][room_type][location] = []
            
            scene["selected_objects"][room_type][location].append(
                (object_name, selected_asset_id)
            )
            added_count += 1
        
        print(f"  Added {added_count} objects to selected_objects")
    
    # If floor_objects already exist and we're not modifying positions directly,
    # clear them to regenerate from selected_objects
    if "floor_objects" in scene:
        if "remove_objects" in modifications:
            if "remove_objects" in modifications:
                original_count = len(scene["floor_objects"])
                scene["floor_objects"] = [
                    obj for obj in scene["floor_objects"]
                    if not any(
                        pattern.lower() in obj.get("object_name", "").lower()
                        for pattern in modifications["remove_objects"]
                    )
                ]
                print(f"  Removed {original_count - len(scene['floor_objects'])} objects from floor_objects")
        
        # Clear floor_objects to regenerate from selected_objects
        # UNLESS we're modifying positions directly (in which case keep them)
        if "modify_object_positions" not in modifications:
            print("  Clearing floor_objects to regenerate from revised selected_objects...")
            scene["floor_objects"] = []
    
    # Clear wall_objects if they exist (will be regenerated)
    if "wall_objects" in scene:
        scene["wall_objects"] = []
    
    # Clear small_objects if they exist (will be regenerated)
    if "small_objects" in scene:
        scene["small_objects"] = []
    
    # Regenerate scene from layout design
    print("\nRegenerating scene from revised layout design...")
    
    # If we modified positions directly, we might want to skip regeneration
    # and just update the objects list
    if "modify_object_positions" in modifications and "floor_objects" in scene and len(scene["floor_objects"]) > 0:
        print("  Using modified floor_objects directly (not regenerating)...")
        # Update objects list
        scene["objects"] = scene["floor_objects"] + scene.get("wall_objects", [])
        # Still need to regenerate wall_objects and small_objects
        scene, save_dir = holodeck.generate_scene_from_layout(
            scene=scene,
            query=scene.get("query", "a living room"),
            save_dir=save_dir,
            used_assets=[],
            add_ceiling=False,
            generate_image=True,
            generate_video=False,
            add_time=True,
            use_constraint=use_constraint,
            random_selection=False,
            use_milp=False,
            start_from="wall_objects",  # Start from wall objects since floor_objects are already modified
        )
    else:
        scene, save_dir = holodeck.generate_scene_from_layout(
            scene=scene,
            query=scene.get("query", "a living room"),
            save_dir=save_dir,
            used_assets=[],
            add_ceiling=False,
            generate_image=True,
            generate_video=False,
            add_time=True,
            use_constraint=use_constraint,  # Constraints preserved!
            random_selection=False,
            use_milp=False,
            start_from="floor_objects",  # Start from floor object placement
        )
    
    print(f"\n✓ Scene regenerated! Saved to: {save_dir}")
    return scene, save_dir


def example_remove_coffee_table():
    """
    Example: Remove coffee table from a living room scene.
    """
    scene_path = "data/scenes/a_living_room-2025-11-17-16-25-02-371328/a_living_room.json"
    
    if not os.path.exists(scene_path):
        print(f"Scene not found: {scene_path}")
        print("Please provide a valid scene path.")
        return
    
    # High-level modification: remove coffee table
    modifications = {
        "remove_objects": ["coffee_table", "coffee table"],
    }
    
    revise_layout_and_regenerate(
        scene_json_path=scene_path,
        modifications=modifications,
        save_dir="./data/scenes",
        use_constraint=True,
    )


def example_modify_object_position():
    """
    Example: Modify object position and rotation.
    """
    scene_path = "data/scenes/a_living_room-2025-11-17-17-59-04-478347/a_living_room.json"
    
    if not os.path.exists(scene_path):
        print(f"Scene not found: {scene_path}")
        return
    
    # Modify object positions/rotations
    modifications = {
        "modify_object_positions": {
            "sofa-0": {
                "position": {"x": 1.0, "z": 2.5},  # Only update x and z
                "rotation": {"y": 270},  # Rotate 270 degrees
            },
            "coffee_table-0": {
                "position": {"x": 2.5, "z": 2.8},  # Move coffee table
            },
        },
    }
    
    revise_layout_and_regenerate(
        scene_json_path=scene_path,
        modifications=modifications,
        save_dir="./data/scenes",
        use_constraint=False,  # Note: Direct position modification bypasses constraints
    )


def example_modify_constraints():
    """
    Example: Modify constraints to change object placement strategy.
    Note: This requires modifying floor_object_generator to use custom constraints.
    """
    scene_path = "data/scenes/a_living_room-2025-11-17-17-59-04-478347/a_living_room.json"
    
    if not os.path.exists(scene_path):
        print(f"Scene not found: {scene_path}")
        return
    
    # Modify constraints (high-level approach)
    modifications = {
        "modify_constraints": {
            "living room": """sofa-0 | middle
coffee_table-0 | middle | near, sofa-0 | in front of, sofa-0
tv_stand-0 | edge | far, coffee_table-0""",
        },
    }
    
    revise_layout_and_regenerate(
        scene_json_path=scene_path,
        modifications=modifications,
        save_dir="./data/scenes",
        use_constraint=True,
    )


def example_remove_multiple_objects():
    """
    Example: Remove multiple objects from a scene.
    """
    scene_path = "data/scenes/a_living_room-2025-11-17-16-25-02-371328/a_living_room.json"
    
    if not os.path.exists(scene_path):
        print(f"Scene not found: {scene_path}")
        return
    
    # High-level modification: remove multiple objects
    modifications = {
        "remove_objects": [
            "coffee_table",
            "coffee table",
            "side_table",
            "end table",
        ],
    }
    
    revise_layout_and_regenerate(
        scene_json_path=scene_path,
        modifications=modifications,
        save_dir="./data/scenes",
        use_constraint=True,
    )


def example_add_objects():
    """
    Example: Add objects to a scene using description-based retrieval.
    """
    scene_path = "data/scenes/a_living_room-2025-11-17-17-59-04-478347/a_living_room.json"
    
    if not os.path.exists(scene_path):
        print(f"Scene not found: {scene_path}")
        return
    
    # Add objects using description (similar to object_selector logic)
    modifications = {
        "add_objects": [
            {
                "room_type": "living room",
                "location": "floor",
                "object_name": "lamp-0",
                "description": "a modern table lamp",
            },
            {
                "room_type": "living room",
                "location": "wall",
                "object_name": "painting-0",
                "description": "a decorative wall painting",
            },
        ],
    }
    
    revise_layout_and_regenerate(
        scene_json_path=scene_path,
        modifications=modifications,
        save_dir="./data/scenes",
        use_constraint=True,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Revise layout design at high-level and regenerate scene"
    )
    parser.add_argument(
        "--scene",
        type=str,
        help="Path to scene JSON file",
        default="data/scenes/a_living_room-2025-11-17-16-25-02-371328/a_living_room.json",
    )
    parser.add_argument(
        "--remove",
        nargs="+",
        help="Object names/patterns to remove (e.g., --remove coffee_table sofa)",
        default=[],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save regenerated scene",
        default="./data/scenes",
    )
    
    args = parser.parse_args()
    
    if args.remove:
        modifications = {
            "remove_objects": args.remove,
        }
        revise_layout_and_regenerate(
            scene_json_path=args.scene,
            modifications=modifications,
            save_dir=args.save_dir,
            use_constraint=True,
        )
    else:
        # Run example
        print("Running example: Remove coffee table")
        example_remove_coffee_table()
