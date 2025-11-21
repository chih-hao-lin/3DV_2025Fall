import json
from copy import deepcopy
from pathlib import Path

# 이 JSON 구조를 기준으로 함:
# - scene["objects"]
# - scene["floor_objects"]
# - scene["wall_objects"]
# - scene["small_objects"]
# 각 object는
#   - "id": "sofa-0 (living room)"
#   - "object_name": "sofa-0"   # 있는 경우
# 를 가짐.

OBJECT_LIST_KEYS = [
    "objects",
    "floor_objects",
    "wall_objects",
    "small_objects",
]

ID_KEY = "id"
OBJECT_NAME_KEY = "object_name"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def object_matches_key(obj, key: str) -> bool:
    """
    패치에서 온 key와 오브젝트가 매칭되는지 판단.
    허용 패턴:
    - obj["object_name"] == key
    - obj["id"] == key
    - obj["id"].startswith(key + " ")
    - obj["id"].startswith(key + "|")
      (예: "sofa-0", "sofa-0 (living room)", "book-3|bookshelf-0 (living room)")
    """
    oid = obj.get(ID_KEY, "")
    oname = obj.get(OBJECT_NAME_KEY, "")

    if oname == key:
        return True
    if oid == key:
        return True
    if oid.startswith(key + " "):
        return True
    if oid.startswith(key + "|"):
        return True
    return False


def iter_all_objects(scene):
    """scene 안의 모든 object 리스트에서 (list_name, list_ref, obj) 튜플을 yield."""
    for list_name in OBJECT_LIST_KEYS:
        objs = scene.get(list_name, [])
        for obj in objs:
            yield list_name, objs, obj


def apply_patch(scene_data: dict, patch: dict) -> dict:
    """
    scene_data: 원본 scene json
    patch: {
      "move_or_rotate": {
        "sofa-0": {
          "position": {"x":..., "y":..., "z":...} 또는 [x,y,z]
          "rotation": {"x":..., "y":..., "z":...} 또는 [rx,ry,rz]
        },
        ...
      },
      "remove": ["bookshelf-0", "armchair-1", ...]
    }
    """
    scene = deepcopy(scene_data)

    move_or_rotate = patch.get("move_or_rotate", {}) or {}
    remove_ids = patch.get("remove", []) or []

    # ---------- 1) move_or_rotate 적용 ----------
    for key, upd in move_or_rotate.items():
        # position/rotation 포맷 정규화
        pos = upd.get("position")
        rot = upd.get("rotation")

        # list -> dict 로 변환 (x,z만 쓰고 y는 그대로 두고 싶으면 거기 맞게 수정해도 됨)
        def vec_to_dict(vec, prev):
            if vec is None:
                return None
            if isinstance(vec, dict):
                return vec
            if isinstance(vec, (list, tuple)) and len(vec) == 3:
                # 기존 y 유지하고 싶으면 prev["y"] 쓰는 식으로 약간 바꿔도 됨
                return {"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])}
            return prev

        # 모든 리스트(objects, floor_objects, ...)에서 매칭되는 애들을 전부 수정
        for list_name, objs, obj in iter_all_objects(scene):
            if not object_matches_key(obj, key):
                continue

            # position
            if pos is not None:
                obj["position"] = vec_to_dict(pos, obj.get("position", {}))

            # rotation
            if rot is not None:
                obj["rotation"] = vec_to_dict(rot, obj.get("rotation", {}))

    # ---------- 2) remove 적용 ----------
    if remove_ids:
        remove_ids_set = set(remove_ids)

        def should_remove(obj):
            # remove 리스트의 key 중 하나와 매칭되면 제거
            return any(object_matches_key(obj, key) for key in remove_ids_set)

        # 각 object 리스트에서 제거
        for list_name in OBJECT_LIST_KEYS:
            objs = scene.get(list_name, [])
            kept = []
            for obj in objs:
                if should_remove(obj):
                    print(f"[INFO] removing {obj.get(ID_KEY)} from {list_name}")
                    continue
                kept.append(obj)
            scene[list_name] = kept

        # selected_objects에서도 제거 (floor / wall 리스트에 첫 요소가 object_name)
        selected = scene.get("selected_objects", {})
        lr = selected.get("living room", {})
        for category in ["floor", "wall"]:
            arr = lr.get(category, [])
            new_arr = []
            for name, asset_id in arr:
                # name 이 remove_ids 중 하나와 매칭되면 제거
                if any(
                    name == key or name.startswith(key + "|") or name.startswith(key + " ")
                    for key in remove_ids_set
                ):
                    print(f"[INFO] removing selected_objects entry '{name}' from {category}")
                    continue
                new_arr.append([name, asset_id])
            lr[category] = new_arr
        selected["living room"] = lr
        scene["selected_objects"] = selected

    return scene


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply JSON patch to scene file")
    parser.add_argument(
        "--scene",
        type=str,
        help="Path to original scene JSON file",
        default="data/scenes/a_living_room-2025-11-17-17-59-04-478347/a_living_room.json",
    )
    parser.add_argument(
        "--patch",
        type=str,
        help="Path to patch JSON file (default: patch.json in same directory as input scene)",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output patched scene JSON file",
        default=None,
    )
    
    args = parser.parse_args()
    
    original_scene_path = Path(args.scene)
    
    # Default patch path: patch.json in same directory as input scene
    if args.patch:
        patch_path = Path(args.patch)
    else:
        patch_path = original_scene_path.parent / "patch.json"
    
    if args.output:
        output_scene_path = Path(args.output)
    else:
        # Default: same directory as input, with _patched suffix
        output_scene_path = original_scene_path.parent / f"{original_scene_path.stem}_patched{original_scene_path.suffix}"

    scene_data = load_json(original_scene_path)
    patch_data = load_json(patch_path)

    patched = apply_patch(scene_data, patch_data)
    save_json(patched, output_scene_path)

    print(f"Patched scene saved to: {output_scene_path}")