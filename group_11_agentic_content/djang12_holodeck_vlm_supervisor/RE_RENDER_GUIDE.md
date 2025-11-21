# Scene Re-rendering Guide

VLM 분석 결과를 기반으로 JSON을 수정한 후, 수정된 scene을 다시 렌더링하는 방법입니다.

## 빠른 사용법

### 기본 사용 (JSON 파일과 같은 디렉토리에 이미지 저장)

```bash
python render_from_json.py --scene_json data/scenes/a_living_room-2025-11-17-16-25-02-371328/a_living_room.json
```

### 출력 경로 지정

```bash
python render_from_json.py \
    --scene_json data/scenes/a_living_room-2025-11-17-16-25-02-371328/a_living_room.json \
    --output_image data/scenes/a_living_room-2025-11-17-16-25-02-371328/a_living_room_updated.png
```

### 고해상도 렌더링

```bash
python render_from_json.py \
    --scene_json scene.json \
    --output_image rendered.png \
    --width 2048 \
    --height 2048
```

## Python 코드에서 직접 사용

```python
from render_from_json import render_scene_from_json

# JSON 파일 로드 및 수정 후 렌더링
image = render_scene_from_json(
    scene_json_path="data/scenes/a_living_room/a_living_room.json",
    output_image_path="rendered.png",
    width=1024,
    height=1024
)

# PIL Image 객체로 반환되므로 추가 처리 가능
image.show()  # 이미지 표시 (GUI 환경에서)
image.save("custom_path.png")  # 다른 경로에 저장
```

## JSON 수정 예시

VLM 분석 결과를 바탕으로 JSON을 수정하는 예시:

```python
import compress_json
import json

# 1. JSON 파일 로드
scene = compress_json.load("a_living_room.json")

# 2. VLM 분석 결과에 따라 객체 위치 수정
# 예: "coffee_table-0"의 위치를 수정
for obj in scene["objects"]:
    if obj.get("object_name") == "coffee_table-0":
        # VLM이 제안한 새로운 위치로 수정
        obj["position"]["x"] = 3.5  # 새로운 x 좌표
        obj["position"]["z"] = 2.0  # 새로운 z 좌표
        print(f"Updated {obj['object_name']} position")

# 3. 수정된 JSON 저장
compress_json.dump(
    scene,
    "a_living_room_updated.json",
    json_kwargs=dict(indent=4)
)

# 4. 렌더링
from render_from_json import render_scene_from_json
render_scene_from_json(
    scene_json_path="a_living_room_updated.json",
    output_image_path="a_living_room_updated.png"
)
```

## JSON 구조 이해

주요 수정 가능한 필드들:

### 객체 위치 수정
```json
{
    "objects": [
        {
            "object_name": "sofa-0",
            "position": {
                "x": 3.5,  // 미터 단위
                "y": 0.5,  // 높이 (일반적으로 객체 높이의 절반)
                "z": 2.0
            },
            "rotation": {
                "x": 0,
                "y": 90,   // Y축 회전 (0-360도)
                "z": 0
            }
        }
    ]
}
```

### 객체 제거
```python
# 특정 객체 제거
scene["objects"] = [
    obj for obj in scene["objects"] 
    if obj.get("object_name") != "unwanted_object-0"
]
```

### 객체 추가
```python
# 새 객체 추가
new_object = {
    "assetId": "some_asset_id",
    "id": "new_object-0 (living room)",
    "object_name": "new_object-0",
    "position": {"x": 1.0, "y": 0.5, "z": 1.0},
    "rotation": {"x": 0, "y": 0, "z": 0},
    "roomId": "living room",
    "kinematic": True
}
scene["objects"].append(new_object)
```

## 주의사항

1. **좌표계**: 
   - X, Z는 수평면 (바닥)
   - Y는 높이 (일반적으로 객체 높이의 절반을 y로 설정)
   - 단위는 미터

2. **방 경계**: 
   - 객체가 방(`roomId`)의 `vertices` 범위 내에 있어야 함
   - 벽과 겹치지 않도록 주의

3. **객체 크기**: 
   - `assetId`에 따라 객체의 실제 크기가 결정됨
   - 위치 수정 시 객체가 다른 객체나 벽과 겹치지 않도록 확인

4. **렌더링 시간**: 
   - 첫 렌더링은 약간 시간이 걸릴 수 있음 (Controller 초기화)
   - 여러 번 렌더링할 경우 약 10-30초 소요

## 자동화 워크플로우

VLM 분석 → JSON 수정 → 재렌더링을 자동화하는 예시:

```python
import subprocess
import json

# 1. VLM 분석 결과 로드 (예: JSON 형식)
with open("vlm_analysis.json", "r") as f:
    analysis = json.load(f)

# 2. JSON 수정
scene = compress_json.load("a_living_room.json")

for issue in analysis.get("critical_issues", []):
    # VLM이 제안한 수정사항 적용
    if "coffee_table" in issue.get("object", ""):
        # 위치 수정 로직
        pass

# 3. 수정된 JSON 저장
compress_json.dump(scene, "a_living_room_fixed.json")

# 4. 재렌더링
subprocess.run([
    "python", "render_from_json.py",
    "--scene_json", "a_living_room_fixed.json",
    "--output_image", "a_living_room_fixed.png"
])
```

## Collision 체크 및 수정

JSON을 수정한 후 collision이 발생할 수 있습니다. 원래 Holodeck 코드와 동일한 collision detection 로직을 사용하여 체크하고 수정할 수 있습니다.

### Collision 체크만 하기

```bash
python check_and_fix_collisions.py --scene_json scene.json --check_only
```

### Collision 자동 수정

```bash
# 작은 객체부터 제거 (기본 전략)
python check_and_fix_collisions.py --scene_json scene.json --output_json scene_fixed.json

# 모든 충돌 객체 제거
python check_and_fix_collisions.py --scene_json scene.json --strategy remove_all
```

### Python 코드에서 사용

```python
from check_and_fix_collisions import check_collisions, fix_collisions_by_removal
from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever

# 1. Scene과 database 로드
scene = compress_json.load("scene.json")
retriever = ObjathorRetriever(...)  # 초기화
database = retriever.database

# 2. Collision 체크
collisions = check_collisions(scene, database)
if collisions:
    print(f"Found {len(collisions)} collisions")
    for obj1, obj2, col_type in collisions:
        print(f"  {obj1} <-> {obj2}")

# 3. Collision 수정
fixed_scene = fix_collisions_by_removal(
    scene, collisions, database, strategy="remove_smaller"
)
compress_json.dump(fixed_scene, "scene_fixed.json")
```

### Collision 처리 방식

원래 Holodeck 코드는 다음과 같이 collision을 처리합니다:

1. **Floor objects**: 2D Polygon intersection 체크 (`DFS_Solver_Floor.filter_collision`)
2. **Small objects**: 3D bounding box intersection 체크 (`SmallObjectGenerator.check_collision`)
3. **Wall objects**: 3D bounding box intersection 체크 (`DFS_Solver_Wall.filter_collision`)

`check_and_fix_collisions.py`는 모든 객체에 대해 3D bounding box intersection을 사용합니다.

## 문제 해결

### "Objaverse asset directory not found" 오류
```bash
# OBJATHOR_ASSETS_DIR 환경 변수 확인
echo $OBJATHOR_ASSETS_DIR

# 또는 직접 지정
python render_from_json.py \
    --scene_json scene.json \
    --objaverse_asset_dir /path/to/objaverse/assets
```

### 렌더링 실패
- JSON 파일이 올바른 형식인지 확인
- `compress_json` 형식인 경우 그대로 사용 가능
- 일반 JSON인 경우 `compress_json.load()` 대신 `json.load()` 사용 후 변환 필요

### 메모리 부족
- 이미지 크기를 줄여서 시도 (`--width 512 --height 512`)
- 또는 배치 처리로 여러 scene을 순차적으로 렌더링

