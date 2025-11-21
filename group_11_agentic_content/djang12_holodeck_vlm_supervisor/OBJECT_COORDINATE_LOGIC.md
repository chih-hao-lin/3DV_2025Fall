# 객체 좌표 계산 로직 위치

이 문서는 Holodeck에서 객체의 좌표(position)를 계산하는 로직의 위치를 정리합니다.

## 1. Floor Objects (바닥 객체)

### use_constraint=True (Constraint 기반 배치)

**파일**: `ai2holodeck/generation/floor_objects.py`

#### Step 1: Constraint Solver가 2D 좌표 계산
- **함수**: `DFS_Solver_Floor.get_solution()` (490줄)
- **입력**: 
  - `bounds`: 방의 polygon
  - `objects_list`: 객체 리스트 (이름, 크기)
  - `constraints`: constraint 딕셔너리
  - `initial_state`: 문/창문 위치
- **출력**: `solution` 딕셔너리
  - `solution[object_name] = ((x_cm, z_cm), rotation_deg, vertices)`
  - 단위: **센티미터 (cm)**

#### Step 2: Solution을 3D Position으로 변환
- **함수**: `FloorObjectGenerator.solution2placement()` (277-300줄)
- **코드**:
```python
placement["position"] = {
    "x": solution[0][0] / 100,        # cm → m 변환
    "y": dimension["y"] / 2,          # 객체 높이의 절반 (바닥에 놓이도록)
    "z": solution[0][1] / 100,        # cm → m 변환
}
placement["rotation"] = {"x": 0, "y": solution[1], "z": 0}
```

**좌표계**:
- `x`, `z`: 방의 vertices 기준 (미터 단위)
- `y`: 객체 높이의 절반 (바닥면이 y=0에 오도록)

---

### use_constraint=False (LLM Baseline 배치)

**파일**: `ai2holodeck/generation/floor_objects.py`

#### Step 1: LLM이 직접 위치 생성
- **함수**: `FloorObjectGenerator.generate_objects_per_room()` (154-213줄)
- **Prompt**: `floor_baseline_prompt` (LLM에게 직접 위치 요청)
- **LLM 출력**: `{"object_name": "...", "position": {"X": 120, "Y": 200}, "rotation": 90}`
  - 단위: **센티미터 (cm)**
  - 좌표계: 방의 bottom-left corner가 (0, 0)

#### Step 2: Room Origin 기준으로 변환
- **코드** (204-208줄):
```python
room_origin = [
    min(v[0] for v in room["vertices"]),  # 방의 최소 x 좌표
    min(v[1] for v in room["vertices"]),  # 방의 최소 z 좌표
]

placement["position"] = {
    "x": room_origin[0] + (data["position"]["X"] / 100),  # room origin + LLM 좌표
    "y": dimension["y"] / 2,                               # 객체 높이의 절반
    "z": room_origin[1] + (data["position"]["Y"] / 100),  # room origin + LLM 좌표
}
```

**좌표계**:
- LLM은 방의 bottom-left를 (0, 0)으로 가정
- 실제 좌표는 `room_origin`을 더해서 계산

---

## 2. Wall Objects (벽 객체)

**파일**: `ai2holodeck/generation/wall_objects.py`

#### Step 1: Constraint Solver가 3D 좌표 계산
- **함수**: `DFS_Solver_Wall.get_solution()` (389줄)
- **출력**: `solution` 딕셔너리
  - `solution[object_name] = ((x_min, y_min, z_min), (x_max, y_max, z_max), rotation, vertices)`
  - 단위: **센티미터 (cm)**

#### Step 2: Bounding Box 중심점 계산
- **함수**: `WallObjectGenerator.solution2placement()` (328-357줄)
- **코드** (336-341줄):
```python
position_x = (solution[0][0] + solution[1][0]) / 200  # (x_min + x_max) / 2, cm → m
position_y = (solution[0][1] + solution[1][1]) / 200  # (y_min + y_max) / 2, cm → m
position_z = (solution[0][2] + solution[1][2]) / 200  # (z_min + z_max) / 2, cm → m

placement["position"] = {"x": position_x, "y": position_y, "z": position_z}
placement["rotation"] = {"x": 0, "y": solution[2], "z": 0}
```

**좌표계**:
- Bounding box의 중심점 사용
- Collision 방지를 위해 약간의 offset 추가 (343-351줄)

---

## 3. Small Objects (작은 객체)

**파일**: `ai2holodeck/generation/small_objects.py`

#### AI2-THOR Controller를 통한 물리 시뮬레이션
- **함수**: `SmallObjectGenerator.place_object()` (68줄)
- **방식**: AI2-THOR의 실제 물리 엔진 사용
- **코드** (74-83줄):
```python
obj = self.place_object(controller, asset_id, receptacle, rotation)
placement["position"] = obj["position"]  # Controller가 계산한 위치
placement["position"]["y"] = (
    obj["position"]["y"] + (asset_height / 2) + 0.001
)  # 높이 조정
```

**좌표계**:
- AI2-THOR의 실제 3D 좌표계
- Receptacle (예: sofa, table) 위에 배치

---

## 4. Ceiling Objects (천장 객체)

**파일**: `ai2holodeck/generation/ceiling_objects.py`

#### 방의 중심점 사용
- **함수**: `CeilingObjectGenerator.generate_ceiling_objects()` (66-83줄)
- **코드** (69-77줄):
```python
floor_polygon = Polygon(room["vertices"])
x = floor_polygon.centroid.x  # 방의 중심 x
z = floor_polygon.centroid.y  # 방의 중심 z (Polygon은 2D이므로 y가 z)
y = scene["wall_height"] - dimension["y"] / 2  # 천장 높이 - 객체 높이/2

ceiling_object["position"] = {"x": x, "y": y, "z": z}
```

**좌표계**:
- 방의 중심점 (centroid)
- y는 천장 높이 기준

---

## 좌표 변환 요약

### 단위 변환
- **Solver 내부**: 센티미터 (cm)
- **최종 JSON**: 미터 (m)
- **변환**: `/ 100` 또는 `/ 200` (wall objects는 bounding box 중심이므로)

### 좌표계 기준점
- **Floor objects**: `room_origin` (방의 최소 x, z 좌표)
- **Wall objects**: 방의 vertices 기준
- **Small objects**: AI2-THOR의 실제 3D 좌표계
- **Ceiling objects**: 방의 중심점 (centroid)

### Y 좌표 (높이)
- **Floor objects**: `dimension["y"] / 2` (바닥면이 y=0)
- **Wall objects**: Solver가 계산한 y_min ~ y_max의 중심
- **Small objects**: Receptacle 위 + 객체 높이/2
- **Ceiling objects**: `wall_height - dimension["y"] / 2`

---

## 주요 파일 위치

1. **Floor Objects (Constraint)**: `floor_objects.py:277-300` (`solution2placement`)
2. **Floor Objects (Baseline)**: `floor_objects.py:204-208` (position 계산)
3. **Wall Objects**: `wall_objects.py:336-341` (position 계산)
4. **Small Objects**: `small_objects.py:74-83` (Controller 사용)
5. **Ceiling Objects**: `ceiling_objects.py:69-77` (centroid 사용)
6. **Constraint Solver**: `floor_objects.py:490` (`DFS_Solver_Floor.get_solution`)


