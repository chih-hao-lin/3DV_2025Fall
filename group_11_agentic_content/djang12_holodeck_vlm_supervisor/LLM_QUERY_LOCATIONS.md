# LLM Query Locations in Holodeck

이 문서는 Holodeck에서 LLM query를 날리는 모든 위치를 정리합니다.

## 1. Floor Plan Generation (방 레이아웃 생성)

**파일**: `ai2holodeck/generation/rooms.py`  
**함수**: `FloorPlanGenerator.generate_rooms()`  
**라인**: 50

```python
# rooms.py:50
raw_floor_plan = self.llm(floor_plan_prompt)
```

**용도**: 사용자 query를 기반으로 방 레이아웃(floor plan) 생성  
**Prompt**: `prompts.floor_plan_prompt`  
**호출 위치**: `holodeck.generate_rooms()` → `holodeck.generate_scene()`

---

## 2. Object Selection (객체 선택)

**파일**: `ai2holodeck/generation/object_selector.py`  
**함수**: `ObjectSelector.plan_room()`  
**라인**: 166, 204

### 첫 번째 LLM 호출 (라인 166)
```python
# object_selector.py:166
output_1 = self.llm(prompt_1).lower()
```

**용도**: 각 방에 대한 객체 선택 계획 생성  
**Prompt**: `prompts.object_selection_prompt_new_1`  
**입력**: room_type, room_size, requirements

### 두 번째 LLM 호출 (라인 204) - 재계획
```python
# object_selector.py:204
output_2 = self.llm(prompt_2).lower()
```

**용도**: 첫 번째 계획이 floor capacity를 80% 이상 채우지 못할 때 추가 객체 선택  
**Prompt**: `prompts.object_selection_prompt_new_2`  
**조건**: `floor_capacity[1] / floor_capacity[0] < 0.8`

**호출 위치**: `holodeck.select_objects()` → `holodeck.generate_scene()`

---

## 3. Door Generation (문 생성)

**파일**: `ai2holodeck/generation/doors.py`  
**함수**: `DoorGenerator.generate_doors()`  
**라인**: 106

```python
# doors.py:106
raw_doorway_plan = self.llm(doorway_prompt)
```

**용도**: 방들 사이의 문 배치 계획 생성  
**Prompt**: `prompts.doorway_prompt`  
**입력**: room_types, room_sizes, room_pairs  
**호출 위치**: `holodeck.generate_doors()` → `holodeck.generate_scene()`

---

## 4. Window Generation (창문 생성)

**파일**: `ai2holodeck/generation/windows.py`  
**함수**: `WindowGenerator.generate_windows()`  
**라인**: 58

```python
# windows.py:58
raw_window_plan = self.llm(window_prompt)
```

**용도**: 각 방의 창문 배치 계획 생성  
**Prompt**: `prompts.window_prompt`  
**입력**: walls, wall_height  
**호출 위치**: `holodeck.generate_windows()` → `holodeck.generate_scene()`

---

## 5. Floor Object Placement Constraints (바닥 객체 배치 제약)

**파일**: `ai2holodeck/generation/floor_objects.py`  
**함수**: `FloorObjectGenerator.generate_objects_per_room()`  
**라인**: 174 (baseline), constraint 부분도 확인 필요

### Baseline Placement (라인 174)
```python
# floor_objects.py:174
completion_text = self.llm(baseline_prompt)
```

**용도**: `use_constraint=False`일 때 객체 배치 위치 생성  
**Prompt**: `prompts.floor_baseline_prompt`  
**조건**: `use_constraint=False`

### Constraint-based Placement
Constraint를 사용할 때는 LLM 대신 DFS solver나 MILP solver를 사용합니다.

**호출 위치**: `holodeck.generate_scene()` → `floor_object_generator.generate_objects()`

---

## 6. Wall Object Placement Constraints (벽 객체 배치 제약)

**파일**: `ai2holodeck/generation/wall_objects.py`  
**함수**: `WallObjectGenerator.generate_wall_objects_per_room()`  
**라인**: 116

```python
# wall_objects.py:116
constraint_plan = self.llm(constraints_prompt)
```

**용도**: 벽 객체의 배치 위치와 높이 제약 생성  
**Prompt**: `prompts.wall_object_constraints_prompt`  
**조건**: `self.constraint_type == "llm" and use_constraint == True`  
**입력**: room_type, wall_height, floor_objects, wall_objects

**호출 위치**: `holodeck.generate_scene()` → `wall_object_generator.generate_wall_objects()`

---

## LLM 초기화

**파일**: `ai2holodeck/generation/holodeck.py`  
**라인**: 70-74

```python
# holodeck.py:70-74
self.llm = OpenAI(
    model_name=LLM_MODEL_NAME,
    max_tokens=2048,
    openai_api_key=openai_api_key,
)
```

**모델**: `LLM_MODEL_NAME` (constants에서 정의)  
**Max Tokens**: 2048

---

## 전체 호출 순서

```
holodeck.generate_scene()
  ├─> generate_rooms()
  │     └─> LLM: Floor plan generation (rooms.py:50)
  │
  ├─> generate_doors()
  │     └─> LLM: Doorway plan (doors.py:106)
  │
  ├─> generate_windows()
  │     └─> LLM: Window plan (windows.py:58)
  │
  ├─> select_objects()
  │     └─> plan_room() (각 방마다)
  │           ├─> LLM: Object selection #1 (object_selector.py:166)
  │           └─> LLM: Object selection #2 (object_selector.py:204) [조건부]
  │
  ├─> floor_object_generator.generate_objects()
  │     └─> generate_objects_per_room()
  │           └─> LLM: Baseline placement (floor_objects.py:174) [use_constraint=False일 때]
  │
  └─> wall_object_generator.generate_wall_objects()
        └─> generate_wall_objects_per_room()
              └─> LLM: Wall object constraints (wall_objects.py:116) [constraint_type=="llm"일 때]
```

---

## 주요 Prompt 템플릿 위치

모든 prompt 템플릿은 `ai2holodeck/generation/prompts.py`에 정의되어 있습니다:

- `floor_plan_prompt`: 방 레이아웃 생성
- `object_selection_prompt_new_1`: 객체 선택 (1차)
- `object_selection_prompt_new_2`: 객체 선택 (2차, 재계획)
- `doorway_prompt`: 문 배치
- `window_prompt`: 창문 배치
- `floor_baseline_prompt`: 바닥 객체 배치 (baseline)
- `object_constraints_prompt`: 객체 배치 제약
- `wall_object_constraints_prompt`: 벽 객체 배치 제약


