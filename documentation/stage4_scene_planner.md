# Стадия 4 — Scene Planner: Подробный анализ

Файл: `project/scene_planner.py`  
Точка входа: `generate_placement_plan(models, room_half_size, room_type, query)`

---

## Общая схема работы

```
generate_placement_plan()
│
├── _generate_plan_in_batches()        ← LLM генерирует семантический план
│   ├── если ≤ 8 объектов:
│   │   └── _generate_plan()           ← единственный LLM вызов
│   └── если > 8 объектов:
│       ├── _determine_priorities()    ← LLM делит на anchors/dependents
│       ├── _generate_plan()           ← генерирует план для anchors
│       ├── _group_by_anchor()         ← распределяет dependents по anchors
│       └── _generate_plan() × N      ← генерирует план для каждой группы
│
├── Инъекция model_loc                 ← добавляет пути к файлам из input
│
├── _DistanceResolver.resolve_positions()
│   ├── строит граф зависимостей
│   ├── проверяет циклы (DFS)
│   ├── топологическая сортировка
│   └── конвертирует relative → absolute для каждого объекта
│
├── _OrientationResolver.resolve_orientations()
│   └── конвертирует relative orientation → абсолютный yaw_deg
│
├── _PlacementExecutor.execute()
│   └── извлекает финальные позиции из плана
│
├── Merge step                         ← мёржит original model fields + placement
│
└── validate_and_repair_layout()       ← физическая коррекция коллизий
    ├── Density check (footprint / room area)
    ├── если плотность > 0.8 → LLM fallback сразу
    ├── gradient_resolve_overlaps()    ← OBB/SAT + gradient push
    └── если коллизии остались → _llm_replan_layout()
```

---

## Шаг 1 — Вход: `generate_placement_plan()`

**Входные данные:**
- `models` — список моделей после стадии 2 (model_loader), каждая содержит:
  - `Model` — имя модели
  - `size` — `[width, height, depth]` в метрах
  - `model_loc` — путь к GLB файлу
  - `scale` — итоговый масштаб
  - `uuid` — уникальный идентификатор
  - `_up_axis` — `"y"` или `"z"` (ось вверх модели)
- `room_half_size` — половина стороны комнаты в метрах (например, 3.5 → комната 7×7м)
- `room_type` — тип комнаты (`"dining"`, `"office"`, `"living"`, ...)
- `query` — исходный пользовательский запрос

**Первое действие** — оборачивает `_llm_request` в `llm_fn(prompt, query_text)` с унифицированной сигнатурой.

---

## Шаг 2 — Генерация семантического плана: `_generate_plan_in_batches()`

### Когда одна попытка (≤ 8 объектов)

Вызывается `_generate_plan()` один раз. Весь список моделей упаковывается в один промпт.

### Когда батчи (> 8 объектов)

#### 2.1 Определение приоритетов: `_determine_priorities()`

Отправляет LLM специальный промпт `_PLACEMENT_PRIORITY_PROMPT` с вопросом:  
*"Раздели эти объекты на anchor objects (мебель-якоря) и dependent objects (объекты, зависящие от якорей)"*

LLM возвращает JSON:
```json
{
  "anchor_objects": [{"index": 0, "reason": "Main table"}],
  "dependent_objects": [{"index": 1, "reason": "Chair around table"}]
}
```

**Эвристический fallback**: если LLM ломается — объекты с `"table"`, `"sofa"`, `"bed"`, `"desk"` в имени становятся anchors, остальные — dependents.

#### 2.2 Генерация плана для anchors (Batch 1)

`_generate_plan()` вызывается только для anchor моделей с `skip_validation=True` (валидация схемы не происходит для промежуточных результатов).

#### 2.3 Распределение dependents: `_group_by_anchor()`

Distributes dependent objects evenly across already-placed anchor objects:
- N anchors, M dependents → каждый anchor получает `M // N` объектов, первые `M % N` anchors получают на 1 больше

#### 2.4 Генерация планов для групп (Batch 2, 3, ...)

Для каждой anchor-группы формируется расширенный запрос:
```
{исходный query}. 

ALREADY PLACED OBJECTS (use these IDs for relative positioning):
  - table_1: Dining Table (size: 1.50m x 1.00m x 0.80m) at x=0.0m, y=0.0m <- TARGET ANCHOR
  - ...

Now place the following new objects using relative positioning to 'table_1'.
```

Это позволяет LLM знать позиции уже размещённых объектов и правильно расставить стулья/лампы вокруг конкретного стола.

---

## Шаг 3 — Единичная генерация плана: `_generate_plan()`

### Формирование промпта

`_build_models_str(models)` форматирует список моделей:
```
  - Model: Dining Chair
    Size: {'width': 0.55, 'length': 0.55, 'height': 0.90}
    uuid: abc123
  
  - Model: Dining Chair
    Size: {'width': 0.55, 'length': 0.55, 'height': 0.90}
    uuid: def456

CRITICAL: You MUST generate EXACTLY 2 objects in your semantic plan.
```

В промпте `_SEMANTIC_PLAN_PROMPT` указаны:
- Размеры комнаты (width, length, height=3.0м)
- Все доступные модели с размерами
- Полная спецификация форматов позиций и ориентаций
- Правила (объектов ровно столько же, сколько моделей в списке)

### Цикл с ретраями (до 3 попыток)

На каждой попытке:
1. Вызов LLM → получение JSON
2. Валидация через `_SchemaValidator`
3. Если ошибка → промпт обновляется текстом ошибок, следующая попытка

---

## Шаг 4 — Валидация схемы: `_SchemaValidator.validate_plan()`

Проверяет структуру JSON от LLM:

| Проверка | Что проверяется |
|---|---|
| Тип | plan — словарь |
| schema_version | Присутствует, значение `"1.0"` |
| room_size | Присутствует, содержит `width`, `length`, `height` |
| objects | Список, не пустой |
| Каждый объект | Наличие полей: `id`, `Model`, `type`, `size`, `is_static`, `position`, `orientation` |
| id | Нет дубликатов |
| size | Содержит `width`, `length`, `height` |
| position | Ровно одно из: `absolute` или `relative` (не оба, не ни одного) |
| orientation | Ровно одно из: `absolute` или `relative` |
| Кросс-ссылки | Все `relative_to` и `facing` ссылаются на существующие `id` |

Если валидация провалилась — ошибки добавляются в следующий промпт как текст:
```
The previous semantic plan had validation errors:
  - Object at index 1: position must specify 'absolute' or 'relative'
  - Object at index 3 missing required field: is_static
Please fix these errors and generate a corrected semantic plan.
```

---

## Шаг 5 — Инъекция model_loc

После получения плана от LLM система добавляет в каждый объект плана реальные данные из входного списка моделей:

```
для каждого obj в semantic_plan["objects"]:
    match = найти модель по Model name из input list
    obj["model_loc"] = match["model_loc"]   ← путь к GLB файлу
    obj["uuid"]      = match["uuid"]
    obj["save_fn"]   = match["save_fn"]
    obj["_up_axis"]  = match["_up_axis"]    ← "y" или "z"
```

Это необходимо, потому что LLM не знает о файловой системе — она оперирует именами, а не путями.

---

## Шаг 6 — Разрешение дистанций: `_DistanceResolver.resolve_positions()`

Конвертирует все `relative` позиции в `absolute` координаты.

### 6.1 Построение графа зависимостей

Для каждого объекта с `relative` позицией строится зависимость:
```
"chair_1" depends on "table_1"
"lamp_1" depends on "table_1"
"book_1" depends on "shelf_1"
```

### 6.2 Проверка циклов (DFS)

Обход в глубину — если найден цикл (A зависит от B, B зависит от A) → `RuntimeError`.

### 6.3 Топологическая сортировка

Порядок обработки:
1. Сначала все объекты с `absolute` позицией (они уже готовы)
2. Затем объекты с `relative` позицией в порядке зависимостей (Kahn's algorithm)

### 6.4 Конвертация relative → absolute: `_calc_abs()`

Для каждого объекта с `relative` позицией:

```python
# 1. Получить опорную точку target объекта
ref_point = _get_ref_point(target, reference_point="center")

# 2. Получить вектор направления (rotated by target's yaw)
dx, dy = _dir_vec(direction="front", target=target)

# 3. Вычислить координаты
x = ref_point.x + dx * distance
y = ref_point.y + dy * distance
z = _calc_z(obj, target, direction)
```

**Опорные точки** (`reference_point`):
| Значение | Точка |
|---|---|
| `"center"` | Геометрический центр target |
| `"front_edge"` | Центр передней грани (y + length/2) |
| `"back_edge"` | Центр задней грани (y - length/2) |
| `"left_edge"` | Центр левой грани (x - width/2) |
| `"right_edge"` | Центр правой грани (x + width/2) |
| `"top_surface"` | Центр верхней грани (z + height/2) |

**Вектора направлений** (до поворота target'а):
| Direction | dx | dy |
|---|---|---|
| `front` | 0 | +1 |
| `back` | 0 | -1 |
| `left` | -1 | 0 |
| `right` | +1 | 0 |
| `front_left` | -0.707 | +0.707 |
| `front_right` | +0.707 | +0.707 |
| `back_left` | -0.707 | -0.707 |
| `back_right` | +0.707 | -0.707 |
| `above` / `below` | 0 | 0 |

Финальный вектор вращается на yaw_deg target'а:
```
dx_rot = dx * cos(yaw) - dy * sin(yaw)
dy_rot = dx * sin(yaw) + dy * cos(yaw)
```

**Вычисление Z** (`_calc_z()`):
- `direction == "above"`: `ref_z + target_height/2 + obj_height/2` (объект лежит сверху на surface)
- `direction == "below"`: `ref_z - target_height/2 - obj_height/2`
- `type == "furniture"`: `obj_height / 2` (стоит на полу, центр на высоте half-height)
- `type == "small_object"` рядом с `furniture`: `surface_z + obj_height/2` (объект стоит на поверхности мебели)

---

## Шаг 7 — Разрешение ориентаций: `_OrientationResolver.resolve_orientations()`

Конвертирует `relative` ориентации (`facing: "table_1"`) в абсолютный `yaw_deg`.

### Алгоритм: `_calc_facing()`

```
1. Определить "сторону" target, куда должен смотреть объект:
   side_angle = radians(target_yaw + offset[facing_direction])
   
   offset["front"]       =    0°
   offset["back"]        =  180°
   offset["left_side"]   =   90°
   offset["right_side"]  =  -90°
   offset["front_left"]  =   45°
   offset["front_right"] =  -45°
   offset["back_left"]   =  135°
   offset["back_right"]  = -135°

2. Вычислить "цель взгляда":
   side_x = target.x + sin(side_angle)
   side_y = target.y + cos(side_angle)

3. Вычислить yaw объекта:
   dx = side_x - obj.x
   dy = side_y - obj.y
   yaw = degrees(atan2(dx, dy))

4. Если facing_away=True: yaw += 180°
```

**Пример**: стул находится перед столом, `facing: "table_1"`, `facing_direction: "front"`.  
Стул смотрит на front-сторону стола → yaw такой, что стул "смотрит на стол".

Если LLM возвращает неизвестный `facing_direction` → WARNING + fallback к `"front"` (не crash).

---

## Шаг 8 — Исполнение: `_PlacementExecutor.execute()`

Самый простой шаг — извлекает из resolved плана данные и возвращает список:

```python
{
    "id": "chair_1",
    "Model": "Dining Chair",
    "type": "furniture",
    "size": {"width": 0.55, "length": 0.55, "height": 0.90},
    "is_static": True,
    "model_loc": "/path/to/chair.glb",
    "uuid": "abc123",
    "save_fn": "chair_uuid_1",
    "_up_axis": "y",
    "position": {"x": 0.75, "y": 0.0, "z": 0.45},
    "orientation": {"yaw_deg": 180.0, "pitch_deg": 0.0, "roll_deg": 0.0},
}
```

---

## Шаг 9 — Merge Step

Мёржит данные из `PlacementExecutor` с оригинальными данными модели (из model_loader):

```python
merged = dict(original_model)   # несёт все оригинальные поля: scale, uuid, model_loc, ...
merged.update({
    "Model": placed_obj["Model"],
    "size": [sz["width"], sz["length"], sz["height"]],   # ← list, не dict
    "is_static": placed_obj["is_static"],
    "Pose": {"x": pos["x"], "y": pos["y"], "z": pos["z"]},
    "yaw_deg": ori["yaw_deg"],
    "save_fn": f"{safe_uuid}_{i}",   # ← уникальный суффикс _i на случай дубликатов
})
```

Ключевой момент: `save_fn` генерируется как `{uuid}_{i}` — это предотвращает конфликты имён когда один и тот же тип объекта (5 стульев) имеет одинаковое базовое имя.

---

## Шаг 10 — Физическая коррекция: `validate_and_repair_layout()`

### 10.1 Проверка z-координаты

Все объекты с `z < 0.01` поднимаются до `z = 0.01` (нет объектов под полом).

### 10.2 Density Check

```python
total_footprint = sum(width * depth для каждого объекта)
available_area  = (2 * room_half_size)²
density_ratio   = total_footprint / available_area
```

Если `density_ratio > 0.8` (≥80% площади занято) → **градиент не сойдётся**, сразу идём к LLM fallback.

Пример: 6 стульев по 0.5×0.5м = 1.5м² в комнате 7×7м = 49м² → density = 0.031 (нормально).  
Пример: 20 больших диванов 3×2м = 120м² в той же комнате → density = 2.45 (невозможно).

### 10.3 Градиентное разрешение коллизий: `gradient_resolve_overlaps()`

#### Адаптивное количество итераций:
```python
base_iterations = repair_iters * 20   # = 8 * 20 = 160
if num_objects > 5:
    bonus = (num_objects - 5) * 50
adaptive = min(base + bonus, 500)
```

При 6 объектах: 160 + 50 = 210 итераций.  
При 10 объектах: 160 + 250 = 410 итераций.  
Максимум: 500 итераций.

#### Сам алгоритм (одна итерация):

**1. Для каждой пары объектов i, j:**

```
a) Проверить Z-overlap: если объекты на разных уровнях высоты → пропустить
b) Если оба z > 0.5м → пропустить (объекты на поверхностях не толкаем)
c) Построить OBB с inflation=0.02м для обоих объектов
d) Применить SAT (Separating Axis Theorem) через obb_separation_vector()
e) Если depth > 0 (есть перекрытие):
   - Выбрать step_multiplier:
       depth > 0.1м → x6  (сильное перекрытие — сильный толчок)
       depth < 0.02м → x2 (лёгкое касание — мягкий толчок)
       иначе → x4
   - scale = step_multiplier * depth / dist_between_centers
   - grad_i += sep_dir * scale * 0.625
   - grad_j -= sep_dir * scale * 0.625  (равное и противоположное)
```

**2. Boundary correction (объект выходит за стены):**

Если AABB объекта выходит за `±room_half_size` → добавить градиент, тянущий внутрь.

**3. Stagnation detection:**

Если последние 20 итераций количество коллизий не уменьшилось → `converged=False`, ранний выход.

**4. Применение градиентов:**

```python
pose["x"] += step_size * grad.x   # step_size = 0.06м
pose["y"] += step_size * grad.y
```

**5. После всех итераций — финальный clamp:**

Объекты, вышедшие за границы, жёстко прижимаются к стенам.
Z-коррекция: если `z` близко к `half_height` (±10%) → выравнивается точно на `half_height` (ставим на пол).

### 10.4 Алгоритм SAT (Separating Axis Theorem)

Используется для точного обнаружения перекрытия двух OBB в 2D:

1. Для каждого из 4 возможных осей разделения (2 оси OBB_A + 2 оси OBB_B):
   - Проецировать все 4 угла каждого OBB на эту ось
   - Вычислить `overlap = min(max_a, max_b) - max(min_a, min_b)`
   - Если `overlap < 0` → оси разделяют фигуры → нет коллизии (ранний выход)
   - Запоминать минимальный overlap

2. `obb_overlap_depth()` возвращает минимальный overlap через все 4 оси (>0 = коллизия)

3. `obb_separation_vector()` возвращает (ось, глубина) для минимального перевода, которое устранит коллизию

### 10.5 LLM Fallback: `_llm_replan_layout()`

Если градиент не смог устранить все коллизии:

**Промпт содержит:**
- Все объекты с текущими позициями и размерами
- Список конкретных коллизий: `{object_a, object_b, overlap_depth}`
- Границы комнаты
- Правила: не менять Z, зазор >= 0.05м

**LLM возвращает:**
```json
{
  "positions": [
    {"index": 0, "x": 1.5, "y": 0.0, "z": 0.4},
    {"index": 1, "x": -1.2, "y": 0.5, "z": 0.4}
  ],
  "reasoning": "Moved chairs apart to avoid overlap"
}
```

**После применения:** проверяется `check_collisions()`. Если коллизии ещё есть → retry с feedback.

Максимум 3 попытки. При частичном успехе (коллизии остались) — возвращает результат с пометкой `True` (best effort).

---

## OBB/SAT геометрические примитивы

### Vec2 — 2D вектор

```
+, -, * (scalar), dot(), length(), normalized()
```

### OBB — Oriented Bounding Box в плоскости XY

```
center: Vec2
half_extents: Vec2   ← [width/2, depth/2]
yaw_rad: float

corners() → 4 угла OBB
axes()    → 2 оси (unit vectors вдоль сторон)
aabb()    → AABB (axis-aligned bounding box для быстрой предварительной проверки)
```

### model_to_obb(model, inflation=0.0)

Строит OBB из словаря модели:
- `size[0]` → ширина (X)
- `size[2]` → глубина (Y) — не `size[1]`! height это Y в GLB, а depth это Z
- Минимальный размер 0.02м

### model_half_height(model)

```python
height = size[1]   # Y-axis в GLB = высота объекта
# Эвристика: если height < 5% от max(size) → использовать max(size)
# (защита от вырожденных mешей)
return max(0.01, height / 2.0)
```

### _z_intervals_overlap(a, b)

Проверяет пересечение объектов по вертикали (Z). Используется перед SAT проверкой — нет смысла проверять 2D коллизию между объектами на разных уровнях высоты.

---

## Итоговый формат выходных данных

Каждый объект в `result` содержит:

```python
{
    # Из model_loader (Stage 2)
    "Model": "Dining Chair",
    "uuid": "abc123",
    "model_loc": "/path/to/datasets/.../chair.glb",
    "scale": 0.9542,
    "_up_axis": "y",
    
    # Из scene_planner (Stage 4)
    "size": [0.55, 0.90, 0.55],     # [width, height, depth] в метрах
    "is_static": True,
    "Pose": {"x": 0.75, "y": 0.0, "z": 0.45},
    "yaw_deg": 180.0,
    "save_fn": "abc123_1",           # уникальное имя для записи на диск
}
```

Этот формат передаётся в Stage 5 (mujoco_assembler).

---

## Логи (что видно при запуске)

```
[scene_planner] Stage 4: generating placement plan for 6 objects
[scene_planner] Room: 7.0m x 7.0m, type: dining
[scene_planner] 6 objects, using single-pass generation
[scene_planner] Generating semantic plan (attempt 1/3)
[scene_planner] Schema validation passed on attempt 1
[scene_planner] LLM produced 6 objects
[scene_planner]   Dining Table: model_loc=/home/.../table.glb
[scene_planner]   Dining Chair: model_loc=/home/.../chair.glb
...
[scene_planner] Distance resolution done
[scene_planner] Orientation resolution done
[scene_planner] Placed 6 objects
[scene_planner]   Dining Table: pos=(0.00, 0.00, 0.40), yaw=0deg
[scene_planner]   Dining Chair: pos=(0.75, 0.00, 0.45), yaw=180deg
...
[scene_planner] Stage 4 complete: 6 models with placement

[validate_and_repair] Starting with 6 models...
[validate_and_repair] Using adaptive iteration limit: 210 iterations (base=160, objects=6)
[validate_and_repair] Density check: footprint=2.15m², available=49.00m², ratio=0.04
[gradient_resolve] Converged successfully at iteration 3
[validate_and_repair] SUCCESS: All collisions resolved after gradient resolution
```

---

## Известные ограничения

1. **Batch generation не сшивает планы**: объекты из разных батчей могут иметь конфликтующие позиции — это исправляется в `validate_and_repair_layout`.

2. **Относительная ориентация требует предварительно разрешённые позиции**: `_OrientationResolver` вызывается ПОСЛЕ `_DistanceResolver`. Если порядок нарушить — будет `ValueError`.

3. **Gradient resolver пропускает объекты с z > 0.5м**: предполагается, что они стоят на мебели и не должны быть сдвинуты в горизонтальной плоскости.

4. **LLM fallback не сохраняет Z**: при LLM переплануровании Z-координаты объектов не меняются — сохраняются оригинальные значения.

5. **Один save_fn на инстанс**: даже если 5 стульев используют одинаковый GLB файл, каждый получает уникальный `save_fn` вида `{uuid}_{0}`, `{uuid}_{1}` и т.д.
