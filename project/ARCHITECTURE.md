# Архитектура системы генерации 3D-сцен — полный разбор

## Обзор

Система принимает короткий текстовый запрос пользователя и генерирует MuJoCo XML сцену с физически корректно расставленными 3D-объектами. Процесс состоит из 6 последовательных стадий.

```
Запрос → [Stage 0] SceneSpec → [Stage 1] Models → [Stage 2] Scaled → [Stage 3] Room →
         [Stage 4] Placement → [Stage 5] MuJoCo XML
```

LLM работает как **семантический планировщик** (что куда, как повернуть, какие отношения), а код — как **геометрический исполнитель** (точные координаты Z, проверка границ, коллизии).

---

## Stage 0 — Расширение запроса (`prompt_expander.py`)

**Вход:** строка запроса пользователя ("Стол и 4 стула")  
**Выход:** объект `SceneSpec`

LLM получает системный промпт из `prompts/expand_prompt.txt` и возвращает JSON:

```json
{
  "expanded_description": "Детальное описание сцены (2-3 предложения)",
  "room_type": "office|bedroom|kitchen|living_room|classroom|warehouse|lab|outdoor|other",
  "room_style": "modern|minimalist|cozy|industrial|academic|other",
  "estimated_objects": [
    {"name": "table", "quantity": 1, "notes": "center"},
    {"name": "chair", "quantity": 4, "notes": "beside table, facing it"}
  ],
  "room_dimensions_hint": "medium (5x5m)"
}
```

### Ограничения (жёстко заданы в промпте)
- Максимум **20 экземпляров** объектов суммарно
- Максимум **6 уникальных типов** для полной сцены
- `name` — только атомарный объект (`"chair"`, не `"set of chairs"`)

### Парсинг
После получения JSON:
1. Валидация `room_type` по белому списку
2. Если сумма экземпляров > 20 — пропорциональное урезание
3. Парсинг `room_dimensions_hint` → `room_half_size`:
   - `small` → 2.0, `medium` → 2.5, `large` → 4.0, `extra_large` → 6.0
   - Или парсинг "NxM" → N/2

---

## Stage 1 — Подбор 3D-моделей (`model_picker.py`)

**Вход:** `SceneSpec.estimated_objects` + каталог моделей  
**Выход:** список `[{Model, uuid, model_loc, _query_name}, ...]`

### Алгоритм выбора модели

Для каждого объекта из `estimated_objects`:

1. **Fallback-замена** имени через словарь `_FALLBACKS`:
   ```python
   _FALLBACKS = {
       "book": "box",
       "sofa": "lounge chair",
       "couch": "lounge chair",
       "tv": "television",
       "tv stand": "table",
       # ...
   }
   ```

2. **Ранжирование** (`_rank`): сортировка каталога по `_score()`:
   - `+20` — точное совпадение имени
   - `+12` — имя содержит строку объекта
   - `+6` — совпадение в тегах/категориях
   - `+4` / `+2` — за каждый токен в имени / в тегах
   - `+5` — токен совпадает с категорией
   - `-15` — категория `abstract`, `icon`, `logo`, `ui`
   - `-10` — `miniature`, `toy` в имени (если не ищем игрушку)
   - `-12` — `wheelchair` при поиске `chair`
   - `-15` — `lamp/light/fan` в имени при поиске `desk/table`
   - Порог: модели со score < 3 отбрасываются

3. **LLM-выбор** (`_llm_pick`): если кандидатов > 1, отправляем топ-10 в LLM.  
   Промпт просит выбрать лучший UUID или вернуть `{"uuid": "none"}`.

4. **Кэширование**: один тип объекта → одна модель. Все экземпляры `chair` получают одну и ту же модель.

---

## Stage 2 — Загрузка и масштабирование (`model_loader.py`)

Загружает GLB-файлы, вычисляет реальные размеры в метрах, добавляет к каждой модели:
- `size: [width, height, depth]` — размеры в метрах
- `scale` — коэффициент масштабирования

---

## Stage 3 — Расчёт размера комнаты (`room_scaler.py`)

**Вход:** список моделей с `size`, тип комнаты  
**Выход:** `room_half_size` (число, половина стороны квадратной комнаты)

### Формула
```
total_footprint = Σ (width × depth) для каждого объекта
room_area = total_footprint × circulation_multiplier[room_type]
room_half = sqrt(room_area) / 2
room_half = clamp(room_half, MIN_ROOM_HALF, 25.0)
room_half = round(room_half * 2) / 2.0   # кратно 0.5м
```

### Коэффициенты circulation и минимальные размеры

| room_type    | circulation | min_half |
|--------------|-------------|----------|
| bedroom      | 8×          | 2.0 м    |
| office       | 8×          | 2.5 м    |
| classroom    | 8×          | 3.0 м    |
| kitchen      | 10×         | 1.8 м    |
| living_room  | 8×          | 2.5 м    |
| warehouse    | 6×          | 3.0 м    |
| lab          | 8×          | 2.5 м    |
| other        | 10×         | 3.0 м    |

---

## Stage 4 — Планирование расстановки (`scene_planner.py`) — ГЛАВНАЯ СТАДИЯ

Центральная стадия системы. Разбита на четыре под-фазы.

### 4.1 — Построение графа сцены (`_build_scene_graph`)

LLM получает промпт `_GRAPH_HIERARCHY_PROMPT` с:
- запросом пользователя
- размерами комнаты
- списком доступных объектов и их количеств
- суммой экземпляров для верификации

LLM возвращает JSON с массивом узлов (`nodes`). Узел графа:

```json
{
  "id": "chair_group_1",
  "model_name": "chair",
  "instances": 4,
  "parent_id": "table_1",
  "anchor_hint": null,
  "relationship": {
    "type": "around",
    "reference": "table_1",
    "distance": 0.5,
    "facing": "inward"
  },
  "place_order": 2,
  "depends_on": "table_1"
}
```

#### Типы связей (relationship.type)

| Тип          | Смысл                                      | Z-поведение      |
|--------------|--------------------------------------------|------------------|
| `on_surface` | Объект лежит на поверхности родителя       | На верху родителя |
| `stacked_on` | Стопкой поверх другого объекта             | На верху родителя |
| `around`     | Кольцом вокруг  | Пол (floor)      |
| `beside`     | Рядом с reference                          | Как у родителя   |
| `in_front`   | Перед reference                            | Как у родителя   |
| `behind`     | Позади reference                           | Как у родителя   |
| `facing`     | Лицом к reference                          | Как у родителя   |

#### Правила для LLM при построении графа

- **GROUP RULE**: одинаковые объекты с одним `parent_id` и одним типом связи → **один** узел с `instances>1` (не N отдельных узлов)
- **AROUND RULE**: для `around` — `relationship.reference` должен совпадать с `parent_id`; объекты стоят на полу, не на поверхности якоря
- `depends_on` обязателен если `relationship.reference != null`
- Уровни LLM НЕ задаёт — они вычисляются из цепочки `parent_id`

#### Постобработка графа

После получения ответа от LLM выполняется:

**1. `_normalize_grouped_graph()`** — нормализация в формат групп:
- Pass 1: Все узлы типа `around` с одинаковым `model_name + reference` → объединить в один групповой узел с суммарным `instances`
- Pass 2: Оставшиеся узлы с одинаковой сигнатурой `(model_name, parent_id, rel_type, rel_ref, rel_facing, distance)` → слить в один
- Перезапись `parent_id`/`depends_on`/`reference` по карте замен `id_map`

**2. `_compute_levels()`** — вычисление уровня каждого узла через DFS по цепочке `parent_id`:
- `parent_id = null` → level 0 (якоря, крупная мебель)
- `parent_id = якорь` → level 1
- `parent_id = level-1 объект` → level 2
- Защита от циклов через `seen`-множество

**3. `_validate_graph()`** — структурная валидация:
- Все `parent_id`, `depends_on`, `relationship.reference` ссылаются на реальные ID
- Нет циклов в parent-цепочке
- Все `model_name` из доступного списка
- `instances >= 1`

**4. `_validate_graph_counts()`** — проверка сумм:
- Для каждого `model_name`: `sum(instances всех узлов с этим именем) == count из входных данных`

**При ошибках валидации** — повтор запроса к LLM (максимум 3 попытки) с добавлением списка ошибок в промпт.  
**После 3 неудач** → `_heuristic_graph()` (без LLM):
- Ключевые слова определяют уровень: `table/sofa/bed/desk...` → level 0, `chair/plate/cup/lamp...` → level 1, `apple/book/pen...` → level 2
- Все объекты уровня 1 и 2 привязываются к первому якорю
- relationship: anchors → `around` @ 0.6m, level 2 → `on_surface`

#### Раскрытие множественных якорей (`_expand_anchor_groups`)

Если level-0 узел имеет `instances > 1` (например, 4 одинаковых стола):

1. Разбить на N отдельных узлов: `table_1_1`, `table_1_2`, ...
2. Распределить потомков:
   - Если `instances % N == 0` → равномерно делить (4 стула на 4 стола → 1 стул каждому столу)
   - Иначе → round-robin (каждый узел потомка целиком отдаётся одному якорю)

---

### 4.2 — Инициализация world_state

После построения графа:
1. Инъекция `size_hint: [width, height, depth]` в каждый узел графа (берётся из реальных размеров модели)
2. Инициализация `world_state`:
   ```python
   world_state = {
       "placed": [],          # список уже размещённых объектов
       "remaining": [...],    # счётчики ещё не размещённых
       "room_half_size": 3.5
   }
   ```

---

### 4.3 — Иерархическое размещение (`_place_all`)

Размещение идёт уровень за уровнем: level 0 → level 1 → level 2...

#### Размещение якорей (level 0)

**Если один якорь** с `instances == 1`:
- `_resolve_single_anchor_pos()` без LLM — читает `anchor_hint.region`:
  - `"center"` → (0, 0)
  - `"near_wall"` → y = -(room_half - 0.6), yaw = 90°
  - `"corner"` → (-limit*0.7, -limit*0.7), yaw = 45°
  - Кастомный → из `offset_x`, `offset_y`

**Если один якорь** с `instances > 1`:
- `_place_anchor_batch_node()` → LLM с `_ANCHOR_BATCH_PLACEMENT_PROMPT`
- Батчами по 6 экземпляров
- Запасная позиция при ошибках: `x += idx * 0.8`

**Если несколько якорей** (разные level-0 узлы):
- `_place_anchors_by_llm()` → LLM с `_ANCHOR_RELATIONS_PROMPT`
- LLM получает: все якоря с размерами + описание всех будущих потомков каждого якоря
- LLM должен учесть clearance (запас места) для будущих зависимых объектов
- При ошибке LLM → равномерная сетка: `offset = (i - (n-1)/2) * (room_half/n)`

#### Размещение зависимых объектов (level > 0)

1. Узлы группируются по `parent_id`
2. Внутри каждой группы — топологическая сортировка (`_topological_sort_group`):
   - DFS по `depends_on` ссылкам внутри группы
   - Tie-breaker: `place_order`, затем `id`
3. Для каждого узла вызов `_place_then_validate(node)`:

**Если `instances == 1`** → `_place_single_node()`:
- Строится `placement_instruction` с полным контекстом:
  - Описание связи (тип, reference, side, distance)
  - Параметры родительской поверхности (позиция, размеры, Z верха)
  - Список размещённых объектов-reference с их позициями
  - Специальное предупреждение: если reference-объекты находятся ВНЕ поверхности родителя (стулья вокруг стола), не копировать их X/Y напрямую — использовать направление от центра родителя
- Промпт: `_SINGLE_PLACEMENT_PROMPT` с полным описанием геометрии
- LLM возвращает `{x, y, z, yaw_deg}`
- **Валидация** (до 6 попыток):
  - `_validate_pos()`: x,y в пределах `±room_half * 1.1`
  - `_validate_xy_within_parent_footprint()`: для `on_surface`/`stacked_on` — X,Y внутри footprint родителя
- На попытке 3 добавляется "RELAXED FALLBACK" в промпт

**Если `instances > 1`** → `_place_batch_node()`:
- Батчами по 6 экземпляров
- Для каждого батча: `_BATCH_PLACEMENT_PROMPT` с:
  - Размерами объекта и явными числовыми границами footprint родителя:
    ```
    x ∈ [parent_x - parent_w/2 + half_x + margin, parent_x + parent_w/2 - half_x - margin]
    y ∈ [parent_y - parent_d/2 + half_y + margin, parent_y + parent_d/2 - half_y - margin]
    ```
  - Примерами позиций по 4 сторонам родителя
  - Правилом yaw для `around`: north → 0°, east → 90°, south → 180°, west → 270°
  - Уже размещёнными объектами этой же группы
  - Позициями reference-объектов (если есть)
  - Инструкцией: reference_instance_i → match с placed_instance_i
- **Валидации** после получения батча:
  - Все позиции в пределах комнаты
  - Для `on_surface`: все внутри footprint родителя
  - `_validate_min_clearance_xy()`: расстояние между любой парой ≥ `max(0.05, 0.6 * max(sw, sd))`
- **Fallback**: если батч провалился → `_place_group_sequential_via_llm()` (по одному объекту, до 8 попыток каждый)

#### LLM-валидатор позиции (`_validate_placement_via_llm`)

Применяется только для одиночных узлов (`instances == 1`) на уровне > 0. До 3 раундов.

Промпт `_VALIDATE_PLACEMENT_PROMPT` содержит:
- Точный bounding box только что размещённого объекта (min/max по X,Y,Z)
- Параметры родительской поверхности
- Параметры reference-объекта
- Историю предыдущих раундов (чтобы LLM не осциллировал)
- Все уже размещённые объекты с их bounding box

LLM проверяет три вещи:
1. **Коллизии 3D**: XY OBB overlap И Z-интервалы overlap → только тогда коллизия. Стопка на поверхности (Z just-touching) — НЕ коллизия. Объект на полу и объект на столе — НЕ коллизия (разные Z)
2. **Bounds поверхности**: для `on_surface` — центр объекта внутри footprint родителя. Фикс не должен выводить объект за пределы поверхности
3. **Ориентация**: для длинных объектов (ручки, ножи) — yaw должен быть одинаковым у всех одинаковых объектов

Возвращает либо `{"valid": true}`, либо `{"valid": false, "reason": "...", "fix": {x, y, z, yaw_deg}}`.  
Если фикс выходит за границы комнаты → игнорируется (принимается текущая позиция).

---

### Вычисление Z-координаты (детерминистически, не через LLM)

Z всегда вычисляется кодом в `_register_results()`, независимо от того, что вернул LLM:

```
если relationship.type == "around":
    z = obj_height / 2         # объект стоит на полу

иначе если parent_id != null:
    z = parent_z + parent_height/2 + obj_height/2   # объект стоит на родителе

иначе (якорь, нет родителя):
    z = obj_height / 2         # объект стоит на полу
```

---

### 4.4 — Объединение результатов (`_merge_to_output`)

Сопоставление по очереди: для каждого `placed` объекта из `world_state` берётся очередная оригинальная модель того же типа и объединяется с позицией.

Итоговая запись содержит: `Model, uuid, model_loc, save_fn, _up_axis, size, is_static, Pose{x,y,z}, yaw_deg`

---

### 4.5 — Коррекция коллизий (`validate_and_repair_layout`)

> **Примечание:** На момент написания вызов `validate_and_repair_layout` закомментирован в `generate_placement_plan()` в целях тестирования. Логика реализована и готова к включению.

**Шаг 1: Проверка плотности**
```
density = total_footprint_area / room_area
если density > 0.8 → пропустить градиент, сразу LLM-переплан
```

**Шаг 2: Градиентное разрешение коллизий (`gradient_resolve_overlaps`)**

Итеративный алгоритм, максимум `adaptive_iterations`:
```
adaptive_iters = base_iters*20 + max(0, n-5)*50
adaptive_iters = min(adaptive_iters, 500)
```

На каждой итерации:
1. Для каждой пары `(i, j)` объектов:
   - Пропустить если `_z_intervals_overlap()` возвращает False
   - Пропустить если оба объекта выше 0.5м (предмет на столе и другой предмет на столе — без помощи — см. примечание ниже)
   - Вычислить OBB overlap через SAT (`obb_separation_vector`)
   - Добавить градиент: `grad_i += sep_dir * scale`, `grad_j -= sep_dir * scale`
   - `scale = step_multiplier * depth / dist`, где step_multiplier: 6 (глубокое >0.1м), 4 (средн.), 2 (мелкое <0.02м)
2. Добавить граничные градиенты (выход за комнату)
3. Применить: `pos += step_size * grad`

**Защита поверхностных объектов**: объекты с `relationship.type in (on_surface, stacked_on)` не двигаются в XY.

**Детектор стагнации**: если кол-во overlap пар не уменьшается за последние 20 итераций → досрочный выход с пометкой "не сошлось".

**Шаг 3: LLM-переплан (`_llm_replan_layout`)** при неудаче градиента:
- LLM получает список всех объектов с текущими позициями и список коллизий
- LLM возвращает новые X,Y позиции
- Поверхностные объекты (`on_surface`, `stacked_on`) не перемещаются
- До 3 попыток с обновлением списка оставшихся коллизий в промпте

---

## Система обнаружения коллизий (OBB + SAT)

### Геометрические структуры

**`Vec2`** — 2D вектор с операциями (dot, length, normalized)

**`OBB`** (Oriented Bounding Box в плоскости XY):
```
center: Vec2
half_extents: Vec2
yaw_rad: float
```
- `corners()`: 4 угла бокса в мировых координатах
- `axes()`: 2 оси (cos/sin, -sin/cos)
- `aabb()`: получить AABB из OBB

**`AABB`** — обычный axis-aligned bounding box

### SAT overlap (obb_overlap)

Для каждой из 4 осей (2 от OBB_A + 2 от OBB_B):
- Проецировать все 4 угла каждого бокса на эту ось
- Если проекции не пересекаются → нет коллизии

### Проверка Z-интервалов (_z_intervals_overlap)

- z_diff = |z_A - z_B|
- **Ключевой нюанс**: если `z_diff > 0.15м` → NOT overlap (разные уровни). Это предотвращает ложные коллизии между объектами на столе и объектами на полу.
- Иначе: проверяется пересечение интервалов `[z-h/2, z+h/2]`

### Полная 3D коллизия

3D коллизия = `_z_intervals_overlap(A, B)` AND `obb_overlap(OBB_A, OBB_B)`

---

## Stage 5 — Сборка MuJoCo XML (`mujoco_assembler.py`)

- Конвертирует GLB → OBJ через `obj2mjcf`
- Для каждой модели определяет физические параметры:
  - `is_static` (якоря → static)
  - Плотность материала по типу объекта
  - Трение
- Генерирует `<worldbody>` с `<body>` для каждого объекта
- Сохраняет в `.cache/worlds/scene_latest.xml`

---

## LLM-интерфейс (`llm_request.py`)

**Бэкенд:** Ollama API (`http://localhost:11434`)  
**Модель по умолчанию:** `deepseek-v3.1:671b-cloud`  
**Параметры:**
- `temperature: 0` (детерминизм)
- `seed: 42`
- `num_predict: 4000` (максимум токенов)
- `timeout: 180s`

**Поддержка изображений:** base64 в `images` поле сообщения (для `model_orientation_detector.py`)

**Стриминг:** ответ читается потоково, токены выводятся в консоль

**JSON-парсинг с защитами:**
1. Извлечение из markdown ` ```json ... ``` ` или ` ``` ... ``` `
2. Поиск первой `{` или `[` в тексте
3. Обрезание текста после последней `}` или `]`
4. Санитизация: замена китайских символов в ключах (известная проблема с `deepseek`)
5. Исправление пробелов в ID-полях: `"computer chair_1"` → `"computer_chair_1"`
6. Исправление склеенных объектов: `}{` → `},{`
7. Лимит: 5000 токенов — защита от бесконечной генерации

---

## Промпты и их назначение

| Промпт | Файл | Назначение |
|--------|------|-----------|
| expand_prompt.txt | prompts/ | Stage 0: Расширение запроса пользователя |
| _GRAPH_HIERARCHY_PROMPT | scene_prompts.py | Stage 4.1: Построение иерархического графа |
| _ANCHOR_RELATIONS_PROMPT | scene_prompts.py | Stage 4.3: Размещение нескольких якорей |
| _SINGLE_PLACEMENT_PROMPT | scene_prompts.py | Stage 4.3: Размещение одного объекта |
| _BATCH_PLACEMENT_PROMPT | scene_prompts.py | Stage 4.3: Размещение батча зависимых объектов |
| _ANCHOR_BATCH_PLACEMENT_PROMPT | scene_prompts.py | Stage 4.3: Размещение батча якорей |
| _VALIDATE_PLACEMENT_PROMPT | scene_prompts.py | Stage 4.3: LLM-валидация позиции |
| _DISAMBIGUATION_PROMPT | model_picker.py | Stage 1: Выбор модели из кандидатов |

---

## Система fallback-ов

На каждом уровне предусмотрены запасные варианты:

```
Построение графа:
  LLM attempt 1 → LLM attempt 2 → LLM attempt 3 → Heuristic graph (без LLM)

Размещение одного объекта:
  LLM attempt 1-3 → RELAXED FALLBACK (ослабленные ограничения) attempt 4-6 → RuntimeError

Размещение батча:
  LLM attempt 1-3 → Sequential LLM fallback (по одному, 8 попыток каждый)

LLM-валидатор:
  Round 1 → Round 2 → Round 3 → Accept current position

Разрешение коллизий:
  Density check → Gradient resolution → LLM replan (3 попытки) → Accept with warnings

Подбор якорей:
  LLM → Grid layout fallback
```

---

## Формат world_state

Центральная структура состояния в процессе размещения:

```python
{
  "placed": [
    {
      "id": "table_1_inst_0",
      "Model": "Dining Table",
      "size": [1.2, 0.75, 0.8],    # [width, height, depth] в метрах
      "Pose": {"x": 0.0, "y": 0.0, "z": 0.375},
      "yaw_deg": 0.0
    },
    ...
  ],
  "remaining": [
    {"model_name": "chair", "count": 3, "size": [0.5, 0.9, 0.5]}
  ],
  "room_half_size": 3.5
}
```

**`_ws_summary()`** — форматирует `placed` в текст для LLM-промптов:
```
- Dining Table [table_1_inst_0]: center=(0.000, 0.000, 0.375), yaw=0°,
  X=[-0.600..0.600], Y=[-0.400..0.400], Z=[-0.000..0.750]
```

---

## Координатная система

- **Начало координат**: центр комнаты (0, 0, 0)
- **X**: ось "право" (East)
- **Y**: ось "вперёд" (North)
- **Z**: вертикаль (вверх)
- Границы комнаты: `±room_half_size` по X и Y
- Пол: Z = 0
- `yaw_deg = 0` → объект ориентирован вдоль +X
- `yaw_deg = 90` → объект ориентирован вдоль +Y
- Правило для `around` + yaw: north(y>parent_y)→0°, east→90°, south→180°, west→270°

---

## Сохранение промежуточных результатов

При `save_outputs=True` (по умолчанию) сохраняются JSON-файлы в `project/output/`:

| Файл | Содержимое |
|------|-----------|
| `stage1_scene_graph_*.json` | Граф сцены: nodes, metadata |
| `stage2_world_state_*.json` | world_state после всего размещения |
| `stage3_merged_output_*.json` | Слитые модели с позициями (до коррекции коллизий) |
| `stage4_final_layout_*.json` | Финальный layout + query + room_type |

---

## Нюансы и известные проблемы

1. **`validate_and_repair_layout` закомментирован** в `generate_placement_plan()` — Stage 4.5 не выполняется в текущей конфигурации

2. **deepseek-v3 иногда возвращает китайские символы в JSON-ключах** — обрабатывается в `_sanitize_json_string()`

3. **Объект "around" с `z > 0.5м` пропускается в градиентном разрешении** — оба объекта в паре должны быть ниже 0.5м, иначе коллизия не разрешается. Это может быть проблемой для объектов на высоких столах.

4. **Surface bounds validation использует `epsilon=1e-3`** — LLM часто возвращает значения точно на границе footprint из-за округления в промптах, epsilon предотвращает ложные ошибки.

5. **Ограничение 20 объектов** действует на уровне Stage 0 и Stage 4 — LLM-промпт Stage 4 не знает этого лимита явно, но Stage 0 уже ограничил список.

6. **Один тип объекта = одна 3D-модель** — все 4 стула в сцене используют одну и ту же GLB-модель (разные позиции, один файл).

7. **Z рассчитывается только по первому родительскому экземпляру** — при нескольких родителях (`parent_instances > 1`) Z берётся от первого в списке.

8. **Топологическая сортировка работает только внутри одной группы** (узлы с одним `parent_id`) — зависимости между разными группами на одном уровне не учитываются.

9. **`_place_anchors_by_llm` вызывается только при наличии нескольких anchor-узлов** — если якорь один, его позиция определяется детерминистически из `anchor_hint.region` без LLM.

---

## Пример полного прохода (из output-файлов)

**Запрос:** "Только стол и 4 стула. и 4 тарелки на столе напротив каждого стула и ваза"

**Stage 0:** `room_type=other`, `room_half_size=2.5`, objects: table×1, chair×4, plate×4, vase×1

**Stage 1:** подобраны модели из каталога (uuid для каждого типа)

**Stage 3:** `room_half = 3.5м` (комната 7×7м)

**Stage 4.1 граф:**
```
Level 0: table_1 (instances=1, anchor)
Level 1: chair_group_1 (instances=4, around table_1, floor)
          plate_group_1 (instances=4, on_surface table_1, ref=chair_group_1)
          vase_1 (instances=1, on_surface table_1)
```

**Stage 4.3 размещение:**
```
table_1:       pos=(0.00, 0.00, 0.375),  yaw=0°
chair N:       pos=(0.00,  1.12, 0.45),  yaw=0°   (north, facing inward → yaw=0)
chair E:       pos=(1.12,  0.00, 0.45),  yaw=90°  (east  → yaw=90)
chair S:       pos=(0.00, -1.12, 0.45),  yaw=180° (south → yaw=180)
chair W:       pos=(-1.12, 0.00, 0.45),  yaw=270° (west  → yaw=270)
plate @ N:     pos=(0.00,  0.25, 0.79),  on_surface table, near north edge
plate @ E:     pos=(0.25,  0.00, 0.79),  on_surface table, near east edge
plate @ S:     pos=(0.00, -0.25, 0.79),  on_surface table, near south edge
plate @ W:     pos=(-0.25, 0.00, 0.79),  on_surface table, near west edge
vase:          pos=(0.00,  0.00, 0.79),  center of table
```
