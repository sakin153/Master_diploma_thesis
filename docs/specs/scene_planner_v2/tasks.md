# Scene Planner v2 — Tasks

## Статус задач

- [ ] = не начата
- [~] = в процессе
- [x] = завершена

---

## Milestone 1: Структуры данных и утилиты

### TASK-01: SceneNode и SceneGraph
**Файл:** `project/scene_planner.py`

Определить структуры данных для представления графа:

- [x] Определить константы типов связей: `REL_AROUND`, `REL_ON_SURFACE`, `REL_BESIDE`, `REL_FACING`, `REL_OPPOSITE`, `REL_IN_FRONT`, `REL_BEHIND`, `REL_ABSOLUTE`
- [x] Функция `make_scene_node(id, model_name, level, instances, parent_id, relationship, anchor_hint, place_order, depends_on)` → dict
  - `place_order`: int, порядок внутри группы родителя (1 = первый, 2 = после place_order=1)
  - `depends_on`: str|None, ID узла от которого зависит позиция (тарелки зависят от стульев)
- [x] Функция `make_scene_graph(nodes, anchor_relations)` → dict
- [x] Функция `make_world_state(room_half_size)` → dict с пустыми `placed` и `remaining`
- [x] Функция `_collect_groups(nodes)` → `{parent_id: [node, ...]}` — группирует узлы по родителю

### TASK-02: Валидатор графа `SceneGraphValidator`
**Файл:** `project/scene_planner.py`

- [x] `validate_graph(graph)` → `(bool, list[str] errors)`
- [x] Проверка: все `parent_id` ссылаются на существующие `id`
- [x] Проверка: level 0 имеют `parent_id = null`
- [x] Проверка: level N+1 имеет parent на level N (нет пропусков)
- [x] Проверка: нет циклов (DFS)
- [x] Проверка: все `model_name` из переданного списка доступных моделей
- [x] Проверка: `instances >= 1`

### TASK-03: WorldState утилиты
**Файл:** `project/scene_planner.py`

- [x] `world_state_add(world_state, node, placed_results)` → обновляет `placed` и `remaining`
- [x] `world_state_summary(world_state, max_items=10)` → строка для промпта (краткое описание)
- [x] `remaining_summary(world_state)` → строка "в сцене ещё будет: 4x Chair, 2x Plate"
- [x] `find_placed(world_state, object_id)` → dict или None

### TASK-04: Эвристический граф `_heuristic_graph`
**Файл:** `project/scene_planner.py`

Fallback когда LLM не справилась с построением графа:

- [x] Определить словари ключевых слов:
  - Anchors: `("table","sofa","bed","desk","shelf","cabinet","bookcase")`
  - Level-1 dependents: `("chair","stool","lamp","plate","bowl","cup","vase")`
  - Level-2 dependents: `("apple","book","pen","bottle","glass","candle","fruit")`
- [x] Группировать одинаковые объекты → `group_type="repeated"`
- [x] Привязывать dependents к ближайшему anchor по типу
- [x] Возвращать валидный SceneGraph

---

## Milestone 2: Построение графа (SceneGraphBuilder)

### TASK-05: Промпт иерархии `GRAPH_HIERARCHY_PROMPT`
**Файл:** `project/scene_planner.py`

- [x] Написать промпт-шаблон для LLM-вызова 1
- [x] Промпт объясняет правила уровней (0/1/2)
- [x] Промпт объясняет group_type (single vs repeated)
- [x] Промпт объясняет типы связей с примерами
- [x] Промпт требует точно N узлов (по числу переданных объектов)
- [x] Добавить раздел с примерами хороших JSON-ответов

### TASK-06: Промпт связей anchors `ANCHOR_RELATIONS_PROMPT`
**Файл:** `project/scene_planner.py`

- [x] Написать промпт-шаблон для LLM-вызова 2 (только если > 1 anchor)
- [x] Промпт включает требования к пространству вокруг каждого anchor
  (`space_requirements` = "вокруг стола нужно 0.6м для стульев с каждой стороны")
- [x] Промпт возвращает `anchor_relations` + `anchor_positions` (абсолютные координаты)

### TASK-07: `SceneGraphBuilder.build_hierarchy`
**Файл:** `project/scene_planner.py`

- [x] Форматировать список моделей для промпта
- [x] Вызвать LLM с `GRAPH_HIERARCHY_PROMPT`
- [x] Валидировать через `SceneGraphValidator`
- [x] При ошибке — retry до 3 раз с feedback
- [x] При 3 провалах — `_heuristic_graph()` fallback
- [x] Логировать: `[graph_builder] Hierarchy: 2 anchors, 3 groups, max_level=2`

### TASK-08: `SceneGraphBuilder.build_anchor_rels`
**Файл:** `project/scene_planner.py`

- [x] Пропустить если anchor_count == 1
- [x] Вычислить `space_requirements` для каждого anchor (из его дочерних узлов)
- [x] Вызвать LLM с `ANCHOR_RELATIONS_PROMPT`
- [x] При ошибке — использовать равномерное распределение anchors по комнате
- [x] Логировать: `[graph_builder] Anchor relations: sofa_1 ↔ table_1 (opposite, 2.0m)`

### TASK-09: `SceneGraphBuilder.build`
**Файл:** `project/scene_planner.py`

- [x] Вызвать `build_hierarchy()` → получить nodes
- [x] Вызвать `build_anchor_rels()` → получить relations + anchor_positions
- [x] Собрать финальный SceneGraph
- [x] Вычислить `metadata` (total_objects, max_level)
- [x] Логировать граф в читаемом виде:
  ```
  [graph_builder] Scene graph:
    Level 0: table_1(Dining Table×1), sofa_1(Sofa×1)
    Level 1: chair_group(Chair×4 → table_1), plate_group(Plate×4 → table_1)
    Level 2: apple_group(Apple×4 → plate_group)
  ```

---

## Milestone 3: Иерархическое размещение (HierarchicalPlacer)

### TASK-10: Промпт одиночного размещения `SINGLE_PLACEMENT_PROMPT`
**Файл:** `project/scene_planner.py`

- [x] Описание объекта (имя, размер)
- [x] Описание связи с родителем (или "anchor, разместить в [region]")
- [x] `remaining_objects` секция ("в сцене ещё будет...")
- [x] `world_state_summary` секция
- [x] Чёткий формат ответа: `{"x": float, "y": float, "z": float, "yaw_deg": float}`
- [x] Примеры правильного и неправильного ответа

### TASK-11: Промпт группового размещения `GROUP_PLACEMENT_PROMPT`
**Файл:** `project/scene_planner.py`

- [x] Описание группы (тип, количество, суммарный батч)
- [x] Описание родителя (позиция, размер, тип связи)
- [x] Секция "уже размещено в группе"
- [x] `world_state_summary`
- [x] Формат ответа: `[{"instance": N, "x": float, "y": float, "z": float, "yaw_deg": float}, ...]`

### TASK-12: `HierarchicalPlacer._place_single`
**Файл:** `project/scene_planner.py`

- [x] Найти parent в world_state (если level > 0)
- [x] Сформировать `relationship_description` из node["relationship"]
- [x] Вызвать LLM с `SINGLE_PLACEMENT_PROMPT`
- [x] Валидировать ответ: координаты в границах комнаты
- [x] При ошибке — retry + fallback позиция
- [x] Логировать: `[placer] Placed table_1 at (0.0, 0.0, 0.375), yaw=0°`

### TASK-13: `HierarchicalPlacer._place_group_of_parent`
**Файл:** `project/scene_planner.py`

Размещает все узлы, привязанные к одному родителю (гетерогенная группа):

- [x] Отсортировать узлы группы по `place_order`, затем по `depends_on` (зависимые — позже)
- [x] Итерировать по узлам в порядке сортировки:
  - `instances == 1` → вызвать `_place_single(node, ...)`
  - `instances > 1` → вызвать `_place_node_batch(node, ...)`
- [x] После каждого узла обновить world_state — следующий узел видит уже размещённых
- [x] Логировать: `[placer] Group(table_1): placed chair×4 → plate×4 → apple×4`

### TASK-13b: `HierarchicalPlacer._place_node_batch`
**Файл:** `project/scene_planner.py`

Размещает несколько экземпляров одного типа объектов (батчинг внутри узла):

- [x] Разбить `instances` на суб-батчи по ≤ 6
- [x] Для каждого суб-батча:
  - [x] Сформировать `already_placed_in_batch` из предыдущих суб-батчей этого узла
  - [x] Если `depends_on` задан — включить в контекст уже размещённые объекты зависимости
  - [x] Вызвать LLM с `GROUP_PLACEMENT_PROMPT`
  - [x] Валидировать ответ (правильное число объектов, координаты в комнате)
  - [x] При ошибке — retry с feedback
  - [x] Обновить world_state через `world_state_add()`
- [x] Логировать: `[placer] Group chair_group: placed 4/4 (1 batch)`

### TASK-14: `HierarchicalPlacer.place_all`
**Файл:** `project/scene_planner.py`

- [x] Инициализировать world_state из anchor_positions (результат Milestone 2)
- [x] Добавить anchors с их позициями в world_state
- [x] Итерировать по уровням: 0 → 1 → 2 → ...
- [x] На каждом уровне: итерировать по узлам
  - `group_type == "single"` → `_place_single()`
  - `group_type == "repeated"` → `_place_group()`
- [x] После каждого размещения — `world_state_add()`
- [x] Возвращать `world_state["placed"]`

### TASK-15: Валидация координат
**Файл:** `project/scene_planner.py`

- [x] `_validate_placement_result(result, room_half_size, node_size)` → `(bool, str)`
- [x] Проверка: x, y в диапазоне [-room_half_size, room_half_size]
- [x] Проверка: z >= 0
- [x] Проверка: yaw_deg в [0, 360)
- [x] Проверка: результат содержит все обязательные поля

### TASK-16: Fallback позиции
**Файл:** `project/scene_planner.py`

- [x] `_fallback_position(node, parent, world_state, room_half_size)` → dict
- [x] level 0 без parent: центр комнаты + случайный offset ≤ 0.5м
- [x] level 1 с parent: рядом с родителем на safe_distance
- [x] level 2 с parent на поверхности: центр верхней поверхности родителя
- [x] Проверить что fallback не перекрывает уже размещённые объекты (AABB check)

---

## Milestone 4: Сборка и интеграция

### TASK-17: Merge world_state → output list
**Файл:** `project/scene_planner.py`

- [x] `_merge_to_output(placed_list, models, graph)` → list of dicts
- [x] Для каждого placed объекта найти оригинальную модель из `models` по `model_name`
- [x] Подставить `model_loc`, `uuid`, `scale`, `_up_axis` из оригинала
- [x] Сгенерировать уникальный `save_fn`: `{sanitized_model_name}_{instance_idx}`
- [x] Конвертировать `size` из dict → list `[width, height, depth]`
- [x] Конвертировать `position` → `Pose` dict с ключами `x, y, z`
- [x] Логировать каждый объект: `[scene_planner] Dining Chair: pos=(0.75, 0.0, 0.45), yaw=180°`

### TASK-18: Обновить `generate_placement_plan`
**Файл:** `project/scene_planner.py`

- [x] Заменить вызов `_generate_plan_in_batches` на:
  1. `SceneGraphBuilder.build()` → граф
  2. `HierarchicalPlacer.place_all()` → placed list
  3. `_merge_to_output()` → output list
- [x] Сохранить вызов `validate_and_repair_layout()` в конце
- [x] Сохранить ту же сигнатуру функции
- [x] Логировать: `[scene_planner] Stage 4 complete: N models with placement`

### TASK-19: Удалить устаревший код
**Файл:** `project/scene_planner.py`

После проверки что новая версия работает:

- [x] Удалить `_generate_plan()`
- [x] Удалить `_generate_plan_in_batches()`
- [x] Удалить `_determine_priorities()`
- [x] Удалить `_group_by_anchor()`
- [x] Удалить `_PLACEMENT_PRIORITY_PROMPT`
- [x] Удалить `_SEMANTIC_PLAN_PROMPT`
- [x] Удалить `_DistanceResolver` (если не нужен)
- [x] Удалить `_OrientationResolver` (если не нужен)
- [x] Удалить `_PlacementExecutor`
- [x] Удалить `_SchemaValidator` (заменить на `SceneGraphValidator`)

---

## Milestone 5: Тестирование

### TASK-20: Базовые сценарии
- [ ] `"стол и 4 стула"` → 5 объектов, 1 anchor, 1 группа
- [ ] `"диван напротив стола"` → 2 anchor с explicit связью
- [ ] `"стол, 4 стула, 4 тарелки на столе"` → 2 уровня иерархии
- [ ] `"стол, 4 стула, тарелки, яблоки на тарелках"` → 3 уровня

### TASK-21: Граничные случаи
- [ ] `"1 объект"` → 1 anchor, никаких групп
- [ ] `"10 стульев"` → 1 группа, 2 суб-батча (6 + 4)
- [ ] `"4 стола, 6 стульев у каждого"` → 4 anchors, 4 группы по 6
- [ ] LLM возвращает невалидный JSON → fallback к `_heuristic_graph`
- [ ] LLM ставит объект за стену → clamp + retry

### TASK-22: Регрессия
- [ ] Убедиться что `validate_and_repair_layout` по-прежнему вызывается
- [ ] Убедиться что выходной формат совместим со Stage 5 (mujoco_assembler)
- [ ] Убедиться что MuJoCo компилирует итоговый XML без ошибок

---

## Порядок выполнения

```
Milestone 1 (структуры)
    ↓
Milestone 2 (SceneGraphBuilder)
    ↓
Milestone 3 (HierarchicalPlacer)
    ↓
Milestone 4 (интеграция)
    ↓
Milestone 5 (тесты)
```

Milestone 1 и начало Milestone 2 можно делать параллельно с Milestone 3 (промпты).

---

## Оценка объёма

| Milestone | Задач | Оценка строк кода |
|---|---|---|
| M1: Структуры | 4 | ~150 |
| M2: GraphBuilder | 5 | ~250 |
| M3: Placer | 7 | ~350 |
| M4: Интеграция | 3 | ~100 |
| M5: Тесты | 3 | ~100 |
| **Итого** | **22** | **~950** |

Текущий `scene_planner.py`: ~1570 строк.  
Новый: ~1800–2000 строк (один файл, как сейчас).
