# Scene Planner v2 — Design

## 1. Архитектурный обзор

```
generate_placement_plan(models, room_half_size, room_type, query)
│
├── ФАЗА 1: Построение графа сцены
│   ├── SceneGraphBuilder.build_hierarchy()    ← LLM вызов 1
│   ├── SceneGraphBuilder.build_groups()       ← LLM вызов 2
│   └── SceneGraphBuilder.build_anchor_rels()  ← LLM вызов 3 (если > 1 anchor)
│
├── ФАЗА 2: Иерархическое размещение
│   ├── HierarchicalPlacer.place_level(0)      ← Anchors (с контекстом будущих объектов)
│   ├── HierarchicalPlacer.place_level(1)      ← Groups relative to anchors
│   └── HierarchicalPlacer.place_level(2+)     ← Sub-groups
│
├── Merge: plan → placed models list
│
└── validate_and_repair_layout()               ← без изменений
```

---

## 2. Структуры данных

### 2.1 SceneNode — узел графа

SceneNode описывает **один тип объекта** (возможно, несколько экземпляров одного типа)
с его привязкой к родителю. Узлы с одним родителем образуют **группу родителя** —
она может быть гетерогенной (разные типы, один parent_id).

```python
# Узел графа — один тип объекта и его привязка к родителю
{
    "id":          "chair_group_1",     # уникальный ID узла
    "model_name":  "Dining Chair",      # имя из каталога
    "level":       1,                   # 0=anchor, 1=level1, 2=level2
    "instances":   4,                   # количество экземпляров этого типа
    
    # привязка к родителю (None для level 0)
    "parent_id":   "table_1",
    "relationship": {
        "type":      "around",          # см. типы связей
        "direction": "all_sides",       # конкретизация
        "distance":  0.6,               # метры, None = вывести из размеров
        "facing":    "inward",          # куда смотреть
    },
    
    # порядок внутри группы родителя (важно если зависит от других узлов)
    # тарелки ставятся после стульев (place_order=2), стулья первыми (place_order=1)
    "place_order": 1,
    
    # ссылка на другой узел той же группы, если позиция зависит от него
    # (тарелки "напротив стульев" → depends_on="chair_group_1")
    "depends_on":  None,

    # позиция для anchor (только level 0)
    "anchor_hint": {
        "region":   "center",           # "center" | "near_wall" | "corner"
        "offset_x": 0.0,
        "offset_y": 0.0,
    },
}
```

**Группа родителя** — это все узлы с одинаковым `parent_id`. Внутри группы:
- Разные типы объектов могут иметь разный `place_order`
- Если узел B зависит от узла A через `depends_on`, B ставится после A
- Одинаковые `model_name` батчуются в один LLM-запрос (≤ 6 экземпляров)

### 2.2 Типы связей (relationship.type)

| Тип | Описание | Пример |
|---|---|---|
| `absolute` | Точные координаты | anchor в центре |
| `around` | Вокруг родителя равномерно | стулья вокруг стола |
| `on_surface` | На поверхности родителя | тарелка на столе |
| `beside` | Рядом с боку | кресло рядом с диваном |
| `facing` | Напротив, лицом к родителю | кресла напротив дивана |
| `opposite` | Напротив другого anchor | стол напротив дивана |
| `in_front` | Перед родителем | кофейный столик перед диваном |
| `behind` | За родителем | тумбочка за диваном |

### 2.3 SceneGraph — граф целиком

```python
{
    "nodes": [SceneNode, ...],       # список всех узлов
    "anchor_relations": [            # связи между anchors
        {
            "from": "sofa_1",
            "to":   "table_1",
            "type": "opposite",
            "distance": 2.0,
        }
    ],
    "metadata": {
        "total_objects": 7,
        "max_level": 2,
        "room_half_size": 4.5,
    }
}
```

### 2.4 WorldState — состояние мира во время размещения

```python
{
    "placed": [
        {
            "id":    "table_1",
            "Model": "Dining Table",
            "size":  {"width": 1.5, "length": 0.9, "height": 0.75},
            "Pose":  {"x": 0.0, "y": 0.0, "z": 0.375},
            "yaw_deg": 0.0,
        },
        ...
    ],
    "remaining": [
        {"model_name": "Dining Chair", "count": 4, "size": {"width": 0.55, "length": 0.55, "height": 0.90}},
        ...
    ]
}
```

---

## 3. Фаза 1 — Построение графа: `SceneGraphBuilder`

### 3.1 Шаг 1: Определение иерархии (`build_hierarchy`)

**LLM-вызов 1**

Промпт передаёт:
- Запрос пользователя
- Список всех объектов с размерами
- Размер комнаты

LLM возвращает:
```json
{
  "nodes": [
    {
      "id": "table_1",
      "model_name": "Dining Table",
      "level": 0,
      "instances": 1,
      "parent_id": null,
      "anchor_hint": {"region": "center"},
      "place_order": 1,
      "depends_on": null
    },
    {
      "id": "chair_group_1",
      "model_name": "Dining Chair",
      "level": 1,
      "instances": 4,
      "parent_id": "table_1",
      "relationship": {
        "type": "around",
        "distance": 0.6,
        "facing": "inward"
      },
      "place_order": 1,
      "depends_on": null
    },
    {
      "id": "plate_group_1",
      "model_name": "Plate",
      "level": 1,
      "instances": 4,
      "parent_id": "table_1",
      "relationship": {
        "type": "on_surface",
        "facing": "opposite_chairs"
      },
      "place_order": 2,
      "depends_on": "chair_group_1"
    }
  ]
}
```

Группа стола содержит два типа объектов (стулья и тарелки) — оба `parent_id = "table_1"`.
Тарелки имеют `place_order = 2` и `depends_on = "chair_group_1"` → ставятся после стульев.

**Валидация:**
- Все parent_id ссылаются на существующие id
- Нет циклов (DFS)
- level 0 объекты имеют parent_id = null
- Все model_name из доступных моделей

**Fallback при ошибке LLM:**
```python
def _heuristic_graph(models):
    # Правило 1: объекты с "table","sofa","bed","desk" в имени → level 0
    # Правило 2: объекты с "chair","lamp","plate","cup" → level 1
    # Правило 3: объекты с "apple","book","pen" → level 2 (на поверхности level 1)
    # Правило 4: одинаковые объекты → group_type="repeated"
```

### 3.2 Шаг 2: Связи между anchors (`build_anchor_rels`)

**LLM-вызов 2** — только если anchor-объектов > 1

Промпт: "У нас N anchors: [список с размерами]. Опиши их взаимное расположение в комнате размером WxL."

LLM возвращает:
```json
{
  "anchor_relations": [
    {
      "from": "sofa_1",
      "to": "table_1",
      "type": "opposite",
      "distance": 2.0,
      "description": "Стол напротив дивана на расстоянии 2м"
    }
  ]
}
```

**Если 1 anchor** — этот шаг пропускается.

---

## 4. Фаза 2 — Иерархическое размещение: `HierarchicalPlacer`

### 4.1 Главный цикл

```python
def place_all(graph, world_state, room_half_size, query):
    max_level = graph["metadata"]["max_level"]
    
    for level in range(max_level + 1):
        nodes_at_level = [n for n in graph["nodes"] if n["level"] == level]

        if level == 0:
            # Anchors: каждый как одиночный объект
            for node in nodes_at_level:
                results = _place_single(node, graph, world_state, room_half_size, query)
                _update_world_state(world_state, node, results)
        else:
            # Уровень 1+: сгруппировать по parent_id, обработать группы
            groups = _collect_groups(nodes_at_level)  # {parent_id: [node, ...]}
            for parent_id, group_nodes in groups.items():
                _place_group_of_parent(
                    group_nodes, parent_id,
                    graph, world_state, room_half_size, query
                )

def _collect_groups(nodes):
    """Собрать узлы одного уровня в группы по parent_id."""
    groups = {}
    for node in nodes:
        pid = node["parent_id"]
        groups.setdefault(pid, []).append(node)
    return groups

def _place_group_of_parent(group_nodes, parent_id, graph, world_state, room_half_size, query):
    """Разместить все узлы группы одного родителя в порядке place_order."""
    # Сортируем по place_order, затем по depends_on (зависимые — позже)
    ordered = sorted(group_nodes, key=lambda n: (n.get("place_order", 1), n["id"]))
    
    for node in ordered:
        # Если зависит от другого узла — он уже в world_state (мы идём по порядку)
        if node["instances"] == 1:
            results = _place_single(node, graph, world_state, room_half_size, query)
        else:
            results = _place_node_batch(node, graph, world_state, room_half_size, query)
        
        # Обновить world_state СРАЗУ — следующий узел в группе уже видит этот
        _update_world_state(world_state, node, results)
```

**Ключевое свойство**: внутри группы одного родителя узлы обрабатываются
последовательно по `place_order`. После каждого узла world_state обновляется —
следующий узел (напр. тарелки) уже видит позиции предыдущих (стульев).

### 4.2 Размещение одиночного объекта `_place_single`

```python
def _place_single(node, graph, world_state, room_half_size, query):
    parent = _find_placed(world_state, node["parent_id"])
    
    prompt = _build_single_placement_prompt(
        node=node,
        parent=parent,
        relationship=node["relationship"],
        world_state=world_state,       # уже размещённые объекты
        room_half_size=room_half_size,
        query=query,
    )
    
    # LLM возвращает {"x": ..., "y": ..., "z": ..., "yaw_deg": ...}
    result = llm_fn(prompt)
    return result
```

**Структура промпта для одиночного объекта:**
```
Разместить: Dining Table (1.5 × 0.9 × 0.75м)
Связь: anchor объект, region=center

Комната: 9.0 × 9.0м

В сцене ТАКЖЕ БУДУТ следующие объекты (учти при размещении):
  - 4x Dining Chair (0.55 × 0.55м) — будут вокруг стола
  - 4x Plate (0.28 × 0.28м) — будут на столе

Уже размещённые объекты: []

Верни JSON: {"x": ..., "y": ..., "z": ..., "yaw_deg": ...}
```

### 4.3 Размещение группы `_place_group`

```python
def _place_group(node, graph, world_state, room_half_size, query):
    instances = node["instances"]
    parent = _find_placed(world_state, node["parent_id"])
    placed_in_group = []
    
    # Разбить на суб-батчи по 6
    batch_size = 6
    batches = [
        list(range(i, min(i + batch_size, instances)))
        for i in range(0, instances, batch_size)
    ]
    
    for batch_indices in batches:
        prompt = _build_group_placement_prompt(
            node=node,
            batch_indices=batch_indices,     # "размести стулья №3, 4, 5"
            parent=parent,
            already_in_group=placed_in_group,  # стулья 1, 2 уже стоят тут
            world_state=world_state,
            room_half_size=room_half_size,
            query=query,
        )
        
        results = llm_fn(prompt)  # список позиций
        placed_in_group.extend(results)
        _update_world_state(world_state, node, results)  # обновляем ПОСЛЕ каждого батча
    
    return placed_in_group
```

**Структура промпта для группы (батч 2/2):**
```
Разместить: Dining Chair (0.55 × 0.55м) — объекты 3, 4 из 4
Связь: around Dining Table (стол в 0.0, 0.0, высота 0.75м), расстояние 0.6м, facing=inward

Уже в этой группе (стулья 1 и 2):
  - chair_1: pos=(0.0, 1.1, 0.45), yaw=180°
  - chair_2: pos=(0.0, -1.1, 0.45), yaw=0°

Другие объекты в сцене: [sofa_1 at (-2.0, 0.0)]

Верни JSON: [{"instance": 3, "x": ..., "y": ..., "yaw_deg": ...}, ...]
```

### 4.4 Обновление WorldState

```python
def _update_world_state(world_state, node, placed_results):
    for result in placed_results:
        world_state["placed"].append({
            "id":    result["id"],
            "Model": node["model_name"],
            "size":  node["size"],
            "Pose":  {"x": result["x"], "y": result["y"], "z": result["z"]},
            "yaw_deg": result["yaw_deg"],
        })
    
    # Убрать из remaining
    world_state["remaining"] = [
        r for r in world_state["remaining"]
        if r["model_name"] != node["model_name"]
        or r["count"] > len(placed_results)
    ]
```

---

## 5. Схема LLM-вызовов

```
Сцена: стол, 4 стула, диван, 4 тарелки
(10 объектов)

LLM Call 1: build_hierarchy
  → граф с 4 узлами

LLM Call 2: build_anchor_rels (2 anchors: стол + диван)
  → соотношение диван↔стол

LLM Call 3: place_single(диван, level=0)
  → {"x": -2.0, "y": 0.0, "yaw_deg": 90}

LLM Call 4: place_single(стол, level=0)
  → {"x": 1.5, "y": 0.0, "yaw_deg": 0}

LLM Call 5: place_group(стулья 1–4, level=1, parent=стол)
  → 4 позиции за один вызов (≤6)

LLM Call 6: place_group(тарелки 1–4, level=1, parent=стол)
  → 4 позиции за один вызов

Итого: 6 LLM-вызовов для 10 объектов
```

Для сравнения — текущая реализация: 1–2 вызова, но качество хуже.

---

## 6. Обработка ошибок

### 6.1 Ошибка в графе (LLM вернула невалидную структуру)

```
Попытка 1 → валидация → ошибки → промпт + ошибки
Попытка 2 → валидация → ошибки → промпт + ошибки
Попытка 3 → если снова ошибки → _heuristic_graph() fallback
```

### 6.2 Ошибка в размещении отдельного объекта

```
Попытка 1 → размещение → провал
Попытка 2 → промпт + "предыдущая попытка дала x=999 (вне комнаты)"
Попытка 3 → если снова провал → fallback позиция:
    - для level 0: центр комнаты + small_random_offset
    - для level 1: рядом с родителем на safe_distance
    - для level 2: центр поверхности родителя
```

### 6.3 Цикл в графе

DFS при построении графа → если цикл обнаружен → разорвать самое слабое ребро
(объект с наименьшим `level` становится anchor).

---

## 7. Взаимодействие с существующим кодом

### 7.1 Что остаётся без изменений

| Компонент | Статус |
|---|---|
| `Vec2`, `OBB`, `AABB`, `obb_overlap*` | Без изменений |
| `gradient_resolve_overlaps()` | Без изменений |
| `check_collisions()` | Без изменений |
| `validate_and_repair_layout()` | Без изменений |
| `_llm_replan_layout()` | Без изменений |

### 7.2 Что заменяется

| Компонент | Статус |
|---|---|
| `_generate_plan()` | Заменяется `SceneGraphBuilder` + `HierarchicalPlacer` |
| `_generate_plan_in_batches()` | Заменяется `HierarchicalPlacer.place_all()` |
| `_determine_priorities()` | Заменяется `SceneGraphBuilder.build_hierarchy()` |
| `_group_by_anchor()` | Заменяется логикой `SceneGraph.nodes` |
| `_SchemaValidator` | Расширяется для валидации графа |
| `_DistanceResolver` | Сохраняется для совместимости, может использоваться для on_surface/above |
| `_OrientationResolver` | Сохраняется, может использоваться |
| `_PlacementExecutor` | Заменяется `_merge_to_output()` |

### 7.3 Сигнатура выходных данных (без изменений)

```python
# На выходе generate_placement_plan() — тот же формат что и сейчас:
[
    {
        "Model": "Dining Table",
        "uuid": "abc123",
        "model_loc": "/path/to/table.glb",
        "save_fn": "abc123_0",
        "_up_axis": "y",
        "size": [1.5, 0.75, 0.9],
        "is_static": True,
        "Pose": {"x": 0.0, "y": 0.0, "z": 0.375},
        "yaw_deg": 0.0,
    },
    ...
]
```

---

## 8. Промпты

### 8.1 `GRAPH_HIERARCHY_PROMPT`

```
Ты — планировщик расстановки мебели. Твоя задача — построить иерархию объектов для сцены.

Запрос пользователя: {query}
Комната: {room_width}м × {room_length}м

Доступные объекты:
{objects_list}

Правила иерархии:
- level 0: основная мебель (стол, диван, кровать, шкаф) — якорные объекты
- level 1: объекты относительно level 0 (стулья вокруг стола, предметы на столе)
- level 2: объекты относительно level 1 (яблоко на тарелке, книга на полке)

Одинаковые объекты с одним родителем → group_type="repeated", instances=N

Верни JSON:
{
  "nodes": [
    {
      "id": "table_1",
      "model_name": "...",
      "level": 0,
      "instances": 1,
      "group_type": "single",
      "parent_id": null,
      "anchor_hint": {"region": "center"},
      "relationship": null
    },
    ...
  ]
}
```

### 8.2 `ANCHOR_RELATIONS_PROMPT`

```
У нас несколько основных объектов (level 0) в комнате {room_width}×{room_length}м:

{anchors_list}

Запрос пользователя: {query}

Опиши как расположить эти объекты относительно друг друга.
Учитывай что вокруг каждого объекта нужно оставить место для:
{space_requirements}

Верни JSON:
{
  "anchor_relations": [
    {
      "from": "sofa_1",
      "to": "table_1",
      "type": "opposite",
      "distance": 2.0
    }
  ],
  "anchor_positions": [
    {
      "id": "sofa_1",
      "x": -2.0, "y": 0.0, "yaw_deg": 90
    }
  ]
}
```

### 8.3 `SINGLE_PLACEMENT_PROMPT`

```
Разместить один объект в сцене.

Объект: {model_name} (размер: {size})
{relationship_description}

Комната: {room_width}×{room_length}м

В сцене ТАКЖЕ БУДУТ (учти при размещении, оставь место):
{remaining_objects}

Уже размещённые объекты:
{world_state_summary}

Верни JSON: {"x": float, "y": float, "z": float, "yaw_deg": float}
```

### 8.4 `GROUP_PLACEMENT_PROMPT`

```
Разместить {batch_count} объектов типа {model_name} (размер: {size}).
Это объекты {from_idx}–{to_idx} из {total} в группе.

Привязка к: {parent_model} {parent_size} в позиции {parent_pos}
Тип расположения: {relationship_type} (расстояние: {distance}м, направление взгляда: {facing})

Уже размещено в этой группе ({already_count} шт.):
{already_placed_in_group}

Другие объекты в сцене:
{world_state_summary}

Верни JSON: [{"instance": N, "x": float, "y": float, "z": float, "yaw_deg": float}, ...]
```
