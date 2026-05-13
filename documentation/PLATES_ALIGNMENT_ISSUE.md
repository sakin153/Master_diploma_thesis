# Проблема с привязкой тарелок к стульям

## Ваши вопросы:

### 1. Почему получилось 2 стола вместо 1?

**Ответ:** Prompt expander интерпретировал "каждого стола" как множественное число.

**Ваш запрос:** "стол, вокруг каждого стола по 4 стула..."

**Интерпретация LLM:** "two identical rectangular tables..."

**Решение:** Явно указывать "**1 стол**" или "**один стол**":
```
"1 стол, вокруг стола 4 стула, на столе 4 тарелки напротив каждого стула"
```

---

### 2. Почему тарелки НЕ напротив стульев?

**Ответ:** Тарелки имеют `parent_id="tables"` вместо привязки к стульям.

**Текущий scene graph:**
```json
{
  "id": "plates",
  "instances": 8,
  "parent_id": "tables",      // ← Родитель: столы
  "relationship": {
    "type": "on_surface",
    "reference": "chairs",    // ← Только reference, но нет 1:1 привязки
    "arrangement": "auto"
  }
}
```

**Проблема:**
- Тарелки размещаются **на столах** (8 тарелок ÷ 2 стола = 4 на каждом)
- `reference: "chairs"` - это только **подсказка**, а не строгая 1:1 привязка
- Система размещает тарелки в сетке на столе, **игнорируя** позиции стульев

**Реальные позиции (стол 1):**
```
Стулья:  (-0.357, -1.2), (-2.043, -1.2), (-1.2, -0.357), (-1.2, -2.043)
Тарелки: (-1.375, -1.375), (-1.025, -1.375), (-1.375, -1.025), (-1.025, -1.025)
```

Видите? Тарелки в сетке, стулья по кругу - **нет соответствия**!

---

## Почему это происходит?

### Код уже поддерживает 1:1 привязку!

В `scene_planner.py` есть механизм:

```python
# Обнаружение 1:1 привязки
ref_id_for_batch = rel.get("reference")
has_1to1_ref = False
if is_surface and ref_id_for_batch:
    ref_list_for_batch = _ws_find_all_by_node(world_state, ref_id_for_batch)
    if ref_list_for_batch and len(ref_list_for_batch) == instances:
        has_1to1_ref = True  // ← Активирует 1:1 размещение
```

**НО!** Проблема в том, что:
1. `plates.parent_id = "tables"` (не "chairs")
2. Система группирует тарелки по столам: 8 тарелок ÷ 2 стола = 4 на каждом
3. Для каждого стола: 4 тарелки, 4 стула вокруг этого стола
4. **Условие срабатывает!** `len(chairs_around_table) == len(plates_on_table)` = 4 == 4
5. **Но** стулья размещены "around" (по кругу), а тарелки "on_surface" (на столе)
6. LLM не понимает, как спроецировать круговое расположение стульев на плоскость стола

---

## Решение

### Вариант 1: Улучшить промпт GROUP_RELATIONS

Добавить более явные инструкции для случая "тарелки напротив стульев":

```python
# В GROUP_RELATIONS_PROMPT добавить:
"""
SPECIAL CASE: plates opposite chairs
====================================
When plates (on_surface) reference chairs (around), the 1:1 alignment means:
  - Each plate should be placed on the table surface
  - At the position that is CLOSEST to its paired chair
  - Project the chair's (x, y) onto the table surface
  - Inset slightly from the table edge toward center
"""
```

### Вариант 2: Добавить post-processing

После размещения тарелок, если обнаружена 1:1 привязка к стульям, скорректировать позиции:

```python
def _adjust_plates_to_chairs(plates, chairs, table):
    """Adjust plate positions to align with chairs."""
    for i, (plate, chair) in enumerate(zip(plates, chairs)):
        # Project chair position onto table surface
        table_center = (table['x'], table['y'])
        chair_pos = (chair['x'], chair['y'])
        
        # Find point on table closest to chair
        direction = normalize(chair_pos - table_center)
        plate_pos = table_center + direction * (table_radius - inset)
        
        plate['Pose']['x'] = plate_pos[0]
        plate['Pose']['y'] = plate_pos[1]
```

### Вариант 3: Изменить иерархию (самый правильный)

Сделать тарелки **дочерними объектами стульев**:

```json
{
  "id": "plates",
  "instances": 8,
  "parent_id": "chairs",     // ← Изменить на chairs!
  "relationship": {
    "type": "in_front",      // ← Тип: перед стулом
    "reference": "chairs",
    "distance": 0.3
  }
}
```

Но это требует изменения логики в CLASSIFY_HIERARCHY.

---

## Текущий статус

✅ **Код поддерживает 1:1 привязку** - механизм есть
✅ **Промпт поддерживает reference** - инструкции есть
❌ **LLM не всегда правильно интерпретирует** - нужно улучшить промпт
❌ **Проекция "around" → "on_surface" не работает** - нужна доработка

---

## Рекомендация

**Краткосрочно:** Улучшить промпт GROUP_RELATIONS с явными инструкциями для случая "plates opposite chairs"

**Долгосрочно:** Добавить специальную логику для проекции круговых расположений на поверхности

**Обратная совместимость:** Все изменения должны быть опциональными и не ломать существующие сцены
