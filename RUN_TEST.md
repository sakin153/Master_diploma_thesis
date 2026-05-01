# Инструкция по тестированию

## Исправления выполнены

Исправлены три проблемы:

1. **В `runner.py`**: Добавлена проверка что `id` не пустой
2. **В `universal_system.py`**: Добавлено автоматическое создание полей `id` и `type` для объектов
3. **В `universal_system.py`**: Исправлены дублирующиеся ID - теперь каждый объект получает уникальный ID с индексом

## Как протестировать

### Вариант 1: Полный тест

```bash
source .venv/bin/activate
python3 main.py "стол и стул"
```

### Вариант 2: Быстрый тест

```bash
source .venv/bin/activate
python3 test_quick.py
```

### После успешной генерации

```bash
# Посмотреть сцену в MuJoCo viewer
python3 run_mujoco_viewer.py
```

## Что было исправлено

### Проблема 1
```
RuntimeError: Object at index 0 missing required field 'id'
```

### Проблема 2
```
RuntimeError: Duplicate object ID: 0632cdfa6b0b596b93c124dcae3d3a9b
```

### Решение

**1. В `creator/runner.py` (строки ~449-458)**:
```python
# Ensure we have a valid id
model_id = model.get("uuid") or model.get("Model") or f"object_{i}"
if not model_id or not str(model_id).strip():
    model_id = f"object_{i}"

models_for_universal.append({
    "id": str(model_id).strip(),  # Гарантируем непустой id
    "Model": model.get("Model", ""),
    "size": model.get("size", [1.0, 1.0, 1.0]),
    "type": "furniture",
    "uuid": model.get("uuid", ""),
})
```

**2. В `creator/placement/universal_system.py` (метод `_generate_semantic_plan`)**:
```python
# Create unique IDs with index to avoid duplicates
for i, obj in enumerate(semantic_plan["objects"]):
    model_name = obj.get("Model", f"object_{i}")
    clean_name = model_name.replace(" ", "_").replace("-", "_")
    obj["id"] = f"{clean_name}_{i}"  # Уникальный ID с индексом
    
    if "type" not in obj:
        obj["type"] = "furniture" if obj.get("is_static", True) else "small_object"

# Update all target references in constraints
for obj in semantic_plan["objects"]:
    if "constraints" in obj:
        for constraint in obj["constraints"]:
            if "target" in constraint:
                # Map old target to new ID
                constraint["target"] = id_mapping.get(constraint["target"])
```

## Ожидаемый результат

```
[pipeline] Stage 4: Layout solving with Universal Placement System
[UniversalPlacementSystem] Stage 1: Command Interpretation
[UniversalPlacementSystem] Stage 2: Semantic Planning
[UniversalPlacementSystem] Stage 3: Scene Graph Construction
[UniversalPlacementSystem] Stage 4: Anchor System Setup
[UniversalPlacementSystem] Stage 5: Constraint Resolution
[UniversalPlacementSystem] Stage 6: Layout Solving
[UniversalPlacementSystem] Stage 7: Physics Validation
[UniversalPlacementSystem] Stage 8: Export
[pipeline] Universal System placed 2 objects
...
[pipeline] Done. Scene saved to: /tmp/ciare_fresh_*/worlds/scene_latest.xml
Generated world at: /tmp/ciare_fresh_*/worlds/scene_latest.xml
```

## Примеры ID

Теперь объекты получают уникальные ID:
- `desk_0` - первый стол
- `computer_chair_1` - первый стул
- `computer_chair_2` - второй стул (если есть)
- `notebook_3` - блокнот

Это гарантирует что не будет дубликатов, даже если используется несколько объектов одной модели.
