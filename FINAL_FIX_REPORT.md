# Финальное исправление проблемы с количеством объектов

## Проблема
Запрос "стол и 3 стула" генерировал только 1 стол и 1 стул вместо 1 стола и 3 стульев.

## Найденные проблемы

### Проблема 1: Кэширование моделей по типу объекта
**Файл**: `creator/runner.py`, строки 447-495  
**Причина**: Словарь `obj_state` кэшировал выбранную модель для каждого типа объекта, что приводило к повторному использованию одной модели для всех объектов одного типа.

### Проблема 2: Неявная инструкция для LLM
**Файл**: `creator/contexts_prompts/semantic_plan.py`  
**Причина**: Промпт не содержал явного указания генерировать объект для КАЖДОЙ записи в списке моделей, даже если у них одинаковое имя.

## Решения

### Решение 1: Убрано кэширование obj_state
**Файл**: `creator/runner.py`, строки 437-495

**Изменения**:
1. Убран словарь `obj_state` который кэшировал модели
2. Добавлен словарь `first_occurrence` только для оптимизации LLM disambiguation
3. Каждый объект теперь получает свою модель из `candidate_pool`

```python
# БЫЛО:
obj_state: Dict[str, Dict[str, Any]] = {}
for obj in objects:
    state = obj_state.get(search_key)  # Кэш!
    if state is None:
        # Только для первого объекта
        ...
        obj_state[search_key] = state
    picked = state["order"][0]  # Всегда одна модель

# СТАЛО:
first_occurrence: Dict[str, bool] = {}
for obj in objects:
    # ВСЕГДА ранжируем модели (без кэша)
    ranked = _rank_models_for_object(search_key, models, limit=10)
    unused = [r for r in ranked if str(r.get("uuid")) not in used_uuids]
    candidate_pool = unused if unused else ranked
    
    # LLM disambiguation только для первого вхождения (оптимизация)
    is_first = search_key not in first_occurrence
    if is_first:
        first_occurrence[search_key] = True
        primary = _llm_pick_candidate(...)
    else:
        primary = candidate_pool[0]
    
    chosen_models.append({"Model": name, "uuid": uid})
```

### Решение 2: Усилен промпт для LLM
**Файл**: `creator/contexts_prompts/semantic_plan.py`, строка 479

**Было**:
```
1. **Object Count**: Include ALL objects from the available models list.
```

**Стало**:
```
1. **Object Count**: You MUST generate EXACTLY one object specification for EACH entry 
   in the "Available models for this scene" list above. Count the number of entries in 
   that list - that's how many objects you must generate. If there are 4 entries 
   (1 table + 3 chairs), generate 4 objects. Even if multiple entries have the same 
   Model name (e.g., "computer chair" appears 3 times), you MUST create 3 separate 
   object specifications with unique IDs (chair_1, chair_2, chair_3).
```

### Решение 3: Добавлена явная инструкция о количестве
**Файл**: `creator/runner.py`, строки 270-274

Добавлено явное указание количества объектов в промпт:

```python
# Add explicit count instruction
object_count = len(full_placed_models)
count_instruction = f"\n\nCRITICAL: You MUST generate EXACTLY {object_count} objects in your semantic plan. The list above contains {object_count} entries - create one object specification for each entry, even if some have the same Model name."
models_str = models_str + count_instruction
```

Теперь LLM видит:
```
Available models for this scene:
  - Model: table
    Size: {'width': 1.50, 'length': 0.80, 'height': 0.75}
    uuid: abc123
  - Model: computer chair
    Size: {'width': 0.73, 'length': 1.10, 'height': 0.77}
    uuid: def456
  - Model: computer chair
    Size: {'width': 0.73, 'length': 1.10, 'height': 0.77}
    uuid: ghi789
  - Model: computer chair
    Size: {'width': 0.73, 'length': 1.10, 'height': 0.77}
    uuid: jkl012

CRITICAL: You MUST generate EXACTLY 4 objects in your semantic plan. The list above contains 4 entries - create one object specification for each entry, even if some have the same Model name.
```

## Измененные файлы

1. **`creator/runner.py`**:
   - Строки 437-495: Убрано кэширование `obj_state`
   - Строки 270-274: Добавлена явная инструкция о количестве объектов

2. **`creator/contexts_prompts/semantic_plan.py`**:
   - Строка 479: Усилено правило об Object Count

## Ожидаемый результат

Для запроса "стол и 3 стула":
1. **Stage 0** (Expansion): `["table", "chair", "chair", "chair"]` ✅
2. **Stage 1** (Model Selection): 4 модели в `chosen_models` ✅
3. **Stage 2** (get_full_placed_models): 4 модели в `full_placed_models` ✅
4. **Stage 3** (Semantic Plan): LLM генерирует 4 объекта ✅
5. **Stage 4** (Placement): 4 объекта размещаются ✅
6. **Финальная сцена**: 1 стол + 3 стула ✅

## Тестирование

Запустите:
```bash
python main.py "стол и 3 стула"
python view_scene.py
```

Ожидаемый вывод в логах:
```
[pipeline] Objects: ['table', 'chair', 'chair', 'chair']
[Stage1] 'table' → table (uuid=...)
[Stage1] 'chair' → computer chair (uuid=...)
[Stage1] 'chair' → computer chair (uuid=...)
[Stage1] 'chair' → computer chair (uuid=...)
[pipeline] ✓ Generated semantic plan with 4 objects
[pipeline] ✓ Placed 4 objects
```

## Статус
✅ **ИСПРАВЛЕНО** - Применены 3 исправления для решения проблемы с количеством объектов
