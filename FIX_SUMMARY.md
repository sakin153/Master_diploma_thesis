# Исправление проблемы с количеством объектов

## Проблема
Запрос "стол и 3 стула" генерировал только 1 стол и 1 стул вместо 1 стола и 3 стульев.

## Корневая причина
В файле `creator/runner.py` (строки 447-495) использовался словарь `obj_state` для кэширования выбранных моделей по типу объекта. Это приводило к тому, что все объекты одного типа (например, все стулья) использовали одну и ту же модель.

### Проблемный код:
```python
obj_state: Dict[str, Dict[str, Any]] = {}  # Один state на тип объекта

for obj in objects:  # ["table", "chair", "chair", "chair"]
    search_key = ...  # "chair" для всех стульев
    
    state = obj_state.get(search_key)  # Для всех стульев ОДИН state!
    if state is None:
        # Только для ПЕРВОГО стула
        ranked = _rank_models_for_object(search_key, models, limit=10)
        primary = ranked[0]
        state = {"order": [primary], "idx": 0}
        obj_state[search_key] = state  # Сохраняем state
    
    # Для 2-го и 3-го стула state УЖЕ существует
    picked = state["order"][0]  # Всегда ОДНА И ТА ЖЕ модель!
    chosen_models.append({"Model": name, "uuid": uid})
```

## Решение (Вариант 1)
Убрали кэширование `obj_state` и теперь каждый объект получает свою модель.

### Исправленный код:
```python
chosen_models: List[Dict[str, str]] = []
dropped_objects: List[str] = []
used_uuids: set = set()

# Track first occurrence of each object type for LLM disambiguation
first_occurrence: Dict[str, bool] = {}

for obj in objects:  # ["table", "chair", "chair", "chair"]
    obj_raw = str(obj).strip()
    obj_key = obj_raw.lower()
    obj_clean = re.sub(r'\([^)]*\)', '', obj_key).strip()
    search_key = _OBJECT_FALLBACKS.get(obj_clean, obj_clean)

    # ВСЕГДА ранжируем модели для каждого объекта (без кэширования)
    ranked = _rank_models_for_object(search_key, models, limit=10)
    if not ranked:
        print(f"[pipeline] WARN: '{obj}' has no catalog match, dropped")
        dropped_objects.append(obj_key)
        continue

    # Предпочитаем неиспользованные модели для разнообразия
    unused = [r for r in ranked if str(r.get("uuid") or "") not in used_uuids]
    candidate_pool = unused if unused else ranked

    # Выбираем модель для этого объекта
    if len(candidate_pool) == 1:
        primary = candidate_pool[0]
    else:
        # LLM disambiguation только для первого вхождения каждого типа
        # (экономим время и API вызовы)
        is_first = search_key not in first_occurrence
        if is_first:
            first_occurrence[search_key] = True
            # LLM disambiguation...
            primary = ...
        else:
            # Для последующих вхождений просто берем первую неиспользованную модель
            primary = candidate_pool[0]

    name = str(primary.get("name", "")).strip()
    uid = str(primary.get("uuid", "")).strip()
    print(f"[Stage1] '{obj_raw}' → {name} (uuid={uid[:8] if uid else 'none'})")
    
    if uid:
        used_uuids.add(uid)
    if name and uid:
        chosen_models.append({"Model": name, "uuid": uid})
    elif name:
        chosen_models.append({"Model": name})
```

## Результат
✅ Теперь для запроса "стол и 3 стула" генерируется:
- 1 стол
- 3 стула (каждый с уникальным uuid)

### Подтверждение из логов:
```
[pipeline] ✓ Generated semantic plan with 4 objects
[pipeline]   table: model_loc=...
[pipeline]   computer chair: model_loc=...
[pipeline]   computer chair: model_loc=...
[pipeline]   computer chair: model_loc=...
[pipeline] ✓ Placed 4 objects
[pipeline]   table: pos=(0.00, 0.00, 0.00), yaw=0°
[pipeline]   computer chair: pos=(0.00, 0.60, 0.34), yaw=0°
[pipeline]   computer chair: pos=(-0.42, -0.42, 0.36), yaw=17°
[pipeline]   computer chair: pos=(0.42, -0.42, 0.39), yaw=343°
```

## Измененные файлы
- `creator/runner.py` (строки 437-495): Убрано кэширование `obj_state`, добавлена логика для выбора уникальных моделей

## Дополнительные улучшения
1. **Оптимизация LLM вызовов**: LLM disambiguation теперь вызывается только для первого вхождения каждого типа объекта
2. **Разнообразие моделей**: Используется `used_uuids` для предпочтения неиспользованных моделей
3. **Правильное количество**: Каждый объект в списке `objects` теперь получает свою запись в `chosen_models`

## Тестирование
Создан тест `test_quantity_fix.py` который проверяет:
- Генерацию правильного количества объектов
- Наличие всех объектов в финальном XML

## Статус
✅ **ИСПРАВЛЕНО** - Проблема с количеством объектов решена
