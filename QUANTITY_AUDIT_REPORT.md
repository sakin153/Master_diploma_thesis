# ОТЧЕТ: Ревизия проблемы с количеством объектов

## Запрос пользователя
"стол и 3 стула" → генерируется 1 стол и 1 стул вместо 1 стола и 3 стульев

## Результаты ревизии

### ✅ STAGE 0: Prompt Expansion (РАБОТАЕТ ПРАВИЛЬНО)
**Файл**: `creator/scene/prompt_expander.py`  
**Результат**: Правильно извлекает количество объектов

Пример для "стол и 3 стула":
```python
estimated_objects = [
    ObjectHint('table' ×1),
    ObjectHint('chair' ×3)
]
```

### ✅ STAGE 1a: Object List Building (РАБОТАЕТ ПРАВИЛЬНО)
**Файл**: `creator/runner.py`, строки 420-428  
**Результат**: Правильно создает список с повторениями

```python
objects = []
for h in (scene_spec.estimated_objects or []):
    name = str(getattr(h, "name", "")).strip()
    qty = max(1, min(int(getattr(h, "quantity", 1)), 20))
    objects.extend([name] * qty)  # ✓ Правильно: ["table", "chair", "chair", "chair"]
```

### ❌ STAGE 1b: Model Selection (ПРОБЛЕМА НАЙДЕНА!)
**Файл**: `creator/runner.py`, строки 447-495  
**Проблема**: Код использует `obj_state` словарь с ключом `search_key`, что приводит к повторному использованию одной и той же модели для одинаковых объектов.

#### Проблемный код (строки 459-487):
```python
chosen_models: List[Dict[str, str]] = []
obj_state: Dict[str, Dict[str, Any]] = {}  # ← ПРОБЛЕМА: один state на тип объекта

for obj in objects:  # ["table", "chair", "chair", "chair"]
    obj_raw = str(obj).strip()
    obj_key = obj_raw.lower()
    obj_clean = re.sub(r'\([^)]*\)', '', obj_key).strip()
    search_key = _OBJECT_FALLBACKS.get(obj_clean, obj_clean)
    
    state = obj_state.get(search_key)  # ← ПРОБЛЕМА: для всех "chair" один state!
    if state is None:
        # Первый раз для этого типа объекта
        ranked = _rank_models_for_object(search_key, models, limit=10)
        primary = ranked[0]
        state = {"order": [primary], "idx": 0}
        obj_state[search_key] = state  # ← Сохраняем state для "chair"
    
    # Для второго и третьего стула state уже существует!
    picked = state["order"][0]  # ← Всегда берем ОДНУ И ТУ ЖЕ модель
    name = str(picked.get("name", "")).strip()
    uid = str(picked.get("uuid", "")).strip()
    
    if name and uid:
        chosen_models.append({"Model": name, "uuid": uid})  # ← Добавляем 3 раза
```

#### Что происходит:
1. **Итерация 1** (obj="table"): 
   - `search_key = "table"`
   - `state` не существует → создаем новый state
   - Добавляем модель стола в `chosen_models`
   
2. **Итерация 2** (obj="chair"):
   - `search_key = "chair"`
   - `state` не существует → создаем новый state
   - Добавляем модель стула в `chosen_models`
   
3. **Итерация 3** (obj="chair"):
   - `search_key = "chair"`
   - `state` **УЖЕ СУЩЕСТВУЕТ** → используем существующий state
   - Добавляем **ТУ ЖЕ** модель стула в `chosen_models`
   
4. **Итерация 4** (obj="chair"):
   - `search_key = "chair"`
   - `state` **УЖЕ СУЩЕСТВУЕТ** → используем существующий state
   - Добавляем **ТУ ЖЕ** модель стула в `chosen_models`

**Результат**: `chosen_models` содержит:
```python
[
    {"Model": "table_name", "uuid": "table_uuid"},
    {"Model": "chair_name", "uuid": "chair_uuid_1"},  # Первый стул
    {"Model": "chair_name", "uuid": "chair_uuid_1"},  # Дубликат!
    {"Model": "chair_name", "uuid": "chair_uuid_1"},  # Дубликат!
]
```

### ❌ STAGE 2: get_full_placed_models (ФИЛЬТРУЕТ ДУБЛИКАТЫ?)
**Файл**: `creator/sim_interfaces/mujoco.py`, строки 224-253  
**Возможная проблема**: Нужно проверить, фильтрует ли этот метод дубликаты по uuid

```python
def get_full_placed_models(self, placed_models: List[Dict], models: List[Dict]) -> List[Dict]:
    full_placed_models = []
    per_name_cursor: Dict[str, int] = {}
    models_by_uuid = {str(m.get("uuid")): m for m in models if m.get("uuid")}
    
    for model in placed_models:  # Проходит по ВСЕМ моделям, включая дубликаты
        uid = str(model.get("uuid") or "").strip()
        selected_entry: Optional[Dict] = None
        
        if uid and uid in models_by_uuid:
            selected_entry = copy.deepcopy(models_by_uuid[uid])  # ← Копирует ТУ ЖЕ модель
        else:
            # Fallback logic
            ...
        
        selected_entry.update(model)
        full_placed_models.append(selected_entry)  # ← Добавляет все, включая дубликаты
    
    return full_placed_models
```

**Вывод**: Этот метод НЕ фильтрует дубликаты. Он просто копирует каждую модель из `placed_models`.

### ❌ STAGE 3: Semantic Plan Generation
**Файл**: `creator/runner.py`, строки 257-270  
**Проблема**: В промпт передается список с дубликатами

```python
models_str_lines = []
for i, model in enumerate(full_placed_models):  # ← Если здесь дубликаты, они попадут в промпт
    model_name = model.get("Model", f"object_{i}")
    size = model.get("size", [1.0, 1.0, 1.0])
    model_uuid = model.get("uuid", "")
    
    models_str_lines.append(
        f"  - Model: {model_name}\n"
        f"    Size: {{'width': {size[0]:.2f}, 'length': {size[1]:.2f}, 'height': {size[2]:.2f}}}\n"
        f"    uuid: {model_uuid}"
    )

models_str = "\n".join(models_str_lines)  # ← Промпт содержит дубликаты
```

НО! В промпте есть правило (строка ~500 в `semantic_plan.py`):
```
1. **Object Count**: Include ALL objects from the available models list.
```

**Проблема**: Если в `models_str` 3 одинаковых стула с одинаковым uuid, LLM может:
- Либо создать 3 объекта с одинаковым Model
- Либо решить, что это ошибка, и создать только 1 объект

## КОРНЕВАЯ ПРИЧИНА

**Строки 459-487 в `creator/runner.py`**: Логика `obj_state` предназначена для **разнообразия** (чтобы "10 яблок" были разных цветов), но она **ломает количество** для одинаковых объектов.

Код был написан с предположением, что:
- Для "10 apples" → выбрать 10 РАЗНЫХ моделей яблок (разные цвета/формы)
- Использовать `obj_state` для round-robin через `state["order"]`

НО реализация неправильная:
- `state["order"]` содержит только ОДНУ модель: `[primary]`
- `state["idx"]` никогда не используется для выбора следующей модели
- Все повторения используют `state["order"][0]` - ОДНУ И ТУ ЖЕ модель

## РЕШЕНИЯ

### Вариант 1: Убрать кэширование obj_state (ПРОСТОЕ)
Убрать строки 459-460 и всегда создавать новый state:

```python
for obj in objects:
    obj_raw = str(obj).strip()
    obj_key = obj_raw.lower()
    obj_clean = re.sub(r'\([^)]*\)', '', obj_key).strip()
    search_key = _OBJECT_FALLBACKS.get(obj_clean, obj_clean)
    
    # УБРАТЬ ЭТО:
    # state = obj_state.get(search_key)
    # if state is None:
    
    # ВСЕГДА создавать новый state:
    ranked = _rank_models_for_object(search_key, models, limit=10)
    if not ranked:
        print(f"[pipeline] WARN: '{obj}' has no catalog match, dropped")
        dropped_objects.append(obj_key)
        continue
    
    unused = [r for r in ranked if str(r.get("uuid") or "") not in used_uuids]
    candidate_pool = unused if unused else ranked
    
    if len(candidate_pool) == 1:
        primary = candidate_pool[0]
    else:
        # LLM disambiguation (только для первого объекта каждого типа)
        ...
    
    # Добавить модель
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

**Плюсы**: Простое решение, каждый объект получает свою модель  
**Минусы**: Для "10 яблок" все будут одинаковые (но это можно исправить позже)

### Вариант 2: Исправить round-robin логику (СЛОЖНОЕ)
Исправить логику `obj_state` чтобы она действительно делала round-robin:

```python
state = obj_state.get(search_key)
if state is None:
    ranked = _rank_models_for_object(search_key, models, limit=10)
    if not ranked:
        continue
    state = {"order": ranked[:5], "idx": 0}  # ← Сохранить несколько моделей
    obj_state[search_key] = state

# Round-robin через модели
picked = state["order"][state["idx"] % len(state["order"])]  # ← Использовать idx
state["idx"] += 1  # ← Инкрементировать для следующего объекта
```

**Плюсы**: Разнообразие для повторяющихся объектов  
**Минусы**: Более сложная логика, нужно тестировать

### Вариант 3: Упростить всю логику (РАДИКАЛЬНОЕ)
Убрать всю логику disambiguation и obj_state, просто брать первую подходящую модель:

```python
chosen_models: List[Dict[str, str]] = []
used_uuids: set = set()

for obj in objects:
    obj_raw = str(obj).strip()
    obj_key = obj_raw.lower()
    obj_clean = re.sub(r'\([^)]*\)', '', obj_key).strip()
    search_key = _OBJECT_FALLBACKS.get(obj_clean, obj_clean)
    
    ranked = _rank_models_for_object(search_key, models, limit=10)
    if not ranked:
        print(f"[pipeline] WARN: '{obj}' has no catalog match, dropped")
        continue
    
    # Взять первую неиспользованную модель
    unused = [r for r in ranked if str(r.get("uuid") or "") not in used_uuids]
    primary = unused[0] if unused else ranked[0]
    
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

**Плюсы**: Максимально простая логика, легко понять и отладить  
**Минусы**: Теряем LLM disambiguation (но это может быть и плюсом - быстрее работает)

## РЕКОМЕНДАЦИЯ

**Вариант 1** - самый простой и безопасный. Он решает текущую проблему с минимальными изменениями.

Если нужно разнообразие для повторяющихся объектов, можно потом добавить **Вариант 2**.

Если нужна максимальная простота и скорость, использовать **Вариант 3**.

## ДОПОЛНИТЕЛЬНАЯ ПРОБЛЕМА

Даже если исправить Stage 1, нужно проверить, что LLM в semantic plan действительно генерирует ВСЕ объекты из списка `models_str`. Возможно, LLM игнорирует дубликаты или неправильно интерпретирует промпт.

**Решение**: Добавить в промпт более явное указание:
```
CRITICAL: You MUST generate EXACTLY {len(full_placed_models)} objects.
The available models list contains {len(full_placed_models)} entries.
Generate one object specification for EACH entry in the list, even if some have the same Model name.
Use unique IDs like: chair_1, chair_2, chair_3 for multiple chairs.
```
