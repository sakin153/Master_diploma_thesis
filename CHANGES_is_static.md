# Изменения: LLM-определение is_static для объектов

## Проблема
Ранее физические свойства объектов (статичный/динамический) определялись жестко закодированным списком в `physics_profile.py`. Это приводило к проблемам:
- Телевизор всегда был статичным, даже когда стоял на столе → висел в воздухе
- Невозможно было гибко управлять физикой объектов
- LLM не могла принимать решения о физических свойствах

## Решение
Удален жестко закодированный список. Теперь LLM определяет `is_static` для каждого объекта в semantic plan.

## Изменения

### 1. Промпт (`creator/contexts_prompts/constraints.py`)
Добавлена инструкция для LLM:
```
PHYSICS PROPERTIES:
For each object, specify "is_static" (boolean):
- true: Heavy furniture, large appliances, mounted objects
- false: Small movable objects that can be picked up or fall
```

### 2. Парсинг (`creator/placement/plan.py`)
- `build_semantic_plan` теперь извлекает `is_static` из ответа LLM
- Автоматическая логика: `wall_mounted` → `is_static=True`

### 3. Физический профиль (`creator/scene/physics_profile.py`)
- Удалены списки `_PROFILE_RULES`, `STATIC_LIGHT`, `STATIC_APPLIANCE`
- Удалены функции `get_physics_profile()`, `is_static_object()`
- `get_physics_profile_for_model()` теперь принимает `is_static` как параметр
- Если `is_static=None` → default `True` (безопасный fallback)

### 4. Использование (`mujoco.py`, `engine_refine.py`)
Все вызовы обновлены:
```python
profile = get_physics_profile_for_model(
    model_name, 
    size=model.get("size"), 
    is_static=model.get("is_static")
)
```

## Результат
- ✅ Телевизор на столе теперь получает `is_static=False` и `free joint` → падает на стол
- ✅ LLM контролирует физику объектов
- ✅ Настенные объекты автоматически статичные
- ✅ Безопасный fallback: неизвестные объекты → статичные (не летают)

## Обратная совместимость
Если LLM не вернет `is_static` для объекта, используется default `True` (статичный).
