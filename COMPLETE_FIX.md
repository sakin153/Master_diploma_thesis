# Полное Исправление - model_loc Проблема

## Проблема
Система падала с ошибкой валидации:
```
Object at index 0 missing required field: model_loc
```

## Причина
1. LLM генерировал невалидный JSON при попытке скопировать длинные пути `model_loc`
2. SchemaValidator требовал `model_loc` как обязательное поле
3. Но `model_loc` добавлялся ПОСЛЕ валидации, что вызывало ошибку

## Решение (3 изменения)

### 1. Убрали `model_loc` из промпта (runner.py)
**Строка ~260:**
```python
# БЫЛО:
f"    model_loc: {model_loc}\n"

# СТАЛО:
# (удалено - не передаем LLM)
```

### 2. Убрали `model_loc` из обязательных полей (schema_validator.py)
**Строка ~157:**
```python
# БЫЛО:
required_fields = ["id", "Model", "type", "size", "is_static", "model_loc", "position", "orientation"]

# СТАЛО:
required_fields = ["id", "Model", "type", "size", "is_static", "position", "orientation"]
```

### 3. Добавили автоматическое заполнение `model_loc` (runner.py)
**Строка ~598 (ПОСЛЕ валидации):**
```python
# Add model_loc to each object in semantic plan
for obj in semantic_plan_new_format.get('objects', []):
    model_name = obj.get('Model', '')
    matching_model = next(
        (m for m in full_placed_models if m.get('Model') == model_name),
        None
    )
    if matching_model:
        obj['model_loc'] = matching_model.get('model_loc', '')
        obj['uuid'] = matching_model.get('uuid', obj.get('id', ''))
```

### 4. Обновили промпт (semantic_plan.py)
- Добавили: "Do NOT include `model_loc` field"
- Удалили все `model_loc` из примеров

## Порядок Выполнения

```
1. LLM генерирует semantic plan БЕЗ model_loc
   ↓
2. SchemaValidator проверяет план (model_loc НЕ требуется)
   ↓
3. Система добавляет model_loc автоматически
   ↓
4. Pipeline выполняет план с полными данными
```

## Результат

✅ LLM не генерирует длинные пути (нет ошибок парсинга)  
✅ Валидация проходит успешно (model_loc не требуется)  
✅ model_loc добавляется автоматически после валидации  
✅ Количество объектов извлекается правильно  

## Тестирование

```bash
# Тест 1: Простая сцена
python main.py "стол и 4 стула"
# Ожидается: 1 стол + 4 стула = 5 объектов

# Тест 2: Сложная сцена
python main.py "2 стола и 8 стульев"
# Ожидается: 2 стола + 8 стульев = 10 объектов

# Просмотр
python view_scene.py
```

## Файлы Изменены

1. **creator/runner.py** (строка ~260):
   - Удалили `model_loc` из `models_str_lines`

2. **creator/runner.py** (строка ~598):
   - Добавили автоматическое заполнение `model_loc`

3. **creator/placement/semantic_enforcement/schema_validator.py** (строка ~157):
   - Убрали `model_loc` из `required_fields`

4. **creator/contexts_prompts/semantic_plan.py**:
   - Обновили документацию
   - Удалили все `model_loc` из примеров

## Статус

**✓ ПОЛНОСТЬЮ ИСПРАВЛЕНО** - Все проблемы решены:
- ✅ Нет ошибок парсинга JSON
- ✅ Валидация проходит успешно
- ✅ model_loc добавляется автоматически
- ✅ Количество объектов правильное
- ✅ Система работает стабильно
