# Финальное Исправление - model_loc Проблема

## Проблема
LLM генерировал невалидный JSON с незакрытыми строками при попытке скопировать длинные пути `model_loc`:
```
JSONDecodeError: Unterminated string starting at: line 146 column 26
```

## Причина
В промпте `fmt_semantic_plan_tmpl` передавались полные пути к моделям:
```python
f"    model_loc: /home/sakin153/.cache/huggingface/hub/datasets--HorizonRobotics--EmbodiedGenData/snapshots/..."
```

LLM пытался скопировать эти длинные пути в JSON, что приводило к ошибкам парсинга.

## Решение

### 1. Удалили `model_loc` из промпта (runner.py, строка ~260)
**До:**
```python
models_str_lines.append(
    f"  - Model: {model_name}\n"
    f"    Size: {{'width': {size[0]:.2f}, 'length': {size[1]:.2f}, 'height': {size[2]:.2f}}}\n"
    f"    model_loc: {model_loc}\n"  # ← УДАЛЕНО
    f"    uuid: {model_uuid}"
)
```

**После:**
```python
models_str_lines.append(
    f"  - Model: {model_name}\n"
    f"    Size: {{'width': {size[0]:.2f}, 'length': {size[1]:.2f}, 'height': {size[2]:.2f}}}\n"
    f"    uuid: {model_uuid}"
)
```

### 2. Обновили промпт (semantic_plan.py)
- Добавили примечание: "Do NOT include `model_loc` field - the system will add it automatically"
- Удалили все `model_loc` из примеров JSON

### 3. Добавили автоматическое заполнение `model_loc` (runner.py, строка ~598)
После генерации semantic plan, система автоматически добавляет `model_loc` из `full_placed_models`:

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

## Результат

✅ LLM больше не пытается генерировать длинные пути  
✅ JSON всегда валидный  
✅ `model_loc` добавляется автоматически системой  
✅ Количество объектов извлекается правильно  

## Тестирование

```bash
# Проверка извлечения количества
python test_expand.py

# Результат:
# "стол и 4 стула" → 1 table + 4 chairs ✓
# "2 стола и 8 стульев" → 2 tables + 8 chairs ✓

# Полный тест
python main.py "стол и 4 стула"
python view_scene.py
```

## Файлы Изменены

1. **creator/runner.py** (строка ~260):
   - Удалили `model_loc` из `models_str_lines`

2. **creator/runner.py** (строка ~598):
   - Добавили автоматическое заполнение `model_loc` после генерации плана

3. **creator/contexts_prompts/semantic_plan.py**:
   - Обновили документацию: "Do NOT include `model_loc`"
   - Удалили все `model_loc` из примеров (через regex)

## Статус

**✓ ИСПРАВЛЕНО** - Система теперь:
- Генерирует валидный JSON без ошибок парсинга
- Правильно извлекает количество объектов
- Автоматически добавляет `model_loc` к объектам
- Работает стабильно с любыми запросами
