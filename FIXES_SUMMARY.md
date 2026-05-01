# Исправления и Инструкции по Тестированию

## Исправленные Проблемы

### 1. ✅ Конфликт имен переменных с модулем `uuid`

**Проблема**: 
```
UnboundLocalError: cannot access local variable 'uuid' where it is not associated with a value
```

**Причина**: В коде использовались локальные переменные с именем `uuid`, которые перекрывали импортированный модуль `uuid`.

**Исправление**:
- В `creator/runner.py`, строка ~262: `uuid = model.get("uuid", "")` → `model_uuid = model.get("uuid", "")`
- В `creator/runner.py`, строка ~714: `uuid = placed_obj.uuid` → `obj_uuid = placed_obj.uuid`

**Статус**: ✅ Исправлено

---

### 2. ✅ Добавлено детальное логирование

**Что добавлено**:
- Print-ы для отладки генерации семантического плана
- Вывод типа исключения при ошибках
- Полный traceback при fallback на старую систему

**Файл**: `creator/runner.py`, функция `generate_world()`, Stage 4

---

## Тестирование

### Шаг 1: Проверка исправления UUID

```bash
python test_uuid_fix.py
```

**Ожидаемый результат**:
```
============================================================
Testing UUID Fix
============================================================
✓ Can import uuid module: <module 'uuid' from '...'>
✓ Can generate UUID: <UUID>
✓ Successfully imported creator.runner
✓ generate_world function exists
✓ _generate_semantic_plan_with_validation function exists

============================================================
✓ All UUID fix tests passed!
============================================================
```

---

### Шаг 2: Проверка промпт-шаблона

```bash
python test_prompt_template.py
```

**Ожидаемый результат**:
```
============================================================
Testing Semantic Plan Prompt Template
============================================================
✓ Successfully imported fmt_semantic_plan_tmpl
✓ Template length: XXXX characters
✓ Contains 'schema_version': True
✓ Contains 'position': True
✓ Contains 'orientation': True
... (все проверки должны пройти)
```

---

### Шаг 3: Тестирование генерации сцены

```bash
python main.py create "Простая столовая где есть 4 стола и 16 стульев у каждого стола по 4 стула"
```

**Что смотреть в логах**:

1. **Если семантический план генерируется успешно**:
   ```
   [pipeline] Attempting to generate semantic plan in new format...
   [SemanticPlan] Generating semantic plan (attempt 1/3)
   [SemanticPlan] Schema validation passed on attempt 1
   [pipeline] Generated semantic plan with 20 objects
   [pipeline] Executing Semantic Enforcement Pipeline
   [Pipeline] Starting semantic enforcement pipeline
   ...
   [pipeline] Semantic Enforcement Pipeline placed 20 objects
   ```

2. **Если происходит fallback**:
   ```
   [pipeline] Attempting to generate semantic plan in new format...
   [pipeline] WARN: Failed to generate semantic plan in new format: <причина>
   [pipeline] Exception type: <тип ошибки>
   <traceback>
   [pipeline] Falling back to old constraint-based system
   [pipeline] Universal System placed 20 objects
   ```

---

## Текущее Состояние

### ✅ Что работает:

1. **Исправлена ошибка с uuid** - система больше не падает с `UnboundLocalError`
2. **Создан новый промпт-шаблон** - `creator/contexts_prompts/semantic_plan.py`
3. **Интегрирован SemanticEnforcementPipeline** - в `creator/runner.py`
4. **Добавлен fallback механизм** - система использует старую систему при ошибках
5. **Добавлено детальное логирование** - можно диагностировать проблемы

### ❓ Что нужно проверить:

1. **Генерирует ли LLM семантический план в новом формате?**
   - Если нет, нужно посмотреть на ошибки в логах
   - Возможные причины: LLM не понимает промпт, ошибка валидации схемы, ошибка парсинга JSON

2. **Работает ли SemanticEnforcementPipeline?**
   - Если план генерируется, но pipeline падает, нужно смотреть traceback

3. **Правильно ли размещаются объекты?**
   - Если pipeline работает, нужно проверить финальную сцену

---

## Следующие Шаги

### Если система использует fallback:

1. **Посмотрите на логи** - найдите строку с `[pipeline] WARN: Failed to generate semantic plan`
2. **Определите причину**:
   - `RuntimeError: Failed to parse LLM response as JSON` → LLM не генерирует валидный JSON
   - `SchemaValidationError: Schema validation failed` → LLM генерирует JSON, но не соответствует схеме
   - Другая ошибка → смотрите traceback

3. **Возможные решения**:
   - Если LLM не генерирует JSON: нужно улучшить промпт
   - Если схема не проходит валидацию: нужно посмотреть, какие поля отсутствуют
   - Если другая ошибка: нужно исправить код

### Если система использует SemanticEnforcementPipeline:

1. **Проверьте финальную сцену**:
   ```bash
   python view_scene.py
   ```

2. **Проверьте pipeline trace**:
   - Файл: `analysis_logs/pipeline_trace_<timestamp>.json`
   - Содержит информацию о каждом этапе pipeline

3. **Если объекты размещены неправильно**:
   - Проверьте семантический план (что LLM сгенерировал)
   - Проверьте, правильно ли работают resolvers (distance, orientation)

---

## Контакты для Отладки

Если нужна помощь, предоставьте:

1. **Полные логи** от запуска `python main.py create "<запрос>"`
2. **Traceback** если есть ошибка
3. **Содержимое** `analysis_logs/pipeline_trace_<latest>.json` если pipeline запустился

---

## Дополнительные Тестовые Скрипты

- `test_uuid_fix.py` - проверка исправления uuid
- `test_prompt_template.py` - проверка промпт-шаблона
- `test_semantic_integration.py` - проверка интеграции pipeline
- `test_dining_room.py` - полный тест с оригинальным запросом
- `view_scene.py` - просмотр сгенерированной сцены

---

## Заключение

Основная ошибка с `uuid` исправлена. Система теперь должна работать, но может использовать fallback на старую систему, если LLM не генерирует семантический план в новом формате.

Следующий шаг - запустить тесты и посмотреть на логи, чтобы понять, почему происходит fallback (если он происходит).
