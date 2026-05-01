# 🚀 ЗАПУСК СИСТЕМЫ

## Быстрый Старт

### Вариант 1: Автоматический запуск (рекомендуется)

```bash
./run_and_view.sh
```

Этот скрипт:
1. Проверит, что все исправления работают
2. Сгенерирует сцену "стол и стул"
3. Откроет viewer для просмотра

---

### Вариант 2: Пошаговый запуск

#### Шаг 1: Проверка системы
```bash
python quick_test.py
```

**Ожидаемый результат:**
```
============================================================
Quick System Test
============================================================
Testing imports...
✓ uuid module: <module 'uuid'>
✓ semantic_plan template: XXXX chars
✓ SemanticEnforcementPipeline: <class>
✓ generate_world: <function>
✓ _generate_semantic_plan_with_validation: <function>

✅ All imports successful!

Testing UUID fix...
✓ Can generate UUID: ...
✓ Can use uuid in dict: ...

✅ UUID fix works!

============================================================
Test Summary
============================================================
✅ PASS: Imports
✅ PASS: UUID Fix

🎉 All tests passed! System is ready.
```

#### Шаг 2: Генерация сцены
```bash
python main.py "стол и стул"
```

**Что смотреть в логах:**

✅ **Если работает новый pipeline:**
```
[pipeline] Attempting to generate semantic plan in new format...
[SemanticPlan] Generating semantic plan (attempt 1/3)
[SemanticPlan] Schema validation passed on attempt 1
[pipeline] Generated semantic plan with 2 objects
[pipeline] Executing Semantic Enforcement Pipeline
[Pipeline] Starting semantic enforcement pipeline
[Pipeline] Stage 1: Schema Validation
[Pipeline] Stage 2: Distance Resolution
[Pipeline] Stage 3: Orientation Resolution
[Pipeline] Stage 4: Placement Execution
[Pipeline] Stage 5: Scene Assembly
[pipeline] Semantic Enforcement Pipeline placed 2 objects
```

⚠️ **Если fallback на старую систему:**
```
[pipeline] Attempting to generate semantic plan in new format...
[pipeline] WARN: Failed to generate semantic plan in new format: <причина>
[pipeline] Falling back to old constraint-based system
[pipeline] Universal System placed 2 objects
```

#### Шаг 3: Просмотр сцены
```bash
python view_scene.py
```

---

## Тестирование с Оригинальным Запросом

```bash
python main.py "Простая столовая где есть 4 стола и 16 стульев у каждого стола по 4 стула"
```

**Ожидаемый результат:**
- 4 стола размещены в комнате
- 16 стульев (по 4 у каждого стола)
- Каждый стул **лицом к своему столу** (относительная ориентация)
- Каждый стул **на правильном расстоянии** от стола (относительная позиция)

---

## Что Исправлено

### ✅ Критические Ошибки

1. **UUID конфликт** - исправлен
   - Было: `uuid = model.get("uuid")`
   - Стало: `model_uuid = model.get("uuid")`

2. **Интеграция pipeline** - завершена
   - Создан промпт-шаблон `semantic_plan.py`
   - Интегрирован `SemanticEnforcementPipeline`
   - Добавлен fallback механизм

3. **Логирование** - улучшено
   - Детальные логи на каждом этапе
   - Traceback при ошибках
   - Диагностика fallback

---

## Архитектура Решения

```
User Query: "стол и стул"
    ↓
LLM генерирует Semantic Plan:
{
  "objects": [
    {
      "id": "table_1",
      "position": {"absolute": {"x": 0, "y": 0, "z": 0}},
      "orientation": {"absolute": {"yaw_deg": 0}}
    },
    {
      "id": "chair_1",
      "position": {
        "relative": {
          "relative_to": "table_1",
          "direction": "front",
          "distance": 0.6
        }
      },
      "orientation": {
        "relative": {
          "facing": "table_1",
          "facing_direction": "front"
        }
      }
    }
  ]
}
    ↓
SemanticEnforcementPipeline:
    ↓
1. SchemaValidator - проверяет структуру
    ↓
2. DistanceResolver - преобразует relative position → absolute
   chair_1: relative "front of table_1, 0.6m" → absolute (0, 0.6, 0)
    ↓
3. OrientationResolver - преобразует relative orientation → absolute
   chair_1: relative "facing table_1" → absolute yaw=180°
    ↓
4. PlacementExecutor - размещает объекты ТОЧНО по плану
    ↓
5. SceneAssembly - создает MuJoCo XML
    ↓
Final Scene: стул стоит лицом к столу на расстоянии 0.6м
```

---

## Диагностика Проблем

### Если система использует fallback:

1. **Посмотрите на логи** - найдите строку:
   ```
   [pipeline] WARN: Failed to generate semantic plan in new format: <причина>
   ```

2. **Возможные причины:**
   - LLM не генерирует валидный JSON
   - Схема не проходит валидацию
   - Ошибка в коде pipeline

3. **Решение:**
   - Проверьте traceback в логах
   - Запустите `python quick_test.py` для проверки импортов
   - Проверьте, что `creator/contexts_prompts/semantic_plan.py` существует

### Если объекты размещены неправильно:

1. **Проверьте semantic plan** - что LLM сгенерировал
2. **Проверьте pipeline trace** - `analysis_logs/pipeline_trace_*.json`
3. **Проверьте resolvers** - правильно ли работают distance/orientation

---

## Файлы для Проверки

- `quick_test.py` - быстрая проверка системы
- `run_and_view.sh` - автоматический запуск
- `main.py` - основной скрипт генерации
- `view_scene.py` - просмотр сцены
- `FIXES_SUMMARY.md` - детальная документация

---

## Контакты

Если нужна помощь, предоставьте:
1. Полные логи от `python main.py "<запрос>"`
2. Результат `python quick_test.py`
3. Содержимое `analysis_logs/pipeline_trace_<latest>.json`

---

## 🎯 ИТОГО

**Все исправлено и готово к запуску!**

Запустите:
```bash
./run_and_view.sh
```

Или пошагово:
```bash
python quick_test.py
python main.py "стол и стул"
python view_scene.py
```

**Система должна:**
- ✅ Генерировать semantic plan с относительными позициями
- ✅ Преобразовывать их в абсолютные координаты
- ✅ Размещать объекты с правильной ориентацией
- ✅ Создавать корректный MuJoCo XML

**Модель (LLM) задает:**
- Связи между объектами ("chair_1 относительно table_1")
- Смещение ("direction: front, distance: 0.6")
- Расстояние между ними (0.6 метра)
- Ориентацию ("facing: table_1")

**Обработчики (Resolvers) преобразуют:**
- DistanceResolver: relative position → absolute coordinates
- OrientationResolver: relative orientation → absolute angles
- PlacementExecutor: размещает ТОЧНО по плану
- SceneAssembly: создает MuJoCo XML

🚀 **ЗАПУСКАЙТЕ!**
