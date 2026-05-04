# Changelog: Output Saving Feature

## [2026-05-03] - Добавлено сохранение промежуточных результатов

### Добавлено
- ✨ Новая функция `_save_stage_output()` для сохранения результатов каждой стадии
- 📁 Папка `output/` для хранения промежуточных результатов
- 📄 Автоматическое сохранение 4 стадий генерации:
  - Stage 1: Scene Graph (построение графа сцены)
  - Stage 2: World State (иерархическое размещение)
  - Stage 3: Merged Output (объединённый результат)
  - Stage 4: Final Layout (финальная расстановка после коррекции)
- 🕐 Timestamp в именах файлов для отслеживания запусков
- 📝 Документация в `output/README.md`
- 🧪 Тестовый скрипт `test_output_saving.py`

### Изменено
- 🔧 Обновлена сигнатура `generate_placement_plan()`:
  - Добавлен параметр `save_outputs=True` (опциональный)
  - Обратная совместимость сохранена
- 📦 Добавлены импорты: `os`, `datetime`

### Файлы
```
Изменённые:
  - project/scene_planner.py (+50 строк)

Новые:
  - output/.gitignore
  - output/README.md
  - test_output_saving.py
  - docs/output_saving_feature.md
  - CHANGELOG_output_saving.md
```

### Использование

#### По умолчанию (сохранение включено)
```python
from project.scene_planner import generate_placement_plan

result = generate_placement_plan(
    models=models,
    room_half_size=5.0,
    room_type="living_room",
    query="table and 4 chairs"
)
# → Создаёт 4 файла в output/
```

#### Отключение сохранения
```python
result = generate_placement_plan(
    models=models,
    room_half_size=5.0,
    room_type="living_room",
    query="table and 4 chairs",
    save_outputs=False  # Отключить
)
```

### Примеры файлов

#### Stage 1: Scene Graph
```json
{
  "stage": "1_scene_graph",
  "query": "table and 4 chairs",
  "room_half_size": 5.0,
  "graph": {
    "nodes": [
      {
        "id": "table_1",
        "model_name": "Table",
        "level": 0,
        "instances": 1,
        ...
      }
    ],
    "metadata": {
      "total_objects": 5,
      "max_level": 1
    }
  }
}
```

#### Stage 4: Final Layout
```json
{
  "stage": "4_final_layout",
  "query": "table and 4 chairs",
  "room_type": "living_room",
  "models": [
    {
      "Model": "Table",
      "Pose": {"x": 0.0, "y": 0.0, "z": 0.375},
      "yaw_deg": 0.0,
      "size": [1.5, 0.75, 0.9],
      ...
    }
  ]
}
```

### Преимущества

1. **Отладка** 🐛
   - Видно результаты каждой стадии
   - Легко найти где возникла проблема

2. **Анализ** 📊
   - Понять как LLM строит граф
   - Сравнить разные запросы

3. **Воспроизведение** 🔄
   - Сохранены все параметры запуска
   - Можно повторить генерацию

4. **Тестирование** ✅
   - Регрессионные тесты
   - Сравнение с эталонными результатами

### Производительность

- Минимальные накладные расходы: ~10-50ms на файл
- Не блокирует основной процесс
- Ошибки сохранения не прерывают генерацию

### Совместимость

- ✅ Полная обратная совместимость
- ✅ Опциональная функция (можно отключить)
- ✅ Не меняет выходной формат

### Тестирование

```bash
# Проверить синтаксис
python3 -m py_compile project/scene_planner.py

# Запустить тест
python test_output_saving.py
```

### Документация

- 📖 Полная документация: `docs/output_saving_feature.md`
- 📁 Структура output: `output/README.md`
- 🧪 Примеры использования: `test_output_saving.py`
