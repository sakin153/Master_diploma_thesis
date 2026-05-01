# Команды для запуска

## Полная последовательность команд

### 1. Активация виртуальной среды
```bash
source .venv/bin/activate
```

### 2. Запуск генерации сцены
```bash
python main.py "Простая столовая где есть 4 стола и 16 стульев у каждого стола по 4 стула"
```

## Альтернативный вариант (без активации среды)

```bash
.venv/bin/python main.py "Простая столовая где есть 4 стола и 16 стульев у каждого стола по 4 стула"
```

## Полная команда одной строкой

```bash
source .venv/bin/activate && python main.py "Простая столовая где есть 4 стола и 16 стульев у каждого стола по 4 стула"
```

## Что должно произойти

1. **Загрузка моделей**:
```
[pipeline] Loaded 452 models from EmbodiedGen dataset
```

2. **Расширение промпта**:
```
[prompt_expander] Expanding query: 'Простая столовая...'
```

3. **Выбор объектов**:
```
[Stage1] 'dining table' → table (uuid=0033f5a7)
[Stage1] 'chair' → computer chair (uuid=aeea9a5e)
```

4. **Размер комнаты**:
```
[room_planner] other: footprint=13.45m², ×10.0 → area=134.5m², room=12m×12m (half=6.0m)
[pipeline] Final room: 12m × 12m
```

5. **КРИТИЧНО - Использование semantic plan**:
```
[UniversalPlacementSystem] Using pre-built semantic plan from runner
```

6. **Размещение объектов**:
```
[pipeline] Universal System placed 20 objects
```

7. **Успех**:
```
[pipeline] Done. Scene saved to: .cache/worlds/scene_latest.xml
```

## Проверка результата

### Открыть сцену в MuJoCo viewer
```bash
python main.py  # без аргументов откроет последнюю сцену
```

### Или указать путь явно
```bash
python -c "import mujoco; import mujoco.viewer; m = mujoco.MjModel.from_xml_path('.cache/worlds/scene_latest.xml'); mujoco.viewer.launch(m)"
```

## Ожидаемый результат в viewer

✅ **4 стола** (коричневые/красные объекты)
✅ **16 стульев** (синие объекты)
✅ **Стулья вокруг столов** (4 стула вокруг каждого стола)
✅ **Стулья повернуты лицом к столам**
✅ **Нет дубликатов**
✅ **Правильное расстояние** между объектами

## Если возникли ошибки

### Ошибка: "command not found: source"
Используйте альтернативный вариант без активации:
```bash
.venv/bin/python main.py "Простая столовая где есть 4 стола и 16 стульев у каждого стола по 4 стула"
```

### Ошибка: "No module named 'mujoco'"
Установите зависимости:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Ошибка: "UnboundLocalError: fmt_constraints_plan_tmpl"
✅ **УЖЕ ИСПРАВЛЕНО** - просто запустите команду снова

### Ошибка: "36 стульев вместо 16"
✅ **УЖЕ ИСПРАВЛЕНО** - semantic_plan теперь передается правильно

## Быстрый тест

Если хотите быстро проверить, что все работает:

```bash
source .venv/bin/activate && python test_universal_system_fix.py
```

Этот скрипт запустит тот же сценарий и покажет результат.

## Готово! 🚀

Скопируйте и выполните команды выше. Система должна сгенерировать правильную сцену!
