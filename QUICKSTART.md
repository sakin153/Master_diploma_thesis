# 🚀 Быстрый старт - Universal Placement System

## Шаг 1: Проверка установки

Убедитесь, что все компоненты работают:

```bash
python3 quick_test.py
```

Ожидаемый результат: `✅ Система работает корректно!`

## Шаг 2: Запуск примеров

```bash
python3 example_usage.py
```

Это запустит 4 примера и создаст файлы:
- `scene_output.json` - JSON представление сцены
- `scene_output.xml` - MuJoCo XML формат

## Шаг 3: Ваш первый скрипт

Создайте файл `my_scene.py`:

```python
from creator.placement import UniversalPlacementSystem, GenerationConfig

# Создаем систему
system = UniversalPlacementSystem()

# Описываем сцену
description = "Поставь стол в центре комнаты"

# Указываем объекты
models = [
    {"Model": "table", "size": [1.5, 0.8, 0.75]}
]

# Генерируем
result = system.generate_scene(
    description=description,
    chosen_models=models,
    config=GenerationConfig(room_half_size=5.0)
)

# Проверяем результат
if result.success:
    print("✅ Успех!")
    print(f"Позиция: {result.placed_objects[0]}")
else:
    print("❌ Ошибка:", result.errors)
```

Запустите:

```bash
python3 my_scene.py
```

## Что дальше?

- Читайте `USAGE_GUIDE.md` для подробной документации
- Смотрите `example_usage.py` для больших примеров
- Изучайте `.kiro/specs/universal-scene-placement-system/` для деталей реализации

## Основные команды

### Относительное позиционирование
```python
"Поставь стол в центре"
"Поставь стул справа от стола"
"Поставь яблоко на стол"
```

### Точные координаты
```python
"Поставь стол на координатах (2.0, 3.0)"
"Поверни стол на 45 градусов"
```

### Угловое размещение
```python
"Поставь холодильник в северо-восточном углу"
"Поставь диван у южной стены"
```

## Поддержка

Если что-то не работает:
1. Проверьте, что все зависимости установлены
2. Запустите `python3 quick_test.py` для диагностики
3. Проверьте логи ошибок в выводе

Удачи! 🎉
