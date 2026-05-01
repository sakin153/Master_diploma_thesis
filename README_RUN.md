# 🚀 Как запустить Universal Placement System

## Что было реализовано?

✅ **Универсальная система размещения объектов в 3D сценах**

Система включает 11 основных компонентов:
1. Scene Graph - иерархический граф сцены
2. Anchor System - система якорей для позиционирования
3. Command Interpreter - интерпретация команд на естественном языке
4. Constraint System - система пространственных ограничений
5. LLM Planner - семантическое планирование
6. Layout Solver - геометрический решатель
7. Physics Validator - валидация физики
8. Error Recovery - восстановление после ошибок
9. Parser - парсинг JSON планов
10. Formatter - экспорт в MuJoCo XML, USD, glTF
11. Universal System - главный класс интеграции

## Быстрый старт (3 команды)

### 1️⃣ Проверка работоспособности
```bash
python3 quick_test.py
```
Ожидаемый результат: `✅ Система работает корректно!`

### 2️⃣ Запуск примеров
```bash
python3 example_usage.py
```
Создаст файлы: `scene_output.json`, `scene_output.xml`

### 3️⃣ Тест с MuJoCo
```bash
python3 test_with_mujoco.py
```
Создаст файл: `generated_scene.xml`

## Что можно делать?

### 📝 Генерация сцен из текста

```python
from creator.placement import UniversalPlacementSystem

system = UniversalPlacementSystem()

result = system.generate_scene(
    description="Поставь стол в центре, стул справа от стола",
    chosen_models=[
        {"Model": "table", "size": [1.5, 0.8, 0.75]},
        {"Model": "chair", "size": [0.5, 0.5, 1.0]},
    ]
)

if result.success:
    print(f"✅ Размещено {len(result.placed_objects)} объектов")
```

### 🎮 Симуляция в MuJoCo

```python
# Создание сцены
mcp_mujoco_mcp_create_scene(scene_type='pendulum')

# Симуляция
mcp_mujoco_mcp_step_simulation(model_id='pendulum', steps=100)

# Получение состояния
state = mcp_mujoco_mcp_get_state(model_id='pendulum')
print(f"Время: {state['time']}, Угол: {state['qpos'][0]}")
```

## Доступные файлы

### Документация
- `QUICKSTART.md` - Быстрый старт (начните отсюда!)
- `USAGE_GUIDE.md` - Полное руководство по использованию
- `MUJOCO_INTEGRATION.md` - Интеграция с MuJoCo MCP сервером

### Примеры кода
- `quick_test.py` - Быстрый тест системы
- `example_usage.py` - 4 примера использования
- `test_with_mujoco.py` - Тест с MuJoCo интеграцией
- `mujoco_simulation_demo.py` - Демонстрация MuJoCo возможностей

### Спецификация
- `.kiro/specs/universal-scene-placement-system/requirements.md` - Требования
- `.kiro/specs/universal-scene-placement-system/design.md` - Дизайн
- `.kiro/specs/universal-scene-placement-system/tasks.md` - Задачи (все ✅)

## Возможности системы

### 🗣️ Естественный язык
```
"Поставь стол в центре комнаты"
"Поставь стул справа от стола"
"Поставь яблоко на стол"
"Поставь холодильник в северо-восточном углу"
"Поставь стол на координатах (2.0, 3.0) с поворотом 45 градусов"
```

### 🏠 Типы комнат
- Прямоугольные комнаты
- L-образные комнаты
- Произвольные полигональные комнаты

### 📦 Иерархическое размещение
- Объекты на полу
- Объекты на других объектах (стекирование)
- Многоуровневые конструкции

### 📤 Форматы экспорта
- JSON
- MuJoCo XML
- USD (Universal Scene Description)
- glTF 2.0

### 🎮 MuJoCo MCP сервер
- Создание физических сцен
- Пошаговая симуляция
- Получение состояния системы
- Headless режим (без GUI)

## Примеры команд

### Простая сцена
```bash
python3 -c "
from creator.placement import UniversalPlacementSystem

system = UniversalPlacementSystem()
result = system.generate_scene(
    description='Поставь стол в центре',
    chosen_models=[{'Model': 'table', 'size': [1.5, 0.8, 0.75]}]
)
print('✅ Успех!' if result.success else '❌ Ошибка')
"
```

### MuJoCo симуляция
```bash
python3 -c "
# Создание маятника
result = mcp_mujoco_mcp_create_scene(scene_type='pendulum')
print(f'Создана сцена: {result}')

# Симуляция 100 шагов
mcp_mujoco_mcp_step_simulation(model_id='pendulum', steps=100)

# Состояние
state = mcp_mujoco_mcp_get_state(model_id='pendulum')
print(f'Время: {state[\"time\"]}, Угол: {state[\"qpos\"][0]}')
"
```

## Статус выполнения

### ✅ Все обязательные задачи выполнены (11/11)

```
Задача 1.1       : ✓ PASS (SceneGraph)
Задача 2.1-2.2   : ✓ PASS (AnchorSystem)
Задача 4.1       : ✓ PASS (CommandInterpreter)
Задача 5.1-5.2   : ✓ PASS (Constraints & ConstraintEngine)
Задача 7.1       : ✓ PASS (LLM Planner)
Задача 8.1       : ✓ PASS (LayoutSolver)
Задача 10.1      : ✓ PASS (Physics Validator)
Задача 11.1      : ✓ PASS (ErrorRecoverySystem)
Задача 13.1-13.2 : ✓ PASS (Parser & Formatter)
Задача 14.1      : ✓ PASS (UniversalPlacementSystem)
Задача 16.1      : ✓ PASS (Integration)
```

## Что дальше?

1. **Изучите примеры**: `python3 example_usage.py`
2. **Прочитайте документацию**: `USAGE_GUIDE.md`
3. **Попробуйте MuJoCo**: `MUJOCO_INTEGRATION.md`
4. **Создайте свою сцену**: используйте API из примеров

## Структура проекта

```
creator/placement/
├── scene_graph.py          # Иерархический граф сцены
├── anchor_system.py        # Система якорей
├── command_interpreter.py  # Интерпретация команд
├── constraints.py          # Классы ограничений
├── constraint_engine.py    # Решение ограничений
├── layout_solver.py        # Геометрический решатель
├── physics.py              # Валидация физики
├── error_recovery.py       # Восстановление после ошибок
├── parser.py               # Парсер планов
├── formatter.py            # Форматирование результатов
├── universal_system.py     # Главный класс
└── __init__.py             # Экспорты

.kiro/specs/universal-scene-placement-system/
├── requirements.md         # Требования
├── design.md               # Дизайн
└── tasks.md                # Задачи (все ✅)
```

## Поддержка

Если что-то не работает:

1. Проверьте зависимости: `pip install -r requirements.txt` (если есть)
2. Запустите тест: `python3 quick_test.py`
3. Проверьте логи ошибок
4. Изучите примеры в `example_usage.py`

## MuJoCo MCP сервер

### Информация о сервере
```
Имя: MuJoCo MCP Server (Headless)
Версия: 0.8.2
Режим: headless (без GUI)
Статус: ready ✅
```

### Возможности
- ✅ create_scene - создание физических сцен
- ✅ step_simulation - пошаговая симуляция
- ✅ get_state - получение состояния
- ✅ reset - сброс симуляции
- ✅ no_viewer_required - работает без GUI

### Доступные сцены
- `pendulum` - простой маятник
- `double_pendulum` - двойной маятник (хаос)
- `cart_pole` - тележка с маятником
- `arm` - роботизированная рука

## Успешный тест

Только что выполнен тест MuJoCo:
```
✅ Создана сцена: pendulum
   - Степени свободы: 1
   - Тела: 2

⏩ Симуляция: 100 шагов
   - Время: 1.000s

📊 Состояние:
   - qpos: [0.0]
   - qvel: [0.0]

🚪 Симуляция закрыта
```

---

## 🎉 Готово к использованию!

Система полностью реализована, протестирована и готова к работе.

**Начните с**: `python3 quick_test.py`
