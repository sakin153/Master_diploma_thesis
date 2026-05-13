# Project - Чистый Pipeline для Генерации Сцен

Эта папка содержит пошаговую сборку финального проекта из проверенных компонентов текущего pipeline.

## Структура папок

```
project/
├── prompts/              # LLM промпты
│   ├── expand_prompt.txt # Промпт для расширения запроса
│   └── README.md         # Документация промптов
├── prompt_expander.py    # Stage 0: Расширение запроса
├── test_stage0.py        # Тест Stage 0
├── API.md                # API документация
└── README.md             # Этот файл
```

## Цель

Собрать минимальный, чистый pipeline для генерации 3D сцен MuJoCo из текстовых запросов, включая только необходимые функции без legacy кода.

## Структура Pipeline

### Stage 0: Prompt Expansion ✅

**Файл:** `prompt_expander.py`

**Промпт:** `prompts/expand_prompt.txt`

**Функция:** `expand_prompt(query, prompt_model_fn, llm_model, verbose=True) -> SceneSpec`

**Что делает:**
- Принимает короткий запрос пользователя (например, "офис с 5 столами")
- Расширяет его в детальную спецификацию сцены через LLM
- Извлекает структурированную информацию:
  - `room_type`: тип комнаты (office, bedroom, kitchen и т.д.)
  - `room_style`: стиль (modern, minimalist, cozy и т.д.)
  - `estimated_objects`: список объектов с количеством
  - `room_half_size`: начальная оценка размера комнаты
  - `anchor_objects`: главные объекты (столы, кровати и т.д.)

**Зависимости:**
- LLM функция `prompt_model_fn(system_prompt, user_query, model_name)`
- Стандартные библиотеки: `re`, `json`

**Входные данные:**
```python
query = "офис с 5 столами"
```

**Выходные данные:**
```python
SceneSpec(
    original_query="офис с 5 столами",
    expanded_description="A professional office space with 5 desks...",
    room_type="office",
    room_style="modern",
    estimated_objects=[
        ObjectHint(name="desk", quantity=5),
        ObjectHint(name="chair", quantity=5),
        ...
    ],
    room_half_size=4.0,  # 8m × 8m room
    anchor_objects=["desk"]
)
```

**Особенности:**
- Поддержка русского и английского языков
- Автоматическое ограничение количества объектов (≤20 экземпляров)
- Встроенный промпт для LLM с правилами генерации сцен
- Извлечение иерархии объектов (anchor objects vs dependent objects)

## Следующие этапы

- [x] Stage 0: Prompt Expansion ✅
- [x] Stage 1: Object Extraction (выбор моделей из каталога) ✅
- [x] Stage 2: Model Loading & Scaling ✅
- [x] Stage 3: Room Sizing (вычисление размера комнаты) ✅
- [x] Stage 4: Semantic Plan Generation + Collision Resolution ✅
- [x] Stage 5: MuJoCo Assembly (сборка XML) ✅
- [x] Stage 6.5: Preview Rendering (рендеринг превью) ✅

## Использование

```python
from project.prompt_expander import expand_prompt

# Предполагается что у вас есть функция для вызова LLM
def my_llm_function(system_prompt, user_query, model_name):
    # Ваша реализация вызова LLM
    # Должна вернуть dict или JSON string
    pass

# Расширение запроса
scene_spec = expand_prompt(
    query="офис с 5 столами",
    prompt_model_fn=my_llm_function,
    llm_model="deepseek-v3.1:671b-cloud",
    verbose=True
)

# Использование результата
print(f"Room type: {scene_spec.room_type}")
print(f"Objects: {scene_spec.estimated_objects}")
print(f"Room size: {scene_spec.room_half_size * 2}m × {scene_spec.room_half_size * 2}m")
```

## Статус

- ✅ Stage 0-5: **ГОТОВО** (Prompt Expansion → MuJoCo Assembly)
- ✅ Stage 6.5: **ГОТОВО** (Preview Rendering)
