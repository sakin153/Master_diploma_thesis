# Prompts - LLM Промпты для Pipeline

Эта папка содержит все промпты для LLM, используемые в pipeline генерации сцен.

## Структура

### Stage 0: Prompt Expansion

**Файл:** `expand_prompt.txt`

**Используется в:** `prompt_expander.py`

**Назначение:** Расширение короткого пользовательского запроса в детальную спецификацию сцены

**Входные данные:**
- Короткий запрос пользователя (например, "офис с 5 столами")

**Выходные данные (JSON):**
```json
{
  "expanded_description": "Детальное описание сцены",
  "room_type": "office|bedroom|kitchen|...",
  "room_style": "modern|minimalist|...",
  "anchor_objects": ["desk"],
  "estimated_objects": [
    {"name": "desk", "quantity": 5, "notes": "primary workspace"}
  ],
  "room_dimensions_hint": "large (8x8m)"
}
```

**Ключевые правила:**
- Максимум 20 экземпляров объектов
- Максимум 6 типов объектов для полных сцен
- Объекты должны быть атомарными (не "набор стульев", а "стул" с quantity=4)
- Anchor objects - главные объекты, вокруг которых размещаются остальные

## Формат промптов

Все промпты хранятся в текстовых файлах `.txt` для удобства редактирования и версионирования.

## Использование

Промпты автоматически загружаются соответствующими модулями при импорте:

```python
from prompt_expander import expand_prompt

# Промпт загружается автоматически из prompts/expand_prompt.txt
scene_spec = expand_prompt(query="офис", prompt_model_fn=my_llm, llm_model="gpt-4")
```

## Редактирование промптов

При редактировании промптов:

1. Сохраняйте формат JSON в примерах
2. Не удаляйте обязательные поля
3. Тестируйте изменения с помощью `test_stage0.py`
4. Документируйте значительные изменения в этом README

## Следующие промпты

По мере добавления новых этапов pipeline, здесь будут появляться новые промпты:

- [ ] `disambiguation_prompt.txt` - Выбор конкретной модели из каталога (Stage 1)
- [ ] `semantic_plan_prompt.txt` - Генерация плана размещения (Stage 3)
- [ ] `collision_resolution_prompt.txt` - Разрешение коллизий (Stage 4)
- [ ] `vlm_validation_prompt.txt` - Валидация сцены (Stage 7)
