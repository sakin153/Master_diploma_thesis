# Диагностика Проблемы с Количеством

## Проблема
Запрос "стол и две коробки на нем, в каждой коробке по 3 яблока" генерирует только:
- 1 стол ✓
- 1 коробка (ожидается 2) ✗
- 1 яблоко (ожидается 6) ✗

## Диагностика

### Шаг 1: Проверить expand_prompt
```bash
python test_specific_query.py
```

Это покажет, что извлекает LLM на этапе расширения запроса.

**Ожидается:**
```
Objects extracted:
  - table x1
  - cardboard_box x2
  - apple x6
Total: 9 objects
```

### Шаг 2: Проверить лог pipeline
Запустите:
```bash
python main.py "стол и две коробки на нем, в каждой коробке по 3 яблока" 2>&1 | grep -A 20 "Objects:"
```

Найдите строку:
```
[pipeline] Objects: ['table', 'cardboard_box', 'cardboard_box', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple']
```

Если список короче, проблема в `expand_prompt`.

### Шаг 3: Проверить chosen_models
Найдите в логе:
```
[Stage1] 'table' → table (uuid=...)
[Stage1] 'cardboard_box' → cardboard_box (uuid=...)
[Stage1] 'cardboard_box' → cardboard_box (uuid=...)  ← Должно быть 2 раза
[Stage1] 'apple' → apple (uuid=...)
[Stage1] 'apple' → apple (uuid=...)  ← Должно быть 6 раз
...
```

Если строк меньше, проблема в выборе моделей.

### Шаг 4: Проверить semantic plan
Найдите в логе:
```
[pipeline] ✓ Generated semantic plan with X objects
```

Если X < 9, проблема в генерации semantic plan.

## Возможные Причины

### Причина 1: expand_prompt неправильно парсит количество
**Симптом**: `test_specific_query.py` показывает меньше объектов, чем ожидается

**Решение**: Проблема в промпте `fmt_expand_system` в `creator/contexts_prompts/expand.py`

### Причина 2: Дубликаты фильтруются
**Симптом**: `[pipeline] Objects:` показывает правильное количество, но `chosen_models` короче

**Решение**: Проблема в логике выбора моделей (строка ~490 в runner.py)

### Причина 3: LLM генерирует меньше объектов в semantic plan
**Симптом**: `expand_prompt` правильный, но semantic plan содержит меньше объектов

**Решение**: Проблема в промпте `fmt_semantic_plan_tmpl` - LLM не генерирует все объекты

### Причина 4: Модели не найдены в каталоге
**Симптом**: Видны предупреждения `WARN: 'apple' has no catalog match, dropped`

**Решение**: Модели отсутствуют в базе данных

## Быстрая Проверка

Запустите:
```bash
python main.py "стол и две коробки на нем, в каждой коробке по 3 яблока" 2>&1 | tee debug.log
```

Затем проверьте `debug.log`:
1. Найдите `[pipeline] Objects:` - сколько объектов?
2. Посчитайте строки `[Stage1]` - сколько моделей выбрано?
3. Найдите `Generated semantic plan with X objects` - сколько в плане?
4. Найдите `✓ Placed X objects` - сколько размещено?

Сравните числа на каждом этапе, чтобы найти, где теряются объекты.
