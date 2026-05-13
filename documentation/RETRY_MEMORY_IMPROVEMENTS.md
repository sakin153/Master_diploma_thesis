# Улучшения Системы Памяти Между Ретраями

**Дата:** 2026-05-11  
**Цель:** Добавить обратную связь о причинах провалов в промпты повторов для уменьшения количества неудачных попыток

---

## Проблема

Как указано в [`АНАЛИЗ_ПРОИЗВОДИТЕЛЬНОСТИ_PIPELINE.md`](АНАЛИЗ_ПРОИЗВОДИТЕЛЬНОСТИ_PIPELINE.md:169):

> **Нет обучения между повторами:** Каждый повтор начинается с нуля, не учится на предыдущих провалах

Система делала 30+ вызовов LLM для простой сцены, причем 67% вызовов были ретраями. Основная причина - LLM повторял те же ошибки, потому что не получал информацию о **причинах** предыдущих провалов.

### Что было:
- [`_llm_json()`](project/scene_planner.py:206) имел обратную связь только о **структуре JSON** (массив vs объект)
- [`_place_node_batch()`](project/scene_planner.py:2584) имел обратную связь о нарушениях валидации, но она **не передавалась** в `_llm_json()`
- Каждый ретрай начинался "с чистого листа"

---

## Решение

Добавлена **трехуровневая система памяти** между ретраями:

### 1. Улучшен [`_llm_json()`](project/scene_planner.py:206-260)

**Добавлен параметр `validation_context`:**
```python
def _llm_json(llm_fn, prompt, query, expect, retries=LLM_RETRIES, 
              validation_context=None, **llm_kwargs):
```

**Структура `validation_context`:**
```python
{
    'previous_attempts': [
        {
            'attempt_num': 1,
            'coordinates': '(0.15,0.20), (0.30,0.20), ...',
            'violations': ['Item 1 overlaps with item 2', ...]
        },
        ...
    ],
    'forbidden_points': [(0.15, 0.20), (0.30, 0.20), ...],
    'context_summary': 'Pattern "row" placement has failed 3 time(s)...'
}
```

**Что делает:**
- Если `validation_context` предоставлен, добавляет его **в начало промпта** перед первой попыткой
- Показывает LLM:
  - Историю последних 3 попыток с координатами и нарушениями
  - Список запрещенных координат (до 10 точек)
  - Краткое резюме проблемы

### 2. Улучшен [`_place_node_batch()`](project/scene_planner.py:2584-3053)

**Добавлено отслеживание истории:**
```python
# Строка 2732
previous_attempts = []

for attempt in range(DOMAIN_RETRIES + 1):
    # ... размещение ...
    
    if violations:
        # Записываем провал
        coords_summary = ", ".join([f"({it['x']:.2f},{it['y']:.2f})" 
                                    for it in items[:3]])
        previous_attempts.append({
            'attempt_num': attempt + 1,
            'coordinates': coords_summary,
            'violations': violations[:5],
        })
```

**Передача контекста в `_llm_json()`:**
```python
# Строки 2768-2775, 2903-2910
validation_ctx = None
if previous_attempts:
    validation_ctx = {
        'previous_attempts': previous_attempts,
        'forbidden_points': [(it['x'], it['y']) for it in items],
        'context_summary': f"Batch placement for {node['model_name']} has failed..."
    }

raw = _llm_json(llm_fn, prompt, query, expect,
                validation_context=validation_ctx,  # <-- Передаем контекст
                temperature=..., seed=...)
```

### 3. Улучшен [`_llm_fix_batch_violations()`](project/scene_planner.py:2496-2581)

**Добавлен параметр `previous_attempts`:**
```python
def _llm_fix_batch_violations(node, items, violations, parent_obj, sw, sd,
                              gap_m, query, llm_fn, batch_count, 
                              previous_attempts=None):  # <-- Новый параметр
```

**Добавлена история в промпт:**
```python
# Строки 2541-2551
history_block = ""
if previous_attempts:
    history_lines = ["\n\nHISTORY OF PREVIOUS FAILED ATTEMPTS:"]
    for prev in previous_attempts[-5:]:
        history_lines.append(f"  Attempt #{prev['attempt_num']}:")
        history_lines.append(f"    Tried: {prev.get('coordinates', 'N/A')}")
        if prev.get('violations'):
            top_violations = prev['violations'][:2]
            history_lines.append(f"    Failed because: {'; '.join(top_violations)}")
    history_lines.append("\nLEARN FROM THESE MISTAKES - do not repeat the same errors!\n")
    history_block = "\n".join(history_lines)

prompt = PLACE_BATCH_FIX_PROMPT.format(...) + history_block
```

**Передача контекста в `_llm_json()`:**
```python
# Строки 2563-2572
validation_ctx = {
    'previous_attempts': previous_attempts,
    'forbidden_points': [(it['x'], it['y']) for it in items],
    'context_summary': (
        f"FIX-BATCH call after {len(previous_attempts)} failed attempts. "
        f"This is the LAST CHANCE to get it right."
    ),
}

raw = _llm_json(llm_fn, prompt, query, expect, 
                validation_context=validation_ctx)
```

---

## Как Это Работает

### Пример: Размещение 5 бананов в коробке

**Попытка 1 (T=0.7):**
- LLM размещает бананы
- Валидация: "Item 1 overlaps with item 2"
- Записывается в `previous_attempts`

**Попытка 2 (T=0.8):**
- `_llm_json()` получает `validation_context`:
  ```
  ============================================================
  CONTEXT FROM PREVIOUS DOMAIN VALIDATION FAILURES:
  Batch placement for banana has failed 1 time(s). 
  Avoid repeating the same coordinates.
  
  PREVIOUS FAILED ATTEMPTS:
    Attempt #1:
      Coordinates: (0.15,0.20), (0.30,0.20), (0.45,0.20)
      Violations: Item 1 overlaps with item 2; Item 2 overlaps with item 3
  
  FORBIDDEN COORDINATES (avoid these):
    - (0.150, 0.200)
    - (0.300, 0.200)
    - (0.450, 0.200)
  ============================================================
  ```
- LLM видит, что координаты слишком близко, и увеличивает расстояние
- Если снова провал - история растет

**Попытка 6 (FIX-BATCH):**
- `_llm_fix_batch_violations()` получает всю историю 5 попыток
- Промпт включает:
  - Детерминистические предложения координат (вычисленные кодом)
  - Историю всех 5 провалов с причинами
  - Контекст "это последний шанс"
- LLM имеет максимум информации для успеха

---

## Ключевые Особенности

### ✅ Память активируется ТОЛЬКО при ретраях
- Первая попытка всегда использует чистый промпт
- Контекст добавляется только если `previous_attempts` не пуст

### ✅ Ограничение размера контекста
- Последние 3-5 попыток (не все)
- Топ-5 нарушений на попытку
- До 10 запрещенных точек
- Предотвращает раздувание промпта

### ✅ Прогрессивное обогащение
- Попытка 1: чистый промпт
- Попытка 2: история 1 провала
- Попытка 3: история 2 провалов
- ...
- FIX-BATCH: полная история + детерминистические предложения

### ✅ Обратная совместимость
- `validation_context` - опциональный параметр
- Старый код без контекста продолжает работать
- Новый код постепенно добавляет контекст

---

## Ожидаемые Результаты

### Метрики До Изменений:
- **Общее время:** 147.6 секунд
- **Вызовы LLM:** 37
- **Повторы:** ~25 (67% вызовов)
- **Этап 4 (планирование):** 96 секунд (65%)

### Ожидаемые Метрики После:
- **Общее время:** ~70-90 секунд (улучшение на 40-50%)
- **Вызовы LLM:** ~15-20 (снижение на 45-60%)
- **Повторы:** ~5-8 (снижение на 70-80%)
- **Этап 4:** ~40-50 секунд (улучшение на 50%)

### Почему Ожидается Улучшение:

1. **LLM учится на ошибках** - не повторяет те же координаты
2. **Видит запрещенные точки** - избегает известных проблемных зон
3. **Понимает паттерн провалов** - например, "всегда перекрытие" → увеличивает расстояние
4. **FIX-BATCH более эффективен** - имеет полную историю для финального решения

---

## Файлы Изменены

- [`project/scene_planner.py`](project/scene_planner.py)
  - Функция `_llm_json()` (строки 206-260)
  - Функция `_place_node_batch()` (строки 2584-3053)
  - Функция `_llm_fix_batch_violations()` (строки 2496-2581)

---

## Тестирование

Для проверки улучшений:

```bash
# Запустить pipeline с той же сценой
python project/main_project.py "стол с двумя коробками, в одной 5 бананов, в другой 5 яблок"

# Сравнить метрики:
# - Количество вызовов LLM (должно уменьшиться)
# - Время этапа 4 (должно уменьшиться)
# - Количество "FIX-BATCH succeeded" сообщений (должно уменьшиться)
```

Ожидаемые логи:
```
[llm] Запрос к qwen3-coder-next:cloud (timeout=180s, T=0.7, seed=None)
[llm] Запрос к qwen3-coder-next:cloud (timeout=180s, T=0.8, seed=None)  # Меньше повторов
✓ Успех без FIX-BATCH
```

Вместо:
```
[llm] Запрос к qwen3-coder-next:cloud (timeout=180s, T=0.7, seed=None)
[llm] Запрос к qwen3-coder-next:cloud (timeout=180s, T=0.8, seed=None)
[llm] Запрос к qwen3-coder-next:cloud (timeout=180s, T=0.9, seed=None)
[llm] Запрос к qwen3-coder-next:cloud (timeout=180s, T=1.0, seed=None)
[llm] Запрос к qwen3-coder-next:cloud (timeout=180s, T=1.0, seed=None)
[scene_planner] FIX-BATCH succeeded for node bananas after DOMAIN_RETRIES exhausted.
```

---

## Дальнейшие Улучшения

Если результаты хорошие, можно добавить:

1. **Кеширование успешных паттернов** - запоминать удачные размещения для похожих объектов
2. **Адаптивные ограничения** - ослаблять валидацию на 10-15% после 2-3 провалов
3. **Эвристические fallback'и** - использовать детерминистические алгоритмы после N провалов
4. **Визуальная обратная связь** - генерировать ASCII-диаграммы размещений для LLM

---

## Заключение

Система памяти между ретраями решает ключевую проблему: **LLM теперь учится на своих ошибках** вместо слепого повтора попыток. Это должно значительно уменьшить количество ретраев и ускорить pipeline на 40-50%.

Изменения минимальны, обратно совместимы и следуют принципу "fail fast, learn faster".