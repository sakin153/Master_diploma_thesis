# Table Count Fix Bugfix Design

## Overview

Этот bugfix решает проблему недетерминированного размещения объектов при запросах типа "4 стола и 4 стула вокруг каждого стола". Основная проблема заключается в том, что LLM не всегда генерирует полный список объектов в семантическом плане - например, вместо 4 столов (table_1, table_2, table_3, table_4) может вернуть только 3 (table_1, table_2, table_3).

**Стратегия исправления:** Преимущественно через улучшение промптов для LLM (80% решения), с минимальными улучшениями в fallback механизме `_ensure_plan_covers_chosen_models` (20% решения). Это обеспечит детерминированное размещение всех запрошенных объектов с правильными пространственными отношениями.

## Glossary

- **Bug_Condition (C)**: Условие, при котором LLM пропускает объекты из `chosen_models` в семантическом плане
- **Property (P)**: Желаемое поведение - LLM генерирует семантический план со всеми N экземплярами каждого типа объектов
- **Preservation**: Существующее поведение для случаев с малым количеством объектов (< 4) и корректных планов должно остаться неизменным
- **chosen_models**: Список моделей объектов, выбранных для размещения в сцене (входной параметр)
- **semantic_plan**: JSON-структура с объектами и их constraints, генерируемая LLM
- **_build_models_str**: Функция в `creator/placement/plan.py`, которая форматирует список объектов для промпта LLM
- **_ensure_plan_covers_chosen_models**: Fallback функция, которая добавляет недостающие объекты в план после генерации LLM
- **grid_threshold**: Порог количества объектов (4), при котором система переключается на grid arrangement
- **fmt_constraints_plan_tmpl**: Основной промпт-шаблон в `creator/contexts_prompts/constraints.py` для генерации семантического плана
- **fmt_seating_plan_tmpl**: Промпт-шаблон для распределения стульев вокруг уже размещенных столов

## Bug Details

### Bug Condition

Баг проявляется когда LLM получает запрос на размещение N экземпляров объекта (где N >= 4), но генерирует семантический план с меньшим количеством экземпляров. Функция `_build_models_str` корректно форматирует список объектов, но промпт не содержит достаточно явных инструкций о критичности включения всех экземпляров.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type {chosen_models: List[Dict], llm_plan: Dict}
  OUTPUT: boolean
  
  LET counts_requested = Counter({normalize_name(m["Model"]): count for m in chosen_models})
  LET counts_in_plan = Counter({normalize_name(obj["Model"]): count for obj in llm_plan["objects"]})
  
  RETURN EXISTS object_type IN counts_requested WHERE
         counts_in_plan[object_type] < counts_requested[object_type]
         AND counts_requested[object_type] >= 4
END FUNCTION
```

### Examples

**Пример 1: Пропуск стола**
- Запрос: "4 стола и 4 стула вокруг каждого стола"
- `chosen_models`: 4 × "table", 16 × "computer chair"
- LLM возвращает: table_1, table_2, table_3 (пропущен table_4)
- Ожидалось: table_1, table_2, table_3, table_4

**Пример 2: Неправильное распределение стульев**
- Запрос: "4 стола и 4 стула вокруг каждого стола"
- `chosen_models`: 4 × "table", 16 × "computer chair"
- LLM возвращает: все 4 стола, но стулья с constraints типа `region: middle` вместо `beside: table_N`
- Ожидалось: 4 стула с `beside: table_1`, 4 стула с `beside: table_2`, и т.д.

**Пример 3: Корректная работа с малым количеством**
- Запрос: "2 стола и 2 стула"
- `chosen_models`: 2 × "table", 2 × "chair"
- LLM корректно возвращает: table_1, table_2, chair_1, chair_2
- Ожидаемое поведение: должно продолжать работать так же

**Пример 4: Edge case - ровно на пороге**
- Запрос: "4 стула вокруг стола"
- `chosen_models`: 1 × "table", 4 × "chair"
- LLM может вернуть только 3 стула
- Ожидалось: все 4 стула с правильными constraints

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Beam search для объектов с количеством < 4 должен продолжать работать без изменений
- Grid arrangement для объектов с количеством >= 4 должен продолжать использоваться
- Функция `_normalize_name` должна продолжать корректно удалять суффиксы `_N` для группировки
- Существующие positional constraints (beside, face_to, on_top_of) должны обрабатываться так же
- Fallback механизм `_ensure_plan_covers_chosen_models` должен продолжать работать для случаев, когда LLM все же пропускает объекты

**Scope:**
Все входные данные, где LLM уже корректно генерирует полный список объектов, должны быть полностью не затронуты этим исправлением. Это включает:
- Запросы с малым количеством объектов (< 4 экземпляров)
- Запросы без явного указания количества
- Сцены с разнородными объектами без повторений
- Любые случаи, где семантический план уже содержит все объекты из `chosen_models`

## Hypothesized Root Cause

На основе анализа кода и описания бага, наиболее вероятные причины:

1. **Недостаточно явная инструкция в промпте**: Промпт `fmt_constraints_plan_tmpl` содержит правило "Output EXACTLY the objects listed above — no more, no less", но это недостаточно сильная инструкция для LLM при работе с большими списками объектов. LLM может интерпретировать это как "примерно столько же" вместо "ровно столько".

2. **Неявное указание количества в _build_models_str**: Функция `_build_models_str` форматирует список как "table x4 → table_1 ... table_4", но использование "..." может быть воспринято LLM как "и так далее, необязательно все". Нужно более явное перечисление для критических случаев.

3. **Отсутствие явного распределения стульев**: В промпте `fmt_seating_plan_tmpl` нет явной инструкции о том, сколько стульев должно быть привязано к каждому столу. LLM должен сам вывести это из контекста, что приводит к ошибкам.

4. **Недостаточная валидация ответа LLM**: После получения ответа от LLM в `_build_plan_single` нет проверки, что все объекты из `chosen_models` присутствуют в плане перед возвратом. Валидация происходит только в `_ensure_plan_covers_chosen_models`, но это fallback, а не основной механизм.

## Correctness Properties

Property 1: Bug Condition - Complete Object Inclusion

_For any_ input where `chosen_models` contains N instances of an object type (where N >= 4), the fixed LLM prompt SHALL cause the LLM to generate a semantic plan that includes all N instances with explicit numbering (object_1, object_2, ..., object_N), and the system SHALL deterministically place them in the scene.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - Small Object Count Behavior

_For any_ input where `chosen_models` contains fewer than 4 instances of each object type, the fixed code SHALL produce exactly the same semantic plan and placement behavior as the original code, preserving beam search logic and existing constraint handling.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

Предполагая, что наш анализ корневой причины верен:

**File**: `creator/contexts_prompts/constraints.py`

**Section**: `fmt_constraints_plan_tmpl`

**Specific Changes**:
1. **Усилить инструкцию о полноте списка**: Заменить "Output EXACTLY the objects listed above — no more, no less" на более явную инструкцию с примером:
   ```
   CRITICAL RULE: You MUST include ALL N instances of each object type shown above.
   - If the list shows "table x4 → table_1, table_2, table_3, table_4", your output MUST contain all 4 tables
   - If the list shows "chair x16 → chair_1 ... chair_16", your output MUST contain all 16 chairs
   - Missing even one instance is a critical error
   ```

2. **Добавить явное напоминание перед JSON примером**: Перед секцией "Output format" добавить:
   ```
   Before generating JSON, verify:
   ✓ Count of each object type in your output matches the count shown in "Objects (N total)" above
   ✓ All instances are numbered sequentially: object_1, object_2, ..., object_N
   ```

**File**: `creator/placement/plan.py`

**Function**: `_build_models_str`

**Specific Changes**:
3. **Явное перечисление для критических случаев**: Изменить логику форматирования, чтобы для объектов с count >= 4 выводить полный список вместо "...":
   ```python
   if count == 1:
       lines.append(f"  {name} x1  → {name}_1")
   elif count <= 3:
       # Small count: show all explicitly
       instances = ", ".join(f"{name}_{i}" for i in range(1, count + 1))
       lines.append(f"  {name} x{count}  → {instances}")
   else:
       # Large count: show first 2, last 2, and total
       lines.append(f"  {name} x{count}  → {name}_1, {name}_2, ..., {name}_{count-1}, {name}_{count}")
       lines.append(f"    (CRITICAL: ALL {count} instances must be included)")
   ```

**File**: `creator/contexts_prompts/constraints.py`

**Section**: `fmt_seating_plan_tmpl`

**Specific Changes**:
4. **Явное распределение стульев по столам**: Изменить промпт для явного указания распределения:
   ```
   Distribution requirement:
   - If there are N anchors and M children, distribute children evenly
   - Example: 4 tables and 16 chairs → 4 chairs per table
   - Explicitly assign: chair_1..chair_4 to table_1, chair_5..chair_8 to table_2, etc.
   ```

5. **Добавить пример с явным распределением**: Расширить JSON пример в `fmt_seating_plan_tmpl`:
   ```json
   // Example: 2 tables, 8 chairs (4 per table)
   {
     "objects": [
       // Chairs for table_1
       {"Model": "chair", "constraints": [{"type": "beside", "target": "table_1", "side": "front", ...}]},
       {"Model": "chair", "constraints": [{"type": "beside", "target": "table_1", "side": "back", ...}]},
       {"Model": "chair", "constraints": [{"type": "beside", "target": "table_1", "side": "left", ...}]},
       {"Model": "chair", "constraints": [{"type": "beside", "target": "table_1", "side": "right", ...}]},
       // Chairs for table_2
       {"Model": "chair", "constraints": [{"type": "beside", "target": "table_2", "side": "front", ...}]},
       ...
     ]
   }
   ```

**File**: `creator/placement/plan.py`

**Function**: `_build_plan_single`

**Specific Changes** (минимальные, как fallback):
6. **Добавить пост-валидацию перед возвратом**: После вызова `_ensure_plan_covers_chosen_models` добавить проверку:
   ```python
   # Validate that all chosen_models are present
   from collections import Counter
   expected_counts = Counter(_normalize_name(_safe_str(m.get("Model") or m.get("name"))) 
                             for m in chosen_models)
   actual_counts = Counter(_normalize_name(_safe_str(obj.get("Model"))) 
                          for obj in objects)
   
   for obj_type, expected_count in expected_counts.items():
       if actual_counts.get(obj_type, 0) < expected_count:
           # Log warning but continue - _ensure_plan_covers_chosen_models should have fixed this
           pass
   ```

**File**: `creator/placement/plan.py`

**Function**: `_ensure_plan_covers_chosen_models`

**Specific Changes** (минимальные улучшения):
7. **Улучшить распределение недостающих стульев**: Текущая логика распределяет недостающие объекты по принципу "least-populated anchor". Добавить явное распределение для стульев:
   ```python
   # For chairs specifically, ensure even distribution across tables
   if "chair" in base.lower() or "stool" in base.lower():
       # Calculate target count per anchor
       target_per_anchor = len([m for m in chosen_models 
                               if _normalize_name(_safe_str(m.get("Model"))) == base]) // len(targets)
       # Distribute to reach target_per_anchor for each anchor
   ```

## Testing Strategy

### Validation Approach

Стратегия тестирования следует двухфазному подходу: сначала выявить counterexamples, демонстрирующие баг на неисправленном коде, затем проверить, что исправление работает корректно и сохраняет существующее поведение.

### Exploratory Bug Condition Checking

**Goal**: Выявить counterexamples, демонстрирующие баг ДО внедрения исправления. Подтвердить или опровергнуть анализ корневой причины. Если опровергнем, потребуется пересмотр гипотезы.

**Test Plan**: Написать тесты, которые вызывают `build_semantic_plan` с различными конфигурациями `chosen_models` (4+ объектов одного типа) и проверяют, что LLM возвращает полный список. Запустить эти тесты на НЕИСПРАВЛЕННОМ коде для наблюдения сбоев и понимания корневой причины.

**Test Cases**:
1. **4 Tables Test**: Запрос "4 стола в сетке 2x2" с `chosen_models` = 4 × "table" (будет падать на неисправленном коде - LLM может вернуть только 3 стола)
2. **16 Chairs Around 4 Tables Test**: Запрос "4 стола и 4 стула вокруг каждого" с `chosen_models` = 4 × "table", 16 × "chair" (будет падать - стулья могут быть размещены неправильно)
3. **Edge Case - Exactly 4 Objects**: Запрос "4 стула вокруг стола" с `chosen_models` = 1 × "table", 4 × "chair" (может падать на неисправленном коде)
4. **Large Count Test**: Запрос "8 столов" с `chosen_models` = 8 × "table" (может падать - LLM может пропустить несколько столов)

**Expected Counterexamples**:
- LLM возвращает (N-1) или (N-2) экземпляров вместо N для объектов с count >= 4
- Стулья получают constraints типа `region: middle` вместо `beside: table_N`
- Возможные причины: недостаточно явная инструкция в промпте, неявное указание количества в `_build_models_str`, отсутствие явного распределения

### Fix Checking

**Goal**: Проверить, что для всех входных данных, где выполняется условие бага, исправленная функция генерирует ожидаемое поведение.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  plan := build_semantic_plan_fixed(input.chosen_models, input.query)
  ASSERT all_objects_present(plan, input.chosen_models)
  ASSERT correct_constraints(plan, input.query)
END FOR
```

**Testing Approach**: Использовать property-based testing для генерации различных конфигураций `chosen_models` с count >= 4 и проверки, что все объекты присутствуют в плане.

### Preservation Checking

**Goal**: Проверить, что для всех входных данных, где условие бага НЕ выполняется, исправленная функция генерирует тот же результат, что и оригинальная.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT build_semantic_plan_original(input) = build_semantic_plan_fixed(input)
END FOR
```

**Testing Approach**: Property-based testing рекомендуется для preservation checking, потому что:
- Автоматически генерирует множество тестовых случаев по всему входному домену
- Выявляет edge cases, которые могут быть пропущены в ручных unit тестах
- Предоставляет сильные гарантии, что поведение не изменилось для всех не-багованных входных данных

**Test Plan**: Наблюдать поведение на НЕИСПРАВЛЕННОМ коде для запросов с малым количеством объектов (< 4), затем написать property-based тесты, фиксирующие это поведение.

**Test Cases**:
1. **Small Count Preservation**: Наблюдать, что запросы с 1-3 объектами работают корректно на неисправленном коде, затем написать тест для проверки, что это продолжает работать после исправления
2. **Beam Search Preservation**: Наблюдать, что объекты с positional constraints используют beam search на неисправленном коде, затем проверить, что это сохраняется
3. **Normalize Name Preservation**: Проверить, что функция `_normalize_name` продолжает корректно группировать экземпляры
4. **Existing Constraints Preservation**: Проверить, что существующие constraints (beside, face_to, on_top_of) обрабатываются так же

### Unit Tests

- Тест `_build_models_str` с различными конфигурациями (1, 2, 4, 8, 16 объектов) для проверки корректного форматирования
- Тест `_ensure_plan_covers_chosen_models` для проверки добавления недостающих объектов
- Тест валидации промпта для проверки наличия критических инструкций
- Тест распределения стульев вокруг столов для проверки корректных constraints

### Property-Based Tests

- Генерировать случайные конфигурации `chosen_models` с count >= 4 и проверять, что все объекты присутствуют в плане
- Генерировать случайные конфигурации с count < 4 и проверять, что поведение не изменилось (preservation)
- Генерировать случайные запросы с "N объектов вокруг M объектов" и проверять корректное распределение constraints
- Тестировать, что все не-клавиатурные входные данные продолжают работать так же

### Integration Tests

- Полный flow: запрос → выбор моделей → генерация плана → размещение в сцене для случая "4 стола и 4 стула вокруг каждого"
- Тест переключения между beam search и grid arrangement в зависимости от количества объектов
- Тест визуальной проверки: сгенерировать сцену и проверить, что все объекты размещены корректно
