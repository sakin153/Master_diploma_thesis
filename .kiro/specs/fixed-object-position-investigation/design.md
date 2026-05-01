# Fixed Object Position Investigation Bugfix Design

## Overview

Система размещения объектов Universal Placement System имеет критическую проблему: объекты, которые должны быть динамическими (яблоки, мелкие предметы, контейнеры), размещаются с флагом `is_static=true`, что приводит к их "зависанию" в воздухе вместо падения под действием гравитации. Это делает сгенерированные сцены физически некорректными.

Проблема проявляется в двух местах:
1. **LLM семантическое планирование**: LLM может неправильно установить `is_static=true` для динамических объектов
2. **Обработка в universal_system.py**: Система может игнорировать или неправильно применять значение `is_static` из семантического плана

Стратегия исправления:
- Добавить эвристику определения `is_static` на основе размера объекта (volume < 0.3 m³ → dynamic)
- Добавить определение типа объекта (контейнеры всегда dynamic)
- Применять эвристику в `_generate_semantic_plan()` после получения плана от LLM
- Сохранить приоритет явно указанного LLM значения `is_static`

## Glossary

- **Bug_Condition (C)**: Условие, при котором объект должен быть динамическим, но установлен как статический
- **Property (P)**: Желаемое поведение - динамические объекты имеют `is_static=false` и падают при симуляции
- **Preservation**: Существующая логика размещения статических объектов (мебель) должна остаться неизменной
- **is_static**: Флаг в метаданных объекта, определяющий, является ли объект статическим (true) или динамическим (false)
- **dynamics determination step**: Отдельный шаг LLM-запроса для определения динамичности объектов после генерации семантического плана
- **robot training environment designer**: Роль LLM при определении динамичности - LLM принимает решение, какие объекты должны быть динамическими для создания реалистичной среды обучения роботов
- **semantic_plan**: Семантический план, сгенерированный LLM, содержащий список объектов с их свойствами
- **universal_system.py**: Главный модуль системы размещения, интегрирующий все компоненты
- **_generate_semantic_plan()**: Метод в `UniversalPlacementSystem`, который генерирует семантический план с помощью LLM
- **_determine_object_dynamics()**: Новый метод для выполнения отдельного LLM-запроса, определяющего динамичность объектов

## Bug Details

### Bug Condition

Баг проявляется, когда объект должен быть динамическим (падать под действием гравитации), но система устанавливает для него `is_static=true`. Это происходит в методе `_generate_semantic_plan()` класса `UniversalPlacementSystem`, где:

1. LLM генерирует семантический план с объектами
2. Система обрабатывает объекты и добавляет поля `id`, `type`, `size`
3. Система использует эвристику (volume < 0.3 m³ или проверка ключевых слов) для определения `is_static`
4. **ПРОБЛЕМА**: Эвристика не учитывает контекст сцены и назначение объектов для обучения роботов
5. **ПРОБЛЕМА**: Отсутствует отдельный LLM-шаг, где LLM как "дизайнер среды" принимает осознанное решение о динамичности

В результате мелкие объекты и контейнеры могут получить `is_static=true` (по умолчанию или от эвристики), что приводит к их "зависанию" в воздухе.

**Formal Specification:**
```
FUNCTION isBugCondition(object)
  INPUT: object of type Dict[str, Any] from semantic_plan["objects"]
  OUTPUT: boolean
  
  size = object.get("size", [1.0, 1.0, 1.0])
  volume = size[0] * size[1] * size[2]
  model_name = object.get("Model", "").lower()
  
  # Object should be dynamic based on physical properties or type
  should_be_dynamic = (volume < 0.3) OR is_container_type(model_name)
  
  # But system set it as static (bug condition)
  is_static = object.get("is_static", True)
  
  RETURN should_be_dynamic AND is_static == True
END FUNCTION

FUNCTION is_container_type(model_name)
  INPUT: model_name of type string
  OUTPUT: boolean
  
  container_keywords = ["box", "basket", "container", "crate", "bin", "bowl", "cup"]
  RETURN any(keyword in model_name for keyword in container_keywords)
END FUNCTION
```

### Examples

**Пример 1: Яблоко (мелкий объект)**
- Model: "apple2"
- size: [0.08, 0.08, 0.08]
- volume: 0.000512 m³ (< 0.3 m³)
- **Текущее поведение**: `is_static=true` → яблоко зависает в воздухе
- **Ожидаемое поведение**: `is_static=false` → яблоко падает в коробку

**Пример 2: Картонная коробка (контейнер)**
- Model: "cardboard_box"
- size: [0.4, 0.3, 0.4]
- volume: 0.048 m³ (< 0.3 m³)
- is_container: True (содержит "box")
- **Текущее поведение**: `is_static=true` → коробка зависает в воздухе
- **Ожидаемое поведение**: `is_static=false` → коробка падает на стол

**Пример 3: Корзина с яблоками**
- Model: "basket"
- size: [0.3, 0.2, 0.3]
- volume: 0.018 m³ (< 0.3 m³)
- is_container: True (содержит "basket")
- **Текущее поведение**: `is_static=true` → корзина зависает
- **Ожидаемое поведение**: `is_static=false` → корзина падает

**Пример 4: Стол (крупная мебель) - должен остаться статическим**
- Model: "wooden_patterned_table"
- size: [1.5, 0.75, 0.8]
- volume: 0.9 m³ (≥ 0.3 m³)
- is_container: False
- **Текущее поведение**: `is_static=true` → стол остается на месте
- **Ожидаемое поведение**: `is_static=true` → стол остается на месте (без изменений)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Статические объекты (мебель, крупные предметы с volume ≥ 0.3 m³) должны продолжать иметь `is_static=true`
- Размещение объектов на полу без вложенности должно работать как раньше
- Все 8 этапов пайплайна генерации должны выполняться в том же порядке
- LLM семантическое планирование через `build_semantic_plan()` должно работать без изменений
- Экспорт сцены в MuJoCo XML должен включать все свойства объектов

**Scope:**
Все объекты, которые НЕ являются мелкими (volume ≥ 0.3 m³) и НЕ являются контейнерами, должны быть полностью не затронуты этим исправлением. Это включает:
- Мебель (столы, стулья, шкафы)
- Крупные предметы (большие ящики, тумбы)
- Архитектурные элементы (стены, двери, окна)

## Hypothesized Root Cause

На основе анализа кода в `creator/placement/universal_system.py`, метод `_generate_semantic_plan()` (строки 380-500), корневая причина:

**Отсутствие отдельного LLM-шага для определения динамичности объектов**

Текущая система использует эвристику (volume < 0.3 m³, проверка ключевых слов "box", "basket" и т.д.) для определения `is_static`. Это приводит к проблемам:

1. **Эвристика не учитывает контекст**: Объем 0.3 m³ - это произвольный порог, который не учитывает назначение объекта в сцене. Например, большая коробка (volume > 0.3 m³) может быть динамической в контексте обучения робота манипуляции.

2. **Проверка ключевых слов недостаточна**: Список ["box", "basket", "container", "crate", "bin", "bowl", "cup"] не покрывает все возможные контейнеры и динамические объекты.

3. **Отсутствие роли "дизайнер среды"**: LLM генерирует семантический план, но не принимает осознанное решение о том, какие объекты должны быть динамическими для создания реалистичной среды обучения роботов.

4. **Конфликт между LLM и эвристикой**: Если LLM установил `is_static=true`, эвристика не применяется (условие `if "is_static" not in obj`). Если LLM не указал `is_static`, эвристика может дать неправильный результат.

**Правильное решение**: Добавить отдельный шаг LLM-запроса после генерации семантического плана, где LLM выступает как "дизайнер сред для обучения роботов" и принимает решение о динамичности каждого объекта на основе:
- Исходного запроса пользователя (prompt)
- Сгенерированной сцены (список объектов)
- Физических свойств объектов (размер, тип)
- Назначения сцены (обучение робота)

## Correctness Properties

Property 1: Bug Condition - Dynamic Objects Have is_static=false

_For any_ object in semantic_plan["objects"] where the bug condition holds (volume < 0.3 m³ OR object is a container), the fixed _generate_semantic_plan() function SHALL set `is_static=false` for that object, ensuring it will fall under gravity during simulation.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Preservation - Static Objects Remain Static

_For any_ object in semantic_plan["objects"] where the bug condition does NOT hold (volume ≥ 0.3 m³ AND object is NOT a container), the fixed _generate_semantic_plan() function SHALL produce the same `is_static` value as the original function, preserving static behavior for furniture and large objects.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

## Fix Implementation

### Changes Required

Вместо использования эвристики, добавить отдельный LLM-шаг для определения динамичности объектов:

**File**: `creator/placement/universal_system.py`

**New Method**: `_determine_object_dynamics()` (добавить после `_generate_semantic_plan()`)

**Specific Changes**:

1. **Создать новый метод `_determine_object_dynamics()`**: Этот метод выполняет отдельный LLM-запрос для определения динамичности объектов.

2. **Системный промпт для LLM**: LLM выступает как "дизайнер сред для обучения роботов" и принимает решение о динамичности на основе:
   - Исходного запроса пользователя (prompt)
   - Сгенерированной сцены (список объектов с их размерами и типами)
   - Физических свойств объектов
   - Назначения сцены (обучение робота манипуляции, навигации и т.д.)

3. **Интеграция в `_generate_semantic_plan()`**: После получения семантического плана от LLM, вызвать `_determine_object_dynamics()` для определения `is_static` для каждого объекта.

4. **Формат LLM-запроса**: LLM получает структурированный запрос с информацией о сцене и возвращает JSON с решениями о динамичности.

5. **Fallback механизм**: Если LLM-запрос не удался, использовать эвристику (volume < 0.3 m³) как fallback.

### Pseudocode for New Method

```python
def _determine_object_dynamics(
    self,
    semantic_plan: Dict[str, Any],
    user_prompt: str,
    prompt_model: Optional[Any],
) -> Dict[str, Any]:
    """Determine object dynamics using LLM as robot training environment designer.
    
    Args:
        semantic_plan: Semantic plan with objects
        user_prompt: Original user prompt describing the scene
        prompt_model: LLM model for dynamics determination
        
    Returns:
        Updated semantic plan with is_static flags set by LLM
    """
    if prompt_model is None:
        # Fallback: use heuristic
        return self._apply_heuristic_dynamics(semantic_plan)
    
    # Build context for LLM
    objects_info = []
    for obj in semantic_plan.get("objects", []):
        model_name = obj.get("Model", "")
        size = obj.get("size", [1.0, 1.0, 1.0])
        volume = size[0] * size[1] * size[2]
        
        objects_info.append({
            "name": model_name,
            "size": size,
            "volume": volume,
            "current_is_static": obj.get("is_static", None),
        })
    
    # System prompt: LLM as robot training environment designer
    system_prompt = """You are a robot training environment designer. Your task is to determine 
which objects in a scene should be dynamic (can move/fall) vs static (fixed in place).

Consider:
- Small objects (fruits, tools, small containers) should typically be dynamic for manipulation training
- Large furniture (tables, chairs, cabinets) should typically be static
- Containers (boxes, baskets, bowls) should be dynamic if they are meant to be manipulated
- The scene purpose: training robots for manipulation, grasping, placing, etc.

Return a JSON object with format:
{
  "objects": [
    {"name": "object_name", "is_static": true/false, "reason": "explanation"},
    ...
  ]
}
"""
    
    # User prompt with scene context
    user_context = f"""Original scene request: "{user_prompt}"

Objects in scene:
{json.dumps(objects_info, indent=2)}

For each object, determine if it should be static (fixed) or dynamic (can move/fall).
Consider the scene purpose and realistic robot training scenarios."""
    
    try:
        # Call LLM
        response = prompt_model(system_prompt + "\n\n" + user_context, user_prompt)
        
        if isinstance(response, dict) and "objects" in response:
            # Apply LLM decisions to semantic plan
            name_to_decision = {
                obj["name"]: obj["is_static"] 
                for obj in response["objects"]
            }
            
            for obj in semantic_plan.get("objects", []):
                model_name = obj.get("Model", "")
                if model_name in name_to_decision:
                    obj["is_static"] = name_to_decision[model_name]
                    print(f"[universal_system] LLM decision: {model_name} is_static={obj['is_static']}")
        else:
            print("[universal_system] LLM response invalid, using fallback heuristic")
            return self._apply_heuristic_dynamics(semantic_plan)
            
    except Exception as e:
        print(f"[universal_system] LLM dynamics determination failed: {e}, using fallback")
        return self._apply_heuristic_dynamics(semantic_plan)
    
    return semantic_plan


def _apply_heuristic_dynamics(
    self,
    semantic_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Fallback: apply heuristic-based dynamics determination.
    
    Args:
        semantic_plan: Semantic plan with objects
        
    Returns:
        Updated semantic plan with is_static flags set by heuristic
    """
    for obj in semantic_plan.get("objects", []):
        model_name = obj.get("Model", "")
        size = obj.get("size", [1.0, 1.0, 1.0])
        volume = size[0] * size[1] * size[2]
        
        # Check if object is a container
        is_container = any(keyword in model_name.lower() 
                           for keyword in ["box", "basket", "container", "crate", 
                                          "bin", "bowl", "cup"])
        
        # Apply heuristic
        if is_container:
            obj["is_static"] = False
            print(f"[universal_system] Heuristic: {model_name} is dynamic (container)")
        elif volume < 0.3:
            obj["is_static"] = False
            print(f"[universal_system] Heuristic: {model_name} is dynamic (volume={volume:.4f} m³)")
        elif "is_static" not in obj:
            obj["is_static"] = True
            print(f"[universal_system] Heuristic: {model_name} is static (volume={volume:.4f} m³)")
    
    return semantic_plan
```

### Integration in _generate_semantic_plan()

```python
def _generate_semantic_plan(
    self,
    description: str,
    chosen_models: List[Dict[str, Any]],
    spatial_command: Any,
    prompt_model: Optional[Any],
    prompt_template: Optional[str],
) -> Dict[str, Any]:
    """Generate semantic plan using LLM."""
    
    # ... existing code to generate semantic_plan ...
    
    # NEW: Determine object dynamics using LLM
    semantic_plan = self._determine_object_dynamics(
        semantic_plan=semantic_plan,
        user_prompt=description,
        prompt_model=prompt_model,
    )
    
    # ... rest of existing code ...
    
    return semantic_plan
```

## Testing Strategy

### Validation Approach

Стратегия тестирования следует двухфазному подходу: сначала продемонстрировать баг на нефиксированном коде, затем проверить, что исправление работает корректно и сохраняет существующее поведение.

### Exploratory Bug Condition Checking

**Goal**: Продемонстрировать баг ДО внесения исправления. Подтвердить или опровергнуть анализ корневой причины. Если опровергнем, нужно будет пересмотреть гипотезу.

**Test Plan**: Написать тесты, которые создают семантический план с мелкими объектами и контейнерами, затем проверяют значение `is_static` в результате. Запустить эти тесты на НЕФИКСИРОВАННОМ коде, чтобы увидеть ошибки и понять корневую причину.

**Test Cases**:
1. **Small Object Test (Apple)**: Создать объект с size=[0.08, 0.08, 0.08] (volume=0.000512 m³), проверить что `is_static=false` (будет fail на нефиксированном коде, если эвристика или LLM установил true)
2. **Container Test (Box)**: Создать объект с Model="cardboard_box", проверить что `is_static=false` (будет fail на нефиксированном коде)
3. **Large Object Test (Table)**: Создать объект с size=[1.5, 0.75, 0.8] (volume=0.9 m³), проверить что `is_static=true` (должен pass даже на нефиксированном коде)
4. **Nested Objects Test**: Создать сцену "стол и две коробки на нем в каждой коробке по 3 яблока", проверить что яблоки и коробки имеют `is_static=false` (будет fail на нефиксированном коде)

**Expected Counterexamples**:
- Мелкие объекты (яблоки) имеют `is_static=true` вместо `false` из-за отсутствия LLM-шага определения динамичности
- Контейнеры (коробки) имеют `is_static=true` вместо `false` из-за того, что эвристика не применяется или LLM не принял правильное решение
- Возможные причины: отсутствие отдельного LLM-запроса с ролью "дизайнер среды", использование только эвристики

### Fix Checking

**Goal**: Проверить, что для всех входных данных, где выполняется условие бага, исправленная функция производит ожидаемое поведение.

**Pseudocode:**
```
FOR ALL object IN semantic_plan["objects"] WHERE isBugCondition(object) DO
  result := _generate_semantic_plan_fixed(...)
  processed_object := find_object_in_result(result, object.id)
  ASSERT processed_object["is_static"] == False
END FOR
```

**Test Plan**: Создать набор тестовых объектов с различными размерами и типами, запустить исправленную функцию, проверить что все мелкие объекты и контейнеры получили `is_static=false`.

**Test Cases**:
1. **Small Objects**: Объекты с volume < 0.3 m³ должны иметь `is_static=false`
2. **Containers**: Объекты с ключевыми словами (box, basket, etc.) должны иметь `is_static=false`
3. **Nested Scenes**: Сложные сцены с вложенными объектами должны корректно обрабатываться
4. **Edge Cases**: Объекты с volume ≈ 0.3 m³ (граничные случаи)

### Preservation Checking

**Goal**: Проверить, что для всех входных данных, где условие бага НЕ выполняется, исправленная функция производит тот же результат, что и оригинальная функция.

**Pseudocode:**
```
FOR ALL object IN semantic_plan["objects"] WHERE NOT isBugCondition(object) DO
  result_original := _generate_semantic_plan_original(...)
  result_fixed := _generate_semantic_plan_fixed(...)
  obj_original := find_object_in_result(result_original, object.id)
  obj_fixed := find_object_in_result(result_fixed, object.id)
  ASSERT obj_original["is_static"] == obj_fixed["is_static"]
END FOR
```

**Testing Approach**: Property-based testing рекомендуется для preservation checking, потому что:
- Автоматически генерирует множество тестовых случаев по всему входному домену
- Находит граничные случаи, которые могут быть пропущены в ручных unit-тестах
- Предоставляет сильные гарантии, что поведение не изменилось для всех не-багованных входных данных

**Test Plan**: Наблюдать поведение на НЕФИКСИРОВАННОМ коде для крупных объектов и мебели, затем написать property-based тесты, фиксирующие это поведение.

**Test Cases**:
1. **Large Furniture Preservation**: Наблюдать, что столы, стулья, шкафы имеют `is_static=true` на нефиксированном коде, затем написать тест для проверки этого после исправления
2. **Floor Placement Preservation**: Наблюдать, что размещение объектов на полу работает корректно на нефиксированном коде, затем написать тест для проверки этого после исправления
3. **Pipeline Preservation**: Наблюдать, что все 8 этапов пайплайна выполняются на нефиксированном коде, затем написать тест для проверки этого после исправления
4. **Export Preservation**: Наблюдать, что экспорт в MuJoCo XML работает корректно на нефиксированном коде, затем написать тест для проверки этого после исправления

### Unit Tests

- Тест для метода `_determine_object_dynamics()` с различными типами объектов (мелкие, крупные, контейнеры)
- Тест для метода `_determine_object_dynamics()` с fallback на эвристику (когда LLM недоступен)
- Тест для проверки, что LLM получает правильный контекст (user prompt, список объектов, размеры)
- Тест для проверки, что LLM-решения корректно применяются к semantic_plan
- Тест для проверки логирования (print-сообщения выводятся корректно)
- Тест для граничных случаев (объекты с неполными данными, отсутствие размеров)

### Property-Based Tests

- Генерировать случайные объекты с различными размерами и проверять, что LLM принимает разумные решения о динамичности
- Генерировать случайные имена объектов и проверять, что fallback эвристика работает корректно
- Генерировать случайные семантические планы и проверять, что крупные объекты сохраняют свое поведение
- Тестировать, что все не-мелкие объекты продолжают работать одинаково до и после исправления (preservation)

### Integration Tests

- Полный тест генерации сцены с промптом "стол и две коробки на нем в каждой коробке по 3 яблока"
- Проверка, что яблоки и коробки имеют `is_static=false` в финальном XML
- Проверка, что стол имеет `is_static=true` в финальном XML
- Визуальная проверка: запустить симуляцию в MuJoCo и убедиться, что яблоки падают в коробки
- Проверка, что не возникает коллизий и объекты размещаются корректно
- Тест с различными промптами для проверки, что LLM корректно определяет динамичность в разных контекстах
