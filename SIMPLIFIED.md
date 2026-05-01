# ✅ УПРОЩЕННАЯ СИСТЕМА

## Что Изменилось

### ❌ УДАЛЕНО:
- Fallback на UniversalPlacementSystem
- Дублирование кода
- Сложная логика с try-except
- Лишние проверки

### ✅ ОСТАЛОСЬ:
- **ТОЛЬКО** SemanticEnforcementPipeline
- Простой линейный поток
- Четкие ошибки если что-то не работает

---

## Новый Алгоритм (УПРОЩЕННЫЙ)

```
User Query
    ↓
LLM генерирует Semantic Plan
    ↓
SemanticEnforcementPipeline:
  1. SchemaValidator
  2. DistanceResolver  
  3. OrientationResolver
  4. PlacementExecutor
  5. SceneAssembly
    ↓
MuJoCo XML
    ↓
Done!
```

**Если ошибка** - система упадет с четким сообщением ЧТО не работает.

---

## Почему Плохо Работала Расстановка?

### Проблема:
Система использовала **UniversalPlacementSystem** (старый код), а не новый **SemanticEnforcementPipeline**.

### Причина:
LLM не генерировал semantic plan в новом формате → fallback на старую систему.

### Решение:
**УБРАЛ FALLBACK**. Теперь система будет падать с ошибкой, если LLM не генерирует правильный план. Это позволит увидеть РЕАЛЬНУЮ проблему.

---

## Запуск

```bash
python main.py "стол и стул"
```

### Если Работает:
```
[pipeline] Stage 4: Semantic Enforcement Pipeline
[pipeline] Generating semantic plan...
[SemanticPlan] Generating semantic plan (attempt 1/3)
[SemanticPlan] Schema validation passed on attempt 1
[pipeline] ✓ Generated semantic plan with 2 objects
[pipeline] Executing Semantic Enforcement Pipeline...
[Pipeline] Starting semantic enforcement pipeline
[Pipeline] Stage 1: Schema Validation
[Pipeline] Stage 2: Distance Resolution
[Pipeline] Stage 3: Orientation Resolution
[Pipeline] Stage 4: Placement Execution
[Pipeline] Stage 5: Scene Assembly
[pipeline] ✓ Placed 2 objects
[pipeline] ✓ Saved MuJoCo XML
```

### Если НЕ Работает:
Система упадет с **ЧЕТКОЙ ОШИБКОЙ**:
- `RuntimeError: Failed to parse LLM response as JSON` → LLM не генерирует JSON
- `SchemaValidationError: Schema validation failed` → JSON не соответствует схеме
- Другая ошибка → смотрите traceback

---

## Что Дальше?

1. **Запустите**: `python main.py "стол и стул"`
2. **Посмотрите логи** - теперь будет видно ЧТО именно не работает
3. **Если ошибка** - покажите мне traceback, исправлю

---

## Преимущества Упрощения

✅ **Проще понять** - один путь вместо двух
✅ **Проще отладить** - четкие ошибки
✅ **Проще поддерживать** - меньше кода
✅ **Быстрее работает** - нет лишних проверок

---

## Файлы

- `creator/runner.py` - упрощен Stage 4 и Stage 5
- Удалено ~200 строк fallback кода
- Осталось только то, что нужно

---

🚀 **ЗАПУСКАЙТЕ И СМОТРИТЕ ЛОГИ!**
