# Текущий статус исправления

## Проблема 1: Mesh Assets Deduplication ✅ ИСПРАВЛЕНО
Ошибка: `Error: mesh 'mesh_chair_2' not found`

**Решение:** Убрана дедупликация в `scene_assembly.py`

## Проблема 2: No Decoder for .glb Files ✅ ИСПРАВЛЕНО
Ошибка: `Error: no decoder found for mesh file '/path/to/model.glb'`

**Решение:** Восстановлен Stage 5 для конвертации через `obj2mjcf`

## Проблема 3: IsADirectoryError ✅ ИСПРАВЛЕНО
Ошибка: `IsADirectoryError: [Errno 21] Is a directory: '/tmp/ciare_fresh_1777597884'`

**Причина:** `add_models()` получал `path_to_save=cache.cache_path` (директория), но ожидал путь к файлу.

**Решение:** Изменен параметр на `path_to_save=world_path` (файл)

```python
saved_models = interface.add_models(
    chosen_models=chosen_models,
    models=models_full,
    query=effective_query,
    path_to_save=world_path,  # ← FIX: файл вместо директории
    world_path=world_path,
    room_half_size=room_half_size,
    pre_placed_models=full_placed_models,
    semantic_plan=semantic_plan,
)
```

## Статус

✅ **ВСЕ ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ**

Готово к тестированию:
```bash
python main.py "стол"
python view_scene.py
```
