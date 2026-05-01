# БЛОКАТОР НАЙДЕН: Mesh Assets Deduplication

## Проблема

При запуске `python main.py "стол и 3 стула"` возникает ошибка:

```
Error: mesh 'mesh_chair_2' not found in geom 7
```

## Корневая причина

**Файл:** `creator/placement/semantic_enforcement/scene_assembly.py`  
**Метод:** `_add_mesh_assets()`  
**Строки:** 148-150

### Проблемный код:

```python
def _add_mesh_assets(self, asset_elem: ET.Element, objects: List[PlacedObject]) -> None:
    """Add mesh asset definitions for all unique models."""
    
    # Track unique model files to avoid duplicate mesh assets
    seen_model_files = {}
    
    for obj in objects:
        # Skip if we've already added a mesh asset for this model file
        if obj.model_loc in seen_model_files:  # ← БЛОКАТОР!
            continue  # ← Пропускает chair_2 и chair_3!
        
        # Add mesh asset
        ET.SubElement(asset_elem, "mesh", name=f"mesh_{obj.id}", file=mesh_file)
        seen_model_files[obj.model_loc] = mesh_file
```

### Что происходит:

1. **Итерация 1:** `chair_1` (model_loc="/path/to/chair.glb")
   - `seen_model_files` пуст
   - Создается `<mesh name="mesh_chair_1" file="/path/to/chair.glb"/>`
   - `seen_model_files["/path/to/chair.glb"] = "/path/to/chair.glb"`

2. **Итерация 2:** `chair_2` (model_loc="/path/to/chair.glb")
   - `obj.model_loc in seen_model_files` → **True**
   - `continue` → **ПРОПУСКАЕТ создание mesh_chair_2!**

3. **Итерация 3:** `chair_3` (model_loc="/path/to/chair.glb")
   - `obj.model_loc in seen_model_files` → **True**
   - `continue` → **ПРОПУСКАЕТ создание mesh_chair_3!**

### Результат в XML:

```xml
<asset>
  <mesh name="mesh_table_1" file="/path/to/table.glb"/>
  <mesh name="mesh_chair_1" file="/path/to/chair.glb"/>
  <!-- mesh_chair_2 ОТСУТСТВУЕТ! -->
  <!-- mesh_chair_3 ОТСУТСТВУЕТ! -->
</asset>

<worldbody>
  <body name="table_1">
    <geom type="mesh" mesh="mesh_table_1"/>  <!-- OK -->
  </body>
  <body name="chair_1">
    <geom type="mesh" mesh="mesh_chair_1"/>  <!-- OK -->
  </body>
  <body name="chair_2">
    <geom type="mesh" mesh="mesh_chair_2"/>  <!-- ERROR: mesh not found! -->
  </body>
  <body name="chair_3">
    <geom type="mesh" mesh="mesh_chair_3"/>  <!-- ERROR: mesh not found! -->
  </body>
</worldbody>
```

## Почему это блокатор

Код был написан с предположением, что **дедупликация mesh assets экономит память**. Это правильно для оптимизации, НО:

1. **MuJoCo требует уникальные имена mesh для каждого geom**
2. Каждый объект создает geom с именем `mesh_{obj.id}`
3. Если mesh asset с таким именем не существует → **ошибка рендеринга**

## Решение

Убрать дедупликацию и создавать mesh asset для **КАЖДОГО объекта**, даже если они используют один и тот же файл модели.

### Исправленный код:

```python
def _add_mesh_assets(self, asset_elem: ET.Element, objects: List[PlacedObject]) -> None:
    """Add mesh asset definitions for ALL objects.
    
    Each object gets its own mesh asset with unique name (mesh_{obj.id}),
    even if multiple objects share the same model file. This allows
    multiple instances of the same model in the scene.
    """
    for obj in objects:
        # Verify model file exists
        model_path = self.workspace_root / obj.model_loc
        if not model_path.exists():
            logger.warning(
                f"[SceneAssembly] Model file not found: {obj.model_loc}. "
                f"Skipping mesh asset for {obj.id}"
            )
            continue

        # Add mesh asset with unique name based on object ID
        mesh_file = str(model_path.resolve())
        ET.SubElement(
            asset_elem, "mesh", name=f"mesh_{obj.id}", file=mesh_file
        )
        
        logger.debug(f"[SceneAssembly] Added mesh asset: mesh_{obj.id} -> {mesh_file}")
```

### Результат после исправления:

```xml
<asset>
  <mesh name="mesh_table_1" file="/path/to/table.glb"/>
  <mesh name="mesh_chair_1" file="/path/to/chair.glb"/>
  <mesh name="mesh_chair_2" file="/path/to/chair.glb"/>  <!-- ✓ Создан! -->
  <mesh name="mesh_chair_3" file="/path/to/chair.glb"/>  <!-- ✓ Создан! -->
</asset>

<worldbody>
  <body name="table_1">
    <geom type="mesh" mesh="mesh_table_1"/>  <!-- ✓ OK -->
  </body>
  <body name="chair_1">
    <geom type="mesh" mesh="mesh_chair_1"/>  <!-- ✓ OK -->
  </body>
  <body name="chair_2">
    <geom type="mesh" mesh="mesh_chair_2"/>  <!-- ✓ OK -->
  </body>
  <body name="chair_3">
    <geom type="mesh" mesh="mesh_chair_3"/>  <!-- ✓ OK -->
  </body>
</worldbody>
```

## Примечание об оптимизации

Да, это создает дублирующиеся mesh assets в XML. Но:

1. **MuJoCo сам оптимизирует загрузку** - файл загружается один раз, даже если на него ссылаются несколько mesh assets
2. **Размер XML файла увеличивается незначительно** - только добавляются строки `<mesh name="..." file="..."/>`
3. **Это единственный способ** поддержать множественные экземпляры одной модели

## Статус

✅ **ИСПРАВЛЕНО** в `creator/placement/semantic_enforcement/scene_assembly.py`

Убрана логика дедупликации (`seen_model_files`), теперь каждый объект получает свой mesh asset.
