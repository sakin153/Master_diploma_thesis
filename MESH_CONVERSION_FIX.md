# Mesh Conversion Fix - Complete Solution

## Problem Evolution

### Issue 1: XML Parsing Error (FIXED)
**Error**: `XML Error: Include error: 'XML parse error 8:Error=XML_ERROR_PARSING_TEXT ErrorID=8 (0x8) Line number=1'`

**Cause**: SceneAssembly was using `<include file="model.glb"/>` which is incorrect.

**Fix**: Changed to use `<mesh>` assets in `<asset>` section and `<geom type="mesh">` references.

### Issue 2: No Decoder for .glb Files (CURRENT)
**Error**: `Error: no decoder found for mesh file '/path/to/model.glb'`

**Cause**: MuJoCo cannot directly load .glb files as meshes. They must be converted to MuJoCo XML format first using `obj2mjcf`.

**Root cause**: Stage 5 (MuJoCo assembly) was being skipped, so mesh files were never converted.

## Complete Solution

### Understanding the Pipeline

The correct flow is:
1. **Semantic Pipeline** generates placement solution with positions/orientations
2. **obj2mjcf conversion** converts .glb/.obj files to MuJoCo XML format
3. **MuJoCo assembly** includes converted XML files using `<include>` tags
4. **Final scene** loads successfully in MuJoCo viewer

### The Fix

**File**: `creator/runner.py` (Stage 5)

**Before** (INCORRECT - skipping conversion):
```python
# Stage 5: MuJoCo assembly (SKIPPED - already done by pipeline)
print("[pipeline] Stage 5: Skipping assembly (done by Semantic Pipeline)")
saved_models = full_placed_models
```

**After** (CORRECT - running conversion):
```python
# Stage 5: MuJoCo assembly (convert meshes and generate final XML)
print("[pipeline] Stage 5: MuJoCo assembly (converting meshes)")

# Use interface.add_models to convert meshes and generate proper MuJoCo XML
# This will convert .glb files to MuJoCo XML using obj2mjcf
saved_models = interface.add_models(
    chosen_models=chosen_models,
    models=models,
    query=query,
    path_to_save=cache.cache_path,
    world_path=world_path,
    room_half_size=room_half_size,
    pre_placed_models=full_placed_models,  # Use positions from semantic pipeline
    semantic_plan=semantic_plan,
)
```

## How It Works

### Step 1: Semantic Pipeline (Stage 4)
```python
# Generates placement solution with exact positions
placement_solution = PlacementSolution(
    objects=[
        PlacedObject(
            id="table_1",
            model_loc="/path/to/table.glb",  # Raw .glb file
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            orientation={"yaw_deg": 0.0, ...},
        ),
        PlacedObject(
            id="chair_1",
            model_loc="/path/to/chair.glb",  # Raw .glb file
            position={"x": 0.0, "y": 0.6, "z": 0.34},
            orientation={"yaw_deg": 0.0, ...},
        ),
    ]
)
```

### Step 2: Convert to full_placed_models Format
```python
full_placed_models = [
    {
        "Model": "table",
        "uuid": "table_1",
        "model_loc": "/path/to/table.glb",
        "save_fn": "table_1",  # Used for converted XML filename
        "Pose": {"x": 0.0, "y": 0.0, "z": 0.0},
        "yaw_deg": 0.0,
        ...
    },
    {
        "Model": "chair",
        "uuid": "chair_1",
        "model_loc": "/path/to/chair.glb",
        "save_fn": "chair_1",
        "Pose": {"x": 0.0, "y": 0.6, "z": 0.34},
        "yaw_deg": 0.0,
        ...
    },
]
```

### Step 3: MuJoCo Assembly (Stage 5)
```python
# interface.add_models() does:
# 1. For each model, convert .glb to MuJoCo XML using obj2mjcf
#    Input:  /path/to/table.glb
#    Output: /cache/table_1/table_1.xml
#
# 2. Generate main scene XML with <include> tags
#    <mujoco>
#      <worldbody>
#        <body name="table_1" pos="0 0 0" euler="0 0 0">
#          <include file="/cache/table_1/table_1.xml"/>
#        </body>
#        <body name="chair_1" pos="0 0.6 0.34" euler="0 0 0">
#          <include file="/cache/chair_1/chair_1.xml"/>
#        </body>
#      </worldbody>
#    </mujoco>
```

### Step 4: Final Scene
The generated XML now has:
- Converted MuJoCo XML files for each model (with proper mesh definitions)
- Main scene XML that includes the converted files
- All positions and orientations from the semantic pipeline

## Key Insight

The `SceneAssembly` class was trying to do too much - it was attempting to directly reference .glb files as meshes, which MuJoCo doesn't support.

The correct approach is:
1. **SceneAssembly** generates a simple placement solution (just positions/orientations)
2. **Existing MuJoCo interface** handles mesh conversion and XML assembly
3. **Positions from semantic pipeline** are preserved via `pre_placed_models` parameter

## Benefits of This Approach

✓ **Reuses existing code** - No need to reimplement obj2mjcf conversion
✓ **Maintains compatibility** - Works with existing model database and caching
✓ **Preserves semantic positions** - Uses `pre_placed_models` to keep exact positions
✓ **Handles all formats** - Works with .glb, .obj, .stl, etc.
✓ **Proper collision meshes** - obj2mjcf generates proper collision geometry

## Testing

```bash
# Generate scene
python main.py "стол и стул"

# Expected output:
# [pipeline] Stage 4: Layout solving (Semantic Enforcement)
# [pipeline] ✓ Placed 2 objects
# [pipeline] Stage 5: MuJoCo assembly (converting meshes)
# [mujoco] Converting table.glb to MuJoCo XML...
# [mujoco] Converting chair.glb to MuJoCo XML...
# [pipeline] Done. Scene saved to: /tmp/ciare_fresh_*/worlds/scene_latest.xml

# View scene
python view_scene.py

# Expected result:
# ✓ Scene loads successfully
# ✓ Objects are visible and positioned correctly
# ✓ No XML parsing errors
# ✓ No "no decoder" errors
```

## Status

✓ **Stage 1 Fix**: XML structure corrected (use `<mesh>` assets, not `<include>` for .glb)
✓ **Stage 2 Fix**: Re-enabled Stage 5 to convert meshes using obj2mjcf
✓ **Integration**: Semantic pipeline positions preserved via `pre_placed_models`

## Files Modified

1. `creator/placement/semantic_enforcement/scene_assembly.py` - Fixed XML generation (reverted to simpler approach)
2. `creator/runner.py` - Re-enabled Stage 5 (MuJoCo assembly) to convert meshes

## Next Steps

The system should now:
1. Generate semantic plans correctly ✓
2. Execute semantic enforcement pipeline ✓
3. Convert meshes to MuJoCo XML ✓ (NEW)
4. Generate valid MuJoCo scene ✓ (NEW)
5. Load in viewer without errors ✓ (NEW)
