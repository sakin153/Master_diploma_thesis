# How to Test the Fix

## Quick Test

```bash
# 1. Generate a simple scene
python main.py "стол и стул"

# 2. View the scene
python view_scene.py
```

## What to Look For

### ✓ Success Indicators

**In the console output:**
```
[pipeline] Stage 4: Layout solving (Semantic Enforcement)
[pipeline] ✓ Generated semantic plan with 2 objects
[Pipeline] Stage 1: Schema Validation
[Pipeline] Stage 2: Distance Resolution
[Pipeline] Stage 3: Orientation Resolution
[Pipeline] Stage 4: Placement Execution
[Pipeline] Stage 5: Scene Assembly
[pipeline] ✓ Placed 2 objects
[pipeline]   table: pos=(0.00, 0.00, 0.00), yaw=0°
[pipeline]   computer chair: pos=(0.00, 0.60, 0.34), yaw=0°
[pipeline] Stage 5: MuJoCo assembly (converting meshes)  ← NEW!
[mujoco] Converting meshes...                             ← NEW!
[pipeline] Done. Scene saved to: /tmp/ciare_fresh_*/worlds/scene_latest.xml
```

**In the viewer:**
- Scene loads without errors
- Table is visible at center
- Chair is visible in front of table (0.6m away)
- No XML parsing errors
- No "no decoder" errors

### ✗ Failure Indicators

**Old errors (should NOT appear):**
```
❌ XML Error: Include error: 'XML parse error 8'
❌ Error: no decoder found for mesh file
```

**If you see these, the fix didn't work.**

## Full Test with Original Query

```bash
# Test with the original complex query
python main.py "Простая столовая где есть 4 стола и 16 стульев у каждого стола по 4 стула"

# View the result
python view_scene.py
```

### Expected Result

**Console output should show:**
```
[pipeline] ✓ Generated semantic plan with 20 objects
[pipeline] ✓ Placed 20 objects
[pipeline]   table_1: pos=(...), yaw=...
[pipeline]   chair_1: pos=(...), yaw=...  ← relative to table_1
[pipeline]   chair_2: pos=(...), yaw=...  ← relative to table_1
[pipeline]   chair_3: pos=(...), yaw=...  ← relative to table_1
[pipeline]   chair_4: pos=(...), yaw=...  ← relative to table_1
[pipeline]   table_2: pos=(...), yaw=...
[pipeline]   chair_5: pos=(...), yaw=...  ← relative to table_2
...
[pipeline] Stage 5: MuJoCo assembly (converting meshes)
[mujoco] Converting 20 models...
[pipeline] Done.
```

**Viewer should show:**
- 4 tables arranged in the room
- 4 chairs around each table
- Chairs facing their respective tables
- Proper spacing between objects
- No collisions or overlaps

## Verification Checklist

- [ ] System generates semantic plan successfully
- [ ] Semantic plan includes relative positioning (e.g., "relative_to": "table_1")
- [ ] Pipeline executes all 5 stages without errors
- [ ] Stage 5 (MuJoCo assembly) runs and converts meshes
- [ ] Scene XML file is generated
- [ ] Scene loads in viewer without XML errors
- [ ] Scene loads in viewer without decoder errors
- [ ] Objects are visible in the viewer
- [ ] Objects are positioned correctly (chairs near tables)
- [ ] Spatial relationships are correct (4 chairs around each table)

## Debugging

### If Stage 5 is still skipped:

Check `creator/runner.py` around line 710. It should say:
```python
print("[pipeline] Stage 5: MuJoCo assembly (converting meshes)")
saved_models = interface.add_models(...)
```

NOT:
```python
print("[pipeline] Stage 5: Skipping assembly (done by Semantic Pipeline)")
```

### If you see "no decoder" errors:

This means Stage 5 is not running or not converting meshes properly. Check:
1. Is `interface.add_models()` being called?
2. Are the mesh files being converted to XML?
3. Check `/tmp/ciare_fresh_*/` for converted XML files

### If you see XML parsing errors:

This means the XML structure is still incorrect. Check:
1. Is the scene using `<include>` tags for converted XML files?
2. Are the converted XML files present?
3. Check the generated `scene_latest.xml` file

## Success Metrics

**Before the fix:**
- ❌ XML parsing error on line 16
- ❌ "no decoder found for mesh file"
- ❌ Scene fails to load in viewer

**After the fix:**
- ✓ No XML parsing errors
- ✓ No decoder errors
- ✓ Scene loads successfully
- ✓ Objects positioned correctly
- ✓ Relative positioning works (chairs near tables)

## Performance Note

The first run may be slower because:
- Models need to be downloaded from HuggingFace
- Meshes need to be converted using obj2mjcf
- Converted files are cached for future runs

Subsequent runs with the same models will be faster.

## Files to Check

If something goes wrong, check these files:

1. **Generated scene**: `/tmp/ciare_fresh_*/worlds/scene_latest.xml`
   - Should have `<include>` tags pointing to converted XML files
   - Should NOT have `<mesh>` tags pointing to .glb files

2. **Converted models**: `/tmp/ciare_fresh_*/table_1/table_1.xml`
   - Should exist for each model
   - Should contain proper MuJoCo XML with mesh definitions

3. **Console output**: Look for Stage 5 execution
   - Should say "MuJoCo assembly (converting meshes)"
   - Should NOT say "Skipping assembly"

## Contact

If the fix doesn't work:
1. Check the console output for errors
2. Check the generated XML files
3. Verify Stage 5 is running
4. Check that `interface.add_models()` is being called

The fix should work if:
- `creator/runner.py` has been updated (Stage 5 re-enabled)
- The system can access obj2mjcf for mesh conversion
- Model files are accessible
