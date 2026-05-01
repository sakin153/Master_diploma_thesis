#!/usr/bin/env python3
"""Full audit to trace where objects are lost."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_full_pipeline():
    """Test the full pipeline and report at each stage."""
    
    query = "стол и 3 стула"
    
    print("=" * 80)
    print(f"FULL AUDIT: {query}")
    print("=" * 80)
    
    from creator.scene.prompt_expander import expand_prompt
    from creator.llm.model import prompt_model
    from creator.model_databases.embodied_gen import EmbodiedGenLoader
    from creator.sim_interfaces.mujoco import MujocoSimInterface
    
    chosen_model = "deepseek-v3.1:671b-cloud"
    
    # Stage 0: Expansion
    print("\n[STAGE 0] Prompt Expansion")
    scene_spec = expand_prompt(query, prompt_model_fn=prompt_model, llm_model=chosen_model, verbose=False)
    
    objects = []
    for h in (scene_spec.estimated_objects or []):
        name = str(getattr(h, "name", "")).strip()
        qty = max(1, min(int(getattr(h, "quantity", 1)), 20))
        objects.extend([name] * qty)
    
    print(f"✓ Objects list: {objects}")
    print(f"✓ Total: {len(objects)}")
    
    # Stage 1: Model selection
    print("\n[STAGE 1] Model Selection")
    
    try:
        loader = EmbodiedGenLoader(dataset_dir=None)
        models, _ = loader.get_models()
        models_full = loader.get_models_full()
    except:
        print("✗ Cannot load EmbodiedGen dataset")
        return
    
    from creator.runner import _rank_models_for_object, _OBJECT_FALLBACKS
    import re
    
    chosen_models = []
    obj_state = {}
    used_uuids = set()
    
    for obj in objects:
        obj_raw = str(obj).strip()
        obj_key = obj_raw.lower()
        obj_clean = re.sub(r'\([^)]*\)', '', obj_key).strip()
        search_key = _OBJECT_FALLBACKS.get(obj_clean, obj_clean)
        
        state = obj_state.get(search_key)
        if state is None:
            ranked = _rank_models_for_object(search_key, models, limit=10)
            if not ranked:
                print(f"  ✗ No match for '{obj}'")
                continue
            primary = ranked[0]
            state = {"order": [primary], "idx": 0}
            obj_state[search_key] = state
        
        picked = state["order"][0]
        name = str(picked.get("name", "")).strip()
        uid = str(picked.get("uuid", "")).strip()
        
        print(f"  '{obj_raw}' → {name} (uuid={uid[:8]})")
        
        if uid:
            used_uuids.add(uid)
        if name and uid:
            chosen_models.append({"Model": name, "uuid": uid})
        elif name:
            chosen_models.append({"Model": name})
    
    print(f"\n✓ chosen_models length: {len(chosen_models)}")
    print(f"✓ chosen_models: {[m.get('Model') for m in chosen_models]}")
    
    # Stage 2: get_full_placed_models
    print("\n[STAGE 2] get_full_placed_models")
    
    interface = MujocoSimInterface(chosen_model, cache_dir=None)
    full_placed_models = interface.get_full_placed_models(chosen_models, models_full)
    
    print(f"✓ full_placed_models length: {len(full_placed_models)}")
    print(f"✓ full_placed_models: {[m.get('Model') for m in full_placed_models]}")
    
    # Stage 3: Check what goes into semantic plan prompt
    print("\n[STAGE 3] Semantic Plan Prompt")
    
    models_str_lines = []
    for i, model in enumerate(full_placed_models):
        model_name = model.get("Model", f"object_{i}")
        size = model.get("size", [1.0, 1.0, 1.0])
        model_uuid = model.get("uuid", "")
        models_str_lines.append(f"  - Model: {model_name}, uuid: {model_uuid}")
    
    print(f"✓ Models in prompt ({len(models_str_lines)} total):")
    for line in models_str_lines:
        print(f"  {line}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Stage 0 (expansion):        {len(objects)} objects")
    print(f"Stage 1 (chosen_models):    {len(chosen_models)} objects")
    print(f"Stage 2 (full_placed):      {len(full_placed_models)} objects")
    print(f"Stage 3 (prompt):           {len(models_str_lines)} objects")
    
    if len(objects) != len(chosen_models):
        print(f"\n⚠️  PROBLEM: Lost {len(objects) - len(chosen_models)} objects between Stage 0 and Stage 1")
        print(f"   Reason: obj_state dictionary uses search_key, so duplicate object types reuse same state")
    
    if len(chosen_models) != len(full_placed_models):
        print(f"\n⚠️  PROBLEM: Lost {len(chosen_models) - len(full_placed_models)} objects between Stage 1 and Stage 2")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_full_pipeline()
