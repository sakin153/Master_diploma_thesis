#!/usr/bin/env python3
"""Find the exact blocker that prevents duplicate models."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_blocker():
    """Test each stage to find where duplicates are filtered."""
    
    query = "стол и 3 стула"
    
    print("=" * 80)
    print(f"BLOCKER DETECTION: {query}")
    print("=" * 80)
    
    from creator.scene.prompt_expander import expand_prompt
    from creator.llm.model import prompt_model
    from creator.model_databases.embodied_gen import EmbodiedGenLoader
    from creator.sim_interfaces.mujoco import MujocoSimInterface
    from creator.runner import _rank_models_for_object, _OBJECT_FALLBACKS, _generate_semantic_plan_with_validation
    from creator.scene.room_planner import compute_room_half_size
    import re
    
    chosen_model = "deepseek-v3.1:671b-cloud"
    
    # Stage 0: Expansion
    print("\n[STAGE 0] Prompt Expansion")
    scene_spec = expand_prompt(query, prompt_model_fn=prompt_model, llm_model=chosen_model, verbose=False)
    
    objects = []
    for h in (scene_spec.estimated_objects or []):
        name = str(getattr(h, "name", "")).strip()
        qty = max(1, min(int(getattr(h, "quantity", 1)), 20))
        objects.extend([name] * qty)
    
    print(f"✓ Objects: {objects} (count={len(objects)})")
    
    # Stage 1: Model selection
    print("\n[STAGE 1] Model Selection")
    
    try:
        loader = EmbodiedGenLoader(dataset_dir=None)
        models, _ = loader.get_models()
        models_full = loader.get_models_full()
    except Exception as e:
        print(f"✗ Cannot load dataset: {e}")
        return
    
    chosen_models = []
    used_uuids = set()
    first_occurrence = {}
    
    for i, obj in enumerate(objects):
        obj_raw = str(obj).strip()
        obj_key = obj_raw.lower()
        obj_clean = re.sub(r'\([^)]*\)', '', obj_key).strip()
        search_key = _OBJECT_FALLBACKS.get(obj_clean, obj_clean)
        
        ranked = _rank_models_for_object(search_key, models, limit=10)
        if not ranked:
            continue
        
        unused = [r for r in ranked if str(r.get("uuid") or "") not in used_uuids]
        candidate_pool = unused if unused else ranked
        
        is_first = search_key not in first_occurrence
        if is_first:
            first_occurrence[search_key] = True
            primary = candidate_pool[0]
        else:
            primary = candidate_pool[0]
        
        name = str(primary.get("name", "")).strip()
        uid = str(primary.get("uuid", "")).strip()
        
        print(f"  {i+1}. '{obj_raw}' → {name} (uuid={uid[:8]})")
        
        if uid:
            used_uuids.add(uid)
        if name and uid:
            chosen_models.append({"Model": name, "uuid": uid})
    
    print(f"\n✓ chosen_models: {len(chosen_models)} objects")
    
    # Check for duplicates
    from collections import Counter
    uuid_counts = Counter([m.get("uuid") for m in chosen_models])
    print(f"✓ UUID counts: {dict(uuid_counts)}")
    
    # Stage 2: get_full_placed_models
    print("\n[STAGE 2] get_full_placed_models")
    
    interface = MujocoSimInterface(chosen_model, cache_dir=None)
    full_placed_models = interface.get_full_placed_models(chosen_models, models_full)
    
    print(f"✓ full_placed_models: {len(full_placed_models)} objects")
    
    # Check for duplicates
    uuid_counts2 = Counter([m.get("uuid") for m in full_placed_models])
    print(f"✓ UUID counts: {dict(uuid_counts2)}")
    
    # Stage 3: Load objects and update sizes
    print("\n[STAGE 3] Load objects and update sizes")
    
    objects_map = interface.load_objects(full_placed_models)
    for i, _ in enumerate(full_placed_models):
        uid = str(full_placed_models[i].get("uuid") or "")
        if not uid:
            uid = str(full_placed_models[i].get("name") or f"asset_{i}")
        full_placed_models[i]["uuid"] = uid
        full_placed_models[i]["model_loc"] = objects_map.get(uid)
        safe_uid = re.sub(r"[^a-zA-Z0-9_]+", "_", uid)
        full_placed_models[i]["save_fn"] = safe_uid + f"_{i}"
    
    full_placed_models = interface.update_model_sizes(full_placed_models)
    full_placed_models = interface.normalize_models_to_realistic_scale(
        full_placed_models, query=query,
    )
    
    print(f"✓ After size update: {len(full_placed_models)} objects")
    
    # Check for duplicates
    uuid_counts3 = Counter([m.get("uuid") for m in full_placed_models])
    print(f"✓ UUID counts: {dict(uuid_counts3)}")
    
    # Stage 4: Compute room size
    print("\n[STAGE 4] Compute room size")
    
    room_half_size = compute_room_half_size(
        full_placed_models,
        room_type=scene_spec.room_type,
        verbose=False,
    )
    
    print(f"✓ Room half size: {room_half_size}m")
    
    # Stage 5: Generate semantic plan
    print("\n[STAGE 5] Generate semantic plan")
    
    semantic_plan = _generate_semantic_plan_with_validation(
        prompt_model_fn=prompt_model,
        llm_model=chosen_model,
        query=query,
        room_half_size=room_half_size,
        full_placed_models=full_placed_models,
        max_retries=3,
    )
    
    print(f"✓ Semantic plan generated with {len(semantic_plan.get('objects', []))} objects")
    
    # Check object IDs
    object_ids = [obj.get("id") for obj in semantic_plan.get("objects", [])]
    print(f"✓ Object IDs: {object_ids}")
    
    # Check Model names
    model_names = [obj.get("Model") for obj in semantic_plan.get("objects", [])]
    model_counts = Counter(model_names)
    print(f"✓ Model counts: {dict(model_counts)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Stage 0 (expansion):        {len(objects)} objects")
    print(f"Stage 1 (chosen_models):    {len(chosen_models)} objects")
    print(f"Stage 2 (full_placed):      {len(full_placed_models)} objects")
    print(f"Stage 5 (semantic_plan):    {len(semantic_plan.get('objects', []))} objects")
    
    if len(semantic_plan.get('objects', [])) < len(full_placed_models):
        print(f"\n❌ BLOCKER FOUND IN STAGE 5!")
        print(f"   LLM generated only {len(semantic_plan.get('objects', []))} objects")
        print(f"   Expected: {len(full_placed_models)} objects")
        print(f"\n   This means the LLM is not following the instruction to generate")
        print(f"   one object for each entry in the models list.")
    elif len(full_placed_models) < len(chosen_models):
        print(f"\n❌ BLOCKER FOUND IN STAGE 2!")
        print(f"   get_full_placed_models returned only {len(full_placed_models)} objects")
        print(f"   Expected: {len(chosen_models)} objects")
    elif len(chosen_models) < len(objects):
        print(f"\n❌ BLOCKER FOUND IN STAGE 1!")
        print(f"   Model selection returned only {len(chosen_models)} objects")
        print(f"   Expected: {len(objects)} objects")
    else:
        print(f"\n✅ NO BLOCKER FOUND IN PIPELINE!")
        print(f"   All stages have correct object count.")

if __name__ == "__main__":
    test_blocker()
