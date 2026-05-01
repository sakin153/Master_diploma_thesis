#!/usr/bin/env python3
"""Full diagnostic to see exactly what's happening at each stage."""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def diagnose():
    query = "стол и 3 стула"
    
    print("=" * 80)
    print(f"FULL DIAGNOSTIC: {query}")
    print("=" * 80)
    
    from creator.scene.prompt_expander import expand_prompt
    from creator.llm.model import prompt_model
    from creator.model_databases.embodied_gen import EmbodiedGenLoader
    from creator.sim_interfaces.mujoco import MujocoSimInterface
    from creator.runner import _rank_models_for_object, _OBJECT_FALLBACKS
    import re
    
    chosen_model = "deepseek-v3.1:671b-cloud"
    
    # Stage 0: Expansion
    print("\n[STAGE 0] Prompt Expansion")
    print("-" * 80)
    scene_spec = expand_prompt(query, prompt_model_fn=prompt_model, llm_model=chosen_model, verbose=False)
    
    objects = []
    for h in (scene_spec.estimated_objects or []):
        name = str(getattr(h, "name", "")).strip()
        qty = max(1, min(int(getattr(h, "quantity", 1)), 20))
        objects.extend([name] * qty)
    
    print(f"Objects list: {objects}")
    print(f"Total: {len(objects)}")
    
    # Stage 1: Model selection
    print("\n[STAGE 1] Model Selection (with fix)")
    print("-" * 80)
    
    try:
        loader = EmbodiedGenLoader(dataset_dir=None)
        models, _ = loader.get_models()
        models_full = loader.get_models_full()
    except Exception as e:
        print(f"Cannot load dataset: {e}")
        return
    
    chosen_models = []
    dropped_objects = []
    used_uuids = set()
    first_occurrence = {}
    
    for i, obj in enumerate(objects):
        obj_raw = str(obj).strip()
        obj_key = obj_raw.lower()
        obj_clean = re.sub(r'\([^)]*\)', '', obj_key).strip()
        search_key = _OBJECT_FALLBACKS.get(obj_clean, obj_clean)
        
        print(f"\n  Object {i+1}/{len(objects)}: '{obj_raw}'")
        print(f"    search_key: {search_key}")
        
        # Always rank models (no caching)
        ranked = _rank_models_for_object(search_key, models, limit=10)
        if not ranked:
            print(f"    ✗ No match")
            dropped_objects.append(obj_key)
            continue
        
        print(f"    Ranked models: {len(ranked)}")
        
        # Prefer unused models
        unused = [r for r in ranked if str(r.get("uuid") or "") not in used_uuids]
        candidate_pool = unused if unused else ranked
        
        print(f"    Unused models: {len(unused)}")
        print(f"    Candidate pool: {len(candidate_pool)}")
        
        # Select model
        if len(candidate_pool) == 1:
            primary = candidate_pool[0]
            print(f"    Selection: Only 1 candidate")
        else:
            is_first = search_key not in first_occurrence
            if is_first:
                first_occurrence[search_key] = True
                print(f"    Selection: First occurrence, using first candidate (skipping LLM)")
                primary = candidate_pool[0]
            else:
                print(f"    Selection: Subsequent occurrence, using first unused")
                primary = candidate_pool[0]
        
        name = str(primary.get("name", "")).strip()
        uid = str(primary.get("uuid", "")).strip()
        
        print(f"    ✓ Selected: {name} (uuid={uid[:8]})")
        
        if uid:
            used_uuids.add(uid)
        if name and uid:
            chosen_models.append({"Model": name, "uuid": uid})
        elif name:
            chosen_models.append({"Model": name})
    
    print(f"\n{'=' * 80}")
    print(f"chosen_models length: {len(chosen_models)}")
    print(f"chosen_models:")
    for i, m in enumerate(chosen_models):
        print(f"  {i+1}. {m.get('Model')} (uuid={m.get('uuid', 'none')[:8]})")
    
    # Stage 2: get_full_placed_models
    print(f"\n[STAGE 2] get_full_placed_models")
    print("-" * 80)
    
    interface = MujocoSimInterface(chosen_model, cache_dir=None)
    full_placed_models = interface.get_full_placed_models(chosen_models, models_full)
    
    print(f"full_placed_models length: {len(full_placed_models)}")
    print(f"full_placed_models:")
    for i, m in enumerate(full_placed_models):
        print(f"  {i+1}. {m.get('Model')} (uuid={m.get('uuid', 'none')[:8]})")
    
    # Stage 3: Check semantic plan prompt
    print(f"\n[STAGE 3] Semantic Plan Prompt Input")
    print("-" * 80)
    
    print(f"Number of models passed to semantic plan: {len(full_placed_models)}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Stage 0 (expansion):        {len(objects)} objects")
    print(f"Stage 1 (chosen_models):    {len(chosen_models)} objects")
    print(f"Stage 2 (full_placed):      {len(full_placed_models)} objects")
    
    if len(objects) == len(chosen_models) == len(full_placed_models):
        print(f"\n✅ ALL STAGES HAVE CORRECT COUNT!")
        print(f"\nNext step: Check if LLM generates all {len(full_placed_models)} objects in semantic plan")
    else:
        print(f"\n❌ MISMATCH DETECTED!")
        if len(objects) != len(chosen_models):
            print(f"   Lost {len(objects) - len(chosen_models)} objects between Stage 0 and Stage 1")
        if len(chosen_models) != len(full_placed_models):
            print(f"   Lost {len(chosen_models) - len(full_placed_models)} objects between Stage 1 and Stage 2")

if __name__ == "__main__":
    diagnose()
