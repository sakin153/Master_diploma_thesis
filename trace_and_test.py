#!/usr/bin/env python3
"""Trace and test script with full logging."""

import sys
import os
import json
import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create trace directory
TRACE_DIR = Path("trace_logs")
TRACE_DIR.mkdir(exist_ok=True)

# Current trace session
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = TRACE_DIR / f"session_{timestamp}"
SESSION_DIR.mkdir(exist_ok=True)

print(f"📁 Trace session: {SESSION_DIR}")

# Capture all logs
log_buffer = []

def log(message, level="INFO"):
    """Log message to buffer and console."""
    log_line = f"[{level}] {message}"
    log_buffer.append(log_line)
    print(log_line)

def save_trace(name, data):
    """Save trace data to JSON file."""
    trace_file = SESSION_DIR / f"{name}.json"
    with open(trace_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    log(f"Saved trace: {trace_file}")

def save_logs():
    """Save all logs to file."""
    log_file = SESSION_DIR / "full_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_buffer))
    log(f"Saved logs: {log_file}")

def test_scene_generation(query):
    """Test scene generation with full tracing."""
    
    log(f"Testing query: {query}", "TEST")
    
    from creator.scene.prompt_expander import expand_prompt
    from creator.llm.model import prompt_model
    from creator.model_databases.embodied_gen import EmbodiedGenLoader
    from creator.sim_interfaces.mujoco import MujocoSimInterface
    from creator.runner import _rank_models_for_object, _OBJECT_FALLBACKS, _generate_semantic_plan_with_validation
    from creator.scene.room_planner import compute_room_half_size
    import re
    
    chosen_model = "deepseek-v3.1:671b-cloud"
    
    # Stage 0: Expansion
    log("Stage 0: Prompt Expansion", "STAGE")
    scene_spec = expand_prompt(query, prompt_model_fn=prompt_model, llm_model=chosen_model, verbose=False)
    
    objects = []
    for h in (scene_spec.estimated_objects or []):
        name = str(getattr(h, "name", "")).strip()
        qty = max(1, min(int(getattr(h, "quantity", 1)), 20))
        objects.extend([name] * qty)
    
    log(f"Objects extracted: {objects} (count={len(objects)})")
    
    save_trace("stage0_expansion", {
        "query": query,
        "expanded_query": scene_spec.effective_query,
        "room_type": scene_spec.room_type,
        "estimated_objects": [str(h) for h in (scene_spec.estimated_objects or [])],
        "objects_list": objects,
        "count": len(objects)
    })
    
    # Stage 1: Model selection
    log("Stage 1: Model Selection", "STAGE")
    
    try:
        loader = EmbodiedGenLoader(dataset_dir=None)
        models, _ = loader.get_models()
        models_full = loader.get_models_full()
        log(f"Loaded {len(models_full)} models from database")
    except Exception as e:
        log(f"Failed to load dataset: {e}", "ERROR")
        return False
    
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
            log(f"No match for '{obj}'", "WARN")
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
        
        log(f"  [{i+1}] '{obj_raw}' → {name} (uuid={uid[:8]})")
        
        if uid:
            used_uuids.add(uid)
        if name and uid:
            chosen_models.append({"Model": name, "uuid": uid})
    
    log(f"chosen_models count: {len(chosen_models)}")
    
    save_trace("stage1_model_selection", {
        "objects_input": objects,
        "chosen_models": chosen_models,
        "count": len(chosen_models)
    })
    
    # Stage 2: get_full_placed_models
    log("Stage 2: get_full_placed_models", "STAGE")
    
    interface = MujocoSimInterface(chosen_model, cache_dir=None)
    full_placed_models = interface.get_full_placed_models(chosen_models, models_full)
    
    log(f"full_placed_models count: {len(full_placed_models)}")
    
    save_trace("stage2_full_placed_models", {
        "full_placed_models": [{"Model": m.get("Model"), "uuid": m.get("uuid")} for m in full_placed_models],
        "count": len(full_placed_models)
    })
    
    # Stage 3: Load and update
    log("Stage 3: Load objects and update sizes", "STAGE")
    
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
    
    log(f"After size update: {len(full_placed_models)} objects")
    
    # Stage 4: Room size
    log("Stage 4: Compute room size", "STAGE")
    
    room_half_size = compute_room_half_size(
        full_placed_models,
        room_type=scene_spec.room_type,
        verbose=False,
    )
    
    log(f"Room half size: {room_half_size}m")
    
    # Stage 5: Semantic plan
    log("Stage 5: Generate semantic plan", "STAGE")
    
    semantic_plan = _generate_semantic_plan_with_validation(
        prompt_model_fn=prompt_model,
        llm_model=chosen_model,
        query=query,
        room_half_size=room_half_size,
        full_placed_models=full_placed_models,
        max_retries=3,
    )
    
    log(f"Semantic plan generated with {len(semantic_plan.get('objects', []))} objects")
    
    # Extract object info
    semantic_objects = []
    for obj in semantic_plan.get('objects', []):
        semantic_objects.append({
            "id": obj.get("id"),
            "Model": obj.get("Model"),
            "type": obj.get("type"),
            "is_static": obj.get("is_static")
        })
    
    save_trace("stage5_semantic_plan", {
        "objects": semantic_objects,
        "count": len(semantic_plan.get('objects', [])),
        "room_size": semantic_plan.get('room_size')
    })
    
    # Stage 6: Full generation
    log("Stage 6: Full scene generation", "STAGE")
    
    from creator.runner import generate_world
    
    try:
        world_path = generate_world(
            query=query,
            cache_dir=None,
            vlm_validation=False,
            max_vlm_iters=0,
            assets_dir=None,
            seed=42,
        )
        
        log(f"Scene generated: {world_path}")
        
        # Analyze XML
        if os.path.exists(world_path):
            with open(world_path, 'r') as f:
                xml_content = f.read()
            
            # Copy XML to trace
            xml_file = SESSION_DIR / "scene.xml"
            with open(xml_file, 'w') as f:
                f.write(xml_content)
            
            # Count bodies
            import re
            bodies = re.findall(r'<body name="([^"]+)"', xml_content)
            meshes = re.findall(r'<mesh name="([^"]+)"', xml_content)
            
            log(f"XML analysis: {len(bodies)} bodies, {len(meshes)} meshes")
            
            save_trace("stage6_final_scene", {
                "world_path": world_path,
                "bodies": bodies,
                "bodies_count": len(bodies),
                "meshes": meshes,
                "meshes_count": len(meshes)
            })
            
            # Summary
            log("=" * 80, "SUMMARY")
            log(f"Stage 0 (expansion):        {len(objects)} objects")
            log(f"Stage 1 (chosen_models):    {len(chosen_models)} objects")
            log(f"Stage 2 (full_placed):      {len(full_placed_models)} objects")
            log(f"Stage 5 (semantic_plan):    {len(semantic_plan.get('objects', []))} objects")
            log(f"Stage 6 (final XML):        {len(bodies)} bodies")
            
            # Check for issues
            if len(bodies) < len(objects):
                log(f"❌ ISSUE: Expected {len(objects)} objects, got {len(bodies)} in final scene", "ERROR")
                
                if len(semantic_plan.get('objects', [])) < len(full_placed_models):
                    log(f"   Problem in Stage 5: LLM generated only {len(semantic_plan.get('objects', []))} objects", "ERROR")
                elif len(bodies) < len(semantic_plan.get('objects', [])):
                    log(f"   Problem in Stage 6: Scene assembly created only {len(bodies)} bodies", "ERROR")
                
                return False
            else:
                log(f"✅ SUCCESS: All {len(objects)} objects present in final scene", "SUCCESS")
                return True
        else:
            log(f"Scene file not found: {world_path}", "ERROR")
            return False
            
    except Exception as e:
        log(f"Scene generation failed: {e}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
        return False

if __name__ == "__main__":
    query = "стол и 3 стула"
    
    log("=" * 80)
    log(f"TRACE AND TEST: {query}")
    log("=" * 80)
    
    success = test_scene_generation(query)
    
    save_logs()
    
    log("=" * 80)
    if success:
        log("✅ TEST PASSED", "RESULT")
    else:
        log("❌ TEST FAILED", "RESULT")
    log(f"📁 Trace saved to: {SESSION_DIR}")
    log("=" * 80)
    
    sys.exit(0 if success else 1)
