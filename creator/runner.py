import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tinydb import TinyDB

from creator.contexts_prompts.constraints import fmt_constraints_plan_tmpl
from creator.contexts_prompts.disambiguation import fmt_disambiguation_tmpl
from creator.contexts_prompts.semantic_plan import fmt_semantic_plan_tmpl
from creator.model_databases.embodied_gen import EmbodiedGenLoader
from creator.placement import (
    build_semantic_plan,
    repair_layout_by_constraints,
    validate_and_repair_layout,
)
from creator.placement.plan import _stabilize_dense_group_constraints
from creator.placement.semantic_enforcement.pipeline import (
    SemanticEnforcementPipeline,
    SchemaValidationError,
)
from creator.postprocess import refine_scene_with_engine
from creator.sim_interfaces.mujoco import MujocoSimInterface
from creator.utils.cache import Cache
from creator.utils.json import NumpyEncoder

def _tokenize(text: str) -> List[str]:
    # Support both Latin and Cyrillic (Russian) characters
    return re.findall(r"[a-zа-яё0-9]+", (text or "").lower())


# Fallback mapping for common objects not in catalog
_OBJECT_FALLBACKS = {
    "book": "box",
    "lamp": "bottle",
    "light": "bottle",
    "pillow": "cushion",
    "cushion": "pillow",
    "tv": "television",
    "tv stand": "table",
    "sofa": "lounge chair",
    "couch": "lounge chair",
    "rug": "carpet",
}


def _singularize(word: str) -> str:
    w = (word or "").strip().lower()
    if len(w) > 3 and w.endswith("ves"):
        return w[:-3] + "f"
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 2 and w.endswith("s"):
        return w[:-1]
    return w


def _score_model_for_object(obj: str, model: Dict[str, Any]) -> int:
    obj_l = (obj or "").strip().lower()
    if not obj_l:
        return 0

    name = str(model.get("name", "")).lower()
    tags = model.get("tags") or []
    categories = model.get("categories") or []
    meta = " ".join([str(x).lower() for x in (tags + categories) if x])

    score = 0

    # Exact match in name - highest priority
    if obj_l == name:
        score += 20
    elif obj_l in name:
        score += 12
    
    # Match in categories/tags
    if obj_l in meta:
        score += 6

    obj_tokens = [t for t in _tokenize(obj_l) if len(t) > 1]
    if not obj_tokens:
        return score

    # Token matching
    for t in obj_tokens:
        if t in name:
            score += 4
        if t in meta:
            score += 2

    head = _singularize(obj_tokens[-1]) if obj_tokens else obj_l

    # Universal category relevance - penalize obviously wrong categories
    category_str = " ".join(str(c).lower() for c in categories)
    
    # If object name appears in categories, boost
    if head in category_str:
        score += 5
    
    # Generic irrelevant category penalties
    irrelevant_for_physical_objects = ["abstract", "icon", "logo", "symbol", "ui", "interface"]
    if any(k in category_str for k in irrelevant_for_physical_objects):
        score -= 15
    
    # Penalize miniatures/toys when looking for real objects
    if any(k in name for k in ["miniature", "toy", "lego", "figurine"]) and head not in {"toy"}:
        score -= 10

    # Common disambiguation
    if head == "chair" and "wheelchair" in name:
        score -= 12
    
    if head in {"desk", "table"}:
        if any(k in name for k in ["lamp", "light", "fan"]):
            score -= 15

    return score


def _rank_models_for_object(
    obj: str,
    models: Sequence[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for m in models:
        s = _score_model_for_object(obj, m)
        if s >= 3:
            scored.append((s, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    seen = set()
    for s, m in scored:
        key = (m.get("uuid"), m.get("name"))
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
        if len(out) >= limit:
            break
    return out


REJECTED = object()  # sentinel: LLM explicitly said no candidate fits


def _llm_pick_candidate(
    *,
    obj: str,
    scene_query: str,
    candidates: Sequence[Dict[str, Any]],
    prompt_model_fn: Any,
    llm_model: str,
) -> Any:
    """Ask the LLM to pick the best candidate uuid from a small ranked list.

    Sends a compact JSON payload (uuid + name + description, ≤10 rows) so the
    prompt stays small.

    Return values:
      - candidate dict        — LLM picked it
      - REJECTED (sentinel)   — LLM said "none" (categorical mismatch);
                                caller should drop the request
      - None                  — call failed / unparsable; caller falls back
                                to candidates[0]
    """
    if not candidates:
        return None

    payload = []
    for c in candidates:
        uid = str(c.get("uuid") or "").strip()
        if not uid:
            continue
        payload.append({
            "uuid": uid,
            "name": str(c.get("name") or ""),
            "description": str(c.get("description") or ""),
        })
    if not payload:
        return None

    by_uuid = {str(c.get("uuid") or ""): c for c in candidates}

    prompt = fmt_disambiguation_tmpl.format(
        scene_query=scene_query,
        object=obj,
        candidates=json.dumps(payload, ensure_ascii=False),
    )

    try:
        raw = prompt_model_fn(prompt, str(obj), llm_model)
    except Exception as e:  # noqa: BLE001
        print(f"[disambiguation] LLM call failed for {obj!r}: {e}")
        return None

    chosen_uuid = ""
    if isinstance(raw, dict):
        chosen_uuid = str(raw.get("uuid") or raw.get("UUID") or "").strip()
    elif isinstance(raw, str):
        if "none" in raw.lower():
            chosen_uuid = "none"
        else:
            m = re.search(r"[0-9a-f]{16,32}", raw)
            if m:
                chosen_uuid = m.group(0)
    elif isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict):
            chosen_uuid = str(first.get("uuid") or "").strip()

    if chosen_uuid.lower() == "none":
        return REJECTED

    return by_uuid.get(chosen_uuid)


def _generate_semantic_plan_with_validation(
    *,
    prompt_model_fn,
    llm_model: str,
    query: str,
    room_half_size: float,
    full_placed_models: List[Dict[str, Any]],
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Generate semantic plan using new format with validation feedback loop.
    
    This function:
    1. Generates semantic plan using the new prompt template
    2. Validates the plan against the schema
    3. If validation fails, sends error back to LLM for correction
    4. Retries up to max_retries times
    
    Args:
        prompt_model_fn: LLM prompt function
        llm_model: LLM model name
        query: User query
        room_half_size: Room half size in meters
        full_placed_models: List of models with sizes and metadata
        max_retries: Maximum number of retry attempts
        
    Returns:
        Valid semantic plan dictionary
        
    Raises:
        RuntimeError: If all retry attempts fail
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Build models string for prompt
    models_str_lines = []
    for i, model in enumerate(full_placed_models):
        model_name = model.get("Model", f"object_{i}")
        size = model.get("size", [1.0, 1.0, 1.0])
        model_uuid = model.get("uuid", "")
        
        models_str_lines.append(
            f"  - Model: {model_name}\n"
            f"    Size: {{'width': {size[0]:.2f}, 'length': {size[1]:.2f}, 'height': {size[2]:.2f}}}\n"
            f"    uuid: {model_uuid}"
        )
    
    models_str = "\n".join(models_str_lines)
    
    # Add explicit count instruction
    object_count = len(full_placed_models)
    count_instruction = f"\n\nCRITICAL: You MUST generate EXACTLY {object_count} objects in your semantic plan. The list above contains {object_count} entries - create one object specification for each entry, even if some have the same Model name."
    models_str = models_str + count_instruction
    
    # Room dimensions
    room_width = room_half_size * 2
    room_length = room_half_size * 2
    room_height = 3.0
    
    # Initial prompt
    prompt = fmt_semantic_plan_tmpl.format(
        query=query,
        room_width=room_width,
        room_length=room_length,
        room_height=room_height,
        models_str=models_str,
    )
    
    # Validation feedback loop
    for attempt in range(max_retries):
        logger.info(f"[SemanticPlan] Generating semantic plan (attempt {attempt + 1}/{max_retries})")
        
        try:
            # Call LLM
            raw_response = prompt_model_fn(prompt, query, llm_model)
            
            # Parse response
            if isinstance(raw_response, dict):
                semantic_plan = raw_response
            elif isinstance(raw_response, str):
                semantic_plan = json.loads(raw_response)
            else:
                raise ValueError(f"Unexpected LLM response type: {type(raw_response)}")
            
            # Validate schema
            from creator.placement.semantic_enforcement.schema_validator import SchemaValidator
            validator = SchemaValidator()
            validation_result = validator.validate_plan(semantic_plan)
            
            if validation_result.success:
                logger.info(f"[SemanticPlan] Schema validation passed on attempt {attempt + 1}")
                return semantic_plan
            else:
                # Validation failed - send error back to LLM
                error_msg = "\n".join(validation_result.errors)
                logger.warning(f"[SemanticPlan] Schema validation failed on attempt {attempt + 1}:\n{error_msg}")
                
                if attempt < max_retries - 1:
                    # Retry with error feedback
                    prompt = (
                        f"The previous semantic plan had validation errors:\n\n{error_msg}\n\n"
                        f"Please fix these errors and generate a corrected semantic plan.\n\n"
                        f"Original request:\n{prompt}"
                    )
                else:
                    raise SchemaValidationError(f"Schema validation failed after {max_retries} attempts:\n{error_msg}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"[SemanticPlan] JSON parsing failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                prompt = (
                    f"The previous response was not valid JSON. Error: {e}\n\n"
                    f"Please generate a valid JSON semantic plan.\n\n"
                    f"Original request:\n{prompt}"
                )
            else:
                raise RuntimeError(f"Failed to parse LLM response as JSON after {max_retries} attempts")
        
        except Exception as e:
            logger.error(f"[SemanticPlan] Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                prompt = (
                    f"An error occurred: {e}\n\n"
                    f"Please try again.\n\n"
                    f"Original request:\n{prompt}"
                )
            else:
                raise RuntimeError(f"Failed to generate semantic plan after {max_retries} attempts: {e}")
    
    raise RuntimeError(f"Failed to generate valid semantic plan after {max_retries} attempts")


def generate_world(
    *,
    query: str,
    cache_dir: Optional[str] = None,
    vlm_validation: bool = True,
    max_vlm_iters: int = 1,
    assets_dir: Optional[str] = None,
    seed: int = 42,
) -> str:
    """Generate a 3D MuJoCo scene from a text query.

    New pipeline (stages):
      0. Prompt expansion    — expand short queries into full scene specs
            1. Object extraction   — LLM selects objects from local assets catalog
      2. Room sizing         — compute room dimensions from object footprints
      3. Semantic plan       — LLM generates placement constraints + scene graph
      4. Layout solving      — floor / wall / surface placement
      5. Assembly            — MuJoCo XML with proper physics (static/dynamic)
      6. Physics refinement  — proxy settle for dynamic objects
      7. VLM validation      — optional LLM layout quality check + repair loop
    """
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    cache = Cache(cache=cache_dir)
    if cache.models_and_worlds_initialized():
        cache.init_models_and_worlds()
    db = TinyDB(os.path.join(cache.worlds_path, "world_db.json"))

    # Make the LLM cache live alongside the rest of the cache so a custom
    # CACHE_DIR (per-run or per-project) gets isolated LLM responses too.
    # Otherwise stale planner outputs from previous runs (different asset
    # catalogues) leak through and pin the new pipeline to old model names.
    os.environ["CIARE_CACHE_DIR"] = cache.cache_path

    from creator.llm.model import prompt_model
    from creator.scene.prompt_expander import expand_prompt
    from creator.scene.room_planner import compute_room_half_size

    chosen_model = "deepseek-v3.1:671b-cloud"

    # Try to load EmbodiedGen dataset, fallback to local assets if not available
    try:
        loader = EmbodiedGenLoader(dataset_dir=assets_dir)
        print(f"[pipeline] Loaded {len(loader.get_models_full())} models from EmbodiedGen dataset: {loader.dataset_dir}")
    except FileNotFoundError as e:
        print(f"[pipeline] EmbodiedGen dataset not found: {e}")
        print(f"[pipeline] Falling back to local assets directory")
        from creator.model_databases.local_assets import LocalAssetsLoader
        loader = LocalAssetsLoader(assets_dir=None)  # Use default assets/ directory
        print(f"[pipeline] Loaded {len(loader.get_models_full())} models from local assets: {loader.assets_dir}")
    
    interface = MujocoSimInterface(chosen_model, cache_dir=cache_dir)

    models, _worlds = loader.get_models()
    models_full = loader.get_models_full()

    # ---------------------------------------------------------------
    # Stage 0: Prompt expansion
    # ---------------------------------------------------------------
    scene_spec = expand_prompt(
        query,
        prompt_model_fn=prompt_model,
        llm_model=chosen_model,
        verbose=True,
    )
    effective_query = scene_spec.effective_query
    print(f"[pipeline] Room type: {scene_spec.room_type}, "
          f"initial half-size estimate: {scene_spec.room_half_size}m")

    # ---------------------------------------------------------------
    # Stage 1: Object extraction
    # ---------------------------------------------------------------
    # Build objects list from prompt expander hints
    objects = []
    for h in (scene_spec.estimated_objects or []):
        name = str(getattr(h, "name", "")).strip()
        qty = max(1, min(int(getattr(h, "quantity", 1)), 20))
        objects.extend([name] * qty)

    if not objects:
        objects = [effective_query]

    print(f"[pipeline] Objects: {objects}")

    # Two-stage matching: local prefilter → small-context LLM disambiguation.
    # 1) For each abstract object, score-rank top-K catalog entries locally
    #    (no LLM, deterministic).
    # 2) Send only those K candidates (uuid + name + description) plus the
    #    original query to the LLM and let it pick the best uuid. This keeps
    #    the LLM context tiny (~10 short rows) while letting it leverage the
    #    rich per-asset descriptions in EmbodiedGen.
    # 3) Each object gets its own model selection (no caching by object type)
    #    to ensure correct quantities (e.g., "3 chairs" → 3 separate chair models).
    chosen_models: List[Dict[str, str]] = []
    dropped_objects: List[str] = []
    used_uuids: set = set()
    
    # Track first occurrence of each object type for LLM disambiguation
    first_occurrence: Dict[str, bool] = {}
    
    for obj in objects:
        obj_raw = str(obj).strip()
        obj_key = obj_raw.lower()

        # Remove parentheses: "Sofa (blue)" -> "Sofa"
        obj_clean = re.sub(r'\([^)]*\)', '', obj_key).strip()

        # Apply fallback mapping
        search_key = _OBJECT_FALLBACKS.get(obj_clean, obj_clean)

        # Always rank models for this object (no caching)
        ranked = _rank_models_for_object(search_key, models, limit=10)
        if not ranked:
            print(f"[pipeline] WARN: '{obj}' has no catalog match, dropped")
            dropped_objects.append(obj_key)
            continue

        # Prefer unused models for variety
        unused = [r for r in ranked if str(r.get("uuid") or "") not in used_uuids]
        candidate_pool = unused if unused else ranked

        # Select model for this object
        if len(candidate_pool) == 1:
            primary = candidate_pool[0]
        else:
            # Use LLM disambiguation only for first occurrence of each object type
            # to save time and API calls
            is_first = search_key not in first_occurrence
            if is_first:
                first_occurrence[search_key] = True
                cand_names = [f"{c.get('name')}({c.get('uuid')[:8]})" for c in candidate_pool[:5]]
                print(f"[disambiguation] '{obj}' → candidates: {', '.join(cand_names)}")
                pick = _llm_pick_candidate(
                    obj=obj,
                    scene_query=effective_query,
                    candidates=candidate_pool,
                    prompt_model_fn=prompt_model,
                    llm_model=chosen_model,
                )
                primary = (pick if pick and pick is not REJECTED else candidate_pool[0])
            else:
                # For subsequent occurrences, just take the first unused model
                primary = candidate_pool[0]

        name = str(primary.get("name", "")).strip()
        uid = str(primary.get("uuid", "")).strip()
        print(f"[Stage1] '{obj_raw}' → {name} (uuid={uid[:8] if uid else 'none'})")
        
        if uid:
            used_uuids.add(uid)
        if name and uid:
            chosen_models.append({"Model": name, "uuid": uid})
        elif name:
            chosen_models.append({"Model": name})

    if dropped_objects:
        from collections import Counter
        cnt = Counter(dropped_objects)
        summary = ", ".join(f"{n}×{c}" if c > 1 else n for n, c in cnt.items())
        print(
            f"[pipeline] Stage 1: {len(dropped_objects)} object instance(s) "
            f"dropped (no catalog match): {summary}"
        )

    # Last-resort: if nothing matched any individual object, rank against the
    # full effective query once and take the top few unique names.
    if not chosen_models:
        fallback_candidates = _rank_models_for_object(effective_query, models, limit=8)
        seen_fb: set = set()
        for m in fallback_candidates:
            name = str(m.get("name", "")).strip()
            if name and name not in seen_fb:
                seen_fb.add(name)
                chosen_models.append({"Model": name})
                if len(chosen_models) >= 6:
                    break

    if not chosen_models:
        raise RuntimeError(
            "No suitable models were found. "
            "Try a more specific prompt (e.g. 'office desk and chair')."
        )

    # ---------------------------------------------------------------
    # Stage 2: Load models, sizes, and normalize scale
    # (room sizing happens after sizes are known)
    # ---------------------------------------------------------------
    full_placed_models = interface.get_full_placed_models(chosen_models, models_full)
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
        full_placed_models, query=effective_query,
    )

    # Compute room size from actual object footprints
    room_half_size = compute_room_half_size(
        full_placed_models,
        room_type=scene_spec.room_type,
        verbose=True,
    )
    print(f"[pipeline] Final room: {room_half_size*2:.0f}m × {room_half_size*2:.0f}m")

    # ---------------------------------------------------------------
    # Stage 3: Semantic plan (constraints + scene graph)
    # ---------------------------------------------------------------
    semantic_plan = build_semantic_plan(
        prompt_model=interface.prompt_model_for_constraints,
        prompt_template=fmt_constraints_plan_tmpl,
        query=effective_query,
        chosen_model=chosen_model,
        chosen_models=chosen_models,
        context_models=models,
        scene_spec=scene_spec,
    )
    if isinstance(semantic_plan.get("objects"), list):
        semantic_plan["objects"] = _stabilize_dense_group_constraints(
            semantic_plan["objects"]
        )
    # Inject room spec into the plan for solvers
    semantic_plan.setdefault("room", {})["half_size"] = room_half_size
    semantic_plan["room"]["type"] = scene_spec.room_type

    interface.save_constraint_graph(
        semantic_plan=semantic_plan,
        query=effective_query,
        output_filename="scene_graph_latest.json",
    )

    # ---------------------------------------------------------------
    # Stage 4: Layout solving (Semantic Enforcement Pipeline ONLY)
    # ---------------------------------------------------------------
    print("[pipeline] Stage 4: Semantic Enforcement Pipeline")
    
    # Generate semantic plan using new format
    print("[pipeline] Generating semantic plan...")
    semantic_plan_new_format = _generate_semantic_plan_with_validation(
        prompt_model_fn=prompt_model,
        llm_model=chosen_model,
        query=effective_query,
        room_half_size=room_half_size,
        full_placed_models=full_placed_models,
        max_retries=3,
    )
    
    print(f"[pipeline] ✓ Generated semantic plan with {len(semantic_plan_new_format.get('objects', []))} objects")
    
    # Add model_loc to each object in semantic plan BEFORE pipeline execution
    # LLM doesn't generate model_loc, so we add it from full_placed_models
    print("[pipeline] Adding model_loc to semantic plan objects...")
    for obj in semantic_plan_new_format.get('objects', []):
        model_name = obj.get('Model', '')
        # Find matching model in full_placed_models
        matching_model = next(
            (m for m in full_placed_models if m.get('Model') == model_name),
            None
        )
        if matching_model:
            obj['model_loc'] = matching_model.get('model_loc', '')
            obj['uuid'] = matching_model.get('uuid', obj.get('id', ''))
            obj['save_fn'] = matching_model.get('save_fn', '')
            obj['_up_axis'] = matching_model.get('_up_axis', 'y')
            print(f"[pipeline]   {model_name}: model_loc={obj['model_loc'][:50]}..." if len(obj['model_loc']) > 50 else f"[pipeline]   {model_name}: model_loc={obj['model_loc']}")
        else:
            print(f"[pipeline] WARNING: No model_loc found for {model_name}, using empty string")
            obj['model_loc'] = ''
            obj['uuid'] = obj.get('id', '')
            obj['save_fn'] = ''
            obj['_up_axis'] = 'y'
    
    # Execute pipeline
    print("[pipeline] Executing Semantic Enforcement Pipeline...")
    pipeline = SemanticEnforcementPipeline(
        workspace_root=".",
        output_dir=cache.cache_path,
        enable_tracking=True,
        enable_collision_check=True,
    )
    
    mujoco_xml, placement_solution, pipeline_trace = pipeline.execute_pipeline(
        semantic_plan_dict=semantic_plan_new_format,
        output_filename="scene_latest_semantic",
    )
    
    print(f"[pipeline] ✓ Placed {len(placement_solution.objects)} objects")
    
    # Convert PlacementSolution back to full_placed_models format
    new_full_placed_models = []
    for i, placed_obj in enumerate(placement_solution.objects):
        obj_uuid = placed_obj.uuid if hasattr(placed_obj, 'uuid') else placed_obj.id
        original = next(
            (m for m in full_placed_models if m.get("uuid") == obj_uuid),
            next((m for m in full_placed_models if m.get("Model") == placed_obj.Model), {})
        )
        
        # Generate unique save_fn for each object instance
        safe_uuid = re.sub(r"[^a-zA-Z0-9_]+", "_", obj_uuid)
        unique_save_fn = f"{safe_uuid}_{i}"
        
        merged = {
            "Model": placed_obj.Model,
            "uuid": obj_uuid,
            "model_loc": placed_obj.model_loc,
            "save_fn": unique_save_fn,  # ← Use unique save_fn for each instance
            "_up_axis": original.get("_up_axis", "y"),
            "size": [placed_obj.size["width"], placed_obj.size["length"], placed_obj.size["height"]],
            "is_static": placed_obj.is_static,
            "Pose": {
                "x": placed_obj.position["x"],
                "y": placed_obj.position["y"],
                "z": placed_obj.position["z"],
            },
            "yaw_deg": placed_obj.orientation["yaw_deg"],
        }
        
        new_full_placed_models.append(merged)
        
        pose = merged.get("Pose", {})
        print(f"[pipeline]   {merged.get('Model')}: pos=({pose.get('x', 0):.2f}, {pose.get('y', 0):.2f}, {pose.get('z', 0):.2f}), yaw={merged.get('yaw_deg', 0):.0f}°")
    
    full_placed_models = new_full_placed_models
    
    # Save the MuJoCo XML from pipeline
    world_name = "scene_latest"
    world_path = (
        os.path.join(cache.worlds_path, world_name) + interface.get_world_extension()
    )
    with open(world_path, "w", encoding="utf-8") as f:
        f.write(mujoco_xml)
    
    print(f"[pipeline] ✓ Saved MuJoCo XML to {world_path}")
    
    # Mark that we used the semantic pipeline
    used_semantic_pipeline = True
    
    # ---------------------------------------------------------------
    # Stage 4.5: Detect and place robots as virtual objects
    # ---------------------------------------------------------------
    from creator.robots.detector import detect_robots
    from creator.robots.placer import place_robots
    
    detected_robots = detect_robots(query)
    robot_placements = []
    if detected_robots:
        print(f"[pipeline] Detected robots: {detected_robots}")
        robot_placements = place_robots(detected_robots, full_placed_models, room_half_size)
        
        # Add robots as virtual objects so small_object solver avoids them
        for placement in robot_placements:
            from creator.robots.catalog import get_robot_info
            robot_info = get_robot_info(placement["robot_id"])
            if not robot_info:
                continue
            
            base_size = robot_info.get("base_size", [0.3, 0.6, 0.3])
            pos = placement["pos"]
            
            # Calculate volume (make it large so small_objects solver skips it)
            volume = base_size[0] * base_size[1] * base_size[2]
            
            # Add as virtual object with is_robot flag
            virtual_robot = {
                "Model": f"robot_{placement['robot_id']}",
                "uuid": f"robot_{placement['robot_id']}",
                "is_robot": True,
                "robot_placement": placement,
                "size": base_size,
                "volume": max(volume, 1.0),  # Ensure volume > small_threshold (0.06)
                "Pose": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "is_static": True,
            }
            full_placed_models.append(virtual_robot)
            print(f"[pipeline] Added virtual robot {placement['robot_id']} at {pos}")
    
    # Note: solve_small_object_placements is removed - Universal System handles all placement
    
    full_placed_models = validate_and_repair_layout(
        full_placed_models,
        room_half_size=room_half_size,
        semantic_plan=semantic_plan,
    )
    full_placed_models = repair_layout_by_constraints(
        full_placed_models, semantic_plan=semantic_plan,
        room_half_size=room_half_size,
    )

    # ---------------------------------------------------------------
    # Stage 5: MuJoCo assembly - Convert meshes using add_models
    # ---------------------------------------------------------------
    print("[pipeline] Stage 5: MuJoCo assembly - converting meshes...")
    
    # Convert .glb files to MuJoCo XML using add_models
    # This will convert meshes and create proper XML with <include> tags
    # We pass full_placed_models as pre_placed_models to preserve positions
    saved_models = interface.add_models(
        chosen_models=chosen_models,
        models=models_full,
        query=effective_query,
        path_to_save=world_path,  # ← FIX: Use world_path (file) instead of cache.cache_path (directory)
        world_path=world_path,
        room_half_size=room_half_size,
        pre_placed_models=full_placed_models,
        semantic_plan=semantic_plan,
    )
    
    print(f"[pipeline] ✓ Converted and assembled {len(saved_models)} models")

    # ---------------------------------------------------------------
    # Stage 5.5: Add robots to XML
    # ---------------------------------------------------------------
    import xml.etree.ElementTree as ET
    
    if robot_placements:
        print(f"[pipeline] Adding {len(robot_placements)} robot(s) to scene")
        # Load existing XML
        tree = ET.parse(world_path)
        root = tree.getroot()
        
        # Add each robot
        for placement in robot_placements:
            print(f"[pipeline] Adding robot: {placement['robot_id']}")
            interface.add_robot(root, placement)
        
        # Save updated XML
        tree.write(world_path, encoding="utf-8", xml_declaration=True)
        print(f"[pipeline] Added {len(robot_placements)} robot(s) to scene")
    else:
        print("[pipeline] No robots detected in query")

    # ---------------------------------------------------------------
    # Stage 6: Physics refinement
    # ---------------------------------------------------------------
    saved_models = refine_scene_with_engine(
        interface=interface,
        world_path=world_path,
        placed_models=saved_models,
        room_half_size=room_half_size,
    )

    # ---------------------------------------------------------------
    # Stage 6.5: Scene preview render
    # ---------------------------------------------------------------
    preview_path = interface.render_preview(world_path)
    if preview_path:
        print(f"[pipeline] Preview saved → {preview_path}")

    # ---------------------------------------------------------------
    # Stage 7: VLM validation loop (optional, enabled by flag)
    # ---------------------------------------------------------------
    if vlm_validation:
        from creator.scene.vlm_validator import validate_and_repair_loop
        saved_models = validate_and_repair_loop(
            saved_models,
            query=effective_query,
            prompt_model_fn=prompt_model,
            llm_model=chosen_model,
            room_half_size=room_half_size,
            world_path=world_path,
            max_iterations=max_vlm_iters,
            verbose=True,
        )
        # Re-assemble after VLM repair
        saved_models = interface.add_models(
            [{"Model": m.get("Model", m.get("name", ""))} for m in saved_models],
            models_full,
            effective_query,
            world_path,
            room_half_size=room_half_size,
        )

    db.insert({
        "id": str(uuid.uuid4()),
        "name": world_name,
        "filepath": world_path,
        "prompt": query,
        "expanded_query": effective_query,
        "room_type": scene_spec.room_type,
        "room_half_size": room_half_size,
        "total_models": json.dumps(saved_models, cls=NumpyEncoder),
        "world_name": "Empty",
    })

    print(f"[pipeline] Done. Scene saved to: {world_path}")
    return world_path
