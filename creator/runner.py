import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tinydb import TinyDB

from creator.contexts_prompts.constraints import fmt_constraints_plan_tmpl
from creator.contexts_prompts.disambiguation import fmt_disambiguation_tmpl
from creator.model_databases.embodied_gen import EmbodiedGenLoader
from creator.placement import (
    build_semantic_plan,
    repair_layout_by_constraints,
    solve_floor_placements,
    solve_small_object_placements,
    solve_wall_placements,
    validate_and_repair_layout,
)
from creator.placement.plan import _stabilize_dense_group_constraints
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

    chosen_model = "qwen3-coder-next:cloud"

    loader = EmbodiedGenLoader(dataset_dir=assets_dir)
    interface = MujocoSimInterface(chosen_model, cache_dir=cache_dir)

    models, _worlds = loader.get_models()
    models_full = loader.get_models_full()
    print(f"[pipeline] Loaded {len(models)} models from EmbodiedGen dataset: {loader.dataset_dir}")

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
    # 3) For repeated occurrences of the same obj (e.g. "10 apples of
    #    different colors"), call the LLM once to anchor the primary pick,
    #    then round-robin through the remaining ranked candidates so the
    #    scene gets visual variety instead of N identical clones.
    chosen_models: List[Dict[str, str]] = []
    obj_state: Dict[str, Dict[str, Any]] = {}
    dropped_objects: List[str] = []
    used_uuids: set = set()
    for obj in objects:
        obj_raw = str(obj).strip()
        obj_key = obj_raw.lower()

        # Remove parentheses: "Sofa (blue)" -> "Sofa"
        obj_clean = re.sub(r'\([^)]*\)', '', obj_key).strip()

        # Apply fallback mapping
        search_key = _OBJECT_FALLBACKS.get(obj_clean, obj_clean)

        state = obj_state.get(search_key)
        if state is None:
            ranked = _rank_models_for_object(search_key, models, limit=10)
            if not ranked:
                print(f"[pipeline] WARN: '{obj}' has no catalog match, dropped")
                dropped_objects.append(obj_key)
                continue

            unused = [r for r in ranked if str(r.get("uuid") or "") not in used_uuids]
            candidate_pool = unused if unused else ranked

            if len(candidate_pool) == 1:
                primary = candidate_pool[0]
            else:
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

            state = {"order": [primary], "idx": 0}
            obj_state[search_key] = state

        picked = state["order"][0]
        name = str(picked.get("name", "")).strip()
        uid = str(picked.get("uuid", "")).strip()
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
    # Stage 4: Layout solving
    # ---------------------------------------------------------------
    full_placed_models = solve_floor_placements(
        full_placed_models=full_placed_models,
        semantic_plan=semantic_plan,
        room_half_size=room_half_size,
        grid_step=None,  # auto-adapt from min footprint
        yaw_candidates_deg=(0.0, 90.0, 180.0, 270.0),
        beam_width=12,
        seed=seed,
    )
    full_placed_models = solve_wall_placements(
        placed_models=full_placed_models,
        semantic_plan=semantic_plan,
        room_half_size=room_half_size,
    )
    
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
    
    full_placed_models = solve_small_object_placements(
        placed_models=full_placed_models,
        semantic_plan=semantic_plan,
        small_threshold_volume=0.06,
        seed=seed,
    )
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
    # Stage 5: MuJoCo assembly
    # ---------------------------------------------------------------
    world_name = "scene_latest"
    world_path = (
        os.path.join(cache.worlds_path, world_name) + interface.get_world_extension()
    )

    # Filter out virtual robots before add_models (they'll be added via add_robot)
    models_for_assembly = [m for m in full_placed_models if not m.get("is_robot", False)]
    
    # Build chosen_models list from models_for_assembly (ensures consistency)
    chosen_models_for_assembly = [
        {"Model": m.get("Model", m.get("name", "")), "uuid": m.get("uuid", "")}
        for m in models_for_assembly
    ]

    saved_models = interface.add_models(
        chosen_models_for_assembly,
        models_full,
        effective_query,
        world_path,
        room_half_size=room_half_size,
        pre_placed_models=models_for_assembly,
        semantic_plan=semantic_plan,
    )

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
