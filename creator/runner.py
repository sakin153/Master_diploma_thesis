import json
import os
import re
import uuid
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from tinydb import TinyDB

from creator.contexts_prompts.constraints import fmt_constraints_plan_tmpl
from creator.contexts_prompts.model import fmt_model_qa_tmpl
from creator.contexts_prompts.objects import fmt_objects_qa_tmpl
from creator.model_databases.local_assets import LocalAssetsLoader
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
from creator.xml.worlds import find_model

Simulator = Literal["mujoco"]

def _tokenize(text: str) -> List[str]:
    # Support both Latin and Cyrillic (Russian) characters
    return re.findall(r"[a-zа-яё0-9]+", (text or "").lower())


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

    if obj_l in name:
        score += 12
    if obj_l in meta:
        score += 6

    obj_tokens = [t for t in _tokenize(obj_l) if len(t) > 1]
    if not obj_tokens:
        return score

    for t in obj_tokens:
        if t in name:
            score += 4
        if t in meta:
            score += 2

    head = _singularize(obj_tokens[-1]) if obj_tokens else obj_l

    # Disambiguate common ambiguous nouns.
    if head in {"desk", "table"}:
        if "table" in name and head == "desk":
            score += 5
        if "desk" in name and head == "table":
            score += 3
        if any(k in name for k in ["lamp", "light", "fan", "desktop"]):
            score -= 18

    if head in {"whiteboard", "blackboard", "board"}:
        if any(k in name for k in ["whiteboard", "blackboard", "board"]):
            score += 5
        if any(k in name for k in ["surfboard", "skateboard", "snowboard"]):
            score -= 20

    if head == "chair" and "wheelchair" in name:
        score -= 12

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


def _objects_from_llm_output(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            if isinstance(item, str):
                if item.strip():
                    out.append(item.strip())
            elif isinstance(item, dict):
                v = item.get("Object") or item.get("object")
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
        return out
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    return []


def _normalize_chosen_models(raw: Any) -> List[Dict[str, str]]:
    if raw is None:
        return []
    out: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                model_name = item.get("Model") or item.get("model")
                if isinstance(model_name, str) and model_name.strip():
                    out.append({"Model": model_name.strip()})
            elif isinstance(item, str) and item.strip():
                out.append({"Model": item.strip()})
    elif isinstance(raw, dict):
        model_name = raw.get("Model") or raw.get("model")
        if isinstance(model_name, str) and model_name.strip():
            out.append({"Model": model_name.strip()})
    elif isinstance(raw, str) and raw.strip():
        out.append({"Model": raw.strip()})
    return out


def _noun_key(text: str) -> str:
    toks = _tokenize(text)
    if not toks:
        return ""
    return _singularize(toks[-1])

def _merge_objects_with_scene_hints(
    objects: Sequence[str],
    scene_spec: Any,
) -> List[str]:
    merged = [str(o).strip() for o in objects if str(o).strip()]

    hints = getattr(scene_spec, "estimated_objects", []) or []
    for h in hints:
        name = str(getattr(h, "name", "") or "").strip()
        if not name:
            continue
        qty = int(getattr(h, "quantity", 1) or 1)
        qty = max(1, min(qty, 20))
        for _ in range(qty):
            merged.append(name)

    return merged


def _model_matches_object(model_name: str, obj_name: str) -> bool:
    m_tokens = {_singularize(t) for t in _tokenize(model_name)}
    o_tokens = [_singularize(t) for t in _tokenize(obj_name)]
    if not m_tokens or not o_tokens:
        return False

    # Exact noun match on head token is strongest.
    if o_tokens[-1] in m_tokens:
        return True

    # Otherwise require at least one semantic token overlap.
    return any(t in m_tokens for t in o_tokens if len(t) > 2)


def _ensure_models_cover_objects(
    chosen_models: Sequence[Dict[str, str]],
    objects: Sequence[str],
    models_catalog: Sequence[Dict[str, Any]],
) -> List[Dict[str, str]]:
    out = [dict(m) for m in chosen_models if isinstance(m, dict) and m.get("Model")]
    if not objects:
        return out

    required_unique: List[str] = []
    seen_req: set = set()
    for obj in objects:
        key = _noun_key(obj) or str(obj).strip().lower()
        if not key or key in seen_req:
            continue
        seen_req.add(key)
        required_unique.append(str(obj))

    for obj in required_unique:
        covered = any(_model_matches_object(str(m.get("Model", "")), obj) for m in out)
        if covered:
            continue
        ranked = _rank_models_for_object(obj, models_catalog, limit=1)
        if not ranked:
            continue
        name = str(ranked[0].get("name", "")).strip()
        if not name:
            continue
        out.append({"Model": name})

    return out


def generate_world(
    *,
    simulator: Simulator,
    query: str,
    cache_dir: Optional[str] = None,
    vlm_validation: bool = True,   # enable VLM layout validation loop
    max_vlm_iters: int = 1,
    assets_dir: Optional[str] = None,
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

    chosen_model = "gpt-oss:120b-cloud"

    if simulator == "mujoco":
        loader = LocalAssetsLoader(assets_dir=assets_dir)
        interface = MujocoSimInterface(chosen_model, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unsupported simulator: {simulator}")

    models, _worlds = loader.get_models()
    models_full = loader.get_models_full()
    print(f"[pipeline] Loaded {len(models)} local assets from: {loader.assets_dir}")

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
    raw_objects = prompt_model(fmt_objects_qa_tmpl, effective_query, chosen_model)
    objects = _objects_from_llm_output(raw_objects)
    objects = _merge_objects_with_scene_hints(objects, scene_spec)
    if not objects:
        hints = [getattr(h, "name", "") for h in (scene_spec.estimated_objects or [])]
        objects = [h for h in hints if isinstance(h, str) and h.strip()] or [effective_query]

    candidates: List[Dict[str, Any]] = []
    for obj in objects:
        candidates.extend(_rank_models_for_object(obj, models, limit=10))
    if not candidates:
        candidates = _rank_models_for_object(effective_query, models, limit=40)

    context: List[Dict[str, Any]] = []
    seen_names: set = set()
    for m in models:
        name = m.get("name")
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        context.append({"name": name, "metadata": {
            "tags": m.get("tags"),
            "categories": m.get("categories"),
            "uuid": m.get("uuid"),
        }})

    content = fmt_model_qa_tmpl.format(context_str=context)
    chosen_models_raw = prompt_model(content, effective_query, chosen_model)
    chosen_models = _normalize_chosen_models(chosen_models_raw)

    filtered_models = [m for m in chosen_models if find_model(m["Model"], models)]
    chosen_models = _ensure_models_cover_objects(filtered_models, objects, models)

    # Last-resort fallback
    if not chosen_models:
        fallback_candidates = _rank_models_for_object(effective_query, models, limit=8)
        seen_fb: set = set()
        fallback: List[Dict[str, str]] = []
        for m in fallback_candidates:
            name = m.get("name")
            if isinstance(name, str) and name and name not in seen_fb:
                seen_fb.add(name)
                fallback.append({"Model": name})
                if len(fallback) >= 6:
                    break
        chosen_models = fallback

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
    )
    if isinstance(semantic_plan.get("objects"), list):
        semantic_plan["objects"] = _stabilize_dense_group_constraints(
            semantic_plan["objects"]
        )
        from creator.placement.plan import _infer_face_to_from_near
        semantic_plan["objects"] = _infer_face_to_from_near(semantic_plan["objects"])
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
        yaw_candidates_deg=(0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        beam_width=24,
    )
    full_placed_models = solve_wall_placements(
        placed_models=full_placed_models,
        semantic_plan=semantic_plan,
        room_half_size=room_half_size,
    )
    full_placed_models = solve_small_object_placements(
        placed_models=full_placed_models,
        semantic_plan=semantic_plan,
        small_threshold_volume=0.06,
    )
    full_placed_models = validate_and_repair_layout(
        full_placed_models, room_half_size=room_half_size,
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

    saved_models = interface.add_models(
        chosen_models,
        models_full,
        effective_query,
        world_path,
        room_half_size=room_half_size,
        pre_placed_models=full_placed_models,
        semantic_plan=semantic_plan,
    )

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
