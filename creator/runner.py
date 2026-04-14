import json
import os
import re
import uuid
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from tinydb import TinyDB

from creator.contexts_prompts.model import fmt_model_qa_tmpl
from creator.contexts_prompts.objects import fmt_objects_qa_tmpl
from creator.model_databases.objaverse import ObjaverseLoader
from creator.sim_interfaces.mujoco import MujocoSimInterface
from creator.utils.cache import Cache
from creator.utils.json import NumpyEncoder
from creator.xml.worlds import find_model

Simulator = Literal["mujoco"]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


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
        if s > 0:
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
        # De-dupe preserving order
        deduped: List[str] = []
        seen = set()
        for o in out:
            ol = o.lower()
            if ol in seen:
                continue
            seen.add(ol)
            deduped.append(o)
        return deduped
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    return []


def generate_world(
    *,
    simulator: Simulator,
    query: str,
    cache_dir: Optional[str] = None,
) -> str:
    """Generate a world without Click/questionary orchestration.

    Notes:
    - LLM calls go through creator.llm.model.prompt_model (Ollama localhost).
        - Model search is local string scoring over model name/tags/categories.
    """

    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    cache = Cache(cache=cache_dir)
    if cache.models_and_worlds_initialized():
        cache.init_models_and_worlds()
    db = TinyDB(os.path.join(cache.worlds_path, "world_db.json"))

    from creator.llm.model import prompt_model

    chosen_model = "gpt-oss:120b-cloud"  # ignored by prompt_model(), kept for compatibility

    if simulator == "mujoco":
        loader = ObjaverseLoader()
        interface = MujocoSimInterface(chosen_model)
    else:
        raise ValueError(f"Unsupported simulator: {simulator}")

    models, _worlds = loader.get_models()

    # 1) LLM: extract objects for the scene
    raw_objects = prompt_model(fmt_objects_qa_tmpl, query, chosen_model)
    objects = _objects_from_llm_output(raw_objects)
    if not objects:
        # Fallback: if the model returns something unexpected,
        # at least search by the full query
        objects = [query]

    # 2) Search models for each object, build context
    candidates: List[Dict[str, Any]] = []
    for obj in objects:
        candidates.extend(_rank_models_for_object(obj, models, limit=10))

    # If search returned nothing (rare), fallback to query search
    if not candidates:
        candidates = _rank_models_for_object(query, models, limit=40)

    context: List[Dict[str, Any]] = []
    seen_names = set()
    for m in candidates:
        name = m.get("name")
        if not name:
            continue
        if name in seen_names:
            continue
        seen_names.add(name)
        metadata = {
            "tags": m.get("tags"),
            "categories": m.get("categories"),
        }
        if m.get("uuid"):
            metadata["uuid"] = m.get("uuid")
        context.append({"name": name, "metadata": metadata})

    template_world_path = os.path.join(cache.worlds_path, "empty.sdf")

    content = fmt_model_qa_tmpl.format(context_str=context)
    chosen_models = prompt_model(content, query, chosen_model)

    filtered_models = []
    for model in chosen_models:
        if find_model(model["Model"], models):
            filtered_models.append(model)
    chosen_models = filtered_models

    cleaned_query = re.sub(r"[<>:;.\,\"/\\|?*]", "", query).strip()
    world_name = f"world_{cleaned_query.replace(' ', '_')}"
    world_path = (
        os.path.join(cache.worlds_path, world_name) + interface.get_world_extension()
    )

    saved_models = interface.add_models(
        chosen_models,
        loader.get_models_full(),
        query,
        world_path,
        template_world_path,
    )

    db.insert(
        {
            "id": str(uuid.uuid4()),
            "name": world_name,
            "filepath": world_path,
            "prompt": query,
            "total_models": json.dumps(saved_models, cls=NumpyEncoder),
            "world_name": "Empty",
        }
    )

    return world_path
