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

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _singularize(word: str) -> str:
    w = (word or "").strip().lower()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 2 and w.endswith("s"):
        return w[:-1]
    return w


def _requested_counts_from_query(query: str) -> Dict[str, int]:
    tokens = _tokenize(query)
    counts: Dict[str, int] = {}
    for i, t in enumerate(tokens[:-1]):
        n = None
        if t.isdigit():
            n = int(t)
        elif t in _NUMBER_WORDS:
            n = _NUMBER_WORDS[t]
        if n is None or n <= 1:
            continue
        noun = _singularize(tokens[i + 1])
        if noun:
            counts[noun] = max(counts.get(noun, 1), min(n, 10))
    return counts


def _expand_models_by_requested_counts(
    chosen_models: Sequence[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    counts = _requested_counts_from_query(query)
    if not counts:
        return list(chosen_models)

    requested_nouns = list(counts.keys())
    grouped: Dict[str, List[Dict[str, Any]]] = {noun: [] for noun in requested_nouns}
    unmatched: List[Dict[str, Any]] = []

    for m in chosen_models:
        name = str(m.get("Model", "")).lower()
        name_tokens = {_singularize(t) for t in _tokenize(name)}
        matched_noun = ""
        for noun in requested_nouns:
            if noun in name_tokens:
                matched_noun = noun
                break
        if matched_noun:
            grouped[matched_noun].append(dict(m))
        else:
            unmatched.append(dict(m))

    expanded: List[Dict[str, Any]] = []
    for noun in requested_nouns:
        target_count = counts[noun]
        pool = grouped.get(noun, [])
        if not pool:
            continue

        # Keep exact requested quantity for this noun.
        i = 0
        current = 0
        while current < target_count:
            expanded.append(dict(pool[i % len(pool)]))
            i += 1
            current += 1

    expanded.extend(unmatched)
    return expanded


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

    chosen_model = "deepseek-v3.1:671b-cloud"  # ignored by prompt_model(), kept for compatibility

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
    chosen_models_raw = prompt_model(content, query, chosen_model)
    chosen_models = _normalize_chosen_models(chosen_models_raw)

    filtered_models = []
    for model in chosen_models:
        if find_model(model["Model"], models):
            filtered_models.append(model)
    chosen_models = _expand_models_by_requested_counts(filtered_models, query)

    # Last-resort fallback: keep scene non-empty even when LLM model-picking fails.
    if not chosen_models:
        fallback_candidates = _rank_models_for_object(query, models, limit=8)
        if not fallback_candidates and objects:
            for obj in objects:
                fallback_candidates.extend(_rank_models_for_object(obj, models, limit=4))

        seen_fallback = set()
        fallback_models: List[Dict[str, str]] = []
        for m in fallback_candidates:
            name = m.get("name")
            if not isinstance(name, str) or not name:
                continue
            if name in seen_fallback:
                continue
            seen_fallback.add(name)
            fallback_models.append({"Model": name})
            if len(fallback_models) >= 6:
                break
        chosen_models = fallback_models

    if not chosen_models:
        raise RuntimeError(
            "No suitable models were found for the prompt. "
            "Try a more specific prompt (for example: 'office desk and chair')."
        )

    world_name = "scene_latest"
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
