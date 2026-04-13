from __future__ import annotations

import json
import os
import re
from typing import List

from mujoco_scene_editor.llm.openrouter import chat_completion


_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "with",
    "on",
    "in",
    "of",
    "to",
    "for",
    "at",
    "by",
    "from",
    "room",
    "dining",
    "living",
    "kitchen",
    "bedroom",
    "office",
    "detailed",
    "scene",
    "robot",
}


def _parse_keywords_json(content: str) -> List[str]:
    # Try direct JSON first.
    try:
        data = json.loads(content)
    except Exception:
        # Then try extracting the first JSON array from free-form text.
        m = re.search(r"\[[\s\S]*?\]", content)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []

    if not isinstance(data, list):
        return []

    out: List[str] = []
    for item in data:
        if not isinstance(item, str):
            continue
        kw = item.strip().lower()
        if not kw:
            continue
        if kw not in out:
            out.append(kw)
        if len(out) >= 6:
            break
    return out


def _heuristic_keywords(scene_prompt: str, limit: int = 6) -> List[str]:
    text = scene_prompt.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    parts = [p for p in text.split() if p and not p.isdigit()]

    # Light normalization (plural stripping).
    tokens: List[str] = []
    for p in parts:
        if p.endswith("ies") and len(p) > 4:
            p = p[:-3] + "y"
        elif p.endswith("es") and len(p) > 3:
            p = p[:-2]
        elif p.endswith("s") and len(p) > 3:
            p = p[:-1]
        if len(p) < 3:
            continue
        if p in _STOPWORDS:
            continue
        if p not in tokens:
            tokens.append(p)

    preferred = [
        "table",
        "chair",
        "sofa",
        "couch",
        "bed",
        "laptop",
        "computer",
        "mug",
        "cup",
        "vase",
        "plant",
        "book",
        "bottle",
        "lamp",
    ]
    out: List[str] = []
    for p in preferred:
        if p in tokens and p not in out:
            out.append(p)
        if len(out) >= limit:
            return out
    for t in tokens:
        if t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def extract_objaverse_keywords(scene_prompt: str) -> List[str]:
    """Extract a small set of English object/category keywords from the user prompt.

    This is used to map a free-form prompt (possibly non-English) to Objaverse LVIS labels.
    Returns an empty list if extraction fails.
    """

    # If you want to completely avoid the extra LLM call, enable this.
    # Note: this is heuristic-based and will influence which Objaverse meshes are used.
    if os.environ.get("MJPROMPT_OBJAVERSE_NO_LLM") == "1":
        return _heuristic_keywords(scene_prompt)

    messages = [
        {
            "role": "system",
            "content": (
                "You extract object category keywords for 3D asset search. "
                "Return ONLY a JSON array of 1 to 6 short English keywords (strings). "
                "No prose, no markdown, no code fences."
            ),
        },
        {
            "role": "user",
            "content": (
                "Prompt:\n" + scene_prompt.strip() + "\n\n" "JSON array of keywords:"
            ),
        },
    ]

    try:
        content = chat_completion(messages)
    except Exception:
        # By default, don't fall back to heuristics (keeps generation unaffected).
        if os.environ.get("MJPROMPT_OBJAVERSE_KEYWORDS_FALLBACK") == "1":
            return _heuristic_keywords(scene_prompt)
        return []

    parsed = _parse_keywords_json(content)
    if parsed:
        return parsed

    if os.environ.get("MJPROMPT_OBJAVERSE_KEYWORDS_FALLBACK") == "1":
        return _heuristic_keywords(scene_prompt)
    return []
