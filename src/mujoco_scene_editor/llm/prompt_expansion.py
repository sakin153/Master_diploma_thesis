from __future__ import annotations

from mujoco_scene_editor.llm.openrouter import query_openrouter

_EXPAND_PREFIX = (
    "You are a scene design assistant for robot manipulation environments.\n"
    "Expand the given brief scene description into a detailed, specific description "
    "suitable for generating a physics simulation.\n"
    "Mention: specific object types, their materials, approximate real-world sizes, "
    "spatial relationships (what is on/in/next_to what), and anything mounted on walls.\n"
    "Keep it to 3-5 sentences. Output only the expanded description, no preamble.\n"
    "\nBrief description:\n"
)

_MIN_WORDS_TO_SKIP = 20


def expand_prompt(prompt: str, *, model: str | None = None) -> str:
    """Expand a short scene prompt into a detailed description via LLM.

    If the prompt is already detailed (>= 20 words) it is returned unchanged.
    If the LLM call fails the original prompt is returned as a safe fallback.
    """
    if len(prompt.split()) >= _MIN_WORDS_TO_SKIP:
        return prompt
    try:
        expanded = query_openrouter(_EXPAND_PREFIX, prompt, model=model)
        return expanded.strip() or prompt
    except Exception:
        return prompt
