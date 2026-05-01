import os
import re
import sys
import sqlite3
from typing import Any, Dict, Optional
import requests

from creator.utils.json import parse_output_to_json

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-v3.1:671b-cloud"  # Changed from qwen3-coder-next:cloud
OLLAMA_TIMEOUT_S = 180  # Increased for cloud models

_DEFAULT_CACHE_DB_PATH = "/var/tmp/ciare/.ollama_cache.sqlite3"


def _resolve_cache_db_path() -> str:
    """LLM cache path. Follows CIARE_CACHE_DIR env var when set so a custom
    CACHE_DIR (e.g. per-run /tmp/ciare_run_*) gets its own LLM cache instead
    of all runs sharing a single global SQLite at /var/tmp/ciare/."""
    custom = os.environ.get("CIARE_CACHE_DIR")
    if custom:
        return os.path.join(custom, ".ollama_cache.sqlite3")
    return _DEFAULT_CACHE_DB_PATH


def _ensure_cache_db() -> None:
    db_path = _resolve_cache_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT NOT NULL,
                model TEXT NOT NULL,
                response TEXT NOT NULL,
                PRIMARY KEY (cache_key, model)
            )
            """
        )


def _cache_get(cache_key: str, model: str) -> Optional[str]:
    _ensure_cache_db()
    with sqlite3.connect(_resolve_cache_db_path()) as conn:
        row = conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ? AND model = ?",
            (cache_key, model),
        ).fetchone()
    return None if row is None else str(row[0])


def _cache_set(cache_key: str, model: str, response: str) -> None:
    _ensure_cache_db()
    with sqlite3.connect(_resolve_cache_db_path()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache(cache_key, model, response) VALUES (?, ?, ?)",
            (cache_key, model, response),
        )


def _ollama_chat(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    timeout_s: int,
    base_url: str,
) -> str:
    import json as _json

    url = f"{base_url}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 2000,  # Increased to allow full JSON responses
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s, stream=True)
    except requests.exceptions.Timeout:
        raise TimeoutError("Timeout while waiting for Ollama response")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Failed to connect to Ollama at {base_url}. Is it running?"
        ) from e

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

    chunks: list = []
    token_count = 0
    max_tokens = 2500  # Safety limit to prevent infinite loops
    
    print("[llm] ", end="", flush=True)
    for line in resp.iter_lines():
        if not line:
            continue
        
        try:
            data = _json.loads(line)
        except _json.JSONDecodeError:
            print(f"\n[llm] WARNING: Failed to parse JSON: {line[:100]}")
            continue
            
        token = (data.get("message") or {}).get("content", "")
        if token:
            # Check if token looks like a prompt template (indicates model issue)
            if "<|im_start|>" in token or "<|im_end|>" in token:
                print("\n[llm] ERROR: Model is outputting prompt template!")
                print(f"[llm] Model '{model}' is not properly loaded.")
                print(f"[llm] Try: ollama pull {model}")
                raise RuntimeError(
                    f"Model '{model}' outputting prompt template"
                )
            
            print(token, end="", flush=True)
            chunks.append(token)
            token_count += 1
            
            # Safety check: stop if too many tokens
            if token_count > max_tokens:
                print("\n[llm] WARNING: Token limit reached, stopping generation")
                break
                
        if data.get("done"):
            break
    print()
    
    result = "".join(chunks)
    
    # Final check: if result contains prompt markers, it's invalid
    if "<|im_start|>" in result or "<|im_end|>" in result:
        raise RuntimeError(
            f"Model '{model}' returned invalid response with markers"
        )
    
    return result


def prompt_model(context: str, prompt: str, model: str = "gpt-3.5-turbo-16k"):
    try:
        # Hardcoded Ollama backend/model as requested.
        # Keep `model` argument for backward compatibility with callers.
        cache_key = repr(
            [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ]
        )
        cached_text = _cache_get(cache_key, OLLAMA_MODEL)
        if cached_text is not None:
            print("Using cached query result.")
            return parse_output_to_json(cached_text)

        print(
            f"[llm] calling {OLLAMA_MODEL} "
            f"(timeout={OLLAMA_TIMEOUT_S}s, "
            f"prompt_len={len(context)+len(prompt)})"
        )
        ans_text = _ollama_chat(
            system_prompt=context,
            user_prompt=prompt,
            model=OLLAMA_MODEL,
            timeout_s=OLLAMA_TIMEOUT_S,
            base_url=OLLAMA_BASE_URL,
        )

        _cache_set(cache_key, OLLAMA_MODEL, ans_text)
    except TimeoutError:
        print(
            "Timeout while waiting for model response."
            "Probably in your prompt there are too many models\n"
            "Re-run the script and adapt the prompt so that model will generate less models."
        )
        sys.exit(os.EX_UNAVAILABLE)
    return parse_output_to_json(ans_text)


def prompt_model_with_image(context: str, prompt: str, image_base64: str) -> Any:
    """Send a multimodal prompt with a rendered image to Ollama VLM.

    The model must support vision (e.g. deepseek-v3.1 multimodal or llava).
    The image should be a base64-encoded PNG/JPEG string.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt, "images": [image_base64]},
        ],
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_S)
    except requests.exceptions.Timeout:
        raise TimeoutError("Timeout waiting for Ollama multimodal response")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}") from e

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

    data: Dict[str, Any] = resp.json()
    message: Optional[Dict[str, Any]] = data.get("message")
    if not message or "content" not in message:
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return parse_output_to_json(str(message["content"]))


def ask_object_height_m(object_name: str) -> Optional[float]:
    """Ask the LLM for the real-world height of an object in meters.

    Used as fallback when the object category is not in the hardcoded table.
    Results are persisted in the SQLite cache (key prefixed with __height__).
    Returns None on failure so the caller can use its own fallback.
    
    Can be disabled via environment variable:
    CIARE_DISABLE_LLM_HEIGHT=1
    """
    name_clean = (object_name or "").strip().lower()
    if not name_clean:
        return None

    # Check if LLM height queries are disabled
    disable_llm = os.getenv("CIARE_DISABLE_LLM_HEIGHT", "0")
    if disable_llm.strip().lower() in {"1", "true", "yes", "on"}:
        return None  # Return None to use fallback value

    cache_key = f"__height__{name_clean}"
    cached = _cache_get(cache_key, OLLAMA_MODEL)
    if cached is not None:
        try:
            val = float(cached)
            if 0.01 < val < 10.0:
                return val
        except (ValueError, TypeError):
            pass

    system_prompt = (
        "You are a 3D scene assistant. "
        "Answer with ONLY a single decimal number in metres, no units, no text."
    )
    user_prompt = (
        f"Total height from floor to top (including backrest for chairs, lid for boxes) "
        f"in metres of a '{object_name}' (common household/furniture item). "
        f"Example answers: 0.75, 1.80, 0.12"
    )
    try:
        print(f"[llm] Querying height for '{object_name}'...")
        ans = _ollama_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=OLLAMA_MODEL,
            timeout_s=180,  # Increased for cloud models
            base_url=OLLAMA_BASE_URL,
        )
        
        # Extract only the first number from response
        nums = re.findall(r"\d+(?:\.\d+)?", ans.strip())
        if nums:
            val = float(nums[0])
            if 0.01 < val < 10.0:
                _cache_set(cache_key, OLLAMA_MODEL, str(val))
                print(f"[llm] Height for '{object_name}': {val}m (cached)")
                return val
        
        print(f"[llm] WARNING: Invalid response for '{object_name}': {ans[:100]}")
    except Exception as exc:
        print(f"[llm] height inference failed for '{object_name}': {exc}")

    return None
