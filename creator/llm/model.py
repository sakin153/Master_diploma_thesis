import os
import sys
import sqlite3
from typing import Any, Dict, Optional
import requests

from creator.utils.json import parse_output_to_json

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-v3.1:671b-cloud"
OLLAMA_TIMEOUT_S = 120

CACHE_DB_PATH = "/var/tmp/ciare/.ollama_cache.sqlite3"


def _ensure_cache_db() -> None:
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    with sqlite3.connect(CACHE_DB_PATH) as conn:
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
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        row = conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ? AND model = ?",
            (cache_key, model),
        ).fetchone()
    return None if row is None else str(row[0])


def _cache_set(cache_key: str, model: str, response: str) -> None:
    _ensure_cache_db()
    with sqlite3.connect(CACHE_DB_PATH) as conn:
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
    url = f"{base_url}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except requests.exceptions.Timeout:
        raise TimeoutError("Timeout while waiting for Ollama response")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Failed to connect to Ollama at {base_url}. Is it running?"
        ) from e

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

    data: Dict[str, Any] = resp.json()
    message: Optional[Dict[str, Any]] = data.get("message")
    if not message or "content" not in message:
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return str(message["content"])


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
