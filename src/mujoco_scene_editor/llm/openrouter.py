from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from openai import NotFoundError, OpenAI

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# NOTE: OpenRouter model availability changes over time. If a model disappears,
# users can override via OPENROUTER_MODEL or mjprompt --model.
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
DEFAULT_OPENROUTER_FALLBACK_MODELS: tuple[str, ...] = (
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-120b:free",
)


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    base_url: str
    model: str
    http_referer: str | None = None
    x_title: str | None = None


def _read_simple_dotenv(path: Path) -> dict[str, str]:
    """Read a minimal .env file without external dependencies.

    Supports lines in the form `KEY=VALUE`, strips surrounding single/double
    quotes, and ignores empty lines/comments.
    """

    out: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return out

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        out[key] = value

    return out


def _is_placeholder_api_key(value: str | None) -> bool:
    if not value:
        return True
    normalized = value.strip().lower()
    return (
        normalized in {"replace_with_new_key", "changeme", "your_key_here"}
        or normalized.startswith("replace_with_")
    )


def load_openrouter_config(environ: Mapping[str, str] | None = None) -> OpenRouterConfig:
    if environ is None:
        merged_env: dict[str, str] = dict(os.environ)
        dotenv_values = _read_simple_dotenv(Path.cwd() / ".env")
        # Real process environment wins over .env values.
        for key, value in dotenv_values.items():
            merged_env.setdefault(key, value)
        env: Mapping[str, str] = merged_env
    else:
        env = environ

    api_key = env.get("OPENROUTER_API_KEY")
    if _is_placeholder_api_key(api_key):
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY. Please export it with export OPENROUTER_API_KEY=... and retry."
        )

    base_url = env.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
    model = env.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)

    http_referer = env.get("OPENROUTER_HTTP_REFERER") or None
    x_title = env.get("OPENROUTER_X_TITLE") or None

    return OpenRouterConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        http_referer=http_referer,
        x_title=x_title,
    )


def build_user_message(prefix: str, prompt: str) -> dict:
    merged = f"{prefix.strip()}\n{prompt.strip()}".strip()
    return {"role": "user", "content": merged}


def build_default_headers(config: OpenRouterConfig) -> dict[str, str] | None:
    headers: dict[str, str] = {}
    if config.http_referer:
        headers["HTTP-Referer"] = config.http_referer
    if config.x_title:
        headers["X-Title"] = config.x_title
    return headers or None


def create_openrouter_client(config: OpenRouterConfig) -> OpenAI:
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        default_headers=build_default_headers(config),
    )


def _load_fallback_models(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    env = os.environ if environ is None else environ
    raw = (env.get("OPENROUTER_FALLBACK_MODELS") or "").strip()
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
    else:
        models = list(DEFAULT_OPENROUTER_FALLBACK_MODELS)

    # Dedupe while preserving order.
    return tuple(dict.fromkeys(models))


def _iter_model_candidates(primary: str, fallbacks: Iterable[str]) -> list[str]:
    out = [primary]
    for m in fallbacks:
        if m and m not in out:
            out.append(m)
    return out


def _is_retryable_provider_error(err: Exception) -> bool:
    code = getattr(err, "status_code", None)
    if code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    text = str(err).lower()
    retry_hints = (
        "temporarily rate-limited",
        "provider returned error",
        "timeout",
        "service unavailable",
        "too many requests",
    )
    return any(hint in text for hint in retry_hints)


def _is_fallback_worthy_provider_error(err: Exception) -> bool:
    """Return True for non-transient errors where trying another model helps."""

    code = getattr(err, "status_code", None)
    if code == 402:
        return True

    text = str(err).lower()
    fallback_hints = (
        "insufficient credits",
        "payment required",
        "never purchased credits",
        "billing",
    )
    return any(hint in text for hint in fallback_hints)


def _error_summary(err: Exception) -> str:
    code = getattr(err, "status_code", None)
    if code is not None:
        return f"HTTP {code}: {err}"
    return str(err) or err.__class__.__name__


def _extract_reset_epoch_ms(text: str) -> int | None:
    # Example from provider payload:
    # 'X-RateLimit-Reset': '1775779200000'
    m = re.search(r"X-RateLimit-Reset'\s*:\s*'?(\d{10,13})'?", text)
    if not m:
        return None
    try:
        value = int(m.group(1))
    except ValueError:
        return None
    if value < 10_000_000_000:
        value *= 1000
    return value


def _format_epoch_ms_utc(epoch_ms: int | None) -> str | None:
    if epoch_ms is None:
        return None
    try:
        dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def chat_completion(
    messages: Iterable[Mapping[str, Any]],
    *,
    model: str | None = None,
    config: OpenRouterConfig | None = None,
) -> str:
    cfg = load_openrouter_config() if config is None else config
    client = create_openrouter_client(cfg)

    chosen_model = model or cfg.model

    def _coerce_content_to_text(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        # Some providers may return content parts.
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str) and text:
                        parts.append(text)
            merged = "".join(parts).strip()
            return merged or None
        if isinstance(value, dict):
            text = value.get("text") or value.get("content")
            if isinstance(text, str):
                return text
        return None

    allow_fallback = os.environ.get("OPENROUTER_DISABLE_FALLBACK") != "1"
    candidates = [chosen_model]
    if allow_fallback:
        candidates = _iter_model_candidates(chosen_model, _load_fallback_models())

    attempt_errors: list[str] = []
    saw_free_per_day_limit = False
    saw_free_per_min_limit = False
    saw_insufficient_credits = False
    rate_limit_resets_ms: list[int] = []
    payload = list(messages)

    for current_model in candidates:
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=payload,
            )
        except NotFoundError as e:
            # OpenRouter returns 404 when the requested model has no routable endpoints.
            attempt_errors.append(f"{current_model}: no endpoints (404)")
            continue
        except Exception as e:
            if _is_retryable_provider_error(e) or _is_fallback_worthy_provider_error(e):
                text = str(e).lower()
                if "free-models-per-day" in text:
                    saw_free_per_day_limit = True
                if "free-models-per-min" in text:
                    saw_free_per_min_limit = True
                if "insufficient credits" in text or "payment required" in text:
                    saw_insufficient_credits = True

                reset_ms = _extract_reset_epoch_ms(str(e))
                if reset_ms is not None:
                    rate_limit_resets_ms.append(reset_ms)

                attempt_errors.append(f"{current_model}: {_error_summary(e)}")
                continue
            raise RuntimeError(
                f"OpenRouter request failed for model '{current_model}': {_error_summary(e)}"
            ) from e

        choices = getattr(response, "choices", None)
        if not choices:
            attempt_errors.append(f"{current_model}: no choices in response")
            continue

        first = choices[0]
        message = getattr(first, "message", None)
        if message is None:
            attempt_errors.append(f"{current_model}: completion has no message")
            continue

        raw_content = getattr(message, "content", None)
        content = _coerce_content_to_text(raw_content)
        if content is None:
            attempt_errors.append(f"{current_model}: empty/non-text content")
            continue

        return content

    attempted = ", ".join(candidates)
    details = "; ".join(attempt_errors) if attempt_errors else "unknown error"

    if saw_free_per_day_limit or saw_free_per_min_limit:
        hints: list[str] = []
        if saw_free_per_day_limit:
            hints.append("daily free-tier limit reached")
        if saw_free_per_min_limit:
            hints.append("per-minute free-tier limit reached")

        reset_hint = None
        if rate_limit_resets_ms:
            reset_hint = _format_epoch_ms_utc(min(rate_limit_resets_ms))

        hint_text = ", ".join(hints) if hints else "free-tier rate limit reached"
        reset_text = f" Earliest reset: {reset_hint}." if reset_hint else ""

        raise RuntimeError(
            "OpenRouter free-model rate limit exceeded "
            f"({hint_text}).{reset_text} "
            f"Attempted: {attempted}. "
            "Use a paid/BYOK model via OPENROUTER_MODEL or wait for reset. "
            f"Details: {details}"
        )

    if saw_insufficient_credits:
        raise RuntimeError(
            "OpenRouter paid-model access failed due to insufficient credits. "
            f"Attempted: {attempted}. "
            "Use a funded account/key, switch OPENROUTER_MODEL to a free model, "
            "or wait for free-tier reset if your account is rate-limited. "
            f"Details: {details}"
        )

    raise RuntimeError(
        "OpenRouter did not return usable output for any candidate model. "
        f"Attempted: {attempted}. Details: {details}. "
        "Set OPENROUTER_MODEL/--model to a working model, or customize OPENROUTER_FALLBACK_MODELS."
    )


def query_openrouter(
    prefix: str,
    prompt: str,
    *,
    model: str | None = None,
    config: OpenRouterConfig | None = None,
) -> str:
    cfg = load_openrouter_config() if config is None else config
    return chat_completion(
        [build_user_message(prefix, prompt)],
        model=model or cfg.model,
        config=cfg,
    )
