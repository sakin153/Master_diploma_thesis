import pytest

from mujoco_scene_editor.llm.openrouter import (
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterConfig,
    build_default_headers,
    build_user_message,
    chat_completion,
    create_openrouter_client,
    load_openrouter_config,
)


def test_load_openrouter_config_requires_key():
    with pytest.raises(RuntimeError, match=r"OPENROUTER_API_KEY"):
        load_openrouter_config({})


def test_load_openrouter_config_defaults():
    cfg = load_openrouter_config({"OPENROUTER_API_KEY": "k"})
    assert cfg.api_key == "k"
    assert cfg.base_url == DEFAULT_OPENROUTER_BASE_URL
    assert cfg.model == DEFAULT_OPENROUTER_MODEL


def test_load_openrouter_config_reads_local_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENROUTER_API_KEY=test-key-from-dotenv",
                "OPENROUTER_MODEL=dotenv-model",
                "OPENROUTER_BASE_URL=https://example.invalid/v1",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    cfg = load_openrouter_config()
    assert cfg.api_key == "test-key-from-dotenv"
    assert cfg.model == "dotenv-model"
    assert cfg.base_url == "https://example.invalid/v1"


def test_load_openrouter_config_env_overrides_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENROUTER_API_KEY=dotenv-key",
                "OPENROUTER_MODEL=dotenv-model",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "env-model")

    cfg = load_openrouter_config()
    assert cfg.api_key == "env-key"
    assert cfg.model == "env-model"


def test_load_openrouter_config_rejects_placeholder_key(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=REPLACE_WITH_NEW_KEY\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match=r"OPENROUTER_API_KEY"):
        load_openrouter_config()


def test_build_user_message_merges_prefix_and_prompt():
    msg = build_user_message("  a ", " b  ")
    assert msg == {"role": "user", "content": "a\nb"}


def test_build_default_headers_only_includes_set_values():
    cfg = OpenRouterConfig(
        api_key="k",
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        model=DEFAULT_OPENROUTER_MODEL,
        http_referer=None,
        x_title=None,
    )
    assert build_default_headers(cfg) is None

    cfg2 = OpenRouterConfig(
        api_key="k",
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        model=DEFAULT_OPENROUTER_MODEL,
        http_referer="https://example.com",
        x_title="app",
    )
    assert build_default_headers(cfg2) == {
        "HTTP-Referer": "https://example.com",
        "X-Title": "app",
    }


def test_create_openrouter_client_passes_base_url_and_headers(monkeypatch):
    captured = {}

    class DummyOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("mujoco_scene_editor.llm.openrouter.OpenAI", DummyOpenAI)

    cfg = OpenRouterConfig(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model=DEFAULT_OPENROUTER_MODEL,
        http_referer="https://example.com",
        x_title="myapp",
    )
    client = create_openrouter_client(cfg)
    assert client is not None

    assert captured["api_key"] == "k"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["default_headers"] == {
        "HTTP-Referer": "https://example.com",
        "X-Title": "myapp",
    }


def test_chat_completion_fallbacks_when_primary_returns_no_choices(monkeypatch):
    calls = []

    class DummyResponse:
        def __init__(self, content):
            class _Message:
                def __init__(self, c):
                    self.content = c

            class _Choice:
                def __init__(self, c):
                    self.message = _Message(c)

            self.choices = [_Choice(content)]

    class DummyCompletions:
        def create(self, *, model, messages):
            calls.append(model)
            if model == "primary-model":
                class BadResponse:
                    choices = None

                return BadResponse()
            return DummyResponse("ok from fallback")

    class DummyClient:
        def __init__(self):
            class _Chat:
                completions = DummyCompletions()

            self.chat = _Chat()

    monkeypatch.setattr(
        "mujoco_scene_editor.llm.openrouter.create_openrouter_client",
        lambda _cfg: DummyClient(),
    )
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "fallback-model")
    monkeypatch.delenv("OPENROUTER_DISABLE_FALLBACK", raising=False)

    cfg = OpenRouterConfig(
        api_key="k",
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        model="primary-model",
    )

    out = chat_completion([
        {"role": "user", "content": "hi"},
    ], config=cfg)

    assert out == "ok from fallback"
    assert calls == ["primary-model", "fallback-model"]


def test_chat_completion_reports_all_attempts_when_all_models_fail(monkeypatch):
    class DummyCompletions:
        def create(self, *, model, messages):
            class BadResponse:
                choices = None

            return BadResponse()

    class DummyClient:
        def __init__(self):
            class _Chat:
                completions = DummyCompletions()

            self.chat = _Chat()

    monkeypatch.setattr(
        "mujoco_scene_editor.llm.openrouter.create_openrouter_client",
        lambda _cfg: DummyClient(),
    )
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "fallback-a,fallback-b")
    monkeypatch.delenv("OPENROUTER_DISABLE_FALLBACK", raising=False)

    cfg = OpenRouterConfig(
        api_key="k",
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        model="primary-model",
    )

    with pytest.raises(RuntimeError, match=r"Attempted: primary-model, fallback-a, fallback-b"):
        chat_completion([
            {"role": "user", "content": "hi"},
        ], config=cfg)


def test_chat_completion_fallbacks_on_insufficient_credits(monkeypatch):
    calls = []

    class InsufficientCreditsError(Exception):
        status_code = 402

    class DummyResponse:
        def __init__(self, content):
            class _Message:
                def __init__(self, c):
                    self.content = c

            class _Choice:
                def __init__(self, c):
                    self.message = _Message(c)

            self.choices = [_Choice(content)]

    class DummyCompletions:
        def create(self, *, model, messages):
            calls.append(model)
            if model == "paid-primary":
                raise InsufficientCreditsError("Insufficient credits")
            return DummyResponse("ok from free fallback")

    class DummyClient:
        def __init__(self):
            class _Chat:
                completions = DummyCompletions()

            self.chat = _Chat()

    monkeypatch.setattr(
        "mujoco_scene_editor.llm.openrouter.create_openrouter_client",
        lambda _cfg: DummyClient(),
    )
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "free-fallback")
    monkeypatch.delenv("OPENROUTER_DISABLE_FALLBACK", raising=False)

    cfg = OpenRouterConfig(
        api_key="k",
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        model="paid-primary",
    )

    out = chat_completion([
        {"role": "user", "content": "hi"},
    ], config=cfg)

    assert out == "ok from free fallback"
    assert calls == ["paid-primary", "free-fallback"]


def test_chat_completion_reports_insufficient_credits_when_all_candidates_fail(monkeypatch):
    class InsufficientCreditsError(Exception):
        status_code = 402

    class DummyCompletions:
        def create(self, *, model, messages):
            raise InsufficientCreditsError("Insufficient credits")

    class DummyClient:
        def __init__(self):
            class _Chat:
                completions = DummyCompletions()

            self.chat = _Chat()

    monkeypatch.setattr(
        "mujoco_scene_editor.llm.openrouter.create_openrouter_client",
        lambda _cfg: DummyClient(),
    )
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODELS", "fallback-a,fallback-b")
    monkeypatch.delenv("OPENROUTER_DISABLE_FALLBACK", raising=False)

    cfg = OpenRouterConfig(
        api_key="k",
        base_url=DEFAULT_OPENROUTER_BASE_URL,
        model="paid-primary",
    )

    with pytest.raises(RuntimeError, match=r"insufficient credits"):
        chat_completion([
            {"role": "user", "content": "hi"},
        ], config=cfg)
