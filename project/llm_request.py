# Базовая универсальная функция для отправки запроса в LLM.
import requests
def ask_llm(prompt: str, system_prompt: str, model: str, timeout: int, base_url: str) -> str:
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 0,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    except requests.exceptions.Timeout:
            raise TimeoutError("Timeout while waiting for Ollama response")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Failed to connect to LLM server at {base_url}. Is it running ma boy?"
        ) from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"LLM server error {e.response.status_code}: {e.response.text}") from e
    