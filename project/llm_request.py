import base64
import json
import re
import sys

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-coder-next:cloud"
DEFAULT_TIMEOUT_S = 180


def _ollama_chat(system, user, model, timeout_s, base_url, images=None):
    url = f"{base_url}/api/chat"

    # Формируем сообщения
    messages = [{"role": "system", "content": system}]

    # Если есть изображения, добавляем их в user message
    if images:
        user_message = {"role": "user", "content": user, "images": []}
        for img_path in images:
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
                user_message["images"].append(img_base64)
        messages.append(user_message)
    else:
        messages.append({"role": "user", "content": user})

    payload = {
        "model": model,
        "stream": True,
        "messages": messages,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 4000,  # Увеличено для больших JSON
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s, stream=True)
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama не ответил за {timeout_s}s")
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(f"Нет соединения с Ollama на {base_url}. Запущен?") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text}")

    chunks = []
    token_count = 0
    print("[llm] ", end="", flush=True)

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        token = (data.get("message") or {}).get("content", "")
        if token:
            # модель выдаёт маркеры шаблона — значит она не загружена
            if "<|im_start|>" in token or "<|im_end|>" in token:
                raise RuntimeError(
                    f"Модель '{model}' выдаёт маркеры шаблона — "
                    f"запусти: ollama pull {model}"
                )
            print(token, end="", flush=True)
            chunks.append(token)
            token_count += 1
            if token_count > 5000:  # защита от бесконечной генерации (увеличено)
                print("\n[llm] WARNING: лимит токенов достигнут, останавливаем")
                break

        if data.get("done"):
            break

    print()
    result = "".join(chunks)

    if "<|im_start|>" in result or "<|im_end|>" in result:
        raise RuntimeError(f"Модель '{model}' вернула ответ с маркерами шаблона")

    return result


def _sanitize_json_string(json_str):
    """Очистить JSON от проблемных символов.
    
    Заменяет китайские символы в ключах JSON на правильные английские слова.
    Исправляет пробелы в идентификаторах.
    """
    # Словарь замен для известных проблем
    replacements = {
        '"anchor_h极"': '"anchor_hint"',
        '"极"': '"facing"',
        '极': 'facing',
    }
    
    result = json_str
    for bad, good in replacements.items():
        result = result.replace(bad, good)
    
    # Исправить пробелы в id полях: "computer chair_inst1" -> "computer_chair_inst1"
    # Паттерн: "id": "слово пробел слово..."
    import re
    
    def fix_id_spaces(match):
        """Заменить пробелы на подчеркивания в значениях id."""
        full_match = match.group(0)
        id_value = match.group(1)
        fixed_value = id_value.replace(' ', '_')
        return full_match.replace(id_value, fixed_value)
    
    # Найти все "id": "значение с пробелами"
    result = re.sub(r'"id"\s*:\s*"([^"]+\s+[^"]+)"', fix_id_spaces, result)
    
    # Также исправить в "reference": "значение с пробелами"
    result = re.sub(r'"reference"\s*:\s*"([^"]+\s+[^"]+)"', fix_id_spaces, result)
    
    # Исправить в "depends_on": "значение с пробелами"
    result = re.sub(r'"depends_on"\s*:\s*"([^"]+\s+[^"]+)"', fix_id_spaces, result)
    
    return result


def _parse_json(text):
    """Парсит JSON из ответа LLM с улучшенной обработкой ошибок."""
    # вырезаем JSON из ```json ... ``` если LLM обернул его в markdown
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end == -1:  # Если закрывающий ``` не найден
            json_str = text[start:].strip()
        else:
            json_str = text[start:end].strip()
    elif "```" in text:
        # Попробуем найти JSON без явного маркера json
        start = text.find("```") + len("```")
        end = text.find("```", start)
        if end == -1:
            json_str = text[start:].strip()
        else:
            json_str = text[start:end].strip()
    else:
        json_str = text.strip()

    # Попытка найти JSON объект/массив в тексте
    if not json_str.startswith(("{", "[")):
        # Ищем первый { или [
        match = re.search(r"[\{\[]", json_str)
        if match:
            json_str = json_str[match.start():]

    # Убираем текст после закрывающей скобки
    if json_str.startswith("{"):
        # Найти последнюю закрывающую }
        last_brace = json_str.rfind("}")
        if last_brace != -1:
            json_str = json_str[:last_brace + 1]
    elif json_str.startswith("["):
        # Найти последнюю закрывающую ]
        last_bracket = json_str.rfind("]")
        if last_bracket != -1:
            json_str = json_str[:last_bracket + 1]

    # НЕ удаляем не-ASCII символы - они могут быть частью валидного JSON

    # Очистить от известных проблемных символов (китайские символы в ключах)
    json_str = _sanitize_json_string(json_str)

    # Чиним слепленные объекты (если есть)
    json_str = re.sub(r"\}\s*[^,\[\]\{\}]+\s*\{", "},{", json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        # Сохраняем проблемный JSON для отладки
        print("[llm] ОШИБКА парсинга JSON:")
        print(f"[llm] Первые 500 символов: {json_str[:500]}")
        print(f"[llm] Последние 500 символов: {json_str[-500:]}")
        print(f"[llm] Ошибка: {exc}")
        raise RuntimeError(f"Не удалось распарсить JSON из ответа LLM: {exc}") from exc

# Публичная функция для запроса к LLM
def request(
    system,
    user,
    model=DEFAULT_MODEL,
    base_url=OLLAMA_BASE_URL,
    timeout_s=DEFAULT_TIMEOUT_S,
    images=None
):
    print(f"[llm] Запрос к {model} (timeout={timeout_s}s)")
    try:
        text = _ollama_chat(system, user, model, timeout_s, base_url, images)
    except TimeoutError:
        print("[llm] Таймаут. Уменьши промпт или увеличь timeout_s.")
        sys.exit(1)

    return _parse_json(text)
