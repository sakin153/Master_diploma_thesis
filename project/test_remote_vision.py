"""
Тестовый запрос к удаленному Ollama серверу с мультимодальной моделью.
"""

import base64
import json
import requests

REMOTE_OLLAMA_URL = "http://94.19.29.186:11434"


def test_remote_vision_model(
    image_paths,
    model="minicpm-v:8b",
    prompt="Describe what you see in these images."
):
    """
    Тестирует мультимодальную модель на удаленном сервере.
    
    Args:
        image_paths: список путей к изображениям
        model: название модели
        prompt: текстовый промпт
    """
    url = f"{REMOTE_OLLAMA_URL}/api/chat"
    
    # Конвертируем изображения в base64
    images_base64 = []
    for img_path in image_paths:
        print(f"[test] Загружаем изображение: {img_path}")
        with open(img_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
            images_base64.append(img_base64)
    
    # Формируем запрос
    payload = {
        "model": model,
        "stream": False,  # Без стриминга для простоты
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_base64
            }
        ],
        "options": {
            "temperature": 0,
            "seed": 42
        }
    }
    
    print(f"[test] Отправляем запрос к {REMOTE_OLLAMA_URL}")
    print(f"[test] Модель: {model}")
    print(f"[test] Количество изображений: {len(images_base64)}")
    print(f"[test] Промпт: {prompt}")
    print()
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code != 200:
            print(f"[test] ОШИБКА: HTTP {response.status_code}")
            print(f"[test] Ответ: {response.text}")
            return None
        
        result = response.json()
        content = result.get("message", {}).get("content", "")
        
        print("="*60)
        print("ОТВЕТ МОДЕЛИ:")
        print("="*60)
        print(content)
        print("="*60)
        
        return content
        
    except requests.exceptions.Timeout:
        print("[test] ОШИБКА: Таймаут (сервер не ответил за 120 секунд)")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"[test] ОШИБКА: Не удалось подключиться к {REMOTE_OLLAMA_URL}")
        print(f"[test] {e}")
        return None
    except Exception as e:
        print(f"[test] ОШИБКА: {e}")
        return None


def test_list_models():
    """Проверяет список доступных моделей на удаленном сервере."""
    url = f"{REMOTE_OLLAMA_URL}/api/tags"
    
    print(f"[test] Запрашиваем список моделей с {REMOTE_OLLAMA_URL}")
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"[test] ОШИБКА: HTTP {response.status_code}")
            return None
        
        result = response.json()
        models = result.get("models", [])
        
        print("="*60)
        print("ДОСТУПНЫЕ МОДЕЛИ:")
        print("="*60)
        for model in models:
            name = model.get("name", "unknown")
            size = model.get("size", 0) / (1024**3)  # В GB
            print(f"  - {name} ({size:.1f} GB)")
        print("="*60)
        
        return models
        
    except Exception as e:
        print(f"[test] ОШИБКА: {e}")
        return None


if __name__ == "__main__":
    print("="*60)
    print("ТЕСТ УДАЛЕННОГО OLLAMA СЕРВЕРА")
    print("="*60)
    print()
    
    # Сначала проверяем доступные модели
    print("ШАГ 1: Проверка доступных моделей")
    print()
    models = test_list_models()
    print()
    
    if not models:
        print("[test] Не удалось получить список моделей.")
        print("[test] Проверьте, что сервер доступен:")
        print(f"[test]   curl {REMOTE_OLLAMA_URL}/api/tags")
        exit(1)
    
    # Проверяем, есть ли мультимодальные модели
    vision_models = [
        m["name"] for m in models
        if any(x in m["name"].lower()
               for x in ["vision", "minicpm", "llava", "qwen"])
    ]
    
    if not vision_models:
        print("[test] ВНИМАНИЕ: Не найдено мультимодальных моделей!")
        print("[test] Попробуем с первой доступной моделью...")
        test_model = models[0]["name"] if models else "minicpm-v:8b"
    else:
        test_model = vision_models[0]
        print(f"[test] Найдена мультимодальная модель: {test_model}")
    
    print()
    print("="*60)
    print("ШАГ 2: Тест с изображениями стула")
    print("="*60)
    print()
    
    # Тестируем с рендерами стула
    image_paths = [
        ".cache/orientation/computer_chair_yaw0.png",
        ".cache/orientation/computer_chair_yaw90.png",
        ".cache/orientation/computer_chair_yaw180.png",
        ".cache/orientation/computer_chair_yaw270.png",
    ]
    
    prompt = """Analyze these 4 views of a chair model:
- Image 1: yaw=0° (default orientation)
- Image 2: yaw=90° (rotated 90° clockwise)
- Image 3: yaw=180° (rotated 180°)
- Image 4: yaw=270° (rotated 270° clockwise)

Which view shows the FRONT of the chair (where a person sits)?
Respond in JSON format:
{
  "front_yaw": <0, 90, 180, or 270>,
  "confidence": "<high, medium, or low>",
  "reasoning": "<brief explanation>"
}"""
    
    result = test_remote_vision_model(image_paths, test_model, prompt)
    
    if result:
        print()
        print("[test] УСПЕХ! Удаленный сервер работает.")
    else:
        print()
        print("[test] ОШИБКА: Не удалось получить ответ от сервера.")
