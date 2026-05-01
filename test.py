import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

def plot_layout_from_json(json_data):
    """
    Принимает JSON-строку или список словарей от LLM и строит планировку.
    """
    # Если входные данные - строка, парсим JSON
    if isinstance(json_data, str):
        try:
            furniture_list = json.loads(json_data)
        except json.JSONDecodeError:
            print("Ошибка: Неверный JSON формат")
            return
    else:
        furniture_list = json_data

    # Проверка на ошибку от LLM
    if isinstance(furniture_list, dict) and "error" in furniture_list:
        print(f"Ошибка от планировщика: {furniture_list['error']}")
        return

    # Нормализация ключей: приводим все к виду 'w' и 'h' для удобства работы скрипта
    normalized_list = []
    for item in furniture_list:
        new_item = item.copy()
        # Если есть width, переименовываем в w
        if 'width' in new_item:
            new_item['w'] = new_item.pop('width')
        # Если есть height, переименовываем в h
        if 'height' in new_item:
            new_item['h'] = new_item.pop('height')
        
        # Гарантируем наличие ключей, если их нет ни в каком виде (на всякий случай)
        if 'w' not in new_item or 'h' not in new_item:
            continue # Пропускаем битые объекты
            
        normalized_list.append(new_item)
    
    furniture_list = normalized_list

    if not furniture_list:
        print("Нет объектов для отображения")
        return

    # Определяем границы комнаты на основе объектов
    # Используем ключи 'w' и 'h', так как мы их нормализовали выше
    max_x = max([item['x'] + item['w'] for item in furniture_list]) + 1.0
    max_y = max([item['y'] + item['h'] for item in furniture_list]) + 1.0
    
    # Задаем минимальный размер комнаты, чтобы не было слишком мелко
    room_w = max(5.0, max_x) 
    room_h = max(5.0, max_y)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    ax.set_xlim(0, room_w)
    ax.set_ylim(0, room_h)
    ax.set_aspect('equal')
    
    # Сетка
    ax.grid(True, linestyle='--', alpha=0.3)
    # Динамические тики сетки
    ax.set_xticks([i for i in range(0, int(room_w) + 1)])
    ax.set_yticks([i for i in range(0, int(room_h) + 1)])

    for item in furniture_list:
        x = item['x']
        y = item['y']
        w = item['w']
        h = item['h']
        color = item.get('color', '#CCCCCC')
        label = item.get('label', item.get('id', 'Object'))
        
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        
        # Текст (рисуем только если объект достаточно большой для текста)
        if w > 0.2 and h > 0.2:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=8, fontweight='bold')

    plt.title('Планировка сгенерирована AI')
    plt.xlabel('Ширина (метры)')
    plt.ylabel('Длина (метры)')
    plt.tight_layout()
    plt.show()

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---

llm_response = """
[
  {
    "id": "sofa_1",
    "label": "Диван",
    "x": 0.5,
    "y": 4.5,
    "w": 2.0,
    "h": 0.8,
    "color": "#3b82f6"
  },
  {
    "id": "table_1",
    "label": "Стол",
    "x": 1.0,
    "y": 2.5,
    "w": 1.0,
    "h": 0.6,
    "color": "#8b5cf6"
  },
  {
    "id": "chair_1",
    "label": "Стул",
    "x": 1.0,
    "y": 1.8,
    "w": 0.5,
    "h": 0.5,
    "color": "#10b981"
  },
  {
    "id": "chair_2",
    "label": "Стул",
    "x": 1.0,
    "y": 3.2,
    "w": 0.5,
    "h": 0.5,
    "color": "#10b981"
  },
  {
    "id": "chair_3",
    "label": "Стул",
    "x": 0.3,
    "y": 2.55,
    "w": 0.5,
    "h": 0.5,
    "color": "#10b981"
  },
  {
    "id": "chair_4",
    "label": "Стул",
    "x": 2.2,
    "y": 2.55,
    "w": 0.5,
    "h": 0.5,
    "color": "#10b981"
  },
  {
    "id": "vase_1",
    "label": "Ваза",
    "x": 1.35,
    "y": 2.75,
    "w": 0.2,
    "h": 0.2,
    "color": "#f43f5e"
  },
  {
    "id": "box_1",
    "label": "Коробка",
    "x": 1.6,
    "y": 2.75,
    "w": 0.2,
    "h": 0.2,
    "color": "#f59e0b"
  },
  {
    "id": "cabinet_1",
    "label": "Шкаф",
    "x": 0.5,
    "y": 0.5,
    "w": 2.0,
    "h": 0.6,
    "color": "#6366f1"
  }
]
"""

# Запуск визуализации
plot_layout_from_json(llm_response)