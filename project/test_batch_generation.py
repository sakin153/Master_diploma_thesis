#!/usr/bin/env python3
"""Тест батчевой генерации графа сцены."""

import sys
sys.path.insert(0, '.')

from project.catalog import load_catalog
from project.prompt_expander import expand_prompt
from project.model_picker import pick_models
from project.model_loader import load_and_scale_models
from project.room_scaler import compute_room_half_size
from project.scene_planner import generate_placement_plan

# Загрузить каталог
catalog = load_catalog()

# Тестовый запрос: 6 столов, у каждого по 4 стула
query = "6 столов у каждого стола по 4 стуля."

print("="*60)
print("ТЕСТ БАТЧЕВОЙ ГЕНЕРАЦИИ ГРАФА СЦЕНЫ")
print("="*60)
print(f"Запрос: {query}")
print()

# Stage 0: Расширение промпта
print("[Stage 0] Расширение промпта...")
spec = expand_prompt(query)
print(f"  Тип комнаты: {spec.room_type}")
print(f"  Размер комнаты: {spec.room_half_size * 2}m x {spec.room_half_size * 2}m")
print(f"  Объектов: {len(spec.estimated_objects)}")
print()

# Stage 1: Выбор моделей
print("[Stage 1] Выбор моделей...")
models = pick_models(spec, catalog)
print(f"  Выбрано моделей: {len(models)}")
for m in models:
    print(f"    - {m.get('Model', 'unknown')}")
print()

# Stage 2: Загрузка и масштабирование
print("[Stage 2] Загрузка и масштабирование...")
models = load_and_scale_models(models)
print(f"  Загружено: {len(models)} моделей")
print()

# Stage 3: Вычисление размера комнаты
print("[Stage 3] Вычисление размера комнаты...")
room_half = compute_room_half_size(models, spec.room_type)
print(f"  Размер комнаты: {room_half * 2}m x {room_half * 2}m")
print()

# Stage 4: Генерация плана размещения (с батчевой генерацией графа)
print("[Stage 4] Генерация плана размещения...")
print("  (используется батчевая генерация графа)")
placement = generate_placement_plan(
    models, room_half, spec.room_type, spec.expanded_description, save_outputs=True
)

print()
print("="*60)
print("РЕЗУЛЬТАТ")
print("="*60)
print(f"Размещено объектов: {len(placement)}")
print()
print("Объекты:")
for i, obj in enumerate(placement, 1):
    pose = obj.get("Pose", {})
    print(
        f"  {i}. {obj.get('Model', 'unknown')}: "
        f"pos=({pose.get('x', 0):.2f}, {pose.get('y', 0):.2f}, {pose.get('z', 0):.2f}), "
        f"yaw={obj.get('yaw_deg', 0):.0f}°"
    )

print()
print("Проверьте файлы в project/output/ для деталей")
print("="*60)
