#!/usr/bin/env python3
"""Просмотр сгенерированной MuJoCo сцены."""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")  # Используем glfw для интерактивного просмотра

import mujoco
import mujoco.viewer

# Путь к сцене (относительно папки project/)
scene_path = ".cache/worlds/scene_latest.xml"

if not os.path.exists(scene_path):
    print(f"❌ Файл сцены не найден: {scene_path}")
    print("Сначала запустите: python main_project.py")
    sys.exit(1)

print(f"Загрузка сцены: {scene_path}")

try:
    # Загрузить модель
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    print(f"✅ Модель загружена успешно")
    print(f"   Объектов в сцене: {model.nbody - 1}")  # -1 для world body
    print(f"   Размер комнаты: {model.stat.extent * 2:.1f}m")
    
    print("\n🎮 Управление:")
    print("   - Левая кнопка мыши: вращение камеры")
    print("   - Правая кнопка мыши: перемещение камеры")
    print("   - Колесико мыши: приближение/отдаление")
    print("   - ESC или закрыть окно: выход")
    print("\nЗапуск viewer...")
    
    # Запустить viewer
    mujoco.viewer.launch(model, data)
    
except Exception as e:
    print(f"❌ Ошибка при загрузке сцены: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
