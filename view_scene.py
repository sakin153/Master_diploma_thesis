#!/usr/bin/env python3
"""Просмотр сгенерированной MuJoCo сцены."""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")  # Используем glfw для интерактивного просмотра

import mujoco
import mujoco.viewer

# Путь к сцене - проверяем оба варианта (creator и project), относительно расположения скрипта
_repo_root = os.path.dirname(os.path.abspath(__file__))
creator_scene_path = os.path.join(_repo_root, "output/cache/worlds/scene_latest.xml")
project_scene_path = os.path.join(_repo_root, "project/.cache/worlds/scene_latest.xml")

# Проверяем, какая сцена существует и какая новее
scene_path = None
scene_source = None

if os.path.exists(creator_scene_path) and os.path.exists(project_scene_path):
    # Обе существуют - выбираем более новую
    creator_time = os.path.getmtime(creator_scene_path)
    project_time = os.path.getmtime(project_scene_path)
    
    if creator_time > project_time:
        scene_path = creator_scene_path
        scene_source = "creator (main.py)"
    else:
        scene_path = project_scene_path
        scene_source = "project (main_project.py)"
elif os.path.exists(creator_scene_path):
    scene_path = creator_scene_path
    scene_source = "creator (main.py)"
elif os.path.exists(project_scene_path):
    scene_path = project_scene_path
    scene_source = "project (main_project.py)"
else:
    print(f"❌ Файл сцены не найден ни в одной из директорий:")
    print(f"   - {creator_scene_path}")
    print(f"   - {project_scene_path}")
    print("\nСначала запустите:")
    print("   python main.py  (для creator)")
    print("   или")
    print("   python project/main_project.py  (для project)")
    sys.exit(1)

print(f"Загрузка сцены: {scene_path}")
print(f"Источник: {scene_source}")

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
