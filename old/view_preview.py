#!/usr/bin/env python3
"""Показать превью изображение сцены."""

import os
import sys
import subprocess

preview_path = "project/.cache/worlds/scene_latest_preview.png"

if not os.path.exists(preview_path):
    print(f"❌ Превью не найдено: {preview_path}")
    print("Сначала запустите: python project/main_project.py")
    sys.exit(1)

print(f"Открытие превью: {preview_path}")

# Попробовать разные способы открыть изображение
try:
    # Linux
    if sys.platform.startswith('linux'):
        subprocess.run(['xdg-open', preview_path])
    # macOS
    elif sys.platform == 'darwin':
        subprocess.run(['open', preview_path])
    # Windows
    elif sys.platform == 'win32':
        os.startfile(preview_path)
    else:
        print(f"Откройте файл вручную: {os.path.abspath(preview_path)}")
except Exception as e:
    print(f"❌ Не удалось открыть изображение: {e}")
    print(f"Откройте файл вручную: {os.path.abspath(preview_path)}")
