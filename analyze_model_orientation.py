"""
Анализ ориентации моделей и euler углов в MuJoCo
"""

import trimesh
import numpy as np

# Загружаем модель стула
chair_path = "/home/sakin153/.cache/huggingface/hub/datasets--HorizonRobotics--EmbodiedGenData/snapshots/0da911ca5b1c26c64533d8579b2fe9d3f042acb4/dataset/basic_furniture/chair/aeea9a5e33665bb3b31544206cb2a929/mesh/computer_chair_44.glb"

print("="*60)
print("АНАЛИЗ ОРИЕНТАЦИИ МОДЕЛИ СТУЛА")
print("="*60)
print()

try:
    mesh = trimesh.load(chair_path, force='mesh')
    
    print(f"Bounds (min): {mesh.bounds[0]}")
    print(f"Bounds (max): {mesh.bounds[1]}")
    print(f"Extents (size): {mesh.extents}")
    print()
    
    # Определяем главные оси
    extents = mesh.extents
    axes = ['X', 'Y', 'Z']
    sorted_axes = sorted(zip(extents, axes), reverse=True)
    
    print("Размеры по осям (от большего к меньшему):")
    for size, axis in sorted_axes:
        print(f"  {axis}: {size:.3f}m")
    print()
    
    # Определяем up_axis (самая длинная ось обычно = высота)
    longest_axis = sorted_axes[0][1]
    print(f"Самая длинная ось (вероятно высота стула): {longest_axis}")
    print()
    
    # Проверяем центр масс
    print(f"Центр масс: {mesh.center_mass}")
    print()
    
    print("="*60)
    print("ИНТЕРПРЕТАЦИЯ:")
    print("="*60)
    print()
    
    if longest_axis == 'Y':
        print("✓ up_axis = 'y' КОРРЕКТЕН")
        print("  Модель стоит вертикально вдоль оси Y")
        print("  В MuJoCo (где Z вверх) нужно повернуть на 90° вокруг X")
        print()
        print("Формула euler для поворота:")
        print("  euler = '90 {yaw} 0'")
        print("  - 90° вокруг X (roll) поворачивает Y→Z (ставит вертикально)")
        print("  - {yaw}° вокруг Y (pitch) поворачивает модель по горизонтали")
        print("  - 0° вокруг Z (yaw) не используется")
    elif longest_axis == 'Z':
        print("✓ up_axis = 'z' КОРРЕКТЕН")
        print("  Модель уже стоит вертикально вдоль оси Z")
        print("  В MuJoCo просто применяем yaw")
        print()
        print("Формула euler для поворота:")
        print("  euler = '0 0 {yaw}'")
    else:
        print("⚠ Модель лежит горизонтально (X - самая длинная ось)")
        print("  Нужна другая формула!")
    
    print()
    print("="*60)
    print("ПРОВЕРКА ТЕКУЩЕЙ ФОРМУЛЫ:")
    print("="*60)
    print()
    
    # Текущая формула
    up_axis = "y"
    test_yaws = [0, 90, 180, 270]
    
    print("Текущая формула: euler = '90 {yaw} 0'")
    print()
    print("Что происходит:")
    print("  1. Поворот на 90° вокруг X (roll)")
    print("     - Ось Y модели → ось Z мира (вертикально вверх) ✓")
    print("     - Ось Z модели → ось -Y мира")
    print("     - Ось X модели → ось X мира")
    print()
    print("  2. Поворот на {yaw}° вокруг Y (pitch)")
    print("     - Это поворот вокруг НОВОЙ оси Y (после первого поворота)")
    print("     - Новая ось Y = старая ось -Z")
    print("     - Поворот вокруг горизонтальной оси!")
    print()
    print("Примеры:")
    for yaw in test_yaws:
        print(f"  yaw={yaw:3d}° → euler='90 {yaw} 0'")
        if yaw == 0:
            print(f"    Стул стоит прямо, смотрит вперёд")
        elif yaw == 90:
            print(f"    Стул наклонён на 90° (лежит на боку!)")
        elif yaw == 180:
            print(f"    Стул перевёрнут вверх ногами!")
        elif yaw == 270:
            print(f"    Стул наклонён на 270° (лежит на другом боку!)")
    
    print()
    print("="*60)
    print("ВЫВОД: ФОРМУЛА НЕПРАВИЛЬНАЯ!")
    print("="*60)
    
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    print()
    print("Попробуем логический анализ...")
    print()
    print("="*60)
    print("ЛОГИЧЕСКИЙ АНАЛИЗ EULER УГЛОВ")
    print("="*60)
    print()
    print("MuJoCo использует порядок XYZ для euler углов:")
    print("  euler = 'roll pitch yaw'")
    print()
    print("Для модели с up_axis='y' (Y вверх в модели, Z вверх в мире):")
    print()
    print("ВАРИАНТ 1: euler = '90 {yaw} 0'")
    print("  1. Roll 90° вокруг X: Y→Z (ставит вертикально)")
    print("  2. Pitch {yaw}° вокруг Y: наклоняет модель (ПЛОХО!)")
    print("  3. Yaw 0° вокруг Z: не поворачивает")
    print("  Результат: модель наклонена ❌")
    print()
    print("ВАРИАНТ 2: euler = '0 {yaw} 90'")
    print("  1. Roll 0° вокруг X: не меняет")
    print("  2. Pitch {yaw}° вокруг Y: поворачивает горизонтально")
    print("  3. Yaw 90° вокруг Z: поворачивает Y→Z (ставит вертикально)")
    print("  Результат: сначала поворот, потом подъём (порядок важен!)")
    print()
    print("ВАРИАНТ 3: euler = '90 0 {yaw}'")
    print("  1. Roll 90° вокруг X: Y→Z (ставит вертикально)")
    print("  2. Pitch 0° вокруг Y: не наклоняет")
    print("  3. Yaw {yaw}° вокруг Z: поворачивает вокруг вертикальной оси")
    print("  Результат: сначала подъём, потом поворот")
    print("  НО: после roll=90°, ось Z модели стала осью -Y мира!")
    print("  Поэтому yaw вокруг Z мира не даст нужный эффект ❌")
    print()
    print("ПРАВИЛЬНЫЙ ВАРИАНТ: Нужно учесть порядок применения!")
