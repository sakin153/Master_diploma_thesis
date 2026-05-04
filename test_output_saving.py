#!/usr/bin/env python3
"""
Тестовый скрипт для проверки сохранения промежуточных результатов.
"""

import json
import os
from project.scene_planner import generate_placement_plan


def test_output_saving():
    """Тест сохранения результатов каждой стадии."""
    
    # Простой тестовый набор моделей
    test_models = [
        {
            "Model": "Table",
            "uuid": "table_001",
            "model_loc": "/path/to/table.glb",
            "size": [1.5, 0.75, 0.9],  # width, height, depth
            "_up_axis": "y",
            "is_static": True,
        },
        {
            "Model": "Chair",
            "uuid": "chair_001",
            "model_loc": "/path/to/chair.glb",
            "size": [0.5, 0.9, 0.5],
            "_up_axis": "y",
            "is_static": True,
        },
        {
            "Model": "Chair",
            "uuid": "chair_002",
            "model_loc": "/path/to/chair.glb",
            "size": [0.5, 0.9, 0.5],
            "_up_axis": "y",
            "is_static": True,
        },
    ]
    
    print("=" * 60)
    print("ТЕСТ: Сохранение промежуточных результатов")
    print("=" * 60)
    
    # Проверить что папка output существует
    if not os.path.exists("output"):
        print("❌ Папка output не существует!")
        return False
    
    print("✓ Папка output существует")
    
    # Подсчитать файлы до генерации
    files_before = set(os.listdir("output"))
    print(f"✓ Файлов в output до генерации: {len(files_before)}")
    
    # Запустить генерацию с сохранением
    print("\nЗапуск генерации с save_outputs=True...")
    try:
        result = generate_placement_plan(
            models=test_models,
            room_half_size=4.5,
            room_type="dining_room",
            query="table with 2 chairs around it",
            save_outputs=True,
        )
        print(f"✓ Генерация завершена: {len(result)} объектов размещено")
    except Exception as e:
        print(f"❌ Ошибка при генерации: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Подсчитать файлы после генерации
    files_after = set(os.listdir("output"))
    new_files = files_after - files_before
    print(f"\n✓ Новых файлов создано: {len(new_files)}")
    
    # Проверить что созданы все 4 стадии
    expected_stages = ["stage1_scene_graph", "stage2_world_state", "stage3_merged_output", "stage4_final_layout"]
    found_stages = set()
    
    for filename in new_files:
        if filename.endswith(".json"):
            for stage in expected_stages:
                if filename.startswith(stage):
                    found_stages.add(stage)
                    print(f"  ✓ {filename}")
    
    # Проверка результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ")
    print("=" * 60)
    
    success = True
    for stage in expected_stages:
        if stage in found_stages:
            print(f"✓ {stage}: найден")
        else:
            print(f"❌ {stage}: НЕ НАЙДЕН")
            success = False
    
    if success:
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print(f"Создано {len(new_files)} файлов в папке output/")
    else:
        print("\n❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
    
    return success


if __name__ == "__main__":
    test_output_saving()
