"""
Тест для определения правильных yaw углов
"""

print("="*70)
print("АНАЛИЗ YAW УГЛОВ ДЛЯ EULER='90 {yaw} 0'")
print("="*70)
print()

print("Согласно комментарию в creator/sim_interfaces/mujoco.py:")
print("  euler='90 yaw 0' означает:")
print("  1. Поворот на 90° вокруг X (Rx)")
print("  2. Поворот на yaw° вокруг Y тела (после Rx)")
print("  3. После Rx(90°), ось Y тела указывает вдоль оси Z мира")
print("  4. Поэтому yaw вокруг Y тела = yaw вокруг Z мира!")
print()

print("Система координат MuJoCo:")
print("  X → восток (вправо)")
print("  Y → север (вперёд)")
print("  Z → вверх")
print()

print("Для модели с up_axis='y' (Y вверх в файле):")
print("  - Передняя часть модели смотрит вдоль +Z модели")
print("  - После euler='90 0 0': передняя часть смотрит вдоль -Y мира (юг)")
print()

print("="*70)
print("ПРАВИЛЬНЫЕ УГЛЫ ДЛЯ 'FACING INWARD':")
print("="*70)
print()

table_pos = (-2.2, -2.2)
print(f"Стол в позиции: ({table_pos[0]}, {table_pos[1]})")
print()

test_cases = [
    # (chair_pos, direction_name, expected_facing, correct_yaw)
    ((-2.2, -1.6), "Север (y > table_y)", "Юг (к столу)", None),
    ((-1.6, -2.2), "Восток (x > table_x)", "Запад (к столу)", None),
    ((-2.2, -2.8), "Юг (y < table_y)", "Север (к столу)", None),
    ((-2.8, -2.2), "Запад (x < table_x)", "Восток (к столу)", None),
]

print("Логика:")
print("  - После euler='90 0 0': модель смотрит на ЮГ (-Y)")
print("  - yaw поворачивает вокруг вертикальной оси (Z мира)")
print("  - yaw=0°   → смотрит на ЮГ (-Y)")
print("  - yaw=90°  → смотрит на ЗАПАД (-X)")
print("  - yaw=180° → смотрит на СЕВЕР (+Y)")
print("  - yaw=270° → смотрит на ВОСТОК (+X)")
print()

for chair_pos, direction, facing, _ in test_cases:
    dx = chair_pos[0] - table_pos[0]
    dy = chair_pos[1] - table_pos[1]
    
    # Определяем куда должен смотреть стул
    if abs(dy) > abs(dx):  # Север/Юг
        if dy > 0:  # Стул севернее стола
            target_dir = "Юг (-Y)"
            correct_yaw = 0  # Смотрит на юг
        else:  # Стул южнее стола
            target_dir = "Север (+Y)"
            correct_yaw = 180  # Смотрит на север
    else:  # Восток/Запад
        if dx > 0:  # Стул восточнее стола
            target_dir = "Запад (-X)"
            correct_yaw = 90  # Смотрит на запад
        else:  # Стул западнее стола
            target_dir = "Восток (+X)"
            correct_yaw = 270  # Смотрит на восток
    
    print(f"Стул {direction}:")
    print(f"  Позиция: {chair_pos}")
    print(f"  Должен смотреть: {target_dir}")
    print(f"  Правильный yaw: {correct_yaw}°")
    print()

print("="*70)
print("СРАВНЕНИЕ С ТЕКУЩИМИ ДАННЫМИ:")
print("="*70)
print()

current_data = [
    ((-2.2, -1.6), 180),  # Север
    ((-1.6, -2.2), 270),  # Восток
    ((-2.2, -2.8), 0),    # Юг
    ((-2.8, -2.2), 90),   # Запад
]

expected = [0, 90, 180, 270]
actual = [180, 270, 0, 90]

print("Позиция стула | Текущий yaw | Ожидаемый yaw | Совпадает?")
print("-" * 70)
for i, ((pos, curr_yaw), exp_yaw) in enumerate(zip(current_data, expected)):
    match = "✓" if curr_yaw == exp_yaw else "✗"
    print(f"{str(pos):20} | {curr_yaw:3d}° | {exp_yaw:3d}° | {match}")

print()
print("="*70)
print("ВЫВОД:")
print("="*70)
print()
print("Текущие yaw углы НЕПРАВИЛЬНЫЕ!")
print()
print("Проблема в промпте _BATCH_PLACEMENT_PROMPT:")
print("  Текущая инструкция: 'object north of parent → yaw=180°'")
print("  Правильная инструкция: 'object north of parent → yaw=0°'")
print()
print("Правильная таблица:")
print("  Стул СЕВЕРНЕЕ стола (y > parent_y) → должен смотреть на ЮГ  → yaw=0°")
print("  Стул ВОСТОЧНЕЕ стола (x > parent_x) → должен смотреть на ЗАПАД → yaw=90°")
print("  Стул ЮЖНЕЕ стола (y < parent_y) → должен смотреть на СЕВЕР → yaw=180°")
print("  Стул ЗАПАДНЕЕ стола (x < parent_x) → должен смотреть на ВОСТОК → yaw=270°")
print()
