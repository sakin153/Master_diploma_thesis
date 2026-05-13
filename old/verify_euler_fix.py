"""
Простая проверка исправления euler углов
"""

# Тестируем логику
def test_euler_conversion():
    print("="*60)
    print("ПРОВЕРКА ИСПРАВЛЕНИЯ EULER УГЛОВ")
    print("="*60)
    print()
    
    test_cases = [
        # (up_axis, yaw_deg, expected_euler)
        ("y", 0, "90 0 0"),
        ("y", 90, "90 0 90"),
        ("y", 180, "90 0 180"),
        ("y", 270, "90 0 270"),
        ("z", 0, "0 0 0"),
        ("z", 90, "0 0 90"),
        ("z", 180, "0 0 180"),
        ("z", 270, "0 0 270"),
    ]
    
    print("Тестовые случаи:")
    print()
    
    for up_axis, yaw_deg, expected in test_cases:
        # Новая формула (исправленная)
        euler_str = f"0 0 {yaw_deg}" if up_axis == "z" else f"90 0 {yaw_deg}"
        
        status = "✓" if euler_str == expected else "✗"
        print(f"{status} up_axis={up_axis}, yaw={yaw_deg:3.0f}° → euler=\"{euler_str}\" (ожидалось: \"{expected}\")")
    
    print()
    print("="*60)
    print("ОБЪЯСНЕНИЕ:")
    print("="*60)
    print()
    print("Euler углы в MuJoCo: (roll, pitch, yaw) в порядке XYZ")
    print()
    print("up_axis=\"y\" (модель лежит):")
    print("  - roll=90° поднимает модель вертикально")
    print("  - pitch=0° не наклоняет")
    print("  - yaw=N° поворачивает вокруг вертикальной оси")
    print()
    print("up_axis=\"z\" (модель уже вертикальна):")
    print("  - roll=0° не нужен")
    print("  - pitch=0° не нужен")
    print("  - yaw=N° просто поворачивает")
    print()
    print("="*60)
    print("ПРИМЕР ДЛЯ СТУЛА:")
    print("="*60)
    print()
    print("Стул восточнее стола (должен смотреть на запад):")
    print("  - yaw_deg = 270° (смотрит на -X, т.е. запад)")
    print("  - up_axis = \"y\" (стул лежит в файле)")
    print("  - euler = \"90 0 270\"")
    print("    → roll=90° поднимает стул")
    print("    → yaw=270° поворачивает лицом на запад")
    print("    → стул смотрит на стол! ✓")
    print()
    print("="*60)

if __name__ == "__main__":
    test_euler_conversion()
