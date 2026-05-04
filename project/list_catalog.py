"""
Скрипт для просмотра каталога моделей из датасета EmbodiedGen.
"""

import sys
from project.catalog import load_catalog


def list_all_models(output_file=None):
    """Выводит список всех моделей из каталога."""
    catalog = load_catalog()
    
    print("="*80)
    print(f"КАТАЛОГ МОДЕЛЕЙ EMBODIEDGEN")
    print("="*80)
    print(f"Всего моделей: {len(catalog)}")
    print("="*80)
    print()
    
    # Группируем по категориям
    by_category = {}
    for model in catalog:
        categories = model.get("categories", ["uncategorized"])
        primary = categories[0] if categories else "uncategorized"
        if primary not in by_category:
            by_category[primary] = []
        by_category[primary].append(model)
    
    # Выводим статистику по категориям
    print("СТАТИСТИКА ПО КАТЕГОРИЯМ:")
    print("-"*80)
    for category in sorted(by_category.keys()):
        count = len(by_category[category])
        print(f"  {category:30s} : {count:4d} моделей")
    print("="*80)
    print()
    
    # Если указан файл, сохраняем туда
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("КАТАЛОГ МОДЕЛЕЙ EMBODIEDGEN\n")
            f.write("="*80 + "\n")
            f.write(f"Всего моделей: {len(catalog)}\n")
            f.write("="*80 + "\n\n")
            
            for category in sorted(by_category.keys()):
                f.write(f"\n{'='*80}\n")
                f.write(f"КАТЕГОРИЯ: {category.upper()}\n")
                f.write(f"{'='*80}\n\n")
                
                for model in by_category[category]:
                    f.write(f"Название: {model['name']}\n")
                    f.write(f"UUID: {model['uuid']}\n")
                    f.write(f"Категории: {', '.join(model['categories'])}\n")
                    f.write(f"Теги: {', '.join(model['tags'][:10])}\n")
                    f.write(f"Путь: {model['model_loc']}\n")
                    if model.get('description'):
                        f.write(f"Описание: {model['description']}\n")
                    f.write("-"*80 + "\n")
        
        print(f"Список сохранён в файл: {output_file}")
    else:
        # Выводим в консоль (первые 50 моделей)
        print("ПЕРВЫЕ 50 МОДЕЛЕЙ:")
        print("-"*80)
        for i, model in enumerate(catalog[:50], 1):
            print(f"{i:3d}. {model['name']:40s} | "
                  f"{model['categories'][0] if model['categories'] else 'N/A':20s}")
        
        if len(catalog) > 50:
            print(f"\n... и ещё {len(catalog) - 50} моделей")
        
        print()
        print("Для полного списка запустите:")
        print("  python list_catalog.py output.txt")


def search_models(query):
    """Ищет модели по запросу."""
    catalog = load_catalog()
    query_lower = query.lower()
    
    results = []
    for model in catalog:
        # Ищем в названии, категориях, тегах
        if (query_lower in model['name'].lower() or
            any(query_lower in cat.lower() for cat in model['categories']) or
            any(query_lower in tag.lower() for tag in model['tags'])):
            results.append(model)
    
    print("="*80)
    print(f"РЕЗУЛЬТАТЫ ПОИСКА: '{query}'")
    print("="*80)
    print(f"Найдено: {len(results)} моделей")
    print("="*80)
    print()
    
    for i, model in enumerate(results, 1):
        print(f"{i:3d}. {model['name']}")
        print(f"     Категории: {', '.join(model['categories'])}")
        print(f"     UUID: {model['uuid']}")
        print(f"     Путь: {model['model_loc']}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        # Если аргумент - файл для сохранения
        if arg.endswith('.txt') or arg.endswith('.md'):
            list_all_models(output_file=arg)
        else:
            # Иначе это поисковый запрос
            search_models(arg)
    else:
        # Без аргументов - показываем краткий список
        list_all_models()
