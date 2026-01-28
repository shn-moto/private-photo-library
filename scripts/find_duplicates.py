#!/usr/bin/env python3
"""
Поиск дубликатов изображений по векторному сходству CLIP эмбеддингов.

Использование:
    # Найти дубликаты и сохранить в файл
    python find_duplicates.py --output duplicates.txt

    # С кастомным порогом (по умолчанию 0.98)
    python find_duplicates.py --threshold 0.99 --output duplicates.txt

    # Удалить дубликаты (после проверки файла)
    python find_duplicates.py --delete duplicates.txt
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Добавить корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def find_duplicates(threshold: float = 0.98, limit: int = 10000) -> list:
    """
    Найти дубликаты по косинусному сходству CLIP эмбеддингов.

    Args:
        threshold: минимальное сходство для считания дубликатом (0.98 = 98%)
        limit: максимальное количество пар для проверки

    Returns:
        Список групп дубликатов: [[(path1, id1), (path2, id2), ...], ...]
    """
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    print(f"Подключение к БД: {settings.DATABASE_URL.split('@')[1]}")
    print(f"Порог сходства: {threshold:.2%}")
    print(f"Поиск дубликатов...")

    try:
        # Находим пары с высоким сходством
        # Используем self-join для сравнения всех пар
        # Условие p1.image_id < p2.image_id исключает дубликаты пар и сравнение с собой
        query = text("""
            WITH pairs AS (
                SELECT
                    p1.image_id as id1,
                    p1.file_path as path1,
                    p1.file_size as size1,
                    p2.image_id as id2,
                    p2.file_path as path2,
                    p2.file_size as size2,
                    1 - (p1.clip_embedding <=> p2.clip_embedding) as similarity
                FROM photo_index p1
                JOIN photo_index p2 ON p1.image_id < p2.image_id
                WHERE p1.clip_embedding IS NOT NULL
                  AND p2.clip_embedding IS NOT NULL
                  AND 1 - (p1.clip_embedding <=> p2.clip_embedding) >= :threshold
                ORDER BY similarity DESC
                LIMIT :limit
            )
            SELECT * FROM pairs
        """)

        result = session.execute(query, {'threshold': threshold, 'limit': limit})
        pairs = list(result)

        print(f"Найдено пар с сходством >= {threshold:.2%}: {len(pairs)}")

        if not pairs:
            return []

        # Группируем дубликаты (Union-Find алгоритм)
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Сохраняем информацию о файлах
        file_info = {}

        for row in pairs:
            id1, path1, size1, id2, path2, size2, similarity = row
            file_info[id1] = {'path': path1, 'size': size1, 'id': id1}
            file_info[id2] = {'path': path2, 'size': size2, 'id': id2}
            union(id1, id2)

        # Собираем группы
        groups = defaultdict(list)
        for file_id, info in file_info.items():
            root = find(file_id)
            groups[root].append(info)

        # Сортируем группы по размеру (больше файлов = больше дубликатов)
        result_groups = []
        for group in groups.values():
            if len(group) > 1:
                # Сортируем внутри группы: первый - оригинал (по пути/дате)
                group.sort(key=lambda x: (x['path']))
                result_groups.append(group)

        # Сортируем группы по количеству дубликатов
        result_groups.sort(key=lambda g: -len(g))

        return result_groups

    finally:
        session.close()
        engine.dispose()


def save_duplicates_report(groups: list, output_file: str, threshold: float):
    """Сохранить отчёт о дубликатах в файл."""

    total_duplicates = sum(len(g) - 1 for g in groups)
    total_groups = len(groups)

    # Подсчитываем потенциально освобождаемое место
    total_size_saved = 0
    for group in groups:
        # Считаем размер всех кроме первого (оригинала)
        for item in group[1:]:
            if item['size']:
                total_size_saved += item['size']

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Отчёт о дубликатах изображений\n")
        f.write(f"# Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Порог сходства: {threshold:.2%}\n")
        f.write(f"# Групп дубликатов: {total_groups}\n")
        f.write(f"# Всего дубликатов: {total_duplicates}\n")
        f.write(f"# Потенциально освобождаемо: {total_size_saved / 1024 / 1024:.1f} MB\n")
        f.write(f"#\n")
        f.write(f"# Формат: первый файл в группе = ОРИГИНАЛ (сохраняется)\n")
        f.write(f"#         остальные = ДУБЛИКАТЫ (можно удалить)\n")
        f.write(f"# Для удаления: python find_duplicates.py --delete {output_file}\n")
        f.write(f"#\n\n")

        for i, group in enumerate(groups, 1):
            f.write(f"=== Группа {i} ({len(group)} файлов) ===\n")

            for j, item in enumerate(group):
                prefix = "KEEP  " if j == 0 else "DELETE"
                size_mb = item['size'] / 1024 / 1024 if item['size'] else 0
                f.write(f"{prefix} [{size_mb:6.1f} MB] {item['path']}\n")

            f.write(f"\n")

    print(f"\nОтчёт сохранён: {output_file}")
    print(f"Групп дубликатов: {total_groups}")
    print(f"Всего дубликатов к удалению: {total_duplicates}")
    print(f"Потенциально освобождаемо: {total_size_saved / 1024 / 1024:.1f} MB")


def delete_duplicates(report_file: str, dry_run: bool = True):
    """
    Удалить дубликаты на основе отчёта.

    Args:
        report_file: путь к файлу отчёта
        dry_run: если True, только показать что будет удалено
    """
    from send2trash import send2trash

    if not os.path.exists(report_file):
        print(f"Файл не найден: {report_file}")
        return

    files_to_delete = []

    with open(report_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DELETE'):
                # Парсим строку: DELETE [  12.3 MB] /path/to/file
                parts = line.split(']', 1)
                if len(parts) == 2:
                    path = parts[1].strip()
                    files_to_delete.append(path)

    if not files_to_delete:
        print("Нет файлов для удаления")
        return

    print(f"Файлов к удалению: {len(files_to_delete)}")

    if dry_run:
        print("\n[DRY RUN] Будут удалены:")
        for path in files_to_delete[:20]:
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {path}")
        if len(files_to_delete) > 20:
            print(f"  ... и ещё {len(files_to_delete) - 20} файлов")
        print(f"\nДля реального удаления добавьте --confirm")
        return

    # Реальное удаление
    deleted = 0
    errors = []

    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        for path in files_to_delete:
            try:
                if os.path.exists(path):
                    send2trash(path)
                    deleted += 1
                    print(f"Удалён: {path}")

                    # Удаляем из БД
                    session.execute(
                        text("DELETE FROM photo_index WHERE file_path = :path"),
                        {'path': path}
                    )
                else:
                    # Файл уже удалён - просто удаляем из БД
                    session.execute(
                        text("DELETE FROM photo_index WHERE file_path = :path"),
                        {'path': path}
                    )
                    print(f"Удалён из БД (файл не существует): {path}")
                    deleted += 1

            except Exception as e:
                errors.append(f"{path}: {e}")
                print(f"Ошибка: {path} - {e}")

        session.commit()

    finally:
        session.close()
        engine.dispose()

    print(f"\nУдалено: {deleted}/{len(files_to_delete)}")
    if errors:
        print(f"Ошибок: {len(errors)}")


def main():
    parser = argparse.ArgumentParser(
        description='Поиск и удаление дубликатов изображений по CLIP эмбеддингам'
    )

    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.98,
        help='Порог сходства (0.0-1.0, по умолчанию 0.98)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='duplicates.txt',
        help='Файл для сохранения отчёта (по умолчанию duplicates.txt)'
    )

    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=50000,
        help='Максимум пар для анализа (по умолчанию 50000)'
    )

    parser.add_argument(
        '--delete', '-d',
        type=str,
        metavar='REPORT_FILE',
        help='Удалить дубликаты из указанного файла отчёта'
    )

    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Подтвердить удаление (без этого флага - dry run)'
    )

    args = parser.parse_args()

    if args.delete:
        # Режим удаления
        delete_duplicates(args.delete, dry_run=not args.confirm)
    else:
        # Режим поиска
        groups = find_duplicates(threshold=args.threshold, limit=args.limit)

        if groups:
            save_duplicates_report(groups, args.output, args.threshold)
        else:
            print("Дубликаты не найдены")


if __name__ == '__main__':
    main()
