#!/usr/bin/env python3
"""Скрипт для переименования видео файлов с неправильным расширением (.HEIC -> .mov/.mp4)"""

import os
import sys
from pathlib import Path
from typing import Optional

# Счётчики
stats = {"scanned": 0, "renamed": 0, "errors": 0, "skipped": 0}


def get_video_extension(file_path: str) -> Optional[str]:
    """Определить правильное расширение для видео файла по magic bytes"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)

        if len(header) >= 8 and header[4:8] == b'ftyp':
            ftyp = header[8:12].lower()
            # QuickTime
            if ftyp in (b'qt  ', b'mqt '):
                return '.mov'
            # MP4 варианты
            if ftyp in (b'mp4 ', b'mp41', b'mp42', b'isom', b'avc1', b'm4v '):
                return '.mp4'
            # MOV
            if ftyp == b'mov ':
                return '.mov'
        return None
    except Exception:
        return None


def fix_extension(file_path: str, dry_run: bool = False) -> bool:
    """Переименовать файл если нужно. Возвращает True если переименован."""
    video_ext = get_video_extension(file_path)
    if video_ext is None:
        return False

    current_ext = os.path.splitext(file_path)[1].lower()

    # Расширение уже правильное
    if current_ext == video_ext:
        stats["skipped"] += 1
        return False

    # Новое имя
    base = os.path.splitext(file_path)[0]
    new_path = base + video_ext

    # Если файл существует, добавляем суффикс
    counter = 1
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{video_ext}"
        counter += 1

    rel_old = os.path.basename(file_path)
    rel_new = os.path.basename(new_path)

    if dry_run:
        print(f"  [DRY] {rel_old} -> {rel_new}")
        stats["renamed"] += 1
        return True

    try:
        os.rename(file_path, new_path)
        print(f"  {rel_old} -> {rel_new}")
        stats["renamed"] += 1
        return True
    except OSError as e:
        print(f"  ERROR: {rel_old}: {e}")
        stats["errors"] += 1
        return False


def scan_directory(directory: str, dry_run: bool = False):
    """Сканировать директорию и переименовать видео файлы"""
    extensions = {'.heic', '.heif', '.jpg', '.jpeg', '.png'}

    print(f"Сканирование: {directory}")
    print(f"Режим: {'DRY RUN (без изменений)' if dry_run else 'РЕАЛЬНОЕ переименование'}")
    print("-" * 60)

    for root, dirs, files in os.walk(directory):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                continue

            file_path = os.path.join(root, filename)
            stats["scanned"] += 1

            fix_extension(file_path, dry_run)

    print("-" * 60)
    print(f"Просканировано: {stats['scanned']}")
    print(f"Переименовано:  {stats['renamed']}")
    print(f"Пропущено:      {stats['skipped']}")
    print(f"Ошибок:         {stats['errors']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Переименование видео файлов с неправильным расширением"
    )
    parser.add_argument(
        "directory",
        help="Директория для сканирования"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Только показать что будет переименовано, без изменений"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Не спрашивать подтверждение"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Ошибка: {args.directory} не является директорией")
        sys.exit(1)

    if not args.dry_run and not args.yes:
        print(f"Будут переименованы видео файлы в: {args.directory}")
        confirm = input("Продолжить? [y/N]: ")
        if confirm.lower() != 'y':
            print("Отменено")
            sys.exit(0)

    scan_directory(args.directory, args.dry_run)


if __name__ == "__main__":
    main()
