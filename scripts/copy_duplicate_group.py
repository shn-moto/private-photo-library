"""
Копирование группы дубликатов из отчёта duplicates.txt для ручной проверки.

Использование:
    python scripts/copy_duplicate_group.py 1
    python scripts/copy_duplicate_group.py 1 5 12
    python scripts/copy_duplicate_group.py 1-10
    python scripts/copy_duplicate_group.py --report reports/other.txt 3
"""

import argparse
import shutil
import sys
from pathlib import Path

PHOTOS_MAP = ("/photos", "D:/PHOTO")
DEFAULT_REPORT = Path(__file__).parent.parent / "reports" / "duplicates.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "reports" / "duplicates"


def parse_groups(report_path: Path) -> dict[int, list[str]]:
    """Парсит отчёт, возвращает {group_nr: [paths]}."""
    groups = {}
    current = None
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("=== Группа "):
                current = int(line.split()[2])
                groups[current] = []
            elif current and (line.startswith("KEEP") or line.startswith("DELETE")):
                path = line.split("]", 1)[1].strip()
                groups[current].append(path)
    return groups


def copy_group(group_nr: int, files: list[str], output_dir: Path):
    """Копирует файлы группы в output_dir/{group_nr}/."""
    dest_dir = output_dir / str(group_nr)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for i, docker_path in enumerate(files):
        local_path = Path(docker_path.replace(PHOTOS_MAP[0], PHOTOS_MAP[1], 1))
        label = "KEEP" if i == 0 else "DELETE"
        dest = dest_dir / f"{i:02d}_{label}_{local_path.name}"

        if not local_path.exists():
            print(f"  NOT FOUND: {local_path}")
            continue

        shutil.copy2(str(local_path), str(dest))
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"  {label:6s} [{size_mb:5.1f} MB] {local_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Копирование группы дубликатов для проверки")
    parser.add_argument("groups", nargs="+", help="Номера групп: 1 5 12 или 1-10")
    parser.add_argument("--report", "-r", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def expand_ranges(args: list[str]) -> list[int]:
    """Раскрывает диапазоны: ['1', '3-5', '8'] -> [1, 3, 4, 5, 8]."""
    result = []
    for arg in args:
        if "-" in arg:
            start, end = arg.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(arg))
    return result


def main():
    args = parse_args()
    group_nrs = expand_ranges(args.groups)

    if not args.report.exists():
        print(f"Отчёт не найден: {args.report}")
        sys.exit(1)

    all_groups = parse_groups(args.report)
    print(f"Отчёт: {args.report} ({len(all_groups)} групп)")

    for nr in group_nrs:
        if nr not in all_groups:
            print(f"\nГруппа {nr} не найдена (доступно: 1-{max(all_groups)})")
            continue

        files = all_groups[nr]
        print(f"\n=== Группа {nr} ({len(files)} файлов) → {OUTPUT_DIR / str(nr)} ===")
        copy_group(nr, files, OUTPUT_DIR)

    print("\nГотово.")


if __name__ == "__main__":
    main()
