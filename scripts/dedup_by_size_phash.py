#!/usr/bin/env python3
"""
Дедупликация фото по точному совпадению file_size + подтверждение по pHash.

Логика:
  1. Берём фото за последние N месяцев (по photo_date, fallback created_at).
  2. Группируем по ТОЧНОМУ совпадению file_size (до байта).
  3. Внутри size-групп подтверждаем дубликаты по pHash (Hamming <= threshold).
  4. Оставляем ПЕРВЫЙ добавленный (минимальный image_id), остальные — в DUPLICATES_DIR,
     записи удаляем из БД (через DuplicateFinder.delete_from_report).

По умолчанию — dry-run (ничего не перемещает). Для применения добавь --apply.

Запуск (в Docker):
  docker exec smart_photo_api python /app/scripts/dedup_by_size_phash.py
  docker exec smart_photo_api python /app/scripts/dedup_by_size_phash.py --apply
  docker exec smart_photo_api python /app/scripts/dedup_by_size_phash.py \
      --months 6 --threshold 6 --path-filter '/photos/_WIFI_SYNC/%' --apply
"""
import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, '/app')

from sqlalchemy import text  # noqa: E402
from config.settings import settings  # noqa: E402
from db.database import DatabaseManager  # noqa: E402
from services.phash_service import PHashService, _load_image_for_hash  # noqa: E402


def hamming_hex(h1: str, h2: str) -> int:
    """Hamming distance между двумя hex-pHash (паддинг справа нулями до 256 бит)."""
    a = int(h1.ljust(64, '0'), 16)
    b = int(h2.ljust(64, '0'), 16)
    return bin(a ^ b).count('1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', type=int, default=6, help='Окно по дате (мес), default 6')
    ap.add_argument('--threshold', type=int, default=6,
                    help='Макс. Hamming distance pHash для дубликата (0=точно), default 6')
    ap.add_argument('--path-filter', default=None,
                    help="SQL LIKE по file_path, напр. '/photos/_WIFI_SYNC/%%'")
    ap.add_argument('--apply', action='store_true',
                    help='Реально переместить дубликаты (по умолчанию dry-run)')
    ap.add_argument('--all-formats', action='store_true',
                    help='Сравнивать и между разными форматами (по умолчанию только в пределах '
                         'одной группы форматов — защита от ложных кросс-форматных совпадений)')
    ap.add_argument('--report', default='/reports/dedup_size_phash.txt')
    args = ap.parse_args()

    db = DatabaseManager(settings.DATABASE_URL)
    cutoff = datetime.now() - timedelta(days=args.months * 31)

    # 1. Кандидаты за последние N месяцев
    session = db.get_session()
    try:
        params = {'cutoff': cutoff}
        path_cond = ''
        if args.path_filter:
            path_cond = 'AND file_path LIKE :pf'
            params['pf'] = args.path_filter
        rows = session.execute(text(f"""
            SELECT image_id, file_path, file_size, phash
            FROM photo_index
            WHERE file_size IS NOT NULL
              AND (photo_date >= :cutoff OR (photo_date IS NULL AND created_at >= :cutoff))
              {path_cond}
            ORDER BY image_id
        """), params).fetchall()
    finally:
        session.close()

    print(f"Кандидатов (за {args.months} мес, cutoff {cutoff:%Y-%m-%d}): {len(rows)}")

    # 2. Группировка по точному размеру (по умолчанию — в пределах одной группы форматов)
    by_size = defaultdict(list)
    for image_id, path, size, phash in rows:
        key = size if args.all_formats else (size, PHashService._get_format(path))
        by_size[key].append([image_id, path, size, phash])
    size_groups = {s: items for s, items in by_size.items() if len(items) > 1}
    candidates = sum(len(v) for v in size_groups.values())
    print(f"Групп с одинаковым размером (>1 файла, same_format={not args.all_formats}): "
          f"{len(size_groups)} ({candidates} файлов)")

    # 3. Досчитать pHash там, где его нет
    need_phash = [it for items in size_groups.values() for it in items if not it[3]]
    print(f"Нужно вычислить pHash: {len(need_phash)}")
    if need_phash:
        upd = db.get_session()
        try:
            computed = failed = 0
            for it in need_phash:
                img = _load_image_for_hash(it[1])
                if img is None:
                    it[3] = None
                    failed += 1
                    continue
                h = PHashService.compute_phash_from_pil(img)
                it[3] = h
                if h:
                    upd.execute(text("UPDATE photo_index SET phash=:h WHERE image_id=:id"),
                                {'h': h, 'id': it[0]})
                    computed += 1
                    if computed % 50 == 0:
                        upd.commit()
                else:
                    failed += 1
            upd.commit()
            print(f"Вычислено pHash: {computed}, не удалось: {failed}")
        finally:
            upd.close()

    # 4. Внутри size-групп — кластеры по pHash (Union-Find, Hamming <= threshold)
    final_groups = []
    for size, items in size_groups.items():
        valid = [it for it in items if it[3]]
        if len(valid) < 2:
            continue
        parent = list(range(len(valid)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                if hamming_hex(valid[i][3], valid[j][3]) <= args.threshold:
                    union(i, j)

        clusters = defaultdict(list)
        for i in range(len(valid)):
            clusters[find(i)].append(valid[i])
        for cl in clusters.values():
            if len(cl) > 1:
                cl.sort(key=lambda x: x[0])  # min image_id первым = KEEP (первый добавленный)
                final_groups.append(cl)

    total_dup = sum(len(g) - 1 for g in final_groups)
    total_mb = sum(it[2] for g in final_groups for it in g[1:]) / 1024 / 1024
    print(f"\nГрупп дубликатов (size+pHash): {len(final_groups)}")
    print(f"Файлов к перемещению (DELETE): {total_dup}")
    print(f"Освободится: {total_mb:.1f} MB")

    # Отчёт в формате, понятном DuplicateFinder.delete_from_report
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(f"# Dedup by size+pHash | {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"# months={args.months} threshold={args.threshold} "
                f"path_filter={args.path_filter}\n")
        f.write(f"# groups={len(final_groups)} delete={total_dup} "
                f"free_mb={total_mb:.1f}\n")
        f.write("# KEEP = первый добавленный (min image_id), DELETE = дубликаты\n\n")
        for i, g in enumerate(final_groups, 1):
            f.write(f"=== Группа {i} ({len(g)} файлов) ===\n")
            for j, it in enumerate(g):
                prefix = "KEEP  " if j == 0 else "DELETE"
                f.write(f"{prefix} [{it[2] / 1024 / 1024:6.1f} MB] {it[1]}\n")
            f.write("\n")
    print(f"Отчёт: {args.report}")

    for g in final_groups[:5]:
        print(f"  KEEP {g[0][0]} {g[0][1]}")
        for it in g[1:]:
            print(f"    DEL {it[0]} {it[1]}")

    if not args.apply:
        print("\n[DRY-RUN] Ничего не перемещено. Добавь --apply для применения.")
        return

    from services.duplicate_finder import DuplicateFinder
    finder = DuplicateFinder(db.get_session, None)
    result = finder.delete_from_report(args.report, dry_run=False)
    print(f"\n[APPLY] Перемещено/удалено: {result['deleted']}, "
          f"ошибок: {len(result.get('errors', []))}")
    for e in result.get('errors', [])[:10]:
        print(f"  ERR {e}")


if __name__ == '__main__':
    main()
