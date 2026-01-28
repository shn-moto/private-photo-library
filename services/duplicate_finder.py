"""Сервис поиска дубликатов изображений по CLIP эмбеддингам."""

import logging
import os
from collections import defaultdict
from datetime import datetime

from sqlalchemy import text
import shutil

logger = logging.getLogger(__name__)


class DuplicateFinder:
    """Поиск и удаление дубликатов по косинусному сходству CLIP эмбеддингов."""

    def __init__(self, session_factory):
        """
        Args:
            session_factory: callable, возвращающий SQLAlchemy session
        """
        self._get_session = session_factory

    @staticmethod
    def _move_to_trash(file_path: str):
        from config.settings import settings
        trash_dir = settings.TRASH_DIR
        rel = os.path.relpath(file_path, "/photos")
        dest = os.path.join(trash_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(file_path, dest)

    def find_groups(self, threshold: float = 0.98, limit: int = 50000,
                    batch_size: int = 500, neighbors: int = 10,
                    path_filter: str = None) -> list:
        """
        Найти группы дубликатов через HNSW индекс (KNN для каждой записи).

        Вместо brute-force self-join (O(N²)) используем HNSW индекс:
        для каждого фото ищем K ближайших соседей — O(N × log N).

        Args:
            threshold: минимальное сходство (0.98 = 98%)
            limit: макс. количество пар в результате
            batch_size: размер батча для обработки
            neighbors: сколько ближайших соседей проверять для каждого фото
            path_filter: фильтр по пути (SQL LIKE, например '%/2024/%')

        Returns:
            Список групп: [[{id, path, size}, ...], ...]
        """
        session = self._get_session()
        try:
            # Увеличиваем ef_search для точности при высоком пороге
            session.execute(text("SET hnsw.ef_search = 100"))

            # Получаем все image_id с эмбеддингами
            path_condition = ""
            params = {}
            if path_filter:
                path_condition = "AND file_path LIKE :path_filter"
                params['path_filter'] = path_filter

            ids_result = session.execute(text(f"""
                SELECT image_id FROM photo_index
                WHERE clip_embedding IS NOT NULL AND indexed = 1
                {path_condition}
                ORDER BY image_id
            """), params)
            all_ids = [row[0] for row in ids_result]

            logger.info(f"Поиск дубликатов среди {len(all_ids)} фото "
                       f"(threshold={threshold}, neighbors={neighbors}"
                       f"{f', filter={path_filter}' if path_filter else ''})")

            # Для каждого фото ищем K ближайших соседей через HNSW
            # Батчим запросы для эффективности
            cosine_threshold = 1.0 - threshold  # pgvector <=> возвращает расстояние
            pairs = set()
            file_info = {}

            for offset in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[offset:offset + batch_size]
                placeholders = ','.join(f"'{id}'" for id in batch_ids)

                # Lateral join: для каждого фото в батче ищем K соседей через HNSW
                query = text(f"""
                    SELECT
                        p.image_id as id1, p.file_path as path1, p.file_size as size1,
                        neighbor.image_id as id2, neighbor.file_path as path2,
                        neighbor.file_size as size2, neighbor.distance
                    FROM photo_index p,
                    LATERAL (
                        SELECT n.image_id, n.file_path, n.file_size,
                               p.clip_embedding <=> n.clip_embedding as distance
                        FROM photo_index n
                        WHERE n.clip_embedding IS NOT NULL
                          AND n.indexed = 1
                          AND n.image_id != p.image_id
                        ORDER BY n.clip_embedding <=> p.clip_embedding
                        LIMIT :neighbors
                    ) neighbor
                    WHERE p.image_id IN ({placeholders})
                      AND neighbor.distance <= :cosine_threshold
                      AND p.file_size = neighbor.file_size
                """)

                result = session.execute(query, {
                    'neighbors': neighbors,
                    'cosine_threshold': cosine_threshold,
                })

                for row in result:
                    id1, path1, size1, id2, path2, size2, distance = row
                    # Нормализуем порядок пары чтобы избежать дублей
                    pair_key = tuple(sorted([id1, id2]))
                    if pair_key not in pairs:
                        pairs.add(pair_key)
                        file_info[id1] = {'id': id1, 'path': path1, 'size': size1 or 0}
                        file_info[id2] = {'id': id2, 'path': path2, 'size': size2 or 0}

                    if len(pairs) >= limit:
                        break

                if len(pairs) >= limit:
                    break

                if (offset + batch_size) % 5000 == 0:
                    logger.info(f"Обработано {offset + batch_size}/{len(all_ids)}, "
                               f"найдено пар: {len(pairs)}")

            logger.info(f"Найдено пар-дубликатов: {len(pairs)}")

            if not pairs:
                return []

            # Union-Find группировка
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

            for id1, id2 in pairs:
                union(id1, id2)

            groups_map = defaultdict(list)
            for file_id, info in file_info.items():
                groups_map[find(file_id)].append(info)

            groups = []
            for group in groups_map.values():
                if len(group) > 1:
                    group.sort(key=lambda x: x['path'])
                    groups.append(group)
            groups.sort(key=lambda g: -len(g))

            return groups

        finally:
            session.close()

    def save_report(self, groups: list, output_file: str, threshold: float) -> dict:
        """
        Сохранить отчёт о дубликатах в текстовый файл.

        Returns:
            dict с total_groups, total_duplicates, size_saved_mb
        """
        total_duplicates = sum(len(g) - 1 for g in groups)
        total_size_saved = sum(
            item['size'] for g in groups for item in g[1:]
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Отчёт о дубликатах изображений\n")
            f.write(f"# Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Порог сходства: {threshold:.2%}\n")
            f.write(f"# Групп дубликатов: {len(groups)}\n")
            f.write(f"# Всего дубликатов: {total_duplicates}\n")
            f.write(f"# Потенциально освобождаемо: {total_size_saved / 1024 / 1024:.1f} MB\n")
            f.write(f"#\n")
            f.write(f"# Формат: первый файл в группе = ОРИГИНАЛ (сохраняется)\n")
            f.write(f"#         остальные = ДУБЛИКАТЫ (можно удалить)\n")
            f.write(f"#\n\n")

            for i, group in enumerate(groups, 1):
                f.write(f"=== Группа {i} ({len(group)} файлов) ===\n")
                for j, item in enumerate(group):
                    prefix = "KEEP  " if j == 0 else "DELETE"
                    size_mb = item['size'] / 1024 / 1024
                    f.write(f"{prefix} [{size_mb:6.1f} MB] {item['path']}\n")
                f.write(f"\n")

        logger.info(f"Отчёт сохранён: {output_file}")

        return {
            "total_groups": len(groups),
            "total_duplicates": total_duplicates,
            "size_saved_mb": round(total_size_saved / 1024 / 1024, 1),
        }

    def delete_from_report(self, report_file: str, dry_run: bool = True) -> dict:
        """
        Удалить дубликаты на основе отчёта.

        Returns:
            dict с deleted, errors
        """
        files_to_delete = []
        with open(report_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('DELETE'):
                    parts = line.split(']', 1)
                    if len(parts) == 2:
                        files_to_delete.append(parts[1].strip())

        if not files_to_delete:
            return {"deleted": 0, "errors": []}

        if dry_run:
            existing = [p for p in files_to_delete if os.path.exists(p)]
            return {
                "dry_run": True,
                "total": len(files_to_delete),
                "existing": len(existing),
                "files": files_to_delete[:50],
            }

        deleted = 0
        errors = []
        session = self._get_session()

        try:
            for path in files_to_delete:
                try:
                    if os.path.exists(path):
                        self._move_to_trash(path)
                        logger.info(f"В корзину: {path}")

                    session.execute(
                        text("DELETE FROM photo_index WHERE file_path = :path"),
                        {'path': path}
                    )
                    deleted += 1
                except Exception as e:
                    errors.append(f"{path}: {e}")
                    logger.error(f"Ошибка удаления {path}: {e}")

            session.commit()
        finally:
            session.close()

        return {"deleted": deleted, "errors": errors}
