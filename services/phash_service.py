"""Perceptual hash (pHash) service for exact/near-duplicate detection."""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image
from sqlalchemy import text

logger = logging.getLogger(__name__)

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    logger.warning("imagehash not installed. pip install imagehash")

# Lookup table: byte value -> number of set bits
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

# Максимальный размер для загрузки (pHash нужно только 32x32)
_MAX_SIZE = 256


def _load_image_for_hash(file_path: str) -> Optional[Image.Image]:
    """
    Load image as small PIL Image for hashing.
    Much faster than loading full-res via ImageProcessor.
    """
    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''

    try:
        # RAW files: need rawpy
        if ext in ('nef', 'cr2', 'arw', 'dng', 'raf', 'orf', 'rw2'):
            try:
                import rawpy
                with rawpy.imread(file_path) as raw:
                    # half_size=True: decode at half resolution (4x faster)
                    rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                    img = Image.fromarray(rgb)
                    img.thumbnail((_MAX_SIZE, _MAX_SIZE))
                    return img
            except Exception as e:
                logger.debug(f"RAW load failed {file_path}: {e}")
                return None

        # HEIC: need pillow-heif
        if ext in ('heic', 'heif'):
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass

        # Standard formats + HEIC (after registration)
        img = Image.open(file_path)
        img.draft('RGB', (_MAX_SIZE, _MAX_SIZE))  # JPEG: fast downscaled load
        img = img.convert('RGB')
        img.thumbnail((_MAX_SIZE, _MAX_SIZE))
        return img

    except Exception as e:
        logger.debug(f"Image load failed {file_path}: {e}")
        return None


class PHashService:
    """Compute and compare perceptual hashes for duplicate detection."""

    def __init__(self, session_factory):
        self._get_session = session_factory

    @staticmethod
    def compute_phash_from_pil(pil_image: Image.Image, hash_size: int = 16) -> Optional[str]:
        """Compute pHash from PIL Image. Returns 64-char hex string (256-bit)."""
        if not HAS_IMAGEHASH:
            return None
        try:
            h = imagehash.phash(pil_image, hash_size=hash_size)
            return str(h)
        except Exception as e:
            logger.warning(f"pHash failed: {e}")
            return None

    @staticmethod
    def _popcount_array(arr: np.ndarray) -> np.ndarray:
        """Vectorized popcount for uint64 array via byte lookup table."""
        result = np.zeros(len(arr), dtype=np.int32)
        for shift in range(0, 64, 8):
            byte_vals = ((arr >> shift) & 0xFF).astype(np.int64)
            result += _POPCOUNT_TABLE[byte_vals]
        return result

    def reindex(self, batch_size: int = 100, progress_callback=None,
                stop_flag=None) -> dict:
        """Compute pHash for all photos that don't have one yet.

        Args:
            stop_flag: callable returning True to stop early (checked per file)
        """
        session = self._get_session()
        stats = {"total": 0, "computed": 0, "failed": 0}

        try:
            stats["total"] = session.execute(
                text("SELECT COUNT(*) FROM photo_index WHERE phash IS NULL")
            ).scalar()

            if stats["total"] == 0:
                logger.info("All photos already have pHash")
                return stats

            logger.info(f"Computing pHash for {stats['total']} photos "
                       f"(imagehash: {HAS_IMAGEHASH}, batch: {batch_size})")

            if progress_callback:
                progress_callback(0, 0, stats["total"], 0, 0)

            start_time = time.time()
            last_id = 0

            while True:
                rows = session.execute(text("""
                    SELECT image_id, file_path FROM photo_index
                    WHERE phash IS NULL AND image_id > :last_id
                    ORDER BY image_id LIMIT :batch_size
                """), {"last_id": last_id, "batch_size": batch_size}).fetchall()

                if not rows:
                    break

                uncommitted = 0
                for image_id, file_path in rows:
                    last_id = image_id
                    pil_img = _load_image_for_hash(file_path)
                    if pil_img is None:
                        stats["failed"] += 1
                    else:
                        phash_hex = self.compute_phash_from_pil(pil_img)
                        if phash_hex is None:
                            stats["failed"] += 1
                        else:
                            session.execute(
                                text("UPDATE photo_index SET phash = :phash WHERE image_id = :id"),
                                {"phash": phash_hex, "id": image_id}
                            )
                            stats["computed"] += 1
                    uncommitted += 1

                    # Commit every 50 files (balance: speed vs resumability)
                    if uncommitted >= 50:
                        session.commit()
                        uncommitted = 0

                    # Check stop flag (commit pending before exit)
                    if stop_flag and stop_flag():
                        if uncommitted > 0:
                            session.commit()
                        logger.info("pHash reindex stopped by user")
                        elapsed = time.time() - start_time
                        logger.info(f"pHash stopped: {stats['computed']} computed, "
                                   f"{stats['failed']} failed in {elapsed:.1f}s")
                        return stats

                    processed = stats["computed"] + stats["failed"]
                    if progress_callback:
                        elapsed = time.time() - start_time
                        speed = processed / elapsed if elapsed > 0 else 0
                        eta = (stats["total"] - processed) / speed if speed > 0 else 0
                        progress_callback(stats["computed"], stats["failed"],
                                          stats["total"], speed, eta)

                # Commit remaining uncommitted files in this batch
                if uncommitted > 0:
                    session.commit()

                processed = stats["computed"] + stats["failed"]
                elapsed = time.time() - start_time
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (stats["total"] - processed) / speed if speed > 0 else 0
                logger.info(f"pHash: {processed}/{stats['total']} "
                           f"({processed * 100 // max(stats['total'], 1)}%) | "
                           f"{speed:.0f} img/s | ETA: {eta:.0f}s")

            elapsed = time.time() - start_time
            logger.info(f"pHash reindex done: {stats['computed']} computed, "
                       f"{stats['failed']} failed in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"pHash reindex error: {e}", exc_info=True)
            session.rollback()
            raise
        finally:
            session.close()

        return stats

    @staticmethod
    def _get_format(path: str) -> str:
        """Extract normalized format group from path."""
        ext = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
        if ext in ('jpg', 'jpeg'):
            return 'jpg'
        if ext in ('heic', 'heif'):
            return 'heic'
        if ext in ('nef', 'cr2', 'arw', 'dng', 'raf', 'orf', 'rw2'):
            return 'raw'
        return ext

    def find_duplicates(self, threshold: int = 0, limit: int = 50000,
                        path_filter: str = None,
                        same_format_only: bool = True) -> list:
        """
        Find duplicate groups by pHash Hamming distance.

        Loads all hashes as uint64, vectorized XOR+popcount.

        Args:
            threshold: max Hamming distance (0=exact, <=6=near-duplicate)
            limit: max duplicate pairs
            path_filter: SQL LIKE filter
            same_format_only: only match within same file format (default True)
        """
        session = self._get_session()
        try:
            path_condition = ""
            params = {}
            if path_filter:
                path_condition = "AND file_path LIKE :path_filter"
                params["path_filter"] = path_filter

            rows = session.execute(text(f"""
                SELECT image_id, file_path, file_size, phash
                FROM photo_index
                WHERE phash IS NOT NULL AND phash != '' {path_condition}
                ORDER BY image_id
            """), params).fetchall()

            if not rows:
                return []

            logger.info(f"pHash search: {len(rows)} photos, threshold={threshold}, "
                       f"same_format={same_format_only}")
            start_time = time.time()

            ids = [r[0] for r in rows]
            paths = [r[1] for r in rows]
            sizes = [r[2] or 0 for r in rows]
            hashes_hex = [r[3] for r in rows]

            # Precompute formats for same-format filtering
            formats = [self._get_format(p) for p in paths] if same_format_only else None

            # Parse 256-bit hashes into 4 x uint64 chunks (columns)
            # Each hex hash = 64 chars = 4 chunks of 16 hex chars
            n = len(hashes_hex)
            num_chunks = 4
            hash_chunks = [np.zeros(n, dtype=np.uint64) for _ in range(num_chunks)]

            for idx, h in enumerate(hashes_hex):
                # Pad short hashes (legacy 16-char) with zeros on the right
                h_padded = h.ljust(64, '0')
                for c in range(num_chunks):
                    chunk_hex = h_padded[c * 16:(c + 1) * 16]
                    hash_chunks[c][idx] = np.uint64(int(chunk_hex, 16))

            pairs = set()
            file_info = {}

            for i in range(n):
                if len(pairs) >= limit:
                    break

                # XOR each chunk and sum popcounts for total Hamming distance
                total_dist = np.zeros(n - i - 1, dtype=np.int32)
                for c in range(num_chunks):
                    xor = np.bitwise_xor(hash_chunks[c][i], hash_chunks[c][i + 1:])
                    total_dist += self._popcount_array(xor)
                match_indices = np.where(total_dist <= threshold)[0]

                for mi in match_indices:
                    j = i + 1 + mi
                    # Skip cross-format pairs unless all-types mode
                    if same_format_only and formats[i] != formats[j]:
                        continue
                    pair_key = (ids[i], ids[j])
                    pairs.add(pair_key)
                    file_info[ids[i]] = {
                        'image_id': ids[i], 'path': paths[i], 'size': sizes[i]
                    }
                    file_info[ids[j]] = {
                        'image_id': ids[j], 'path': paths[j], 'size': sizes[j]
                    }

                    if len(pairs) >= limit:
                        break

                if (i + 1) % 10000 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"pHash: {i + 1}/{n}, пар: {len(pairs)}, {elapsed:.1f}s")

            elapsed = time.time() - start_time
            logger.info(f"pHash: {len(pairs)} пар за {elapsed:.1f}s")

            if not pairs:
                return []

            # Union-Find grouping
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
                    group.sort(key=lambda x: (-(x['size']), x['path']))
                    groups.append(group)
            groups.sort(key=lambda g: -len(g))

            return groups

        finally:
            session.close()

    def save_report(self, groups: list, output_file: str, threshold: int) -> dict:
        """Save duplicate report in same format as DuplicateFinder."""
        total_duplicates = sum(len(g) - 1 for g in groups)
        total_size_saved = sum(item['size'] for g in groups for item in g[1:])

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Отчёт о дубликатах (pHash)\n")
            f.write(f"# Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Hamming distance threshold: {threshold}\n")
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

        logger.info(f"Report saved: {output_file}")

        return {
            "total_groups": len(groups),
            "total_duplicates": total_duplicates,
            "size_saved_mb": round(total_size_saved / 1024 / 1024, 1),
        }
