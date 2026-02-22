#!/usr/bin/env python3
"""
Находит фото с координатами (0, 0) у которых есть pHash-дубликаты с реальными GPS,
и копирует координаты из дубликата.

Запуск в Docker:
    docker exec smart_photo_api python /app/scripts/fix_gps_from_duplicates.py
    docker exec smart_photo_api python /app/scripts/fix_gps_from_duplicates.py --apply
    docker exec smart_photo_api python /app/scripts/fix_gps_from_duplicates.py --threshold 10 --apply

Опции:
    --threshold N   Максимальное Hamming-расстояние (0=точные дубли, default: 6)
    --apply         Применить изменения в БД (по умолчанию dry-run)
    --all-formats   Не ограничивать совпадения одним форматом файла
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from db.database import DatabaseManager
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Lookup table: byte value → number of set bits
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def _hamming_one_vs_all(hash_i: list, chunks_b: list) -> np.ndarray:
    """
    Compute Hamming distance between one hash and all hashes in B.
    hash_i: list of 4 uint64 scalars
    chunks_b: list of 4 arrays shape (nb,) dtype uint64
    Returns: (nb,) int32 distances
    """
    nb = len(chunks_b[0])
    dist = np.zeros(nb, dtype=np.int32)
    for c in range(4):
        xor = np.bitwise_xor(hash_i[c], chunks_b[c])  # (nb,)
        for shift in range(0, 64, 8):
            dist += _POPCOUNT[((xor >> shift) & 0xFF).astype(np.int64)]
    return dist


def _parse_hashes(hashes_hex: list) -> list:
    """Parse list of 64-char hex strings into 4 uint64 chunk arrays."""
    n = len(hashes_hex)
    chunks = [np.zeros(n, dtype=np.uint64) for _ in range(4)]
    for idx, h in enumerate(hashes_hex):
        h_padded = h.ljust(64, '0')
        for c in range(4):
            chunks[c][idx] = np.uint64(int(h_padded[c * 16:(c + 1) * 16], 16))
    return chunks


def _format_group(path: str) -> str:
    """Normalize file format for same-format matching."""
    ext = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
    if ext in ('jpg', 'jpeg'):
        return 'jpg'
    if ext in ('heic', 'heif'):
        return 'heic'
    if ext in ('nef', 'cr2', 'arw', 'dng', 'raf', 'orf', 'rw2'):
        return 'raw'
    return ext


def main():
    parser = argparse.ArgumentParser(description='Copy GPS from pHash duplicates to (0,0) photos')
    parser.add_argument('--threshold', type=int, default=6,
                        help='Max Hamming distance for match (default: 6)')
    parser.add_argument('--apply', action='store_true',
                        help='Write changes to DB (default: dry-run)')
    parser.add_argument('--all-formats', action='store_true',
                        help='Match across different file formats')
    args = parser.parse_args()

    mode = 'APPLY' if args.apply else 'DRY-RUN'
    logger.info(f"Mode: {mode} | threshold: {args.threshold} | all-formats: {args.all_formats}")

    db = DatabaseManager(settings.DATABASE_URL)
    session = db.get_session()

    try:
        # --- 1. Load zero-GPS photos with pHash ---
        zero_rows = session.execute(text("""
            SELECT image_id, file_path, file_name, phash
            FROM photo_index
            WHERE latitude = 0 AND longitude = 0
              AND phash IS NOT NULL AND phash != ''
            ORDER BY image_id
        """)).fetchall()

        if not zero_rows:
            logger.info("No photos with (0,0) coordinates and pHash found. Nothing to do.")
            return

        logger.info(f"Photos with (0,0) coords: {len(zero_rows)}")

        # --- 2. Load valid-GPS photos with pHash ---
        gps_rows = session.execute(text("""
            SELECT image_id, file_path, latitude, longitude, phash
            FROM photo_index
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
              AND NOT (latitude = 0 AND longitude = 0)
              AND phash IS NOT NULL AND phash != ''
            ORDER BY image_id
        """)).fetchall()

        if not gps_rows:
            logger.info("No photos with valid GPS and pHash found. Nothing to do.")
            return

        logger.info(f"Photos with valid GPS: {len(gps_rows)}")

        # --- 3. Build numpy arrays for GPS set (loaded once) ---
        gps_paths  = [r[1] for r in gps_rows]
        gps_lats   = [r[2] for r in gps_rows]
        gps_lons   = [r[3] for r in gps_rows]
        gps_hashes = [r[4] for r in gps_rows]
        gps_formats = [_format_group(p) for p in gps_paths]

        chunks_gps = _parse_hashes(gps_hashes)
        gps_formats_arr = np.array(gps_formats)

        # --- 4. For each (0,0) photo compute distances against full GPS set ---
        logger.info("Computing pHash distances (one-vs-all, no matrix)...")

        updates = []

        for i, row in enumerate(zero_rows):
            image_id, file_path, file_name, phash_hex = row

            if (i + 1) % 500 == 0:
                logger.info(f"  {i + 1}/{len(zero_rows)} processed, {len(updates)} matches so far")

            # Parse this single hash into 4 uint64 scalars
            h = phash_hex.ljust(64, '0')
            hash_i = [np.uint64(int(h[c * 16:(c + 1) * 16], 16)) for c in range(4)]

            # Distances: (n_gps,)
            dists = _hamming_one_vs_all(hash_i, chunks_gps)

            # Mask cross-format pairs unless --all-formats
            if not args.all_formats:
                zero_fmt = _format_group(file_path)
                dists[gps_formats_arr != zero_fmt] = 9999

            best_j = int(np.argmin(dists))
            best_dist = int(dists[best_j])

            if best_dist <= args.threshold:
                updates.append((
                    image_id,
                    file_path,
                    file_name,
                    gps_lats[best_j],
                    gps_lons[best_j],
                    gps_paths[best_j],
                    best_dist,
                ))

        # --- 5. Report ---
        logger.info(f"\n{'='*60}")
        logger.info(f"Found {len(updates)} matches (threshold ≤ {args.threshold}):")
        logger.info(f"{'='*60}")
        for image_id, path, name, lat, lon, src_path, dist in updates:
            logger.info(
                f"  [{image_id}] {name}\n"
                f"    → lat={lat:.6f}, lon={lon:.6f}  (dist={dist})\n"
                f"    from: {src_path}"
            )

        no_match = len(zero_rows) - len(updates)
        logger.info(f"\nMatched: {len(updates)}, No match: {no_match}")

        if not updates:
            return

        # --- 6. Apply ---
        if args.apply:
            logger.info(f"\nApplying {len(updates)} GPS updates...")
            for image_id, path, name, lat, lon, src_path, dist in updates:
                session.execute(text("""
                    UPDATE photo_index
                    SET latitude = :lat, longitude = :lon
                    WHERE image_id = :id
                """), {"lat": lat, "lon": lon, "id": image_id})
            session.commit()
            logger.info(f"Done. Updated {len(updates)} photos.")
        else:
            logger.info(f"\nDry-run: no changes made. Add --apply to write to DB.")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        session.rollback()
        sys.exit(1)
    finally:
        session.close()


if __name__ == '__main__':
    main()
