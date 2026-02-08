#!/usr/bin/env python3
"""
Restore false duplicates from .photo_duplicates back to PHOTO directory.

Reads a deletion report, computes 256-bit pHash for KEEP and DELETE files,
restores DELETE files that are NOT true pHash duplicates of their KEEP file.

Usage:
    python scripts/restore_false_duplicates.py --report reports/duplicates_deleted4.txt
    python scripts/restore_false_duplicates.py --report reports/duplicates_deleted4.txt --dry-run
    python scripts/restore_false_duplicates.py --report reports/duplicates_deleted4.txt --threshold 4
"""

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

import imagehash

PHOTOS_HOST_PATH = os.getenv("PHOTOS_HOST_PATH", "D:/PHOTO").replace("\\", "/")
DUPLICATES_HOST_PATH = os.getenv("DUPLICATES_HOST_PATH",
                                  str(Path(PHOTOS_HOST_PATH).parent / ".photo_duplicates")).replace("\\", "/")
DOCKER_PHOTOS_PATH = "/photos"
MAX_SIZE = 256


def docker_to_host(docker_path: str) -> str:
    """Convert Docker /photos/... path to Windows host path."""
    return docker_path.replace(DOCKER_PHOTOS_PATH, PHOTOS_HOST_PATH, 1)


def docker_to_duplicates(docker_path: str) -> str:
    """Convert Docker /photos/... path to duplicates host path."""
    return docker_path.replace(DOCKER_PHOTOS_PATH, DUPLICATES_HOST_PATH, 1)


def compute_phash(file_path: str) -> str | None:
    ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
    try:
        if ext in ('nef', 'cr2', 'arw', 'dng', 'raf', 'orf', 'rw2'):
            if not HAS_RAWPY:
                return None
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                img = Image.fromarray(rgb)
                img.thumbnail((MAX_SIZE, MAX_SIZE))
        else:
            img = Image.open(file_path)
            img.draft('RGB', (MAX_SIZE, MAX_SIZE))
            img = img.convert('RGB')
            img.thumbnail((MAX_SIZE, MAX_SIZE))

        return str(imagehash.phash(img, hash_size=16))
    except Exception:
        return None


def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    # Pad to 64 chars
    h1 = h1.ljust(64, '0')
    h2 = h2.ljust(64, '0')
    val1 = int(h1, 16)
    val2 = int(h2, 16)
    return bin(val1 ^ val2).count('1')


def parse_report(report_path: Path) -> list[dict]:
    """Parse report, return list of groups with keep/delete files."""
    groups = []
    current = None
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("=== Группа "):
                if current:
                    groups.append(current)
                nr = int(line.split()[2])
                current = {"nr": nr, "keep": None, "delete": []}
            elif current is not None:
                if line.startswith("KEEP"):
                    path = line.split("]", 1)[1].strip()
                    current["keep"] = path
                elif line.startswith("DELETE"):
                    path = line.split("]", 1)[1].strip()
                    current["delete"].append(path)
    if current:
        groups.append(current)
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Restore false duplicates from .photo_duplicates")
    parser.add_argument("--report", "-r", type=Path, required=True,
                       help="Deletion report file")
    parser.add_argument("--threshold", "-t", type=int, default=0,
                       help="Max Hamming distance to consider as duplicate (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only show what would be restored, don't move files")
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--output", "-o", type=Path,
                       default=Path(__file__).parent.parent / "reports" / "restore_results.txt")
    args = parser.parse_args()

    if not args.report.exists():
        print(f"Report not found: {args.report}")
        sys.exit(1)

    groups = parse_report(args.report)
    print(f"Report: {args.report}")
    print(f"Groups: {len(groups)}")
    print(f"Threshold: {args.threshold}")
    print(f"Photos dir: {PHOTOS_HOST_PATH}")
    print(f"Duplicates dir: {DUPLICATES_HOST_PATH}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'RESTORE'}\n")

    # Collect all files to hash
    all_files = {}  # host_path -> docker_path
    for g in groups:
        if g["keep"]:
            host = docker_to_host(g["keep"])
            if os.path.exists(host):
                all_files[host] = g["keep"]
        for dp in g["delete"]:
            # Deleted files are in duplicates dir
            dup_host = docker_to_duplicates(dp)
            if os.path.exists(dup_host):
                all_files[dup_host] = dp

    print(f"Files to hash: {len(all_files)}")

    # Compute hashes in parallel
    hashes = {}  # docker_path -> hash
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compute_phash, hp): dp
                   for hp, dp in all_files.items()}
        done = 0
        for future in as_completed(futures):
            docker_path = futures[future]
            result = future.result()
            done += 1
            if result:
                hashes[docker_path] = result
            if done % 100 == 0:
                print(f"  Hashing: {done}/{len(futures)}...", end='\r')

    elapsed = time.time() - start
    print(f"Hashed: {len(hashes)} in {elapsed:.1f}s\n")

    # Analyze each group
    to_restore = []  # (docker_path, dup_host_path, original_host_path, distance)
    true_dupes = 0
    missing_keep = 0
    missing_delete = 0
    no_hash = 0

    for g in groups:
        keep_path = g["keep"]
        keep_hash = hashes.get(keep_path)

        if not keep_hash:
            # KEEP file missing or unhashable — check if it exists
            keep_host = docker_to_host(keep_path)
            if not os.path.exists(keep_host):
                missing_keep += 1
            else:
                no_hash += 1
            continue

        for del_path in g["delete"]:
            dup_host = docker_to_duplicates(del_path)
            orig_host = docker_to_host(del_path)

            if not os.path.exists(dup_host):
                missing_delete += 1
                continue

            del_hash = hashes.get(del_path)
            if not del_hash:
                no_hash += 1
                continue

            dist = hamming_distance(keep_hash, del_hash)
            if dist > args.threshold:
                to_restore.append((del_path, dup_host, orig_host, dist))
            else:
                true_dupes += 1

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        def out(line=""):
            print(line)
            f.write(line + "\n")

        out(f"# Restore false duplicates")
        out(f"# Report: {args.report}")
        out(f"# Threshold: {args.threshold}")
        out(f"# Mode: {'DRY RUN' if args.dry_run else 'RESTORE'}")
        out()
        out(f"Total groups: {len(groups)}")
        out(f"True duplicates (kept deleted): {true_dupes}")
        out(f"False duplicates (to restore): {len(to_restore)}")
        out(f"Missing KEEP files: {missing_keep}")
        out(f"Missing DELETE files: {missing_delete}")
        out(f"Unhashable: {no_hash}")
        out()

        # Restore files
        restored = 0
        errors = 0
        for del_path, dup_host, orig_host, dist in to_restore:
            name = del_path.rsplit('/', 1)[-1]
            out(f"RESTORE [dist={dist:3d}] {name}")
            out(f"  from: {dup_host}")
            out(f"  to:   {orig_host}")

            if not args.dry_run:
                try:
                    os.makedirs(os.path.dirname(orig_host), exist_ok=True)
                    shutil.move(dup_host, orig_host)
                    restored += 1
                    out(f"  OK")
                except Exception as e:
                    errors += 1
                    out(f"  ERROR: {e}")
            out()

        out(f"{'='*60}")
        if args.dry_run:
            out(f"DRY RUN: {len(to_restore)} files would be restored")
        else:
            out(f"Restored: {restored}, Errors: {errors}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
