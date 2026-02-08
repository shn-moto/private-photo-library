#!/usr/bin/env python3
"""
Test 256-bit pHash on files from previous (64-bit) duplicate report.

Computes new 256-bit hashes for all files mentioned in the report,
then runs duplicate detection to verify accuracy before full reindex.

Usage:
    python scripts/test_phash256.py
    python scripts/test_phash256.py --report reports/duplicates_phash.txt --threshold 0
    python scripts/test_phash256.py --threshold 4       # near-duplicates
    python scripts/test_phash256.py --all-types         # match across formats (HEIC vs JPG)
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
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
DOCKER_PHOTOS_PATH = "/photos"
MAX_SIZE = 256
DEFAULT_REPORT = Path(__file__).parent.parent / "reports" / "duplicates_phash.txt"

# Popcount lookup table
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def docker_to_host(docker_path: str) -> str:
    return docker_path.replace(DOCKER_PHOTOS_PATH, PHOTOS_HOST_PATH, 1)


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
    except Exception as e:
        return None


def parse_report(report_path: Path) -> list[list[str]]:
    """Parse report, return list of groups (each group = list of docker paths)."""
    groups = []
    current = []
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("=== Группа "):
                if current:
                    groups.append(current)
                current = []
            elif line.startswith("KEEP") or line.startswith("DELETE"):
                path = line.split("]", 1)[1].strip()
                current.append(path)
    if current:
        groups.append(current)
    return groups


def popcount_array(arr: np.ndarray) -> np.ndarray:
    result = np.zeros(len(arr), dtype=np.int32)
    for shift in range(0, 64, 8):
        byte_vals = ((arr >> shift) & 0xFF).astype(np.int64)
        result += _POPCOUNT_TABLE[byte_vals]
    return result


def _get_format(path: str) -> str:
    """Extract normalized format group from path."""
    ext = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
    # Group similar formats
    if ext in ('jpg', 'jpeg'):
        return 'jpg'
    if ext in ('heic', 'heif'):
        return 'heic'
    if ext in ('nef', 'cr2', 'arw', 'dng', 'raf', 'orf', 'rw2'):
        return 'raw'
    return ext


def find_duplicates(hashes: dict[str, str], threshold: int,
                    same_format_only: bool = True) -> list[list[str]]:
    """Find duplicate groups from {path: hash_hex} dict."""
    paths = list(hashes.keys())
    hex_vals = list(hashes.values())
    n = len(paths)

    # Precompute formats for same-format filtering
    formats = [_get_format(p) for p in paths] if same_format_only else None

    # Parse into 4 x uint64 chunks
    num_chunks = 4
    chunks = [np.zeros(n, dtype=np.uint64) for _ in range(num_chunks)]
    for idx, h in enumerate(hex_vals):
        h_padded = h.ljust(64, '0')
        for c in range(num_chunks):
            chunk_hex = h_padded[c * 16:(c + 1) * 16]
            chunks[c][idx] = np.uint64(int(chunk_hex, 16))

    # Find pairs
    pairs = set()
    for i in range(n):
        total_dist = np.zeros(n - i - 1, dtype=np.int32)
        for c in range(num_chunks):
            xor = np.bitwise_xor(chunks[c][i], chunks[c][i + 1:])
            total_dist += popcount_array(xor)
        match_indices = np.where(total_dist <= threshold)[0]
        for mi in match_indices:
            j = i + 1 + mi
            # Skip cross-format pairs unless all-types mode
            if same_format_only and formats[i] != formats[j]:
                continue
            pairs.add((i, j))

    if not pairs:
        return []

    # Union-Find
    parent = {}
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for a, b in pairs:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    groups_map = defaultdict(list)
    all_in_pairs = set()
    for a, b in pairs:
        all_in_pairs.add(a)
        all_in_pairs.add(b)
    for idx in all_in_pairs:
        groups_map[find(idx)].append(paths[idx])

    return [sorted(g) for g in groups_map.values() if len(g) > 1]


def main():
    parser = argparse.ArgumentParser(description="Test 256-bit pHash on report files")
    parser.add_argument("--report", "-r", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--threshold", "-t", type=int, default=0)
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--all-types", action="store_true",
                       help="Match across file formats (default: same format only)")
    parser.add_argument("--output", "-o", type=Path,
                       default=Path(__file__).parent.parent / "reports" / "test_phash256.txt")
    args = parser.parse_args()

    if not args.report.exists():
        print(f"Report not found: {args.report}")
        sys.exit(1)

    # 1. Parse old report
    old_groups = parse_report(args.report)
    all_paths = set()
    for g in old_groups:
        all_paths.update(g)

    print(f"Report: {args.report}")
    print(f"Old 64-bit groups: {len(old_groups)}")
    print(f"Unique files: {len(all_paths)}")
    print(f"Threshold: {args.threshold}")
    print(f"Match mode: {'all types' if args.all_types else 'same format only'}")
    print(f"Hash size: 16 (256-bit)\n")

    # 2. Compute 256-bit pHash
    tasks = []
    for docker_path in all_paths:
        host_path = docker_to_host(docker_path)
        tasks.append((docker_path, host_path))

    hashes = {}
    failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for docker_path, host_path in tasks:
            if not os.path.exists(host_path):
                failed += 1
                continue
            futures[pool.submit(compute_phash, host_path)] = docker_path

        done = 0
        for future in as_completed(futures):
            docker_path = futures[future]
            result = future.result()
            done += 1
            if result:
                hashes[docker_path] = result
            else:
                failed += 1
            if done % 100 == 0:
                print(f"  Computing: {done}/{len(futures)}...", end='\r')

    elapsed = time.time() - start
    print(f"Computed: {len(hashes)} hashes, {failed} failed in {elapsed:.1f}s "
          f"({len(hashes)/max(elapsed,0.1):.0f} img/s)\n")

    # 3. Find duplicates with 256-bit hashes
    same_format_only = not args.all_types
    new_groups = find_duplicates(hashes, args.threshold, same_format_only=same_format_only)
    new_groups.sort(key=lambda g: -len(g))

    # 4. Write results to file and console
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        def out(line=""):
            print(line)
            f.write(line + "\n")

        mode = "all types" if args.all_types else "same format only"
        out(f"# Test 256-bit pHash (hash_size=16)")
        out(f"# Source report: {args.report}")
        out(f"# Threshold: {args.threshold}, match mode: {mode}")
        out(f"# Computed: {len(hashes)} hashes, {failed} failed in {elapsed:.1f}s")
        out()
        out(f"{'='*60}")
        out(f"RESULTS: 256-bit pHash, threshold={args.threshold}")
        out(f"{'='*60}")
        out(f"Old 64-bit groups: {len(old_groups)}")
        out(f"New 256-bit groups: {len(new_groups)}")
        out(f"Reduction: {len(old_groups)} -> {len(new_groups)} "
              f"({len(old_groups) - len(new_groups)} false positives removed)")
        out()

        # New duplicate groups (same format as save_report — compatible with copy_duplicate_group.py)
        for i, group in enumerate(new_groups, 1):
            # Sort: largest file first (KEEP), rest DELETE
            sized = []
            for path in group:
                host = docker_to_host(path)
                size = os.path.getsize(host) if os.path.exists(host) else 0
                sized.append((size, path))
            sized.sort(key=lambda x: (-x[0], x[1]))

            out(f"=== Группа {i} ({len(sized)} файлов) ===")
            for j, (size, path) in enumerate(sized):
                prefix = "KEEP  " if j == 0 else "DELETE"
                size_mb = size / 1024 / 1024
                out(f"{prefix} [{size_mb:6.1f} MB] {path}")
            out()

        # Show false positives that were removed
        out(f"{'='*60}")
        out(f"FALSE POSITIVES REMOVED (old groups now split):")
        out(f"{'='*60}")

        new_membership = {}
        for gi, group in enumerate(new_groups):
            for path in group:
                new_membership[path] = gi

        split_count = 0
        for old_idx, old_g in enumerate(old_groups, 1):
            new_gids = set()
            for p in old_g:
                if p in new_membership:
                    new_gids.add(new_membership[p])
                else:
                    new_gids.add(f"solo_{p}")
            if len(new_gids) > 1 or (len(new_gids) == 1 and isinstance(list(new_gids)[0], str)):
                split_count += 1
                out(f"\nOld group {old_idx} ({len(old_g)} files) -> now {len(new_gids)} separate:")
                for p in old_g:
                    name = p.rsplit('/', 1)[-1]
                    h = hashes.get(p, "N/A")
                    tag = f"new group {new_membership[p]+1}" if p in new_membership else "no match"
                    out(f"  {name:40s} hash={h[:20]}... -> {tag}")

        out(f"\nTotal old groups split/removed: {split_count}")
        out(f"Total old groups kept as-is: {len(old_groups) - split_count}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
