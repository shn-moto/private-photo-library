#!/usr/bin/env python3
"""
Compute perceptual hashes on Windows host (fast local I/O).

Reads file list from API, computes pHash locally, sends results back.
Much faster than computing inside Docker (avoids slow volume mount I/O).

Requirements (host):
    pip install imagehash Pillow pillow-heif rawpy httpx python-dotenv

Usage:
    python scripts/compute_phash.py
    python scripts/compute_phash.py --api-url http://localhost:8000
    python scripts/compute_phash.py --batch 1000 --workers 4
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import httpx
import imagehash
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

# Docker path mapping
PHOTOS_HOST_PATH = os.getenv("PHOTOS_HOST_PATH", "D:/PHOTO").replace("\\", "/")
DOCKER_PHOTOS_PATH = "/photos"
MAX_SIZE = 256


def docker_to_host(docker_path: str) -> str:
    """Convert Docker path to Windows host path."""
    return docker_path.replace(DOCKER_PHOTOS_PATH, PHOTOS_HOST_PATH, 1)


def compute_phash(file_path: str) -> str | None:
    """Compute pHash for a single file. Returns 64-char hex (256-bit) or None."""
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


def process_file(item: dict) -> tuple[int, str | None]:
    """Process a single file: returns (image_id, phash_hex or None)."""
    host_path = docker_to_host(item["path"])
    if not os.path.exists(host_path):
        return item["id"], None
    return item["id"], compute_phash(host_path)


def main():
    parser = argparse.ArgumentParser(description="Compute pHash on Windows host")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--batch", type=int, default=5000,
                       help="Files per API request (default: 5000)")
    parser.add_argument("--send-batch", type=int, default=500,
                       help="Hashes per update request (default: 500)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Parallel workers (default: 4)")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.api_url, timeout=60)
    total_computed = 0
    total_failed = 0
    start_time = time.time()

    print(f"API: {args.api_url}")
    print(f"Photos: {PHOTOS_HOST_PATH}")
    print(f"Workers: {args.workers}")

    while True:
        # Get pending files
        resp = client.get(f"/phash/pending?limit={args.batch}")
        if resp.status_code != 200:
            print(f"ERROR: API returned {resp.status_code}: {resp.text[:200]}")
            print("Did you rebuild the container? docker-compose build api && docker-compose up -d api")
            sys.exit(1)
        data = resp.json()
        files = data["files"]

        if not files:
            print("All photos have pHash. Done.")
            break

        remaining = data["count"]
        print(f"\nPending: {remaining} | Processing batch of {len(files)}...")

        # Compute pHash in parallel, send results incrementally
        pending_hashes = {}
        pending_failed = []
        batch_computed = 0
        batch_failed = 0

        def flush_results():
            """Send accumulated hashes/failures to API."""
            nonlocal pending_hashes, pending_failed
            if pending_hashes:
                for i in range(0, len(pending_hashes), args.send_batch):
                    items = list(pending_hashes.items())[i:i + args.send_batch]
                    resp = client.post("/phash/update", json={"hashes": dict(items)})
                    resp.raise_for_status()
                pending_hashes = {}
            if pending_failed:
                for i in range(0, len(pending_failed), args.send_batch):
                    chunk = pending_failed[i:i + args.send_batch]
                    resp = client.post("/phash/update", json={"hashes": {}, "failed": chunk})
                    resp.raise_for_status()
                pending_failed = []

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_file, item): item for item in files}
            done = 0

            for future in as_completed(futures):
                image_id, phash_hex = future.result()
                done += 1

                if phash_hex:
                    pending_hashes[str(image_id)] = phash_hex
                    batch_computed += 1
                else:
                    pending_failed.append(image_id)
                    batch_failed += 1

                # Send results every send_batch files
                if len(pending_hashes) + len(pending_failed) >= args.send_batch:
                    flush_results()

                if done % 200 == 0:
                    elapsed = time.time() - start_time
                    total = total_computed + batch_computed
                    speed = total / elapsed if elapsed > 0 else 0
                    print(f"  {done}/{len(files)} | total: {total} | "
                          f"{speed:.0f} img/s", end='\r')

        # Send remaining results
        flush_results()

        total_computed += batch_computed
        total_failed += batch_failed
        elapsed = time.time() - start_time
        speed = total_computed / elapsed if elapsed > 0 else 0
        eta = (remaining - len(files)) / speed if speed > 0 else 0

        print(f"  Batch done: {batch_computed} ok, {batch_failed} failed (marked) | "
              f"Total: {total_computed} | {speed:.0f} img/s | ETA: {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\nDone: {total_computed} computed, {total_failed} failed in {elapsed:.0f}s "
          f"({total_computed / elapsed:.0f} img/s)")


if __name__ == "__main__":
    main()
