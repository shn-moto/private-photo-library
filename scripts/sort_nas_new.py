#!/usr/bin/env python3
# Sort photos in NAS_new into 3 categories:
# 1. GPS photos  -> NAS_SORTED/gps/YYYY/MM/  (by shooting date from EXIF)
# 2. Good photos -> NAS_SORTED/photos/jpg|png|heic/  (JPG/PNG/HEIC > 100KB)
# 3. Web junk    -> NAS_SORTED/web_junk/  (WEBP or < 100KB)

import os
import shutil
import struct
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# --- Config ---
SOURCE_DIR = Path(r"D:\NAS_new")
OUTPUT_DIR = Path(r"D:\NAS_SORTED")

GPS_DIR = OUTPUT_DIR / "gps"
PHOTOS_DIR = OUTPUT_DIR / "photos"
JUNK_DIR = OUTPUT_DIR / "web_junk"

SIZE_THRESHOLD = 100 * 1024  # 100 KB
GOOD_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.heic', '.heif'})
IMAGE_EXTENSIONS = frozenset({
    '.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.bmp',
    '.nef', '.cr2', '.arw', '.dng', '.raf', '.orf', '.rw2',
})


def get_exif_gps_and_date(file_path: Path) -> tuple:
    """Extract GPS coords and date from EXIF. Returns (lat, lon, date) or (None, None, None)."""
    try:
        img = Image.open(file_path)
        exif_data = img.getexif()
        if not exif_data:
            return None, None, None

        # Get date
        date = None
        # DateTimeOriginal (36867) or DateTime (306)
        date_str = exif_data.get(36867) or exif_data.get(306)
        if date_str:
            try:
                date = datetime.strptime(date_str.strip(), "%Y:%m:%d %H:%M:%S")
            except (ValueError, AttributeError):
                pass

        # Get GPS from IFD
        gps_info = {}
        ifd = exif_data.get_ifd(0x8825)  # GPSInfo IFD
        if ifd:
            for tag_id, value in ifd.items():
                tag_name = GPSTAGS.get(tag_id, tag_id)
                gps_info[tag_name] = value

        lat = _convert_gps(gps_info.get('GPSLatitude'), gps_info.get('GPSLatitudeRef'))
        lon = _convert_gps(gps_info.get('GPSLongitude'), gps_info.get('GPSLongitudeRef'))

        return lat, lon, date
    except Exception:
        return None, None, None


def _convert_gps(coords, ref) -> float | None:
    """Convert GPS DMS tuple to decimal degrees."""
    if not coords or not ref:
        return None
    try:
        d, m, s = coords
        d = float(d)
        m = float(m)
        s = float(s)
        result = d + m / 60 + s / 3600
        if ref in ('S', 'W'):
            result = -result
        return result
    except Exception:
        return None


def scan_files(root: Path) -> list[Path]:
    """Find all image files."""
    files = []
    for entry in root.rglob('*'):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(entry)
    return files


def move_file(src: Path, dest: Path):
    """Move file, handle name collisions."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        i = 1
        while dest.exists():
            dest = dest.parent / f"{stem}_{i}{suffix}"
            i += 1
    shutil.move(str(src), str(dest))


def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source dir not found: {SOURCE_DIR}")
        sys.exit(1)

    print(f"Scanning {SOURCE_DIR} ...")
    files = scan_files(SOURCE_DIR)
    print(f"Found {len(files)} images")

    if not files:
        return

    # Create output dirs
    for d in (GPS_DIR, PHOTOS_DIR, JUNK_DIR):
        d.mkdir(parents=True, exist_ok=True)

    gps_count = 0
    photo_count = 0
    junk_count = 0
    other_count = 0

    for fpath in tqdm(files, desc="Sorting", unit="img"):
        ext = fpath.suffix.lower()
        size = fpath.stat().st_size

        # Rule 3: WEBP or < 100KB → web junk
        if ext == '.webp' or size < SIZE_THRESHOLD:
            dest = JUNK_DIR / fpath.name
            move_file(fpath, dest)
            junk_count += 1
            continue

        # Rule 1: Has GPS → sort by year/month
        lat, lon, date = get_exif_gps_and_date(fpath)
        if lat is not None and lon is not None:
            if date:
                year = str(date.year)
                month = f"{date.month:02d}"
            else:
                year = "unknown"
                month = "unknown"
            dest = GPS_DIR / year / month / fpath.name
            move_file(fpath, dest)
            gps_count += 1
            continue

        # Rule 2: JPG/PNG/HEIC > 100KB → by type
        if ext in GOOD_EXTENSIONS:
            type_folder = ext.lstrip('.').lower()
            if type_folder == 'jpeg':
                type_folder = 'jpg'
            elif type_folder == 'heif':
                type_folder = 'heic'
            dest = PHOTOS_DIR / type_folder / fpath.name
            move_file(fpath, dest)
            photo_count += 1
            continue

        # Everything else (RAW, BMP, etc. without GPS)
        dest = PHOTOS_DIR / "other" / fpath.name
        move_file(fpath, dest)
        other_count += 1

    print(f"\n{'=' * 50}")
    print(f"Total:      {len(files)}")
    print(f"GPS sorted: {gps_count}")
    print(f"Photos:     {photo_count}")
    print(f"Web junk:   {junk_count}")
    print(f"Other:      {other_count}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
