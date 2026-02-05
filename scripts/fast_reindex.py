#!/usr/bin/env python3
"""
Fast reindex script using NTFS USN Journal (Windows only).

Run this script on Windows host to quickly detect new/modified files
and send them to the API for indexing.

Usage:
    python scripts/fast_reindex.py [--api-url http://localhost:8000] [--model SigLIP]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config.settings import settings

# Docker path mapping: PHOTOS_HOST_PATH -> /photos
PHOTOS_HOST_PATH = os.getenv("PHOTOS_HOST_PATH", "").replace("\\", "/")
DOCKER_PHOTOS_PATH = "/photos"

try:
    import httpx
except ImportError:
    print("Error: httpx not installed. Run: pip install httpx")
    sys.exit(1)


def host_to_docker_path(host_path: str) -> str:
    """Convert Windows host path to Docker container path.

    Example: H:/PHOTO/2024/img.jpg -> /photos/2024/img.jpg
    """
    if not PHOTOS_HOST_PATH:
        raise ValueError("PHOTOS_HOST_PATH not set in .env")

    # Normalize path separators
    normalized = str(host_path).replace("\\", "/")
    host_prefix = PHOTOS_HOST_PATH.rstrip("/")

    if not normalized.lower().startswith(host_prefix.lower()):
        raise ValueError(f"Path {host_path} is not under PHOTOS_HOST_PATH ({PHOTOS_HOST_PATH})")

    # Replace host prefix with docker path
    relative = normalized[len(host_prefix):]
    return DOCKER_PHOTOS_PATH + relative


def docker_to_host_path(docker_path: str) -> str:
    """Convert Docker container path to Windows host path.

    Example: /photos/2024/img.jpg -> H:/PHOTO/2024/img.jpg
    """
    if not PHOTOS_HOST_PATH:
        raise ValueError("PHOTOS_HOST_PATH not set in .env")

    if not docker_path.startswith(DOCKER_PHOTOS_PATH):
        raise ValueError(f"Path {docker_path} is not under {DOCKER_PHOTOS_PATH}")

    relative = docker_path[len(DOCKER_PHOTOS_PATH):]
    return PHOTOS_HOST_PATH.rstrip("/") + relative

try:
    from services.ntfs_change_tracker import NTFSChangeTracker, IS_WINDOWS, HAS_WIN32
except ImportError as e:
    print(f"Error importing NTFS tracker: {e}")
    IS_WINDOWS = os.name == 'nt'
    HAS_WIN32 = False

# Try to use Everything SDK for fast file listing (no admin required)
HAS_EVERYTHING = False
try:
    import ctypes
    from ctypes import wintypes

    # Load Everything SDK DLL
    EVERYTHING_DLL = None
    for dll_path in [
        "Everything64.dll",
        "Everything32.dll",
        os.path.join(os.path.dirname(__file__), "Everything64.dll"),
        r"C:\Program Files\Everything\Everything64.dll",
    ]:
        try:
            EVERYTHING_DLL = ctypes.WinDLL(dll_path)
            break
        except OSError:
            continue

    if EVERYTHING_DLL:
        # Define Everything API functions
        EVERYTHING_DLL.Everything_SetSearchW.argtypes = [ctypes.c_wchar_p]
        EVERYTHING_DLL.Everything_QueryW.argtypes = [ctypes.c_int]
        EVERYTHING_DLL.Everything_QueryW.restype = ctypes.c_int
        EVERYTHING_DLL.Everything_GetNumResults.restype = ctypes.c_int
        EVERYTHING_DLL.Everything_GetResultFullPathNameW.argtypes = [ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int]
        EVERYTHING_DLL.Everything_GetResultFullPathNameW.restype = ctypes.c_int
        EVERYTHING_DLL.Everything_GetLastError.restype = ctypes.c_int

        HAS_EVERYTHING = True
        print("Everything SDK loaded - fast file listing available")
except Exception as e:
    pass


def get_checkpoint_from_api(api_url: str, drive_letter: str) -> int:
    """Get last USN checkpoint from API."""
    try:
        response = httpx.get(f"{api_url}/scan/checkpoint/{drive_letter}", timeout=10)
        if response.status_code == 200:
            return response.json().get("last_usn", 0)
    except Exception as e:
        print(f"Warning: Could not get checkpoint from API: {e}")
    return 0


def save_checkpoint_to_api(api_url: str, drive_letter: str, usn: int, files_count: int = 0):
    """Save USN checkpoint to API."""
    try:
        response = httpx.post(
            f"{api_url}/scan/checkpoint",
            json={
                "drive_letter": drive_letter,
                "last_usn": usn,
                "files_count": files_count
            },
            timeout=10
        )
        if response.status_code == 200:
            print(f"Saved checkpoint: USN={usn}")
    except Exception as e:
        print(f"Warning: Could not save checkpoint to API: {e}")


def get_known_files_from_api(api_url: str) -> dict:
    """Get known files from API (for matching filenames to paths)."""
    try:
        response = httpx.get(f"{api_url}/files/index", timeout=60)
        if response.status_code == 200:
            data = response.json()
            return {Path(f["file_path"]).name: f["file_path"] for f in data.get("files", [])}
    except Exception as e:
        print(f"Warning: Could not get file index from API: {e}")
    return {}


def get_unindexed_files_from_api(api_url: str, model: str = None) -> list:
    """Get files that are not indexed for the specified model."""
    try:
        params = {}
        if model:
            params["model"] = model
        response = httpx.get(f"{api_url}/files/unindexed", params=params, timeout=60)
        if response.status_code == 200:
            data = response.json()
            count = data.get("count", 0)
            if count > 0:
                print(f"Found {count} unindexed files for model {data.get('model')}")
            return data.get("files", [])
    except Exception as e:
        print(f"Warning: Could not get unindexed files from API: {e}")
    return []


import gzip
import io

POLL_INTERVAL = 2  # Seconds between status checks


def wait_for_indexing_complete(api_url: str, timeout: int = 3600) -> bool:
    """Wait for current indexing to complete."""
    start_time = time.time()
    last_processed = -1
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{api_url}/reindex/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                if not status.get("running", False):
                    return True
                # Show progress
                processed = status.get("processed_files", 0)
                total = status.get("total_files", 0)
                if total > 0 and processed != last_processed:
                    pct = processed * 100 // total
                    print(f"\r  Progress: {processed}/{total} ({pct}%)   ", end="", flush=True)
                    last_processed = processed
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)
    return False


def cleanup_orphaned_via_api(api_url: str) -> bool:
    """Cleanup orphaned records (files that no longer exist on disk).
    
    Fast mode: gets all file paths from API, checks locally on host,
    sends list of missing files to API for deletion.
    
    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        print("Getting file list from database...")
        
        # Получить все пути из БД
        response = httpx.get(f"{api_url}/files/index", timeout=120)
        if response.status_code != 200:
            print(f"Warning: Could not get file list: HTTP {response.status_code}")
            return False
        
        data = response.json()
        all_files = data.get("files", [])
        total = len(all_files)
        
        if total == 0:
            print("No files in database")
            return True
        
        print(f"Checking {total} files on local filesystem...")
        
        # Проверить существование файлов на хосте
        missing_docker_paths = []
        checked = 0
        
        for file_info in all_files:
            docker_path = file_info["file_path"]
            
            try:
                # Конвертировать Docker path в host path
                host_path = docker_to_host_path(docker_path)
                
                # Проверить существование на хосте (быстро)
                if not Path(host_path).exists():
                    missing_docker_paths.append(docker_path)
                
                checked += 1
                
                # Прогресс каждые 5000 файлов
                if checked % 5000 == 0:
                    print(f"  Checked {checked}/{total} ({len(missing_docker_paths)} missing)...")
                    
            except ValueError:
                # Путь не под PHOTOS_HOST_PATH, пропускаем
                pass
        
        print(f"✓ Checked {checked} files, found {len(missing_docker_paths)} missing")
        
        if len(missing_docker_paths) == 0:
            print("✓ No orphaned records found")
            return True
        
        # Отправить список missing файлов в API для удаления (gzip compressed)
        print(f"Deleting {len(missing_docker_paths)} orphaned records from database...")
        
        # Create JSON and compress with gzip (same as trigger_reindex)
        json_data = json.dumps(missing_docker_paths).encode("utf-8")
        json_size = len(json_data)
        compressed = gzip.compress(json_data)
        compressed_size = len(compressed)
        ratio = compressed_size * 100 // json_size if json_size > 0 else 0
        
        print(f"  JSON size: {json_size / 1024:.1f} KB -> gzip: {compressed_size / 1024:.1f} KB ({ratio}%)")
        
        # Send as multipart POST
        files = {
            "file_list": ("orphaned.json.gz", io.BytesIO(compressed), "application/gzip")
        }
        
        response = httpx.post(
            f"{api_url}/cleanup/orphaned",
            files=files,
            timeout=300
        )
        
        if response.status_code == 200:
            data = response.json()
            deleted = data.get("deleted", 0)
            print(f"✓ Cleanup completed: deleted {deleted} orphaned records")
            return True
        else:
            print(f"Warning: Cleanup failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Warning: Could not cleanup orphaned records: {e}")
        import traceback
        traceback.print_exc()
        return False


def trigger_reindex(api_url: str, file_paths: list, model: str = None):
    """Trigger reindex for specific files via API.

    Sends gzipped JSON file list as multipart POST to handle large file counts.
    """
    if not file_paths:
        print("No files to index.")
        return

    # Convert host paths to Docker container paths
    docker_paths = []
    for host_path in file_paths:
        try:
            docker_path = host_to_docker_path(host_path)
            docker_paths.append(docker_path)
        except ValueError as e:
            print(f"Warning: Skipping {host_path}: {e}")

    if not docker_paths:
        print("No valid files to index after path conversion.")
        return

    total_files = len(docker_paths)
    print(f"Preparing {total_files} files for indexing...")
    print(f"  Host path prefix: {PHOTOS_HOST_PATH}")
    print(f"  Docker path prefix: {DOCKER_PHOTOS_PATH}")
    if docker_paths:
        print(f"  Example: {file_paths[0]} -> {docker_paths[0]}")

    # Create JSON and compress with gzip
    json_data = json.dumps(docker_paths).encode("utf-8")
    json_size = len(json_data)

    compressed = gzip.compress(json_data)
    compressed_size = len(compressed)
    ratio = compressed_size * 100 // json_size if json_size > 0 else 0

    print(f"  JSON size: {json_size / 1024 / 1024:.1f} MB -> gzip: {compressed_size / 1024 / 1024:.1f} MB ({ratio}%)")

    # Send as multipart POST
    try:
        params = {}
        if model:
            params["model"] = model

        files = {
            "file_list": ("files.json.gz", io.BytesIO(compressed), "application/gzip")
        }

        response = httpx.post(
            f"{api_url}/reindex/files",
            params=params,
            files=files,
            timeout=60  # Longer timeout for large uploads
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Indexing started: {result}")
            print(f"\nMonitoring progress (Ctrl+C to stop watching)...")

            # Monitor progress
            try:
                while True:
                    if wait_for_indexing_complete(api_url, timeout=10):
                        print("\nIndexing complete!")
                        # Get final stats
                        status_resp = httpx.get(f"{api_url}/reindex/status", timeout=10)
                        if status_resp.status_code == 200:
                            final = status_resp.json()
                            print(f"  Processed: {final.get('processed_files', 0)}")
                            print(f"  Successful: {final.get('successful', 0)}")
                            print(f"  Failed: {final.get('failed', 0)}")
                            print(f"  Skipped: {final.get('skipped', 0)}")
                        break
            except KeyboardInterrupt:
                print("\n\nStopped watching. Indexing continues in background.")
                print(f"Check status: curl {api_url}/reindex/status")

        elif response.status_code == 409:
            print("Indexing already in progress. Wait for it to complete.")
            print(f"Check status: curl {api_url}/reindex/status")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error triggering reindex: {e}")


def everything_scan(storage_path: str, supported_formats: set) -> list:
    """Fast file listing using Everything SDK (requires Everything to be running)."""
    if not HAS_EVERYTHING:
        return None

    print(f"Using Everything for fast file listing...")
    start_time = time.time()

    # Build search query: path filter + extension filter
    # Example: "H:\PHOTO\" ext:jpg|heic|png|nef
    storage_normalized = storage_path.replace("/", "\\")
    if not storage_normalized.endswith("\\"):
        storage_normalized += "\\"

    # Build extension filter
    extensions = "|".join(ext.lstrip(".") for ext in supported_formats)
    query = f'"{storage_normalized}" ext:{extensions}'

    print(f"  Query: {query}")

    try:
        EVERYTHING_DLL.Everything_SetSearchW(query)
        if not EVERYTHING_DLL.Everything_QueryW(1):  # 1 = wait for results
            error = EVERYTHING_DLL.Everything_GetLastError()
            if error == 2:  # EVERYTHING_ERROR_IPC
                print("  Error: Everything is not running. Start Everything first.")
                return None
            print(f"  Error: Everything query failed (error {error})")
            return None

        num_results = EVERYTHING_DLL.Everything_GetNumResults()
        print(f"  Found {num_results} files")

        files = []
        buffer = ctypes.create_unicode_buffer(1024)

        for i in range(num_results):
            EVERYTHING_DLL.Everything_GetResultFullPathNameW(i, buffer, 1024)
            files.append(buffer.value)

        elapsed = time.time() - start_time
        print(f"Everything scan completed: {len(files)} files in {elapsed:.2f}s")

        return files

    except Exception as e:
        print(f"  Everything error: {e}")
        return None


def full_scan(storage_path: str, supported_formats: set) -> list:
    """Full directory scan. Uses Everything if available, falls back to os.walk."""

    # Try Everything first (much faster)
    if HAS_EVERYTHING:
        result = everything_scan(storage_path, supported_formats)
        if result is not None:
            return result

    print(f"Starting directory scan of {storage_path}...")
    start_time = time.time()

    files = []
    file_count = 0

    # Use os.scandir for better performance than rglob
    def scan_dir(path):
        nonlocal file_count
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            if Path(entry.name).suffix.lower() in supported_formats:
                                files.append(entry.path)
                                file_count += 1
                                if file_count % 10000 == 0:
                                    print(f"  Scanning: {file_count} files...")
                        elif entry.is_dir(follow_symlinks=False):
                            scan_dir(entry.path)
                    except (PermissionError, OSError):
                        continue
        except (PermissionError, OSError):
            pass

    scan_dir(storage_path)

    elapsed = time.time() - start_time
    print(f"Scan completed: {len(files)} files in {elapsed:.1f}s")

    return files


def main():
    parser = argparse.ArgumentParser(description="Fast reindex using NTFS USN Journal")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--model", default=None, help="CLIP model (ViT-B/32, ViT-B/16, ViT-L/14, SigLIP)")
    parser.add_argument("--storage-path", default=None, help="Photo storage path (default: PHOTOS_HOST_PATH from .env)")
    parser.add_argument("--full-scan", action="store_true", help="Force full scan (ignore USN Journal)")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup orphaned records before indexing (slow, checks all files)")
    args = parser.parse_args()

    # Use PHOTOS_HOST_PATH from .env (the mapped directory), not settings.PHOTO_STORAGE_PATH
    storage_path = args.storage_path or PHOTOS_HOST_PATH
    if not storage_path:
        print("Error: PHOTOS_HOST_PATH not set in .env and --storage-path not provided")
        sys.exit(1)

    supported_formats = set(settings.SUPPORTED_FORMATS)

    print(f"Storage path (host): {storage_path}")
    print(f"Docker mapping: {storage_path} -> {DOCKER_PHOTOS_PATH}")
    print(f"API URL: {args.api_url}")
    print(f"Supported formats: {supported_formats}")

    if not Path(storage_path).exists():
        print(f"Error: Storage path does not exist: {storage_path}")
        sys.exit(1)

    # Optional cleanup orphaned records (slow - checks all files in DB)
    if args.cleanup:
        print("=" * 60)
        print("Running cleanup (this may take several minutes for large databases)...")
        cleanup_orphaned_via_api(args.api_url)
        print("=" * 60)
        print()

    # Check if NTFS USN Journal is available
    if not IS_WINDOWS:
        print("Warning: Not running on Windows. Using full scan.")
        files = full_scan(storage_path, supported_formats)
        trigger_reindex(args.api_url, files, args.model)
        return

    if not HAS_WIN32:
        print("Warning: pywin32 not installed. Using full scan.")
        print("Install with: pip install pywin32")
        files = full_scan(storage_path, supported_formats)
        trigger_reindex(args.api_url, files, args.model)
        return

    if args.full_scan:
        print("Forcing full scan (--full-scan flag)")
        files = full_scan(storage_path, supported_formats)
        trigger_reindex(args.api_url, files, args.model)
        return

    # Use NTFS USN Journal
    try:
        storage_path_obj = Path(storage_path).resolve()
        drive_letter = str(storage_path_obj.drive)

        # Get checkpoint
        last_usn = get_checkpoint_from_api(args.api_url, drive_letter)
        print(f"Last USN checkpoint: {last_usn}")

        tracker = NTFSChangeTracker(storage_path, supported_formats)

        if last_usn == 0:
            # First run - get current USN and do full scan
            current_usn = tracker.get_current_usn()
            print(f"First run - saving USN checkpoint: {current_usn}")

            files = full_scan(storage_path, supported_formats)
            save_checkpoint_to_api(args.api_url, drive_letter, current_usn, len(files))
            trigger_reindex(args.api_url, files, args.model)
            return

        # Get changes from USN Journal
        print("Reading NTFS USN Journal...")
        start_time = time.time()
        changes = tracker.get_changes_since(last_usn)
        elapsed = time.time() - start_time
        print(f"USN Journal read in {elapsed:.2f}s")

        if changes.get('full_scan_required'):
            print("USN Journal overflow - falling back to full scan")
            files = full_scan(storage_path, supported_formats)
            new_usn = changes.get('next_usn', 0)
            if new_usn:
                save_checkpoint_to_api(args.api_url, drive_letter, new_usn, len(files))
            trigger_reindex(args.api_url, files, args.model)
            return

        # Match filenames to full paths
        print(f"Changes detected: {len(changes['added'])} added, {len(changes['modified'])} modified, {len(changes['deleted'])} deleted")

        # For added files, search on disk
        added_paths = []
        for filename in changes['added']:
            for found_path in storage_path_obj.rglob(filename):
                if found_path.is_file() and found_path.suffix.lower() in supported_formats:
                    added_paths.append(str(found_path))
                    break

        # Get known files from API for matching modified/deleted
        # Note: API returns docker paths (/photos/...), we need to convert to host paths for exists() check
        known_files = get_known_files_from_api(args.api_url)

        # Match modified files (check host path exists, but keep host path for trigger_reindex)
        modified_paths = []
        for filename in changes['modified']:
            if filename in known_files:
                docker_path = known_files[filename]
                try:
                    host_path = docker_to_host_path(docker_path)
                    if Path(host_path).exists():
                        modified_paths.append(host_path)  # Use host path, will be converted in trigger_reindex
                except ValueError:
                    pass

        # Report deleted files (check host path doesn't exist)
        deleted_paths = []
        for filename in changes['deleted']:
            if filename in known_files:
                docker_path = known_files[filename]
                try:
                    host_path = docker_to_host_path(docker_path)
                    if not Path(host_path).exists():
                        deleted_paths.append(docker_path)
                        print(f"Deleted: {docker_path}")
                except ValueError:
                    pass

        # Save new checkpoint
        new_usn = changes.get('next_usn', 0)
        if new_usn:
            save_checkpoint_to_api(args.api_url, drive_letter, new_usn)

        # Cleanup deleted files from database
        if deleted_paths:
            print(f"\nRemoving {len(deleted_paths)} deleted file(s) from database...")
            
            # Send deleted paths to cleanup API (gzip compressed)
            json_data = json.dumps(deleted_paths).encode("utf-8")
            compressed = gzip.compress(json_data)
            
            files_param = {
                "file_list": ("deleted.json.gz", io.BytesIO(compressed), "application/gzip")
            }
            
            try:
                response = httpx.post(
                    f"{args.api_url}/cleanup/orphaned",
                    files=files_param,
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    deleted_count = data.get("deleted", 0)
                    print(f"✓ Removed {deleted_count} record(s) from database")
                else:
                    print(f"Warning: Cleanup failed with status {response.status_code}")
            except Exception as e:
                print(f"Warning: Could not cleanup deleted files: {e}")
            print()

        # Also check for unindexed files in DB (e.g., if previous indexing was interrupted)
        unindexed_docker_paths = get_unindexed_files_from_api(args.api_url, args.model)
        unindexed_host_paths = []
        for docker_path in unindexed_docker_paths:
            try:
                host_path = docker_to_host_path(docker_path)
                if Path(host_path).exists():
                    unindexed_host_paths.append(host_path)
            except ValueError:
                pass

        # Trigger reindex for new/modified/unindexed files
        files_to_index = list(set(added_paths + modified_paths + unindexed_host_paths))
        print(f"Files to index: {len(files_to_index)} ({len(added_paths)} new from USN, {len(modified_paths)} modified, {len(unindexed_host_paths)} unindexed in DB)")

        if files_to_index:
            trigger_reindex(args.api_url, files_to_index, args.model)
        else:
            print("No new files to index.")

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to full scan...")
        files = full_scan(storage_path, supported_formats)
        trigger_reindex(args.api_url, files, args.model)


if __name__ == "__main__":
    main()
