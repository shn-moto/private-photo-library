"""NTFS USN Journal change tracker for fast file change detection on Windows"""

import logging
import os
import struct
import time
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from enum import IntFlag

logger = logging.getLogger(__name__)

# Check if we're on Windows
IS_WINDOWS = os.name == 'nt'

if IS_WINDOWS:
    try:
        import win32file
        import win32con
        import winioctlcon
        import pywintypes
        HAS_WIN32 = True
    except ImportError:
        HAS_WIN32 = False
        logger.warning("pywin32 not installed. NTFS USN Journal tracking disabled.")
else:
    HAS_WIN32 = False


class UsnReason(IntFlag):
    """USN Journal change reasons"""
    DATA_OVERWRITE = 0x00000001
    DATA_EXTEND = 0x00000002
    DATA_TRUNCATION = 0x00000004
    NAMED_DATA_OVERWRITE = 0x00000010
    NAMED_DATA_EXTEND = 0x00000020
    NAMED_DATA_TRUNCATION = 0x00000040
    FILE_CREATE = 0x00000100
    FILE_DELETE = 0x00000200
    EA_CHANGE = 0x00000400
    SECURITY_CHANGE = 0x00000800
    RENAME_OLD_NAME = 0x00001000
    RENAME_NEW_NAME = 0x00002000
    INDEXABLE_CHANGE = 0x00004000
    BASIC_INFO_CHANGE = 0x00008000
    HARD_LINK_CHANGE = 0x00010000
    COMPRESSION_CHANGE = 0x00020000
    ENCRYPTION_CHANGE = 0x00040000
    OBJECT_ID_CHANGE = 0x00080000
    REPARSE_POINT_CHANGE = 0x00100000
    STREAM_CHANGE = 0x00200000
    CLOSE = 0x80000000


@dataclass
class FileChange:
    """Represents a file change from USN Journal"""
    path: str
    reason: int
    is_directory: bool
    usn: int
    timestamp: float


class NTFSChangeTracker:
    """
    Fast file change detection using NTFS USN Journal.

    The USN (Update Sequence Number) Journal is a feature of NTFS that
    logs all changes to files and directories. Reading the journal is
    much faster than scanning the entire file system.
    """

    # IOCTL codes
    FSCTL_QUERY_USN_JOURNAL = 0x000900f4
    FSCTL_READ_USN_JOURNAL = 0x000900bb
    FSCTL_ENUM_USN_DATA = 0x000900b3

    def __init__(self, storage_path: str, supported_formats: Set[str] = None):
        self.storage_path = Path(storage_path).resolve()
        self.supported_formats = supported_formats or {'.heic', '.heif', '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.nef', '.cr2', '.arw', '.dng', '.raf', '.orf', '.rw2'}
        self.drive_letter = self.storage_path.drive
        self.volume_handle = None
        self.journal_id = None
        self.last_usn = 0

        if not IS_WINDOWS:
            raise RuntimeError("NTFSChangeTracker only works on Windows")
        if not HAS_WIN32:
            raise RuntimeError("pywin32 is required for NTFS USN Journal access. Install with: pip install pywin32")

    def _open_volume(self):
        """Open handle to the volume"""
        if self.volume_handle:
            return

        volume_path = f"\\\\.\\{self.drive_letter}"
        try:
            self.volume_handle = win32file.CreateFile(
                volume_path,
                win32con.GENERIC_READ,
                win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
                None,
                win32con.OPEN_EXISTING,
                0,
                None
            )
            logger.info(f"Opened volume {volume_path}")
        except pywintypes.error as e:
            raise RuntimeError(f"Failed to open volume {volume_path}: {e}. Run as Administrator.")

    def _close_volume(self):
        """Close volume handle"""
        if self.volume_handle:
            win32file.CloseHandle(self.volume_handle)
            self.volume_handle = None

    def _query_journal(self) -> Tuple[int, int, int]:
        """Query USN Journal info. Returns (journal_id, first_usn, next_usn)"""
        self._open_volume()

        # USN_JOURNAL_DATA structure: UsnJournalID (8), FirstUsn (8), NextUsn (8), ...
        output_buffer = win32file.DeviceIoControl(
            self.volume_handle,
            self.FSCTL_QUERY_USN_JOURNAL,
            None,
            64  # Output buffer size
        )

        journal_id, first_usn, next_usn = struct.unpack('<QQQ', output_buffer[:24])
        self.journal_id = journal_id

        logger.debug(f"Journal ID: {journal_id}, First USN: {first_usn}, Next USN: {next_usn}")
        return journal_id, first_usn, next_usn

    def _parse_usn_record_v2(self, data: bytes, offset: int) -> Tuple[Optional[FileChange], int]:
        """Parse a USN_RECORD_V2 structure"""
        if offset + 8 > len(data):
            return None, 0

        # USN_RECORD_V2 structure
        record_length = struct.unpack_from('<I', data, offset)[0]
        if record_length == 0 or offset + record_length > len(data):
            return None, 0

        # Parse fixed fields
        # Offset 0: RecordLength (4)
        # Offset 4: MajorVersion (2), MinorVersion (2)
        # Offset 8: FileReferenceNumber (8)
        # Offset 16: ParentFileReferenceNumber (8)
        # Offset 24: Usn (8)
        # Offset 32: TimeStamp (8) - FILETIME
        # Offset 40: Reason (4)
        # Offset 44: SourceInfo (4)
        # Offset 48: SecurityId (4)
        # Offset 52: FileAttributes (4)
        # Offset 56: FileNameLength (2)
        # Offset 58: FileNameOffset (2)
        # Offset 60+: FileName (Unicode)

        try:
            usn = struct.unpack_from('<Q', data, offset + 24)[0]
            timestamp_raw = struct.unpack_from('<Q', data, offset + 32)[0]
            reason = struct.unpack_from('<I', data, offset + 40)[0]
            file_attributes = struct.unpack_from('<I', data, offset + 52)[0]
            filename_length = struct.unpack_from('<H', data, offset + 56)[0]
            filename_offset = struct.unpack_from('<H', data, offset + 58)[0]

            # Extract filename
            filename_start = offset + filename_offset
            filename = data[filename_start:filename_start + filename_length].decode('utf-16-le')

            # Convert FILETIME to Unix timestamp
            # FILETIME is 100-nanosecond intervals since January 1, 1601
            timestamp = (timestamp_raw - 116444736000000000) / 10000000 if timestamp_raw > 0 else time.time()

            is_directory = bool(file_attributes & win32con.FILE_ATTRIBUTE_DIRECTORY)

            change = FileChange(
                path=filename,  # Just filename, not full path
                reason=reason,
                is_directory=is_directory,
                usn=usn,
                timestamp=timestamp
            )

            return change, record_length

        except Exception as e:
            logger.warning(f"Failed to parse USN record at offset {offset}: {e}")
            return None, record_length

    def _get_file_path_from_ref(self, file_ref: int) -> Optional[str]:
        """Get full file path from file reference number (slow, use sparingly)"""
        # This requires opening the file by ID which is complex
        # For now we'll use a simpler approach - scan for the filename
        return None

    def is_supported_file(self, filename: str) -> bool:
        """Check if file has supported format"""
        return Path(filename).suffix.lower() in self.supported_formats

    def get_changes_since(self, last_usn: int = 0) -> Dict[str, List[str]]:
        """
        Get file changes since the specified USN.

        Returns dict with keys: 'added', 'modified', 'deleted'
        Note: Due to USN Journal limitations, we can only get filenames,
        not full paths. This method returns filenames that need to be
        matched against a full scan or database.
        """
        start_time = time.time()

        journal_id, first_usn, next_usn = self._query_journal()

        if last_usn < first_usn:
            logger.warning(f"Last USN {last_usn} is before journal start {first_usn}. Full scan required.")
            return {'added': [], 'modified': [], 'deleted': [], 'full_scan_required': True}

        added_files = set()
        modified_files = set()
        deleted_files = set()

        # Read USN Journal entries
        # READ_USN_JOURNAL_DATA_V0 structure
        start_usn = last_usn
        reason_mask = (
            UsnReason.FILE_CREATE |
            UsnReason.FILE_DELETE |
            UsnReason.DATA_OVERWRITE |
            UsnReason.DATA_EXTEND |
            UsnReason.DATA_TRUNCATION |
            UsnReason.RENAME_NEW_NAME |
            UsnReason.CLOSE
        )

        read_data = struct.pack('<QIIQQQ',
            start_usn,          # StartUsn (8 bytes)
            reason_mask,        # ReasonMask (4 bytes)
            0,                  # ReturnOnlyOnClose (4 bytes)
            0,                  # Timeout (8 bytes)
            0,                  # BytesToWaitFor (8 bytes)
            journal_id          # UsnJournalID (8 bytes)
        )

        buffer_size = 65536  # 64KB buffer
        records_processed = 0

        try:
            while True:
                try:
                    output = win32file.DeviceIoControl(
                        self.volume_handle,
                        self.FSCTL_READ_USN_JOURNAL,
                        read_data,
                        buffer_size
                    )
                except pywintypes.error as e:
                    if e.winerror == 1181:  # ERROR_JOURNAL_ENTRY_DELETED
                        logger.warning("Some journal entries were deleted. Full scan may be required.")
                        break
                    raise

                if len(output) <= 8:
                    break

                # First 8 bytes is next USN
                next_start_usn = struct.unpack('<Q', output[:8])[0]

                # Parse records
                offset = 8
                while offset < len(output):
                    change, record_len = self._parse_usn_record_v2(output, offset)
                    if not change or record_len == 0:
                        break

                    offset += record_len
                    records_processed += 1

                    # Skip directories
                    if change.is_directory:
                        continue

                    # Filter by supported formats
                    if not self.is_supported_file(change.path):
                        continue

                    # Categorize by reason
                    if change.reason & UsnReason.FILE_DELETE:
                        deleted_files.add(change.path)
                        added_files.discard(change.path)
                        modified_files.discard(change.path)
                    elif change.reason & UsnReason.FILE_CREATE:
                        added_files.add(change.path)
                        deleted_files.discard(change.path)
                    elif change.reason & (UsnReason.DATA_OVERWRITE | UsnReason.DATA_EXTEND | UsnReason.DATA_TRUNCATION):
                        if change.path not in added_files:
                            modified_files.add(change.path)

                # Update for next iteration
                if next_start_usn == start_usn or next_start_usn >= next_usn:
                    break

                start_usn = next_start_usn
                read_data = struct.pack('<QIIQQQ',
                    start_usn, reason_mask, 0, 0, 0, journal_id
                )

        finally:
            self._close_volume()

        elapsed = time.time() - start_time
        logger.info(f"USN Journal scan: {records_processed} records in {elapsed:.2f}s")
        logger.info(f"Changes: {len(added_files)} added, {len(modified_files)} modified, {len(deleted_files)} deleted")

        self.last_usn = next_usn

        return {
            'added': list(added_files),
            'modified': list(modified_files),
            'deleted': list(deleted_files),
            'next_usn': next_usn,
            'full_scan_required': False
        }

    def get_current_usn(self) -> int:
        """Get current USN position (to save as checkpoint)"""
        _, _, next_usn = self._query_journal()
        self._close_volume()
        return next_usn

    def match_filenames_to_paths(self, filenames: Set[str], known_paths: Dict[str, str]) -> List[str]:
        """
        Match USN filenames to full paths from known paths index.

        Args:
            filenames: Set of filenames from USN Journal
            known_paths: Dict mapping filename -> full path (from DB or previous scan)

        Returns:
            List of matched full paths
        """
        matched = []
        for filename in filenames:
            if filename in known_paths:
                matched.append(known_paths[filename])
        return matched


def scan_with_usn_acceleration(
    storage_path: str,
    supported_formats: Set[str],
    last_usn: int,
    known_files: Dict[str, Dict]
) -> Tuple[Dict[str, List[str]], int]:
    """
    Hybrid scan: use USN Journal for incremental updates, fall back to full scan if needed.

    Args:
        storage_path: Root path to scan
        supported_formats: Set of supported file extensions
        last_usn: Last checkpoint USN (0 for full scan)
        known_files: Dict of known files from previous scan/DB {path: {size, mtime}}

    Returns:
        Tuple of (changes dict, new_usn)
    """
    if not IS_WINDOWS or not HAS_WIN32:
        # Fall back to full scan on non-Windows
        return _fallback_full_scan(storage_path, supported_formats, known_files), 0

    try:
        tracker = NTFSChangeTracker(storage_path, supported_formats)

        if last_usn == 0:
            # First run - just get current USN for next time
            current_usn = tracker.get_current_usn()
            logger.info(f"First run - saving USN checkpoint: {current_usn}")
            return _fallback_full_scan(storage_path, supported_formats, known_files), current_usn

        changes = tracker.get_changes_since(last_usn)

        if changes.get('full_scan_required'):
            logger.warning("Full scan required due to journal overflow")
            return _fallback_full_scan(storage_path, supported_formats, known_files), changes.get('next_usn', 0)

        # Build filename -> path index from known files
        filename_index = {}
        for path in known_files:
            filename = Path(path).name
            filename_index[filename] = path

        # Match filenames to full paths
        storage_path_obj = Path(storage_path).resolve()

        # For added files, we need to find them on disk
        added_paths = []
        for filename in changes['added']:
            # Quick search in storage path
            for found_path in storage_path_obj.rglob(filename):
                if found_path.is_file():
                    added_paths.append(str(found_path))
                    break

        # For modified/deleted, match against known files
        modified_paths = []
        for filename in changes['modified']:
            if filename in filename_index:
                path = filename_index[filename]
                if Path(path).exists():
                    modified_paths.append(path)

        deleted_paths = []
        for filename in changes['deleted']:
            if filename in filename_index:
                path = filename_index[filename]
                if not Path(path).exists():
                    deleted_paths.append(path)

        return {
            'added': added_paths,
            'modified': modified_paths,
            'deleted': deleted_paths
        }, changes.get('next_usn', 0)

    except Exception as e:
        logger.error(f"USN Journal access failed: {e}. Falling back to full scan.")
        return _fallback_full_scan(storage_path, supported_formats, known_files), 0


def _fallback_full_scan(
    storage_path: str,
    supported_formats: Set[str],
    known_files: Dict[str, Dict]
) -> Dict[str, List[str]]:
    """Full directory scan fallback"""
    logger.info("Performing full directory scan...")
    start_time = time.time()

    storage_path_obj = Path(storage_path)
    current_files = {}

    for file_path in storage_path_obj.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in supported_formats:
            continue

        try:
            stat = file_path.stat()
            current_files[str(file_path)] = {
                'size': stat.st_size,
                'mtime': stat.st_mtime
            }
        except OSError:
            continue

    # Calculate changes
    added = [p for p in current_files if p not in known_files]
    deleted = [p for p in known_files if p not in current_files]
    modified = []
    for p in current_files:
        if p in known_files:
            if (current_files[p]['size'] != known_files[p].get('size') or
                current_files[p]['mtime'] != known_files[p].get('mtime')):
                modified.append(p)

    elapsed = time.time() - start_time
    logger.info(f"Full scan completed: {len(current_files)} files in {elapsed:.1f}s")

    return {
        'added': added,
        'modified': modified,
        'deleted': deleted
    }
