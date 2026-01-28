"""Мониторинг файловой системы и обнаружение изменений"""

import logging
import os
import time
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class FileMonitor:
    """Монитор файловой системы для отслеживания изменений"""

    def __init__(self, storage_path: str, supported_formats: List[str] = None):
        self.storage_path = Path(storage_path)
        self.supported_formats = set(f.lower() for f in (supported_formats or ['.heic', '.heif', '.jpg', '.jpeg', '.png']))
        self.file_index: Dict[str, Dict] = {}
        self.last_scan_time = 0

        logger.info(f"Монитор инициализирован для {self.storage_path}")

    def is_supported_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_formats

    def scan_directory(self) -> Dict[str, Dict]:
        """Быстрое сканирование директории (без хэширования)"""
        file_index = {}

        if not self.storage_path.exists():
            logger.error(f"Путь не существует: {self.storage_path}")
            return file_index

        scan_start = time.time()
        file_count = 0

        try:
            for file_path in self.storage_path.rglob("*"):
                if not file_path.is_file():
                    continue

                if not self.is_supported_file(file_path):
                    continue

                try:
                    stat = file_path.stat()
                    file_index[str(file_path)] = {
                        'path': str(file_path),
                        'size': stat.st_size,
                        'modified_time': stat.st_mtime,
                    }
                    file_count += 1

                    if file_count % 1000 == 0:
                        logger.info(f"Сканирование: {file_count} файлов...")

                except Exception as e:
                    logger.warning(f"Ошибка сканирования {file_path}: {e}")

        except Exception as e:
            logger.error(f"Ошибка сканирования {self.storage_path}: {e}")

        scan_time = time.time() - scan_start
        logger.info(f"Сканирование завершено: {len(file_index)} файлов за {scan_time:.1f}s")

        self.last_scan_time = time.time()
        self.file_index = file_index
        return file_index

    def get_changes(self) -> Dict[str, List[str]]:
        """Получить изменения (новые, измененные, удаленные)"""
        current_index = self.scan_directory()

        added = []
        modified = []
        deleted = []

        for file_path in current_index:
            if file_path not in self.file_index:
                added.append(file_path)

        for file_path, info in current_index.items():
            if file_path in self.file_index:
                old_info = self.file_index[file_path]
                if (old_info['size'] != info['size'] or
                    old_info['modified_time'] != info['modified_time']):
                    modified.append(file_path)

        for file_path in self.file_index:
            if file_path not in current_index:
                deleted.append(file_path)

        self.file_index = current_index

        return {'added': added, 'modified': modified, 'deleted': deleted}

    def print_stats(self):
        """Статистика сканирования"""
        total_size = sum(info['size'] for info in self.file_index.values())
        total_files = len(self.file_index)
        logger.info(f"Статистика: {total_files} файлов, {total_size / (1024**3):.2f} GB")
