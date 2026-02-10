"""Сервис индексации изображений (батчевая обработка на GPU)"""

import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Set
from pathlib import Path
import time

from config.settings import settings
from services.image_processor import ImageProcessor

try:
    from services.clip_embedder import CLIPEmbedder
except ImportError:
    CLIPEmbedder = None
    logging.warning("CLIP embedder not available - transformers not installed")

# NTFS USN Journal support (Windows only)
try:
    from services.ntfs_change_tracker import NTFSChangeTracker, IS_WINDOWS, HAS_WIN32
except ImportError:
    NTFSChangeTracker = None
    IS_WINDOWS = os.name == 'nt'
    HAS_WIN32 = False

from db.database import DatabaseManager, PhotoIndexRepository

logger = logging.getLogger(__name__)


class IndexingService:
    """Сервис индексации фотографий (батчевая обработка GPU)"""

    def __init__(self, model_name: Optional[str] = None, clip_embedder: Optional['CLIPEmbedder'] = None):
        """Инициализация сервиса индексации
        
        Args:
            model_name: Имя CLIP модели (ViT-B/32, ViT-B/16, ViT-L/14, SigLIP).
                       Если None - используется модель из settings.CLIP_MODEL
            clip_embedder: Уже инициализированный CLIPEmbedder (для повторного использования).
                          Если None - создается новый экземпляр.
        """
        self.db_manager = DatabaseManager(settings.DATABASE_URL)
        self.photo_repo = PhotoIndexRepository(self.db_manager)
        self.image_processor = ImageProcessor(settings.IMAGE_MAX_SIZE)
        self.batch_size = settings.BATCH_SIZE_CLIP

        # CLIP embedder - используем переданный или создаем новый
        if clip_embedder:
            self.clip_embedder = clip_embedder
            logger.info(f"Использую существующий CLIP embedder: {clip_embedder.model_name}")
        elif CLIPEmbedder:
            try:
                model = model_name or settings.CLIP_MODEL
                self.clip_embedder = CLIPEmbedder(model, settings.CLIP_DEVICE)
            except Exception as e:
                logger.error(f"Failed to initialize CLIP embedder: {e}")
                self.clip_embedder = None
        else:
            self.clip_embedder = None

        # Внутреннее состояние для отслеживания прогресса
        self._progress = {
            "total_files": 0,
            "processed_files": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "current_batch": 0,
            "total_batches": 0,
            "speed_imgs_per_sec": 0.0,
            "eta_seconds": 0,
        }
        self._start_time = 0  # Время начала индексации для вычисления скорости
        self._stop_requested = False  # Флаг остановки индексации

        logger.info("=" * 50)
        logger.info("Сервис индексации инициализирован")
        logger.info(f"  CLIP: {self.clip_embedder.model_name if self.clip_embedder else 'DISABLED'}")
        logger.info(f"  Device: {self.clip_embedder.device if self.clip_embedder else 'N/A'}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info("=" * 50)

    def get_indexed_paths(self) -> Set[str]:
        """Получить множество уже проиндексированных путей для текущей модели"""
        session = self.db_manager.get_session()
        try:
            from models.data_models import PhotoIndex

            # Получаем имя колонки для текущей модели
            if not self.clip_embedder:
                return set()
            embedding_column_name = self.clip_embedder.get_embedding_column()
            embedding_column = getattr(PhotoIndex, embedding_column_name)

            # Ищем пути, где есть эмбеддинг для этой модели
            paths = session.query(PhotoIndex.file_path).filter(embedding_column != None).all()
            return {p.file_path for p in paths}
        finally:
            session.close()

    def get_all_known_files(self) -> Dict[str, Dict]:
        """Получить все известные файлы из БД (для NTFS tracker fallback)"""
        session = self.db_manager.get_session()
        try:
            from models.data_models import PhotoIndex

            photos = session.query(PhotoIndex.file_path, PhotoIndex.file_size).all()
            return {
                p.file_path: {'size': p.file_size, 'mtime': 0}
                for p in photos
            }
        finally:
            session.close()

    def get_usn_checkpoint(self, drive_letter: str) -> int:
        """Получить USN checkpoint для диска"""
        session = self.db_manager.get_session()
        try:
            from models.data_models import ScanCheckpoint

            checkpoint = session.query(ScanCheckpoint).filter_by(drive_letter=drive_letter).first()
            return checkpoint.last_usn if checkpoint else 0
        except Exception as e:
            logger.debug(f"No checkpoint found for {drive_letter}: {e}")
            return 0
        finally:
            session.close()

    def save_usn_checkpoint(self, drive_letter: str, usn: int, files_count: int = 0):
        """Сохранить USN checkpoint для диска"""
        session = self.db_manager.get_session()
        try:
            from models.data_models import ScanCheckpoint

            checkpoint = session.query(ScanCheckpoint).filter_by(drive_letter=drive_letter).first()
            if checkpoint:
                checkpoint.last_usn = usn
                checkpoint.last_scan_time = datetime.now()
                checkpoint.files_count = files_count
            else:
                checkpoint = ScanCheckpoint(
                    drive_letter=drive_letter,
                    last_usn=usn,
                    files_count=files_count
                )
                session.add(checkpoint)

            session.commit()
            logger.info(f"Saved USN checkpoint for {drive_letter}: USN={usn}")
        except Exception as e:
            logger.error(f"Failed to save USN checkpoint: {e}")
            session.rollback()
        finally:
            session.close()

    def fast_scan_files(self, storage_path: str) -> List[str]:
        """
        Быстрое сканирование файлов с использованием NTFS USN Journal (Windows)
        или fallback на полное сканирование.

        Returns:
            Список путей к новым/измененным файлам для индексации
        """
        storage_path_obj = Path(storage_path).resolve()
        supported_formats = self.image_processor.SUPPORTED_FORMATS

        # Try NTFS USN Journal on Windows
        if IS_WINDOWS and HAS_WIN32 and NTFSChangeTracker:
            try:
                drive_letter = str(storage_path_obj.drive)
                last_usn = self.get_usn_checkpoint(drive_letter)

                logger.info(f"Using NTFS USN Journal for fast scanning (last USN: {last_usn})")

                tracker = NTFSChangeTracker(storage_path, supported_formats)

                if last_usn == 0:
                    # First run - get current USN and do full scan
                    current_usn = tracker.get_current_usn()
                    logger.info(f"First run - saving USN checkpoint: {current_usn}")

                    # Do full scan
                    files = self._full_scan_files(storage_path, supported_formats)
                    self.save_usn_checkpoint(drive_letter, current_usn, len(files))
                    return files

                # Get changes from USN Journal
                changes = tracker.get_changes_since(last_usn)

                if changes.get('full_scan_required'):
                    logger.warning("USN Journal overflow - falling back to full scan")
                    files = self._full_scan_files(storage_path, supported_formats)
                    new_usn = changes.get('next_usn', 0)
                    if new_usn:
                        self.save_usn_checkpoint(drive_letter, new_usn, len(files))
                    return files

                # Build filename -> path index from known files
                known_files = self.get_all_known_files()
                filename_index = {Path(p).name: p for p in known_files}

                # Match added files to full paths
                added_paths = []
                for filename in changes['added']:
                    # Search in storage path
                    for found_path in storage_path_obj.rglob(filename):
                        if found_path.is_file() and found_path.suffix.lower() in supported_formats:
                            added_paths.append(str(found_path))
                            break

                # Match modified files
                modified_paths = []
                for filename in changes['modified']:
                    if filename in filename_index:
                        path = filename_index[filename]
                        if Path(path).exists():
                            modified_paths.append(path)

                # Handle deleted files (cleanup DB)
                deleted_paths = []
                for filename in changes['deleted']:
                    if filename in filename_index:
                        path = filename_index[filename]
                        if not Path(path).exists():
                            deleted_paths.append(path)

                # Cleanup deleted files from DB
                if deleted_paths:
                    self._cleanup_deleted_files(deleted_paths)

                # Save new checkpoint
                new_usn = changes.get('next_usn', 0)
                if new_usn:
                    self.save_usn_checkpoint(drive_letter, new_usn)

                logger.info(f"NTFS fast scan: {len(added_paths)} new, {len(modified_paths)} modified, {len(deleted_paths)} deleted")

                return added_paths + modified_paths

            except Exception as e:
                logger.warning(f"NTFS USN Journal failed: {e}. Falling back to full scan.")

        # Fallback to full scan
        return self._full_scan_files(storage_path, supported_formats)

    def _full_scan_files(self, storage_path: str, supported_formats: Set[str]) -> List[str]:
        """Полное сканирование директории (fallback)"""
        logger.info(f"Starting full directory scan of {storage_path}...")
        start_time = time.time()

        files = []
        storage_path_obj = Path(storage_path)
        file_count = 0

        for file_path in storage_path_obj.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in supported_formats:
                continue

            files.append(str(file_path))
            file_count += 1

            if file_count % 10000 == 0:
                logger.info(f"Scanning: {file_count} files...")

        elapsed = time.time() - start_time
        logger.info(f"Full scan completed: {len(files)} files in {elapsed:.1f}s")

        return files

    def _cleanup_deleted_files(self, deleted_paths: List[str]):
        """Удалить записи для удалённых файлов"""
        if not deleted_paths:
            return

        session = self.db_manager.get_session()
        try:
            from models.data_models import PhotoIndex

            for path in deleted_paths:
                session.query(PhotoIndex).filter_by(file_path=path).delete()

            session.commit()
            logger.info(f"Cleaned up {len(deleted_paths)} deleted files from DB")
        except Exception as e:
            logger.error(f"Failed to cleanup deleted files: {e}")
            session.rollback()
        finally:
            session.close()

    def cleanup_missing_files(self, check_only: bool = False) -> Dict[str, int]:
        """
        Удалить записи для файлов, которых нет на диске
        
        Args:
            check_only: Если True, только проверить (не удалять)
            
        Returns:
            {checked: количество проверено, missing: найдено потеряных, deleted: удалено}
        """
        session = self.db_manager.get_session()
        stats = {'checked': 0, 'missing': 0, 'deleted': 0}
        
        try:
            from models.data_models import PhotoIndex
            
            # Получить общее количество для прогресса
            total_count = session.query(PhotoIndex).count()
            logger.info(f"Проверка {total_count} записей на orphaned файлы...")
            
            batch_size = 1000
            offset = 0
            to_delete = []
            
            while offset < total_count:
                # Обрабатываем батчами для прогресса
                photos = session.query(PhotoIndex).offset(offset).limit(batch_size).all()
                
                for photo in photos:
                    stats['checked'] += 1
                    
                    if not Path(photo.file_path).exists():
                        stats['missing'] += 1
                        to_delete.append(photo.image_id)
                        
                        if stats['missing'] <= 10:  # Логируем только первые 10
                            logger.warning(f"Missing file: {photo.file_path}")
                
                # Прогресс каждые 5000 записей
                if stats['checked'] % 5000 == 0:
                    logger.info(f"Проверено {stats['checked']}/{total_count} ({stats['missing']} missing)")
                
                offset += batch_size
            
            # Удаление найденных orphaned записей
            if to_delete and not check_only:
                logger.info(f"Удаление {len(to_delete)} orphaned записей...")
                
                # Удаляем батчами по 500
                for i in range(0, len(to_delete), 500):
                    batch_ids = to_delete[i:i+500]
                    session.query(PhotoIndex).filter(PhotoIndex.image_id.in_(batch_ids)).delete(synchronize_session=False)
                    stats['deleted'] += len(batch_ids)
                    
                    if (i + 500) % 5000 == 0:
                        logger.info(f"Удалено {stats['deleted']}/{len(to_delete)}")
                
                session.commit()
                logger.info(f"Удалено {stats['deleted']} orphaned записей")
            elif check_only:
                session.rollback()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            session.rollback()
        finally:
            session.close()
        
        return stats

    def index_batch(self, file_paths: List[str]) -> Dict[str, int]:
        """
        Индексировать батч изображений с GPU ускорением

        Args:
            file_paths: Список путей к файлам

        Returns:
            {successful: кол-во успешных, failed: кол-во ошибок, skipped: пропущено}
        """
        if not self.clip_embedder:
            logger.error("CLIP embedder не инициализирован!")
            return {'successful': 0, 'failed': len(file_paths), 'skipped': 0}

        results = {'successful': 0, 'failed': 0, 'skipped': 0}
        total = len(file_paths)

        if total == 0:
            logger.info("Нет файлов для индексации")
            return results

        # Фильтруем уже проиндексированные файлы
        logger.info("Проверка уже проиндексированных файлов...")
        indexed_paths = self.get_indexed_paths()

        files_to_index = []
        for fp in file_paths:
            if fp in indexed_paths:
                results['skipped'] += 1
            elif self.image_processor.is_supported_format(fp):
                files_to_index.append(fp)
            else:
                results['failed'] += 1

        if results['skipped'] > 0:
            logger.info(f"Пропущено (уже в индексе): {results['skipped']}")

        total_to_process = len(files_to_index)
        if total_to_process == 0:
            logger.info("Все файлы уже проиндексированы")
            return results

        # Инициализация прогресса
        self._progress["total_files"] = total_to_process
        self._progress["processed_files"] = 0
        self._progress["successful"] = 0
        self._progress["failed"] = 0
        self._progress["skipped"] = results['skipped']  # Уже пропущенные
        self._progress["current_batch"] = 0
        self._progress["total_batches"] = (total_to_process + self.batch_size - 1) // self.batch_size
        self._progress["speed_imgs_per_sec"] = 0.0
        self._progress["eta_seconds"] = 0

        logger.info(f"Начинаю индексацию {total_to_process} файлов (batch_size={self.batch_size})")
        start_time = time.time()
        self._start_time = start_time
        self._stop_requested = False

        # Обрабатываем батчами
        for batch_start in range(0, total_to_process, self.batch_size):
            # Проверяем запрос на остановку
            if self._stop_requested:
                logger.info(f"CLIP indexing stopped by user request after {results['successful']} files")
                break

            batch_end = min(batch_start + self.batch_size, total_to_process)
            batch_files = files_to_index[batch_start:batch_end]
            batch_num = batch_start // self.batch_size + 1
            total_batches = (total_to_process + self.batch_size - 1) // self.batch_size

            batch_start_time = time.time()

            # Получаем эмбеддинги батчем на GPU
            embeddings = self.clip_embedder.embed_images_batch(batch_files, batch_size=len(batch_files))

            gpu_time = time.time() - batch_start_time

            # Сохраняем в БД
            db_start_time = time.time()
            session = self.db_manager.get_session()

            try:
                from models.data_models import PhotoIndex
                embedding_column_name = self.clip_embedder.get_embedding_column()

                for i, (file_path, embedding) in enumerate(zip(batch_files, embeddings)):
                    if embedding is None:
                        results['failed'] += 1
                        continue

                    try:
                        # Проверяем, есть ли уже запись для этого файла
                        existing = session.query(PhotoIndex).filter_by(file_path=file_path).first()

                        if existing:
                            # UPDATE существующей записи
                            setattr(existing, embedding_column_name, embedding.tolist())

                            # Также обновляем EXIF данные, если их ещё нет
                            if existing.latitude is None or existing.longitude is None or existing.photo_date is None:
                                try:
                                    metadata = self.image_processor.get_full_metadata(file_path)
                                    if metadata.get('latitude') is not None and existing.latitude is None:
                                        existing.latitude = metadata['latitude']
                                        existing.longitude = metadata['longitude']
                                    if metadata.get('photo_date') is not None and existing.photo_date is None:
                                        existing.photo_date = metadata['photo_date']
                                except Exception as exif_err:
                                    logger.debug(f"EXIF extraction failed for {file_path}: {exif_err}")
                        else:
                            # INSERT новой записи
                            file_info = self.image_processor.get_file_info(file_path)

                            # Извлекаем EXIF данные (GPS, дата съёмки)
                            latitude = None
                            longitude = None
                            photo_date = None
                            exif_data = None
                            try:
                                metadata = self.image_processor.get_full_metadata(file_path)
                                latitude = metadata.get('latitude')
                                longitude = metadata.get('longitude')
                                photo_date = metadata.get('photo_date')
                                exif_data = metadata.get('exif_data')
                            except Exception as exif_err:
                                logger.debug(f"EXIF extraction failed for {file_path}: {exif_err}")

                            photo_data = {
                                'file_path': file_path,
                                'file_name': file_info.get('file_name'),
                                'file_size': file_info.get('file_size'),
                                'file_format': file_info.get('file_format'),
                                'width': file_info.get('width'),
                                'height': file_info.get('height'),
                                'latitude': latitude,
                                'longitude': longitude,
                                'photo_date': photo_date,
                                'exif_data': exif_data,
                                embedding_column_name: embedding.tolist(),
                            }
                            self.photo_repo.add_photo(session, photo_data)
                        
                        session.commit()
                        results['successful'] += 1

                    except Exception as e:
                        session.rollback()
                        logger.warning(f"Ошибка сохранения {file_path}: {e}")
                        results['failed'] += 1
            finally:
                session.close()

            db_time = time.time() - db_start_time
            total_batch_time = time.time() - batch_start_time
            imgs_per_sec = len(batch_files) / total_batch_time if total_batch_time > 0 else 0

            # Прогресс и ETA
            processed = batch_end
            elapsed = time.time() - self._start_time
            eta = (elapsed / processed) * (total_to_process - processed) if processed > 0 else 0
            speed = processed / elapsed if elapsed > 0 else 0

            # Обновление прогресса
            self._progress["processed_files"] = processed
            self._progress["successful"] = results['successful']
            self._progress["failed"] = results['failed']
            self._progress["skipped"] = results['skipped']
            self._progress["current_batch"] = batch_num
            self._progress["total_batches"] = total_batches
            self._progress["speed_imgs_per_sec"] = round(speed, 2)
            self._progress["eta_seconds"] = int(eta)

            logger.info(
                f"[{batch_num}/{total_batches}] "
                f"{processed}/{total_to_process} | "
                f"GPU: {gpu_time:.2f}s, DB: {db_time:.2f}s | "
                f"{imgs_per_sec:.1f} img/s | "
                f"ETA: {eta:.0f}s"
            )

        total_time = time.time() - start_time
        avg_speed = results['successful'] / total_time if total_time > 0 else 0

        logger.info("=" * 50)
        logger.info(f"Индексирование завершено за {total_time:.1f}s")
        logger.info(f"  Успешно: {results['successful']}")
        logger.info(f"  Ошибок: {results['failed']}")
        logger.info(f"  Пропущено: {results['skipped']}")
        logger.info(f"  Скорость: {avg_speed:.1f} img/s")
        logger.info("=" * 50)

        return results

    def index_image(self, file_path: str) -> Optional[str]:
        """Индексировать одно изображение (обёртка для совместимости)"""
        results = self.index_batch([file_path])
        return file_path if results['successful'] > 0 else None

    def delete_index_entry(self, image_id: str):
        """Удалить запись из индекса"""
        session = self.db_manager.get_session()
        try:
            self.photo_repo.delete_photo(session, image_id)
            logger.info(f"Запись удалена: {image_id}")
        finally:
            session.close()

    def get_indexing_status(self) -> Dict:
        """Получить статус индексирования для текущей модели"""
        session = self.db_manager.get_session()
        try:
            from models.data_models import PhotoIndex
            total_photos = session.query(PhotoIndex).count()
            
            # Проверяем наличие эмбеддинга для текущей модели
            if self.clip_embedder:
                embedding_column_name = self.clip_embedder.get_embedding_column()
                embedding_column = getattr(PhotoIndex, embedding_column_name)
                indexed_photos = session.query(PhotoIndex).filter(embedding_column != None).count()
            else:
                indexed_photos = 0
            
            return {
                'total': total_photos,
                'indexed': indexed_photos,
                'pending': total_photos - indexed_photos,
                'percentage': (indexed_photos / total_photos * 100) if total_photos > 0 else 0
            }
        finally:
            session.close()
    
    def request_stop(self):
        """Запросить остановку индексации после текущего батча"""
        self._stop_requested = True
        logger.info("CLIP indexing stop requested")

    def get_progress(self) -> Dict:
        """Получить текущий прогресс индексации (live data)"""
        return dict(self._progress)

    def clear_index(self):
        """Очистить весь индекс"""
        session = self.db_manager.get_session()
        try:
            from models.data_models import PhotoIndex
            count = session.query(PhotoIndex).delete()
            session.commit()
            logger.info(f"Индекс очищен, удалено {count} записей")
        finally:
            session.close()
