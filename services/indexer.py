"""Сервис индексации изображений (батчевая обработка на GPU)"""

import logging
import uuid
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

from db.database import DatabaseManager, PhotoIndexRepository

logger = logging.getLogger(__name__)


class IndexingService:
    """Сервис индексации фотографий (батчевая обработка GPU)"""

    def __init__(self, model_name: Optional[str] = None):
        """Инициализация сервиса индексации
        
        Args:
            model_name: Имя CLIP модели (ViT-B/32, ViT-B/16, ViT-L/14, SigLIP).
                       Если None - используется модель из settings.CLIP_MODEL
        """
        self.db_manager = DatabaseManager(settings.DATABASE_URL)
        self.photo_repo = PhotoIndexRepository(self.db_manager)
        self.image_processor = ImageProcessor(settings.IMAGE_MAX_SIZE)
        self.batch_size = settings.BATCH_SIZE_CLIP

        # CLIP embedder
        if CLIPEmbedder:
            try:
                model = model_name or settings.CLIP_MODEL
                self.clip_embedder = CLIPEmbedder(model, settings.CLIP_DEVICE)
            except Exception as e:
                logger.error(f"Failed to initialize CLIP embedder: {e}")
                self.clip_embedder = None
        else:
            self.clip_embedder = None

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
            
            photos = session.query(PhotoIndex).all()
            stats['checked'] = len(photos)
            
            for photo in photos:
                if not Path(photo.file_path).exists():
                    stats['missing'] += 1
                    logger.warning(f"Missing file: {photo.file_path}")
                    
                    if not check_only:
                        # Удалить саму фотографию
                        session.delete(photo)
                        stats['deleted'] += 1
            
            if not check_only and stats['deleted'] > 0:
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

        logger.info(f"Начинаю индексацию {total_to_process} файлов (batch_size={self.batch_size})")
        start_time = time.time()

        # Обрабатываем батчами
        for batch_start in range(0, total_to_process, self.batch_size):
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

            # Прогресс
            processed = batch_end
            elapsed = time.time() - start_time
            eta = (elapsed / processed) * (total_to_process - processed) if processed > 0 else 0

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
