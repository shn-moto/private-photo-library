#!/usr/bin/env python3
"""
Скрипт для извлечения EXIF данных из всех фотографий в БД.
Заполняет колонки: photo_date, latitude, longitude, exif_data

Запуск:
    python scripts/populate_exif_data.py [--batch-size 100] [--force]

Опции:
    --batch-size N  Количество фото в пакете (default: 100)
    --force         Перезаписать существующие данные
    --dry-run       Только показать статистику, не записывать
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Добавить корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from sqlalchemy import text

from config.settings import settings
from db.database import DatabaseManager
from services.image_processor import ImageProcessor
from models.data_models import PhotoIndex

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def populate_exif_data(batch_size: int = 100, force: bool = False, dry_run: bool = False):
    """
    Извлечь EXIF данные из всех фотографий и записать в БД

    Args:
        batch_size: размер пакета для обработки
        force: перезаписать существующие данные
        dry_run: только показать статистику
    """
    logger.info("Подключение к БД...")
    db_manager = DatabaseManager(settings.DATABASE_URL)
    session = db_manager.get_session()

    processor = ImageProcessor()

    try:
        # Получить общее количество записей
        if force:
            # Обработать все записи
            query = session.query(PhotoIndex)
        else:
            # Только записи без EXIF данных
            query = session.query(PhotoIndex).filter(
                PhotoIndex.exif_data == None
            )

        total_count = query.count()
        logger.info(f"Всего записей для обработки: {total_count}")

        if dry_run:
            # Показать статистику
            total = session.query(PhotoIndex).count()
            with_gps = session.query(PhotoIndex).filter(
                PhotoIndex.latitude != None,
                PhotoIndex.longitude != None
            ).count()
            with_date = session.query(PhotoIndex).filter(
                PhotoIndex.photo_date != None
            ).count()
            with_exif = session.query(PhotoIndex).filter(
                PhotoIndex.exif_data != None
            ).count()

            logger.info(f"\n=== Статистика БД ===")
            logger.info(f"Всего фотографий: {total}")
            logger.info(f"С GPS координатами: {with_gps} ({100*with_gps/total:.1f}%)" if total > 0 else "С GPS: 0")
            logger.info(f"С датой съемки: {with_date} ({100*with_date/total:.1f}%)" if total > 0 else "С датой: 0")
            logger.info(f"С EXIF данными: {with_exif} ({100*with_exif/total:.1f}%)" if total > 0 else "С EXIF: 0")
            return

        if total_count == 0:
            logger.info("Нет записей для обработки")
            return

        # Счетчики
        processed = 0
        updated_gps = 0
        updated_date = 0
        updated_exif = 0
        errors = 0
        not_found = 0

        # Получить все ID для обработки (чтобы избежать проблем с OFFSET при изменении данных)
        all_ids = [r.image_id for r in query.order_by(PhotoIndex.image_id).with_entities(PhotoIndex.image_id).all()]
        total_count = len(all_ids)

        logger.info(f"Получено {total_count} ID для обработки")

        # Обработка пакетами по ID
        pbar = tqdm(total=total_count, desc="Обработка EXIF")

        for i in range(0, total_count, batch_size):
            batch_ids = all_ids[i:i + batch_size]
            batch = session.query(PhotoIndex).filter(PhotoIndex.image_id.in_(batch_ids)).all()

            for photo in batch:
                try:
                    file_path = photo.file_path

                    # Проверить существование файла
                    if not os.path.exists(file_path):
                        not_found += 1
                        pbar.update(1)
                        continue

                    # Извлечь метаданные
                    metadata = processor.get_full_metadata(file_path)

                    # Обновить GPS
                    if metadata.get('latitude') is not None and metadata.get('longitude') is not None:
                        if force or photo.latitude is None:
                            photo.latitude = metadata['latitude']
                            photo.longitude = metadata['longitude']
                            updated_gps += 1

                    # Обновить дату съемки
                    if metadata.get('photo_date') is not None:
                        if force or photo.photo_date is None:
                            photo.photo_date = metadata['photo_date']
                            updated_date += 1

                    # Обновить EXIF
                    if metadata.get('exif_data'):
                        if force or photo.exif_data is None:
                            photo.exif_data = metadata['exif_data']
                            updated_exif += 1

                    processed += 1

                except Exception as e:
                    logger.debug(f"Ошибка обработки {photo.file_path}: {e}")
                    errors += 1

                pbar.update(1)

            # Коммит пакета
            try:
                session.commit()
            except Exception as e:
                logger.error(f"Ошибка коммита пакета: {e}")
                session.rollback()

        pbar.close()

        # Итоговая статистика
        logger.info(f"\n=== Результаты ===")
        logger.info(f"Обработано: {processed}")
        logger.info(f"Обновлено GPS: {updated_gps}")
        logger.info(f"Обновлено дат: {updated_date}")
        logger.info(f"Обновлено EXIF: {updated_exif}")
        logger.info(f"Файлов не найдено: {not_found}")
        logger.info(f"Ошибок: {errors}")

        # Финальная статистика БД
        total = session.query(PhotoIndex).count()
        with_gps = session.query(PhotoIndex).filter(
            PhotoIndex.latitude != None,
            PhotoIndex.longitude != None
        ).count()
        with_date = session.query(PhotoIndex).filter(
            PhotoIndex.photo_date != None
        ).count()

        logger.info(f"\n=== Статистика БД ===")
        logger.info(f"Всего фотографий: {total}")
        logger.info(f"С GPS координатами: {with_gps} ({100*with_gps/total:.1f}%)" if total > 0 else "С GPS: 0")
        logger.info(f"С датой съемки: {with_date} ({100*with_date/total:.1f}%)" if total > 0 else "С датой: 0")

    finally:
        session.close()
        db_manager.close()


def main():
    parser = argparse.ArgumentParser(
        description='Извлечь EXIF данные из всех фотографий в БД'
    )
    parser.add_argument(
        '--batch-size', type=int, default=100,
        help='Количество фото в пакете (default: 100)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Перезаписать существующие данные'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Только показать статистику, не записывать'
    )

    args = parser.parse_args()

    logger.info(f"Запуск скрипта populate_exif_data")
    logger.info(f"  batch_size: {args.batch_size}")
    logger.info(f"  force: {args.force}")
    logger.info(f"  dry_run: {args.dry_run}")

    populate_exif_data(
        batch_size=args.batch_size,
        force=args.force,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
