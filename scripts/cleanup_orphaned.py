"""Скрипт очистки orphaned записей в индексе"""

import logging
import sys
from pathlib import Path
import os

# Добавить родительскую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from db.database import DatabaseManager, PhotoIndexRepository
from models.data_models import PhotoIndex

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Маппинг путей между контейнером и хостом
CONTAINER_PHOTOS_PATH = "/photos"
HOST_PHOTOS_PATH = os.getenv("PHOTOS_HOST_PATH", "H:/PHOTO")


def map_container_path_to_host(container_path: str) -> str:
    """
    Преобразовать путь из контейнера в путь на хосте
    
    Пример:
        /photos/СЕМЬЯ/foto.jpg -> H:/PHOTO/СЕМЬЯ/foto.jpg
    """
    if container_path.startswith(CONTAINER_PHOTOS_PATH):
        # Заменить /photos на локальный путь
        relative_path = container_path[len(CONTAINER_PHOTOS_PATH):].lstrip("/")
        host_path = os.path.join(HOST_PHOTOS_PATH, relative_path)
        # Нормализовать слэши для Windows
        return host_path.replace("\\", "/")
    return container_path


def cleanup_orphaned_records(dry_run: bool = True) -> dict:
    """
    Удалить записи в индексе для файлов, которых нет на диске
    
    Args:
        dry_run: Если True, только показать что будет удалено (не удалять)
        
    Returns:
        Статистика очистки
    """
    db_manager = DatabaseManager(settings.DATABASE_URL)
    photo_repo = PhotoIndexRepository(db_manager)
    
    session = db_manager.get_session()
    stats = {
        'total_records': 0,
        'missing_files': 0,
        'deleted': 0,
        'errors': 0,
    }
    
    try:
        logger.info("=" * 60)
        logger.info("ОЧИСТКА ORPHANED ЗАПИСЕЙ")
        logger.info(f"Маппинг путей: {CONTAINER_PHOTOS_PATH} -> {HOST_PHOTOS_PATH}")
        logger.info(f"Режим: {'DRY RUN (только проверка)' if dry_run else 'РЕАЛЬНОЕ УДАЛЕНИЕ'}")
        logger.info("=" * 60)
        
        # Получить все записи из индекса
        photos = session.query(PhotoIndex).all()
        stats['total_records'] = len(photos)
        
        logger.info(f"Всего записей в индексе: {stats['total_records']}")
        
        # Проверить каждый файл
        for photo in photos:
            file_path = photo.file_path
            # Преобразовать путь из контейнера в путь на хосте
            host_file_path = map_container_path_to_host(file_path)
            
            # Проверить существование файла
            if not Path(host_file_path).exists():
                stats['missing_files'] += 1
                logger.warning(f"  ❌ MISSING: {file_path} -> {host_file_path}")
                
                if not dry_run:
                    try:
                        # Удалить саму фотографию
                        session.delete(photo)
                        stats['deleted'] += 1
                        
                    except Exception as e:
                        stats['errors'] += 1
                        logger.error(f"  Error deleting {file_path}: {e}")
        
        # Коммитить если не dry_run
        if not dry_run and stats['deleted'] > 0:
            session.commit()
            logger.info(f"\nИзменения сохранены в БД")
        elif dry_run:
            session.rollback()
        
        # Статистика
        logger.info("=" * 60)
        logger.info("СТАТИСТИКА:")
        logger.info(f"  Всего записей: {stats['total_records']}")
        logger.info(f"  Потеряны файлы: {stats['missing_files']}")
        logger.info(f"  Удалено записей о фото: {stats['deleted']}")
        logger.info(f"  Ошибок при удалении: {stats['errors']}")
        logger.info("=" * 60)
        
        return stats
        
    finally:
        session.close()
        db_manager.close()


def reindex_all_existing_files(image_dir: str = None) -> dict:
    """
    Переиндексировать все существующие файлы в директории
    
    Args:
        image_dir: Директория с изображениями (если None, используется DEFAULT_IMAGE_DIR)
        
    Returns:
        Статистика переиндексации
    """
    if image_dir is None:
        image_dir = settings.DEFAULT_IMAGE_DIR
    
    image_path = Path(image_dir)
    if not image_path.exists():
        logger.error(f"Директория не найдена: {image_path}")
        return {'total': 0, 'successful': 0, 'failed': 0}
    
    logger.info("=" * 60)
    logger.info("ПЕРЕИНДЕКСИРОВАНИЕ СУЩЕСТВУЮЩИХ ФАЙЛОВ")
    logger.info(f"Директория: {image_path}")
    logger.info("=" * 60)
    
    from services.indexer import IndexingService
    
    indexer = IndexingService()
    
    # Получить все поддерживаемые файлы
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.heic')
    image_files = []
    
    for fmt in supported_formats:
        image_files.extend(image_path.glob(f'**/*{fmt}'))
        image_files.extend(image_path.glob(f'**/*{fmt.upper()}'))
    
    # Преобразовать в строки пути
    file_paths = [str(f.absolute()) for f in image_files]
    
    logger.info(f"Найдено {len(file_paths)} файлов для индексации")
    
    if not file_paths:
        logger.warning("Не найдены поддерживаемые файлы")
        return {'total': 0, 'successful': 0, 'failed': 0}
    
    # Индексировать
    results = indexer.index_batch(file_paths)
    
    logger.info("=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ПЕРЕИНДЕКСИРОВАНИЯ:")
    logger.info(f"  Всего файлов: {len(file_paths)}")
    logger.info(f"  Успешно индексировано: {results.get('successful', 0)}")
    logger.info(f"  Ошибок: {results.get('failed', 0)}")
    logger.info(f"  Пропущено (уже в индексе): {results.get('skipped', 0)}")
    logger.info("=" * 60)
    
    return {
        'total': len(file_paths),
        'successful': results.get('successful', 0),
        'failed': results.get('failed', 0),
        'skipped': results.get('skipped', 0),
    }


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("УТИЛИТА ОЧИСТКИ И ПЕРЕИНДЕКСИРОВАНИЯ")
    print("=" * 60)
    print("\nДоступные операции:")
    print("  1. Проверить orphaned файлы (dry-run)")
    print("  2. Удалить orphaned записи (реально удалит)")
    print("  3. Переиндексировать все существующие файлы")
    print("  4. Полная очистка и переиндексирование")
    print("\nУказать номер операции:")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Выбор (1-4): ").strip()
    
    try:
        if choice == '1':
            # Dry run - только проверка
            cleanup_orphaned_records(dry_run=True)
            
        elif choice == '2':
            # Реальное удаление
            confirm = input("\n⚠️  ВНИМАНИЕ! Это действительно удалит orphaned записи из БД.\n"
                          "Введите 'да' для подтверждения: ").strip().lower()
            if confirm == 'да':
                cleanup_orphaned_records(dry_run=False)
            else:
                logger.info("Операция отменена")
                
        elif choice == '3':
            # Переиндексирование
            image_dir = None
            if len(sys.argv) > 2:
                image_dir = sys.argv[2]
            
            reindex_all_existing_files(image_dir)
            
        elif choice == '4':
            # Полная очистка и переиндексирование
            confirm = input("\n⚠️  ВНИМАНИЕ! Это удалит orphaned записи и переиндексирует все файлы.\n"
                          "Введите 'да' для подтверждения: ").strip().lower()
            if confirm == 'да':
                # Сначала очистка
                cleanup_orphaned_records(dry_run=False)
                
                # Потом переиндексирование
                print("\n")
                image_dir = None
                if len(sys.argv) > 2:
                    image_dir = sys.argv[2]
                
                reindex_all_existing_files(image_dir)
            else:
                logger.info("Операция отменена")
        else:
            print("Неизвестная операция")
            
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
