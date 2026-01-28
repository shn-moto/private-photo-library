#!/usr/bin/env python3
"""Тестирование очистки и переиндексирования"""

import logging
from pathlib import Path
from config.settings import settings
from db.database import DatabaseManager
from models.data_models import PhotoIndex, FaceRecord

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_orphaned_detection():
    """Тест: обнаружение orphaned записей"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Обнаружение orphaned записей")
    logger.info("=" * 60)
    
    db_manager = DatabaseManager(settings.DATABASE_URL)
    session = db_manager.get_session()
    
    try:
        # Получить все записи
        photos = session.query(PhotoIndex).all()
        logger.info(f"Всего записей в индексе: {len(photos)}")
        
        # Проверить существование файлов
        missing_count = 0
        existing_count = 0
        
        for photo in photos:
            path = Path(photo.file_path)
            if path.exists():
                existing_count += 1
            else:
                missing_count += 1
                logger.warning(f"  ❌ ORPHANED: {photo.file_path}")
        
        logger.info(f"\nРезультаты:")
        logger.info(f"  Существующих файлов: {existing_count}")
        logger.info(f"  Orphaned записей: {missing_count}")
        logger.info(f"  Процент orphaned: {100*missing_count/len(photos):.1f}%")
        
        return missing_count == 0  # Тест пройден если нет orphaned
        
    finally:
        session.close()
        db_manager.close()


def test_cleanup_function():
    """Тест: функция очистки orphaned"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Функция cleanup_missing_files()")
    logger.info("=" * 60)
    
    from services.indexer import IndexingService
    
    indexer = IndexingService()
    
    # Проверить (dry run)
    logger.info("Запуск проверки (check_only=True)...")
    stats = indexer.cleanup_missing_files(check_only=True)
    
    logger.info(f"Результаты:")
    logger.info(f"  Проверено записей: {stats['checked']}")
    logger.info(f"  Найдено missing: {stats['missing']}")
    logger.info(f"  Удалено записей: {stats['deleted']} (должно быть 0 в check_only режиме)")
    
    # Проверка
    test_passed = stats['deleted'] == 0  # В dry-run режиме ничего не должно быть удалено
    
    logger.info(f"Тест {'ПРОЙДЕН ✓' if test_passed else 'ПРОВАЛЕН ✗'}")
    return test_passed


def test_indexer_skip_logic():
    """Тест: логика пропуска уже проиндексированных файлов"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Логика пропуска в индексере")
    logger.info("=" * 60)
    
    from services.indexer import IndexingService
    
    indexer = IndexingService()
    
    # Получить уже проиндексированные пути
    indexed_paths = indexer.get_indexed_paths()
    
    logger.info(f"Всего проиндексировано: {len(indexed_paths)}")
    
    # Проверить существование каждого пути
    missing_in_indexed = 0
    for path in indexed_paths:
        if not Path(path).exists():
            missing_in_indexed += 1
            logger.warning(f"  Missing: {path}")
    
    logger.info(f"\nВ индексе есть missing файлов: {missing_in_indexed}")
    
    if missing_in_indexed > 0:
        logger.warning("⚠️ ПРОБЛЕМА: В индексе есть несуществующие файлы!")
        logger.info("Рекомендация: Запустите cleanup_orphaned.py операцию 2")
    
    return missing_in_indexed == 0


def print_summary():
    """Вывести сводку по статусу"""
    logger.info("\n" + "=" * 60)
    logger.info("СВОДКА")
    logger.info("=" * 60)
    
    db_manager = DatabaseManager(settings.DATABASE_URL)
    session = db_manager.get_session()
    
    try:
        total_photos = session.query(PhotoIndex).count()
        indexed = session.query(PhotoIndex).filter_by(indexed=1).count()
        total_faces = session.query(FaceRecord).count()
        
        logger.info(f"Всего фотографий в индексе: {total_photos}")
        logger.info(f"  - Проиндексировано: {indexed}")
        logger.info(f"  - Не проиндексировано: {total_photos - indexed}")
        logger.info(f"Всего лиц в БД: {total_faces}")
        
    finally:
        session.close()
        db_manager.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ОЧИСТКИ И ПЕРЕИНДЕКСИРОВАНИЯ")
    print("=" * 60)
    
    results = {
        'orphaned_detection': test_orphaned_detection(),
        'cleanup_function': test_cleanup_function(),
        'indexer_skip_logic': test_indexer_skip_logic(),
    }
    
    print_summary()
    
    logger.info("\n" + "=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ТЕСТОВ:")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ ПРОЙДЕН" if passed else "✗ ПРОВАЛЕН"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "=" * 60)
    logger.info(f"ОБЩИЙ РЕЗУЛЬТАТ: {'✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ' if all_passed else '✗ ЕСТЬ ОШИБКИ'}")
    logger.info("=" * 60)
    
    if not all_passed:
        logger.warning("\n⚠️ Рекомендации:")
        logger.warning("1. Запустите: python scripts/cleanup_orphaned.py 1")
        logger.warning("2. Проверьте orphaned файлы")
        logger.warning("3. Если есть orphaned, запустите: python scripts/cleanup_orphaned.py 2")
        logger.warning("4. После этого переиндексируйте: python scripts/cleanup_orphaned.py 3")
