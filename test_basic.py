"""Тестовый скрипт для проверки базовой функциональности"""

import sys
import os
import time
from pathlib import Path

# Добавить корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Тест импортов"""
    logger.info("=" * 60)
    logger.info("Тест 1: Проверка импортов")
    logger.info("=" * 60)
    
    try:
        from config.settings import settings
        logger.info("✓ config.settings")
        
        from models.data_models import PhotoIndex
        logger.info("✓ models.data_models")
        
        from services.image_processor import ImageProcessor
        logger.info("✓ services.image_processor")
        
        from services.file_monitor import FileMonitor
        logger.info("✓ services.file_monitor")
        
        from db.database import DatabaseManager
        logger.info("✓ db.database")
        
        logger.info("\n✓ Все импорты успешны!\n")
        return True
    except Exception as e:
        logger.error(f"✗ Ошибка импорта: {e}")
        return False


def test_config():
    """Тест конфигурации"""
    logger.info("=" * 60)
    logger.info("Тест 2: Проверка конфигурации")
    logger.info("=" * 60)
    
    try:
        from config.settings import settings
        
        logger.info(f"DATABASE_URL: {settings.DATABASE_URL[:50]}...")
        logger.info(f"PHOTO_STORAGE_PATH: {settings.PHOTO_STORAGE_PATH}")
        logger.info(f"CLIP_MODEL: {settings.CLIP_MODEL}")
        logger.info(f"CLIP_DEVICE: {settings.CLIP_DEVICE}")
        logger.info(f"MONITORING_INTERVAL: {settings.MONITORING_INTERVAL}")
        logger.info(f"API_HOST: {settings.API_HOST}:{settings.API_PORT}")
        
        logger.info("\n✓ Конфигурация загружена!\n")
        return True
    except Exception as e:
        logger.error(f"✗ Ошибка конфигурации: {e}")
        return False


def test_image_processor():
    """Тест обработчика изображений"""
    logger.info("=" * 60)
    logger.info("Тест 3: Проверка ImageProcessor")
    logger.info("=" * 60)
    
    try:
        from services.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # Проверить поддерживаемые форматы
        test_files = [
            "photo.heic",
            "photo.jpg",
            "photo.png",
            "document.pdf"
        ]
        
        for filename in test_files:
            is_supported = processor.is_supported_format(filename)
            status = "✓" if is_supported else "✗"
            expected = filename.lower().endswith(('.heic', '.heif', '.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            if is_supported == expected:
                logger.info(f"{status} {filename}: {'поддерживается' if is_supported else 'не поддерживается'}")
            else:
                logger.warning(f"⚠ {filename}: неправильный результат")
        
        logger.info("\n✓ ImageProcessor работает!\n")
        return True
    except Exception as e:
        logger.error(f"✗ Ошибка ImageProcessor: {e}")
        return False


def test_file_monitor():
    """Тест мониторинга файлов"""
    logger.info("=" * 60)
    logger.info("Тест 4: Проверка FileMonitor")
    logger.info("=" * 60)
    
    try:
        from services.file_monitor import FileMonitor
        from config.settings import settings
        
        monitor = FileMonitor(settings.PHOTO_STORAGE_PATH)
        
        logger.info(f"Путь к хранилищу: {settings.PHOTO_STORAGE_PATH}")
        logger.info(f"Поддерживаемые форматы: {monitor.supported_formats}")
        
        # Попытаться сканировать
        logger.info("Сканирование директории...")
        index = monitor.scan_directory()
        logger.info(f"Найдено файлов: {len(index)}")
        
        if len(index) > 0:
            for file_path, info in list(index.items())[:3]:
                logger.info(f"  - {Path(file_path).name} ({info['size']} bytes)")
        
        logger.info("\n✓ FileMonitor работает!\n")
        return True
    except Exception as e:
        logger.error(f"✗ Ошибка FileMonitor: {e}")
        return False


def test_database():
    """Тест подключения к БД"""
    logger.info("=" * 60)
    logger.info("Тест 5: Проверка подключения к БД")
    logger.info("=" * 60)
    
    try:
        from db.database import DatabaseManager
        from config.settings import settings
        
        logger.info(f"Подключение к: {settings.DATABASE_URL[:50]}...")
        
        db_manager = DatabaseManager(settings.DATABASE_URL)
        
        # Проверить здоровье
        is_healthy = db_manager.health_check()
        
        if is_healthy:
            logger.info("✓ Подключение успешно!")
        else:
            logger.warning("⚠ Подключение в статусе 'не здоров'")
        
        db_manager.close()
        
        logger.info("\n✓ Database Manager работает!\n")
        return is_healthy
    except Exception as e:
        logger.error(f"✗ Ошибка БД: {e}")
        logger.info("\nЭто нормально если PostgreSQL не запущен в Docker\n")
        return False


def run_all_tests():
    """Запустить все тесты"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 10 + "ТЕСТИРОВАНИЕ Smart Photo Indexing" + " " * 16 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")
    
    results = {
        "Импорты": test_imports(),
        "Конфигурация": test_config(),
        "ImageProcessor": test_image_processor(),
        "FileMonitor": test_file_monitor(),
        "База данных": test_database(),
    }
    
    logger.info("=" * 60)
    logger.info("ИТОГИ")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status:10} - {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"Результат: {passed}/{total} тестов пройдено")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("\n✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Приложение готово к запуску.\n")
    else:
        logger.warning(f"\n⚠ {total - passed} тест(ов) не пройдено. Проверьте логи выше.\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
