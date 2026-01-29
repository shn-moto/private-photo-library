"""Главный файл приложения"""

import logging
import logging.handlers
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Добавить корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent))

# Отключить прогресс-бары transformers/huggingface (до импорта!)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

from config.settings import settings
from services.indexer import IndexingService
from services.file_monitor import FileMonitor

# Настройка логирования с записью в файл
def setup_logging():
    """Настроить логирование: консоль + файл"""
    log_level = getattr(logging, settings.LOG_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Корневой логгер
    root_logger = logging.getLogger()
    
    # Проверить, не настроено ли уже логирование
    if root_logger.handlers:
        return
    
    root_logger.setLevel(log_level)

    # Консоль (только WARNING+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Файл (все уровни) с ротацией
    if settings.LOG_FILE:
        log_dir = Path(settings.LOG_FILE).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Заглушить шумные библиотеки
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Главная функция приложения"""
    logger.info("=" * 60)
    logger.info("Smart Photo Indexing Service запущен")
    logger.info(f"Хранилище: {settings.PHOTO_STORAGE_PATH}")
    logger.info(f"БД: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'unknown'}")
    logger.info("=" * 60)
    
    # Инициализировать сервисы
    indexing_service = IndexingService()
    file_monitor = FileMonitor(
        settings.PHOTO_STORAGE_PATH,
        settings.SUPPORTED_FORMATS
    )
    
    # Начальное сканирование
    logger.info("Начальное сканирование хранилища...")
    initial_files = file_monitor.scan_directory()
    file_monitor.print_stats()

    # Первичная индексация всех найденных файлов
    if initial_files:
        logger.info(f"Найдено {len(initial_files)} файлов для индексации")
        indexing_service.index_batch(list(initial_files.keys()))
        status = indexing_service.get_indexing_status()
        logger.info(f"Первичная индексация завершена: {status['indexed']}/{status['total']}")

    # Мониторинг отключён — используйте POST /reindex через API
    logger.info("Автоматический мониторинг отключён. Используйте POST /reindex для ручной переиндексации.")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Получен сигнал выхода (Ctrl+C)")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Завершение работы сервиса...")
        indexing_service.db_manager.close()
        logger.info("Smart Photo Indexing Service остановлен")


if __name__ == "__main__":
    main()
