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
    """Главная функция приложения - idle daemon, индексация только через API"""
    logger.info("=" * 60)
    logger.info("Smart Photo Indexing Service запущен (idle mode)")
    logger.info(f"Хранилище: {settings.PHOTO_STORAGE_PATH}")
    logger.info("Индексация запускается через API: POST /reindex/files")
    logger.info("=" * 60)

    # Просто держим контейнер живым, индексация через API
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Получен сигнал выхода (Ctrl+C)")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Smart Photo Indexing Service остановлен")


if __name__ == "__main__":
    main()
