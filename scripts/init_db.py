"""Script для инициализации базы данных"""

import logging
import sys
from pathlib import Path

# Добавить корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from db.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Инициализировать базу данных"""
    try:
        db_manager = DatabaseManager(settings.DATABASE_URL)
        
        logger.info("Создание таблиц в БД...")
        db_manager.init_tables()
        
        logger.info("БД успешно инициализирована!")
        
    except Exception as e:
        logger.error(f"Ошибка инициализации БД: {e}")
        sys.exit(1)


if __name__ == "__main__":
    init_database()
