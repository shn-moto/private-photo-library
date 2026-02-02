import os
from typing import Optional
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Конфигурация приложения"""
    
    model_config = ConfigDict(
        extra='ignore',
        env_file='.env',
        env_file_encoding='utf-8'
    )
    
    # PostgreSQL (используем psycopg вместо psycopg2)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+psycopg://postgres:password@localhost:5432/smart_photo_index"
    )
    
    # CLIP модель (HuggingFace transformers)
    CLIP_MODEL: str = "SigLIP"  # или ViT-B/32, ViT-B/16, ViT-L/14
    CLIP_DEVICE: str = "cuda"  # или "cpu"

    # InsightFace (face detection and embedding)
    FACE_MODEL: str = "buffalo_l"  # InsightFace model name
    FACE_DEVICE: str = "cuda"  # или "cpu"
    FACE_DET_SIZE: tuple = (640, 640)  # Detection input size
    FACE_MIN_SCORE: float = 0.65  # Minimum detection confidence
    FACE_BATCH_SIZE: int = 8  # Batch size for face indexing

    # Face similarity thresholds
    FACE_SIMILARITY_THRESHOLD: float = 0.6  # For auto-assignment
    FACE_SEARCH_THRESHOLD: float = 0.5  # For search results
    
    # Мониторинг файловой системы
    PHOTO_STORAGE_PATH: str = os.path.expanduser("~/Pictures")
    MONITORING_INTERVAL: int = 30000  # секунды (500 минут)
    SUPPORTED_FORMATS: list = [".heic", ".heif", ".jpg", ".jpeg", ".png", ".nef", ".cr2", ".arw"]
    
    # Обработка изображений
    IMAGE_MAX_SIZE: tuple = (1024, 1024)
    BATCH_SIZE_CLIP: int = 16  # Уменьшено для стабильности на Windows (TDR timeout)
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = False
    
    # Логирование (в Docker: /logs/app.log через bind mount)
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None  # задаётся через .env
    
    # Корзина для удалённых файлов
    TRASH_DIR: str = "/.trash"
    
    # Папка для дубликатов (отдельно от обычной корзины)
    DUPLICATES_DIR: str = "/.photo_duplicates"

    # Кэширование
    CACHE_EMBEDDINGS: bool = True
    CACHE_DIR: str = "cache"


settings = Settings()
