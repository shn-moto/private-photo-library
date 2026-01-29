"""Модели данных для сервиса индексации"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
import uuid

Base = declarative_base()

# Размерности эмбиддингов
CLIP_EMBEDDING_DIM = 1152  # SigLIP so400m (legacy, для обратной совместимости)

# Маппинг моделей на колонки БД
CLIP_MODEL_COLUMNS = {
    "ViT-B/32": "clip_embedding_vit_b32",
    "ViT-B/16": "clip_embedding_vit_b16",
    "ViT-L/14": "clip_embedding_vit_l14",
    "SigLIP": "clip_embedding_siglip",
}

CLIP_MODEL_DIMS = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "SigLIP": 1152,
}


# ==================== Pydantic модели ====================


class ImageMetadata(BaseModel):
    """Метаданные об изображении"""
    image_id: Optional[int] = None
    file_path: str
    file_name: str
    file_size: int
    file_format: str  # heic, jpg, png, etc

    # Дата и время
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)
    photo_date: Optional[datetime] = None  # Дата съемки из EXIF

    # Размеры
    width: int
    height: int

    # CLIP embedding
    clip_embedding: Optional[List[float]] = None

    # Дополнительные метаданные
    exif_data: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Запрос для поиска"""
    query_type: str  # "text", "image", "face"
    query: Union[str, bytes]  # текст или путь/данные изображения
    top_k: int = 10
    similarity_threshold: float = 0.5


class SearchResult(BaseModel):
    """Результат поиска"""
    image_id: int
    file_path: str
    similarity: float


# ==================== SQLAlchemy модели ====================

class PhotoIndex(Base):
    """Таблица индекса фотографий"""
    __tablename__ = "photo_index"

    image_id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(1024), nullable=False, unique=True)
    file_name = Column(String(256), nullable=False)
    file_size = Column(Integer)
    file_format = Column(String(10))

    width = Column(Integer)
    height = Column(Integer)

    # Временные метки
    created_at = Column(DateTime, default=datetime.now)
    modified_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    photo_date = Column(DateTime, nullable=True)

    # Model-specific embedding колонки
    clip_embedding_vit_b32 = Column(Vector(512), nullable=True)
    clip_embedding_vit_b16 = Column(Vector(512), nullable=True)
    clip_embedding_vit_l14 = Column(Vector(768), nullable=True)
    clip_embedding_siglip = Column(Vector(1152), nullable=True)

    # Метаданные
    exif_data = Column(JSONB, nullable=True)

    # Индексы для быстрого поиска
    __table_args__ = (
        Index('idx_photo_index_file_format', 'file_format'),
    )
