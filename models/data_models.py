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
CLIP_EMBEDDING_DIM = 1152  # SigLIP so400m (was 512 for ViT-B/32)
FACE_EMBEDDING_DIM = 512   # ArcFace default


# ==================== Pydantic модели ====================

class FaceMetadata(BaseModel):
    """Метаданные о лице"""
    face_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    x1: float  # Координаты bounding box
    y1: float
    x2: float
    y2: float
    confidence: float  # Уверенность детекции

    # Атрибуты лица
    age: Optional[int] = None
    gender: Optional[str] = None  # "M" или "F"
    emotion: Optional[str] = None  # "happy", "sad", "neutral", etc
    ethnicity: Optional[str] = None
    landmarks: Optional[List[tuple]] = None  # Ключевые точки лица

    embedding: Optional[List[float]] = None  # Face embedding вектор


class ImageMetadata(BaseModel):
    """Метаданные об изображении"""
    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
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

    # Лица на фото
    faces: List[FaceMetadata] = []

    # Дополнительные метаданные
    exif_data: Optional[Dict[str, Any]] = None

    # Статус
    indexed: bool = False
    indexed_at: Optional[datetime] = None


class SearchRequest(BaseModel):
    """Запрос для поиска"""
    query_type: str  # "text", "image", "face"
    query: Union[str, bytes]  # текст или путь/данные изображения
    top_k: int = 10
    similarity_threshold: float = 0.5


class SearchResult(BaseModel):
    """Результат поиска"""
    image_id: str
    file_path: str
    similarity: float
    matched_faces: Optional[List[Dict[str, Any]]] = None


# ==================== SQLAlchemy модели ====================

class PhotoIndex(Base):
    """Таблица индекса фотографий"""
    __tablename__ = "photo_index"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(String(256), unique=True, nullable=False, index=True)
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

    # CLIP Embedding - используем pgvector!
    clip_embedding = Column(Vector(CLIP_EMBEDDING_DIM), nullable=True)

    # Метаданные
    exif_data = Column(JSONB, nullable=True)

    # Статус индексации
    indexed = Column(Integer, default=0)  # 0/1
    indexed_at = Column(DateTime, nullable=True)

    # Дополнительные данные
    meta_data = Column(JSONB, default={})

    # Индексы для быстрого поиска
    __table_args__ = (
        Index('idx_photo_index_indexed', 'indexed'),
        Index('idx_photo_index_file_format', 'file_format'),
    )


class FaceRecord(Base):
    """Таблица записей о лицах"""
    __tablename__ = "faces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    face_id = Column(String(256), unique=True, nullable=False, index=True)
    photo_id = Column(String(256), nullable=False, index=True)

    # Координаты bounding box
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    confidence = Column(Float)

    # Атрибуты лица
    age = Column(Integer, nullable=True)
    gender = Column(String(1), nullable=True)  # M/F
    emotion = Column(String(20), nullable=True)
    ethnicity = Column(String(50), nullable=True)
    landmarks = Column(JSON, nullable=True)

    # Face embedding - используем pgvector!
    face_embedding = Column(Vector(FACE_EMBEDDING_DIM), nullable=True)

    # Временные метки
    created_at = Column(DateTime, default=datetime.now)
    meta_data = Column(JSONB, default={})

    # Индексы
    __table_args__ = (
        Index('idx_faces_photo_id', 'photo_id'),
        Index('idx_faces_age', 'age'),
        Index('idx_faces_gender', 'gender'),
    )


class IndexingLog(Base):
    """Лог операций индексации"""
    __tablename__ = "indexing_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    operation = Column(String(50))  # "add", "update", "delete", "scan"
    status = Column(String(20))  # "success", "failed", "skipped"
    file_path = Column(String(1024), nullable=True)
    error_message = Column(String(1024), nullable=True)
    processing_time = Column(Float, nullable=True)  # секунды
    details = Column(JSONB, default={})
