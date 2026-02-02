"""Модели данных для сервиса индексации"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Index, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
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

# Face embedding dimensions
FACE_EMBEDDING_DIM = 512  # InsightFace buffalo_l model


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


# ==================== Face & Person Pydantic Models ====================


class FaceInfo(BaseModel):
    """Информация о лице на фото"""
    face_id: int
    bbox: List[float]  # [x1, y1, x2, y2] в пикселях
    det_score: float
    age: Optional[int] = None
    gender: Optional[int] = None  # 0=female, 1=male
    person_id: Optional[int] = None
    person_name: Optional[str] = None


class PersonCreate(BaseModel):
    """Запрос на создание персоны"""
    name: str
    description: Optional[str] = None
    initial_face_id: Optional[int] = None


class PersonUpdate(BaseModel):
    """Запрос на обновление персоны"""
    name: Optional[str] = None
    description: Optional[str] = None
    cover_face_id: Optional[int] = None


class PersonInfo(BaseModel):
    """Информация о персоне"""
    person_id: int
    name: str
    description: Optional[str] = None
    cover_face_id: Optional[int] = None
    face_count: int = 0
    photo_count: int = 0
    created_at: Optional[datetime] = None


class FaceAssignRequest(BaseModel):
    """Запрос на привязку лица к персоне"""
    person_id: Optional[int] = None  # Существующая персона
    new_person_name: Optional[str] = None  # Или создать новую


class FaceSearchResult(BaseModel):
    """Результат поиска по лицу"""
    image_id: int
    face_id: int
    file_path: str
    similarity: float
    person_id: Optional[int] = None
    person_name: Optional[str] = None


class PersonClipSearchRequest(BaseModel):
    """Запрос комбинированного поиска: персона + CLIP"""
    person_id: int
    query: str
    top_k: int = 20
    translate: bool = True
    model: Optional[str] = None


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

    # Геолокация
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Model-specific embedding колонки
    clip_embedding_vit_b32 = Column(Vector(512), nullable=True)
    clip_embedding_vit_b16 = Column(Vector(512), nullable=True)
    clip_embedding_vit_l14 = Column(Vector(768), nullable=True)
    clip_embedding_siglip = Column(Vector(1152), nullable=True)

    # Метаданные
    exif_data = Column(JSONB, nullable=True)
    
    # Флаг индексации лиц (для оптимизации skip_indexed)
    faces_indexed = Column(Integer, nullable=False, server_default='0')
    # 0 = не индексировалось, 1 = индексировано (есть или нет лиц)

    # Индексы для быстрого поиска
    __table_args__ = (
        Index('idx_photo_index_file_format', 'file_format'),
        Index('idx_photo_index_geo', 'latitude', 'longitude',
              postgresql_where='latitude IS NOT NULL AND longitude IS NOT NULL'),
        Index('idx_photo_index_photo_date', 'photo_date',
              postgresql_where='photo_date IS NOT NULL'),
        Index('idx_photo_index_faces_indexed', 'faces_indexed'),
    )

    # Relationship to faces
    faces = relationship("Face", back_populates="photo", cascade="all, delete-orphan")


class Person(Base):
    """Таблица персон (людей)"""
    __tablename__ = "person"

    person_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    cover_face_id = Column(Integer, ForeignKey("faces.face_id", ondelete="SET NULL"), nullable=True)

    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    faces = relationship("Face", back_populates="person", foreign_keys="Face.person_id")

    __table_args__ = (
        Index('idx_person_name', 'name'),
    )


class Face(Base):
    """Таблица обнаруженных лиц на фотографиях"""
    __tablename__ = "faces"

    face_id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey("photo_index.image_id", ondelete="CASCADE"), nullable=False)
    person_id = Column(Integer, ForeignKey("person.person_id", ondelete="SET NULL"), nullable=True)

    # Bounding box (pixel coordinates)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)

    # Detection confidence
    det_score = Column(Float, nullable=False)

    # Facial landmarks (JSON array)
    landmarks = Column(JSONB, nullable=True)

    # Estimated attributes
    age = Column(Integer, nullable=True)
    gender = Column(Integer, nullable=True)  # 0=female, 1=male

    # Face embedding (InsightFace buffalo_l = 512-dim)
    face_embedding = Column(Vector(FACE_EMBEDDING_DIM), nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    photo = relationship("PhotoIndex", back_populates="faces")
    person = relationship("Person", back_populates="faces", foreign_keys=[person_id])

    __table_args__ = (
        Index('idx_faces_image_id', 'image_id'),
        Index('idx_faces_person_id', 'person_id'),
    )
