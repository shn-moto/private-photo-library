"""Работа с базой данных PostgreSQL"""

import logging
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session

from models.data_models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Менеджер для работы с PostgreSQL"""

    def __init__(self, database_url: str):
        """
        Инициализация подключения к БД

        Args:
            database_url: URL подключения (postgresql://user:password@host:port/dbname)
        """
        self.database_url = database_url
        self.engine: Engine = None
        self.SessionLocal = None

        self._init_database()

    def _init_database(self):
        """Инициализировать подключение к БД"""
        try:
            # Создать engine с connection pooling
            self.engine = create_engine(
                self.database_url,
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
            
            # Проверить подключение
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Создать session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Подключение к БД установлено успешно")
        
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            raise
    
    def init_tables(self):
        """Создать все таблицы в БД"""
        try:
            # Создать расширение pgvector если не существует
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            
            # Создать все таблицы
            Base.metadata.create_all(bind=self.engine)
            logger.info("Таблицы БД инициализированы")
        
        except Exception as e:
            logger.error(f"Ошибка инициализации таблиц: {e}")
            raise
    
    def get_session(self) -> Session:
        """Получить сессию для работы с БД"""
        if self.SessionLocal is None:
            raise RuntimeError("БД не инициализирована")
        return self.SessionLocal()
    
    def close(self):
        """Закрыть подключение к БД"""
        if self.engine:
            self.engine.dispose()
            logger.info("Подключение к БД закрыто")
    
    def execute_query(self, query: str):
        """
        Выполнить произвольный SQL запрос
        
        Args:
            query: SQL запрос
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                conn.commit()
                return result
        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            raise
    
    def health_check(self) -> bool:
        """Проверить здоровье подключения к БД"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


class PhotoIndexRepository:
    """Репозиторий для работы с индексом фотографий"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def add_photo(self, session: Session, photo_data: dict) -> str:
        """
        Добавить фотографию в индекс
        
        Args:
            session: SQLAlchemy сессия
            photo_data: Словарь с данными фото
            
        Returns:
            ID добавленной фотографии
        """
        from models.data_models import PhotoIndex
        
        try:
            photo = PhotoIndex(**photo_data)
            session.add(photo)
            session.commit()
            logger.info(f"Фотография добавлена: {photo_data.get('file_path')}")
            return str(photo.id)
        except Exception as e:
            session.rollback()
            logger.error(f"Ошибка добавления фотографии: {e}")
            raise
    
    def update_photo(self, session: Session, image_id: str, photo_data: dict):
        """Обновить данные фотографии"""
        from models.data_models import PhotoIndex
        
        try:
            photo = session.query(PhotoIndex).filter_by(image_id=image_id).first()
            if photo:
                for key, value in photo_data.items():
                    setattr(photo, key, value)
                session.commit()
                logger.info(f"Фотография обновлена: {image_id}")
            else:
                logger.warning(f"Фотография не найдена: {image_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Ошибка обновления фотографии: {e}")
            raise
    
    def get_photo(self, session: Session, image_id: str) -> dict:
        """Получить данные фотографии"""
        from models.data_models import PhotoIndex
        
        try:
            photo = session.query(PhotoIndex).filter_by(image_id=image_id).first()
            if photo:
                return {
                    'id': str(photo.id),
                    'image_id': photo.image_id,
                    'file_path': photo.file_path,
                    'indexed': bool(photo.indexed)
                }
            return None
        except Exception as e:
            logger.error(f"Ошибка получения фотографии: {e}")
            return None
    
    def get_unindexed_photos(self, session: Session, limit: int = 100) -> list:
        """Получить список неиндексированных фотографий"""
        from models.data_models import PhotoIndex
        
        try:
            photos = session.query(PhotoIndex).filter_by(indexed=0).limit(limit).all()
            return [
                {
                    'image_id': photo.image_id,
                    'file_path': photo.file_path,
                    'file_size': photo.file_size
                }
                for photo in photos
            ]
        except Exception as e:
            logger.error(f"Ошибка получения неиндексированных фотографий: {e}")
            return []
    
    def delete_photo(self, session: Session, image_id: str):
        """Удалить фотографию из индекса"""
        from models.data_models import PhotoIndex, FaceRecord
        
        try:
            # Удалить лица для этой фотографии
            session.query(FaceRecord).filter_by(photo_id=image_id).delete()
            
            # Удалить саму фотографию
            session.query(PhotoIndex).filter_by(image_id=image_id).delete()
            session.commit()
            logger.info(f"Фотография удалена: {image_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Ошибка удаления фотографии: {e}")
            raise


class FaceRepository:
    """Репозиторий для работы с записями о лицах"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def add_face(self, session: Session, face_data: dict) -> str:
        """Добавить запись о лице"""
        from models.data_models import FaceRecord
        
        try:
            face = FaceRecord(**face_data)
            session.add(face)
            session.commit()
            return str(face.id)
        except Exception as e:
            session.rollback()
            logger.error(f"Ошибка добавления лица: {e}")
            raise
    
    def get_faces_for_photo(self, session: Session, photo_id: str) -> list:
        """Получить все лица для фотографии"""
        from models.data_models import FaceRecord
        
        try:
            faces = session.query(FaceRecord).filter_by(photo_id=photo_id).all()
            return [
                {
                    'face_id': face.face_id,
                    'x1': face.x1,
                    'y1': face.y1,
                    'x2': face.x2,
                    'y2': face.y2,
                    'confidence': face.confidence,
                    'age': face.age,
                    'gender': face.gender,
                    'emotion': face.emotion,
                    'ethnicity': face.ethnicity
                }
                for face in faces
            ]
        except Exception as e:
            logger.error(f"Ошибка получения лиц для фотографии: {e}")
            return []
