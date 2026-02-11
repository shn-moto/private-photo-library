"""REST API для поиска в индексе фотографий"""

import logging
import logging.handlers
import datetime
import json
import gzip
import threading
from collections import OrderedDict
from fastapi import FastAPI, UploadFile, File, Query, Body, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path
from PIL import Image

# Добавить корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from db.database import DatabaseManager, PhotoIndexRepository

# Настройка логирования: консоль (WARNING+) + файл (INFO+)
def setup_logging():
    """Настроить логирование для API: консоль + файл"""
    log_level = getattr(logging, settings.LOG_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

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
    logging.getLogger("httpx").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

# Опциональные импорты
try:
    from services.clip_embedder import CLIPEmbedder
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    logger.warning("CLIP embedder not available")

# Переводчик для русских запросов
try:
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source="auto", target="en")
    HAS_TRANSLATOR = True
    logger.info("Translator initialized (auto -> en)")
except ImportError:
    translator = None
    HAS_TRANSLATOR = False
    logger.warning("deep-translator not available, queries won't be translated")

# Face detection и person management
try:
    from services.face_embedder import FaceEmbedder
    from services.face_indexer import FaceIndexingService, FaceRepository
    from services.person_service import PersonService
    HAS_FACE_DETECTOR = True
except ImportError as e:
    HAS_FACE_DETECTOR = False
    logger.warning(f"Face detection not available: {e}")

# Инициализация FastAPI приложения
app = FastAPI(
    title="Smart Photo Indexing API",
    description="API для поиска фотографий по тексту, изображению и лицам",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== In-memory thumbnail LRU cache ====================

class ThumbnailMemoryCache:
    """Thread-safe in-memory LRU cache for thumbnail bytes.
    Eliminates Docker bind mount I/O for repeated views."""

    def __init__(self, max_bytes: int = 150 * 1024 * 1024):  # 150 MB
        self._cache: OrderedDict = OrderedDict()  # key -> (bytes, len)
        self._current_bytes = 0
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> bytes | None:
        with self._lock:
            item = self._cache.get(key)
            if item is not None:
                self._cache.move_to_end(key)
                self._hits += 1
                return item[0]
            self._misses += 1
            return None

    def put(self, key: str, data: bytes):
        size = len(data)
        with self._lock:
            if key in self._cache:
                self._current_bytes -= self._cache[key][1]
            self._cache[key] = (data, size)
            self._cache.move_to_end(key)
            self._current_bytes += size
            # Evict oldest entries if over limit
            while self._current_bytes > self._max_bytes and self._cache:
                _, (_, evicted_size) = self._cache.popitem(last=False)
                self._current_bytes -= evicted_size

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._cache),
                "size_bytes": self._current_bytes,
                "size_mb": round(self._current_bytes / (1024 * 1024), 1),
                "max_mb": round(self._max_bytes / (1024 * 1024), 1),
                "hits": self._hits,
                "misses": self._misses,
            }


_thumb_mem_cache = ThumbnailMemoryCache(max_bytes=150 * 1024 * 1024)  # ~5000 thumbs

# Глобальные сервисы
db_manager: Optional[DatabaseManager] = None
clip_embedders: dict = {}  # Кэш загруженных моделей {model_name: CLIPEmbedder}
clip_embedder: Optional['CLIPEmbedder'] = None  # Модель по умолчанию

# CLIP Indexing service (используется в _run_reindex)
active_indexing_service: Optional['IndexingService'] = None

# Face detection сервисы
face_embedder: Optional['FaceEmbedder'] = None
face_indexer: Optional['FaceIndexingService'] = None
person_service: Optional['PersonService'] = None

# Album service
album_service: Optional['AlbumService'] = None


@app.on_event("startup")
async def startup():
    """Инициализация при запуске приложения"""
    global db_manager, clip_embedder, clip_embedders
    global face_embedder, face_indexer, person_service
    global album_service

    logger.info("Инициализация API сервера...")

    db_manager = DatabaseManager(settings.DATABASE_URL)

    # Album service (не зависит от CLIP/Face)
    from services.album_service import AlbumService
    album_service = AlbumService(db_manager.get_session)
    logger.info("Album service инициализирован")

    if HAS_CLIP:
        try:
            # Инициализировать модель по умолчанию
            clip_embedder = CLIPEmbedder(settings.CLIP_MODEL, settings.CLIP_DEVICE)
            clip_embedders[settings.CLIP_MODEL] = clip_embedder
            logger.info(f"CLIP embedder инициализирован: {settings.CLIP_MODEL}")
        except Exception as e:
            logger.error(f"Ошибка инициализации CLIP: {e}", exc_info=True)

    # Инициализация face detection сервисов
    if HAS_FACE_DETECTOR:
        try:
            person_service = PersonService(db_manager.get_session)
            logger.info("Person service инициализирован")

            # Прогрев: загрузка InsightFace модели при старте (иначе первый запрос медленный)
            from services.face_embedder import FaceEmbedder
            face_embedder = FaceEmbedder(device=settings.FACE_DEVICE)
            logger.info("Face embedder прогрет при старте")
        except Exception as e:
            logger.error(f"Ошибка инициализации Face сервисов: {e}", exc_info=True)

    # Pre-scan cache stats in background (takes ~30s via bind mount, don't block startup)
    import threading
    threading.Thread(target=_scan_cache_stats_sync, daemon=True).start()

    logger.info("API сервер готов к работе")


@app.on_event("shutdown")
async def shutdown():
    """Очистка при выключении приложения"""
    global db_manager

    logger.info("Завершение работы API сервера...")
    if db_manager:
        db_manager.close()


# ==================== Модели запросов ====================

class TextSearchRequest(BaseModel):
    """Запрос для текстового поиска"""
    query: str = ""
    top_k: int = 10
    similarity_threshold: float = 0.1  # Lowered for single-word queries
    formats: Optional[List[str]] = None  # Фильтр по форматам: ["jpg", "nef", "heic"]
    translate: bool = True  # Автоперевод на английский
    model: Optional[str] = None  # Модель CLIP для поиска (если None - используется модель по умолчанию)
    person_ids: Optional[List[int]] = None  # Фильтр по персонам (AND: все должны быть на фото)


class SearchResult(BaseModel):
    """Результат поиска"""
    image_id: int
    file_path: str
    similarity: float
    file_format: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class TextSearchResponse(BaseModel):
    """Ответ текстового поиска"""
    results: List[SearchResult]
    translated_query: Optional[str] = None  # Показать что было переведено
    model: Optional[str] = None  # Какая модель использовалась для поиска


class DeleteRequest(BaseModel):
    """Запрос на удаление файлов"""
    image_ids: List[str]


class DeleteResponse(BaseModel):
    """Ответ на удаление файлов"""
    deleted: int
    errors: List[str] = []


class MoveToTrashRequest(BaseModel):
    """Запрос на перемещение файлов в корзину по текстовому запросу"""
    query: str
    model: Optional[str] = None  # Модель CLIP для поиска (если None - используется модель по умолчанию)
    top_k: int = 100  # Максимальное количество файлов для перемещения
    similarity_threshold: float = 0.1  # Порог схожести
    translate: bool = True  # Автоперевод на английский


class MoveToTrashResponse(BaseModel):
    """Ответ на перемещение файлов в корзину"""
    moved: int  # Количество перемещенных файлов
    found: int  # Количество найденных файлов
    errors: List[str] = []  # Ошибки при перемещении
    query: str  # Исходный запрос
    translated_query: Optional[str] = None  # Переведенный запрос


class AlbumCreateRequest(BaseModel):
    """Запрос на создание альбома"""
    title: str
    description: Optional[str] = None
    is_public: bool = False


class AlbumUpdateRequest(BaseModel):
    """Запрос на обновление альбома"""
    title: Optional[str] = None
    description: Optional[str] = None
    cover_image_id: Optional[int] = None
    is_public: Optional[bool] = None


class AlbumAddPhotosRequest(BaseModel):
    """Запрос на добавление фото в альбом"""
    image_ids: List[int]


class AlbumRemovePhotosRequest(BaseModel):
    """Запрос на удаление фото из альбома"""
    image_ids: List[int]


# ==================== Endpoints ====================

@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "ok",
        "db_connected": db_manager.health_check() if db_manager else False,
        "clip_available": clip_embedder is not None,
        "face_detector_available": HAS_FACE_DETECTOR,
        "person_service_available": person_service is not None
    }


@app.get("/models")
async def get_available_models():
    """Получить список доступных CLIP моделей (только с данными в БД)"""
    from models.data_models import PhotoIndex, CLIP_MODEL_COLUMNS
    from sqlalchemy import func
    
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")
    
    session = db_manager.get_session()
    try:
        # Проверить какие модели имеют данные в БД
        models_with_data = []
        for model_name, column_name in CLIP_MODEL_COLUMNS.items():
            column = getattr(PhotoIndex, column_name)
            count = session.query(func.count(PhotoIndex.image_id)).filter(column != None).scalar() or 0
            if count > 0:
                models_with_data.append({
                    "name": model_name,
                    "count": count
                })
        
        return {
            "models": [m["name"] for m in models_with_data],
            "default": settings.CLIP_MODEL if any(m["name"] == settings.CLIP_MODEL for m in models_with_data) else (models_with_data[0]["name"] if models_with_data else None),
            "loaded": list(clip_embedders.keys()),
            "details": models_with_data
        }
    finally:
        session.close()


@app.get("/stats")
async def get_stats():
    """Получить статистику индексирования по моделям"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex, CLIP_MODEL_COLUMNS
    from sqlalchemy import func

    session = db_manager.get_session()
    try:
        total_photos = session.query(PhotoIndex).count()
        
        # Статистика по моделям - считаем не-null значения для каждой колонки
        indexed_by_model = {}
        for model_name, column_name in CLIP_MODEL_COLUMNS.items():
            column = getattr(PhotoIndex, column_name)
            count = session.query(func.count(PhotoIndex.image_id)).filter(column != None).scalar() or 0
            indexed_by_model[model_name] = count
                
        # Общее число проиндексированных (хотя бы одной моделью)
        indexed_photos = session.query(PhotoIndex).filter(
            (PhotoIndex.clip_embedding_vit_b32 != None) |
            (PhotoIndex.clip_embedding_vit_b16 != None) |
            (PhotoIndex.clip_embedding_vit_l14 != None) |
            (PhotoIndex.clip_embedding_siglip != None)
        ).count()

        # Faces stats
        from models.data_models import Face
        total_faces = session.query(func.count(Face.face_id)).scalar() or 0

        # pHash stats
        phash_count = session.query(func.count(PhotoIndex.image_id)).filter(
            PhotoIndex.phash != None, PhotoIndex.phash != ''
        ).scalar() or 0

        return {
            "total_photos": total_photos,
            "indexed_photos": indexed_photos,
            "pending_photos": total_photos - indexed_photos,
            "indexed_by_model": indexed_by_model,
            "active_model": clip_embedder.model_name if clip_embedder else None,
            "total_faces": total_faces,
            "phash_count": phash_count,
        }
    finally:
        session.close()


@app.get("/files/unindexed")
async def get_unindexed_files(model: Optional[str] = Query(None, description="CLIP model name")):
    """Get list of file paths that are not indexed for the specified model"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    from models.data_models import PhotoIndex, CLIP_MODEL_COLUMNS

    # Determine which model to check
    model_name = model or (clip_embedder.model_name if clip_embedder else settings.CLIP_MODEL)

    if model_name not in CLIP_MODEL_COLUMNS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    column_name = CLIP_MODEL_COLUMNS[model_name]
    column = getattr(PhotoIndex, column_name)

    session = db_manager.get_session()
    try:
        # Get file paths where embedding is NULL for this model
        unindexed = session.query(PhotoIndex.file_path).filter(column == None).all()
        return {
            "model": model_name,
            "count": len(unindexed),
            "files": [row.file_path for row in unindexed]
        }
    finally:
        session.close()


def get_clip_embedder(model_name: Optional[str] = None) -> 'CLIPEmbedder':
    """
    Получить CLIP embedder для указанной модели.
    Если модель уже загружена - вернуть из кэша, иначе загрузить и закэшировать.
    
    Args:
        model_name: Имя модели (ViT-B/32, ViT-B/16, ViT-L/14, SigLIP) или None для модели по умолчанию
        
    Returns:
        CLIPEmbedder instance
    """
    global clip_embedders, clip_embedder
    
    if not HAS_CLIP:
        raise HTTPException(status_code=503, detail="CLIP не доступен")
    
    # Использовать модель по умолчанию
    if model_name is None:
        if clip_embedder is None:
            raise HTTPException(status_code=503, detail="CLIP embedder не инициализирован")
        return clip_embedder
    
    # Проверить, загружена ли уже эта модель
    if model_name in clip_embedders:
        return clip_embedders[model_name]
    
    # Загрузить новую модель
    try:
        logger.info(f"Загрузка новой модели CLIP: {model_name}")
        new_embedder = CLIPEmbedder(model_name, settings.CLIP_DEVICE)
        clip_embedders[model_name] = new_embedder
        logger.info(f"Модель {model_name} загружена и закэширована")
        return new_embedder
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить модель {model_name}")


def translate_query(query: str) -> str:
    """Перевести запрос на английский если нужно"""
    if not HAS_TRANSLATOR or not translator:
        return query
    try:
        translated = translator.translate(query)
        if translated and translated != query:
            logger.info(f"Translated: '{query}' -> '{translated}'")
            return translated
        return query
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return query


@app.post("/search/text", response_model=TextSearchResponse)
async def search_by_text(request: TextSearchRequest):
    """
    Поиск фотографий по текстовому описанию (CLIP)

    Если query пустой — возвращает фото по фильтрам (formats, person_ids).
    Запросы автоматически переводятся на английский для лучшего качества поиска.

    Пример: {"query": "кошка на диване", "top_k": 10, "model": "SigLIP"}
    """
    try:
        # Получить embedder для указанной модели или использовать модель по умолчанию
        embedder = get_clip_embedder(request.model)

        # Если запрос пустой — поиск только по фильтрам (без CLIP)
        if not request.query.strip():
            results = search_by_filters_only(
                top_k=request.top_k,
                formats=request.formats,
                person_ids=request.person_ids
            )
            return TextSearchResponse(
                results=results,
                translated_query=None,
                model=embedder.model_name
            )

        # Перевести запрос на английский если включено
        translated = None
        if request.translate:
            query = translate_query(request.query)
            if query != request.query:
                translated = query
        else:
            query = request.query

        # Получить эмбиддинг текста
        text_embedding = embedder.embed_text(query)

        if text_embedding is None:
            raise HTTPException(status_code=400, detail="Ошибка обработки текста")

        # Выполнить поиск через pgvector
        results = search_by_clip_embedding(
            embedding=text_embedding.tolist(),
            top_k=request.top_k,
            threshold=request.similarity_threshold,
            model_name=embedder.model_name,
            formats=request.formats,
            person_ids=request.person_ids
        )

        return TextSearchResponse(
            results=results,
            translated_query=translated,
            model=embedder.model_name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка текстового поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", response_model=TextSearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=100),
    similarity_threshold: float = Query(0.1, ge=0, le=1),  # Lowered for better recall
    model: Optional[str] = Query(None, description="CLIP model (ViT-B/32, ViT-B/16, ViT-L/14, SigLIP)")
):
    """
    Поиск похожих фотографий по загруженному изображению
    """
    try:
        # Получить embedder для указанной модели
        embedder = get_clip_embedder(model)
        
        import io
        from PIL import Image
        import numpy as np

        # Прочитать загруженный файл
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_array = np.array(image)

        # Получить эмбиддинг изображения
        image_embedding = embedder.embed_image(image_array)

        if image_embedding is None:
            raise HTTPException(status_code=400, detail="Ошибка обработки изображения")

        # Выполнить поиск
        results = search_by_clip_embedding(
            embedding=image_embedding.tolist(),
            top_k=top_k,
            threshold=similarity_threshold,
            model_name=embedder.model_name
        )

        return TextSearchResponse(results=results, model=embedder.model_name)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка поиска по изображению: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/photo/{image_id}")
async def get_photo_info(image_id: int):
    """Получить информацию о фотографии"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    try:
        from models.data_models import PhotoIndex

        session = db_manager.get_session()

        try:
            photo = session.query(PhotoIndex).filter_by(image_id=image_id).first()

            if not photo:
                raise HTTPException(status_code=404, detail="Фотография не найдена")

            return {
                "image_id": photo.image_id,
                "file_path": photo.file_path,
                "file_name": photo.file_name,
                "file_format": photo.file_format,
                "width": photo.width,
                "height": photo.height,
                "file_size": photo.file_size,
                "photo_date": photo.photo_date,
                "latitude": photo.latitude,
                "longitude": photo.longitude,
                "exif_data": photo.exif_data
            }

        finally:
            session.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения информации о фотографии: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Endpoints для отдачи изображений ====================

def get_photo_path(image_id: str) -> Optional[str]:
    """Получить путь к файлу по image_id"""
    from models.data_models import PhotoIndex
    session = db_manager.get_session()
    try:
        photo = session.query(PhotoIndex).filter_by(image_id=image_id).first()
        return photo.file_path if photo else None
    finally:
        session.close()


# RAW форматы для обработки через rawpy
RAW_EXTENSIONS = {'.nef', '.cr2', '.arw', '.dng', '.raf', '.orf', '.rw2'}


def _apply_raw_orientation_pil(file_path: str, img: 'Image.Image') -> 'Image.Image':
    """
    Apply EXIF orientation to PIL Image from RAW file.

    RAW files processed with rawpy don't have EXIF in the resulting image,
    so we need to read orientation from the original file and apply it manually.

    Args:
        file_path: Path to original RAW file
        img: PIL Image to rotate

    Returns:
        Rotated PIL Image
    """
    try:
        import exifread

        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='Orientation')

        orientation_tag = tags.get('Image Orientation')
        if not orientation_tag:
            return img

        orientation = str(orientation_tag)

        # Apply rotation based on EXIF orientation value
        from PIL import Image

        if 'Rotated 90 CW' in orientation or orientation == '6':
            return img.transpose(Image.Transpose.ROTATE_270)
        elif 'Rotated 180' in orientation or orientation == '3':
            return img.transpose(Image.Transpose.ROTATE_180)
        elif 'Rotated 90 CCW' in orientation or 'Rotated 270 CW' in orientation or orientation == '8':
            return img.transpose(Image.Transpose.ROTATE_90)
        elif orientation == '2':
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation == '4':
            return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif orientation == '5':
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return img.transpose(Image.Transpose.ROTATE_270)
        elif orientation == '7':
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return img.transpose(Image.Transpose.ROTATE_90)

        return img
    except ImportError:
        logger.debug("exifread not available for RAW orientation")
        return img
    except Exception as e:
        logger.debug(f"Failed to apply RAW orientation: {e}")
        return img


def load_image_any_format(file_path: str, fast_mode: bool = False) -> 'Image.Image':
    """
    Загрузить изображение любого формата (включая RAW)

    Args:
        file_path: путь к файлу
        fast_mode: True для быстрой загрузки (embedded JPEG для RAW) - для превью
                   False для полного качества - для просмотра
    """
    from PIL import Image, ImageOps
    import os
    import io

    ext = os.path.splitext(file_path)[1].lower()

    if ext in RAW_EXTENSIONS:
        try:
            import rawpy

            # Для превью: извлекаем встроенный JPEG (очень быстро)
            # Embedded JPEG обычно уже повернут правильно камерой
            if fast_mode:
                try:
                    with rawpy.imread(file_path) as raw:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img = Image.open(io.BytesIO(thumb.data))
                            # Apply EXIF orientation if present in embedded JPEG
                            return ImageOps.exif_transpose(img)
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            # BITMAP doesn't have EXIF, need manual rotation
                            img = Image.fromarray(thumb.data)
                            return _apply_raw_orientation_pil(file_path, img)
                except Exception:
                    # Если нет встроенного превью, используем half_size
                    pass

            # Полная обработка RAW (или fallback для превью)
            # rawpy.postprocess() automatically applies rotation based on raw.sizes.flip
            # DO NOT apply additional EXIF rotation - it would double-rotate!
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8,
                    half_size=fast_mode  # half_size для превью если нет embedded JPEG
                )
            img = Image.fromarray(rgb)
            return img

        except ImportError:
            logger.warning(f"rawpy не установлен, не могу загрузить {file_path}")
            raise
        except Exception as e:
            logger.error(f"Ошибка загрузки RAW {file_path}: {e}")
            raise

    # HEIC/HEIF
    if ext in {'.heic', '.heif'}:
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass

    # Load image and apply EXIF orientation correction
    img = Image.open(file_path)
    # CRITICAL: Apply EXIF transpose to match the orientation that face detection sees
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img)
    return img


@app.get("/image/{image_id}/thumb")
def get_image_thumbnail(
    image_id: str,
    size: int = Query(400, ge=50, le=800, description="Max thumbnail size in pixels")
):
    """Получить миниатюру изображения: memory cache → disk cache → generate"""
    import os, io
    cache_key = f"{image_id}_{size}"
    _cache_headers = {"Cache-Control": "public, max-age=86400"}

    # 1. Memory cache — instant, no I/O at all
    cached_bytes = _thumb_mem_cache.get(cache_key)
    if cached_bytes is not None:
        return Response(
            content=cached_bytes,
            media_type="image/jpeg",
            headers={**_cache_headers, "X-Cache": "MEM"}
        )

    # 2. Disk cache — read file, store in memory for next time
    cache_file = os.path.join(settings.THUMB_CACHE_DIR, f"{cache_key}.jpg")
    if os.path.exists(cache_file):
        try:
            data = open(cache_file, 'rb').read()
            _thumb_mem_cache.put(cache_key, data)
            return Response(
                content=data,
                media_type="image/jpeg",
                headers={**_cache_headers, "X-Cache": "DISK"}
            )
        except OSError:
            pass  # fall through to generate

    # 3. Generate — cache miss, need file_path from DB
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    file_path = get_photo_path(image_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Изображение не найдено")

    try:
        # fast_mode=True для RAW: half_size ускоряет в ~4 раза
        img = load_image_any_format(file_path, fast_mode=True)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)

        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        quality = 85 if size >= 300 else 75

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        data = buffer.getvalue()

        # Save to disk cache
        os.makedirs(settings.THUMB_CACHE_DIR, exist_ok=True)
        try:
            with open(cache_file, 'wb') as f:
                f.write(data)
        except OSError as cache_err:
            logger.warning(f"Failed to cache thumb {cache_file}: {cache_err}")

        # Store in memory cache
        _thumb_mem_cache.put(cache_key, data)

        return Response(
            content=data,
            media_type="image/jpeg",
            headers={**_cache_headers, "X-Cache": "MISS"}
        )

    except Exception as e:
        logger.error(f"Ошибка создания миниатюры {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image/{image_id}/full")
def get_image_full(image_id: str):
    """Получить полное изображение"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    file_path = get_photo_path(image_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Изображение не найдено")

    try:
        import io

        # fast_mode=False для полного качества при просмотре
        img = load_image_any_format(file_path, fast_mode=False)

        # Ограничить размер для веба (макс 2000px)
        max_size = 2000
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Ошибка загрузки изображения {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Удаление файлов ====================

@app.post("/photos/delete", response_model=DeleteResponse)
async def delete_photos(request: DeleteRequest):
    """
    Удалить фотографии (переместить в корзину)

    Файлы перемещаются в системную корзину, а не удаляются безвозвратно.
    Записи из БД удаляются.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex
    import os

    deleted = 0
    errors = []

    session = db_manager.get_session()
    try:
        for image_id in request.image_ids:
            try:
                # Найти запись в БД
                photo = session.query(PhotoIndex).filter_by(image_id=image_id).first()
                if not photo:
                    errors.append(f"{image_id}: не найден в БД")
                    continue

                file_path = photo.file_path

                # Проверить существование файла
                if not os.path.exists(file_path):
                    # Файл уже удалён - просто удалим запись из БД
                    session.delete(photo)
                    session.commit()
                    deleted += 1
                    logger.info(f"Запись удалена (файл не существует): {file_path}")
                    continue

                # Переместить файл в корзину
                try:
                    import shutil
                    trash_dir = settings.TRASH_DIR
                    rel = os.path.relpath(file_path, "/photos")
                    dest = os.path.join(trash_dir, rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.move(file_path, dest)
                    logger.info(f"Файл перемещён в корзину: {dest}")
                except Exception as e:
                    errors.append(f"{image_id}: ошибка удаления файла - {e}")
                    continue

                # Удалить запись о фото
                session.delete(photo)
                session.commit()

                deleted += 1

            except Exception as e:
                session.rollback()
                errors.append(f"{image_id}: {str(e)}")
                logger.error(f"Ошибка удаления {image_id}: {e}")

        return DeleteResponse(deleted=deleted, errors=errors)

    finally:
        session.close()


@app.post("/photos/move-to-trash", response_model=MoveToTrashResponse)
async def move_photos_to_trash(request: MoveToTrashRequest):
    """
    Найти фотографии по текстовому запросу и переместить их в корзину.
    
    Файлы перемещаются в папку trash с сохранением структуры папок.
    Записи из БД НЕ удаляются (согласно требованию).
    
    Пример: {"query": "мусор на улице", "model": "SigLIP", "top_k": 50}
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")
    
    from models.data_models import PhotoIndex
    import os
    import shutil
    
    # КРИТИЧЕСКАЯ ПРОВЕРКА: убедиться что TRASH_DIR существует и замаплен
    # Это предотвращает потерю файлов при неправильной конфигурации
    if not os.path.exists(settings.TRASH_DIR):
        raise HTTPException(
            status_code=500, 
            detail=f"TRASH_DIR не существует или не замаплен: {settings.TRASH_DIR}. "
                   f"Проверьте volume mapping в docker-compose.yml. "
                   f"Файлы НЕ будут перемещены для безопасности!"
        )
    
    if not os.access(settings.TRASH_DIR, os.W_OK):
        raise HTTPException(
            status_code=500,
            detail=f"TRASH_DIR не доступен для записи: {settings.TRASH_DIR}"
        )

    try:
        # Получить embedder для указанной модели
        embedder = get_clip_embedder(request.model)
        
        # Перевести запрос на английский если включено
        translated = None
        if request.translate:
            query = translate_query(request.query)
            if query != request.query:
                translated = query
        else:
            query = request.query

        # Получить эмбиддинг текста
        text_embedding = embedder.embed_text(query)

        if text_embedding is None:
            raise HTTPException(status_code=400, detail="Ошибка обработки текста")

        # Выполнить поиск через pgvector
        results = search_by_clip_embedding(
            embedding=text_embedding.tolist(),
            top_k=request.top_k,
            threshold=request.similarity_threshold,
            model_name=embedder.model_name
        )

        logger.info(f"Найдено {len(results)} файлов по запросу '{request.query}' (модель: {embedder.model_name})")

        # Переместить найденные файлы в корзину
        moved = 0
        errors = []
        moved_files = []  # Список перемещенных файлов для отчета
        
        for result in results:
            try:
                file_path = result.file_path
                
                # Проверить существование файла
                if not os.path.exists(file_path):
                    errors.append(f"{result.image_id}: файл не существует - {file_path}")
                    continue

                # Использовать тот же подход что и в duplicate_finder
                # Относительный путь от /photos (замапленная папка на хосте)
                trash_dir = settings.TRASH_DIR
                rel = os.path.relpath(file_path, "/photos")
                dest_path = os.path.join(trash_dir, rel)
                dest_dir = os.path.dirname(dest_path)
                
                # КРИТИЧЕСКИ ВАЖНО: Проверить что trash_dir существует и доступен для записи
                # Это предотвращает потерю файлов при неправильном маппинге
                if not os.path.exists(trash_dir):
                    raise Exception(f"TRASH_DIR не существует или не замаплен: {trash_dir}. Файлы НЕ будут перемещены для безопасности!")
                
                if not os.access(trash_dir, os.W_OK):
                    raise Exception(f"TRASH_DIR не доступен для записи: {trash_dir}")
                
                # Создать директории если нужно
                os.makedirs(dest_dir, exist_ok=True)
                
                # Переместить файл (это физическая операция!)
                shutil.move(file_path, dest_path)
                moved += 1
                moved_files.append({
                    'source': file_path,
                    'destination': dest_path,
                    'image_id': result.image_id,
                    'similarity': result.similarity
                })
                logger.info(f"Файл перемещен в корзину: {file_path} -> {dest_path}")
                
            except Exception as e:
                errors.append(f"{result.image_id}: ошибка перемещения - {str(e)}")
                logger.error(f"Ошибка перемещения файла {result.image_id}: {e}")

        # Создать отчет о перемещении
        if moved > 0 or errors:
            from datetime import datetime
            report_path = f"/reports/trash_moved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Отчет о перемещении файлов в корзину\n")
                    f.write(f"# Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Запрос: {request.query}\n")
                    if translated:
                        f.write(f"# Переведено: {translated}\n")
                    f.write(f"# Модель: {embedder.model_name}\n")
                    f.write(f"# Порог схожести: {request.similarity_threshold} ({request.similarity_threshold*100:.1f}%)\n")
                    f.write(f"# Top K: {request.top_k}\n")
                    f.write(f"# Найдено: {len(results)}\n")
                    f.write(f"# Перемещено: {moved}\n")
                    f.write(f"# Ошибок: {len(errors)}\n")
                    f.write(f"#\n\n")
                    
                    if moved_files:
                        f.write(f"## Перемещенные файлы ({moved}):\n\n")
                        for item in moved_files:
                            f.write(f"[{item['image_id']}] Similarity: {item['similarity']:.3f}\n")
                            f.write(f"  FROM: {item['source']}\n")
                            f.write(f"  TO:   {item['destination']}\n\n")
                    
                    if errors:
                        f.write(f"\n## Ошибки ({len(errors)}):\n\n")
                        for error in errors:
                            f.write(f"- {error}\n")
                
                logger.info(f"Отчет о перемещении сохранен: {report_path}")
            except Exception as e:
                logger.error(f"Ошибка создания отчета: {e}")

        return MoveToTrashResponse(
            moved=moved,
            found=len(results),
            errors=errors,
            query=request.query,
            translated_query=translated
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при перемещении файлов в корзину: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Вспомогательные функции с pgvector ====================


def search_by_filters_only(top_k: int, formats: Optional[List[str]] = None, person_ids: Optional[List[int]] = None) -> List[SearchResult]:
    """Поиск фото только по фильтрам (без текстового запроса), сортировка по дате"""
    from sqlalchemy import text as sa_text

    session = db_manager.get_session()

    try:
        # Фильтр по форматам
        format_filter = ""
        if formats and len(formats) > 0:
            normalized_formats = [f.lower().lstrip('.') for f in formats]
            formats_str = ','.join(f"'{f}'" for f in normalized_formats)
            format_filter = f"AND file_format IN ({formats_str})"

        # Фильтр по персонам (AND логика)
        person_filter = ""
        if person_ids and len(person_ids) > 0:
            pids = ','.join(str(int(p)) for p in person_ids)
            person_filter = f"""AND image_id IN (
                SELECT image_id FROM faces
                WHERE person_id IN ({pids})
                GROUP BY image_id
                HAVING COUNT(DISTINCT person_id) = {len(person_ids)}
            )"""

        query = sa_text(f"""
            SELECT image_id, file_path, file_format, latitude, longitude
            FROM photo_index
            WHERE 1=1
              {format_filter}
              {person_filter}
            ORDER BY photo_date DESC NULLS LAST, image_id DESC
            LIMIT :top_k
        """)

        result = session.execute(query, {'top_k': top_k})

        results = []
        for row in result:
            results.append(SearchResult(
                image_id=row.image_id,
                file_path=row.file_path,
                similarity=1.0,
                file_format=row.file_format,
                latitude=row.latitude,
                longitude=row.longitude
            ))

        return results

    finally:
        session.close()


def search_by_clip_embedding(embedding: List[float], top_k: int, threshold: float, model_name: str, formats: Optional[List[str]] = None, person_ids: Optional[List[int]] = None) -> List[SearchResult]:
    """Поиск по CLIP эмбиддингу через pgvector для конкретной модели"""
    from models.data_models import PhotoIndex, CLIP_MODEL_COLUMNS
    from sqlalchemy import text

    session = db_manager.get_session()

    try:
        # Получаем имя колонки для модели
        embedding_column = CLIP_MODEL_COLUMNS.get(model_name)
        if not embedding_column:
            raise ValueError(f"Неизвестная модель: {model_name}")

        embedding_str = '[' + ','.join(map(str, embedding)) + ']'

        # Фильтр по форматам
        format_filter = ""
        if formats and len(formats) > 0:
            normalized_formats = [f.lower().lstrip('.') for f in formats]
            formats_str = ','.join(f"'{f}'" for f in normalized_formats)
            format_filter = f"AND file_format IN ({formats_str})"

        # Фильтр по персонам (AND логика: ВСЕ выбранные персоны должны быть на фото)
        person_filter = ""
        if person_ids and len(person_ids) > 0:
            pids = ','.join(str(int(p)) for p in person_ids)
            person_filter = f"""AND image_id IN (
                SELECT image_id FROM faces
                WHERE person_id IN ({pids})
                GROUP BY image_id
                HAVING COUNT(DISTINCT person_id) = {len(person_ids)}
            )"""

        query = text(f"""
            SELECT
                image_id,
                file_path,
                file_format,
                latitude,
                longitude,
                1 - ({embedding_column} <=> '{embedding_str}'::vector) as similarity
            FROM photo_index
            WHERE {embedding_column} IS NOT NULL
              AND 1 - ({embedding_column} <=> '{embedding_str}'::vector) >= :threshold
              {format_filter}
              {person_filter}
            ORDER BY {embedding_column} <=> '{embedding_str}'::vector
            LIMIT :top_k
        """)

        result = session.execute(query, {
            'threshold': threshold,
            'top_k': top_k
        })

        results = []
        for row in result:
            results.append(SearchResult(
                image_id=row.image_id,
                file_path=row.file_path,
                similarity=float(row.similarity),
                file_format=row.file_format,
                latitude=row.latitude,
                longitude=row.longitude
            ))

        return results

    finally:
        session.close()


# ==================== Переиндексация ====================


# Состояние фоновой переиндексации
_reindex_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "total_files": 0,
    "processed_files": 0,
    "successful": 0,
    "failed": 0,
    "skipped": 0,
    "cleaned": 0,
    "current_batch": 0,
    "total_batches": 0,
    "speed_imgs_per_sec": 0.0,
    "eta_seconds": 0,
    "model": None,
    "error": None,
}


def _run_reindex(model_name: Optional[str] = None):
    """Фоновая задача переиндексации
    
    Args:
        model_name: Имя CLIP модели для индексации (если None - используется модель по умолчанию)
    """
    global active_indexing_service, clip_embedder, clip_embedders
    
    import datetime
    import time
    from services.indexing_lock import IndexingLock
    
    # Захватываем блокировку на весь процесс индексации
    indexing_lock = IndexingLock("clip_indexing")
    if not indexing_lock.acquire(timeout=0):
        logger.warning("Индексация уже запущена другим процессом!")
        _reindex_state["error"] = "Индексация уже запущена другим процессом"
        return
    
    try:
        _reindex_state["running"] = True
        _reindex_state["started_at"] = datetime.datetime.now().isoformat()
        _reindex_state["finished_at"] = None
        _reindex_state["model"] = model_name or settings.CLIP_MODEL
        _reindex_state["error"] = None
        _reindex_state["processed_files"] = 0
        _reindex_state["successful"] = 0
        _reindex_state["failed"] = 0
        _reindex_state["skipped"] = 0
        _reindex_state["current_batch"] = 0
        _reindex_state["total_batches"] = 0

        start_time = time.time()
        from services.indexer import IndexingService

        # Получить или создать clip_embedder для нужной модели
        target_model = model_name or settings.CLIP_MODEL
        
        # Использовать уже загруженную модель если она совпадает
        if target_model in clip_embedders:
            embedder_to_use = clip_embedders[target_model]
            logger.info(f"Переиспользую загруженную модель: {target_model}")
        elif clip_embedder and clip_embedder.model_name == target_model:
            embedder_to_use = clip_embedder
            logger.info(f"Переиспользую модель по умолчанию: {target_model}")
        else:
            # Модель не загружена - создаем новую
            embedder_to_use = None
            logger.info(f"Будет создана новая модель: {target_model}")
        
        indexing_service = IndexingService(model_name=model_name, clip_embedder=embedder_to_use)
        active_indexing_service = indexing_service  # Сохраняем ссылку для /reindex/status

        # Cleanup orphaned убран отсюда - он выполняется в fast_reindex.py через /cleanup/orphaned
        # Disk scan через Docker volume слишком медленный

        # Используем быстрое сканирование (NTFS USN Journal на Windows)
        logger.info("Ручная переиндексация: сканирование хранилища (fast scan)...")
        file_list = indexing_service.fast_scan_files(settings.PHOTO_STORAGE_PATH)
        _reindex_state["total_files"] = len(file_list)

        if file_list:
            logger.info(f"Ручная переиндексация: найдено {len(file_list)} файлов, запуск индексации...")

            # Передаем все файлы одним вызовом - index_batch сам разобьет на батчи и обновит прогресс
            results = indexing_service.index_batch(file_list)
            
            # Копируем финальную статистику из live progress (не из results, т.к. там нет skipped)
            live_progress = indexing_service.get_progress()
            _reindex_state["total_files"] = live_progress.get("total_files", len(file_list))
            _reindex_state["processed_files"] = live_progress.get("processed_files", 0)
            _reindex_state["successful"] = live_progress.get("successful", 0)
            _reindex_state["failed"] = live_progress.get("failed", 0)
            _reindex_state["skipped"] = live_progress.get("skipped", 0)
            _reindex_state["speed_imgs_per_sec"] = live_progress.get("speed_imgs_per_sec", 0.0)

        status = indexing_service.get_indexing_status()
        logger.info(f"Ручная переиндексация завершена: {status['indexed']}/{status['total']}")

    except Exception as e:
        logger.error(f"Ошибка переиндексации: {e}", exc_info=True)
        _reindex_state["error"] = str(e)
    finally:
        active_indexing_service = None  # Очищаем ссылку
        _reindex_state["running"] = False
        _reindex_state["finished_at"] = datetime.datetime.now().isoformat()
        _reindex_state["eta_seconds"] = 0
        # Освобождаем блокировку в самом конце
        indexing_lock.release()


@app.post("/reindex")
async def reindex(
    background_tasks: BackgroundTasks,
    model: Optional[str] = Query(None, description="CLIP модель (ViT-B/32, ViT-B/16, ViT-L/14, SigLIP). Если не указана - используется модель по умолчанию")
):
    """
    Запуск переиндексации в фоне.
    Проверяйте прогресс через GET /reindex/status или GET /stats.
    
    Args:
        model: CLIP модель для индексации (опционально)
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    # Проверяем состояние или наличие блокировки
    if _reindex_state["running"]:
        raise HTTPException(status_code=409, detail="Переиндексация уже запущена через API")
    if _cache_warm_state["running"]:
        raise HTTPException(status_code=409, detail="Cannot start indexing while cache warm is running")

    from services.indexing_lock import IndexingLock
    lock_check = IndexingLock("clip_indexing")
    if lock_check.is_locked():
        raise HTTPException(status_code=409, detail="Индексация уже запущена другим процессом")

    background_tasks.add_task(_run_reindex, model)

    return {
        "status": "started",
        "model": model or settings.CLIP_MODEL,
        "message": "Переиндексация запущена в фоне. Проверяйте прогресс: GET /reindex/status или GET /stats"
    }


@app.get("/reindex/status")
async def reindex_status():
    """Статус фоновой переиндексации с детальным прогрессом"""
    result = dict(_reindex_state)
    
    # Если индексация идет, получаем live данные из IndexingService
    if _reindex_state["running"] and active_indexing_service:
        try:
            live_progress = active_indexing_service.get_progress()
            # Обновляем данные из live progress
            result["total_files"] = live_progress.get("total_files", result["total_files"])
            result["processed_files"] = live_progress.get("processed_files", result["processed_files"])
            result["successful"] = live_progress.get("successful", result["successful"])
            result["failed"] = live_progress.get("failed", result["failed"])
            result["skipped"] = live_progress.get("skipped", result["skipped"])
            result["current_batch"] = live_progress.get("current_batch", result["current_batch"])
            result["total_batches"] = live_progress.get("total_batches", result["total_batches"])
            result["speed_imgs_per_sec"] = live_progress.get("speed_imgs_per_sec", result["speed_imgs_per_sec"])
            result["eta_seconds"] = live_progress.get("eta_seconds", result["eta_seconds"])
        except Exception as e:
            logger.debug(f"Не удалось получить live progress: {e}")
    
    # Добавляем процент выполнения
    if result["total_files"] > 0:
        result["percentage"] = round((result["processed_files"] / result["total_files"]) * 100, 1)
    else:
        result["percentage"] = 0
    
    # Форматируем ETA в читаемый вид
    if result["eta_seconds"] > 0:
        eta_mins = result["eta_seconds"] // 60
        eta_secs = result["eta_seconds"] % 60
        result["eta_formatted"] = f"{eta_mins}m {eta_secs}s"
    else:
        result["eta_formatted"] = "N/A"

    if db_manager and clip_embedder:
        from models.data_models import PhotoIndex, CLIP_MODEL_COLUMNS
        from sqlalchemy import func
        session = db_manager.get_session()
        try:
            total = session.query(PhotoIndex).count()
            
            # Проверяем наличие эмбеддинга для текущей модели
            embedding_column_name = clip_embedder.get_embedding_column()
            embedding_column = getattr(PhotoIndex, embedding_column_name)
            indexed = session.query(PhotoIndex).filter(embedding_column != None).count()
            
            result["db_stats"] = {
                "total_in_db": total,
                "indexed": indexed,
                "pending": total - indexed,
                "db_percentage": round(indexed / total * 100, 1) if total > 0 else 0,
            }
        finally:
            session.close()

    return result


@app.post("/reindex/stop")
async def stop_reindex():
    """Остановить CLIP индексацию. Текущий батч завершится, прогресс сохранён."""
    if not _reindex_state["running"]:
        raise HTTPException(status_code=409, detail="CLIP reindex is not running")
    if active_indexing_service:
        active_indexing_service.request_stop()
    return {"status": "stopping", "message": "Will stop after current batch completes"}


# ==================== Scan Checkpoint API (for fast_reindex.py) ====================

class ScanCheckpointRequest(BaseModel):
    """Request to save scan checkpoint"""
    drive_letter: str
    last_usn: int
    files_count: int = 0


@app.get("/scan/checkpoint/{drive_letter}")
async def get_scan_checkpoint(drive_letter: str):
    """Get scan checkpoint for a drive"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    session = db_manager.get_session()
    try:
        from models.data_models import ScanCheckpoint

        checkpoint = session.query(ScanCheckpoint).filter_by(drive_letter=drive_letter).first()
        if checkpoint:
            return {
                "drive_letter": checkpoint.drive_letter,
                "last_usn": checkpoint.last_usn,
                "last_scan_time": checkpoint.last_scan_time.isoformat() if checkpoint.last_scan_time else None,
                "files_count": checkpoint.files_count
            }
        return {"drive_letter": drive_letter, "last_usn": 0, "files_count": 0}
    finally:
        session.close()


@app.post("/scan/checkpoint")
async def save_scan_checkpoint(request: ScanCheckpointRequest):
    """Save scan checkpoint for a drive"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    session = db_manager.get_session()
    try:
        from models.data_models import ScanCheckpoint
        from datetime import datetime

        checkpoint = session.query(ScanCheckpoint).filter_by(drive_letter=request.drive_letter).first()
        if checkpoint:
            checkpoint.last_usn = request.last_usn
            checkpoint.last_scan_time = datetime.now()
            checkpoint.files_count = request.files_count
        else:
            checkpoint = ScanCheckpoint(
                drive_letter=request.drive_letter,
                last_usn=request.last_usn,
                files_count=request.files_count
            )
            session.add(checkpoint)

        session.commit()
        return {"status": "saved", "drive_letter": request.drive_letter, "last_usn": request.last_usn}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.get("/files/index")
async def get_files_index():
    """Get index of all known files (for filename matching in fast_reindex)"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    session = db_manager.get_session()
    try:
        from models.data_models import PhotoIndex

        photos = session.query(PhotoIndex.file_path, PhotoIndex.file_size).all()
        return {
            "files": [{"file_path": p.file_path, "file_size": p.file_size} for p in photos],
            "count": len(photos)
        }
    finally:
        session.close()


@app.post("/cleanup/orphaned")
async def cleanup_orphaned_records(
    file_list: Optional[UploadFile] = File(None, description="Gzipped JSON array of Docker file paths to delete")
):
    """
    Удалить записи из БД для указанных файлов или проверить все файлы.
    
    Если file_list указан - удаляет только эти пути (быстро, используется fast_reindex.py).
    Если file_list = None - проверяет все файлы на диске (медленно).
    
    Args:
        file_list: Gzipped JSON array of Docker paths (/photos/...) - файлы для удаления
    
    Returns:
        {status, checked, missing, deleted}
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    session = db_manager.get_session()
    
    try:
        from models.data_models import PhotoIndex
        
        if file_list is not None:
            # Быстрый режим: удалить только указанные пути (from fast_reindex.py)
            # Read and decompress file list
            try:
                compressed_data = await file_list.read()
                json_data = gzip.decompress(compressed_data)
                file_paths = json.loads(json_data.decode("utf-8"))

                if not isinstance(file_paths, list):
                    raise HTTPException(status_code=400, detail="File list must be a JSON array")

                logger.info(f"Received {len(file_paths)} orphaned paths for deletion (compressed: {len(compressed_data)} bytes)")

            except gzip.BadGzipFile:
                raise HTTPException(status_code=400, detail="Invalid gzip file")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
            
            logger.info(f"Удаление {len(file_paths)} orphaned записей по списку...")
            
            deleted = 0
            # Удаляем батчами по 500
            for i in range(0, len(file_paths), 500):
                batch_paths = file_paths[i:i+500]
                result = session.query(PhotoIndex).filter(PhotoIndex.file_path.in_(batch_paths)).delete(synchronize_session=False)
                deleted += result
                
                if (i + 500) % 5000 == 0:
                    logger.info(f"Удалено {deleted}/{len(file_paths)}")
            
            session.commit()
            logger.info(f"Удалено {deleted} orphaned записей")
            
            return {
                "status": "completed",
                "checked": 0,
                "missing": len(file_paths),
                "deleted": deleted
            }
        else:
            # Медленный режим: проверить все файлы через IndexingService
            from services.indexer import IndexingService
            
            indexing_service = IndexingService(model_name=None, clip_embedder=None)
            
            logger.info("Запуск полной проверки orphaned записей...")
            stats = indexing_service.cleanup_missing_files(check_only=False)
            logger.info(f"Cleanup завершен: проверено {stats['checked']}, удалено {stats['deleted']} orphaned записей")
            
            return {
                "status": "completed",
                "checked": stats.get("checked", 0),
                "missing": stats.get("missing", 0),
                "deleted": stats.get("deleted", 0)
            }
        
    except Exception as e:
        logger.error(f"Ошибка cleanup orphaned: {e}", exc_info=True)
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.post("/reindex/files")
async def reindex_files(
    background_tasks: BackgroundTasks,
    file_list: UploadFile = File(..., description="Gzipped JSON array of file paths"),
    model: Optional[str] = Query(None, description="CLIP model")
):
    """
    Reindex specific files (used by fast_reindex.py script).

    Accepts gzipped JSON array of file paths as multipart upload.
    This allows sending large file lists (100k+) efficiently.

    Example:
        files.json.gz contains: ["/photos/2024/img1.jpg", "/photos/2024/img2.jpg", ...]
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if _reindex_state["running"]:
        raise HTTPException(status_code=409, detail="Reindexing already in progress")

    from services.indexing_lock import IndexingLock
    lock_check = IndexingLock("clip_indexing")
    if lock_check.is_locked():
        raise HTTPException(status_code=409, detail="Indexing locked by another process")

    # Read and decompress file list
    try:
        compressed_data = await file_list.read()
        json_data = gzip.decompress(compressed_data)
        file_paths = json.loads(json_data.decode("utf-8"))

        if not isinstance(file_paths, list):
            raise HTTPException(status_code=400, detail="File list must be a JSON array")

        logger.info(f"Received {len(file_paths)} files for reindexing (compressed: {len(compressed_data)} bytes)")

    except gzip.BadGzipFile:
        raise HTTPException(status_code=400, detail="Invalid gzip file")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    def _run_files_reindex(file_paths: List[str], model_name: Optional[str]):
        global active_indexing_service, clip_embedder, clip_embedders
        import datetime
        import time
        from services.indexing_lock import IndexingLock

        indexing_lock = IndexingLock("clip_indexing")
        if not indexing_lock.acquire(timeout=0):
            _reindex_state["error"] = "Indexing locked"
            return

        try:
            _reindex_state["running"] = True
            _reindex_state["started_at"] = datetime.datetime.now().isoformat()
            _reindex_state["finished_at"] = None
            _reindex_state["model"] = model_name or settings.CLIP_MODEL
            _reindex_state["error"] = None
            _reindex_state["total_files"] = len(file_paths)

            from services.indexer import IndexingService
            
            # Получить или создать clip_embedder для нужной модели
            target_model = model_name or settings.CLIP_MODEL
            
            # Использовать уже загруженную модель если она совпадает
            if target_model in clip_embedders:
                embedder_to_use = clip_embedders[target_model]
                logger.info(f"Переиспользую загруженную модель: {target_model}")
            elif clip_embedder and clip_embedder.model_name == target_model:
                embedder_to_use = clip_embedder
                logger.info(f"Переиспользую модель по умолчанию: {target_model}")
            else:
                # Модель не загружена - создаем новую
                embedder_to_use = None
                logger.info(f"Будет создана новая модель: {target_model}")
            
            indexing_service = IndexingService(model_name=model_name, clip_embedder=embedder_to_use)
            active_indexing_service = indexing_service

            # Не делаем cleanup здесь - он уже выполнен в fast_reindex.py через /cleanup/orphaned
            # Cleanup здесь замедлял бы каждую батч-индексацию
            results = indexing_service.index_batch(file_paths)

            live_progress = indexing_service.get_progress()
            _reindex_state["processed_files"] = live_progress.get("processed_files", 0)
            _reindex_state["successful"] = live_progress.get("successful", 0)
            _reindex_state["failed"] = live_progress.get("failed", 0)
            _reindex_state["skipped"] = live_progress.get("skipped", 0)

        except Exception as e:
            logger.error(f"Files reindex error: {e}", exc_info=True)
            _reindex_state["error"] = str(e)
        finally:
            active_indexing_service = None
            _reindex_state["running"] = False
            _reindex_state["finished_at"] = datetime.datetime.now().isoformat()
            _reindex_state["eta_seconds"] = 0
            indexing_lock.release()

    background_tasks.add_task(_run_files_reindex, file_paths, model)

    return {
        "status": "started",
        "files_count": len(file_paths),
        "model": model or settings.CLIP_MODEL
    }


# ==================== Поиск дубликатов ====================

class DuplicatesRequest(BaseModel):
    """Запрос на поиск дубликатов"""
    threshold: float = 0.98
    limit: int = 50000
    path_filter: Optional[str] = None  # SQL LIKE: '%/2024/%', '%/DCIM/%'


@app.post("/duplicates")
async def find_duplicates_endpoint(request: DuplicatesRequest):
    """
    Найти дубликаты по косинусному сходству CLIP эмбеддингов.
    Возвращает группы дубликатов и сохраняет отчёт в файл.

    path_filter: фильтр по пути (SQL LIKE), например '%/2024/%'
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    try:
        from services.duplicate_finder import DuplicateFinder

        finder = DuplicateFinder(db_manager.get_session, clip_embedder)
        groups = finder.find_groups(
            threshold=request.threshold,
            limit=request.limit,
            path_filter=request.path_filter
        )

        if not groups:
            return {"status": "ok", "groups": [], "total_groups": 0, "total_duplicates": 0}

        # Сохраняем отчёт
        report_path = "/reports/duplicates.txt"
        stats = finder.save_report(groups, report_path, request.threshold)

        return {
            "status": "ok",
            **stats,
            "report_file": report_path,
            "groups": [
                {
                    "files": [
                        {
                            "action": "KEEP" if j == 0 else "DELETE",
                            "image_id": item['image_id'],
                            "path": item['path'],
                            "size_mb": round(item['size'] / 1024 / 1024, 1)
                        }
                        for j, item in enumerate(group)
                    ]
                }
                for group in groups
            ]
        }

    except Exception as e:
        logger.error(f"Ошибка поиска дубликатов: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/duplicates")
async def delete_duplicates_endpoint(
    threshold: float = Query(1.0, ge=0.9, le=1.0),
    path_filter: Optional[str] = Query(None, description="Фильтр по пути, например */2024/* (используйте * вместо %)")
):
    """
    Найти и удалить дубликаты (в корзину).
    Сначала ищет группы по threshold, затем удаляет все кроме первого в каждой группе.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    # Заменяем * на % для SQL LIKE (удобнее в URL)
    if path_filter:
        path_filter = path_filter.replace("*", "%")

    try:
        from services.duplicate_finder import DuplicateFinder

        finder = DuplicateFinder(db_manager.get_session, clip_embedder)

        # Найти дубликаты
        groups = finder.find_groups(threshold=threshold, path_filter=path_filter)
        if not groups:
            return {"status": "ok", "deleted": 0, "errors": [], "message": "Дубликаты не найдены"}

        # Сохранить отчёт перед удалением
        report_path = "/reports/duplicates_deleted.txt"
        stats = finder.save_report(groups, report_path, threshold)

        # Удалить дубликаты (всё кроме KEEP)
        result = finder.delete_from_report(report_path, dry_run=False)

        # Сформировать детали по группам
        details = []
        for group in groups:
            details.append({
                "keep": group[0]["path"],
                "deleted": [item["path"] for item in group[1:]],
            })

        return {
            "status": "ok",
            **stats,
            "deleted": result["deleted"],
            "errors": result.get("errors", []),
            "groups": details,
        }

    except Exception as e:
        logger.error(f"Ошибка удаления дубликатов: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== pHash Duplicate Detection ====================


class PHashDuplicatesRequest(BaseModel):
    """Запрос на поиск дубликатов по perceptual hash"""
    threshold: int = 0  # Hamming distance: 0=exact, <=6=near-duplicate
    limit: int = 50000
    path_filter: Optional[str] = None
    all_types: bool = False  # True = match across formats, False = same format only


_phash_reindex_state = {
    "running": False,
    "stop_requested": False,
    "total": 0,
    "computed": 0,
    "failed": 0,
    "speed": 0.0,
    "eta_seconds": 0,
    "error": None,
}


@app.post("/duplicates/phash")
async def find_phash_duplicates(request: PHashDuplicatesRequest):
    """
    Найти дубликаты по perceptual hash (Hamming distance).

    threshold: 0 = только точные дубликаты, <=6 = near-duplicates.
    Быстрее и точнее CLIP для настоящих дубликатов (копии, ресайзы, перекодировки).
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    try:
        from services.phash_service import PHashService

        service = PHashService(db_manager.get_session)
        groups = service.find_duplicates(
            threshold=request.threshold,
            limit=request.limit,
            path_filter=request.path_filter,
            same_format_only=not request.all_types
        )

        if not groups:
            return {"status": "ok", "groups": [], "total_groups": 0,
                    "total_duplicates": 0, "threshold": request.threshold}

        report_path = "/reports/duplicates_phash.txt"
        stats = service.save_report(groups, report_path, request.threshold)

        return {
            "status": "ok",
            **stats,
            "threshold": request.threshold,
            "report_file": report_path,
            "groups": [
                {
                    "files": [
                        {
                            "action": "KEEP" if j == 0 else "DELETE",
                            "image_id": item['image_id'],
                            "path": item['path'],
                            "size_mb": round(item['size'] / 1024 / 1024, 1)
                        }
                        for j, item in enumerate(group)
                    ]
                }
                for group in groups
            ]
        }

    except Exception as e:
        logger.error(f"pHash duplicate search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/duplicates/phash")
async def delete_phash_duplicates(request: PHashDuplicatesRequest):
    """
    Найти дубликаты по pHash и удалить (переместить в DUPLICATES_DIR).
    Сначала ищет группы, сохраняет отчёт, затем удаляет все DELETE файлы.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    try:
        from services.phash_service import PHashService
        from services.duplicate_finder import DuplicateFinder

        service = PHashService(db_manager.get_session)
        groups = service.find_duplicates(
            threshold=request.threshold,
            limit=request.limit,
            path_filter=request.path_filter,
            same_format_only=not request.all_types
        )

        if not groups:
            return {"status": "ok", "deleted": 0, "errors": [],
                    "message": "Дубликаты не найдены"}

        # Сохранить отчёт перед удалением
        report_path = "/reports/duplicates_phash_deleted.txt"
        stats = service.save_report(groups, report_path, request.threshold)

        # Удалить дубликаты (всё кроме KEEP) через DuplicateFinder
        finder = DuplicateFinder(db_manager.get_session, clip_embedder)
        result = finder.delete_from_report(report_path, dry_run=False)

        details = []
        for group in groups:
            details.append({
                "keep": group[0]["path"],
                "deleted": [item["path"] for item in group[1:]],
            })

        return {
            "status": "ok",
            **stats,
            "threshold": request.threshold,
            "deleted": result["deleted"],
            "errors": result.get("errors", []),
            "report_file": report_path,
            "groups": details,
        }

    except Exception as e:
        logger.error(f"pHash delete duplicates error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/phash/reindex")
async def reindex_phash(
    background_tasks: BackgroundTasks,
    batch_size: int = Query(500, ge=10, le=5000)
):
    """Вычислить pHash для всех фото без хеша. Фоновая задача."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    if _phash_reindex_state["running"]:
        raise HTTPException(status_code=409, detail="pHash reindex already running")
    if _cache_warm_state["running"]:
        raise HTTPException(status_code=409, detail="Cannot start pHash indexing while cache warm is running")

    def _run(bs: int):
        try:
            _phash_reindex_state["running"] = True
            _phash_reindex_state["stop_requested"] = False
            _phash_reindex_state["error"] = None
            _phash_reindex_state["computed"] = 0
            _phash_reindex_state["failed"] = 0

            from services.phash_service import PHashService
            service = PHashService(db_manager.get_session)

            def on_progress(computed, failed, total, speed, eta):
                _phash_reindex_state["total"] = total
                _phash_reindex_state["computed"] = computed
                _phash_reindex_state["failed"] = failed
                _phash_reindex_state["speed"] = round(speed, 1)
                _phash_reindex_state["eta_seconds"] = int(eta)

            result = service.reindex(
                batch_size=bs,
                progress_callback=on_progress,
                stop_flag=lambda: _phash_reindex_state["stop_requested"]
            )
            _phash_reindex_state["total"] = result["total"]
            _phash_reindex_state["computed"] = result["computed"]
            _phash_reindex_state["failed"] = result["failed"]

        except Exception as e:
            logger.error(f"pHash reindex error: {e}", exc_info=True)
            _phash_reindex_state["error"] = str(e)
        finally:
            _phash_reindex_state["running"] = False

    background_tasks.add_task(_run, batch_size)
    return {"status": "started", "message": "GET /phash/reindex/status for progress"}


@app.get("/phash/reindex/status")
async def phash_reindex_status():
    """Статус вычисления pHash — читаем прогресс из БД."""
    if not db_manager:
        return dict(_phash_reindex_state)
    from sqlalchemy import text
    session = db_manager.get_session()
    try:
        row = session.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE phash IS NOT NULL AND phash != '') AS computed,
                COUNT(*) FILTER (WHERE phash IS NULL) AS pending,
                COUNT(*) AS total
            FROM photo_index
        """)).fetchone()
        speed = _phash_reindex_state.get("speed", 0.0)
        eta_sec = _phash_reindex_state.get("eta_seconds", 0)
        if eta_sec > 0:
            eta_mins = eta_sec // 60
            eta_secs = eta_sec % 60
            eta_formatted = f"{eta_mins}m {eta_secs}s"
        else:
            eta_formatted = "N/A"

        return {
            "running": _phash_reindex_state["running"],
            "total": row[2],
            "computed": row[0],
            "pending": row[1],
            "speed_imgs_per_sec": speed,
            "eta_formatted": eta_formatted,
            "error": _phash_reindex_state.get("error"),
        }
    finally:
        session.close()


@app.post("/phash/reindex/stop")
async def stop_phash_reindex():
    """Остановить фоновое вычисление pHash. Прогресс сохранён — можно продолжить позже."""
    if not _phash_reindex_state["running"]:
        raise HTTPException(status_code=409, detail="pHash reindex is not running")
    _phash_reindex_state["stop_requested"] = True
    return {"status": "stopping", "message": "Will stop after current file completes"}


@app.get("/phash/pending")
async def phash_pending(limit: int = Query(5000, ge=1, le=50000)):
    """Получить файлы без pHash (для вычисления на хосте)."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")
    from sqlalchemy import text
    session = db_manager.get_session()
    try:
        rows = session.execute(text("""
            SELECT image_id, file_path FROM photo_index
            WHERE phash IS NULL
            ORDER BY image_id
            LIMIT :limit
        """), {"limit": limit}).fetchall()
        return {"count": len(rows), "files": [{"id": r[0], "path": r[1]} for r in rows]}
    finally:
        session.close()


@app.post("/phash/update")
async def phash_update(data: dict = Body(...)):
    """
    Обновить pHash для фото (массово, от хост-скрипта).
    Body: {"hashes": {"image_id": "phash_hex", ...}}
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")
    from sqlalchemy import text
    hashes = data.get("hashes", {})
    failed_ids = data.get("failed", [])
    if not hashes and not failed_ids:
        return {"updated": 0, "marked_failed": 0}
    session = db_manager.get_session()
    try:
        count = 0
        for image_id, phash_hex in hashes.items():
            session.execute(
                text("UPDATE photo_index SET phash = :phash WHERE image_id = :id"),
                {"phash": phash_hex, "id": int(image_id)}
            )
            count += 1
        # Mark failed files with empty string so they're excluded from /phash/pending
        for image_id in failed_ids:
            session.execute(
                text("UPDATE photo_index SET phash = '' WHERE image_id = :id"),
                {"id": int(image_id)}
            )
        session.commit()
        return {"updated": count, "marked_failed": len(failed_ids)}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


# ==================== Map API ====================


class MapClusterRequest(BaseModel):
    """Запрос на получение кластеров для карты"""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    zoom: int = 5
    date_from: Optional[str] = None  # YYYY-MM-DD
    date_to: Optional[str] = None    # YYYY-MM-DD
    formats: Optional[List[str]] = None  # File formats filter (e.g., ["jpg", "heic"])
    person_ids: Optional[List[int]] = None  # Фильтр по персонам (OR: фото любого из выбранных)


class MapCluster(BaseModel):
    """Кластер фотографий на карте"""
    latitude: float
    longitude: float
    count: int
    # Границы кластера для drill-down
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    preview_ids: List[int] = []  # Up to 4 image_ids for thumbnail preview


class MapPhotoItem(BaseModel):
    """Фотография для карты"""
    image_id: int
    latitude: float
    longitude: float
    photo_date: Optional[str] = None
    file_format: Optional[str] = None


@app.get("/map/stats")
async def get_map_stats():
    """Получить статистику по гео-данным"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex
    from sqlalchemy import func

    session = db_manager.get_session()
    try:
        total = session.query(PhotoIndex).count()
        with_gps = session.query(PhotoIndex).filter(
            PhotoIndex.latitude != None,
            PhotoIndex.longitude != None
        ).count()
        with_date = session.query(PhotoIndex).filter(
            PhotoIndex.photo_date != None
        ).count()

        # Диапазон дат
        date_range = session.query(
            func.min(PhotoIndex.photo_date),
            func.max(PhotoIndex.photo_date)
        ).filter(PhotoIndex.photo_date != None).first()

        # Географические границы
        geo_bounds = session.query(
            func.min(PhotoIndex.latitude),
            func.max(PhotoIndex.latitude),
            func.min(PhotoIndex.longitude),
            func.max(PhotoIndex.longitude)
        ).filter(
            PhotoIndex.latitude != None,
            PhotoIndex.longitude != None
        ).first()

        return {
            "total_photos": total,
            "with_gps": with_gps,
            "with_date": with_date,
            "gps_percentage": round(100 * with_gps / total, 1) if total > 0 else 0,
            "date_range": {
                "min": date_range[0].isoformat() if date_range and date_range[0] else None,
                "max": date_range[1].isoformat() if date_range and date_range[1] else None,
            },
            "geo_bounds": {
                "min_lat": geo_bounds[0] if geo_bounds else None,
                "max_lat": geo_bounds[1] if geo_bounds else None,
                "min_lon": geo_bounds[2] if geo_bounds else None,
                "max_lon": geo_bounds[3] if geo_bounds else None,
            }
        }
    finally:
        session.close()


@app.post("/map/clusters")
async def get_map_clusters(request: MapClusterRequest):
    """
    Получить кластеры фотографий для карты.

    Кластеризация выполняется на стороне сервера на основе zoom level.
    При увеличении масштаба кластеры дробятся на более мелкие.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex
    from sqlalchemy import func, and_, text
    from datetime import datetime

    session = db_manager.get_session()
    try:
        # Размер ячейки сетки зависит от zoom
        # zoom 1-3: крупные регионы (10 градусов)
        # zoom 4-6: страны (2 градуса)
        # zoom 7-9: регионы (0.5 градуса)
        # zoom 10-12: города (0.1 градуса)
        # zoom 13+: точные координаты (0.01 градуса)
        grid_sizes = {
            1: 30, 2: 20, 3: 10,
            4: 5, 5: 2, 6: 1,
            7: 0.5, 8: 0.2, 9: 0.1,
            10: 0.05, 11: 0.02, 12: 0.01,
            13: 0.005, 14: 0.002, 15: 0.001
        }
        grid_size = grid_sizes.get(request.zoom, 0.001 if request.zoom > 15 else 30)

        # Базовый фильтр по bounding box
        filters = [
            PhotoIndex.latitude != None,
            PhotoIndex.longitude != None,
            PhotoIndex.latitude >= request.min_lat,
            PhotoIndex.latitude <= request.max_lat,
            PhotoIndex.longitude >= request.min_lon,
            PhotoIndex.longitude <= request.max_lon
        ]

        # Фильтр по дате
        if request.date_from:
            try:
                date_from = datetime.strptime(request.date_from, "%Y-%m-%d")
                filters.append(PhotoIndex.photo_date >= date_from)
            except ValueError:
                pass

        if request.date_to:
            try:
                date_to = datetime.strptime(request.date_to, "%Y-%m-%d")
                # Включить весь день
                date_to = datetime(date_to.year, date_to.month, date_to.day, 23, 59, 59)
                filters.append(PhotoIndex.photo_date <= date_to)
            except ValueError:
                pass

        # Фильтр по формату файла
        if request.formats:
            format_list = [f.lower() for f in request.formats]
            filters.append(func.lower(PhotoIndex.file_format).in_(format_list))

        # Фильтр по персонам (OR логика: фото любого из выбранных)
        if request.person_ids:
            from models.data_models import Face as FaceModel
            person_photo_subq = session.query(FaceModel.image_id).filter(
                FaceModel.person_id.in_(request.person_ids)
            )
            filters.append(PhotoIndex.image_id.in_(person_photo_subq))

        # Группировка по ячейкам сетки
        # FLOOR(lat / grid_size) * grid_size дает левую границу ячейки
        lat_cell = func.floor(PhotoIndex.latitude / grid_size) * grid_size
        lon_cell = func.floor(PhotoIndex.longitude / grid_size) * grid_size

        query = session.query(
            lat_cell.label('lat_cell'),
            lon_cell.label('lon_cell'),
            func.count(PhotoIndex.image_id).label('count'),
            func.avg(PhotoIndex.latitude).label('avg_lat'),
            func.avg(PhotoIndex.longitude).label('avg_lon')
        ).filter(and_(*filters)).group_by(lat_cell, lon_cell)

        results = query.all()

        # Fetch up to 4 preview image_ids per grid cell in ONE query
        preview_lookup = {}  # (lat_cell, lon_cell) -> [image_id, ...]
        if results:
            # Build dynamic WHERE clause for raw SQL
            where_parts = [
                "latitude IS NOT NULL", "longitude IS NOT NULL",
                "latitude >= :min_lat", "latitude <= :max_lat",
                "longitude >= :min_lon", "longitude <= :max_lon"
            ]
            params = {
                "grid_size": grid_size,
                "min_lat": request.min_lat, "max_lat": request.max_lat,
                "min_lon": request.min_lon, "max_lon": request.max_lon,
            }
            if request.date_from:
                where_parts.append("photo_date >= :date_from")
                params["date_from"] = request.date_from
            if request.date_to:
                where_parts.append("photo_date <= :date_to_end")
                params["date_to_end"] = request.date_to + " 23:59:59"
            if request.formats:
                format_list = [f.lower() for f in request.formats]
                placeholders = ", ".join(f":fmt_{i}" for i in range(len(format_list)))
                where_parts.append(f"LOWER(file_format) IN ({placeholders})")
                for i, fmt in enumerate(format_list):
                    params[f"fmt_{i}"] = fmt
            if request.person_ids:
                pid_placeholders = ", ".join(f":pid_{i}" for i in range(len(request.person_ids)))
                where_parts.append(f"image_id IN (SELECT image_id FROM faces WHERE person_id IN ({pid_placeholders}))")
                for i, pid in enumerate(request.person_ids):
                    params[f"pid_{i}"] = pid

            where_clause = " AND ".join(where_parts)
            preview_sql = text(f"""
                SELECT image_id, grid_lat, grid_lon FROM (
                    SELECT image_id,
                           FLOOR(latitude / :grid_size) * :grid_size as grid_lat,
                           FLOOR(longitude / :grid_size) * :grid_size as grid_lon,
                           ROW_NUMBER() OVER (
                               PARTITION BY FLOOR(latitude / :grid_size), FLOOR(longitude / :grid_size)
                               ORDER BY image_id
                           ) as rn
                    FROM photo_index
                    WHERE {where_clause}
                ) sub
                WHERE rn = 1
            """)
            from collections import defaultdict
            preview_lookup = defaultdict(list)
            for row in session.execute(preview_sql, params):
                key = (float(row.grid_lat), float(row.grid_lon))
                preview_lookup[key].append(row.image_id)

        clusters = []
        for row in results:
            lat_key = float(row.lat_cell)
            lon_key = float(row.lon_cell)
            clusters.append(MapCluster(
                latitude=float(row.avg_lat),
                longitude=float(row.avg_lon),
                count=row.count,
                min_lat=lat_key,
                max_lat=lat_key + grid_size,
                min_lon=lon_key,
                max_lon=lon_key + grid_size,
                preview_ids=preview_lookup.get((lat_key, lon_key), [])
            ))

        return {
            "clusters": clusters,
            "total_clusters": len(clusters),
            "total_photos": sum(c.count for c in clusters),
            "grid_size": grid_size,
            "zoom": request.zoom
        }
    finally:
        session.close()


@app.get("/map/photos")
async def get_map_photos(
    min_lat: float = Query(..., description="Минимальная широта"),
    max_lat: float = Query(..., description="Максимальная широта"),
    min_lon: float = Query(..., description="Минимальная долгота"),
    max_lon: float = Query(..., description="Максимальная долгота"),
    date_from: Optional[str] = Query(None, description="Дата от (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Дата до (YYYY-MM-DD)"),
    formats: Optional[str] = Query(None, description="Форматы файлов через запятую (jpg,heic)"),
    person_ids: Optional[str] = Query(None, description="ID персон через запятую (OR логика)"),
    limit: int = Query(100, ge=1, le=1000, description="Максимальное количество фото"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации")
):
    """
    Получить фотографии в заданном bounding box.
    Используется при клике на кластер для получения списка фото.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex
    from sqlalchemy import and_, func
    from datetime import datetime

    session = db_manager.get_session()
    try:
        # Фильтры
        filters = [
            PhotoIndex.latitude != None,
            PhotoIndex.longitude != None,
            PhotoIndex.latitude >= min_lat,
            PhotoIndex.latitude <= max_lat,
            PhotoIndex.longitude >= min_lon,
            PhotoIndex.longitude <= max_lon
        ]

        # Фильтр по дате
        if date_from:
            try:
                df = datetime.strptime(date_from, "%Y-%m-%d")
                filters.append(PhotoIndex.photo_date >= df)
            except ValueError:
                pass

        if date_to:
            try:
                dt = datetime.strptime(date_to, "%Y-%m-%d")
                dt = datetime(dt.year, dt.month, dt.day, 23, 59, 59)
                filters.append(PhotoIndex.photo_date <= dt)
            except ValueError:
                pass

        # Фильтр по формату файла
        if formats:
            format_list = [f.strip().lower() for f in formats.split(',')]
            filters.append(func.lower(PhotoIndex.file_format).in_(format_list))

        # Фильтр по персонам (OR логика)
        if person_ids:
            from models.data_models import Face as FaceModel
            pid_list = [int(p.strip()) for p in person_ids.split(',') if p.strip()]
            if pid_list:
                person_photo_subq = session.query(FaceModel.image_id).filter(
                    FaceModel.person_id.in_(pid_list)
                )
                filters.append(PhotoIndex.image_id.in_(person_photo_subq))

        # Общее количество
        total_query = session.query(func.count(PhotoIndex.image_id)).filter(and_(*filters))
        total = total_query.scalar()

        # Получить фото с пагинацией
        query = session.query(PhotoIndex).filter(and_(*filters))
        query = query.order_by(PhotoIndex.photo_date.desc().nullslast())
        query = query.offset(offset).limit(limit)

        photos = []
        for photo in query.all():
            photos.append(MapPhotoItem(
                image_id=photo.image_id,
                latitude=photo.latitude,
                longitude=photo.longitude,
                photo_date=photo.photo_date.isoformat() if photo.photo_date else None,
                file_format=photo.file_format
            ))

        return {
            "photos": photos,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
    finally:
        session.close()


@app.post("/map/search")
async def search_in_area(
    min_lat: float = Query(..., description="Минимальная широта"),
    max_lat: float = Query(..., description="Максимальная широта"),
    min_lon: float = Query(..., description="Минимальная долгота"),
    max_lon: float = Query(..., description="Максимальная долгота"),
    request: TextSearchRequest = Body(...)
):
    """
    Текстовый поиск в пределах географической области.
    Комбинирует CLIP поиск с географической фильтрацией.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex, CLIP_MODEL_COLUMNS
    from sqlalchemy import text

    try:
        # Получить embedder
        embedder = get_clip_embedder(request.model)

        # Перевести запрос
        translated = None
        query_text = request.query
        if request.translate:
            query_text = translate_query(request.query)
            if query_text != request.query:
                translated = query_text

        # Получить эмбиддинг
        text_embedding = embedder.embed_text(query_text)
        if text_embedding is None:
            raise HTTPException(status_code=400, detail="Ошибка обработки текста")

        # Построить SQL запрос с geo фильтром
        embedding_column = CLIP_MODEL_COLUMNS.get(embedder.model_name)
        embedding_str = '[' + ','.join(map(str, text_embedding.tolist())) + ']'

        threshold = request.similarity_threshold
        top_k = request.top_k

        session = db_manager.get_session()
        try:
            query = text(f"""
                SELECT
                    image_id,
                    file_path,
                    file_format,
                    latitude,
                    longitude,
                    photo_date,
                    1 - ({embedding_column} <=> '{embedding_str}'::vector) as similarity
                FROM photo_index
                WHERE {embedding_column} IS NOT NULL
                  AND latitude IS NOT NULL
                  AND longitude IS NOT NULL
                  AND latitude >= :min_lat AND latitude <= :max_lat
                  AND longitude >= :min_lon AND longitude <= :max_lon
                  AND 1 - ({embedding_column} <=> '{embedding_str}'::vector) >= :threshold
                ORDER BY {embedding_column} <=> '{embedding_str}'::vector
                LIMIT :top_k
            """)

            result = session.execute(query, {
                'min_lat': min_lat,
                'max_lat': max_lat,
                'min_lon': min_lon,
                'max_lon': max_lon,
                'threshold': threshold,
                'top_k': top_k
            })

            results = []
            for row in result:
                results.append({
                    "image_id": row.image_id,
                    "file_path": row.file_path,
                    "similarity": float(row.similarity),
                    "file_format": row.file_format,
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                    "photo_date": row.photo_date.isoformat() if row.photo_date else None
                })

            return {
                "results": results,
                "translated_query": translated,
                "model": embedder.model_name,
                "geo_bounds": {
                    "min_lat": min_lat,
                    "max_lat": max_lat,
                    "min_lon": min_lon,
                    "max_lon": max_lon
                }
            }
        finally:
            session.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка geo-поиска: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Face Detection & Indexing ====================


def get_face_indexer() -> 'FaceIndexingService':
    """Получить или создать FaceIndexingService (lazy initialization)"""
    global face_indexer, face_embedder

    if not HAS_FACE_DETECTOR:
        raise HTTPException(status_code=503, detail="Face detection не доступен")

    if face_indexer is None:
        # Создаем face_embedder если еще не создан (переиспользуем между запросами)
        if face_embedder is None:
            from services.face_embedder import FaceEmbedder
            face_embedder = FaceEmbedder(device=settings.FACE_DEVICE)
            logger.info(f"Face embedder инициализирован (device={settings.FACE_DEVICE})")
        
        # Передаем face_embedder в FaceIndexingService для переиспользования
        face_indexer = FaceIndexingService(
            db_manager.get_session,
            device=settings.FACE_DEVICE,
            face_embedder=face_embedder
        )
        logger.info("Face indexer инициализирован (переиспользует face_embedder)")

    return face_indexer


# Состояние фоновой индексации лиц
_face_reindex_state = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "total": 0,
    "processed": 0,
    "faces_found": 0,
    "error": None,
}


def _run_face_reindex(skip_indexed: bool = True, batch_size: int = 8):
    """Фоновая задача индексации лиц"""
    from services.indexing_lock import IndexingLock
    global _face_reindex_state

    # Захватываем блокировку на весь процесс индексации лиц
    face_lock = IndexingLock("face_indexing")
    if not face_lock.acquire(timeout=0):
        logger.warning("Индексация лиц уже запущена другим процессом!")
        _face_reindex_state["error"] = "Индексация лиц уже запущена другим процессом"
        return

    try:
        _face_reindex_state["running"] = True
        _face_reindex_state["started_at"] = datetime.datetime.now().isoformat()
        _face_reindex_state["finished_at"] = None
        _face_reindex_state["error"] = None

        indexer = get_face_indexer()
        stats = indexer.reindex_all(skip_indexed=skip_indexed, batch_size=batch_size)

        _face_reindex_state["total"] = stats.get("total", 0)
        _face_reindex_state["processed"] = stats.get("processed", 0)
        _face_reindex_state["faces_found"] = stats.get("total_faces", 0)

        logger.info(f"Face indexing completed: {stats}")

    except Exception as e:
        _face_reindex_state["error"] = str(e)
        logger.error(f"Face indexing failed: {e}", exc_info=True)
    finally:
        _face_reindex_state["running"] = False
        _face_reindex_state["finished_at"] = datetime.datetime.now().isoformat()
        # Освобождаем блокировку в самом конце
        face_lock.release()


@app.post("/faces/reindex")
async def reindex_faces(
    background_tasks: BackgroundTasks,
    skip_indexed: bool = Query(True, description="Пропустить уже проиндексированные фото"),
    batch_size: int = Query(8, ge=1, le=64, description="Количество воркеров для параллельной обработки на GPU")
):
    """
    Запустить индексацию лиц в фоне.
    Проверяйте прогресс через GET /faces/reindex/status.
    """
    if not HAS_FACE_DETECTOR:
        raise HTTPException(status_code=503, detail="Face detection не доступен")

    # Проверяем состояние или наличие блокировки
    if _face_reindex_state["running"]:
        raise HTTPException(status_code=409, detail="Индексация лиц уже запущена через API")
    if _cache_warm_state["running"]:
        raise HTTPException(status_code=409, detail="Cannot start face indexing while cache warm is running")

    from services.indexing_lock import IndexingLock
    lock_check = IndexingLock("face_indexing")
    if lock_check.is_locked():
        raise HTTPException(status_code=409, detail="Индексация лиц уже запущена другим процессом")

    background_tasks.add_task(_run_face_reindex, skip_indexed, batch_size)

    return {
        "status": "started",
        "skip_indexed": skip_indexed,
        "batch_size": batch_size,
        "message": "Индексация лиц запущена. Проверяйте прогресс: GET /faces/reindex/status"
    }


@app.get("/faces/reindex/status")
async def get_face_reindex_status():
    """Статус индексации лиц с детальным прогрессом"""
    result = dict(_face_reindex_state)

    if db_manager and HAS_FACE_DETECTOR:
        try:
            indexer = get_face_indexer()
            # Получаем реальное состояние из indexer (обновляется в процессе)
            live_status = indexer.get_indexing_status()
            
            # Обновляем основные поля из live_status если индексация идет
            if live_status.get("running", False):
                result["total"] = live_status.get("total", 0)
                result["processed"] = live_status.get("processed", 0)
                result["with_faces"] = live_status.get("with_faces", 0)
                result["faces_found"] = live_status.get("faces_found", 0)
                result["failed"] = live_status.get("failed", 0)
                result["current_batch"] = live_status.get("current_batch", 0)
                result["total_batches"] = live_status.get("total_batches", 0)
                result["speed_imgs_per_sec"] = live_status.get("speed_imgs_per_sec", 0.0)
                result["eta_seconds"] = live_status.get("eta_seconds", 0)
                result["eta_formatted"] = live_status.get("eta_formatted", "N/A")
                result["percentage"] = live_status.get("percentage", 0)
            
            result["db_stats"] = {
                "total_faces": live_status.get("total_faces_in_db", 0),
                "unassigned_faces": live_status.get("unassigned_faces", 0)
            }
        except Exception as e:
            result["db_stats_error"] = str(e)

    return result


@app.post("/faces/reindex/stop")
async def stop_face_reindex():
    """Остановить индексацию лиц. Текущий батч завершится, прогресс сохранён."""
    if not _face_reindex_state["running"]:
        raise HTTPException(status_code=409, detail="Face reindex is not running")
    try:
        indexer = get_face_indexer()
        indexer.request_stop()
    except Exception:
        pass
    return {"status": "stopping", "message": "Will stop after current batch completes"}


@app.get("/photo/{image_id}/faces")
async def get_photo_faces(image_id: int):
    """Получить все лица на фотографии"""
    if not HAS_FACE_DETECTOR:
        raise HTTPException(status_code=503, detail="Face detection не доступен")

    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    try:
        from models.data_models import PhotoIndex

        indexer = get_face_indexer()
        faces = indexer.get_faces_for_photo(image_id)

        # Получить размер изображения из БД
        # rawpy.postprocess() уже применяет поворот, и БД хранит повернутые размеры
        # bbox координаты также сохранены относительно повернутого изображения
        session = db_manager.get_session()
        try:
            photo = session.query(PhotoIndex.width, PhotoIndex.height).filter(
                PhotoIndex.image_id == image_id
            ).first()
            original_width = photo.width if photo else None
            original_height = photo.height if photo else None
        finally:
            session.close()

        return {
            "image_id": image_id,
            "faces": faces,
            "count": len(faces),
            "original_width": original_width,
            "original_height": original_height
        }

    except Exception as e:
        logger.error(f"Ошибка получения лиц для фото {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/photo/{image_id}/faces/auto-assign")
async def auto_assign_photo_faces(
    image_id: int,
    threshold: float = Query(0.6, ge=0.3, le=0.95, description="Минимальное сходство для авто-привязки")
):
    """
    Авто-привязка лиц на фотографии к известным персонам.

    Для каждого неназначенного лица ищет похожие лица среди уже привязанных.
    Если найдено совпадение выше порога - автоматически привязывает.
    """
    if not HAS_FACE_DETECTOR:
        raise HTTPException(status_code=503, detail="Face detection не доступен")

    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    try:
        from models.data_models import PhotoIndex

        indexer = get_face_indexer()
        result = indexer.auto_assign_faces_for_photo(image_id, threshold)

        # Get image size from DB (already stores rotated dimensions)
        session = db_manager.get_session()
        try:
            photo = session.query(PhotoIndex.width, PhotoIndex.height).filter(
                PhotoIndex.image_id == image_id
            ).first()
            original_width = photo.width if photo else None
            original_height = photo.height if photo else None
        finally:
            session.close()

        return {
            "image_id": image_id,
            "assigned": result["assigned"],
            "total_faces": result["total_faces"],
            "faces": result["faces"],
            "original_width": original_width,
            "original_height": original_height
        }

    except Exception as e:
        logger.error(f"Ошибка авто-привязки лиц для фото {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faces/{face_id}/thumb")
def get_face_thumbnail(face_id: int):
    """Получить миниатюру лица (обрезка по bbox)"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import Face as FaceModel, PhotoIndex
    import io

    session = db_manager.get_session()
    try:
        face = session.query(FaceModel).filter(FaceModel.face_id == face_id).first()
        if not face:
            raise HTTPException(status_code=404, detail="Лицо не найдено")

        photo = session.query(PhotoIndex.file_path, PhotoIndex.width, PhotoIndex.height).filter(
            PhotoIndex.image_id == face.image_id
        ).first()
        if not photo:
            raise HTTPException(status_code=404, detail="Фото не найдено")

        file_path = photo.file_path

        # Load image and crop face
        # fast_mode may load embedded JPEG (smaller than original for RAW)
        img = load_image_any_format(file_path, fast_mode=True)

        # Scale bbox to loaded image dimensions
        # bbox in DB is relative to original image size (photo.width x photo.height)
        orig_w = photo.width or img.width
        orig_h = photo.height or img.height
        scale_x = img.width / orig_w if orig_w else 1
        scale_y = img.height / orig_h if orig_h else 1

        bx1 = face.bbox_x1 * scale_x
        by1 = face.bbox_y1 * scale_y
        bx2 = face.bbox_x2 * scale_x
        by2 = face.bbox_y2 * scale_y

        # Add padding around face (20%)
        bbox_w = bx2 - bx1
        bbox_h = by2 - by1
        pad_x = bbox_w * 0.2
        pad_y = bbox_h * 0.2

        x1 = max(0, int(bx1 - pad_x))
        y1 = max(0, int(by1 - pad_y))
        x2 = min(img.width, int(bx2 + pad_x))
        y2 = min(img.height, int(by2 + pad_y))

        face_img = img.crop((x1, y1, x2, y2))
        face_img.thumbnail((160, 160), Image.Resampling.LANCZOS)

        if face_img.mode in ('RGBA', 'P'):
            face_img = face_img.convert('RGB')

        buffer = io.BytesIO()
        face_img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)

        return Response(
            content=buffer.read(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка создания миниатюры лица {face_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


# ==================== Face Search ====================


@app.post("/search/face")
async def search_by_face(
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=100),
    similarity_threshold: float = Query(0.5, ge=0, le=1)
):
    """
    Поиск фотографий по загруженному лицу.
    Загрузите фото с лицом - система найдет похожие лица в базе.
    """
    if not HAS_FACE_DETECTOR:
        raise HTTPException(status_code=503, detail="Face detection не доступен")

    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    try:
        import io
        import numpy as np

        # Прочитать загруженный файл
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_array = np.array(image)

        # Получить face embedder
        indexer = get_face_indexer()
        indexer._ensure_embedder()

        # Детектировать лица на загруженном фото
        faces = indexer.face_embedder.detect_faces(image_array)

        if not faces:
            return {
                "results": [],
                "message": "Лицо не обнаружено на загруженном фото"
            }

        # Использовать первое (самое уверенное) лицо
        best_face = max(faces, key=lambda f: f.det_score)

        # Normalize embedding for search
        embedding = best_face.embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Поиск похожих лиц
        results = indexer.search_by_face(
            embedding=embedding.tolist(),
            top_k=top_k,
            threshold=similarity_threshold
        )

        return {
            "results": results,
            "detected_faces": len(faces),
            "search_face": {
                "det_score": best_face.det_score,
                "age": best_face.age,
                "gender": best_face.gender
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка поиска по лицу: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/face/by_id/{face_id}")
async def search_by_face_id(
    face_id: int,
    top_k: int = Query(10, ge=1, le=100),
    similarity_threshold: float = Query(0.5, ge=0, le=1)
):
    """
    Найти фотографии с похожими лицами по ID известного лица.
    """
    if not HAS_FACE_DETECTOR:
        raise HTTPException(status_code=503, detail="Face detection не доступен")

    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    try:
        from models.data_models import Face

        session = db_manager.get_session()
        try:
            # Найти лицо по ID
            face = session.query(Face).filter(Face.face_id == face_id).first()
            if not face:
                raise HTTPException(status_code=404, detail="Лицо не найдено")

            embedding = list(face.face_embedding)

        finally:
            session.close()

        # Поиск похожих лиц
        indexer = get_face_indexer()
        results = indexer.search_by_face(
            embedding=embedding,
            top_k=top_k,
            threshold=similarity_threshold
        )

        # Исключить само искомое лицо из результатов
        results = [r for r in results if r["face_id"] != face_id]

        return {
            "results": results,
            "source_face_id": face_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка поиска по face_id {face_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Person Management ====================


class PersonCreateRequest(BaseModel):
    """Запрос на создание персоны"""
    name: str
    description: Optional[str] = None
    initial_face_id: Optional[int] = None


class PersonUpdateRequest(BaseModel):
    """Запрос на обновление персоны"""
    name: Optional[str] = None
    description: Optional[str] = None
    cover_face_id: Optional[int] = None


class FaceAssignRequest(BaseModel):
    """Запрос на привязку лица к персоне"""
    person_id: Optional[int] = None
    new_person_name: Optional[str] = None


@app.get("/persons")
async def list_persons(
    search: Optional[str] = Query(None, description="Поиск по имени"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Получить список всех персон"""
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    try:
        persons = person_service.list_persons(search=search, limit=limit, offset=offset)
        return {
            "persons": persons,
            "count": len(persons),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Ошибка получения списка персон: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/persons")
async def create_person(request: PersonCreateRequest):
    """Создать новую персону"""
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    try:
        person_id, is_new = person_service.create_person(
            name=request.name,
            description=request.description,
            initial_face_id=request.initial_face_id
        )

        return {
            "person_id": person_id,
            "name": request.name,
            "created_new_person": is_new,
            "message": "Персона создана" if is_new else f"Использована существующая персона '{request.name}'"
        }

    except Exception as e:
        logger.error(f"Ошибка создания персоны: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persons/{person_id}")
async def get_person(person_id: int):
    """Получить информацию о персоне"""
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    try:
        person = person_service.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Персона не найдена")

        return person

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения персоны {person_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/persons/{person_id}")
async def update_person(person_id: int, request: PersonUpdateRequest):
    """Обновить информацию о персоне"""
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    try:
        success = person_service.update_person(
            person_id=person_id,
            name=request.name,
            description=request.description,
            cover_face_id=request.cover_face_id
        )

        if not success:
            raise HTTPException(status_code=404, detail="Персона не найдена")

        return {"status": "updated", "person_id": person_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обновления персоны {person_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/persons/{person_id}")
async def delete_person(person_id: int):
    """Удалить персону (лица становятся неназначенными)"""
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    try:
        success = person_service.delete_person(person_id)
        if not success:
            raise HTTPException(status_code=404, detail="Персона не найдена")

        return {"status": "deleted", "person_id": person_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка удаления персоны {person_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/persons/{person_id}/merge/{target_person_id}")
async def merge_persons(person_id: int, target_person_id: int):
    """Объединить две персоны (перенести все лица в target)"""
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    if person_id == target_person_id:
        raise HTTPException(status_code=400, detail="Нельзя объединить персону с самой собой")

    try:
        success = person_service.merge_persons(person_id, target_person_id)
        if not success:
            raise HTTPException(status_code=404, detail="Одна из персон не найдена")

        return {
            "status": "merged",
            "source_person_id": person_id,
            "target_person_id": target_person_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка объединения персон {person_id} -> {target_person_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persons/{person_id}/photos")
async def get_person_photos(
    person_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Получить все фотографии с этой персоной"""
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    try:
        photos = person_service.get_photos_by_person(person_id, limit=limit, offset=offset)
        total = person_service.get_photo_count_by_person(person_id)

        return {
            "photos": photos,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }

    except Exception as e:
        logger.error(f"Ошибка получения фото персоны {person_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Face-Person Assignment ====================


@app.post("/faces/{face_id}/assign")
async def assign_face_to_person(face_id: int, request: FaceAssignRequest):
    """
    Привязать лицо к персоне.
    Можно указать существующий person_id или создать новую персону через new_person_name.
    """
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    if not request.person_id and not request.new_person_name:
        raise HTTPException(
            status_code=400,
            detail="Укажите person_id или new_person_name"
        )

    try:
        # Создать новую персону если нужно
        if request.new_person_name:
            person_id, is_new = person_service.create_person(
                name=request.new_person_name,
                initial_face_id=face_id
            )
            return {
                "status": "assigned",
                "face_id": face_id,
                "person_id": person_id,
                "person_name": request.new_person_name,
                "created_new_person": is_new,
                "message": "Создана новая персона" if is_new else f"Привязано к существующей персоне '{request.new_person_name}'"
            }

        # Привязать к существующей персоне
        success = person_service.assign_face_to_person(face_id, request.person_id)
        if not success:
            raise HTTPException(status_code=404, detail="Лицо или персона не найдены")

        return {
            "status": "assigned",
            "face_id": face_id,
            "person_id": request.person_id,
            "created_new_person": False
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка привязки лица {face_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/faces/{face_id}/assign")
async def unassign_face(face_id: int):
    """Отвязать лицо от персоны"""
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    try:
        success = person_service.unassign_face(face_id)
        if not success:
            raise HTTPException(status_code=404, detail="Лицо не найдено")

        return {"status": "unassigned", "face_id": face_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка отвязки лица {face_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/persons/{person_id}/auto-assign")
async def auto_assign_faces_to_person(
    person_id: int,
    threshold: float = Query(0.6, ge=0.4, le=0.9, description="Порог сходства для авто-привязки")
):
    """
    Автоматически привязать неназначенные лица к персоне на основе сходства.
    Использует среднее значение эмбеддингов существующих лиц персоны.
    """
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    try:
        result = person_service.auto_assign_faces(person_id, threshold=threshold)

        return {
            "status": "ok",
            "person_id": person_id,
            "threshold": threshold,
            **result
        }

    except Exception as e:
        logger.error(f"Ошибка авто-привязки лиц к персоне {person_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/persons/maintenance/recalculate-covers")
async def recalculate_person_covers():
    """
    Пересчитать cover_face_id для всех персон на основе лучшего det_score.

    Это административный метод для исправления cover_face_id, которые
    могли быть установлены на лица с низким качеством.
    """
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    try:
        result = person_service.recalculate_all_cover_faces()
        return {
            "status": "ok",
            **result
        }

    except Exception as e:
        logger.error(f"Ошибка пересчёта cover_faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Combined Search (Person + CLIP) ====================


class PersonClipSearchRequest(BaseModel):
    """Запрос комбинированного поиска: персона + CLIP"""
    person_id: int
    query: str
    top_k: int = 20
    translate: bool = True
    model: Optional[str] = None


@app.post("/search/person-clip")
async def search_person_with_clip(request: PersonClipSearchRequest):
    """
    Комбинированный поиск: найти фотографии с конкретным человеком,
    соответствующие текстовому описанию.

    Пример: {"person_id": 5, "query": "в горах"} - найти фото Тани в горах
    """
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    if not HAS_CLIP:
        raise HTTPException(status_code=503, detail="CLIP не доступен")

    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    try:
        from models.data_models import PhotoIndex, Face, CLIP_MODEL_COLUMNS
        from sqlalchemy import text

        # Получить embedder
        embedder = get_clip_embedder(request.model)

        # Перевести запрос
        translated = None
        query_text = request.query
        if request.translate:
            query_text = translate_query(request.query)
            if query_text != request.query:
                translated = query_text

        # Получить CLIP эмбиддинг
        text_embedding = embedder.embed_text(query_text)
        if text_embedding is None:
            raise HTTPException(status_code=400, detail="Ошибка обработки текста")

        # Получить все image_ids с этой персоной
        session = db_manager.get_session()
        try:
            # Найти все фото с персоной
            person_image_ids = session.query(Face.image_id).filter(
                Face.person_id == request.person_id
            ).distinct().all()

            if not person_image_ids:
                return {
                    "results": [],
                    "message": "Фотографии с этой персоной не найдены",
                    "person_id": request.person_id,
                    "model": embedder.model_name
                }

            image_ids = [row[0] for row in person_image_ids]

            # Поиск по CLIP среди этих фото
            embedding_column = CLIP_MODEL_COLUMNS.get(embedder.model_name)
            embedding_str = '[' + ','.join(map(str, text_embedding.tolist())) + ']'

            # SQL с фильтром по image_ids
            image_ids_str = ','.join(map(str, image_ids))

            query = text(f"""
                SELECT
                    image_id,
                    file_path,
                    file_format,
                    photo_date,
                    1 - ({embedding_column} <=> '{embedding_str}'::vector) as similarity
                FROM photo_index
                WHERE {embedding_column} IS NOT NULL
                  AND image_id IN ({image_ids_str})
                ORDER BY {embedding_column} <=> '{embedding_str}'::vector
                LIMIT :top_k
            """)

            result = session.execute(query, {'top_k': request.top_k})

            results = []
            for row in result:
                results.append({
                    "image_id": row.image_id,
                    "file_path": row.file_path,
                    "file_format": row.file_format,
                    "photo_date": row.photo_date.isoformat() if row.photo_date else None,
                    "similarity": float(row.similarity)
                })

            # Получить имя персоны
            person = person_service.get_person(request.person_id)
            person_name = person["name"] if person else None

            return {
                "results": results,
                "person_id": request.person_id,
                "person_name": person_name,
                "translated_query": translated,
                "model": embedder.model_name,
                "photos_with_person": len(image_ids)
            }

        finally:
            session.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка комбинированного поиска: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Face Stats ====================


@app.get("/faces/stats")
async def get_face_stats():
    """Получить статистику по лицам и персонам"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import Face, Person, PhotoIndex
    from sqlalchemy import func

    session = db_manager.get_session()
    try:
        total_faces = session.query(func.count(Face.face_id)).scalar() or 0
        assigned_faces = session.query(func.count(Face.face_id)).filter(
            Face.person_id != None
        ).scalar() or 0
        unassigned_faces = total_faces - assigned_faces

        total_persons = session.query(func.count(Person.person_id)).scalar() or 0

        # Фото с лицами
        photos_with_faces = session.query(func.count(func.distinct(Face.image_id))).scalar() or 0
        total_photos = session.query(func.count(PhotoIndex.image_id)).scalar() or 0

        return {
            "total_faces": total_faces,
            "assigned_faces": assigned_faces,
            "unassigned_faces": unassigned_faces,
            "total_persons": total_persons,
            "photos_with_faces": photos_with_faces,
            "total_photos": total_photos,
            "face_detection_percentage": round(100 * photos_with_faces / total_photos, 1) if total_photos > 0 else 0
        }
    finally:
        session.close()


# ==================== Albums API ====================


@app.get("/albums")
async def list_albums(
    user_id: int = Query(1, description="User ID (1=admin)"),
    search: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Список альбомов пользователя"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    albums = album_service.list_albums(
        user_id=user_id, search=search, limit=limit, offset=offset
    )
    return {"albums": albums, "count": len(albums), "limit": limit, "offset": offset}


@app.post("/albums")
async def create_album(
    request: AlbumCreateRequest,
    user_id: int = Query(1, description="User ID (1=admin)")
):
    """Создать новый альбом"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    try:
        album_id = album_service.create_album(
            user_id=user_id,
            title=request.title,
            description=request.description,
            is_public=request.is_public
        )
        return {"album_id": album_id, "title": request.title, "message": "Альбом создан"}
    except Exception as e:
        logger.error(f"Ошибка создания альбома: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/albums/{album_id}")
async def get_album(album_id: int):
    """Получить информацию об альбоме"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    album = album_service.get_album(album_id)
    if not album:
        raise HTTPException(status_code=404, detail="Альбом не найден")
    return album


@app.put("/albums/{album_id}")
async def update_album(album_id: int, request: AlbumUpdateRequest):
    """Обновить альбом"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    success = album_service.update_album(
        album_id=album_id,
        title=request.title,
        description=request.description,
        cover_image_id=request.cover_image_id,
        is_public=request.is_public
    )
    if not success:
        raise HTTPException(status_code=404, detail="Альбом не найден")
    return {"status": "updated", "album_id": album_id}


@app.delete("/albums/{album_id}")
async def delete_album(album_id: int):
    """Удалить альбом"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    success = album_service.delete_album(album_id)
    if not success:
        raise HTTPException(status_code=404, detail="Альбом не найден")
    return {"status": "deleted", "album_id": album_id}


@app.get("/albums/{album_id}/photos")
async def get_album_photos(
    album_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Получить фотографии в альбоме"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    # Check album exists
    album = album_service.get_album(album_id)
    if not album:
        raise HTTPException(status_code=404, detail="Альбом не найден")

    photos, total = album_service.get_album_photos(album_id, limit, offset)
    return {
        "photos": photos,
        "total": total,
        "album_id": album_id,
        "album_title": album["title"],
        "limit": limit,
        "offset": offset
    }


@app.post("/albums/{album_id}/photos")
async def add_photos_to_album(album_id: int, request: AlbumAddPhotosRequest):
    """Добавить фотографии в альбом"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    # Check album exists
    album = album_service.get_album(album_id)
    if not album:
        raise HTTPException(status_code=404, detail="Альбом не найден")

    try:
        result = album_service.add_photos(album_id, request.image_ids)
        return {"album_id": album_id, **result}
    except Exception as e:
        logger.error(f"Ошибка добавления фото в альбом: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/albums/{album_id}/photos")
async def remove_photos_from_album(album_id: int, request: AlbumRemovePhotosRequest):
    """Удалить фотографии из альбома"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    try:
        removed = album_service.remove_photos(album_id, request.image_ids)
        return {"album_id": album_id, "removed": removed}
    except Exception as e:
        logger.error(f"Ошибка удаления фото из альбома: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/albums/{album_id}/cover/{image_id}")
async def set_album_cover(album_id: int, image_id: int):
    """Установить обложку альбома"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    success = album_service.set_cover(album_id, image_id)
    if not success:
        raise HTTPException(status_code=404, detail="Альбом или фото не найдены")
    return {"status": "ok", "album_id": album_id, "cover_image_id": image_id}


@app.get("/photo/{image_id}/albums")
async def get_photo_albums(image_id: int):
    """Получить альбомы, содержащие данное фото"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    albums = album_service.get_photo_albums(image_id)
    return {"albums": albums, "image_id": image_id}


# ==================== Geo Assignment API ====================


class GeoAssignRequest(BaseModel):
    """Запрос на привязку координат к фото"""
    image_ids: Optional[List[int]] = None  # Конкретные ID фото
    folder: Optional[str] = None  # Или папка целиком (все фото без GPS)
    formats: Optional[List[str]] = None  # Фильтр по форматам (при привязке по папке)
    latitude: float
    longitude: float


@app.get("/geo/stats")
async def get_geo_stats():
    """Получить статистику по фото без GPS"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex
    from sqlalchemy import or_

    session = db_manager.get_session()
    try:
        total = session.query(PhotoIndex).count()
        # Фото без GPS: latitude/longitude = NULL или = 0
        without_gps = session.query(PhotoIndex).filter(
            or_(
                PhotoIndex.latitude == None,
                PhotoIndex.longitude == None,
                PhotoIndex.latitude == 0,
                PhotoIndex.longitude == 0
            )
        ).count()
        with_gps = total - without_gps

        return {
            "total_photos": total,
            "with_gps": with_gps,
            "without_gps": without_gps,
            "gps_percentage": round(100 * with_gps / total, 1) if total > 0 else 0
        }
    finally:
        session.close()


@app.get("/geo/folders")
async def get_folders_without_gps(
    formats: Optional[str] = Query(None, description="Фильтр по форматам (через запятую: jpg,heic,nef)")
):
    """
    Получить список папок с фото без GPS.
    Возвращает только конечные папки (без вложенных).
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex
    from sqlalchemy import or_
    import os

    session = db_manager.get_session()
    try:
        # Базовый запрос - фото без GPS (NULL или 0)
        query = session.query(PhotoIndex.file_path, PhotoIndex.file_format).filter(
            or_(
                PhotoIndex.latitude == None,
                PhotoIndex.longitude == None,
                PhotoIndex.latitude == 0,
                PhotoIndex.longitude == 0
            )
        )

        # Фильтр по форматам
        if formats:
            format_list = [f.strip().lower() for f in formats.split(',') if f.strip()]
            if format_list:
                format_conditions = [PhotoIndex.file_format.ilike(f) for f in format_list]
                # Also check for jpeg when jpg is requested
                if 'jpg' in format_list and 'jpeg' not in format_list:
                    format_conditions.append(PhotoIndex.file_format.ilike('jpeg'))
                if 'heic' in format_list and 'heif' not in format_list:
                    format_conditions.append(PhotoIndex.file_format.ilike('heif'))
                query = query.filter(or_(*format_conditions))

        photos = query.all()

        # Собрать уникальные папки и подсчитать файлы
        folder_counts = {}
        for file_path, file_format in photos:
            # Нормализуем путь (Windows/Linux)
            folder = os.path.dirname(file_path).replace("\\", "/")
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

        # Сортировать по пути
        folders = [
            {"path": path, "count": count}
            for path, count in sorted(folder_counts.items())
        ]

        return {
            "folders": folders,
            "total_folders": len(folders),
            "total_photos": sum(f["count"] for f in folders)
        }
    finally:
        session.close()


@app.get("/geo/photos")
async def get_photos_without_gps(
    folder: Optional[str] = Query(None, description="Фильтр по папке"),
    formats: Optional[str] = Query(None, description="Фильтр по форматам (через запятую: jpg,heic,nef)"),
    limit: int = Query(200, ge=1, le=1000, description="Максимальное количество фото"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации")
):
    """
    Получить фото без GPS, опционально фильтруя по папке и формату.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex
    from sqlalchemy import or_

    session = db_manager.get_session()
    try:
        # Базовый фильтр - фото без GPS (NULL или 0)
        query = session.query(PhotoIndex).filter(
            or_(
                PhotoIndex.latitude == None,
                PhotoIndex.longitude == None,
                PhotoIndex.latitude == 0,
                PhotoIndex.longitude == 0
            )
        )

        # Фильтр по папке (путь содержит папку)
        if folder:
            # Нормализуем путь для поиска
            folder_normalized = folder.replace("\\", "/")
            # Ищем файлы в этой папке (не в подпапках)
            # file_path должен начинаться с folder и после folder не должно быть /
            query = query.filter(
                PhotoIndex.file_path.like(f"{folder_normalized}/%")
            ).filter(
                ~PhotoIndex.file_path.like(f"{folder_normalized}/%/%")
            )

        # Фильтр по форматам
        if formats:
            format_list = [f.strip().lower() for f in formats.split(',') if f.strip()]
            if format_list:
                format_conditions = [PhotoIndex.file_format.ilike(f) for f in format_list]
                # Also check for jpeg when jpg is requested
                if 'jpg' in format_list and 'jpeg' not in format_list:
                    format_conditions.append(PhotoIndex.file_format.ilike('jpeg'))
                if 'heic' in format_list and 'heif' not in format_list:
                    format_conditions.append(PhotoIndex.file_format.ilike('heif'))
                query = query.filter(or_(*format_conditions))

        # Общее количество
        total = query.count()

        # Получить фото с пагинацией (сортировка по дате съемки)
        photos = query.order_by(PhotoIndex.photo_date.asc().nullslast()).offset(offset).limit(limit).all()

        results = []
        for photo in photos:
            results.append({
                "image_id": photo.image_id,
                "file_path": photo.file_path,
                "file_name": photo.file_name,
                "file_format": photo.file_format,
                "file_size": photo.file_size,
                "photo_date": photo.photo_date.isoformat() if photo.photo_date else None
            })

        return {
            "photos": results,
            "total": total,
            "limit": limit,
            "offset": offset,
            "folder": folder
        }
    finally:
        session.close()


@app.post("/geo/assign")
async def assign_geo_coordinates(request: GeoAssignRequest):
    """
    Привязать GPS координаты к выбранным фото.
    Можно указать:
    - image_ids: конкретные ID фото
    - folder: путь к папке (обновит все фото без GPS в этой папке)
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    if not request.image_ids and not request.folder:
        raise HTTPException(status_code=400, detail="Укажите image_ids или folder")

    from models.data_models import PhotoIndex
    from sqlalchemy import or_

    session = db_manager.get_session()
    try:
        if request.image_ids:
            # Привязка по конкретным ID
            updated = session.query(PhotoIndex).filter(
                PhotoIndex.image_id.in_(request.image_ids)
            ).update(
                {
                    PhotoIndex.latitude: request.latitude,
                    PhotoIndex.longitude: request.longitude
                },
                synchronize_session=False
            )
        else:
            # Привязка по папке - все фото без GPS
            folder_normalized = request.folder.replace("\\", "/")

            # Базовый фильтр - фото без GPS в указанной папке
            query = session.query(PhotoIndex).filter(
                or_(
                    PhotoIndex.latitude == None,
                    PhotoIndex.longitude == None,
                    PhotoIndex.latitude == 0,
                    PhotoIndex.longitude == 0
                )
            ).filter(
                PhotoIndex.file_path.like(f"{folder_normalized}/%")
            ).filter(
                ~PhotoIndex.file_path.like(f"{folder_normalized}/%/%")
            )

            # Фильтр по форматам
            if request.formats:
                format_list = [f.strip().lower() for f in request.formats if f.strip()]
                if format_list:
                    format_conditions = [PhotoIndex.file_format.ilike(f) for f in format_list]
                    if 'jpg' in format_list and 'jpeg' not in format_list:
                        format_conditions.append(PhotoIndex.file_format.ilike('jpeg'))
                    if 'heic' in format_list and 'heif' not in format_list:
                        format_conditions.append(PhotoIndex.file_format.ilike('heif'))
                    query = query.filter(or_(*format_conditions))

            updated = query.update(
                {
                    PhotoIndex.latitude: request.latitude,
                    PhotoIndex.longitude: request.longitude
                },
                synchronize_session=False
            )

        session.commit()

        logger.info(f"Assigned GPS ({request.latitude}, {request.longitude}) to {updated} photos")

        return {
            "success": True,
            "updated": updated,
            "latitude": request.latitude,
            "longitude": request.longitude
        }
    except Exception as e:
        session.rollback()
        logger.error(f"Error assigning GPS: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


# ==================== Static Files (Web UI) ====================

# Путь к статическим файлам
static_path = Path(__file__).parent / "static"

# ==================== Admin: Index All Queue ====================

_index_all_state = {
    "running": False,
    "stop_requested": False,
    "current_task": None,       # "clip:SigLIP", "faces", "phash", "cache_warm"
    "queue": [],                # remaining tasks
    "completed": [],            # done tasks
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def _run_index_all(models: List[str], include_faces: bool, include_phash: bool,
                    include_cache_warm: bool = False, cache_warm_heavy_only: bool = True):
    """Последовательное выполнение всех задач индексации"""
    global _index_all_state

    # Строим очередь задач
    queue = []
    for model in models:
        queue.append(f"clip:{model}")
    if include_faces:
        queue.append("faces")
    if include_phash:
        queue.append("phash")
    if include_cache_warm:
        queue.append("cache_warm")

    _index_all_state.update({
        "running": True,
        "stop_requested": False,
        "current_task": None,
        "queue": list(queue),
        "completed": [],
        "started_at": datetime.datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
    })

    try:
        for task in list(queue):
            if _index_all_state["stop_requested"]:
                logger.info("Index All stopped by user")
                break

            _index_all_state["current_task"] = task
            if task in _index_all_state["queue"]:
                _index_all_state["queue"].remove(task)

            logger.info(f"Index All: starting task '{task}'")

            try:
                if task.startswith("clip:"):
                    model_name = task.split(":", 1)[1]
                    _run_reindex(model_name)
                elif task == "faces":
                    _run_face_reindex(skip_indexed=True, batch_size=8)
                elif task == "phash":
                    # pHash - вызываем синхронно
                    from services.phash_service import PHashService
                    _phash_reindex_state["running"] = True
                    _phash_reindex_state["stop_requested"] = False
                    _phash_reindex_state["error"] = None
                    _phash_reindex_state["computed"] = 0
                    _phash_reindex_state["failed"] = 0

                    service = PHashService(db_manager.get_session)

                    def on_progress(computed, failed, total, speed):
                        _phash_reindex_state["computed"] = computed
                        _phash_reindex_state["failed"] = failed
                        _phash_reindex_state["total"] = total
                        _phash_reindex_state["speed"] = speed

                    result = service.reindex(
                        batch_size=500,
                        progress_callback=on_progress,
                        stop_flag=lambda: _index_all_state["stop_requested"] or _phash_reindex_state["stop_requested"]
                    )
                    _phash_reindex_state["total"] = result.get("total", 0)
                    _phash_reindex_state["computed"] = result.get("computed", 0)
                    _phash_reindex_state["running"] = False
                elif task == "cache_warm":
                    # Cache warm - вызываем синхронно
                    from models.data_models import PhotoIndex
                    with db_manager.get_session() as session:
                        query = session.query(PhotoIndex.image_id, PhotoIndex.file_path)
                        if cache_warm_heavy_only:
                            query = query.filter(PhotoIndex.file_format.in_(list(HEAVY_FORMATS)))
                        photos = query.order_by(PhotoIndex.image_id).all()
                    if photos:
                        photo_list = [(p.image_id, p.file_path) for p in photos]
                        _cache_warm_state["heavy_only"] = cache_warm_heavy_only
                        _run_cache_warm(photo_list, [200, 400])
                    else:
                        logger.info("Cache warm: no photos to process")
            except Exception as e:
                logger.error(f"Index All: task '{task}' failed: {e}", exc_info=True)
                # Продолжаем со следующей задачей, не прерываем очередь

            _index_all_state["completed"].append(task)

    except Exception as e:
        _index_all_state["error"] = str(e)
        logger.error(f"Index All error: {e}", exc_info=True)
    finally:
        _index_all_state["running"] = False
        _index_all_state["current_task"] = None
        _index_all_state["finished_at"] = datetime.datetime.now().isoformat()


class IndexAllRequest(BaseModel):
    """Запрос на запуск очереди задач"""
    models: List[str] = ["SigLIP"]
    include_faces: bool = True
    include_phash: bool = True
    include_cache_warm: bool = False
    cache_warm_heavy_only: bool = True


@app.post("/admin/index-all")
async def index_all(request: IndexAllRequest, background_tasks: BackgroundTasks):
    """Запустить очередь задач: CLIP модели -> Лица -> pHash -> Кэш"""
    if _index_all_state["running"]:
        raise HTTPException(status_code=409, detail="Queue already running")
    if _reindex_state["running"] or _face_reindex_state["running"] or _phash_reindex_state["running"]:
        raise HTTPException(status_code=409, detail="Another indexing task is already running")
    if _cache_warm_state["running"]:
        raise HTTPException(status_code=409, detail="Cache warm is already running")

    # Валидация моделей
    from models.data_models import CLIP_MODEL_COLUMNS
    for m in request.models:
        if m not in CLIP_MODEL_COLUMNS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {m}. Available: {list(CLIP_MODEL_COLUMNS.keys())}")

    background_tasks.add_task(
        _run_index_all, request.models, request.include_faces,
        request.include_phash, request.include_cache_warm, request.cache_warm_heavy_only
    )

    # Формируем очередь для ответа
    queue = [f"clip:{m}" for m in request.models]
    if request.include_faces:
        queue.append("faces")
    if request.include_phash:
        queue.append("phash")
    if request.include_cache_warm:
        queue.append("cache_warm")

    return {
        "status": "started",
        "queue": queue,
        "message": "Queue started. Check progress: GET /admin/index-all/status"
    }


@app.get("/admin/index-all/status")
async def index_all_status():
    """Статус очереди индексации Index All"""
    result = dict(_index_all_state)

    # Включаем прогресс текущей подзадачи
    current = _index_all_state.get("current_task")
    if current:
        if current.startswith("clip:"):
            sub = dict(_reindex_state)
            if _reindex_state["running"] and active_indexing_service:
                try:
                    live = active_indexing_service.get_progress()
                    sub.update({
                        "total_files": live.get("total_files", sub["total_files"]),
                        "processed_files": live.get("processed_files", sub["processed_files"]),
                        "speed_imgs_per_sec": live.get("speed_imgs_per_sec", sub["speed_imgs_per_sec"]),
                        "eta_seconds": live.get("eta_seconds", sub["eta_seconds"]),
                    })
                except Exception:
                    pass
            if sub.get("total_files", 0) > 0:
                sub["percentage"] = round((sub["processed_files"] / sub["total_files"]) * 100, 1)
            result["sub_progress"] = sub
        elif current == "faces":
            sub = dict(_face_reindex_state)
            if _face_reindex_state["running"]:
                try:
                    indexer = get_face_indexer()
                    live = indexer.get_indexing_status()
                    if live.get("running"):
                        sub.update({k: live[k] for k in live if k in sub})
                        sub["percentage"] = live.get("percentage", 0)
                except Exception:
                    pass
            result["sub_progress"] = sub
        elif current == "phash":
            result["sub_progress"] = dict(_phash_reindex_state)
        elif current == "cache_warm":
            sub = dict(_cache_warm_state)
            if sub["total"] > 0:
                sub["percentage"] = round(sub["processed"] / sub["total"] * 100, 1)
            else:
                sub["percentage"] = 0
            result["sub_progress"] = sub

    return result


@app.post("/admin/index-all/stop")
async def stop_index_all():
    """Остановить очередь Index All. Текущая задача завершится, остальные отменяются."""
    if not _index_all_state["running"]:
        raise HTTPException(status_code=409, detail="Index All is not running")

    _index_all_state["stop_requested"] = True

    # Останавливаем текущую подзадачу
    current = _index_all_state.get("current_task")
    if current:
        if current.startswith("clip:") and active_indexing_service:
            active_indexing_service.request_stop()
        elif current == "faces":
            try:
                indexer = get_face_indexer()
                indexer.request_stop()
            except Exception:
                pass
        elif current == "phash":
            _phash_reindex_state["stop_requested"] = True
        elif current == "cache_warm":
            _cache_warm_state["stop_requested"] = True

    return {"status": "stopping", "message": "Queue will stop after current task completes"}



# ==================== Thumbnail Cache Management ====================

@app.get("/admin/cache/stats")
async def get_cache_stats():
    """Статистика кэша миниатюр (кэшируется в памяти, обновляется раз в 60 сек)."""
    _maybe_refresh_cache_stats()
    total_size = _cache_stats["total_size_bytes"]

    # Human-readable size
    if total_size < 1024:
        size_human = f"{total_size} B"
    elif total_size < 1024 * 1024:
        size_human = f"{total_size / 1024:.1f} KB"
    elif total_size < 1024 * 1024 * 1024:
        size_human = f"{total_size / (1024 * 1024):.1f} MB"
    else:
        size_human = f"{total_size / (1024 * 1024 * 1024):.2f} GB"

    mem = _thumb_mem_cache.stats
    return {
        "file_count": _cache_stats["file_count"],
        "total_size_bytes": total_size,
        "total_size_human": size_human,
        "cache_dir": settings.THUMB_CACHE_DIR,
        "memory_cache": mem
    }


@app.post("/admin/cache/clear")
async def clear_cache():
    """Очистить кэш миниатюр."""
    import os
    cache_dir = settings.THUMB_CACHE_DIR
    if not os.path.exists(cache_dir):
        return {"status": "ok", "deleted": 0}

    deleted = 0
    errors = 0
    try:
        for entry in os.scandir(cache_dir):
            if entry.is_file() and entry.name.endswith('.jpg'):
                try:
                    os.remove(entry.path)
                    deleted += 1
                except OSError:
                    errors += 1
    except OSError as e:
        logger.error(f"Error clearing cache: {e}")

    logger.info(f"Cache cleared: {deleted} files deleted, {errors} errors")
    _cache_stats.update({"file_count": 0, "total_size_bytes": 0, "updated_at": 0})
    _thumb_mem_cache.clear()
    return {"status": "ok", "deleted": deleted, "errors": errors}


# ==================== Cache Warm (pre-generate thumbnails) ====================

HEAVY_FORMATS = {'nef', 'cr2', 'arw', 'dng', 'raf', 'orf', 'rw2', 'heic', 'heif'}

_cache_warm_state = {
    "running": False,
    "stop_requested": False,
    "total": 0,
    "processed": 0,
    "cached": 0,
    "skipped": 0,
    "errors": 0,
    "speed_imgs_per_sec": 0,
    "eta_formatted": "N/A",
    "heavy_only": False,
    "sizes": [],
}

# In-memory cache for cache stats (avoid 30s scandir on 70K+ files via bind mount)
_cache_stats = {"file_count": 0, "total_size_bytes": 0, "updated_at": 0, "scanning": False}


def _scan_cache_stats_sync():
    """Scan cache dir (runs in background thread). Takes ~30s for 70K files via bind mount."""
    import os, time
    _cache_stats["scanning"] = True
    cache_dir = settings.THUMB_CACHE_DIR
    if not os.path.exists(cache_dir):
        _cache_stats.update({"file_count": 0, "total_size_bytes": 0, "updated_at": time.time(), "scanning": False})
        return
    file_count = 0
    total_size = 0
    try:
        for entry in os.scandir(cache_dir):
            if entry.is_file() and entry.name.endswith('.jpg'):
                file_count += 1
                total_size += entry.stat().st_size
    except OSError as e:
        logger.error(f"Error scanning cache dir: {e}")
    _cache_stats.update({"file_count": file_count, "total_size_bytes": total_size,
                         "updated_at": time.time(), "scanning": False})
    logger.info(f"Cache stats scanned: {file_count} files, {total_size / (1024*1024):.1f} MB")


def _maybe_refresh_cache_stats():
    """Trigger background scan if stats are stale (>60s). Never blocks."""
    import time, threading
    if _cache_stats["scanning"]:
        return  # scan already in progress
    if time.time() - _cache_stats["updated_at"] < 60:
        return  # fresh enough
    threading.Thread(target=_scan_cache_stats_sync, daemon=True).start()


def _run_cache_warm(photo_ids_and_paths: list, sizes: list):
    """Background task: генерация кэша миниатюр для списка фото."""
    import os, time

    state = _cache_warm_state
    cache_dir = settings.THUMB_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Pre-filter: scan cache dir once, skip already-cached photos instantly
    existing_files = set()
    try:
        for f in os.listdir(cache_dir):
            if f.endswith('.jpg'):
                existing_files.add(f)
    except OSError:
        pass

    uncached = []
    already_cached = 0
    for image_id, file_path in photo_ids_and_paths:
        if all(f"{image_id}_{size}.jpg" in existing_files for size in sizes):
            already_cached += 1
        else:
            uncached.append((image_id, file_path))

    state.update({
        "running": True,
        "stop_requested": False,
        "total": len(uncached),
        "processed": 0,
        "cached": 0,
        "skipped": already_cached,
        "errors": 0,
        "speed_imgs_per_sec": 0,
        "eta_formatted": "N/A",
        "sizes": sizes,
    })

    logger.info(f"Cache warm: {len(uncached)} to generate, {already_cached} already cached "
                f"(of {len(photo_ids_and_paths)} total)")

    start_time = time.time()

    for image_id, file_path in uncached:
        if state["stop_requested"] or _index_all_state.get("stop_requested", False):
            logger.info(f"Cache warm stopped by user after {state['processed']} files")
            break

        for size in sizes:
            cache_file = os.path.join(cache_dir, f"{image_id}_{size}.jpg")
            if f"{image_id}_{size}.jpg" in existing_files:
                continue  # this size already exists

            try:
                img = load_image_any_format(file_path, fast_mode=True)
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                quality = 85 if size >= 300 else 75
                img.save(cache_file, format='JPEG', quality=quality)
                state["cached"] += 1
            except Exception as e:
                logger.debug(f"Cache warm error {image_id} size={size}: {e}")
                state["errors"] += 1

        state["processed"] += 1

        # Throttle: yield I/O so thumbnail/API requests aren't blocked
        time.sleep(0.02)

        # Update speed/ETA every 10 files
        if state["processed"] % 10 == 0:
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed = state["processed"] / elapsed
                state["speed_imgs_per_sec"] = round(speed, 1)
                remaining = state["total"] - state["processed"]
                if speed > 0:
                    eta_sec = int(remaining / speed)
                    if eta_sec >= 3600:
                        state["eta_formatted"] = f"{eta_sec // 3600}h {(eta_sec % 3600) // 60}m"
                    elif eta_sec >= 60:
                        state["eta_formatted"] = f"{eta_sec // 60}m {eta_sec % 60}s"
                    else:
                        state["eta_formatted"] = f"{eta_sec}s"

    elapsed = time.time() - start_time
    logger.info(f"Cache warm done: {state['cached']} new thumbnails, {already_cached} pre-cached, "
                f"{state['errors']} errors in {elapsed:.1f}s")
    state["running"] = False
    _cache_stats["updated_at"] = 0  # force refresh on next stats request


@app.post("/admin/cache/warm")
async def warm_cache(
    background_tasks: BackgroundTasks,
    heavy_only: bool = Query(False, description="Only heavy formats (RAW, HEIC)"),
    sizes: str = Query("400", description="Comma-separated sizes to cache, e.g. 200,400")
):
    """Прогреть кэш: сгенерировать миниатюры для всех (или тяжёлых) фото."""
    if _cache_warm_state["running"]:
        raise HTTPException(status_code=409, detail="Cache warm already running")
    if _index_all_state["running"] or _reindex_state["running"] or _face_reindex_state["running"] or _phash_reindex_state["running"]:
        raise HTTPException(status_code=409, detail="Cannot start cache warm while indexing is running")

    # Parse sizes
    try:
        size_list = [int(s.strip()) for s in sizes.split(',') if s.strip()]
        size_list = [s for s in size_list if 50 <= s <= 800]
        if not size_list:
            size_list = [400]
    except ValueError:
        size_list = [400]

    # Get photos from DB
    with db_manager.get_session() as session:
        from models.data_models import PhotoIndex
        query = session.query(PhotoIndex.image_id, PhotoIndex.file_path)
        if heavy_only:
            query = query.filter(PhotoIndex.file_format.in_(list(HEAVY_FORMATS)))
        photos = query.order_by(PhotoIndex.image_id).all()

    if not photos:
        return {"status": "ok", "message": "No photos to cache", "total": 0}

    photo_list = [(p.image_id, p.file_path) for p in photos]
    _cache_warm_state["heavy_only"] = heavy_only

    background_tasks.add_task(_run_cache_warm, photo_list, size_list)
    return {
        "status": "started",
        "total": len(photo_list),
        "heavy_only": heavy_only,
        "sizes": size_list
    }


@app.get("/admin/cache/warm/status")
async def cache_warm_status():
    """Статус прогрева кэша."""
    state = _cache_warm_state
    pct = 0
    if state["total"] > 0:
        pct = round(state["processed"] / state["total"] * 100, 1)
    return {
        "running": state["running"],
        "total": state["total"],
        "processed": state["processed"],
        "cached": state["cached"],
        "skipped": state["skipped"],
        "errors": state["errors"],
        "percentage": pct,
        "speed_imgs_per_sec": state["speed_imgs_per_sec"],
        "eta_formatted": state["eta_formatted"],
        "heavy_only": state["heavy_only"],
        "sizes": state["sizes"],
    }


@app.post("/admin/cache/warm/stop")
async def stop_cache_warm():
    """Остановить прогрев кэша."""
    if not _cache_warm_state["running"]:
        raise HTTPException(status_code=409, detail="Cache warm not running")
    _cache_warm_state["stop_requested"] = True
    return {"status": "stopping"}


# Монтируем статику в корень (после API endpoints)
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
    logger.info(f"Static files mounted from {static_path}")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Запуск API сервера на {settings.API_HOST}:{settings.API_PORT}")

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
