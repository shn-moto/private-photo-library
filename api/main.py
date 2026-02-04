"""REST API для поиска в индексе фотографий"""

import logging
import logging.handlers
import datetime
import json
import gzip
from fastapi import FastAPI, UploadFile, File, Query, Body, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
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


@app.on_event("startup")
async def startup():
    """Инициализация при запуске приложения"""
    global db_manager, clip_embedder, clip_embedders
    global face_embedder, face_indexer, person_service

    logger.info("Инициализация API сервера...")

    db_manager = DatabaseManager(settings.DATABASE_URL)

    if HAS_CLIP:
        try:
            # Инициализировать модель по умолчанию
            clip_embedder = CLIPEmbedder(settings.CLIP_MODEL, settings.CLIP_DEVICE)
            clip_embedders[settings.CLIP_MODEL] = clip_embedder
            logger.info(f"CLIP embedder инициализирован: {settings.CLIP_MODEL}")
        except Exception as e:
            logger.error(f"Ошибка инициализации CLIP: {e}", exc_info=True)

    # Инициализация face detection сервисов (lazy - при первом использовании)
    if HAS_FACE_DETECTOR:
        try:
            person_service = PersonService(db_manager.get_session)
            logger.info("Person service инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации Person service: {e}", exc_info=True)

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
    query: str
    top_k: int = 10
    similarity_threshold: float = 0.1  # Lowered for single-word queries
    formats: Optional[List[str]] = None  # Фильтр по форматам: ["jpg", "nef", "heic"]
    translate: bool = True  # Автоперевод на английский
    model: Optional[str] = None  # Модель CLIP для поиска (если None - используется модель по умолчанию)


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

        return {
            "total_photos": total_photos,
            "indexed_photos": indexed_photos,
            "pending_photos": total_photos - indexed_photos,
            "indexed_by_model": indexed_by_model,
            "active_model": clip_embedder.model_name if clip_embedder else None,
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

    Запросы автоматически переводятся на английский для лучшего качества поиска.

    Пример: {"query": "кошка на диване", "top_k": 10, "model": "SigLIP"}
    """
    try:
        # Получить embedder для указанной модели или использовать модель по умолчанию
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
            model_name=embedder.model_name,
            formats=request.formats
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


def load_image_any_format(file_path: str, fast_mode: bool = False) -> 'Image.Image':
    """
    Загрузить изображение любого формата (включая RAW)

    Args:
        file_path: путь к файлу
        fast_mode: True для быстрой загрузки (embedded JPEG для RAW) - для превью
                   False для полного качества - для просмотра
    """
    from PIL import Image
    import os
    import io

    ext = os.path.splitext(file_path)[1].lower()

    if ext in RAW_EXTENSIONS:
        try:
            import rawpy

            # Для превью: извлекаем встроенный JPEG (очень быстро)
            if fast_mode:
                try:
                    with rawpy.imread(file_path) as raw:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            return Image.open(io.BytesIO(thumb.data))
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            return Image.fromarray(thumb.data)
                except Exception:
                    # Если нет встроенного превью, используем half_size
                    pass

            # Полная обработка RAW (или fallback для превью)
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8,
                    half_size=fast_mode  # half_size для превью если нет embedded JPEG
                )
            return Image.fromarray(rgb)

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
async def get_image_thumbnail(image_id: str):
    """Получить миниатюру изображения (400px)"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    file_path = get_photo_path(image_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Изображение не найдено")

    try:
        import io

        # fast_mode=True для RAW: half_size ускоряет в ~4 раза
        img = load_image_any_format(file_path, fast_mode=True)
        img.thumbnail((400, 400), Image.Resampling.LANCZOS)

        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Ошибка создания миниатюры {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image/{image_id}/full")
async def get_image_full(image_id: str):
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

def search_by_clip_embedding(embedding: List[float], top_k: int, threshold: float, model_name: str, formats: Optional[List[str]] = None) -> List[SearchResult]:
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
    global active_indexing_service
    
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

        indexing_service = IndexingService(model_name=model_name)
        active_indexing_service = indexing_service  # Сохраняем ссылку для /reindex/status

        logger.info("Ручная переиндексация: очистка orphaned записей...")
        cleanup = indexing_service.cleanup_missing_files(check_only=False)
        _reindex_state["cleaned"] = cleanup.get("deleted", 0)
        logger.info(f"Очистка завершена: удалено {cleanup.get('deleted', 0)} orphaned записей из БД")

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
        global active_indexing_service
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
            indexing_service = IndexingService(model_name=model_name)
            active_indexing_service = indexing_service

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
    from sqlalchemy import func, and_
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

        clusters = []
        for row in results:
            clusters.append(MapCluster(
                latitude=float(row.avg_lat),
                longitude=float(row.avg_lon),
                count=row.count,
                min_lat=float(row.lat_cell),
                max_lat=float(row.lat_cell + grid_size),
                min_lon=float(row.lon_cell),
                max_lon=float(row.lon_cell + grid_size)
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
        face_indexer = FaceIndexingService(
            db_manager.get_session,
            device=settings.FACE_DEVICE
        )
        logger.info("Face indexer инициализирован")

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

        # Получить размер оригинального изображения из БД
        # (bbox координаты сохранены относительно этого размера)
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

        # Get original image size for bbox scaling
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
        person_id = person_service.create_person(
            name=request.name,
            description=request.description,
            initial_face_id=request.initial_face_id
        )

        return {
            "person_id": person_id,
            "name": request.name,
            "message": "Персона создана"
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
            person_id = person_service.create_person(
                name=request.new_person_name,
                initial_face_id=face_id
            )
            return {
                "status": "assigned",
                "face_id": face_id,
                "person_id": person_id,
                "person_name": request.new_person_name,
                "created_new_person": True
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


# ==================== Static Files (Web UI) ====================

# Путь к статическим файлам
static_path = Path(__file__).parent / "static"

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
