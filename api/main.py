"""REST API для поиска в индексе фотографий"""

import logging
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, BackgroundTasks
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

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# Face detector отключен - будет реализован позже в отдельном воркере
HAS_FACE_DETECTOR = False

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
clip_embedder: Optional['CLIPEmbedder'] = None
face_detector = None  # отключен, будет реализован позже


@app.on_event("startup")
async def startup():
    """Инициализация при запуске приложения"""
    global db_manager, clip_embedder

    logger.info("Инициализация API сервера...")

    db_manager = DatabaseManager(settings.DATABASE_URL)

    if HAS_CLIP:
        try:
            clip_embedder = CLIPEmbedder(settings.CLIP_MODEL, settings.CLIP_DEVICE)
            logger.info("CLIP embedder инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации CLIP: {e}", exc_info=True)

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


class SearchResult(BaseModel):
    """Результат поиска"""
    image_id: int
    file_path: str
    similarity: float
    file_format: Optional[str] = None


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


# ==================== Endpoints ====================

@app.get("/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "ok",
        "db_connected": db_manager.health_check() if db_manager else False,
        "clip_available": clip_embedder is not None,
        "face_detector_available": False  # будет реализован позже
    }


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

    Пример: {"query": "кошка на диване", "top_k": 10}
    """
    if not clip_embedder:
        raise HTTPException(status_code=503, detail="CLIP embedder не доступен")

    try:
        # Перевести запрос на английский если включено
        translated = None
        if request.translate:
            query = translate_query(request.query)
            if query != request.query:
                translated = query
        else:
            query = request.query

        # Получить эмбиддинг текста
        text_embedding = clip_embedder.embed_text(query)

        if text_embedding is None:
            raise HTTPException(status_code=400, detail="Ошибка обработки текста")

        # Выполнить поиск через pgvector
        results = search_by_clip_embedding(
            embedding=text_embedding.tolist(),
            top_k=request.top_k,
            threshold=request.similarity_threshold,
            model_name=clip_embedder.model_name,
            formats=request.formats
        )

        return TextSearchResponse(
            results=results, 
            translated_query=translated,
            model=clip_embedder.model_name
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
    similarity_threshold: float = Query(0.1, ge=0, le=1)  # Lowered for better recall
):
    """
    Поиск похожих фотографий по загруженному изображению
    """
    if not clip_embedder:
        raise HTTPException(status_code=503, detail="CLIP embedder не доступен")

    try:
        import io
        from PIL import Image
        import numpy as np

        # Прочитать загруженный файл
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_array = np.array(image)

        # Получить эмбиддинг изображения
        image_embedding = clip_embedder.embed_image(image_array)

        if image_embedding is None:
            raise HTTPException(status_code=400, detail="Ошибка обработки изображения")

        # Выполнить поиск
        results = search_by_clip_embedding(
            embedding=image_embedding.tolist(),
            top_k=top_k,
            threshold=similarity_threshold,
            model_name=clip_embedder.model_name
        )

        return TextSearchResponse(results=results, model=clip_embedder.model_name)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка поиска по изображению: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/photo/{image_id}")
async def get_photo_info(image_id: str):
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
                "indexed_at": photo.indexed_at,
                "photo_date": photo.photo_date,
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

    return Image.open(file_path)


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
                file_format=row.file_format
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
    "cleaned": 0,
    "error": None,
}


def _run_reindex():
    """Фоновая задача переиндексации"""
    import datetime
    _reindex_state["running"] = True
    _reindex_state["started_at"] = datetime.datetime.now().isoformat()
    _reindex_state["finished_at"] = None
    _reindex_state["error"] = None

    try:
        from services.indexer import IndexingService
        from services.file_monitor import FileMonitor

        indexing_service = IndexingService()
        file_monitor = FileMonitor(
            settings.PHOTO_STORAGE_PATH,
            settings.SUPPORTED_FORMATS
        )

        cleanup = indexing_service.cleanup_missing_files(check_only=False)
        _reindex_state["cleaned"] = cleanup.get("deleted", 0)

        logger.info("Ручная переиндексация: сканирование хранилища...")
        files = file_monitor.scan_directory()
        _reindex_state["total_files"] = len(files)

        if files:
            logger.info(f"Ручная переиндексация: найдено {len(files)} файлов, запуск индексации...")
            indexing_service.index_batch(list(files.keys()))

        status = indexing_service.get_indexing_status()
        logger.info(f"Ручная переиндексация завершена: {status['indexed']}/{status['total']}")

    except Exception as e:
        logger.error(f"Ошибка переиндексации: {e}", exc_info=True)
        _reindex_state["error"] = str(e)
    finally:
        import datetime
        _reindex_state["running"] = False
        _reindex_state["finished_at"] = datetime.datetime.now().isoformat()


@app.post("/reindex")
async def reindex(background_tasks: BackgroundTasks):
    """
    Запуск переиндексации в фоне.
    Проверяйте прогресс через GET /reindex/status или GET /stats.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    if _reindex_state["running"]:
        raise HTTPException(status_code=409, detail="Переиндексация уже запущена")

    background_tasks.add_task(_run_reindex)

    return {
        "status": "started",
        "message": "Переиндексация запущена в фоне. Проверяйте прогресс: GET /reindex/status или GET /stats"
    }


@app.get("/reindex/status")
async def reindex_status():
    """Статус фоновой переиндексации"""
    result = dict(_reindex_state)

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
            
            result["progress"] = {
                "total_in_db": total,
                "indexed": indexed,
                "pending": total - indexed,
                "percentage": round(indexed / total * 100, 1) if total > 0 else 0,
            }
        finally:
            session.close()

    return result


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
