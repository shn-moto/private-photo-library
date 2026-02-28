"""REST API для поиска в индексе фотографий"""

import asyncio
import logging
import logging.handlers
import datetime
import json
import gzip
import secrets
import threading
import time
from collections import OrderedDict
from fastapi import FastAPI, Request, UploadFile, File, Query, Body, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import sys
from pathlib import Path
from PIL import Image
import httpx

# Добавить корневую папку в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
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


# ==================== Auth middleware ====================

SESSION_TIMEOUT_MINUTES = settings.SESSION_TIMEOUT_MINUTES

# Пути, публично доступные без токена (auth bootstrap)
_AUTH_PUBLIC_PATHS = {"/health", "/auth/session", "/auth/check", "/auth/logout", "/auth/me"}

# Пути, заблокированные через Cloudflare tunnel (403 даже для авторизованных)
_TUNNEL_BLOCKED_PATHS = {"/admin.html", "/geo_assign.html", "/duplicates.html"}
_TUNNEL_BLOCKED_PREFIXES = (
    "/admin/", "/reindex/", "/faces/reindex", "/phash/reindex",
    "/phash/pending", "/phash/update", "/cleanup/", "/scan/", "/files/unindexed", "/geo/assign",
)
_TUNNEL_BLOCKED_METHODS = {
    ("POST", "/photos/delete"),
    ("DELETE", "/duplicates"),
    ("DELETE", "/duplicates/phash"),
}

# Throttle: не обновлять last_active_at в БД чаще раза в 60 сек на токен
_session_update_times: dict = {}


def _is_tunnel_request(request: Request) -> bool:
    """True если запрос пришёл через Cloudflare tunnel."""
    if request.headers.get("CF-Ray"):
        return True
    if "trycloudflare.com" in request.headers.get("Host", ""):
        return True
    return False


def _get_session_user_sync(token: str):
    """Валидировать токен сессии. Возвращает dict или None. Вызывать в threadpool."""
    if not db_manager:
        return None
    try:
        session = db_manager.get_session()
        try:
            row = session.execute(
                text(
                    f"SELECT us.user_id, au.is_admin, au.display_name "
                    f"FROM user_session us JOIN app_user au USING(user_id) "
                    f"WHERE us.token = :token "
                    f"  AND us.last_active_at > NOW() - INTERVAL '{SESSION_TIMEOUT_MINUTES} minutes'"
                ),
                {"token": token},
            ).fetchone()
            if row:
                return {"user_id": row.user_id, "is_admin": bool(row.is_admin), "display_name": row.display_name}
            return None
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Session validation error: {e}")
        return None


async def _update_session_db(token: str):
    """Обновить last_active_at (async, fire-and-forget)."""
    loop = asyncio.get_event_loop()
    def _do():
        try:
            session = db_manager.get_session()
            try:
                session.execute(text("UPDATE user_session SET last_active_at = NOW() WHERE token = :token"), {"token": token})
                session.commit()
            finally:
                session.close()
        except Exception as e:
            logger.debug(f"Session update error: {e}")
    await loop.run_in_executor(None, _do)


def _maybe_update_session(token: str):
    """Планирует обновление last_active_at не чаще раза в 60 сек."""
    now = time.time()
    if now - _session_update_times.get(token, 0) > 60:
        _session_update_times[token] = now
        asyncio.create_task(_update_session_db(token))


def _no_cache_html(path: str, response):
    """Запрещаем кэширование HTML страниц (Telegram-браузер и Safari кэшируют агрессивно)."""
    if path.endswith(".html") or path == "/":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method

    # Статику и /auth/* bootstrap пути — пропускаем без проверки
    if (path.startswith("/static/")
            or path in _AUTH_PUBLIC_PATHS
            or path == "/favicon.ico"):
        request.state.user_id = 1
        request.state.is_admin = True
        return _no_cache_html(path, await call_next(request))

    # Localhost / внутренняя сеть (нет CF-Ray) → доверенный → admin
    if not _is_tunnel_request(request):
        request.state.user_id = 1
        request.state.is_admin = True
        return _no_cache_html(path, await call_next(request))

    # === Запрос через Cloudflare tunnel ===

    # Блокируем admin/maintenance пути
    if (path in _TUNNEL_BLOCKED_PATHS
            or any(path.startswith(p) for p in _TUNNEL_BLOCKED_PREFIXES)
            or (method, path) in _TUNNEL_BLOCKED_METHODS
            or path.endswith("/faces/reindex")):
        return JSONResponse({"detail": "Not available via public access"}, status_code=403)

    # /s/{token} — короткий редирект: валидируем токен → cookie → /map.html
    if path.startswith("/s/") and len(path) > 3:
        short_token = path[3:]
        loop = asyncio.get_event_loop()
        user = await loop.run_in_executor(None, _get_session_user_sync, short_token)
        if not user:
            return JSONResponse(
                {"detail": "Ссылка недействительна или истекла. Получите новую через бота (/map)."},
                status_code=401,
            )
        # cache-bust: уникальный URL чтобы Telegram/Safari не отдал старый кэш
        resp = RedirectResponse(url=f"/map.html?_={short_token[:6]}", status_code=302)
        resp.set_cookie(key="session", value=short_token, path="/", max_age=86400, httponly=True, samesite="lax")
        return resp

    # /sf/{token} — короткий редирект для ленты: валидируем токен → cookie → /timeline.html
    if path.startswith("/sf/") and len(path) > 4:
        short_token = path[4:]
        loop = asyncio.get_event_loop()
        user = await loop.run_in_executor(None, _get_session_user_sync, short_token)
        if not user:
            return JSONResponse(
                {"detail": "Ссылка недействительна или истекла. Получите новую через бота (/feed)."},
                status_code=401,
            )
        resp = RedirectResponse(url=f"/timeline.html?_={short_token[:6]}", status_code=302)
        resp.set_cookie(key="session", value=short_token, path="/", max_age=86400, httponly=True, samesite="lax")
        return resp

    # Читаем токен: сначала из URL ?token=, затем из cookie
    token = request.query_params.get("token")
    from_url = bool(token)
    if not token:
        token = request.cookies.get("session")

    if not token:
        if path.endswith(".html") or path == "/":
            return JSONResponse(
                {"detail": "Authentication required. Use /map command in Telegram bot to get a link."},
                status_code=401,
            )
        return JSONResponse({"detail": "Authentication required"}, status_code=401)

    # Валидируем токен в threadpool (sync DB)
    loop = asyncio.get_event_loop()
    user = await loop.run_in_executor(None, _get_session_user_sync, token)

    if not user:
        resp = JSONResponse({"detail": "Session expired or invalid. Request a new link via Telegram bot."}, status_code=401)
        resp.delete_cookie("session")
        return resp

    # Токен из URL → ставим cookie и редиректим на чистый URL
    if from_url:
        # Собираем URL без параметра token
        params = dict(request.query_params)
        params.pop("token", None)
        clean_path = path
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            clean_path = f"{path}?{qs}"
        resp = RedirectResponse(url=clean_path, status_code=302)
        resp.set_cookie(
            key="session",
            value=token,
            path="/",
            max_age=86400,
            httponly=True,
            samesite="lax",
        )
        return resp

    # Токен из cookie — всё ок, обновляем активность
    request.state.user_id = user["user_id"]
    request.state.is_admin = user["is_admin"]
    _maybe_update_session(token)

    return _no_cache_html(path, await call_next(request))



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

    def evict_by_prefix(self, prefix: str):
        """Remove all cache entries whose key starts with prefix (e.g. '56417_')."""
        with self._lock:
            to_delete = [k for k in self._cache if k.startswith(prefix)]
            for k in to_delete:
                self._current_bytes -= self._cache[k][1]
                del self._cache[k]

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


def _unload_clip_model(model_name: str) -> bool:
    """Выгрузить CLIP модель из памяти GPU и вернуть True если модель была загружена."""
    global clip_embedder, clip_embedders
    import gc

    embedder = clip_embedders.pop(model_name, None)
    if embedder is None:
        return False

    # Release PyTorch tensors
    try:
        del embedder.model
    except Exception:
        pass
    try:
        del embedder.processor
    except Exception:
        pass

    # Update default embedder reference if it pointed to this model
    if clip_embedder is not None and clip_embedder.model_name == model_name:
        clip_embedder = None

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info(f"CLIP модель выгружена из памяти: {model_name}")
    return True


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
            logger.info(f"CLIP embedder инициализирован (lazy mode): {settings.CLIP_MODEL}")
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
    offset: int = 0  # Для пагинации фильтрового поиска
    similarity_threshold: float = 0.1  # Lowered for single-word queries
    formats: Optional[List[str]] = None  # Фильтр по форматам: ["jpg", "nef", "heic"]
    translate: bool = True  # Автоперевод на английский
    model: Optional[str] = None  # Модель CLIP для поиска (если None - используется модель по умолчанию)
    person_ids: Optional[List[int]] = None  # Фильтр по персонам (AND: все должны быть на фото)
    multi_model: bool = False  # Мультимодельный поиск (RRF по всем загруженным моделям)
    date_from: Optional[str] = None  # Фильтр по дате от (YYYY-MM-DD)
    date_to: Optional[str] = None  # Фильтр по дате до (YYYY-MM-DD)
    min_lat: Optional[float] = None  # Гео-фильтр: минимальная широта
    max_lat: Optional[float] = None  # Гео-фильтр: максимальная широта
    min_lon: Optional[float] = None  # Гео-фильтр: минимальная долгота
    max_lon: Optional[float] = None  # Гео-фильтр: максимальная долгота
    tag_ids: Optional[List[int]] = None  # Фильтр по тегам (AND: фото должно иметь ВСЕ теги)
    exclude_tag_ids: Optional[List[int]] = None  # Исключить фото с любым из этих тегов (OR: фото не должно иметь НИ ОДНОГО)
    include_hidden: bool = False  # Только для админа: включить скрытые фото (с системными тегами)


class TagResponse(BaseModel):
    """Информация о теге"""
    tag_id: int
    name: str
    is_system: bool
    color: str


class CreateTagRequest(BaseModel):
    """Запрос на создание тега"""
    name: str
    color: str = '#6b7280'


class PhotoTagsRequest(BaseModel):
    """Запрос на добавление/удаление тегов к фото"""
    tag_ids: List[int]


class BulkTagRequest(BaseModel):
    """Запрос на пакетное тегирование"""
    image_ids: List[int]
    tag_ids: List[int]
    mode: str = "add"  # "add" | "remove"


class SearchResult(BaseModel):
    """Результат поиска"""
    image_id: int
    file_path: str
    file_name: Optional[str] = None
    similarity: float
    file_format: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    photo_date: Optional[str] = None
    rotation: int = 0
    tags: Optional[List[TagResponse]] = None


class TextSearchResponse(BaseModel):
    """Ответ текстового поиска"""
    results: List[SearchResult]
    translated_query: Optional[str] = None  # Показать что было переведено
    model: Optional[str] = None  # Какая модель использовалась для поиска
    has_more: bool = False  # Есть ли ещё результаты (для пагинации)


class DeleteRequest(BaseModel):
    """Запрос на удаление файлов"""
    image_ids: List[int]


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


class AIAssistantRequest(BaseModel):
    """Запрос к AI ассистенту"""
    message: str
    conversation_history: List[dict] = []
    current_state: dict = {}


# ==================== Endpoints ====================

# ==================== Auth API ====================


class AuthSessionRequest(BaseModel):
    telegram_id: int
    display_name: str


@app.post("/auth/session")
async def create_auth_session(request_body: AuthSessionRequest, request: Request):
    """Создать сессию для Telegram пользователя. Только с доверенного IP (без CF-Ray)."""
    if _is_tunnel_request(request):
        raise HTTPException(status_code=403, detail="Only available from internal network")

    loop = asyncio.get_event_loop()

    def _create():
        session = db_manager.get_session()
        try:
            # Upsert app_user по telegram_id
            row = session.execute(
                text("SELECT user_id FROM app_user WHERE telegram_id = :tid"),
                {"tid": request_body.telegram_id},
            ).fetchone()
            if row:
                user_id = row.user_id
                session.execute(
                    text("UPDATE app_user SET display_name = :dn, last_seen_at = NOW() WHERE user_id = :uid"),
                    {"dn": request_body.display_name, "uid": user_id},
                )
            else:
                result = session.execute(
                    text("INSERT INTO app_user (telegram_id, display_name, username, is_admin) "
                         "VALUES (:tid, :dn, :uname, FALSE) RETURNING user_id"),
                    {"tid": request_body.telegram_id, "dn": request_body.display_name,
                     "uname": str(request_body.telegram_id)},
                )
                user_id = result.fetchone().user_id

            # Удалить истёкшие сессии пользователя
            session.execute(
                text(f"DELETE FROM user_session WHERE user_id = :uid "
                     f"AND last_active_at < NOW() - INTERVAL '{SESSION_TIMEOUT_MINUTES} minutes'"),
                {"uid": user_id},
            )

            # Создать новую сессию
            token = secrets.token_urlsafe(16)  # 22 chars, 128-bit entropy
            session.execute(
                text("INSERT INTO user_session (token, user_id) VALUES (:token, :uid)"),
                {"token": token, "uid": user_id},
            )
            session.commit()
            return {"token": token, "user_id": user_id}
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    result = await loop.run_in_executor(None, _create)
    return result


@app.get("/auth/check")
async def check_auth_session(token: str = Query(...)):
    """Проверить валидность токена сессии."""
    loop = asyncio.get_event_loop()
    user = await loop.run_in_executor(None, _get_session_user_sync, token)
    if not user:
        return {"valid": False}
    return {"valid": True, "display_name": user["display_name"], "is_admin": user["is_admin"]}


@app.get("/auth/me")
async def get_me(request: Request):
    """Возвращает возможности текущей сессии (для UI)."""
    is_tunnel = _is_tunnel_request(request)
    return {
        "user_id": getattr(request.state, "user_id", 1),
        "is_admin": getattr(request.state, "is_admin", True),
        "can_delete": not is_tunnel,   # Удаление файлов только с localhost
        "via_tunnel": is_tunnel,
    }


@app.get("/auth/logout")
async def logout(request: Request):
    """Завершить сессию, удалить cookie."""
    token = request.cookies.get("session")
    if token:
        loop = asyncio.get_event_loop()
        def _del():
            session = db_manager.get_session()
            try:
                session.execute(text("DELETE FROM user_session WHERE token = :token"), {"token": token})
                session.commit()
            finally:
                session.close()
        await loop.run_in_executor(None, _del)
        _session_update_times.pop(token, None)
    resp = JSONResponse({"status": "logged out"})
    resp.delete_cookie("session")
    return resp


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

        # Failed files count
        failed_count = session.query(func.count(PhotoIndex.image_id)).filter(
            PhotoIndex.index_failed == True
        ).scalar() or 0

        return {
            "total_photos": total_photos,
            "indexed_photos": indexed_photos,
            "pending_photos": total_photos - indexed_photos,
            "indexed_by_model": indexed_by_model,
            "active_model": clip_embedder.model_name if clip_embedder else None,
            "total_faces": total_faces,
            "phash_count": phash_count,
            "failed_count": failed_count,
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
        # Get file paths where embedding is NULL for this model AND not marked as permanently failed
        unindexed = session.query(PhotoIndex.file_path).filter(
            column == None,
            PhotoIndex.index_failed != True,
        ).all()
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
async def search_by_text(request: TextSearchRequest, fastapi_request: Request):
    """
    Поиск фотографий по текстовому описанию (CLIP)

    Если query пустой — возвращает фото по фильтрам (formats, person_ids).
    multi_model=True — мультимодельный поиск (RRF по всем загруженным моделям).

    Пример: {"query": "кошка на диване", "top_k": 10, "model": "SigLIP"}
    """
    try:
        # include_hidden доступен только администраторам
        is_admin = getattr(fastapi_request.state, "is_admin", False)
        include_hidden = request.include_hidden and is_admin

        # Build geo_filters if bounds provided
        geo_filters = None
        if request.min_lat is not None and request.max_lat is not None:
            geo_filters = {
                'min_lat': request.min_lat,
                'max_lat': request.max_lat,
                'min_lon': request.min_lon,
                'max_lon': request.max_lon
            }

        # Если запрос пустой — поиск только по фильтрам (без CLIP)
        if not request.query.strip():
            embedder = get_clip_embedder(request.model)
            results = search_by_filters_only(
                top_k=request.top_k,
                offset=request.offset,
                formats=request.formats,
                person_ids=request.person_ids,
                date_from=request.date_from,
                date_to=request.date_to,
                geo_filters=geo_filters,
                tag_ids=request.tag_ids,
                exclude_tag_ids=request.exclude_tag_ids,
                include_hidden=include_hidden
            )
            return TextSearchResponse(
                results=results,
                translated_query=None,
                model=embedder.model_name,
                has_more=len(results) == request.top_k
            )

        # Перевести запрос на английский если включено
        translated = None
        if request.translate:
            query = translate_query(request.query)
            if query != request.query:
                translated = query
        else:
            query = request.query

        # Мультимодельный поиск (RRF по всем загруженным моделям)
        if request.multi_model:
            image_ids = clip_search_image_ids(
                clip_query=query,
                top_k=500,
                formats=request.formats,
                person_ids=request.person_ids,
                date_from=request.date_from,
                date_to=request.date_to,
                geo_filters=geo_filters,
                tag_ids=request.tag_ids,
                exclude_tag_ids=request.exclude_tag_ids,
                include_hidden=include_hidden
            )
            # Ограничить результаты по top_k
            image_ids = image_ids[:request.top_k]
            results = fetch_search_results_by_ids(image_ids, formats=request.formats,
                                                   date_from=request.date_from, date_to=request.date_to,
                                                   geo_filters=geo_filters,
                                                   exclude_tag_ids=request.exclude_tag_ids,
                                                   include_hidden=include_hidden)

            return TextSearchResponse(
                results=results,
                translated_query=translated,
                model="Multi-model RRF"
            )

        # Одномодельный поиск
        embedder = get_clip_embedder(request.model)
        text_embedding = embedder.embed_text(query)

        if text_embedding is None:
            raise HTTPException(status_code=400, detail="Ошибка обработки текста")

        results = search_by_clip_embedding(
            embedding=text_embedding.tolist(),
            top_k=request.top_k,
            threshold=request.similarity_threshold,
            model_name=embedder.model_name,
            formats=request.formats,
            person_ids=request.person_ids,
            date_from=request.date_from,
            date_to=request.date_to,
            geo_filters=geo_filters,
            tag_ids=request.tag_ids,
            exclude_tag_ids=request.exclude_tag_ids,
            include_hidden=include_hidden
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


@app.get("/search/similar/{image_id}", response_model=TextSearchResponse)
async def search_similar_by_id(
    image_id: int,
    top_k: int = Query(50, ge=1, le=200),
    similarity_threshold: float = Query(0.01, ge=0, le=1),
    model: Optional[str] = Query(None)
):
    """
    Найти похожие фотографии по хранящемуся в БД эмбеддингу.
    Не требует повторной обработки изображения — использует уже вычисленный вектор.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import CLIP_MODEL_COLUMNS
    from sqlalchemy import text as sa_text

    model_name = model or (clip_embedder.model_name if clip_embedder else "SigLIP")
    embedding_col = CLIP_MODEL_COLUMNS.get(model_name)
    if not embedding_col:
        raise HTTPException(status_code=400, detail=f"Неизвестная модель: {model_name}")

    session = db_manager.get_session()
    try:
        query = sa_text(f"""
            SELECT
                pi.image_id, pi.file_path, pi.file_name, pi.file_format,
                pi.latitude, pi.longitude, pi.photo_date, pi.exif_data,
                1 - (pi.{embedding_col} <=> ref.emb) AS similarity
            FROM photo_index pi
            CROSS JOIN (
                SELECT {embedding_col} AS emb FROM photo_index WHERE image_id = :ref_id
            ) ref
            WHERE pi.{embedding_col} IS NOT NULL
              AND pi.image_id != :ref_id
              AND 1 - (pi.{embedding_col} <=> ref.emb) >= :threshold
            ORDER BY pi.{embedding_col} <=> ref.emb
            LIMIT :top_k
        """)

        rows = session.execute(query, {
            "ref_id": image_id,
            "threshold": similarity_threshold,
            "top_k": top_k
        }).fetchall()

        if not rows and session.execute(
            sa_text(f"SELECT 1 FROM photo_index WHERE image_id = :id AND {embedding_col} IS NOT NULL"),
            {"id": image_id}
        ).first() is None:
            raise HTTPException(status_code=404, detail="Эмбеддинг для этого фото не найден")

        results = []
        for row in rows:
            exif = row.exif_data if isinstance(row.exif_data, dict) else {}
            results.append(SearchResult(
                image_id=row.image_id,
                file_path=row.file_path,
                file_name=row.file_name,
                similarity=round(float(row.similarity), 4),
                file_format=row.file_format,
                latitude=row.latitude,
                longitude=row.longitude,
                photo_date=row.photo_date.isoformat() if row.photo_date else None,
                rotation=exif.get("UserRotation", 0) or 0
            ))

        return TextSearchResponse(results=results, model=model_name)

    finally:
        session.close()


@app.get("/photo/{image_id}")
async def get_photo_info(image_id: int):
    """Получить информацию о фотографии"""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    try:
        from models.data_models import PhotoIndex, PhotoTag, Tag

        session = db_manager.get_session()

        try:
            photo = session.query(PhotoIndex).filter_by(image_id=image_id).first()

            if not photo:
                raise HTTPException(status_code=404, detail="Фотография не найдена")

            # Загрузить теги фото
            tags = session.query(Tag).join(PhotoTag, PhotoTag.tag_id == Tag.tag_id).filter(
                PhotoTag.image_id == image_id
            ).all()
            tags_data = [{"tag_id": t.tag_id, "name": t.name, "is_system": t.is_system, "color": t.color}
                         for t in tags]

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
                "exif_data": photo.exif_data,
                "rotation": (photo.exif_data or {}).get("UserRotation", 0) if isinstance(photo.exif_data, dict) else 0,
                "is_hidden": photo.is_hidden,
                "tags": tags_data,
            }

        finally:
            session.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения информации о фотографии: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Поворот фотографии ====================


@app.post("/photo/{image_id}/rotate")
async def rotate_photo(
    image_id: int,
    degrees: int = Query(90, description="Degrees to rotate CW (+) or CCW (-), multiple of 90"),
):
    """
    Повернуть фотографию (не разрушающий поворот).

    Поворот хранится в exif_data['UserRotation'] таблицы photo_index.
    Оригинальный файл не изменяется — поворот применяется при отдаче изображений.
    Bbox лиц пересчитываются математически без повторной детекции.

    degrees: +90 = по часовой стрелке, -90 = против (или 270)
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    # Normalize degrees to 90 / 180 / 270
    degrees = degrees % 360
    if degrees not in (0, 90, 180, 270):
        raise HTTPException(status_code=400, detail="degrees must be a multiple of 90")
    if degrees == 0:
        return {"image_id": image_id, "rotation": 0, "message": "No change"}

    from models.data_models import PhotoIndex, Face as FaceModel
    from sqlalchemy.orm.attributes import flag_modified
    import glob as _glob

    session = db_manager.get_session()
    try:
        photo = session.query(PhotoIndex).filter_by(image_id=image_id).first()
        if not photo:
            raise HTTPException(status_code=404, detail="Фотография не найдена")

        # Read current user rotation from exif_data
        exif = photo.exif_data if isinstance(photo.exif_data, dict) else {}
        old_rotation = exif.get("UserRotation", 0) or 0
        new_rotation = (old_rotation + degrees) % 360

        # Current DB dimensions (post-EXIF + post-old-rotation)
        W = photo.width or 1
        H = photo.height or 1

        # Transform face bboxes for the applied delta
        faces = session.query(FaceModel).filter_by(image_id=image_id).all()
        for face in faces:
            x1, y1, x2, y2 = face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2
            if degrees == 90:
                face.bbox_x1 = H - y2
                face.bbox_y1 = x1
                face.bbox_x2 = H - y1
                face.bbox_y2 = x2
            elif degrees == 180:
                face.bbox_x1 = W - x2
                face.bbox_y1 = H - y2
                face.bbox_x2 = W - x1
                face.bbox_y2 = H - y1
            elif degrees == 270:
                face.bbox_x1 = y1
                face.bbox_y1 = W - x2
                face.bbox_x2 = y2
                face.bbox_y2 = W - x1

        # Swap width/height for 90°/270°
        if degrees in (90, 270):
            photo.width, photo.height = H, W

        # Save rotation into exif_data (creates dict if was NULL)
        new_exif = dict(exif)
        new_exif["UserRotation"] = new_rotation
        photo.exif_data = new_exif
        flag_modified(photo, "exif_data")  # tell SQLAlchemy JSONB was mutated
        session.commit()

        # Invalidate thumbnail cache for this image
        _thumb_mem_cache.evict_by_prefix(f"{image_id}_")
        import os as _os
        cache_dir = settings.THUMB_CACHE_DIR
        for cache_file in _glob.glob(_os.path.join(cache_dir, f"{image_id}_*.jpg")):
            try:
                _os.remove(cache_file)
            except OSError:
                pass

        return {
            "image_id": image_id,
            "rotation": new_rotation,
            "width": photo.width,
            "height": photo.height,
        }

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка поворота фото {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


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

        from PIL import Image

        # Use integer value from exifread tag (EXIF spec: 1-8)
        orientation_val = orientation_tag.values[0] if orientation_tag.values else 1

        if orientation_val == 6:    # Rotated 90 CW
            return img.transpose(Image.Transpose.ROTATE_270)
        elif orientation_val == 3:  # Rotated 180
            return img.transpose(Image.Transpose.ROTATE_180)
        elif orientation_val == 8:  # Rotated 90 CCW
            return img.transpose(Image.Transpose.ROTATE_90)
        elif orientation_val == 2:  # Mirrored horizontal
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif orientation_val == 4:  # Mirrored vertical
            return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif orientation_val == 5:  # Mirrored horizontal + Rotated 90 CW
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return img.transpose(Image.Transpose.ROTATE_270)
        elif orientation_val == 7:  # Mirrored horizontal + Rotated 90 CCW
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


def _apply_user_rotation(img: 'Image.Image', rotation: int) -> 'Image.Image':
    """Apply user-stored CW rotation (0/90/180/270) to a PIL image."""
    from PIL import Image as _Image
    if rotation == 90:
        return img.transpose(_Image.Transpose.ROTATE_270)   # 90° CW
    elif rotation == 180:
        return img.transpose(_Image.Transpose.ROTATE_180)
    elif rotation == 270:
        return img.transpose(_Image.Transpose.ROTATE_90)    # 90° CCW = 270° CW
    return img


def _get_photo_file_and_rotation(image_id: str):
    """Return (file_path, rotation) from DB using exif_data['UserRotation'], or (None, 0)."""
    from models.data_models import PhotoIndex
    session = db_manager.get_session()
    try:
        photo = session.query(
            PhotoIndex.file_path, PhotoIndex.exif_data
        ).filter_by(image_id=image_id).first()
        if photo:
            rotation = 0
            if photo.exif_data and isinstance(photo.exif_data, dict):
                rotation = photo.exif_data.get("UserRotation", 0) or 0
            return photo.file_path, rotation
        return None, 0
    finally:
        session.close()


@app.get("/image/{image_id}/thumb")
def get_image_thumbnail(
    image_id: str,
    size: int = Query(400, ge=50, le=800, description="Max thumbnail size in pixels"),
    r: int = Query(0, description="Rotation hint for memory cache keying (0/90/180/270)")
):
    """Получить миниатюру изображения: memory cache → disk cache → generate"""
    import os, io
    mem_key = f"{image_id}_{size}_{r}"
    _cache_headers = {"Cache-Control": "public, max-age=86400"}

    # 1. Memory cache — instant, no I/O at all
    cached_bytes = _thumb_mem_cache.get(mem_key)
    if cached_bytes is not None:
        return Response(
            content=cached_bytes,
            media_type="image/jpeg",
            headers={**_cache_headers, "X-Cache": "MEM"}
        )

    # 2. DB query — needed before disk cache check to include rotation in disk key.
    #    Cheap primary-key lookup; avoids serving stale pre-rotation thumbnails.
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    file_path, rotation = _get_photo_file_and_rotation(image_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Изображение не найдено")

    # Disk cache key encodes rotation so stale pre-rotation files are ignored.
    # rotation=0 uses the original key format for backward compatibility.
    disk_key = f"{image_id}_{size}" if not rotation else f"{image_id}_{size}_{rotation}"
    cache_file = os.path.join(settings.THUMB_CACHE_DIR, f"{disk_key}.jpg")

    # 3. Disk cache — read file, store in memory for next time
    if os.path.exists(cache_file):
        try:
            data = open(cache_file, 'rb').read()
            _thumb_mem_cache.put(mem_key, data)
            return Response(
                content=data,
                media_type="image/jpeg",
                headers={**_cache_headers, "X-Cache": "DISK"}
            )
        except OSError:
            pass  # fall through to generate

    # 4. Generate — cache miss
    try:
        # fast_mode=True для RAW: half_size ускоряет в ~4 раза
        img = load_image_any_format(file_path, fast_mode=True)
        if rotation:
            img = _apply_user_rotation(img, rotation)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)

        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        quality = 85 if size >= 300 else 75

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        data = buffer.getvalue()

        # Save to disk cache (rotation-aware key)
        os.makedirs(settings.THUMB_CACHE_DIR, exist_ok=True)
        try:
            with open(cache_file, 'wb') as f:
                f.write(data)
        except OSError as cache_err:
            logger.warning(f"Failed to cache thumb {cache_file}: {cache_err}")

        # Store in memory cache
        _thumb_mem_cache.put(mem_key, data)

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

    file_path, rotation = _get_photo_file_and_rotation(image_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Изображение не найдено")

    try:
        import io

        # fast_mode=False для полного качества при просмотре
        img = load_image_any_format(file_path, fast_mode=False)
        if rotation:
            img = _apply_user_rotation(img, rotation)

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


# Whitelist of allowed file format values — used to prevent SQL injection in format filters
ALLOWED_FORMATS = frozenset({
    "jpg", "jpeg", "heic", "heif", "png", "nef", "cr2", "arw",
    "dng", "raf", "orf", "rw2", "webp", "bmp", "tiff", "tif",
})


def _build_date_filter_sql(date_from: Optional[str] = None, date_to: Optional[str] = None) -> str:
    """Build SQL date filter clause for photo_date. Validates format to prevent injection."""
    from datetime import datetime as _dt
    parts = []
    if date_from:
        try:
            safe = _dt.strptime(date_from, "%Y-%m-%d").strftime("%Y-%m-%d")
            parts.append(f"AND photo_date >= '{safe}'::date")
        except ValueError:
            pass
    if date_to:
        try:
            safe = _dt.strptime(date_to, "%Y-%m-%d").strftime("%Y-%m-%d")
            parts.append(f"AND photo_date < ('{safe}'::date + interval '1 day')")
        except ValueError:
            pass
    return " ".join(parts)


def _build_format_filter_sql(formats: Optional[List[str]] = None) -> str:
    """Build SQL file_format IN (...) filter. Only allows whitelisted format values."""
    if not formats:
        return ""
    safe = [f.lower().lstrip('.') for f in formats if f.lower().lstrip('.') in ALLOWED_FORMATS]
    if not safe:
        return ""
    fmt_str = ','.join(f"'{f}'" for f in safe)
    return f"AND file_format IN ({fmt_str})"


def _build_geo_filter_sql(geo_filters: Optional[dict] = None) -> str:
    """Build SQL bounding-box geo filter clause."""
    if not geo_filters or geo_filters.get('min_lat') is None:
        return ""
    return (
        f"AND latitude >= {float(geo_filters['min_lat'])} "
        f"AND latitude <= {float(geo_filters['max_lat'])} "
        f"AND longitude >= {float(geo_filters['min_lon'])} "
        f"AND longitude <= {float(geo_filters['max_lon'])} "
        f"AND latitude IS NOT NULL AND longitude IS NOT NULL"
    )


def _build_person_filter_sql(person_ids: Optional[List[int]] = None) -> str:
    """Build SQL person filter (AND logic: ALL selected persons must appear on the same photo)."""
    if not person_ids:
        return ""
    pids = ','.join(str(int(p)) for p in person_ids)
    count = len(person_ids)
    return f"""AND image_id IN (
        SELECT image_id FROM faces
        WHERE person_id IN ({pids})
        GROUP BY image_id
        HAVING COUNT(DISTINCT person_id) = {count}
    )"""


def _build_hidden_filter_sql(include_hidden: bool = False) -> str:
    """Exclude photos with is_hidden=TRUE (have system tags). Default: hide them."""
    return "" if include_hidden else "AND NOT is_hidden"


def _build_tag_filter_sql(tag_ids: Optional[List[int]] = None) -> str:
    """Filter photos that have ALL specified tags (AND logic). Safe: uses integer cast."""
    if not tag_ids:
        return ""
    tids = ','.join(str(int(t)) for t in tag_ids)
    count = len(tag_ids)
    if count == 1:
        return f"AND image_id IN (SELECT image_id FROM photo_tag WHERE tag_id = {tids})"
    return f"""AND image_id IN (
        SELECT image_id FROM photo_tag
        WHERE tag_id IN ({tids})
        GROUP BY image_id
        HAVING COUNT(DISTINCT tag_id) = {count}
    )"""


def _build_tag_exclude_filter_sql(exclude_tag_ids: Optional[List[int]] = None) -> str:
    """Exclude photos that have ANY of the specified tags (OR exclusion logic). Safe: uses integer cast."""
    if not exclude_tag_ids:
        return ""
    tids = ','.join(str(int(t)) for t in exclude_tag_ids)
    return f"AND image_id NOT IN (SELECT image_id FROM photo_tag WHERE tag_id IN ({tids}))"


def _batch_load_tags(session, image_ids: list) -> dict:
    """Batch-load tags for a list of image IDs. Returns {image_id: [tag_dicts]}."""
    if not image_ids:
        return {}
    from models.data_models import PhotoTag, Tag as TagModel
    rows = session.query(
        PhotoTag.image_id, TagModel.tag_id, TagModel.name,
        TagModel.is_system, TagModel.color
    ).join(TagModel, PhotoTag.tag_id == TagModel.tag_id).filter(
        PhotoTag.image_id.in_(image_ids)
    ).all()
    result: dict = {}
    for row in rows:
        result.setdefault(row.image_id, []).append({
            "tag_id": row.tag_id, "name": row.name,
            "is_system": row.is_system, "color": row.color,
        })
    return result


def search_by_filters_only(top_k: int, offset: int = 0, formats: Optional[List[str]] = None, person_ids: Optional[List[int]] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, geo_filters: dict = None, tag_ids: Optional[List[int]] = None, exclude_tag_ids: Optional[List[int]] = None, include_hidden: bool = False) -> List[SearchResult]:
    """Поиск фото только по фильтрам (без текстового запроса), сортировка по дате"""
    from sqlalchemy import text as sa_text

    session = db_manager.get_session()

    try:
        format_filter = _build_format_filter_sql(formats)
        person_filter = _build_person_filter_sql(person_ids)
        date_filter = _build_date_filter_sql(date_from, date_to)
        geo_sql = _build_geo_filter_sql(geo_filters)
        tag_filter = _build_tag_filter_sql(tag_ids)
        tag_exclude_filter = _build_tag_exclude_filter_sql(exclude_tag_ids)
        hidden_filter = _build_hidden_filter_sql(include_hidden)

        query = sa_text(f"""
            SELECT image_id, file_path, file_name, file_format, latitude, longitude, photo_date, exif_data
            FROM photo_index
            WHERE 1=1
              {hidden_filter}
              {format_filter}
              {person_filter}
              {date_filter}
              {geo_sql}
              {tag_filter}
              {tag_exclude_filter}
            ORDER BY photo_date DESC NULLS LAST, image_id DESC
            LIMIT :top_k OFFSET :offset
        """)

        result = session.execute(query, {'top_k': top_k, 'offset': offset})

        rows_list = list(result)
        tags_map = _batch_load_tags(session, [r.image_id for r in rows_list])

        results = []
        for row in rows_list:
            exif = row.exif_data if isinstance(row.exif_data, dict) else {}
            results.append(SearchResult(
                image_id=row.image_id,
                file_path=row.file_path,
                file_name=row.file_name,
                similarity=1.0,
                file_format=row.file_format,
                latitude=row.latitude,
                longitude=row.longitude,
                photo_date=row.photo_date.isoformat() if row.photo_date else None,
                rotation=exif.get("UserRotation", 0) or 0,
                tags=tags_map.get(row.image_id, []),
            ))

        return results

    finally:
        session.close()


def search_by_clip_embedding(embedding: List[float], top_k: int, threshold: float, model_name: str, formats: Optional[List[str]] = None, person_ids: Optional[List[int]] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, geo_filters: dict = None, tag_ids: Optional[List[int]] = None, exclude_tag_ids: Optional[List[int]] = None, include_hidden: bool = False) -> List[SearchResult]:
    """Поиск по CLIP эмбиддингу через pgvector для конкретной модели"""
    from models.data_models import CLIP_MODEL_COLUMNS
    from sqlalchemy import text

    session = db_manager.get_session()

    try:
        embedding_column = CLIP_MODEL_COLUMNS.get(model_name)
        if not embedding_column:
            raise ValueError(f"Неизвестная модель: {model_name}")

        embedding_str = '[' + ','.join(map(str, embedding)) + ']'

        format_filter = _build_format_filter_sql(formats)
        person_filter = _build_person_filter_sql(person_ids)
        date_filter = _build_date_filter_sql(date_from, date_to)
        geo_sql = _build_geo_filter_sql(geo_filters)
        tag_filter = _build_tag_filter_sql(tag_ids)
        tag_exclude_filter = _build_tag_exclude_filter_sql(exclude_tag_ids)
        hidden_filter = _build_hidden_filter_sql(include_hidden)

        query = text(f"""
            SELECT
                image_id,
                file_path,
                file_name,
                file_format,
                latitude,
                longitude,
                photo_date,
                exif_data,
                1 - ({embedding_column} <=> '{embedding_str}'::vector) as similarity
            FROM photo_index
            WHERE {embedding_column} IS NOT NULL
              AND 1 - ({embedding_column} <=> '{embedding_str}'::vector) >= :threshold
              {hidden_filter}
              {format_filter}
              {person_filter}
              {date_filter}
              {geo_sql}
              {tag_filter}
              {tag_exclude_filter}
            ORDER BY {embedding_column} <=> '{embedding_str}'::vector
            LIMIT :top_k
        """)

        result = session.execute(query, {
            'threshold': threshold,
            'top_k': top_k
        })

        results = []
        for row in result:
            exif = row.exif_data if isinstance(row.exif_data, dict) else {}
            results.append(SearchResult(
                image_id=row.image_id,
                file_path=row.file_path,
                file_name=row.file_name,
                similarity=float(row.similarity),
                file_format=row.file_format,
                latitude=row.latitude,
                longitude=row.longitude,
                photo_date=row.photo_date.isoformat() if row.photo_date else None,
                rotation=exif.get("UserRotation", 0) or 0
            ))

        return results

    except Exception as e:
        logger.error(f"search_by_clip_embedding error: {e}", exc_info=True)
        return []
    finally:
        session.close()


def fetch_search_results_by_ids(image_ids: List[int], formats: Optional[List[str]] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, geo_filters: dict = None, exclude_tag_ids: Optional[List[int]] = None, include_hidden: bool = False) -> List[SearchResult]:
    """Fetch SearchResult objects for given image_ids, preserving order (RRF rank)."""
    if not image_ids or not db_manager:
        return []

    from sqlalchemy import text as sa_text
    session = db_manager.get_session()
    try:
        ids_str = ','.join(str(int(i)) for i in image_ids)

        format_filter = _build_format_filter_sql(formats)
        date_filter = _build_date_filter_sql(date_from, date_to)
        geo_sql = _build_geo_filter_sql(geo_filters)
        tag_exclude_filter = _build_tag_exclude_filter_sql(exclude_tag_ids)
        hidden_filter = _build_hidden_filter_sql(include_hidden)

        query = sa_text(f"""
            SELECT image_id, file_path, file_name, file_format, latitude, longitude, photo_date, exif_data
            FROM photo_index
            WHERE image_id IN ({ids_str})
            {hidden_filter}
            {format_filter}
            {date_filter}
            {geo_sql}
            {tag_exclude_filter}
        """)

        rows_by_id = {}
        for row in session.execute(query):
            rows_by_id[row.image_id] = row

        tags_map = _batch_load_tags(session, list(rows_by_id.keys()))

        # Build results in RRF rank order with normalized similarity
        results = []
        total = len(image_ids)
        for rank, img_id in enumerate(image_ids):
            row = rows_by_id.get(img_id)
            if not row:
                continue
            # Normalized similarity: rank 1 → 1.0, last → ~0.5
            similarity = 1.0 - (rank / (total * 2)) if total > 1 else 1.0
            exif = row.exif_data if isinstance(row.exif_data, dict) else {}
            results.append(SearchResult(
                image_id=row.image_id,
                file_path=row.file_path,
                file_name=row.file_name,
                similarity=round(similarity, 4),
                file_format=row.file_format,
                latitude=row.latitude,
                longitude=row.longitude,
                photo_date=row.photo_date.isoformat() if row.photo_date else None,
                rotation=exif.get("UserRotation", 0) or 0,
                tags=tags_map.get(img_id, []),
            ))

        return results

    finally:
        session.close()


# Per-model minimum similarity thresholds — below this, results are noise
MODEL_MIN_THRESHOLDS = {
    "SigLIP": 0.06,     # SigLIP range: good 0.15-0.30, noise < 0.06
    "ViT-B/32": 0.18,   # ViT-B/32 range: good 0.25-0.40, noise < 0.18
    "ViT-B/16": 0.18,   # ViT-B/16 range: good 0.25-0.40, noise < 0.18
    "ViT-L/14": 0.15,   # ViT-L/14 range: good 0.22-0.35, noise < 0.15
}


def clip_search_image_ids(clip_query: str, top_k: int = 500, threshold: float = 0.01,
                          geo_filters: dict = None, relative_cutoff: float = 0.65,
                          rrf_cutoff: float = 0.35,
                          candidate_ids: List[int] = None,
                          formats: Optional[List[str]] = None,
                          person_ids: Optional[List[int]] = None,
                          date_from: Optional[str] = None,
                          date_to: Optional[str] = None,
                          tag_ids: Optional[List[int]] = None,
                          exclude_tag_ids: Optional[List[int]] = None,
                          include_hidden: bool = False) -> List[int]:
    """
    Multi-model CLIP search using Reciprocal Rank Fusion (RRF).

    Each model has its own similarity range (SigLIP ~0.06-0.30, ViT-B/32 ~0.18-0.40).
    RRF merges ranked lists without needing score normalization.

    Algorithm:
    1. For each loaded model: get top_k results sorted by similarity
    2. Per-model minimum threshold: discard anything below MODEL_MIN_THRESHOLDS
    3. Per-model adaptive cutoff: keep only results >= best_score * relative_cutoff
    4. RRF score for each photo: sum of 1/(k + rank) across all models where found
       - k=60 (standard RRF constant) prevents top-1 from dominating
    5. Final adaptive cutoff: keep results >= best_rrf_score * rrf_cutoff

    If candidate_ids is provided, search ONLY among those photos.
    Used when person filter narrows the pool first, then CLIP ranks within it.
    """
    from models.data_models import CLIP_MODEL_COLUMNS
    from sqlalchemy import text as sa_text

    if not db_manager:
        return []

    session = db_manager.get_session()
    try:
        rrf_scores = {}  # image_id -> rrf_score
        vote_counts = {}  # image_id -> num models that found it
        num_models = 0
        model_stats = []
        k = 60  # RRF constant

        # Build shared filter SQL once — same for all models
        geo_sql = _build_geo_filter_sql(geo_filters)
        format_sql = _build_format_filter_sql(formats)
        person_sql = _build_person_filter_sql(person_ids)
        date_sql = _build_date_filter_sql(date_from, date_to)
        tag_sql = _build_tag_filter_sql(tag_ids)
        tag_exclude_sql = _build_tag_exclude_filter_sql(exclude_tag_ids)
        hidden_sql = _build_hidden_filter_sql(include_hidden)
        candidate_sql = ""
        if candidate_ids:
            cid_list = ",".join(str(int(i)) for i in candidate_ids)
            candidate_sql = f"AND image_id IN ({cid_list})"

        for model_name, column_name in CLIP_MODEL_COLUMNS.items():
            try:
                embedder = get_clip_embedder(model_name)
            except Exception:
                continue

            text_embedding = embedder.embed_text(clip_query)
            if text_embedding is None:
                continue

            embedding_str = '[' + ','.join(map(str, text_embedding)) + ']'

            query = sa_text(f"""
                SELECT image_id,
                       1 - ({column_name} <=> '{embedding_str}'::vector) as similarity
                FROM photo_index
                WHERE {column_name} IS NOT NULL
                  AND 1 - ({column_name} <=> '{embedding_str}'::vector) >= :threshold
                  {hidden_sql}
                  {geo_sql}
                  {candidate_sql}
                  {format_sql}
                  {person_sql}
                  {date_sql}
                  {tag_sql}
                  {tag_exclude_sql}
                ORDER BY {column_name} <=> '{embedding_str}'::vector
                LIMIT :top_k
            """)

            raw_results = []
            for row in session.execute(query, {'threshold': threshold, 'top_k': top_k}):
                raw_results.append((row.image_id, float(row.similarity)))

            if not raw_results:
                continue

            num_models += 1

            # Per-model minimum threshold — discard noise below model-specific floor
            min_thresh = MODEL_MIN_THRESHOLDS.get(model_name, 0.10)
            raw_results = [(img_id, sim) for img_id, sim in raw_results if sim >= min_thresh]
            if not raw_results:
                model_stats.append(f"{model_name}: 0 (all below min_thresh={min_thresh:.2f})")
                continue

            # Per-model adaptive cutoff (stricter when searching within candidate set)
            best_sim = raw_results[0][1]
            effective_cutoff = 0.80 if candidate_ids else relative_cutoff
            cutoff_sim = best_sim * effective_cutoff
            filtered = [(img_id, sim) for img_id, sim in raw_results if sim >= cutoff_sim]

            # RRF: score = 1 / (k + rank), rank starts at 1
            for rank, (img_id, sim) in enumerate(filtered, 1):
                rrf_scores[img_id] = rrf_scores.get(img_id, 0.0) + 1.0 / (k + rank)
                vote_counts[img_id] = vote_counts.get(img_id, 0) + 1

            model_stats.append(f"{model_name}: {len(raw_results)}→{len(filtered)} "
                               f"(best={best_sim:.4f}, cut={cutoff_sim:.4f})")

        if not rrf_scores:
            return []

        # Sort by RRF score descending
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        if candidate_ids:
            # When searching within a candidate set (e.g. person's photos),
            # use strict cutoff — small pool means even weak matches get decent RRF scores
            best_rrf = ranked[0][1]
            min_rrf = best_rrf * 0.65
            result_ids = [img_id for img_id, score in ranked if score >= min_rrf]
        else:
            # Two-phase cutoff:
            # 1. Adaptive: keep results above rrf_cutoff of best RRF score
            # 2. Hard limit: max 300 results (tighter to reduce noise)
            best_rrf = ranked[0][1]
            min_rrf = best_rrf * rrf_cutoff
            result_ids = [img_id for img_id, score in ranked if score >= min_rrf]
            result_ids = result_ids[:300]

        multi_vote = sum(1 for img_id in result_ids if vote_counts.get(img_id, 0) >= 2)
        logger.info(f"Multi-model RRF search '{clip_query[:60]}': "
                     f"{len(rrf_scores)} unique → {len(result_ids)} final "
                     f"({num_models} models, {multi_vote} multi-vote). "
                     + " | ".join(model_stats))
        return result_ids

    except Exception as e:
        logger.error(f"Multi-model CLIP search error: {e}", exc_info=True)
        return []
    finally:
        session.close()


def clip_search_image_ids_single(clip_query: str, top_k: int = 500, threshold: float = 0.01,
                                  geo_filters: dict = None, relative_cutoff: float = 0.55) -> List[int]:
    """
    Single-model CLIP search using the default model.
    Returns list of image_ids sorted by similarity (descending).

    Uses adaptive threshold: fetches top_k results, then keeps only those
    within `relative_cutoff` of the best match's similarity score.
    E.g. if best=0.12 and cutoff=0.55, keeps results with similarity >= 0.066.
    """
    from models.data_models import CLIP_MODEL_COLUMNS
    from sqlalchemy import text as sa_text

    if not db_manager or not clip_embedder:
        return []

    session = db_manager.get_session()
    try:
        model_name = clip_embedder.model_name
        column_name = CLIP_MODEL_COLUMNS.get(model_name)
        if not column_name:
            return []

        text_embedding = clip_embedder.embed_text(clip_query)
        if text_embedding is None:
            return []

        embedding_str = '[' + ','.join(map(str, text_embedding)) + ']'

        geo_sql = ""
        if geo_filters:
            parts = []
            if geo_filters.get('min_lat') is not None:
                parts.append(f"AND latitude >= {float(geo_filters['min_lat'])}")
                parts.append(f"AND latitude <= {float(geo_filters['max_lat'])}")
                parts.append(f"AND longitude >= {float(geo_filters['min_lon'])}")
                parts.append(f"AND longitude <= {float(geo_filters['max_lon'])}")
                parts.append("AND latitude IS NOT NULL AND longitude IS NOT NULL")
            geo_sql = " ".join(parts)

        query = sa_text(f"""
            SELECT image_id,
                   1 - ({column_name} <=> '{embedding_str}'::vector) as similarity
            FROM photo_index
            WHERE {column_name} IS NOT NULL
              AND 1 - ({column_name} <=> '{embedding_str}'::vector) >= :threshold
              {geo_sql}
            ORDER BY {column_name} <=> '{embedding_str}'::vector
            LIMIT :top_k
        """)

        rows = []
        for row in session.execute(query, {'threshold': threshold, 'top_k': top_k}):
            rows.append((row.image_id, float(row.similarity)))

        if not rows:
            return []

        # Adaptive cutoff: keep only results within relative_cutoff of the best score
        best_sim = rows[0][1]
        min_sim = best_sim * relative_cutoff
        result_ids = [img_id for img_id, sim in rows if sim >= min_sim]

        logger.info(f"CLIP search '{clip_query[:50]}': {len(rows)} raw → {len(result_ids)} after cutoff "
                     f"(best={best_sim:.4f}, min={min_sim:.4f})")
        return result_ids

    except Exception as e:
        logger.error(f"Single-model CLIP search error: {e}", exc_info=True)
        return []
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


def _run_reindex(model_name: Optional[str] = None, file_list: Optional[List[str]] = None):
    """Фоновая задача переиндексации.

    Args:
        model_name: Имя CLIP модели (None = модель по умолчанию).
        file_list:  Готовый список файлов для индексации (передаётся из _run_index_all,
                    чтобы не делать сканирование ФС повторно).
                    Если None — сканируем файловую систему самостоятельно.
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

        # Выгрузить все CLIP модели кроме целевой — освобождает VRAM для батч-индексации
        other_models = [m for m in list(clip_embedders.keys()) if m != target_model]
        if other_models:
            for m in other_models:
                _unload_clip_model(m)
            logger.info(f"Выгружены модели {other_models} перед индексацией, освобождена VRAM")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Использовать уже загруженную модель если она совпадает
        if target_model in clip_embedders:
            embedder_to_use = clip_embedders[target_model]
            logger.info(f"Переиспользую загруженную модель: {target_model}")
        elif clip_embedder and clip_embedder.model_name == target_model:
            embedder_to_use = clip_embedder
            logger.info(f"Переиспользую модель по умолчанию: {target_model}")
        else:
            embedder_to_use = None
            logger.info(f"Будет создана новая модель: {target_model}")

        indexing_service = IndexingService(model_name=model_name, clip_embedder=embedder_to_use)
        active_indexing_service = indexing_service  # Сохраняем ссылку для /reindex/status

        if file_list is None:
            # Сканируем файловую систему — обнаруживаем как новые файлы (не в БД),
            # так и существующие без эмбеддинга. Медленно через Docker bind mount,
            # но при вызове из _run_index_all это делается один раз на всю очередь.
            logger.info("Сканирование файловой системы для обнаружения новых файлов...")
            file_list = indexing_service.fast_scan_files(settings.PHOTO_STORAGE_PATH)
            logger.info(f"Найдено {len(file_list)} файлов на диске")
        else:
            logger.info(f"Используем предварительно просканированный список: {len(file_list)} файлов")

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

            # Выгрузить все CLIP модели кроме целевой — освобождает VRAM для батч-индексации
            other_models = [m for m in list(clip_embedders.keys()) if m != target_model]
            if other_models:
                for m in other_models:
                    _unload_clip_model(m)
                logger.info(f"Выгружены модели {other_models} перед индексацией, освобождена VRAM")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

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

        # Enrich with GPS, format, file_name from DB
        all_ids = [item['image_id'] for group in groups for item in group]
        session = db_manager.get_session()
        try:
            rows = session.execute(text(
                "SELECT image_id, file_name, file_format, latitude, longitude "
                "FROM photo_index WHERE image_id = ANY(:ids)"
            ), {"ids": all_ids}).fetchall()
        finally:
            session.close()
        info_map = {r[0]: {"file_name": r[1], "file_format": r[2],
                            "latitude": r[3], "longitude": r[4]} for r in rows}

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
                            "file_name": info_map.get(item['image_id'], {}).get("file_name", ""),
                            "file_format": info_map.get(item['image_id'], {}).get("file_format", ""),
                            "size_bytes": item['size'],
                            "latitude": info_map.get(item['image_id'], {}).get("latitude"),
                            "longitude": info_map.get(item['image_id'], {}).get("longitude"),
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
    person_ids: Optional[List[int]] = None  # Фильтр по персонам
    person_mode: str = "or"  # "or" = любой из выбранных, "and" = все на одном фото
    clip_query: Optional[str] = None  # Optimized CLIP query (English) for text search filter
    original_query: Optional[str] = None  # Original user query (for display)
    clip_image_ids: Optional[List[int]] = None  # Cached CLIP result IDs (skip re-search)
    tag_ids: Optional[List[int]] = None  # Фильтр по тегам (AND логика)
    exclude_tag_ids: Optional[List[int]] = None  # Исключить фото с этими тегами (OR логика)


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
    file_name: Optional[str] = None
    rotation: int = 0
    tags: Optional[list] = None


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
async def get_map_clusters(request: MapClusterRequest, fastapi_request: Request):
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
        is_admin = getattr(fastapi_request.state, "is_admin", True)

        # Базовый фильтр по bounding box
        filters = [
            PhotoIndex.latitude != None,
            PhotoIndex.longitude != None,
            PhotoIndex.latitude >= request.min_lat,
            PhotoIndex.latitude <= request.max_lat,
            PhotoIndex.longitude >= request.min_lon,
            PhotoIndex.longitude <= request.max_lon,
        ]
        if not is_admin:
            filters.append(PhotoIndex.is_hidden == False)  # noqa: E712

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

        # Фильтр по персонам
        if request.person_ids:
            from models.data_models import Face as FaceModel
            if request.person_mode == "and" and len(request.person_ids) > 1:
                # AND: фото где есть ВСЕ выбранные персоны
                pids = ','.join(str(int(p)) for p in request.person_ids)
                and_subq = text(f"""
                    SELECT image_id FROM faces
                    WHERE person_id IN ({pids})
                    GROUP BY image_id
                    HAVING COUNT(DISTINCT person_id) = {len(request.person_ids)}
                """)
                filters.append(PhotoIndex.image_id.in_(and_subq))
            else:
                # OR: фото любого из выбранных
                person_photo_subq = session.query(FaceModel.image_id).filter(
                    FaceModel.person_id.in_(request.person_ids)
                )
                filters.append(PhotoIndex.image_id.in_(person_photo_subq))

        # Фильтр по тегам (AND логика — фото должно содержать все указанные теги)
        if request.tag_ids:
            from models.data_models import PhotoTag as PhotoTagModel
            for tid in request.tag_ids:
                filters.append(
                    PhotoIndex.image_id.in_(
                        session.query(PhotoTagModel.image_id).filter(PhotoTagModel.tag_id == tid)
                    )
                )

        # Исключить фото с любым из этих тегов (OR логика)
        if request.exclude_tag_ids:
            from models.data_models import PhotoTag as PhotoTagExcludeModel
            exc_subq = session.query(PhotoTagExcludeModel.image_id).filter(
                PhotoTagExcludeModel.tag_id.in_(request.exclude_tag_ids)
            )
            filters.append(~PhotoIndex.image_id.in_(exc_subq))

        # Фильтр по CLIP текстовому поиску
        clip_ids = None
        logger.info(f"[CLUSTERS] clip_query={request.clip_query}, clip_image_ids={len(request.clip_image_ids) if request.clip_image_ids else None}, "
                     f"person_ids={request.person_ids}, bounds=[{request.min_lat:.1f}..{request.max_lat:.1f}, {request.min_lon:.1f}..{request.max_lon:.1f}]")
        if request.clip_image_ids:
            # Use cached IDs from previous search (no re-search on scroll/zoom)
            clip_ids = request.clip_image_ids
            filters.append(PhotoIndex.image_id.in_(clip_ids))
        elif request.clip_query:
            # When person filter active: get person's photos first, then CLIP ranks within them
            person_candidate_ids = None
            if request.person_ids:
                from models.data_models import Face as FaceModel2
                person_photo_q = session.query(FaceModel2.image_id).filter(
                    FaceModel2.person_id.in_(request.person_ids)
                ).distinct()
                person_candidate_ids = [row[0] for row in person_photo_q.all()]
                logger.info(f"[CLUSTERS] Person {request.person_ids} has {len(person_candidate_ids)} photos, CLIP will search within them")

            # Pass geo bounds to CLIP search when not searching globally
            geo_for_clip = None
            is_global = (request.min_lat <= -80 and request.max_lat >= 80
                         and request.min_lon <= -170 and request.max_lon >= 170)
            if not is_global:
                geo_for_clip = {
                    'min_lat': request.min_lat,
                    'max_lat': request.max_lat,
                    'min_lon': request.min_lon,
                    'max_lon': request.max_lon,
                }
                logger.info(f"[CLUSTERS] CLIP search with geo filter: {geo_for_clip}")

            clip_ids = clip_search_image_ids(
                request.clip_query, top_k=500, threshold=0.01,
                rrf_cutoff=0.35,
                candidate_ids=person_candidate_ids,
                geo_filters=geo_for_clip
            )
            if clip_ids:
                filters.append(PhotoIndex.image_id.in_(clip_ids))
            else:
                return {"clusters": [], "total_clusters": 0, "total_photos": 0,
                        "grid_size": grid_size, "zoom": request.zoom, "clip_image_ids": []}

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
                "longitude >= :min_lon", "longitude <= :max_lon",
                "NOT is_hidden"
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
                format_list = [f.lower().lstrip('.') for f in request.formats
                               if f.lower().lstrip('.') in ALLOWED_FORMATS]
                if format_list:
                    placeholders = ", ".join(f":fmt_{i}" for i in range(len(format_list)))
                    where_parts.append(f"LOWER(file_format) IN ({placeholders})")
                    for i, fmt in enumerate(format_list):
                        params[f"fmt_{i}"] = fmt
            if request.person_ids:
                pid_placeholders = ", ".join(f":pid_{i}" for i in range(len(request.person_ids)))
                if request.person_mode == "and" and len(request.person_ids) > 1:
                    where_parts.append(f"image_id IN (SELECT image_id FROM faces WHERE person_id IN ({pid_placeholders}) GROUP BY image_id HAVING COUNT(DISTINCT person_id) = {len(request.person_ids)})")
                else:
                    where_parts.append(f"image_id IN (SELECT image_id FROM faces WHERE person_id IN ({pid_placeholders}))")
                for i, pid in enumerate(request.person_ids):
                    params[f"pid_{i}"] = pid
            if clip_ids:
                # Reuse clip_ids from CLIP search or cached IDs
                clip_id_list = ",".join(str(int(i)) for i in clip_ids[:2000])
                where_parts.append(f"image_id IN ({clip_id_list})")

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

        result = {
            "clusters": clusters,
            "total_clusters": len(clusters),
            "total_photos": sum(c.count for c in clusters),
            "grid_size": grid_size,
            "zoom": request.zoom
        }
        # Return CLIP image IDs for frontend caching (only on first search)
        if clip_ids is not None and request.clip_query and not request.clip_image_ids:
            result["clip_image_ids"] = clip_ids
        logger.info(f"[CLUSTERS] → {len(clusters)} clusters, {sum(c.count for c in clusters)} photos"
                     f"{f', clip_ids={len(clip_ids)}' if clip_ids else ''}")
        return result
    finally:
        session.close()


@app.get("/map/photos")
async def get_map_photos(
    fastapi_request: Request,
    min_lat: float = Query(..., description="Минимальная широта"),
    max_lat: float = Query(..., description="Максимальная широта"),
    min_lon: float = Query(..., description="Минимальная долгота"),
    max_lon: float = Query(..., description="Максимальная долгота"),
    date_from: Optional[str] = Query(None, description="Дата от (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Дата до (YYYY-MM-DD)"),
    formats: Optional[str] = Query(None, description="Форматы файлов через запятую (jpg,heic)"),
    person_ids: Optional[str] = Query(None, description="ID персон через запятую"),
    person_mode: str = Query("or", description="Режим фильтра персон: or/and"),
    clip_query: Optional[str] = Query(None, description="CLIP текстовый поиск"),
    clip_image_ids: Optional[str] = Query(None, description="Cached CLIP image IDs (comma-separated)"),
    tag_ids: Optional[str] = Query(None, description="Фильтр по тегам через запятую (AND логика)"),
    exclude_tag_ids: Optional[str] = Query(None, description="Исключить фото с этими тегами (OR логика)"),
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
    from sqlalchemy import and_, func, text
    from datetime import datetime

    is_admin = getattr(fastapi_request.state, "is_admin", True)
    session = db_manager.get_session()
    try:
        # Фильтры
        filters = [
            PhotoIndex.latitude != None,
            PhotoIndex.longitude != None,
            PhotoIndex.latitude >= min_lat,
            PhotoIndex.latitude <= max_lat,
            PhotoIndex.longitude >= min_lon,
            PhotoIndex.longitude <= max_lon,
        ]
        if not is_admin:
            filters.append(PhotoIndex.is_hidden == False)  # noqa: E712

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

        # Фильтр по персонам
        if person_ids:
            from models.data_models import Face as FaceModel
            pid_list = [int(p.strip()) for p in person_ids.split(',') if p.strip()]
            if pid_list:
                if person_mode == "and" and len(pid_list) > 1:
                    pids = ','.join(str(p) for p in pid_list)
                    and_subq = text(f"""
                        SELECT image_id FROM faces
                        WHERE person_id IN ({pids})
                        GROUP BY image_id
                        HAVING COUNT(DISTINCT person_id) = {len(pid_list)}
                    """)
                    filters.append(PhotoIndex.image_id.in_(and_subq))
                else:
                    person_photo_subq = session.query(FaceModel.image_id).filter(
                        FaceModel.person_id.in_(pid_list)
                    )
                    filters.append(PhotoIndex.image_id.in_(person_photo_subq))

        # Фильтр по тегам (AND логика — фото должно содержать все указанные теги)
        if tag_ids:
            from models.data_models import PhotoTag as PhotoTagModel
            tid_list = [int(t.strip()) for t in tag_ids.split(',') if t.strip()]
            for tid in tid_list:
                filters.append(
                    PhotoIndex.image_id.in_(
                        session.query(PhotoTagModel.image_id).filter(PhotoTagModel.tag_id == tid)
                    )
                )

        # Исключить фото с любым из этих тегов (OR логика)
        if exclude_tag_ids:
            from models.data_models import PhotoTag as PhotoTagExcGeo
            etid_list = [int(t.strip()) for t in exclude_tag_ids.split(',') if t.strip()]
            if etid_list:
                exc_subq = session.query(PhotoTagExcGeo.image_id).filter(
                    PhotoTagExcGeo.tag_id.in_(etid_list)
                )
                filters.append(~PhotoIndex.image_id.in_(exc_subq))

        # Фильтр по CLIP текстовому поиску
        clip_match_ids = None
        if clip_image_ids:
            # Use cached IDs from frontend
            clip_match_ids = [int(x.strip()) for x in clip_image_ids.split(',') if x.strip()]
            if clip_match_ids:
                filters.append(PhotoIndex.image_id.in_(clip_match_ids))
        elif clip_query:
            # When person filter active: CLIP searches within person's photos only
            person_candidate_ids = None
            if person_ids:
                from models.data_models import Face as FaceModel3
                pid_list2 = [int(p.strip()) for p in person_ids.split(',') if p.strip()]
                if pid_list2:
                    person_photo_q = session.query(FaceModel3.image_id).filter(
                        FaceModel3.person_id.in_(pid_list2)
                    ).distinct()
                    person_candidate_ids = [row[0] for row in person_photo_q.all()]
            clip_match_ids = clip_search_image_ids(
                clip_query, top_k=500, threshold=0.01,
                rrf_cutoff=0.35,
                candidate_ids=person_candidate_ids
            )
            if clip_match_ids:
                filters.append(PhotoIndex.image_id.in_(clip_match_ids))
            else:
                return {"photos": [], "total": 0, "limit": limit, "offset": offset, "has_more": False}

        # Общее количество
        total_query = session.query(func.count(PhotoIndex.image_id)).filter(and_(*filters))
        total = total_query.scalar()

        # Получить фото с пагинацией
        query = session.query(PhotoIndex).filter(and_(*filters))
        if clip_match_ids:
            # Sort by CLIP relevance: preserve RRF rank order
            # First get IDs that exist in current viewport (intersection of clip_match_ids + geo filters)
            visible_ids_set = set(
                row[0] for row in session.query(PhotoIndex.image_id).filter(and_(*filters)).all()
            )
            # Filter clip_match_ids to only those visible, preserving rank order
            ranked_visible = [cid for cid in clip_match_ids if cid in visible_ids_set]
            # Paginate from the ranked visible list
            page_ids = ranked_visible[offset:offset + limit]
            total = len(ranked_visible)  # Override total with filtered count
            if page_ids:
                order_case = text("CASE " + " ".join(
                    f"WHEN image_id = {img_id} THEN {i}" for i, img_id in enumerate(page_ids)
                ) + " END")
                query = session.query(PhotoIndex).filter(
                    and_(*filters, PhotoIndex.image_id.in_(page_ids))
                ).order_by(order_case)
            else:
                query = query.limit(0)
        else:
            query = query.order_by(
                PhotoIndex.photo_date.desc().nullslast(),
                PhotoIndex.image_id.desc()  # deterministic tiebreaker — prevents duplicates across pages
            )
            query = query.offset(offset).limit(limit)

        photo_rows = query.all()
        image_ids = [p.image_id for p in photo_rows]
        tags_map = _batch_load_tags(session, image_ids)

        photos = []
        for photo in photo_rows:
            exif = photo.exif_data if isinstance(photo.exif_data, dict) else {}
            photos.append(MapPhotoItem(
                image_id=photo.image_id,
                latitude=photo.latitude,
                longitude=photo.longitude,
                photo_date=photo.photo_date.isoformat() if photo.photo_date else None,
                file_format=photo.file_format,
                file_name=photo.file_name,
                rotation=exif.get("UserRotation", 0) or 0,
                tags=tags_map.get(photo.image_id) or None,
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
    fastapi_request: Request,
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
                  {_build_hidden_filter_sql(getattr(fastapi_request.state, 'is_admin', True))}
                  {_build_tag_filter_sql(request.tag_ids)}
                  {_build_tag_exclude_filter_sql(request.exclude_tag_ids)}
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


@app.post("/photo/{image_id}/faces/reindex")
async def reindex_photo_faces(
    image_id: int,
    threshold: float = Query(0.6, ge=0.3, le=0.95, description="Порог авто-привязки лиц"),
    det_thresh: float = Query(0.45, ge=0.05, le=0.8, description="Порог детекции лиц (ниже = больше лиц)"),
    hd: bool = Query(False, description="HD детекция 1280px (лучше для маленьких/дальних лиц, медленнее)")
):
    """Переиндексировать лица на одном фото:
    1. Удаляет ВСЕ лица для этого фото из таблицы faces (назначенные и нет).
    2. Сбрасывает faces_indexed=0 для этого фото.
    3. Запускает детекцию лиц заново с указанным порогом det_thresh (и опционально 1280px).
    4. Запускает авто-привязку найденных лиц к известным персонам с порогом threshold.
    Синхронный — возвращает результат сразу.
    """
    if not HAS_FACE_DETECTOR:
        raise HTTPException(status_code=503, detail="Face detection не доступен")
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex, Face as FaceModel

    session = db_manager.get_session()
    try:
        photo = session.query(
            PhotoIndex.image_id, PhotoIndex.file_path, PhotoIndex.width, PhotoIndex.height,
            PhotoIndex.exif_data
        ).filter(PhotoIndex.image_id == image_id).first()
        if not photo:
            raise HTTPException(status_code=404, detail=f"Photo {image_id} not found")
        file_path = photo.file_path
        user_rotation = (photo.exif_data or {}).get("UserRotation", 0) if isinstance(photo.exif_data, dict) else 0

        # 1. Delete ALL existing faces for this photo (assigned + unassigned)
        deleted = session.query(FaceModel).filter(FaceModel.image_id == image_id).delete(synchronize_session=False)
        # 2. Reset faces_indexed flag
        session.execute(
            text("UPDATE photo_index SET faces_indexed = 0 WHERE image_id = :id"),
            {"id": image_id}
        )
        session.commit()
        logger.info(f"Photo {image_id}: deleted {deleted} faces, reset faces_indexed")
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

    # 3. Re-detect faces with custom detection threshold and optional HD resolution
    try:
        indexer = get_face_indexer()
        det_size = (1280, 1280) if hd else None

        # If user rotation is applied, pre-load the image with rotation so detection
        # sees the same orientation as the user (bboxes stored relative to rotated dims)
        image_data = None
        if user_rotation:
            import numpy as np
            pil_img = load_image_any_format(file_path, fast_mode=False)
            pil_img = _apply_user_rotation(pil_img, user_rotation)
            image_data = np.array(pil_img.convert("RGB"))

        new_face_ids = indexer.index_image(image_id, file_path, min_det_score=det_thresh, det_size=det_size, image_data=image_data)
        logger.info(f"Photo {image_id}: detected {len(new_face_ids)} faces with det_thresh={det_thresh}, hd={hd}")
    except Exception as e:
        logger.error(f"Face detection failed for photo {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {e}")

    # 4. Auto-assign faces to known persons
    try:
        result = indexer.auto_assign_faces_for_photo(image_id, threshold)
    except Exception as e:
        logger.warning(f"Auto-assign failed for photo {image_id}: {e}")
        result = {"assigned": 0, "total_faces": len(new_face_ids), "faces": indexer.get_faces_for_photo(image_id)}

    session2 = db_manager.get_session()
    try:
        photo_dim = session2.query(PhotoIndex.width, PhotoIndex.height).filter(PhotoIndex.image_id == image_id).first()
        original_width = photo_dim.width if photo_dim else None
        original_height = photo_dim.height if photo_dim else None
    finally:
        session2.close()

    return {
        "image_id": image_id,
        "deleted_faces": deleted,
        "detected_faces": len(new_face_ids),
        "assigned": result["assigned"],
        "total_faces": result["total_faces"],
        "faces": result["faces"],
        "original_width": original_width,
        "original_height": original_height,
    }


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

        photo = session.query(
            PhotoIndex.file_path, PhotoIndex.width, PhotoIndex.height, PhotoIndex.exif_data
        ).filter(
            PhotoIndex.image_id == face.image_id
        ).first()
        if not photo:
            raise HTTPException(status_code=404, detail="Фото не найдено")

        file_path = photo.file_path
        exif = photo.exif_data if isinstance(photo.exif_data, dict) else {}
        photo_rotation = exif.get("UserRotation", 0) or 0

        # Load image and crop face
        # fast_mode may load embedded JPEG (smaller than original for RAW)
        img = load_image_any_format(file_path, fast_mode=True)
        if photo_rotation:
            img = _apply_user_rotation(img, photo_rotation)

        # Scale bbox to loaded image dimensions
        # bbox in DB is relative to rotated image size (photo.width x photo.height)
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


# ==================== Auto-Assign All Persons ====================

_auto_assign_all_state = {
    "running": False,
    "stop_requested": False,
    "total_persons": 0,
    "processed_persons": 0,
    "total_assigned": 0,
    "total_candidates": 0,
    "current_person": None,
    "error": None,
}


@app.post("/persons/auto-assign-all")
async def auto_assign_all_persons(
    background_tasks: BackgroundTasks,
    threshold: float = Query(0.5, ge=0.3, le=0.95, description="Порог сходства для авто-привязки")
):
    """
    Автоматически привязать неназначенные лица ко всем персонам в БД.
    Фоновая задача — итерирует по всем персонам и вызывает auto_assign_faces для каждой.
    Прогресс: GET /persons/auto-assign-all/status
    """
    if not HAS_FACE_DETECTOR or not person_service:
        raise HTTPException(status_code=503, detail="Person service не доступен")

    if _auto_assign_all_state["running"]:
        raise HTTPException(status_code=409, detail="Auto-assign already running")

    def _run(thresh: float):
        try:
            _auto_assign_all_state["running"] = True
            _auto_assign_all_state["stop_requested"] = False
            _auto_assign_all_state["error"] = None
            _auto_assign_all_state["processed_persons"] = 0
            _auto_assign_all_state["total_assigned"] = 0
            _auto_assign_all_state["total_candidates"] = 0
            _auto_assign_all_state["current_person"] = None

            # Получить всех персон через сырой SQL (быстро, без лишних join)
            session = db_manager.get_session()
            try:
                rows = session.execute(
                    text("SELECT person_id, name FROM person ORDER BY person_id")
                ).fetchall()
                person_list = [(r[0], r[1]) for r in rows]
            finally:
                session.close()

            _auto_assign_all_state["total_persons"] = len(person_list)
            logger.info(f"Auto-assign all: {len(person_list)} персон, threshold={thresh}")

            for person_id, name in person_list:
                if _auto_assign_all_state["stop_requested"]:
                    logger.info("Auto-assign all: остановлен по запросу")
                    break

                _auto_assign_all_state["current_person"] = name
                try:
                    result = person_service.auto_assign_faces(person_id, threshold=thresh)
                    _auto_assign_all_state["total_assigned"] += result.get("assigned", 0)
                    _auto_assign_all_state["total_candidates"] += result.get("candidates", 0)
                except Exception as e:
                    logger.warning(f"Auto-assign failed для персоны {person_id} ({name}): {e}")

                _auto_assign_all_state["processed_persons"] += 1

            _auto_assign_all_state["current_person"] = None
            logger.info(
                f"Auto-assign all завершён: {_auto_assign_all_state['processed_persons']} персон, "
                f"назначено={_auto_assign_all_state['total_assigned']}, "
                f"кандидатов={_auto_assign_all_state['total_candidates']}"
            )

        except Exception as e:
            logger.error(f"Auto-assign all error: {e}", exc_info=True)
            _auto_assign_all_state["error"] = str(e)
        finally:
            _auto_assign_all_state["running"] = False

    background_tasks.add_task(_run, threshold)
    return {"status": "started", "message": "GET /persons/auto-assign-all/status for progress"}


@app.get("/persons/auto-assign-all/status")
async def auto_assign_all_status():
    """Статус фоновой авто-привязки лиц ко всем персонам."""
    state = dict(_auto_assign_all_state)
    total = state.get("total_persons", 0)
    processed = state.get("processed_persons", 0)
    state["percentage"] = round(processed / total * 100, 1) if total > 0 else 0
    return state


@app.post("/persons/auto-assign-all/stop")
async def stop_auto_assign_all():
    """Остановить авто-привязку лиц (текущий person завершится, остальные отменяются)."""
    if not _auto_assign_all_state["running"]:
        raise HTTPException(status_code=409, detail="Auto-assign не запущен")
    _auto_assign_all_state["stop_requested"] = True
    return {"status": "stopping", "message": "Остановится после текущей персоны"}


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


def _check_album_access(album: dict, request: Request, write: bool = False) -> None:
    """Проверяет право доступа к альбому. Бросает HTTPException если нет прав."""
    if not album:
        raise HTTPException(status_code=404, detail="Альбом не найден")
    user_id = getattr(request.state, "user_id", 1)
    is_admin = getattr(request.state, "is_admin", True)
    if write:
        if not is_admin and album.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Нет прав на изменение этого альбома")
    else:
        # Чтение: свой альбом или публичный или admin
        if not is_admin and album.get("user_id") != user_id and not album.get("is_public"):
            raise HTTPException(status_code=403, detail="Нет доступа к этому альбому")


@app.get("/albums")
async def list_albums(
    request: Request,
    search: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Список альбомов пользователя"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    user_id = getattr(request.state, "user_id", 1)
    is_admin = getattr(request.state, "is_admin", True)
    # Admin видит все альбомы; обычный пользователь — только свои + публичные
    effective_user_id = None if is_admin else user_id

    albums = album_service.list_albums(
        user_id=effective_user_id, search=search, limit=limit, offset=offset
    )
    return {"albums": albums, "count": len(albums), "limit": limit, "offset": offset}


@app.post("/albums")
async def create_album(
    request_body: AlbumCreateRequest,
    request: Request,
):
    """Создать новый альбом"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    user_id = getattr(request.state, "user_id", 1)
    is_admin = getattr(request.state, "is_admin", True)
    # Обычный пользователь не может создавать публичные альбомы
    is_public = request_body.is_public if is_admin else False

    try:
        album_id = album_service.create_album(
            user_id=user_id,
            title=request_body.title,
            description=request_body.description,
            is_public=is_public
        )
        return {"album_id": album_id, "title": request_body.title, "message": "Альбом создан"}
    except Exception as e:
        logger.error(f"Ошибка создания альбома: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/albums/{album_id}")
async def get_album(album_id: int, request: Request):
    """Получить информацию об альбоме"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    album = album_service.get_album(album_id)
    _check_album_access(album, request, write=False)
    return album


@app.put("/albums/{album_id}")
async def update_album(album_id: int, request_body: AlbumUpdateRequest, request: Request):
    """Обновить альбом"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    album = album_service.get_album(album_id)
    _check_album_access(album, request, write=True)

    is_admin = getattr(request.state, "is_admin", True)
    # Не-admin не может сделать альбом публичным
    is_public = request_body.is_public if is_admin else False

    success = album_service.update_album(
        album_id=album_id,
        title=request_body.title,
        description=request_body.description,
        cover_image_id=request_body.cover_image_id,
        is_public=is_public
    )
    if not success:
        raise HTTPException(status_code=404, detail="Альбом не найден")
    return {"status": "updated", "album_id": album_id}


@app.delete("/albums/{album_id}")
async def delete_album(album_id: int, request: Request):
    """Удалить альбом"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    album = album_service.get_album(album_id)
    _check_album_access(album, request, write=True)

    success = album_service.delete_album(album_id)
    if not success:
        raise HTTPException(status_code=404, detail="Альбом не найден")
    return {"status": "deleted", "album_id": album_id}


@app.get("/albums/{album_id}/photos")
async def get_album_photos(
    album_id: int,
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Получить фотографии в альбоме"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    # Check album exists and access
    album = album_service.get_album(album_id)
    _check_album_access(album, request, write=False)
    if not album:
        raise HTTPException(status_code=404, detail="Альбом не найден")

    # Публичные альбомы для не-администраторов — скрывать фото с системными тегами
    is_admin = getattr(request.state, "is_admin", True)
    apply_hidden = album.get("is_public", False) and not is_admin

    photos, total = album_service.get_album_photos(album_id, limit, offset, apply_hidden_filter=apply_hidden)
    return {
        "photos": photos,
        "total": total,
        "album_id": album_id,
        "album_title": album["title"],
        "limit": limit,
        "offset": offset
    }


@app.post("/albums/{album_id}/photos")
async def add_photos_to_album(album_id: int, request_body: AlbumAddPhotosRequest, request: Request):
    """Добавить фотографии в альбом"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    album = album_service.get_album(album_id)
    _check_album_access(album, request, write=True)

    try:
        result = album_service.add_photos(album_id, request_body.image_ids)
        return {"album_id": album_id, **result}
    except Exception as e:
        logger.error(f"Ошибка добавления фото в альбом: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/albums/{album_id}/photos")
async def remove_photos_from_album(album_id: int, request_body: AlbumRemovePhotosRequest, request: Request):
    """Удалить фотографии из альбома"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    album = album_service.get_album(album_id)
    _check_album_access(album, request, write=True)

    try:
        removed = album_service.remove_photos(album_id, request_body.image_ids)
        return {"album_id": album_id, "removed": removed}
    except Exception as e:
        logger.error(f"Ошибка удаления фото из альбома: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/albums/{album_id}/cover/{image_id}")
async def set_album_cover(album_id: int, image_id: int, request: Request):
    """Установить обложку альбома"""
    if not album_service:
        raise HTTPException(status_code=503, detail="Album service не доступен")

    album = album_service.get_album(album_id)
    _check_album_access(album, request, write=True)

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
            "has_more": offset + limit < total,
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

    # Если очередь содержит CLIP задачи — сканируем файловую систему ОДИН РАЗ
    # и передаём готовый список каждой модели. Так избегаем N медленных сканирований
    # при N моделях в очереди.
    clip_tasks = [t for t in queue if t.startswith("clip:")]
    discovered_files: Optional[List[str]] = None
    if clip_tasks:
        try:
            from services.indexer import IndexingService as _IS
            _tmp = _IS(clip_embedder=clip_embedder or next(iter(clip_embedders.values()), None))
            logger.info("Index All: сканирование файловой системы (один раз)...")
            discovered_files = _tmp.fast_scan_files(settings.PHOTO_STORAGE_PATH)
            logger.info(f"Index All: обнаружено {len(discovered_files)} файлов на диске")
        except Exception as _e:
            logger.warning(f"Index All: сканирование не удалось, продолжаю без списка файлов: {_e}")
            discovered_files = None

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
                    # Передаём уже просканированный список — ФС не сканируется повторно
                    _run_reindex(model_name, file_list=discovered_files)
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

                    def on_progress(computed, failed, total, speed, eta):
                        _phash_reindex_state["computed"] = computed
                        _phash_reindex_state["failed"] = failed
                        _phash_reindex_state["total"] = total
                        _phash_reindex_state["speed"] = round(speed, 1)
                        _phash_reindex_state["eta_seconds"] = int(eta)

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



# ==================== GPU Stats ====================

@app.get("/admin/gpu/stats")
async def get_gpu_stats():
    """Статистика GPU памяти: всего, занято, свободно, список загруженных моделей с памятью."""
    import subprocess

    result: dict = {
        "cuda_available": False,
        "device_name": None,
        "total_memory_gb": None,
        "used_memory_gb": None,
        "free_memory_gb": None,
        "utilization_pct": None,
        "temperature_c": None,
        "pytorch_allocated_gb": None,
        "pytorch_reserved_gb": None,
        "models": [],
        "face_model": None,
    }

    # PyTorch CUDA info
    try:
        import torch
        if torch.cuda.is_available():
            result["cuda_available"] = True
            result["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            result["total_memory_gb"] = round(props.total_memory / 1024**3, 2)
            result["pytorch_allocated_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
            result["pytorch_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1024**3, 2)
    except Exception:
        pass

    # nvidia-smi for used/free/utilization/temperature
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        parts = [p.strip() for p in smi_out.split(",")]
        if len(parts) >= 4:
            result["used_memory_gb"] = round(int(parts[0]) / 1024, 2)
            result["free_memory_gb"] = round(int(parts[1]) / 1024, 2)
            result["utilization_pct"] = int(parts[2])
            result["temperature_c"] = int(parts[3])
    except Exception:
        # Fallback: estimate from PyTorch if nvidia-smi unavailable
        if result["total_memory_gb"] and result["pytorch_allocated_gb"]:
            result["used_memory_gb"] = result["pytorch_allocated_gb"]
            result["free_memory_gb"] = round(result["total_memory_gb"] - result["pytorch_allocated_gb"], 2)

    # Loaded CLIP models with per-model memory
    for model_name, embedder in clip_embedders.items():
        result["models"].append({
            "name": model_name,
            "device": embedder.device,
            "embedding_dim": embedder.embedding_dim,
            "gpu_memory_gb": embedder.gpu_memory_gb,
        })

    # InsightFace model (if loaded)
    if face_embedder is not None:
        result["face_model"] = {
            "name": "InsightFace buffalo_l",
            "loaded": True,
        }

    return result


# ==================== Model Load / Unload Management ====================

@app.get("/admin/models/status")
async def get_models_status():
    """Статус загрузки всех CLIP моделей: loaded/unloaded, память GPU, устройство."""
    from models.data_models import CLIP_MODEL_COLUMNS, CLIP_MODEL_DIMS
    models = []
    for model_name in CLIP_MODEL_COLUMNS:
        embedder = clip_embedders.get(model_name)
        models.append({
            "name": model_name,
            "loaded": embedder is not None,
            "is_default": model_name == settings.CLIP_MODEL,
            "embedding_dim": CLIP_MODEL_DIMS.get(model_name),
            "gpu_memory_gb": embedder.gpu_memory_gb if embedder else None,
            "device": embedder.device if embedder else None,
        })
    return {"models": models}


class ModelActionRequest(BaseModel):
    model: str


@app.post("/admin/models/warm")
async def warm_model(request: ModelActionRequest):
    """Загрузить CLIP модель в память GPU (если ещё не загружена)."""
    global clip_embedder, clip_embedders
    from models.data_models import CLIP_MODEL_COLUMNS
    if request.model not in CLIP_MODEL_COLUMNS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
    if request.model in clip_embedders:
        return {"status": "already_loaded", "model": request.model}
    try:
        embedder = CLIPEmbedder(request.model, settings.CLIP_DEVICE)
        clip_embedders[request.model] = embedder
        # Restore default reference if this is the default model
        if request.model == settings.CLIP_MODEL:
            clip_embedder = embedder
        return {"status": "loaded", "model": request.model, "gpu_memory_gb": embedder.gpu_memory_gb}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/models/unload")
async def unload_model(request: ModelActionRequest):
    """Выгрузить CLIP модель из памяти GPU."""
    from models.data_models import CLIP_MODEL_COLUMNS
    if request.model not in CLIP_MODEL_COLUMNS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
    unloaded = _unload_clip_model(request.model)
    return {"status": "unloaded" if unloaded else "not_loaded", "model": request.model}


# ==================== Failed Files Management ====================

@app.get("/admin/failed-files")
async def get_failed_files(limit: int = Query(500, ge=1, le=5000)):
    """Список файлов с ошибкой индексации (index_failed=True)."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    from models.data_models import PhotoIndex

    session = db_manager.get_session()
    try:
        rows = session.query(
            PhotoIndex.image_id, PhotoIndex.file_path, PhotoIndex.fail_reason
        ).filter(PhotoIndex.index_failed == True).order_by(PhotoIndex.file_path).limit(limit).all()

        total = session.query(PhotoIndex).filter(PhotoIndex.index_failed == True).count()

        return {
            "total": total,
            "returned": len(rows),
            "files": [
                {"image_id": r.image_id, "file_path": r.file_path, "fail_reason": r.fail_reason}
                for r in rows
            ]
        }
    finally:
        session.close()


@app.post("/admin/failed-files/reset")
async def reset_failed_files(image_ids: Optional[List[int]] = None):
    """Сбросить флаг index_failed (все или указанные image_ids) — позволяет переиндексировать."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    from models.data_models import PhotoIndex

    session = db_manager.get_session()
    try:
        q = session.query(PhotoIndex).filter(PhotoIndex.index_failed == True)
        if image_ids:
            q = q.filter(PhotoIndex.image_id.in_(image_ids))
        count = q.count()
        q.update({"index_failed": False, "fail_reason": None}, synchronize_session=False)
        session.commit()
        return {"status": "ok", "reset": count}
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


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
    """Trigger background scan if stats are stale (>300s). Never blocks."""
    import time, threading
    if _cache_stats["scanning"]:
        return  # scan already in progress
    if time.time() - _cache_stats["updated_at"] < 300:
        return  # fresh enough (5 min TTL — avoids noisy scans of 80k+ files every 60s)
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


# ==================== AI Assistant ====================

async def _optimize_clip_prompt(user_query: str, clip_model: str = None) -> dict:
    """
    Reusable function: optimize a user's natural language query for CLIP visual search.
    Calls Gemini to transform the query into an effective CLIP prompt.
    Returns: {"clip_prompt": str, "original_query": str}
    """
    if not settings.GEMINI_API_KEY:
        # Fallback: return as-is
        return {"clip_prompt": user_query, "original_query": user_query}

    model_name = clip_model or (clip_embedder.model_name if clip_embedder else settings.CLIP_MODEL)
    is_siglip = "siglip" in model_name.lower() or model_name == "SigLIP"

    system = f"""You are an expert Prompt Engineer for CLIP-based image retrieval systems specializing in {'SigLIP (multilingual)' if is_siglip else 'CLIP (English)'} embeddings.
Your task: convert the user's query into a highly descriptive, visually-oriented prompt that maximizes cosine similarity in CLIP embeddings.
Style: focus on lighting, textures, composition, colors, specific artistic styles, and scene details. Use concrete visual descriptors, avoid abstract metaphors.

CLIP model in use: {model_name}
{"This model (SigLIP) supports Russian and English natively. You can use Russian if the original query is in Russian, but English often gives better results for specific visual concepts." if is_siglip else "This model only works well with English. You MUST translate to English."}

OPTIMIZATION RULES:
1. Focus on VISUAL elements only — what the camera would capture: objects, scenes, colors, composition, lighting, actions
2. Remove abstract/non-visual concepts (emotions, dates, "my favorite", story context)
3. Remove person names — CLIP doesn't know who "Sasha" is. Replace with visual descriptions: "a person", "a woman", "a child", "two people"
4. Be concise: 5-15 words is optimal for CLIP
5. Use descriptive adjectives: "ancient stone temple ruins" not just "temple"
6. Describe the SCENE, not the request: "sunset over ocean beach with palm trees" not "find me sunset photos"
7. If the query mentions a specific place, describe its visual appearance: "tropical beach with turquoise water" not "beach in Thailand"

EXAMPLES:
"фото Саши на фоне храма в Камбодже" → "person standing near ancient stone temple ruins in tropical forest"
"дети играют на пляже" → "children playing on sandy beach near ocean waves"
"закат в горах" → "dramatic sunset over mountain peaks with orange sky"
"бабушка с внуками на даче" → "elderly woman with children in garden near wooden house"
"красивый вид с балкона отеля" → "scenic view from hotel balcony overlooking city or sea"
"еда в ресторане" → "food plates on restaurant table"
"котик спит на диване" → "cat sleeping on sofa"

Return ONLY the optimized prompt text, nothing else. No quotes, no explanation."""

    try:
        gemini_url = (
            f"https://generativelanguage.googleapis.com/v1beta"
            f"/models/{settings.GEMINI_MODEL}:generateContent"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_query}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "temperature": 0.5,
                "topP": 0.8,
                "maxOutputTokens": 2048,
            }
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(gemini_url, json=payload,
                                         headers={"x-goog-api-key": settings.GEMINI_API_KEY})
            response.raise_for_status()
            data = response.json()

        if data.get("candidates"):
            clip_prompt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Remove quotes if Gemini wrapped it
            clip_prompt = clip_prompt.strip('"\'')
            logger.info(f"CLIP prompt optimized: '{user_query}' → '{clip_prompt}'")
            return {"clip_prompt": clip_prompt, "original_query": user_query}
    except Exception as e:
        logger.warning(f"CLIP prompt optimization failed, using original: {e}")

    return {"clip_prompt": user_query, "original_query": user_query}


@app.post("/ai/clip-prompt")
async def optimize_clip_prompt(
    query: str = Body(..., embed=True),
    model: Optional[str] = Body(None, embed=True)
):
    """
    Optimize a natural language query for CLIP visual search.
    Reusable endpoint — used by map AI assistant and search page.
    """
    result = await _optimize_clip_prompt(query, model)
    return result


ALLOWED_AI_ACTIONS = {"set_bounds", "set_persons", "set_date_range", "set_formats", "clear_filters", "text_search"}


def _load_persons_for_ai() -> List[dict]:
    """Load persons list for AI assistant context (only those with at least one face)."""
    if not person_service:
        return []
    try:
        persons_data = person_service.list_persons(limit=500)
        return [
            {"person_id": p["person_id"], "name": p["name"]}
            for p in persons_data
            if p.get("face_count", 0) > 0
        ]
    except Exception as e:
        logger.warning(f"Could not load persons for AI context: {e}")
        return []


def _load_tags_for_ai() -> List[dict]:
    """Load all tags for AI assistant context."""
    if not db_manager:
        return []
    try:
        from models.data_models import Tag as TagModel
        session = db_manager.get_session()
        try:
            tags = session.query(TagModel).order_by(TagModel.tag_id).all()
            return [{"tag_id": t.tag_id, "name": t.name, "is_system": t.is_system} for t in tags]
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Could not load tags for AI context: {e}")
        return []


async def _call_gemini_api(
    messages: list,
    system_prompt: str,
    allowed_actions: set,
    request_message: str,
    conversation_history: list,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> dict:
    """
    Call Gemini API with retry on 429, parse JSON response, repair truncated JSON.
    Returns {"actions": [...], "message": str, "conversation_history": [...]}.
    Raises HTTPException on unrecoverable errors.
    """
    import asyncio

    gemini_url = (
        f"https://generativelanguage.googleapis.com/v1beta"
        f"/models/{settings.GEMINI_MODEL}:generateContent"
    )
    gemini_headers = {"x-goog-api-key": settings.GEMINI_API_KEY}
    payload = {
        "contents": messages,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.95,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        }
    }

    gemini_response = None
    last_error = None
    for attempt in range(3):
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(gemini_url, json=payload, headers=gemini_headers)
            if response.status_code == 429:
                wait = (attempt + 1) * 5  # 5, 10, 15 seconds
                logger.warning(f"Gemini API rate limited (429), retry {attempt+1}/3 in {wait}s")
                last_error = f"Rate limited: {response.text[:200]}"
                await asyncio.sleep(wait)
                continue
            response.raise_for_status()
            gemini_response = response.json()
            break

    if gemini_response is None:
        raise HTTPException(status_code=429, detail=f"Gemini API перегружен, попробуйте через минуту. {last_error or ''}")

    if not gemini_response.get("candidates"):
        logger.error(f"No candidates in Gemini response: {json.dumps(gemini_response)[:500]}")
        raise HTTPException(status_code=500, detail="Gemini не вернул ответ")

    candidate = gemini_response["candidates"][0]
    finish_reason = candidate.get("finishReason", "")

    if "content" not in candidate or "parts" not in candidate.get("content", {}):
        block_reason = gemini_response.get("promptFeedback", {}).get("blockReason", "")
        logger.error(f"Gemini candidate has no content. finishReason={finish_reason}, "
                     f"blockReason={block_reason}, candidate={json.dumps(candidate)[:300]}")
        detail = "Gemini не смог сгенерировать ответ"
        if block_reason:
            detail += f" (блокировка: {block_reason})"
        elif finish_reason:
            detail += f" (причина: {finish_reason})"
        raise HTTPException(status_code=500, detail=f"{detail}. Попробуйте переформулировать запрос.")

    raw_text = candidate["content"]["parts"][0]["text"].strip()

    if finish_reason not in ("STOP", ""):
        logger.warning(f"Gemini finishReason={finish_reason}, response may be truncated: {raw_text[:200]}")

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    def _parse_and_validate(text: str) -> dict:
        data = json.loads(text)
        actions = data.get("actions", [])
        actions = [a for a in actions if a.get("type") in allowed_actions]
        message = data.get("message", "")
        return actions, message

    try:
        actions, message = _parse_and_validate(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}\nRaw: {raw_text[:500]}")
        # Try to repair truncated JSON using a proper nesting stack
        try:
            repaired = raw_text
            # Walk char-by-char tracking open containers, ignoring string content
            stack = []
            in_string = False
            escape_next = False
            for ch in repaired:
                if escape_next:
                    escape_next = False
                    continue
                if in_string:
                    if ch == '\\':
                        escape_next = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch in ('{', '['):
                    stack.append(ch)
                elif ch in ('}', ']'):
                    if stack:
                        stack.pop()
            # If truncated mid-string, close the string first
            if in_string:
                repaired += '"'
            # Strip trailing comma/whitespace before closing
            stripped = repaired.rstrip()
            if stripped.endswith(','):
                repaired = stripped[:-1]
            # Close open containers in correct reverse order
            repaired += ''.join('}' if ch == '{' else ']' for ch in reversed(stack))
            actions, message = _parse_and_validate(repaired)
            message = message or "AI ответ был обрезан, но основные действия применены."
            logger.info(f"Repaired truncated JSON, extracted {len(actions)} actions")
        except Exception as repair_err:
            logger.error(f"JSON repair also failed: {repair_err}")
            raise HTTPException(status_code=500, detail="AI вернул некорректный ответ. Попробуйте переформулировать запрос.")

    updated_history = list(conversation_history) + [
        {"role": "user", "content": request_message},
        {"role": "assistant", "content": message}
    ]
    return {"actions": actions, "message": message, "conversation_history": updated_history}


def _build_ai_system_prompt(persons: List[dict], current_state: dict) -> str:
    """Build system prompt for Gemini with context and action schema."""
    person_list = "\n".join(
        [f"- ID: {p['person_id']}, Name: {p['name']}" for p in persons]
    ) if persons else "No persons in database."

    return f"""You are a smart map filter assistant for a photo archive application.
The user is looking at a world map of their geotagged photos and wants to filter them using natural language.

AVAILABLE PERSONS IN DATABASE:
{person_list}

CURRENT MAP STATE:
- Visible area: lat [{current_state.get('min_lat', '?')}, {current_state.get('max_lat', '?')}], lon [{current_state.get('min_lon', '?')}, {current_state.get('max_lon', '?')}]
- Date filters: from {current_state.get('date_from', 'none')} to {current_state.get('date_to', 'none')}
- Format filters: {current_state.get('formats', 'all')}
- Selected person IDs: {current_state.get('person_ids', [])}
- Zoom level: {current_state.get('zoom', '?')}

AVAILABLE FILE FORMATS: jpg, jpeg, heic, heif, png, nef, cr2, arw, dng, raf, orf, rw2

YOUR TASK:
Interpret the user's natural language query and return a JSON object with "actions" array and "message" string.

ACTION TYPES (use only these):
1. set_bounds — move/zoom the map to a location. You MUST geocode place names to GPS bounds yourself.
   {{"type": "set_bounds", "min_lat": float, "max_lat": float, "min_lon": float, "max_lon": float}}

2. set_persons — filter by person(s). Match names case-insensitively. Use person_ids from the list above.
   {{"type": "set_persons", "person_ids": [int], "mode": "and"|"or"}}
   - Use "and" when the user wants photos with ALL listed persons TOGETHER on the same photo (e.g. "Sasha with grandma", "Sasha and Alex together")
   - Use "or" when the user wants photos of ANY of the listed persons (e.g. "photos of Sasha or Alex", "show me the kids")
   - Default: "and" for 2+ persons (most natural interpretation), "or" for single person

3. set_date_range — filter by date. Use YYYY-MM-DD format.
   {{"type": "set_date_range", "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD"}}

4. set_formats — filter file types.
   {{"type": "set_formats", "formats": ["jpg", "heic", ...]}}

5. clear_filters — remove all filters, show everything.
   {{"type": "clear_filters"}}

6. text_search — semantic CLIP/SigLIP visual search. Use when the user describes WHAT is in the photo.
   {{"type": "text_search", "query": "original visual description", "clip_prompt": "optimized English prompt for CLIP"}}
   - "query": the visual description in the user's language (for display to user)
   - "clip_prompt": optimized English prompt for CLIP visual search (5-15 words, visual-only)
   - Replace person names with visual descriptions (person, woman, child, etc.) but KEEP their visual attributes
   - clip_prompt rules: focus on VISUAL elements only, remove non-visual concepts, be descriptive
   - Examples:
     - "Сашу в синей рубашке на фоне храма" → query: "человек в синей рубашке на фоне храма", clip_prompt: "person in blue shirt standing near ancient stone temple ruins"
     - "дети на пляже" → query: "дети играют на пляже", clip_prompt: "children playing on sandy beach near ocean waves"
     - "закат в горах" → query: "закат в горах", clip_prompt: "dramatic sunset over mountain peaks with orange sky"
   - Can be combined with set_bounds, set_persons, set_formats

RULES:
- Return ONLY valid JSON. No markdown, no explanation outside JSON.
- Multiple actions can be combined in one response.
- For locations, use your geographic knowledge for accurate GPS bounding boxes.
- If a person name is not found in the list, return empty person_ids and explain in message.
- Be concise but friendly in your messages (Russian language preferred).
- For "iPhone photos" or similar, use formats ["jpg", "jpeg", "heic", "heif"].
- IMPORTANT: If the user describes visual content (temple, beach, sunset, mountains, etc.), ALWAYS include text_search action.

EXAMPLES:

User: "найди Сашу в Камбодже"
{{"actions": [{{"type": "set_bounds", "min_lat": 10.0, "max_lat": 14.7, "min_lon": 102.3, "max_lon": 107.6}}, {{"type": "set_persons", "person_ids": [5], "mode": "or"}}], "message": "Показываю фото Саши в Камбодже"}}

User: "покажи Аэлиту с бабушкой Лидой"
{{"actions": [{{"type": "set_persons", "person_ids": [3, 7], "mode": "and"}}], "message": "Показываю фото где Аэлита и бабушка Лида вместе"}}

User: "покажи фото детей"
{{"actions": [{{"type": "set_persons", "person_ids": [3, 4], "mode": "or"}}], "message": "Показываю фото любого из детей"}}

User: "фото за лето 2024"
{{"actions": [{{"type": "set_date_range", "date_from": "2024-06-01", "date_to": "2024-08-31"}}], "message": "Фильтрую фото за лето 2024 (июнь-август)"}}

User: "только RAW файлы"
{{"actions": [{{"type": "set_formats", "formats": ["nef", "cr2", "arw", "dng", "raf", "orf", "rw2"]}}], "message": "Показываю только RAW файлы"}}

User: "фото Саши в Камбодже в синей рубашке на фоне храма"
{{"actions": [{{"type": "set_bounds", "min_lat": 10.0, "max_lat": 14.7, "min_lon": 102.3, "max_lon": 107.6}}, {{"type": "set_persons", "person_ids": [5], "mode": "or"}}, {{"type": "text_search", "query": "человек в синей рубашке на фоне храма", "clip_prompt": "person in blue shirt standing near ancient stone temple ruins in tropical forest"}}], "message": "Ищу фото Саши в синей рубашке у храма в Камбодже"}}

User: "закат на пляже"
{{"actions": [{{"type": "text_search", "query": "закат на пляже", "clip_prompt": "dramatic sunset over sandy ocean beach with warm orange sky"}}], "message": "Ищу фото заката на пляже"}}

User: "сбрось всё"
{{"actions": [{{"type": "clear_filters"}}], "message": "Все фильтры сброшены"}}"""


@app.post("/ai/assistant")
async def ai_assistant(request: AIAssistantRequest):
    """AI assistant — interprets natural language and returns structured filter commands."""
    if not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API key not configured (set GEMINI_API_KEY in .env)")

    try:
        persons = _load_persons_for_ai()
        system_prompt = _build_ai_system_prompt(persons, request.current_state)

        gemini_messages = []
        for msg in request.conversation_history:
            role = "user" if msg.get("role") == "user" else "model"
            gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
        gemini_messages.append({"role": "user", "parts": [{"text": request.message}]})

        return await _call_gemini_api(
            messages=gemini_messages,
            system_prompt=system_prompt,
            allowed_actions=ALLOWED_AI_ACTIONS,
            request_message=request.message,
            conversation_history=request.conversation_history,
            max_tokens=8192,
        )
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"Gemini API error: {e.response.status_code} {e.response.text[:300]}")
        if e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Gemini API перегружен. Попробуйте через минуту.")
        raise HTTPException(status_code=502, detail=f"Ошибка Gemini API: {e.response.status_code}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gemini API не ответил вовремя")
    except Exception as e:
        logger.error(f"AI assistant error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка AI ассистента: {str(e)}")


def _build_search_ai_system_prompt(persons: List[dict], current_state: dict, tags: List[dict] = None) -> str:
    """Build system prompt for Gemini search assistant (index.html)."""
    person_list = "\n".join(
        [f"- ID: {p['person_id']}, Name: {p['name']}" for p in persons]
    ) if persons else "No persons in database."

    tag_list = "\n".join(
        [f"- ID: {t['tag_id']}, Name: {t['name']}{' (system)' if t.get('is_system') else ''}" for t in (tags or [])]
    ) if tags else "No tags in database."

    return f"""You are a smart search assistant for a photo archive application.
The user wants to search their photo collection using natural language.
The search uses CLIP/SigLIP visual embeddings — multi-model Reciprocal Rank Fusion across all available models.

AVAILABLE PERSONS IN DATABASE:
{person_list}

AVAILABLE TAGS IN DATABASE:
{tag_list}

CURRENT SEARCH STATE:
- Active query: {current_state.get('query', 'none')}
- Geo bounds: {current_state.get('geo_bounds', 'none')}
- Date filters: from {current_state.get('date_from', 'none')} to {current_state.get('date_to', 'none')}
- Format filters: {current_state.get('formats', 'all')}
- Selected person IDs: {current_state.get('person_ids', [])}
- Required tag IDs: {current_state.get('tag_ids', [])}
- Excluded tag IDs: {current_state.get('exclude_tag_ids', [])}

AVAILABLE FILE FORMATS: jpg, jpeg, heic, heif, png, nef, cr2, arw, dng, raf, orf, rw2

YOUR TASK:
Interpret the user's natural language query and return a JSON object with "actions" array and "message" string.

ACTION TYPES (use only these):
1. text_search — semantic CLIP/SigLIP visual search. Use when the user describes WHAT is in the photo.
   {{"type": "text_search", "query": "original visual description", "clip_prompt": "optimized English prompt for CLIP", "tag_ids": [int], "exclude_tag_ids": [int]}}
   - "query": the visual description in the user's language (for display to user)
   - "clip_prompt": optimized English prompt for CLIP visual search (5-15 words, visual-only)
   - "tag_ids": optional list of tag IDs that photos MUST have (AND logic). Include only when user explicitly wants only photos WITH certain tags.
   - "exclude_tag_ids": optional list of tag IDs to exclude (OR logic — excludes photos with ANY of these tags). Include when user says "excluding", "без тега", "не помеченные как", etc.
   - Replace person names with visual descriptions (person, woman, child, etc.) but KEEP their visual attributes
   - clip_prompt rules: focus on VISUAL elements only, remove non-visual concepts, be descriptive
   - Examples:
     - "Сашу в синей рубашке на фоне храма" → query: "человек в синей рубашке на фоне храма", clip_prompt: "person in blue shirt standing near ancient stone temple ruins"
     - "дети на пляже" → query: "дети играют на пляже", clip_prompt: "children playing on sandy beach near ocean waves"
     - "закат в горах" → query: "закат в горах", clip_prompt: "dramatic sunset over mountain peaks with orange sky"

2. set_bounds — limit search to a geographic area. You MUST geocode place names to GPS bounds yourself.
   {{"type": "set_bounds", "min_lat": float, "max_lat": float, "min_lon": float, "max_lon": float}}
   - IMPORTANT: When the user mentions a city, country, region, or any geographic location, ALWAYS include set_bounds action.
   - Use approximate bounding box for the place (city: ~0.1-0.3 degrees, country: wider).
   - Examples:
     - "Бельско-Бяла" → min_lat: 49.78, max_lat: 49.88, min_lon: 19.00, max_lon: 19.10
     - "Крым" → min_lat: 44.3, max_lat: 46.2, min_lon: 32.5, max_lon: 36.7
     - "Таиланд" → min_lat: 5.6, max_lat: 20.5, min_lon: 97.3, max_lon: 105.7

3. set_persons — filter by person(s). Match names case-insensitively. Use person_ids from the list above.
   {{"type": "set_persons", "person_ids": [int], "mode": "and"|"or"}}
   - Use "and" when the user wants photos with ALL listed persons TOGETHER on the same photo
   - Use "or" when the user wants photos of ANY of the listed persons
   - Default: "and" for 2+ persons, "or" for single person

4. set_date_range — filter by date. Use YYYY-MM-DD format.
   {{"type": "set_date_range", "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD"}}
   - Can use date_from only, date_to only, or both
   - For "лето 2024" use date_from: "2024-06-01", date_to: "2024-08-31"
   - For "день рождения" or specific date events, use a narrow range (e.g. 3-5 days around the date)
   - IMPORTANT: When the user mentions a year or time period, ALWAYS include set_date_range action

5. set_formats — filter file types.
   {{"type": "set_formats", "formats": ["jpg", "heic", ...]}}

6. clear_filters — remove all filters, clear search.
   {{"type": "clear_filters"}}

RULES:
- Return ONLY valid JSON. No markdown, no explanation outside JSON.
- Multiple actions can be combined in one response.
- If a person name is not found in the list, return empty person_ids and explain in message.
- If a tag name is not found in the list, explain in message and skip the tag filter.
- Be concise but friendly in your messages (Russian language preferred).
- For "iPhone photos" or similar, use formats ["jpg", "jpeg", "heic", "heif"].
- IMPORTANT: If the user describes visual content, ALWAYS include text_search action.
- IMPORTANT: If the user mentions a year, season, month, or specific date — ALWAYS include set_date_range action.
- IMPORTANT: If the user mentions a city, country, region, or any geographic location — ALWAYS include set_bounds action. You must geocode the place name to approximate GPS bounding box coordinates.
- IMPORTANT: If the user says "исключая", "без тега", "кроме помеченных", "не [tag]" — add exclude_tag_ids to text_search action.
- IMPORTANT: If the user wants only photos WITH a specific tag — add tag_ids to text_search action.
- Can combine text_search with set_bounds, set_persons, set_date_range, set_formats, tag_ids, and exclude_tag_ids.

EXAMPLES:

User: "найди Сашу в синей рубашке на фоне храма"
{{"actions": [{{"type": "set_persons", "person_ids": [5], "mode": "or"}}, {{"type": "text_search", "query": "человек в синей рубашке на фоне храма", "clip_prompt": "person in blue shirt standing near ancient stone temple ruins in tropical forest"}}], "message": "Ищу фото Саши в синей рубашке у храма"}}

User: "закат на пляже"
{{"actions": [{{"type": "text_search", "query": "закат на пляже", "clip_prompt": "dramatic sunset over sandy ocean beach with warm orange sky"}}], "message": "Ищу фото заката на пляже"}}

User: "рыжий кот в Бельско-Бяла"
{{"actions": [{{"type": "set_bounds", "min_lat": 49.78, "max_lat": 49.88, "min_lon": 19.00, "max_lon": 19.10}}, {{"type": "text_search", "query": "рыжий кот", "clip_prompt": "orange ginger cat"}}], "message": "Ищу фото рыжего кота в районе Бельско-Бяла"}}

User: "покажи фото детей"
{{"actions": [{{"type": "set_persons", "person_ids": [3, 4], "mode": "or"}}], "message": "Показываю фото любого из детей"}}

User: "фото за лето 2024"
{{"actions": [{{"type": "set_date_range", "date_from": "2024-06-01", "date_to": "2024-08-31"}}], "message": "Фильтрую фото за лето 2024 (июнь-август)"}}

User: "день рождения Аэлиты 2022"
{{"actions": [{{"type": "set_persons", "person_ids": [3], "mode": "or"}}, {{"type": "set_date_range", "date_from": "2022-01-01", "date_to": "2022-12-31"}}, {{"type": "text_search", "query": "день рождения торт праздник", "clip_prompt": "birthday celebration with cake candles and party decorations"}}], "message": "Ищу фото дня рождения Аэлиты в 2022 году"}}

User: "фото из Камбоджи"
{{"actions": [{{"type": "set_bounds", "min_lat": 10.0, "max_lat": 14.7, "min_lon": 102.3, "max_lon": 107.6}}], "message": "Показываю фото из Камбоджи"}}

User: "только RAW файлы"
{{"actions": [{{"type": "set_formats", "formats": ["nef", "cr2", "arw", "dng", "raf", "orf", "rw2"]}}], "message": "Показываю только RAW файлы"}}

User: "скриншоты исключая trash"
{{"actions": [{{"type": "text_search", "query": "скриншот интерфейс", "clip_prompt": "screenshot computer interface application window", "exclude_tag_ids": [2]}}], "message": "Ищу скриншоты без тега 'trash'"}}

User: "найди все частные фото" (tag 'private' has ID 1)
{{"actions": [{{"type": "text_search", "query": "все фото", "clip_prompt": "", "tag_ids": [1]}}], "message": "Показываю фото с тегом 'private'"}}

User: "сбрось всё"
{{"actions": [{{"type": "clear_filters"}}], "message": "Все фильтры сброшены"}}"""


@app.post("/ai/search-assistant")
async def ai_search_assistant(request: AIAssistantRequest):
    """AI assistant for search page — interprets natural language and returns structured search commands."""
    if not settings.GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API key not configured (set GEMINI_API_KEY in .env)")

    try:
        persons = _load_persons_for_ai()
        tags = _load_tags_for_ai()
        system_prompt = _build_search_ai_system_prompt(persons, request.current_state, tags)

        gemini_messages = []
        for msg in request.conversation_history:
            role = "user" if msg.get("role") == "user" else "model"
            gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
        gemini_messages.append({"role": "user", "parts": [{"text": request.message}]})

        return await _call_gemini_api(
            messages=gemini_messages,
            system_prompt=system_prompt,
            allowed_actions=ALLOWED_AI_ACTIONS,
            request_message=request.message,
            conversation_history=request.conversation_history,
            max_tokens=8192,
        )
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"Gemini API error: {e.response.status_code} {e.response.text[:300]}")
        if e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Gemini API перегружен. Попробуйте через минуту.")
        raise HTTPException(status_code=502, detail=f"Ошибка Gemini API: {e.response.status_code}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gemini API не ответил вовремя")
    except Exception as e:
        logger.error(f"Search AI assistant error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка AI ассистента: {str(e)}")


@app.get("/timeline/photos")
async def get_timeline_photos(
    request: Request,
    limit: int = Query(50, ge=1, le=200, description="Количество фото за запрос"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
    date_from: Optional[str] = Query(None, description="Дата от (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Дата до (YYYY-MM-DD)"),
    include_hidden: bool = Query(False, description="Включить скрытые фото (только для админа)"),
):
    """
    Хронологическая лента фотографий (от новых к старым).
    Возвращает фото с photo_date DESC, image_id DESC.
    Используется страницей timeline.html.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import PhotoIndex
    from sqlalchemy import func
    from datetime import datetime

    is_admin = getattr(request.state, "is_admin", False)
    show_hidden = include_hidden and is_admin

    session = db_manager.get_session()
    try:
        filters = []

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

        from sqlalchemy import and_
        # Скрывать фото с системными тегами, если не админ с include_hidden
        if not show_hidden:
            filters.append(PhotoIndex.is_hidden == False)  # noqa: E712
        base_q = session.query(PhotoIndex)
        if filters:
            base_q = base_q.filter(and_(*filters))

        total = base_q.count()

        from sqlalchemy import nullslast, desc
        photos_q = base_q.order_by(
            nullslast(desc(PhotoIndex.photo_date)),
            desc(PhotoIndex.image_id)
        ).offset(offset).limit(limit).all()

        photos = []
        for p in photos_q:
            exif = p.exif_data if isinstance(p.exif_data, dict) else {}
            rotation = exif.get("UserRotation", 0) or 0
            photos.append({
                "image_id": p.image_id,
                "file_name": p.file_name or "",
                "file_format": p.file_format,
                "photo_date": p.photo_date.isoformat() if p.photo_date else None,
                "width": p.width or 0,
                "height": p.height or 0,
                "rotation": rotation,
                "file_size": p.file_size,
                "tags": [],
            })

        # Batch-load tags for all photos in one query
        if photos_q:
            from models.data_models import PhotoTag, Tag as TagModel
            image_ids_list = [p.image_id for p in photos_q]
            tag_rows = session.query(
                PhotoTag.image_id, TagModel.tag_id, TagModel.name,
                TagModel.is_system, TagModel.color
            ).join(TagModel, PhotoTag.tag_id == TagModel.tag_id).filter(
                PhotoTag.image_id.in_(image_ids_list)
            ).all()
            tags_map: dict = {}
            for row in tag_rows:
                tags_map.setdefault(row.image_id, []).append({
                    "tag_id": row.tag_id, "name": row.name,
                    "is_system": row.is_system, "color": row.color
                })
            for photo_dict in photos:
                photo_dict["tags"] = tags_map.get(photo_dict["image_id"], [])

        return {
            "photos": photos,
            "total": total,
            "has_more": offset + limit < total,
            "offset": offset,
            "limit": limit,
        }
    finally:
        session.close()


# ==================== Tags API ====================

# ── Tag helpers (bulk-optimized) ──

def _validate_tags(session, tag_ids: List[int], is_admin: bool) -> list:
    """Validate tags exist and check system-tag permissions. Returns list of Tag objects."""
    from models.data_models import Tag
    tags = session.query(Tag).filter(Tag.tag_id.in_(tag_ids)).all()
    if not tags:
        raise HTTPException(status_code=400, detail="Теги не найдены")
    for tag in tags:
        if tag.is_system and not is_admin:
            raise HTTPException(status_code=403, detail=f"Только администратор может управлять системным тегом '{tag.name}'")
    return tags


def _bulk_add_tags(session, image_ids: List[int], tag_ids: List[int]) -> int:
    """Bulk-insert photo_tag rows via ON CONFLICT DO NOTHING. Returns count of rows actually inserted."""
    from sqlalchemy import text as sa_text
    if not image_ids or not tag_ids:
        return 0
    # Generate VALUES list: (img1, tag1), (img1, tag2), ..., (img2, tag1), ...
    values = ', '.join(f'({int(img)}, {int(tid)})' for img in image_ids for tid in tag_ids)
    result = session.execute(sa_text(f"""
        INSERT INTO photo_tag (image_id, tag_id)
        VALUES {values}
        ON CONFLICT (image_id, tag_id) DO NOTHING
    """))
    return result.rowcount


def _bulk_remove_tags(session, image_ids: List[int], tag_ids: List[int]) -> int:
    """Bulk-delete photo_tag rows.  Returns count of rows deleted."""
    from sqlalchemy import text as sa_text
    if not image_ids or not tag_ids:
        return 0
    img_csv = ', '.join(str(int(i)) for i in image_ids)
    tag_csv = ', '.join(str(int(t)) for t in tag_ids)
    result = session.execute(sa_text(f"""
        DELETE FROM photo_tag
        WHERE image_id IN ({img_csv}) AND tag_id IN ({tag_csv})
    """))
    return result.rowcount


def _bulk_sync_is_hidden(session, image_ids: List[int]) -> None:
    """Recalculate is_hidden for a batch of photos in 1 query.

    Sets is_hidden = TRUE  if the photo has ≥1 system tag,
          is_hidden = FALSE otherwise.
    """
    from sqlalchemy import text as sa_text
    if not image_ids:
        return
    img_csv = ', '.join(str(int(i)) for i in image_ids)
    session.execute(sa_text(f"""
        UPDATE photo_index p
        SET is_hidden = EXISTS (
            SELECT 1 FROM photo_tag pt
            JOIN tag t ON pt.tag_id = t.tag_id
            WHERE pt.image_id = p.image_id AND t.is_system = TRUE
        )
        WHERE p.image_id IN ({img_csv})
    """))


def _sync_is_hidden(session, image_id: int) -> None:
    """Convenience wrapper for single-photo is_hidden sync."""
    _bulk_sync_is_hidden(session, [image_id])


@app.get("/tags")
async def list_tags(request: Request):
    """Получить список тегов.

    Администраторы видят все теги (включая системные).
    Обычные пользователи видят только пользовательские теги (is_system=FALSE).
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import Tag

    is_admin = getattr(request.state, "is_admin", True)
    session = db_manager.get_session()
    try:
        q = session.query(Tag)
        if not is_admin:
            q = q.filter(Tag.is_system == False)  # noqa: E712
        tags = q.order_by(Tag.is_system.desc(), Tag.name).all()
        return [{"tag_id": t.tag_id, "name": t.name, "is_system": t.is_system, "color": t.color}
                for t in tags]
    finally:
        session.close()


@app.post("/tags", status_code=201)
async def create_tag(body: CreateTagRequest, request: Request):
    """Создать новый пользовательский тег. Системные теги создаёт только администратор."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import Tag

    session = db_manager.get_session()
    try:
        name = body.name.strip().lower()
        if not name:
            raise HTTPException(status_code=400, detail="Название тега не может быть пустым")
        existing = session.query(Tag).filter(Tag.name == name).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Тег '{name}' уже существует")
        tag = Tag(name=name, color=body.color, is_system=False)
        session.add(tag)
        session.commit()
        return {"tag_id": tag.tag_id, "name": tag.name, "is_system": tag.is_system, "color": tag.color}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка создания тега: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.delete("/tags/{tag_id}")
async def delete_tag(tag_id: int, request: Request):
    """Удалить пользовательский тег. Системные теги удалить нельзя."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import Tag

    session = db_manager.get_session()
    try:
        tag = session.query(Tag).filter(Tag.tag_id == tag_id).first()
        if not tag:
            raise HTTPException(status_code=404, detail="Тег не найден")
        if tag.is_system:
            raise HTTPException(status_code=403, detail="Системные теги нельзя удалять")
        session.delete(tag)
        session.commit()
        return {"status": "deleted", "tag_id": tag_id}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка удаления тега: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.get("/photo/{image_id}/tags")
async def get_photo_tags(image_id: int, request: Request):
    """Получить теги фотографии.

    Обычные пользователи видят только не-системные теги.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import Tag, PhotoTag

    is_admin = getattr(request.state, "is_admin", True)
    session = db_manager.get_session()
    try:
        q = session.query(Tag).join(PhotoTag, PhotoTag.tag_id == Tag.tag_id).filter(
            PhotoTag.image_id == image_id
        )
        if not is_admin:
            q = q.filter(Tag.is_system == False)  # noqa: E712
        tags = q.order_by(Tag.is_system.desc(), Tag.name).all()
        return [{"tag_id": t.tag_id, "name": t.name, "is_system": t.is_system, "color": t.color}
                for t in tags]
    finally:
        session.close()


@app.post("/photo/{image_id}/tags", status_code=200)
async def add_photo_tags(image_id: int, body: PhotoTagsRequest, request: Request):
    """Добавить теги к фотографии.

    Системные теги может добавлять только администратор.
    Добавление системного тега устанавливает is_hidden=TRUE.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import Tag, PhotoTag

    is_admin = getattr(request.state, "is_admin", True)
    session = db_manager.get_session()
    try:
        _validate_tags(session, body.tag_ids, is_admin)
        added = _bulk_add_tags(session, [image_id], body.tag_ids)
        if added > 0:
            _bulk_sync_is_hidden(session, [image_id])
        session.commit()

        tags = session.query(Tag).join(PhotoTag, PhotoTag.tag_id == Tag.tag_id).filter(
            PhotoTag.image_id == image_id
        ).all()
        return {
            "added": added,
            "tags": [{"tag_id": t.tag_id, "name": t.name, "is_system": t.is_system, "color": t.color} for t in tags]
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка добавления тегов: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.delete("/photo/{image_id}/tags")
async def remove_photo_tags(image_id: int, body: PhotoTagsRequest, request: Request):
    """Убрать теги с фотографии.

    Системные теги может убирать только администратор.
    Если не осталось системных тегов — is_hidden сбрасывается в FALSE.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    from models.data_models import Tag, PhotoTag

    is_admin = getattr(request.state, "is_admin", True)
    session = db_manager.get_session()
    try:
        _validate_tags(session, body.tag_ids, is_admin)
        removed = _bulk_remove_tags(session, [image_id], body.tag_ids)
        if removed > 0:
            _bulk_sync_is_hidden(session, [image_id])
        session.commit()

        tags = session.query(Tag).join(PhotoTag, PhotoTag.tag_id == Tag.tag_id).filter(
            PhotoTag.image_id == image_id
        ).all()
        return {
            "removed": removed,
            "tags": [{"tag_id": t.tag_id, "name": t.name, "is_system": t.is_system, "color": t.color} for t in tags]
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка удаления тегов: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.post("/photos/tags/bulk")
async def bulk_tag_photos(body: BulkTagRequest, request: Request):
    """Пакетное тегирование/снятие тегов с нескольких фото.

    mode='add': добавить теги к фото (INSERT ON CONFLICT DO NOTHING)
    mode='remove': убрать теги с фото (bulk DELETE)
    Системные теги доступны только для администраторов.
    """
    if not db_manager:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")

    is_admin = getattr(request.state, "is_admin", True)
    if body.mode not in ("add", "remove"):
        raise HTTPException(status_code=400, detail="mode должен быть 'add' или 'remove'")

    session = db_manager.get_session()
    try:
        _validate_tags(session, body.tag_ids, is_admin)

        if body.mode == "add":
            affected = _bulk_add_tags(session, body.image_ids, body.tag_ids)
        else:
            affected = _bulk_remove_tags(session, body.image_ids, body.tag_ids)

        _bulk_sync_is_hidden(session, body.image_ids)
        session.commit()
        return {"affected_photos": len(body.image_ids), "affected_tags": affected, "mode": body.mode}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Ошибка пакетного тегирования: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse(url="/favicon.svg", status_code=301)


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
