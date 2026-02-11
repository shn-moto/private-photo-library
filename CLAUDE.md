# Smart Photo Indexing - Claude Context

## Overview

Сервис индексации домашнего фотоархива с семантическим поиском по текстовому описанию (SigLIP).

**Stack:** Python 3.11 + PyTorch 2.6 + HuggingFace Transformers + PostgreSQL/pgvector + FastAPI + Docker (GPU)

## Quick Start

```bash
# 1. БД (один раз)
psql -U dev -d smart_photo_index -f sql/init_db.sql

# 2. Сборка и запуск Docker
docker-compose build
docker-compose up -d db         # PostgreSQL + pgvector
docker-compose up -d api        # API + Web UI на :8000
docker-compose up -d bot        # Telegram бот (optional)

# 3. Установка утилит на хосте (Windows)
pip install httpx pywin32 python-dotenv

# 4. Запуск индексации (с хоста Windows)
python scripts/fast_reindex.py --model SigLIP

# 5. Web UI
http://localhost:8000/
```

## Host Setup (Windows)

Индексация запускается скриптом с хоста Windows, а не из Docker контейнера. Это позволяет использовать NTFS USN Journal для мгновенного обнаружения изменений.

### Требования на хосте

```bash
# Python зависимости для скрипта индексации
pip install httpx pywin32 python-dotenv

# Опционально: Everything SDK для еще более быстрого сканирования
# Скачать Everything с https://www.voidtools.com/ и запустить
```

### Скрипт индексации (fast_reindex.py)

```bash
# Первый запуск - полное сканирование + сохранение USN checkpoint
python scripts/fast_reindex.py --model SigLIP

# Последующие запуски - только изменения через USN Journal (~0 сек)
python scripts/fast_reindex.py --model SigLIP

# Принудительное полное сканирование
python scripts/fast_reindex.py --model SigLIP --full-scan

# Указать другую модель
python scripts/fast_reindex.py --model ViT-L/14
```

### Как работает fast_reindex.py

1. **USN Journal** — читает NTFS журнал изменений (мгновенно, ~0 сек)
2. **Детекция изменений** — определяет added/modified/deleted файлы
3. **Cleanup deleted** — автоматически удаляет записи из БД для удаленных файлов
4. **API /files/unindexed** — проверяет файлы без эмбеддингов в БД
5. **Gzip + Multipart** — отправляет список файлов в API (100k файлов = 0.4 MB)
6. **API /reindex/files** — индексация в Docker с GPU

### Fallback при ошибках

- Если USN Journal недоступен → os.scandir (~12 сек на 100k файлов)
- Если Everything запущен → Everything SDK (~1 сек на 100k файлов)
- Если индексация была прервана → автоматически доиндексирует из /files/unindexed

### Cleanup orphaned (опционально)

```bash
# Проверка всех файлов в БД на существование (медленно, для больших баз)
python scripts/fast_reindex.py --cleanup
```

USN Journal детектит удаление файлов автоматически, но можно запустить полную проверку вручную.

## Project Structure

```
smart_photo_indexing/
├── main.py                 # Entry point (indexer daemon)
├── config/
│   └── settings.py         # Pydantic settings (.env)
├── services/
│   ├── clip_embedder.py    # SigLIP/CLIP via HuggingFace transformers
│   ├── image_processor.py  # HEIC/JPG/PNG/RAW loading, EXIF
│   ├── indexer.py          # Orchestrates indexing pipeline (batch GPU, upsert)
│   ├── file_monitor.py     # File system scanning
│   ├── duplicate_finder.py # Duplicate detection & deletion (cosine similarity)
│   ├── phash_service.py    # Perceptual hash duplicate detection (256-bit DCT)
│   └── album_service.py    # Album CRUD + photo management
├── api/
│   ├── main.py             # FastAPI endpoints + async reindex
│   └── static/
│       ├── index.html      # Web UI (search page)
│       ├── map.html        # Photo map with clusters (Leaflet)
│       ├── results.html    # Cluster results page
│       ├── admin.html      # Admin dashboard (indexing management)
│       ├── albums.html     # Album list page
│       ├── album_detail.html # Album detail & photo viewer
│       ├── album_picker.js # Reusable album picker component
│       └── person_selector.js # Reusable person picker component
├── bot/
│   └── telegram_bot.py     # Telegram bot for photo search
├── db/
│   └── database.py         # SQLAlchemy + pgvector
├── models/
│   └── data_models.py      # Pydantic + ORM models
├── scripts/
│   ├── fast_reindex.py     # Main indexing script (run from Windows host)
│   ├── find_duplicates.py  # CLI: find duplicates & generate report
│   ├── populate_exif_data.py # Extract EXIF/GPS from all photos in DB
│   ├── compute_phash.py    # Compute pHash on Windows host (fast, parallel)
│   ├── test_phash256.py    # Test 256-bit pHash on old report files
│   ├── restore_false_duplicates.py # Restore falsely deleted files from .photo_duplicates
│   ├── copy_duplicate_group.py # Copy duplicate group for manual review
│   ├── export_person_faces.py # Export assigned faces to folders (720p thumbnails)
│   ├── start_bot.sh        # Bot startup script (waits for cloudflared tunnel)
│   ├── test_cleanup.py     # Test cleanup logic
│   └── test_db.py          # Test DB connection
├── util/
│   ├── cleanup_orphaned.py # CLI: remove DB records for missing files
│   └── fix_video_extensions.py  # Rename misnamed video files
├── backups/
│   └── backup_db.bat        # DB backup script
├── sql/
│   ├── init_db.sql         # DB schema + HNSW indexes (1152-dim)
│   └── migrate_*.sql       # DB migrations
├── reference/              # Reference scripts (not used in production)
├── docker-compose.yml      # 4 services: db, api, cloudflared, bot
├── Dockerfile              # PyTorch 2.6 + CUDA 12.4
├── run.bat                 # Windows launch script
├── test_basic.py           # Basic tests
└── requirements.txt        # Python dependencies
```

## Supported Image Formats

| Format | Extensions | Library |
|--------|------------|---------|
| JPEG | `.jpg`, `.jpeg` | Pillow/OpenCV |
| PNG | `.png` | Pillow/OpenCV |
| HEIC/HEIF | `.heic`, `.heif` | pillow-heif |
| WebP | `.webp` | Pillow |
| BMP | `.bmp` | Pillow/OpenCV |
| Nikon RAW | `.nef` | rawpy |
| Canon RAW | `.cr2` | rawpy |
| Sony RAW | `.arw` | rawpy |
| Adobe DNG | `.dng` | rawpy |
| Fujifilm | `.raf` | rawpy |
| Olympus | `.orf` | rawpy |
| Panasonic | `.rw2` | rawpy |

## Key Files

| File | Purpose |
|------|---------|
| `.env` | Config: DB, paths, CLIP model, device, Telegram token |
| `docker-compose.yml` | 4 services (db, api, cloudflared, bot) with GPU |
| `Dockerfile` | Base: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| `sql/init_db.sql` | DB schema + HNSW indexes for pgvector (1152-dim) |
| `requirements.txt` | Python deps (torch is in Docker image) |

## Database Schema

```sql
-- photo_index: основная таблица
## Database Schema (Multi-Model Support)

```sql
-- photo_index: основная таблица
CREATE TABLE photo_index (
    image_id SERIAL PRIMARY KEY,           -- единственный ID (UUID удален)
    file_path VARCHAR(1024) UNIQUE NOT NULL,
    file_name VARCHAR(256) NOT NULL,
    file_size INTEGER,
    file_format VARCHAR(10),
    width INTEGER, height INTEGER,
    created_at TIMESTAMP, modified_at TIMESTAMP,
    photo_date TIMESTAMP,

    -- Геолокация (GPS координаты из EXIF)
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,

    -- Мульти-модельные эмбеддинги (каждая модель в своей колонке)
    clip_embedding_vit_b32 vector(512),    -- ViT-B/32 (openai/clip-vit-base-patch32)
    clip_embedding_vit_b16 vector(512),    -- ViT-B/16 (openai/clip-vit-base-patch16)
    clip_embedding_vit_l14 vector(768),    -- ViT-L/14 (openai/clip-vit-large-patch14)
    clip_embedding_siglip vector(1152),    -- SigLIP (google/siglip-so400m-patch14-384)

    exif_data JSONB,
    faces_indexed INTEGER NOT NULL DEFAULT 0  -- Флаг индексации лиц
);

-- person: персоны (люди на фотографиях)
CREATE TABLE person (
    person_id SERIAL PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    description TEXT,
    cover_face_id INTEGER,  -- Лучшее лицо для аватара
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- faces: лица на фотографиях
CREATE TABLE faces (
    face_id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES photo_index(image_id) ON DELETE CASCADE,
    person_id INTEGER REFERENCES person(person_id) ON DELETE SET NULL,

    -- Bounding box (координаты в пикселях)
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,

    -- Уверенность детекции (0.0 - 1.0)
    det_score REAL NOT NULL,

    -- Ключевые точки лица (JSON массив)
    landmarks JSONB,

    -- Атрибуты от InsightFace
    age INTEGER,
    gender INTEGER,  -- 0 = female, 1 = male

    -- Эмбеддинг лица (InsightFace buffalo_l = 512 измерений)
    face_embedding vector(512) NOT NULL,

    created_at TIMESTAMP DEFAULT NOW()
);

-- HNSW индексы для каждой модели (cosine similarity)
CREATE INDEX idx_clip_siglip_hnsw ON photo_index USING hnsw (clip_embedding_siglip vector_cosine_ops);
CREATE INDEX idx_clip_vit_b32_hnsw ON photo_index USING hnsw (clip_embedding_vit_b32 vector_cosine_ops);
CREATE INDEX idx_clip_vit_b16_hnsw ON photo_index USING hnsw (clip_embedding_vit_b16 vector_cosine_ops);
CREATE INDEX idx_clip_vit_l14_hnsw ON photo_index USING hnsw (clip_embedding_vit_l14 vector_cosine_ops);

-- HNSW индекс для поиска похожих лиц
CREATE INDEX idx_faces_embedding_hnsw ON faces USING hnsw (face_embedding vector_cosine_ops);

-- Индексы для геопоиска
CREATE INDEX idx_photo_index_geo ON photo_index (latitude, longitude) WHERE latitude IS NOT NULL;
CREATE INDEX idx_photo_index_photo_date ON photo_index (photo_date) WHERE photo_date IS NOT NULL;

-- scan_checkpoint: хранение USN Journal checkpoint
CREATE TABLE scan_checkpoint (
    id SERIAL PRIMARY KEY,
    drive_letter VARCHAR(10) NOT NULL UNIQUE,  -- e.g., "H:"
    last_usn BIGINT NOT NULL DEFAULT 0,        -- NTFS USN Journal position
    last_scan_time TIMESTAMP DEFAULT NOW(),
    files_count INTEGER DEFAULT 0
);

-- app_user: пользователи приложения
CREATE TABLE app_user (
    user_id SERIAL PRIMARY KEY,
    telegram_id BIGINT UNIQUE,
    username VARCHAR(128),
    display_name VARCHAR(256) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_seen_at TIMESTAMP DEFAULT NOW()
);

-- album: фотоальбомы
CREATE TABLE album (
    album_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
    title VARCHAR(512) NOT NULL,
    description TEXT,
    cover_image_id INTEGER REFERENCES photo_index(image_id) ON DELETE SET NULL,
    is_public BOOLEAN DEFAULT FALSE,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- album_photo: связь альбомов с фотографиями (many-to-many)
CREATE TABLE album_photo (
    album_id INTEGER REFERENCES album(album_id) ON DELETE CASCADE,
    image_id INTEGER REFERENCES photo_index(image_id) ON DELETE CASCADE,
    sort_order INTEGER DEFAULT 0,
    added_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (album_id, image_id)
);
```

**Изменения в схеме БД:**
- **Удалены колонки:** `id` (UUID), `clip_embedding` (legacy), `clip_model`, `indexed`, `indexed_at`, `meta_data`
- **Мульти-модельная поддержка:** каждая CLIP модель хранится в отдельной колонке с правильной размерностью
- **image_id** - единственный первичный ключ (SERIAL, автоинкремент)
- **Проверка индексации:** `WHERE <embedding_column> IS NOT NULL` вместо `indexed=1`
- **Face detection:** таблицы `faces` и `person` реализованы и работают (InsightFace buffalo_l, 512 dim)

**Миграция:**
```bash
# 1. Создать новые колонки и перенести данные
psql -U dev -d smart_photo_index -f scripts/migrate_multi_model.sql

# 2. Удалить legacy колонки (после проверки)
psql -U dev -d smart_photo_index -f scripts/cleanup_legacy_columns.sql
```

## API Endpoints

```
GET    /health                  # service status
GET    /models                  # list available CLIP models with data in DB
GET    /stats                   # indexed photos count BY MODEL (показывает статистику по каждой модели)
POST   /search/text             # {"query": "cat on sofa", "top_k": 10, "translate": true, "model": "SigLIP", "formats": ["jpg", "heic"]}
                                # Response: {results: [...], translated_query: str, model: str}
POST   /search/image            # multipart file upload (find similar), query param: model (optional)
                                # Response: {results: [...], model: str}
GET    /photo/{image_id}        # photo details (включая данные о лицах)
GET    /image/{image_id}/thumb  # thumbnail 400px (JPEG), 3-tier cache: memory → disk → generate
GET    /image/{image_id}/full   # full image max 2000px (JPEG)
POST   /photos/delete           # {"image_ids": [123, 456]} - move to TRASH_DIR
POST   /cleanup/orphaned        # удалить записи в БД для несуществующих файлов
                                # Body: ["path1", "path2"] - удалить указанные пути (fast)
                                # Body: null - проверить все файлы на диске (slow)
POST   /reindex/files           # multipart gzipped JSON file list + model param (used by fast_reindex.py)
GET    /reindex/status          # reindex progress (running, total, indexed, percentage, model)
GET    /files/unindexed?model=X # files without embeddings for model (used by fast_reindex.py)
GET    /scan/checkpoint/{drive} # get USN checkpoint for drive (e.g., "H:")
POST   /scan/checkpoint         # save USN checkpoint {drive_letter, last_usn, files_count}
POST   /duplicates              # find duplicates by CLIP (JSON: threshold, limit, path_filter)
DELETE /duplicates              # find & delete duplicates (query: threshold, path_filter)

# pHash Duplicate Detection (perceptual hash)
POST   /duplicates/phash        # find duplicates by pHash {threshold: 0, limit: 50000, path_filter: null, all_types: false}
                                # threshold: 0 = exact, <=6 = near-duplicates. all_types: match across formats
DELETE /duplicates/phash        # find & delete pHash duplicates (move to .photo_duplicates)
POST   /phash/reindex           # compute pHash for photos without it (background task in Docker)
GET    /phash/reindex/status    # progress: {running, total, computed, pending, speed_imgs_per_sec, eta_formatted}
POST   /phash/reindex/stop      # stop background pHash reindex (progress saved)
GET    /phash/pending           # files without pHash (for host script compute_phash.py)
POST   /phash/update            # batch update pHash {hashes: {id: hex}, failed: [id]}

# Map API (геолокация)
GET    /map/stats               # статистика по гео-данным (with_gps, date_range, geo_bounds)
POST   /map/clusters            # кластеры для карты {"min_lat", "max_lat", "min_lon", "max_lon", "zoom", "date_from?", "date_to?"}
GET    /map/photos              # фото в bounding box (query: min_lat, max_lat, min_lon, max_lon, date_from?, date_to?, limit, offset)
POST   /map/search              # текстовый поиск в географической области (query params: min_lat..., body: TextSearchRequest)

# Geo Assignment API (привязка GPS координат)
GET    /geo/stats               # статистика по фото без GPS (total, with_gps, without_gps)
GET    /geo/folders             # папки с фото без GPS (path, count)
GET    /geo/photos              # фото без GPS (query: folder, limit, offset)
POST   /geo/assign              # привязать GPS к фото {"image_ids": [1,2,3], "latitude": 54.5, "longitude": 16.5}

# Face Detection & Recognition API (InsightFace)
POST   /faces/reindex           # индексация лиц (body: {skip_indexed: bool, batch_size: int})
GET    /faces/reindex/status    # статус индексации лиц
GET    /photo/{image_id}/faces  # все лица на фото
POST   /photo/{image_id}/faces/auto-assign  # автоматическое назначение лиц на основе сходства
POST   /search/face             # поиск похожих лиц по загруженному фото
POST   /search/face/by_id/{face_id}  # поиск похожих лиц по face_id из БД

# Person Management API
GET    /persons                 # список всех персон (with_stats: face_count, photo_count)
POST   /persons                 # создание персоны {"name": "John Doe", "description": "..."}
GET    /persons/{person_id}     # информация о персоне
DELETE /persons/{person_id}     # удаление персоны (faces становятся unassigned)
POST   /persons/{person_id}/merge/{target_person_id}  # объединение двух персон
GET    /persons/{person_id}/photos  # все фото с этой персоной
POST   /faces/{face_id}/assign  # назначить лицо персоне {"person_id": 123}
DELETE /faces/{face_id}/assign  # отменить назначение лица
POST   /persons/{person_id}/auto-assign  # автоматически назначить похожие лица персоне
POST   /persons/maintenance/recalculate-covers  # пересчитать обложки для всех персон

# Admin API (indexing management)
POST   /reindex/stop             # остановить CLIP индексацию (текущий батч завершится)
POST   /faces/reindex/stop       # остановить индексацию лиц (текущий батч завершится)
POST   /admin/index-all          # запустить последовательную индексацию {models, include_faces, include_phash, shutdown_after}
GET    /admin/index-all/status   # статус очереди индексации + прогресс текущей подзадачи
POST   /admin/index-all/stop     # остановить очередь (текущая задача завершится, остальные отменяются)
POST   /admin/shutdown-flag      # установить флаг выключения PC
GET    /admin/shutdown-flag      # проверить флаг выключения + статус завершения
GET    /admin/cache/stats        # статистика кэша миниатюр (file_count, total_size, memory_cache)
POST   /admin/cache/clear        # очистить кэш миниатюр (диск + память)
POST   /admin/cache/warm         # прогреть кэш (query: heavy_only, sizes)
GET    /admin/cache/warm/status   # статус прогрева кэша
POST   /admin/cache/warm/stop    # остановить прогрев кэша

# Album API (фотоальбомы)
GET    /albums                    # список альбомов (query: user_id, search, limit, offset)
POST   /albums                    # создать альбом {"title", "description", "is_public"}
GET    /albums/{album_id}         # информация об альбоме (с photo_count)
PUT    /albums/{album_id}         # обновить альбом {title, description, cover_image_id, is_public}
DELETE /albums/{album_id}         # удалить альбом (cascade album_photos)
GET    /albums/{album_id}/photos  # фото в альбоме (query: limit, offset)
POST   /albums/{album_id}/photos  # добавить фото {"image_ids": [1,2,3]}
DELETE /albums/{album_id}/photos  # удалить фото {"image_ids": [1,2,3]}
POST   /albums/{album_id}/cover/{image_id}  # установить обложку альбома
GET    /photo/{image_id}/albums   # альбомы, содержащие фото
```

**Изменения в API:**
- Все поиски и статистика работают с моделью, указанной в `CLIP_MODEL` (.env)
- `SearchResult.image_id` теперь `int` (было `str`)
- Face detection endpoints полностью реализованы (InsightFace buffalo_l)
- Ответы поиска включают `model` для отображения используемой модели
- Ответы текстового поиска включают `translated_query` если запрос был переведен
```

## Postman Collection

**File:** `Smart_Photo_Indexing_API.postman_collection.json`

**Import:** File → Import in Postman

**Contains:**
- All API endpoints with example requests
- Environment variable: `{{base_url}}` = `http://localhost:8000`
- Examples for all CLIP models (ViT-B/32, ViT-B/16, ViT-L/14, SigLIP)
- Reindex with model selection
- Duplicate detection and deletion

## Web UI

Available at `http://localhost:8000/` when API is running.

**Layout:** Adaptive horizontal toolbar — filters left, search center, actions right. Stacks vertically on narrow screens (<900px).

**Features:**
- Text search with optional auto-translation (ru -> en, toggle via checkbox)
- SigLIP supports Russian natively, translation is optional
- Adjustable similarity threshold (0-50%)
- Results count selector (10/20/50/100)
- Results sorted by relevance (best match top-left)
- **File type filters** — checkboxes for JPG, HEIC, PNG, NEF
- **Auto-translate EN** — checkbox to toggle query translation (default: on)
- **Select mode** — click "Select" to enable multi-selection
- **Delete to trash** — move selected files to TRASH_DIR (preserving folder structure)
- **GPS badge (🌐)** on thumbnails when coordinates exist
- Lightbox preview (click on photo) with GPS button to open map
- Format badge on each thumbnail
- **Navigation** — links between Search and Map pages

## Map UI

Available at `http://localhost:8000/map.html` when API is running.

**Features:**
- World map with photo clusters (Leaflet.js) and base layer switcher
  - Default: OpenStreetMap Standard
  - Optional: Topographic (OpenTopoMap), Satellite (Esri), Dark (CartoDB)
- **File type filters** — checkboxes for JPG, HEIC, PNG, NEF (instant apply on change)
  - PNG unchecked by default
  - Filters passed to results.html when opening clusters
- **Date filters** — From/To date pickers for filtering photos
- **Server-side clustering** — clusters adapt to zoom level
- **Click on cluster** — zoom in or open photos in new tab
- **Photos view** (results.html) — gallery with pagination
  - Search/date filters shown when pagination is needed or when filters are active
  - File type filters from map are preserved
- **Text search within area** — CLIP search limited to geographic bounds
- Lightbox preview on results page with file path and image ID in status bar
- **Fullscreen mode** — button in toolbar to hide UI and maximize map
  - Native Fullscreen API on desktop/Android
  - CSS fallback on iOS (hides toolbar, maximizes map)

## Geo Assignment UI

Available at `http://localhost:8000/geo_assign.html` when API is running.

**Purpose:** Simplified bulk GPS coordinate assignment to photos without leaving the browser.

**Layout:** 4-part grid:
- **Top toolbar** — navigation, select mode toggle, assign button, stats
- **Top-left panel** — list of folders with photos without GPS
- **Top-right panel** — interactive map with marker placement
- **Bottom panel** — photo thumbnails grid

**Features:**
- **Folder list** — shows only folders containing photos without GPS coordinates
  - Click folder to load its photos in the bottom grid
  - Folder count shows number of unassigned photos
  - Folders auto-hide when all photos are assigned
- **Map marker** — click anywhere to place/move marker (draggable)
  - Coordinates displayed in the info bar below map
  - Layer switcher: OpenStreetMap / Satellite
- **Photo selection** — two modes:
  - Default: assign coordinates to all photos in selected folder
  - Select mode: click "Выбрать фото" to enable multi-selection
- **Assign coordinates** — applies selected map point to chosen photos
  - If no photos are selected, assigns to all photos in the selected folder (regardless of UI pagination)
  - Photos disappear from grid after assignment
  - Stats update automatically

**Workflow:**
1. Select a folder from the left panel
2. Click on map to place marker at desired location
3. Either assign to all folder photos, or enable select mode and pick specific ones
4. Click "Привязать координаты" button

## Admin UI

Available at `http://localhost:8000/admin.html` when API is running.

**Purpose:** Centralized dashboard for managing all indexing tasks (CLIP, Faces, pHash).

**Features:**
- **DB Stats bar** — live counts: total photos, per-model CLIP counts, faces, pHash
- **Index All** — sequential queue: CLIP models -> Faces -> pHash
  - Checkboxes to select models and task types
  - Queue visualization: completed/current/pending tasks
  - Option to shutdown PC after completion
- **Individual indexer controls** — separate Start/Stop for each:
  - CLIP (with model selector dropdown)
  - Face detection
  - pHash computation
- **Progress bars** — same visual style as index.html (red CLIP, purple Faces, yellow pHash)
- **Quick links** — GPS Assignment, Search, Map
- **Polling** — status updates every 2 seconds, stats every 30 seconds

**Admin API Endpoints:**
```
POST   /reindex/stop             # stop CLIP indexing (current batch completes)
POST   /faces/reindex/stop       # stop face indexing (current batch completes)
POST   /admin/index-all          # start sequential indexing queue
                                 # body: {models: ["SigLIP"], include_faces: true, include_phash: true, shutdown_after: false}
GET    /admin/index-all/status   # queue status + sub-task progress
POST   /admin/index-all/stop     # stop queue (current task completes, remaining cancelled)
POST   /admin/shutdown-flag      # set shutdown flag
GET    /admin/shutdown-flag      # check shutdown flag + indexing completion status
GET    /admin/cache/stats        # thumbnail cache stats (file_count, total_size)
POST   /admin/cache/clear        # clear thumbnail cache
POST   /admin/cache/warm         # warm cache (query: heavy_only, sizes)
GET    /admin/cache/warm/status   # warm cache progress
POST   /admin/cache/warm/stop    # stop cache warm
```

## Config (.env)

```env
# PostgreSQL
POSTGRES_USER=dev
POSTGRES_PASSWORD=secret
POSTGRES_DB=smart_photo_index
DATABASE_URL=postgresql://dev:secret@localhost:5432/smart_photo_index

# Photos path (use / not \ on Windows)
PHOTOS_HOST_PATH=H:/PHOTO

# Model
CLIP_MODEL=SigLIP      # or ViT-B/32, ViT-B/16, ViT-L/14
CLIP_DEVICE=cuda        # or cpu

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=/logs/indexer.log
LOGS_HOST_PATH=./logs

# Trash (deleted files moved here, preserving folder structure)
TRASH_DIR=/photos/.trash

# Telegram bot (optional)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_ALLOWED_USERS=123456789
```

## Models

| Model | HuggingFace ID | Dim | Quality | Speed | Multilingual |
|-------|---------------|-----|---------|-------|--------------|
| **SigLIP so400m** | `google/siglip-so400m-patch14-384` | 1152 | Best | ~3.5 img/s | Yes |
| ViT-B/32 | `openai/clip-vit-base-patch32` | 512 | Good | ~15 img/s | No |
| ViT-B/16 | `openai/clip-vit-base-patch16` | 512 | Better | ~10 img/s | No |
| ViT-L/14 | `openai/clip-vit-large-patch14` | 768 | Great | ~5 img/s | No |

**Default:** SigLIP so400m. Requires `sentencepiece` and `protobuf` packages.
Uses explicit `SiglipTokenizer` + `AutoImageProcessor` + `SiglipProcessor` (AutoProcessor broken in transformers 5.0).

## Dependencies (requirements.txt)

```
# Database
psycopg2-binary, sqlalchemy, pgvector

# Image processing
Pillow, opencv-python-headless, pillow-heif, rawpy

# Model
transformers, sentencepiece, protobuf

# Utilities
numpy, tqdm, pydantic, pydantic-settings, python-dotenv, watchdog

# Translation
deep-translator (Google Translate, optional)

# API
fastapi, uvicorn, python-multipart, httpx

# Telegram bot
python-telegram-bot

# Logging
loguru
```

**Note:** PyTorch is included in the Docker base image, not in requirements.txt.

## Indexing Architecture

Индексация запускается скриптом `fast_reindex.py` с Windows хоста, который отправляет список файлов в API.

**Скрипт:** [scripts/fast_reindex.py](scripts/fast_reindex.py)
- Использует NTFS USN Journal для мгновенного обнаружения изменений (~0 сек)
- Проверяет `/files/unindexed` API для файлов без эмбеддингов
- Отправляет gzip-сжатый список файлов в `POST /reindex/files`
- Сохраняет checkpoint в БД (таблица `scan_checkpoint`)

**API:** [api/main.py](api/main.py)
- `POST /reindex/files` — принимает список файлов и запускает индексацию на GPU
- `GET /files/unindexed?model=X` — возвращает файлы без эмбеддингов для модели
- `GET/POST /scan/checkpoint` — управление USN checkpoint

**Indexer Service:** [services/indexer.py](services/indexer.py)
- **Multi-model support:** сохраняет эмбеддинги в колонку для указанной модели
- **Upsert logic:** if record exists (by file_path) — UPDATE; otherwise INSERT
- **Batch processing:** 16 изображений на GPU за раз

**CLIPEmbedder:** [services/clip_embedder.py](services/clip_embedder.py)
- Поддерживает 4 модели: ViT-B/32, ViT-B/16, ViT-L/14, SigLIP
- Выбор модели через `.env` → `CLIP_MODEL` (default: SigLIP)
- `get_embedding_column()` возвращает имя колонки БД для текущей модели
- Маппинг: `CLIP_MODEL_COLUMNS` в [models/data_models.py](models/data_models.py)

**DuplicateFinder:** [services/duplicate_finder.py](services/duplicate_finder.py)
- Использует HNSW индекс для поиска дубликатов (K-NN вместо brute-force)
- Работает с текущей моделью (передается CLIPEmbedder instance)
- Threshold по умолчанию: 0.98 (98% сходство)
- `save_report()` сохраняет отчет в текстовый файл
- `delete_from_report()` удаляет дубликаты на основе отчета (с dry_run режимом)

**Database Changes:**
- `PhotoIndexRepository.add_photo()` возвращает `image_id` (int) вместо UUID
- `get_unindexed_photos()` принимает параметр `embedding_column` для фильтрации по модели
- Удален `FaceRepository` и все функции работы с лицами

**Web UI Changes:** [api/static/index.html](api/static/index.html)
- Добавлен переключатель размера плиток (XL/L/M/S) в Windows-стиле
- Фиксированные размеры плиток: 300px/200px/150px/100px
- Автоматическая grid-сетка вместо адаптивных колонок
- Отображение используемой модели и переведенного запроса в результатах поиска
- Статистика показывает проиндексировано для текущей модели

## Telegram bot (telegram_bot.py)

**Default model:** ViT-L/14 (can be changed via `/model` command)

**Features:**
- Text search with optional auto-translation (ru -> en)
- Image search (upload photo to find similar)
- **Model selection menu** — `/model` command shows inline keyboard with available models:
  - ViT-L/14 (default, 768 dim, best quality)
  - SigLIP so400m (1152 dim, multilingual)
  - ViT-B/32 (512 dim, fastest)
  - ViT-B/16 (512 dim, medium)
- Selected model is saved per user session
- Format filter: `BOT_FORMATS` env variable (default: jpg,jpeg,heic,heif,nef)
- Sends full-size images (not thumbnails)
- Shows current model in search messages
- **Photo map** — `/map` command returns link to map via cloudflared tunnel
- **User whitelist** — `TELEGRAM_ALLOWED_USERS` env variable limits access

**Commands:**
- `/start` — bot info and current model
- `/model` — open model selection menu
- `/map` — link to photo map (via cloudflared tunnel)

**Cloudflared Integration:**
- Bot waits for cloudflared tunnel URL on startup (`scripts/start_bot.sh`)
- Gets tunnel URL from cloudflared metrics endpoint
- `/map` command returns public trycloudflare.com URL

**Usage:**
```bash
# Set environment variables
BOT_TOKEN=your_telegram_bot_token
TELEGRAM_ALLOWED_USERS=123456789,987654321  # comma-separated user IDs
API_URL=http://api:8000
TOP_K=3
BOT_FORMATS=jpg,jpeg,heic,heif,nef

# Run bot (starts cloudflared automatically)
docker-compose up -d cloudflared bot
```

**Model selection UI:**
Interactive inline keyboard with checkmarks showing current model:
```
✅ ViT-L/14
   SigLIP
   ViT-B/32
   ViT-B/16
```

Click any model to switch, selection persists for user session.

## Common Tasks

### Rebuild & restart API
```bash
docker-compose build --no-cache api
docker-compose up -d api
```

### Run indexing (from Windows host)
```bash
# Incremental (uses USN Journal)
python scripts/fast_reindex.py --model SigLIP

# Full scan
python scripts/fast_reindex.py --model SigLIP --full-scan
```

### View logs
```bash
docker logs smart_photo_api -f
# Detailed logs: logs\indexer.log
```

### Recreate database
```bash
psql -U dev -c "DROP DATABASE smart_photo_index;"
psql -U dev -c "CREATE DATABASE smart_photo_index;"
psql -U dev -d smart_photo_index -f sql/init_db.sql
```

### Migrate to multi-model schema
```bash
# 1. Add new columns and migrate data
psql -U dev -d smart_photo_index -f sql/migrate_multi_model.sql

# 2. Cleanup legacy columns (after verification)
psql -U dev -d smart_photo_index -f sql/cleanup_legacy_columns.sql

# 3. Reindex with new model
python scripts/fast_reindex.py --model ViT-L/14 --full-scan
```

### Test GPU in container
```bash
docker run --rm --gpus all pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.is_available())"
```

## Known Issues

1. **Windows TDR timeout** — reduce BATCH_SIZE_CLIP to 16 if GPU resets
2. **RAW processing is slow** — rawpy decodes on CPU
3. **Windows paths** — use `/` instead of `\` in PHOTOS_HOST_PATH
4. **SigLIP cache** — stored in `/root/.cache/huggingface` (Docker volume)
5. **transformers 5.0** — AutoProcessor/AutoTokenizer broken for SigLIP, must use explicit SiglipTokenizer

## Recent Changes (January 2026)

### Database Schema Refactoring
- **Multi-model support:** каждая CLIP модель теперь хранится в отдельной колонке с правильной размерностью
- **Удален UUID:** `image_id` теперь SERIAL PRIMARY KEY (автоинкремент integer)
- **Удалены legacy колонки:** `clip_embedding`, `clip_model`, `indexed`, `indexed_at`, `meta_data`
- **Face detection:** функционал распознавания лиц СОХРАНЕН (face_embedder, FaceIndexingService, person_service)
- **Новая логика индексации:** проверка `WHERE <embedding_column> IS NOT NULL` вместо `indexed=1`

### Code Changes
- **API:** все endpoints обновлены для работы с мульти-модельной схемой
  - `/stats` показывает статистику по каждой модели
  - Ответы поиска включают `model` и `translated_query` (если применимо)
  - `SearchResult.image_id` теперь `int` вместо `str`
  - Face detection endpoints сохранены и работают (lazy initialization)
- **Indexer:** `get_indexed_paths()` фильтрует по текущей модели
- **DuplicateFinder:** принимает `CLIPEmbedder` для определения используемой модели
- **Database:** `add_photo()` возвращает `int` вместо UUID string
- **Web UI:** добавлен переключатель размера плиток, отображение модели в результатах
- **Telegram Bot:** фильтр форматов, отправка полноразмерных изображений

### Migration Scripts
- `sql/migrate_multi_model.sql` — создание новых колонок и миграция данных
- `sql/cleanup_legacy_columns.sql` — удаление устаревших колонок
- `util/cleanup_orphaned.py` — удаление записей для несуществующих файлов (обновлен)
- `scripts/find_duplicates.py` — поиск дубликатов с поддержкой выбора модели

### Photo Map Feature
- **map.html:** интерактивная карта с кластерами фотографий (Leaflet + CartoDB Dark tiles)
  - Кластеры группируют фото по геолокации
  - Клик по кластеру → открывает results.html с фотографиями в этой области
  - Hover → popup с количеством фото
  - Фильтр по дате (от/до)
- **results.html:** просмотр фотографий кластера
  - Поддержка текстового поиска внутри географической области
  - Пагинация, lightbox просмотр
- **Map API endpoints:** `/map/stats`, `/map/clusters`, `/map/photos`, `/map/search`

### EXIF Data Population
- **populate_exif_data.py:** скрипт для извлечения EXIF из всех фото в БД
  - Использует `exifread` для надежного извлечения GPS и даты
  - Поддержка HEIC/HEIF через pillow-heif
  - Обработка батчами с ID-based pagination (исправлен баг с OFFSET)
  - Запуск: `docker exec smart_photo_api python /app/scripts/populate_exif_data.py`
- **image_processor.py:** исправлена функция `extract_exif()` — возвращает `None` вместо `{}` для файлов без EXIF
- **Indexer:** теперь извлекает EXIF при индексации новых файлов

### Cloudflared Tunnel Integration
- **docker-compose.yml:** добавлен сервис `cloudflared` для публичного доступа к API
  - Quick tunnel через trycloudflare.com (без регистрации)
  - Автоматический запуск после healthy API
  - Метрики на порту 2000 для получения URL туннеля
- **scripts/start_bot.sh:** скрипт запуска бота с ожиданием URL туннеля
  - Получает URL из cloudflared metrics endpoint
  - Передает URL через `TUNNEL_URL` env variable
- **telegram_bot.py:** команда `/map` возвращает ссылку на карту через туннель
  - Работает только для пользователей из whitelist
  - Показывает статистику (фото с GPS / всего)
- **map.html:** добавлена кнопка fullscreen для мобильных устройств
  - Native Fullscreen API на desktop/Android
  - CSS fallback на iOS (скрывает toolbar)

## Recent Changes (February 2026)

### Orphaned Records Cleanup (Feb 5, 2026)
- **fast_reindex.py автоматически удаляет записи для удаленных файлов**
  - USN Journal детектит удаленные файлы мгновенно
  - Автоматическая отправка deleted файлов в `/cleanup/orphaned`
  - Записи удаляются из БД сразу после детекции
- **Новый endpoint `/cleanup/orphaned`** — удаление orphaned записей
  - Fast mode: принимает gzip-сжатый список путей (используется fast_reindex.py)
  - Slow mode: проверяет все файлы через Docker volume (медленно)
  - Опциональный флаг `--cleanup` для полной проверки всех файлов
- **Оптимизация cleanup** — проверка существования на Windows хосте
  - Список файлов получается через `/files/index`
  - Проверка Path.exists() на локальной FS (быстро)
  - Отправка только missing файлов в API для удаления
  - Использует gzip сжатие как `/reindex/files`

### GPU Memory Optimization (Feb 5, 2026)
- **Problem:** IndexingService создавал новую копию CLIP модели вместо переиспользования из API
  - API: 3.27 GB (SigLIP)
  - Indexing: 6.54 GB (новая копия SigLIP)
  - **Итого:** ~10 GB при 8 GB доступных → GPU переполнение → падение скорости с 4-15 img/s до 0.1 img/s
- **Solution:** Переиспользование моделей через параметры конструкторов
  - `IndexingService(clip_embedder=...)` — принимает готовый embedder из API
  - `FaceIndexingService(face_embedder=...)` — принимает готовый face embedder
  - `api/main.py` — передает глобальные embedders в сервисы индексации
  - Добавлено логирование: "Переиспользую загруженную модель" / "Создаю новую модель"
- **Result:** Индексация использует только одну копию модели, GPU память в норме, скорость восстановлена

### Map Format Filters
- **map.html:** добавлены фильтры по типам файлов (JPG, HEIC, PNG, NEF)
  - Мгновенное применение при изменении чекбокса
  - PNG по умолчанию не выбран
  - Фильтры передаются в results.html при открытии кластера
- **results.html:** поддержка фильтров из URL параметра `formats`
- **API endpoints:** `/map/clusters` и `/map/photos` поддерживают параметр `formats`

### RAW/NEF Dimension Fixes
- **image_processor.py:** `get_image_dimensions()` теперь корректно возвращает размеры RAW файлов
  - Использует `rawpy` вместо PIL (PIL читает только встроенный thumbnail)
  - Учитывает `raw.sizes.flip` для 90° поворотов
- **face_embedder.py:** добавлена поддержка RAW файлов через `rawpy.postprocess()`
  - rawpy автоматически применяет поворот через `flip` — дополнительная EXIF ротация не нужна
- **api/main.py:** упрощена логика для face bbox
  - БД хранит повёрнутые размеры, API возвращает их напрямую
- **scripts/fix_nef_dimensions.py:** скрипт для исправления размеров NEF в БД
- **util/fix_photo_dimensions_report.py:** скрипт для исправления ориентации по EXIF

### Lightbox Improvements (Feb 6-7, 2026)
- **Face count display fix** ([index.html](api/static/index.html), [results.html](api/static/results.html)):
  - Исправлено отображение количества лиц в статусной строке lightbox
  - Формат: `Лица: X/Y` где X = распознанные (assigned), Y = всего лиц
  - Функция `loadFaceCount()` теперь загружает данные о лицах для корректного подсчета
  - Данные загружаются сразу при открытии фото, без необходимости нажимать кнопку показа лиц
- **results.html:** в статусной строке lightbox отображаются:
  - image_id
  - Путь к файлу (сокращённый)
  - Количество лиц (распознанные/всего)

### Map UI Improvements (Feb 7, 2026)
- **Cluster popup improvements** ([map.html](api/static/map.html)):
  - Убран бесполезный попап с координатами (координаты видны по положению кластера)
  - Новый попап показывает миниатюры первых 10 фотографий из кластера
  - Задержка показа 0.5 сек — попап не мешает при простом проведении мыши
  - Таймер отменяется при уходе мыши, предотвращая лишние запросы к API
  - Grid-сетка 5×2 с квадратными миниатюрами (aspect-ratio 1:1)
  - Состояния: "Загрузка превью..." → миниатюры или "Ошибка загрузки"
- **Убран loading popup при скролле карты:**
  - Моргающий попап "Загрузка кластеров..." отключён
  - Функция `showLoading()` теперь пустышка
  - Кластеры загружаются тихо в фоне без визуального отвлечения

### Geo Assignment UI Enhancements (Feb 7, 2026)
- **Delete functionality** ([geo_assign.html](api/static/geo_assign.html)):
  - Кнопка удаления выбранных фото (перемещение в корзину)
  - Диалог подтверждения удаления с escape-клавишей
  - Автоматическое обновление статистики и списка папок после удаления
  - Интеграция с API `/photos/delete` endpoint
- **Photo info header:**
  - Grid layout header: left (название + счетчик) / center (детали фото) / right (резерв)
  - При клике на фото показывается: image_id, размер файла, полный путь
  - Форматирование размера файла (B/KB/MB/GB)
  - Центральная секция появляется только при выборе фото
- **UX improvements:**
  - Фото исчезают из grid сразу после назначения координат или удаления
  - Папки автоматически скрываются когда все фото обработаны
  - Счётчик фото обновляется в реальном времени

### Person Service Fix (Feb 7, 2026)
- **PersonService.auto_assign_similar_faces** ([person_service.py](services/person_service.py)):
  - Исправлен pgvector query с bind parameters
  - Использование f-string для embedding interpolation вместо `:embedding` parameter
  - Решена проблема с SQL execution и vector type casting
  - Query теперь работает корректно с pgvector extension

### Face Export Script (Feb 7, 2026)
- **export_person_faces.py** ([scripts/export_person_faces.py](scripts/export_person_faces.py)):
  - Новый скрипт для экспорта лиц в отдельные папки по персонам
  - Создание 720p thumbnails с cropped face regions (margin 30%)
  - Поддержка всех форматов: JPEG, PNG, HEIC, RAW (через rawpy)
  - Применение EXIF orientation correction для правильного отображения
  - Progress bar с tqdm для отслеживания прогресса
  - Опция skip_existing для пропуска уже экспортированных файлов
  - Полезно для создания training datasets для face recognition
  - Запуск: `docker exec smart_photo_api python /app/scripts/export_person_faces.py --person-id 1 --output-dir /reports/faces`

### Person Filter on Map & Search (Feb 7, 2026)
- **PersonSelector component** ([person_selector.js](api/static/person_selector.js)):
  - Reusable JS class for selecting persons (face filter)
  - Dropdown with face thumbnails via `/faces/{face_id}/thumb`, text search, multi-select
  - Methods: `togglePerson()`, `removePerson()`, `clearSelection()`, `getSelectedIds()`
  - Loads persons from `/persons?limit=500`, filters those with `face_count > 0`
- **Face thumbnail endpoint** (`/faces/{face_id}/thumb`):
  - Crops face from photo using bbox with 20% padding
  - Scales bbox for fast_mode dimension mismatch (RAW embedded JPEG vs original)
- **Person filter API** — `person_ids` param added to:
  - `TextSearchRequest` — AND logic via `HAVING COUNT(DISTINCT person_id) = N`
  - `MapClusterRequest` — OR logic via subquery `SELECT image_id FROM faces WHERE person_id IN (...)`
  - `/map/photos` — OR logic via query param (comma-separated)
- **Map page** ([map.html](api/static/map.html)):
  - Person selector button in toolbar
  - Floating person chips on map with face avatars and close buttons
  - Map wrapped in `.map-wrapper` (position: relative) for correct chip positioning
  - Person chips hidden in fullscreen mode
- **Search page** ([index.html](api/static/index.html)):
  - Person selector in controls row
  - Selected persons shown as text tags with close buttons
- **Results page** ([results.html](api/static/results.html)):
  - `person_ids` passed from URL params to `/map/photos` API
- **Cover face fallback** ([person_service.py](services/person_service.py)):
  - `list_persons()` uses `COALESCE(cover_face_id, best_face_subquery)` — falls back to face with highest `det_score` when `cover_face_id` is NULL

### Instant Filters & iPad Layout (Feb 7, 2026)
- **Removed "Применить"/"Сбросить" buttons** from map.html and results.html
  - All filters (formats, dates, persons) now apply instantly on change
  - Date inputs trigger `loadClusters()`/`loadPhotos()` via `change` event
- **results.html iPad optimization:**
  - Compact toolbar: smaller padding (8px), gaps (8px), font sizes (12-13px)
  - Fixed date input width (130px), search box max-width 300px
  - Info panel pushed right with `margin-left: auto`
  - Filters panel always visible (search within area always useful)
  - Responsive breakpoints: tablet (1100px), phone (600px)
- **map.html layout fix:**
  - Map wrapped in `.map-wrapper` with `position: relative; flex: 1`
  - Person chips positioned relative to map area, not viewport
  - Fullscreen CSS targets `.map-wrapper` instead of `#map`

### Geo Assignment Thumbnail Improvements (Feb 7, 2026)
- **Sorting by date** ([api/main.py](api/main.py)):
  - Photos now sorted by `photo_date` ascending (oldest first) with `nullslast()`
  - Changed from filename alphabetical sorting to chronological order
  - Makes it easier to assign GPS to photos taken in sequence
- **Larger thumbnails** ([geo_assign.html](api/static/geo_assign.html)):
  - Increased thumbnail size from 120px to 150px
  - Better visibility of photo details for GPS assignment workflow
- **Photo date display:**
  - Added date column to photo info header (ID | Дата | Размер | Путь)
  - Shows photo capture date/time in format: DD.MM.YYYY HH:MM
  - Date stored in `data-date` attribute and displayed on thumbnail click
  - Added `formatPhotoDate()` helper function for ISO date formatting
- **API enhancement:**
  - `/geo/photos` endpoint now returns `file_size` and `photo_date` fields

### pHash Duplicate Detection (Feb 8, 2026)
- **Perceptual hash (pHash)** — pixel-level duplicate detection (vs CLIP semantic similarity)
  - CLIP at 0.99 threshold matches semantically similar but different photos
  - pHash matches only true duplicates: copies, resizes, re-encodings
  - 256-bit DCT hash via `imagehash` library (hash_size=16), stored as 64-char hex in `phash VARCHAR(64)`
- **New service** ([phash_service.py](services/phash_service.py)):
  - `PHashService.reindex()` — compute pHash for all photos, per-file commit, stop_flag support
  - `PHashService.find_duplicates(threshold, limit, path_filter, same_format_only)` — in-memory vectorized comparison
  - `same_format_only=True` (default): only match within same format group (jpg/jpeg, heic/heif, raw)
  - Loads all hashes as 4 x `np.uint64` chunks, XOR + popcount via byte lookup table
  - Union-Find grouping for transitive duplicates, ~5-10 seconds for 82K photos
- **API endpoints:**
  - `POST /duplicates/phash` — find duplicates, save report, return groups (`all_types: false` by default)
  - `DELETE /duplicates/phash` — find & delete pHash duplicates (move to `.photo_duplicates` dir)
  - `POST /phash/reindex` — background task to compute pHash in Docker
  - `GET /phash/reindex/status` — progress from DB (computed, pending, speed, ETA)
  - `POST /phash/reindex/stop` — stop background reindex (progress saved)
  - `GET /phash/pending` + `POST /phash/update` — for host-side computation
- **Host-side script** ([compute_phash.py](scripts/compute_phash.py)):
  - Computes pHash on Windows host (bypasses Docker volume I/O), ~10 img/s on i9-9900K
  - ThreadPoolExecutor, sends results incrementally every `send_batch` files (no waiting for full batch)
  - Marks failed files with `phash=''` to avoid infinite retry loop
- **Test & restore scripts:**
  - [test_phash256.py](scripts/test_phash256.py) — test 256-bit hashes on old report files before full reindex
  - [restore_false_duplicates.py](scripts/restore_false_duplicates.py) — restore falsely deleted files from `.photo_duplicates`
- **UI progress bar** ([index.html](api/static/index.html)):
  - Yellow progress bar for pHash indexing (like red CLIP / purple faces)
  - Shows computed/total, percent, pending, speed (img/s), ETA
  - Polls `/phash/reindex/status` every 2 seconds
- **DB changes:**
  - `phash VARCHAR(64)` column on `photo_index` + btree index
  - Migration: [migrate_add_phash.sql](sql/migrate_add_phash.sql) — uses DO block (avoids PG UNION type warning)
  - Failed files stored as `phash=''` (excluded from duplicate search)
- **Duplicate finder optimization** ([duplicate_finder.py](services/duplicate_finder.py)):
  - Adaptive ef_search: 40 for threshold>=0.95, 80 otherwise
  - Batch size 500→2000, added timing/ETA logging
  - Removed unused `distance` from SELECT

### Admin UI (Feb 9, 2026)
- **New page** ([admin.html](api/static/admin.html)):
  - Centralized dashboard for indexing management
  - DB stats bar: total photos, per-model CLIP counts, faces, pHash
  - "Index All" sequential queue: CLIP models -> Faces -> pHash with queue visualization
  - Individual Start/Stop controls for CLIP (with model selector), Faces, pHash
  - Progress bars with shimmer animation (same style as index.html)
  - Shutdown PC option after indexing completes
  - Quick links to GPS Assignment, Search, Map
  - 2-second polling for progress, 30-second polling for stats
- **Stop endpoints:**
  - `POST /reindex/stop` — graceful CLIP stop (added `request_stop()` to IndexingService)
  - `POST /faces/reindex/stop` — graceful face stop (added `request_stop()` to FaceIndexingService)
  - Both stop after current batch completes, progress is saved
- **Index All queue:**
  - `POST /admin/index-all` — sequential execution of CLIP, faces, pHash
  - `GET /admin/index-all/status` — queue state + sub-task progress
  - `POST /admin/index-all/stop` — stops current task + cancels remaining
- **Shutdown flag:** `POST/GET /admin/shutdown-flag` for host-side shutdown polling
- **Stats endpoint enhanced:** `/stats` now includes `total_faces` and `phash_count`
- **Navigation:** Admin link (gear icon) added to all pages (index, map, results, geo_assign)

### Thumbnail Disk Cache (Feb 9, 2026)
- **Disk cache for thumbnails** — generated thumbnails cached to `/.thumb_cache/`
  - Cache key: `{image_id}_{size}.jpg` — unique per image and requested size
  - Cache stored on host-mapped folder (like trash/duplicates), not in Docker volume
  - Docker: `${PHOTOS_HOST_PATH}/../.thumb_cache:/.thumb_cache`
  - Cache validation: if source file modified after cache, thumbnail regenerated
  - First request: generates + saves to disk (X-Cache: MISS)
  - Subsequent requests: served directly via `FileResponse` (X-Cache: HIT, ~10x faster)
  - Fallback: if cache write fails, serves from memory as before
- **Cache warm (pre-generate)**:
  - `POST /admin/cache/warm?heavy_only=true&sizes=200,400` — background task
  - `heavy_only=true`: only RAW + HEIC formats (slow to decode)
  - `GET /admin/cache/warm/status` — progress (processed, cached, skipped, speed, ETA)
  - `POST /admin/cache/warm/stop` — graceful stop
  - Heavy formats: nef, cr2, arw, dng, raf, orf, rw2, heic, heif
- **Cache management endpoints:**
  - `GET /admin/cache/stats` — file count, total size (human-readable)
  - `POST /admin/cache/clear` — delete all cached thumbnails
- **Admin UI:** Thumbnail Cache card with stats, Warm/Stop/Clear buttons, progress bar
- **Config:** `THUMB_CACHE_DIR` env var (default: `/.thumb_cache`)

### Album Feature (Feb 11, 2026)
- **New feature: photo albums** — organize photos into named collections
- **Database:** 3 new tables: `app_user`, `album`, `album_photo` (many-to-many)
  - Migration: [migrate_add_albums.sql](sql/migrate_add_albums.sql)
  - ORM models: `AppUser`, `Album`, `AlbumPhoto` in [data_models.py](models/data_models.py)
- **Service:** [album_service.py](services/album_service.py) — `AlbumService` + `AlbumRepository`
  - CRUD for albums, add/remove photos, auto-cover selection
  - Initialized on API startup, uses session factory
- **API endpoints:** full CRUD for albums + photo management (see Album API section above)
- **UI pages:**
  - [albums.html](api/static/albums.html) — album list with grid cards, search, create/edit/delete
  - [album_detail.html](api/static/album_detail.html) — album viewer with photo grid, select mode, lightbox
  - [album_picker.js](api/static/album_picker.js) — reusable modal for adding photos to albums from any page
    - `AlbumPicker` class with `open(imageIds)`, `close()`, `destroy()`
    - Used from search results and album detail pages
- **Navigation:** Albums link added to all page toolbars

### Thumbnail Performance Optimization (Feb 11, 2026)
- **Problem:** Opening a cluster with 100+ cached thumbnails took 1.5+ seconds
  - Root cause: `async def` endpoints blocked the asyncio event loop
  - All blocking I/O (`os.path.exists`, `FileResponse`, `load_image_any_format`) ran sequentially
  - Even cache HITs waited for any cache MISS to complete
- **Fix 1: `async def` → `def`** for image-serving endpoints
  - `/image/{image_id}/thumb`, `/image/{image_id}/full`, `/faces/{face_id}/thumb`
  - FastAPI runs `def` endpoints in threadpool (40 parallel threads vs 1 event loop)
  - Result: 1.5s → 300ms per thumbnail
- **Fix 2: In-memory LRU cache** (`ThumbnailMemoryCache` class)
  - 3-tier caching: **MEM** (Python dict) → **DISK** (bind mount) → **MISS** (generate)
  - Thread-safe `OrderedDict` with LRU eviction, 150 MB limit (~5000 thumbnails)
  - `X-Cache` header: `MEM` / `DISK` / `MISS` for debugging
  - Memory cache stats exposed in `/admin/cache/stats` response
  - Clear cache also clears memory cache
  - First cluster view: ~300ms/thumb (DISK). Repeat view: <1ms/thumb (MEM)
- **Removed:** `FileResponse` import — all responses now use `Response(content=bytes)`

## Not Implemented

- Video file indexing — detected and skipped
