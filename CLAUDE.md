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
2. **API /files/unindexed** — проверяет файлы без эмбеддингов в БД
3. **Gzip + Multipart** — отправляет список файлов в API (100k файлов = 0.4 MB)
4. **API /reindex/files** — индексация в Docker с GPU

### Fallback при ошибках

- Если USN Journal недоступен → os.scandir (~12 сек на 100k файлов)
- Если Everything запущен → Everything SDK (~1 сек на 100k файлов)
- Если индексация была прервана → автоматически доиндексирует из /files/unindexed

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
│   └── duplicate_finder.py # Duplicate detection & deletion (cosine similarity)
├── api/
│   ├── main.py             # FastAPI endpoints + async reindex
│   └── static/
│       ├── index.html      # Web UI (search page)
│       ├── map.html        # Photo map with clusters (Leaflet)
│       └── results.html    # Cluster results page
├── bot/
│   └── telegram_bot.py     # Telegram bot for photo search
├── db/
│   └── database.py         # SQLAlchemy + pgvector
├── models/
│   └── data_models.py      # Pydantic + ORM models
├── scripts/
│   ├── fast_reindex.py     # Main indexing script (run from Windows host)
│   ├── find_duplicates.py  # CLI: find duplicates & generate report
│   ├── init_db.py          # DB initialization script
│   ├── populate_exif_data.py # Extract EXIF/GPS from all photos in DB
│   ├── start_bot.sh        # Bot startup script (waits for cloudflared tunnel)
│   ├── test_cleanup.py     # Test cleanup logic
│   └── test_db.py          # Test DB connection
├── util/
│   ├── cleanup_orphaned.py # CLI: remove DB records for missing files
│   └── fix_video_extensions.py  # Rename misnamed video files
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

    exif_data JSONB
);

-- HNSW индексы для каждой модели (cosine similarity)
CREATE INDEX idx_clip_siglip_hnsw ON photo_index USING hnsw (clip_embedding_siglip vector_cosine_ops);
CREATE INDEX idx_clip_vit_b32_hnsw ON photo_index USING hnsw (clip_embedding_vit_b32 vector_cosine_ops);
CREATE INDEX idx_clip_vit_b16_hnsw ON photo_index USING hnsw (clip_embedding_vit_b16 vector_cosine_ops);
CREATE INDEX idx_clip_vit_l14_hnsw ON photo_index USING hnsw (clip_embedding_vit_l14 vector_cosine_ops);

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
```

**Изменения в схеме БД:**
- **Удалены колонки:** `id` (UUID), `clip_embedding` (legacy), `clip_model`, `indexed`, `indexed_at`, `meta_data`
- **Мульти-модельная поддержка:** каждая CLIP модель хранится в отдельной колонке с правильной размерностью
- **image_id** - единственный первичный ключ (SERIAL, автоинкремент)
- **Проверка индексации:** `WHERE <embedding_column> IS NOT NULL` вместо `indexed=1`
- **Удалена таблица `faces`** и все связанные функции распознавания лиц

**Миграция:**
```bash
# 1. Создать новые колонки и перенести данные
psql -U dev -d smart_photo_index -f scripts/migrate_multi_model.sql

# 2. Удалить legacy колонки (после проверки)
psql -U dev -d smart_photo_index -f scripts/cleanup_legacy_columns.sql
```

-- Indexes: HNSW (vector_cosine_ops) для быстрого поиска
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
GET    /photo/{image_id}        # photo details (БЕЗ данных о лицах)
GET    /image/{image_id}/thumb  # thumbnail 400px (JPEG)
GET    /image/{image_id}/full   # full image max 2000px (JPEG)
POST   /photos/delete           # {"image_ids": [123, 456]} - move to TRASH_DIR
POST   /reindex/files           # multipart gzipped JSON file list + model param (used by fast_reindex.py)
GET    /reindex/status          # reindex progress (running, total, indexed, percentage, model)
GET    /files/unindexed?model=X # files without embeddings for model (used by fast_reindex.py)
GET    /scan/checkpoint/{drive} # get USN checkpoint for drive (e.g., "H:")
POST   /scan/checkpoint         # save USN checkpoint {drive_letter, last_usn, files_count}
POST   /duplicates              # find duplicates (JSON: threshold, limit, path_filter)
DELETE /duplicates              # find & delete duplicates (query: threshold, path_filter)

# Map API (геолокация)
GET    /map/stats               # статистика по гео-данным (with_gps, date_range, geo_bounds)
POST   /map/clusters            # кластеры для карты {"min_lat", "max_lat", "min_lon", "max_lon", "zoom", "date_from?", "date_to?"}
GET    /map/photos              # фото в bounding box (query: min_lat, max_lat, min_lon, max_lon, date_from?, date_to?, limit, offset)
POST   /map/search              # текстовый поиск в географической области (query params: min_lat..., body: TextSearchRequest)
```

**Изменения в API:**
- Все поиски и статистика работают с моделью, указанной в `CLIP_MODEL` (.env)
- `SearchResult.image_id` теперь `int` (было `str`)
- Удалены endpoints для работы с лицами: `/search/face`, `/search/face/attributes`
- Ответы поиска включают `model` для отображения используемой модели
- Ответы текстового поиска включают `translated_query` если запрос был переведен
```

**Note:** Face search endpoints exist but are disabled (not implemented yet).

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
- Lightbox preview (click on photo)
- Format badge on each thumbnail
- **Navigation** — links between Search and Map pages

## Map UI

Available at `http://localhost:8000/map.html` when API is running.

**Features:**
- World map with photo clusters (Leaflet.js + CartoDB Dark theme)
- **Date filters** — From/To date pickers for filtering photos
- **Server-side clustering** — clusters adapt to zoom level
- **Click on cluster** — zoom in or open photos in new tab
- **Photos view** (results.html) — gallery with pagination
- **Text search within area** — CLIP search limited to geographic bounds
- Lightbox preview on results page
- **Fullscreen mode** — button in toolbar to hide UI and maximize map
  - Native Fullscreen API on desktop/Android
  - CSS fallback on iOS (hides toolbar, maximizes map)

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
- **Удалена таблица `faces`:** все функции распознавания лиц полностью удалены из кодовой базы
- **Новая логика индексации:** проверка `WHERE <embedding_column> IS NOT NULL` вместо `indexed=1`

### Code Changes
- **API:** все endpoints обновлены для работы с мульти-модельной схемой
  - `/stats` показывает статистику по каждой модели
  - Ответы поиска включают `model` и `translated_query` (если применимо)
  - `SearchResult.image_id` теперь `int` вместо `str`
  - Удалены endpoints для лиц: `/search/face`, `/search/face/attributes`
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

## Not Implemented / Removed

- **Face detection and recognition** — полностью удалено из кодовой базы
- Video file indexing — detected and skipped
