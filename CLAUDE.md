# Smart Photo Indexing - Claude Context

## Overview

Сервис индексации домашнего фотоархива с семантическим поиском по текстовому описанию (SigLIP).

**Stack:** Python 3.11 + PyTorch 2.6 + HuggingFace Transformers + PostgreSQL/pgvector + FastAPI + Docker (GPU)

## Quick Start

```bash
# 1. БД (один раз)
psql -U dev -d smart_photo_index -f init_db.sql

# 2. Сборка и запуск
docker-compose build
docker-compose up -d db         # PostgreSQL + pgvector
docker-compose up -d indexer    # индексация
docker-compose up -d api        # API + Web UI на :8000
docker-compose up -d bot        # Telegram бот

# 3. Web UI
http://localhost:8000/
```

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
│   └── static/index.html   # Web UI (adaptive layout)
├── bot/
│   └── telegram_bot.py     # Telegram bot for photo search
├── db/
│   └── database.py         # SQLAlchemy + pgvector
├── models/
│   └── data_models.py      # Pydantic + ORM models
├── scripts/
│   ├── init_db.py          # DB initialization script
│   ├── fix_video_extensions.py  # Rename misnamed video files
│   ├── find_duplicates.py  # CLI: find duplicates & generate report
│   ├── cleanup_orphaned.py # CLI: remove DB records for missing files
│   ├── test_cleanup.py     # Test cleanup logic
│   └── test_db.py          # Test DB connection
├── reference/              # Reference scripts (not used in production)
├── docker-compose.yml      # 4 services: db, indexer, api, bot
├── Dockerfile              # PyTorch 2.6 + CUDA 12.4
├── init_db.sql             # DB schema + HNSW indexes (1152-dim)
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
| `docker-compose.yml` | 4 services (db, indexer, api, bot) with GPU |
| `Dockerfile` | Base: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` |
| `init_db.sql` | DB schema + HNSW indexes for pgvector (1152-dim) |
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
GET    /stats                   # indexed photos count BY MODEL (показывает статистику по каждой модели)
POST   /search/text             # {"query": "cat on sofa", "top_k": 10, "translate": true, "formats": ["jpg", "heic"]}
                                # Response: {results: [...], translated_query: str, model: str}
POST   /search/image            # multipart file upload (find similar)
                                # Response: {results: [...], model: str}
GET    /photo/{image_id}        # photo details (БЕЗ данных о лицах)
GET    /image/{image_id}/thumb  # thumbnail 400px (JPEG)
GET    /image/{image_id}/full   # full image max 2000px (JPEG)
POST   /photos/delete           # {"image_ids": [123, 456]} - move to TRASH_DIR
POST   /reindex                 # async: scan storage, cleanup missing, index new files
GET    /reindex/status          # reindex progress (running, total, indexed, percentage) - для ТЕКУЩЕЙ модели
POST   /duplicates              # find duplicates (JSON: threshold, limit, path_filter) - использует ТЕКУЩУЮ модель
DELETE /duplicates              # find & delete duplicates (query: threshold, path_filter) - использует ТЕКУЩУЮ модель
```

**Изменения в API:**
- Все поиски и статистика работают с моделью, указанной в `CLIP_MODEL` (.env)
- `SearchResult.image_id` теперь `int` (было `str`)
- Удалены endpoints для работы с лицами: `/search/face`, `/search/face/attributes`
- Ответы поиска включают `model` для отображения используемой модели
- Ответы текстового поиска включают `translated_query` если запрос был переведен
```

**Note:** Face search endpoints exist but are disabled (not implemented yet).

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

## Indexer Behavior (Multi-Model Support)

**File:** [services/indexer.py](services/indexer.py)

- On startup: scans PHOTOS_HOST_PATH, indexes new files in batches of 16 on GPU
- **Multi-model support:** индексер сохраняет эмбеддинги в колонку для текущей модели (из .env)
- **Upsert logic:** if record exists in DB (by file_path) — UPDATE embedding column; otherwise INSERT
- `get_indexed_paths()` фильтрует по наличию эмбеддинга для текущей модели (`WHERE <column> IS NOT NULL`)
- Automatic monitoring is disabled; use `POST /reindex` for manual re-indexing
- Console logs: WARNING+; detailed INFO logs in `/logs/indexer.log`
- After initial indexing, enters idle loop (`while True: sleep(3600)`)

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

**Telegram Bot Changes:** [bot/telegram_bot.py](bot/telegram_bot.py)
- Добавлен параметр `BOT_FORMATS` для фильтрации форматов (по умолчанию: jpg,jpeg,heic,heif,nef)
- Отправляет полноразмерные изображения вместо thumbnails
- API response теперь имеет структуру `{results: [...], model: ...}` вместо массива

## Common Tasks

### Rebuild & restart
```bash
docker-compose build --no-cache indexer api
docker-compose up -d indexer api
```

### View logs
```bash
docker logs smart_photo_indexer -f    # WARNING+ only
# Detailed logs (INFO):
# Windows: logs\indexer.log
```

### Recreate database
```bash
psql -U dev -c "DROP DATABASE smart_photo_index;"
psql -U dev -c "CREATE DATABASE smart_photo_index;"
psql -U dev -d smart_photo_index -f init_db.sql
```

### Migrate to multi-model schema
```bash
# 1. Add new columns and migrate data
psql -U dev -d smart_photo_index -f scripts/migrate_multi_model.sql

# 2. Cleanup legacy columns (after verification)
psql -U dev -d smart_photo_index -f scripts/cleanup_legacy_columns.sql

# 3. Reindex with new model (if needed)
# Set CLIP_MODEL in .env, then:
docker-compose up -d indexer
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
- `scripts/migrate_multi_model.sql` — создание новых колонок и миграция данных
- `scripts/cleanup_legacy_columns.sql` — удаление устаревших колонок
- `scripts/cleanup_orphaned.py` — удаление записей для несуществующих файлов (обновлен)
- `scripts/find_duplicates.py` — поиск дубликатов с поддержкой выбора модели

## Not Implemented / Removed

- **Face detection and recognition** — полностью удалено из кодовой базы
- Video file indexing — detected and skipped
