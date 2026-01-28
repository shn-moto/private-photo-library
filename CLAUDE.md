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
clip_embedding vector(1152)  -- SigLIP so400m embedding
file_path, file_name, file_size, width, height
exif_data JSONB, indexed INTEGER, indexed_at TIMESTAMP

-- faces: таблица лиц (пока не используется)
face_embedding vector(512), bbox coords, attributes

-- Indexes: HNSW (vector_cosine_ops) для быстрого поиска
```

## API Endpoints

```
GET    /health                  # service status
GET    /stats                   # indexed photos count
POST   /search/text             # {"query": "cat on sofa", "top_k": 10, "translate": true}
POST   /search/image            # multipart file upload (find similar)
GET    /photo/{image_id}        # photo details
GET    /image/{image_id}/thumb  # thumbnail 400px (JPEG)
GET    /image/{image_id}/full   # full image max 2000px (JPEG)
POST   /photos/delete           # {"image_ids": ["id1", "id2"]} - move to TRASH_DIR
POST   /reindex                 # async: scan storage, cleanup missing, index new files
GET    /reindex/status           # reindex progress (running, total, indexed, percentage)
POST   /duplicates              # find duplicates (JSON body: threshold, limit, path_filter)
DELETE /duplicates              # find & delete duplicates (query: threshold, path_filter)
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

## Indexer Behavior

- On startup: scans PHOTOS_HOST_PATH, indexes new files in batches of 16 on GPU
- **Upsert logic:** if record exists in DB (by file_path) — UPDATE embedding; otherwise INSERT
- `get_indexed_paths()` filters by `indexed=1` only
- Automatic monitoring is disabled; use `POST /reindex` for manual re-indexing
- Console logs: WARNING+; detailed INFO logs in `/logs/indexer.log`
- After initial indexing, enters idle loop (`while True: sleep(3600)`)

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

## Not Implemented Yet

- Face detection and recognition (tables exist, code disabled)
- Video file indexing (detected and skipped)
