# Smart Photo Indexing Service

Сервис индексации домашнего хранилища фотографий с семантическим поиском по текстовому описанию.

## Возможности

- **Семантический поиск** — поиск по текстовому описанию на русском и английском (SigLIP so400m, мультиязычная модель)
- **Автоперевод** — опциональный перевод запросов на английский (Google Translate), отключается в UI
- **Поддержка RAW** — Nikon NEF, Canon CR2, Sony ARW, DNG и другие форматы через rawpy
- **HEIC/HEIF** — полная поддержка Apple фотографий через pillow-heif
- **PostgreSQL + pgvector** — быстрый векторный поиск с HNSW индексами (1152-dim)
- **GPU ускорение** — CUDA 12.4 + PyTorch 2.6 для индексации на GPU
- **Web UI** — адаптивный интерфейс для поиска с фильтрами и управлением файлами
- **Telegram бот** — поиск фотографий прямо из Telegram
- **Async reindex** — фоновая переиндексация через API с отслеживанием прогресса
- **Удаление в корзину** — безопасное удаление через Web UI и API (с сохранением структуры папок)
- **Поиск дубликатов** — обнаружение и удаление дубликатов по косинусному сходству CLIP-эмбеддингов

## Требования

- Docker с NVIDIA Container Toolkit (для GPU)
- PostgreSQL 15+ с расширением pgvector
- NVIDIA GPU с CUDA 12.4+ (опционально, можно CPU)

## Поддерживаемые форматы

| Формат | Расширения |
|--------|------------|
| JPEG | `.jpg`, `.jpeg` |
| PNG | `.png` |
| HEIC/HEIF | `.heic`, `.heif` |
| WebP, BMP | `.webp`, `.bmp` |
| Nikon RAW | `.nef` |
| Canon RAW | `.cr2` |
| Sony RAW | `.arw` |
| Adobe DNG | `.dng` |
| Fujifilm | `.raf` |
| Olympus | `.orf` |
| Panasonic | `.rw2` |

## Быстрый старт

### 1. Настройка PostgreSQL

```bash
psql -U dev -c "CREATE DATABASE smart_photo_index;"
psql -U dev -d smart_photo_index -f init_db.sql
```

### 2. Конфигурация

Скопируйте `.env.example` в `.env` и отредактируйте:

```env
# PostgreSQL
POSTGRES_USER=dev
POSTGRES_PASSWORD=secret
POSTGRES_DB=smart_photo_index
DATABASE_URL=postgresql://dev:secret@localhost:5432/smart_photo_index

# Путь к фотографиям (используйте / вместо \ на Windows)
PHOTOS_HOST_PATH=H:/PHOTO

# Модель (SigLIP по умолчанию)
CLIP_MODEL=SigLIP       # или ViT-B/32, ViT-B/16, ViT-L/14
CLIP_DEVICE=cuda         # или cpu

# Telegram бот (опционально)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_USERS=123456789,987654321
```

### 3. Запуск

```bash
docker-compose build
docker-compose up -d db         # PostgreSQL + pgvector
docker-compose up -d indexer    # индексация (GPU)
docker-compose up -d api        # API + Web UI на :8000
docker-compose up -d bot        # Telegram бот (опционально)
```

### 4. Использование

- **Web UI:** http://localhost:8000/
- **API docs:** http://localhost:8000/docs

## Модели

| Модель | Размерность | Качество | Скорость | Мультиязычность |
|--------|-------------|----------|----------|-----------------|
| **SigLIP so400m** | 1152 | Лучшее | ~3.5 img/s | Да (русский, английский и др.) |
| ViT-B/32 | 512 | Хорошее | ~15 img/s | Только английский |
| ViT-B/16 | 512 | Лучше | ~10 img/s | Только английский |
| ViT-L/14 | 768 | Отличное | ~5 img/s | Только английский |

По умолчанию используется **SigLIP so400m** (`google/siglip-so400m-patch14-384`) — мультиязычная модель с лучшим качеством поиска.

## API

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Статус сервисов |
| `/stats` | GET | Статистика индексации |
| `/models` | GET | Список доступных CLIP моделей |
| `/search/text` | POST | Поиск по текстовому описанию |
| `/search/image` | POST | Поиск похожих изображений |
| `/photo/{id}` | GET | Информация о фото |
| `/image/{id}/thumb` | GET | Миниатюра 400px |
| `/image/{id}/full` | GET | Полное изображение (max 2000px) |
| `/photos/delete` | POST | Удалить файлы в корзину |
| `/reindex` | POST | Запустить фоновую переиндексацию (опц. параметр `model`) |
| `/reindex/status` | GET | Прогресс переиндексации |
| `/duplicates` | POST | Найти дубликаты |
| `/duplicates` | DELETE | Найти и удалить дубликаты |

### Postman Collection

Для удобного тестирования API используйте готовую коллекцию:
- **Файл:** `Smart_Photo_Indexing_API.postman_collection.json`
- **Импорт:** File → Import в Postman
- **Содержит:** все endpoints с примерами запросов
- **Переменная окружения:** `{{base_url}}` = `http://localhost:8000`

### Примеры cURL

```bash
# Текстовый поиск (с автопереводом)
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "кошка на диване", "top_k": 10}'

# Текстовый поиск (без перевода, напрямую на SigLIP)
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "кошка на диване", "top_k": 10, "translate": false}'

# Запустить переиндексацию (модель по умолчанию)
curl -X POST http://localhost:8000/reindex

# Переиндексация конкретной моделью
curl -X POST "http://localhost:8000/reindex?model=SigLIP"

# Проверить прогресс
curl http://localhost:8000/reindex/status
```

## Web UI

Доступен по адресу http://localhost:8000/

### Функции интерфейса

- **Поиск** — текстовое описание на любом языке
- **Автоперевод EN** — чекбокс для включения/отключения автоперевода на английский
- **Фильтры по формату** — JPG, HEIC, PNG, NEF
- **Порог сходства** — слайдер 0–50%
- **Количество результатов** — 10/20/50/100
- **Режим выбора** — кнопка "Select" включает мультивыбор
- **Удаление** — выбранные файлы перемещаются в корзину (TRASH_DIR)
- **Lightbox** — клик по фото открывает полноразмерный просмотр

### Горячие клавиши

- `Enter` — выполнить поиск
- `Escape` — закрыть lightbox / выйти из режима выбора

## Архитектура

```
smart_photo_indexing/
├── main.py                 # Entry point (indexer daemon)
├── config/settings.py      # Pydantic settings (.env)
├── services/
│   ├── clip_embedder.py    # SigLIP/CLIP через HuggingFace transformers
│   ├── image_processor.py  # HEIC/JPG/PNG/RAW загрузка + EXIF
│   ├── indexer.py          # Оркестратор индексации (batch GPU)
│   ├── file_monitor.py     # Сканирование файловой системы
│   └── duplicate_finder.py # Поиск дубликатов по косинусному сходству
├── api/
│   ├── main.py             # FastAPI сервер + async reindex
│   └── static/index.html   # Web UI
├── bot/
│   └── telegram_bot.py     # Telegram бот для поиска
├── db/database.py          # SQLAlchemy + pgvector
├── models/data_models.py   # ORM модели
├── scripts/                # Утилиты (init_db, find_duplicates, cleanup)
├── docker-compose.yml      # 4 сервиса: db, indexer, api, bot
├── Dockerfile              # PyTorch 2.6 + CUDA 12.4
└── init_db.sql             # Схема БД + HNSW индексы
```

## Docker сервисы

| Сервис | Контейнер | Описание |
|--------|-----------|----------|
| `db` | smart_photo_db | PostgreSQL 15 + pgvector |
| `indexer` | smart_photo_indexer | Индексация на GPU (SigLIP) |
| `api` | smart_photo_api | FastAPI + Web UI на порту 8000 |
| `bot` | smart_photo_bot | Telegram бот |

## Troubleshooting

### GPU не доступен

```bash
docker run --rm --gpus all pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.is_available())"
```

### Windows TDR timeout

Если GPU "зависает" при индексации, уменьшите `BATCH_SIZE_CLIP` в `.env` до 16 или меньше.

### Логи

```bash
# Docker логи (WARNING+)
docker logs smart_photo_indexer -f
docker logs smart_photo_api -f

# Подробные логи (INFO) — в файле
# Windows: logs\indexer.log (маппинг через docker volume)
```

### Пересоздать БД

```bash
psql -U dev -c "DROP DATABASE smart_photo_index;"
psql -U dev -c "CREATE DATABASE smart_photo_index;"
psql -U dev -d smart_photo_index -f init_db.sql
```

## Планы развития

- [ ] Распознавание лиц (таблицы созданы, код не реализован)
- [ ] Индексация видео (ключевые кадры)
- [ ] Кластеризация по событиям/датам

## License

MIT
