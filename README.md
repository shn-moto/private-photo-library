# Smart Photo Indexing Service

Сервис индексации домашнего хранилища фотографий с семантическим поиском по текстовому описанию.

## Возможности

- **Семантический поиск** — поиск по текстовому описанию на русском и английском (SigLIP so400m, мультиязычная модель)
- **Поиск по изображению** — найти похожие фотографии, загрузив картинку (кнопка с иконкой фото)
- **Автоперевод** — опциональный перевод запросов на английский (Google Translate), отключается в UI
- **Распознавание лиц** — детекция и индексация лиц через InsightFace buffalo_l с HNSW индексами; переиндексация отдельного фото прямо из лайтбокса с настройкой порогов и HD-режимом (1280px)
- **Управление персонами** — создание персон, привязка лиц, автоназначение похожих лиц
- **Поиск по лицам** — найти похожие лица, фото с конкретным человеком
- **Комбинированный поиск** — "Таня в горах" (персона + CLIP запрос)
- **Мульти-модельный RRF** — Reciprocal Rank Fusion по 4 CLIP моделям для максимального качества поиска
- **AI помощник (Gemini)** — умный поиск на естественном языке ("найди Сашу в Камбоджек", "закат на пляже")
- **Поворот фотографий** — неразрушающий поворот в lightbox (кнопки ↺↻); хранится в `exif_data["UserRotation"]`, оригинал не трогается; bbox лиц пересчитываются автоматически
- **Поддержка RAW** — Nikon NEF, Canon CR2, Sony ARW, DNG и другие форматы через rawpy
- **HEIC/HEIF** — полная поддержка Apple фотографий через pillow-heif
- **PostgreSQL + pgvector** — быстрый векторный поиск с HNSW индексами (CLIP 1152-dim, лица 512-dim)
- **GPU ускорение** — CUDA 12.4 + PyTorch 2.6 для индексации на GPU
- **Web UI** — адаптивный интерфейс с детекцией лиц, управлением персонами
- **Хронологическая лента** — Google Photos-style feed с justified grid, группировкой по дням и infinite scroll
- **Telegram бот** — поиск фотографий прямо из Telegram; команды `/feed` (лента) и `/map` (карта) открывают Web UI через защищённый туннель
- **Async reindex** — фоновая переиндексация через API с отслеживанием прогресса
- **Удаление в корзину** — безопасное удаление через Web UI и API (с сохранением структуры папок)
- **Поиск дубликатов** — обнаружение и удаление дубликатов по косинусному сходству CLIP-эмбеддингов
- **Теги и фильтрация** — пользовательские и системные теги с 3-состояниями фильтра (включение/исключение/выкл); скрытие фото с системными тегами; массовое назначение
- **Привязка GPS** — геокодирование текстовых адресов (Nominatim + Gemini AI fallback), поддержка координат, DMS, Google Maps URL; массовое назначение GPS из режима выбора
- **Ctrl+Click на 🌐** — копирование GPS-координат в буфер обмена; Ctrl+Hover меняет иконку на 📋; зелёный toast подтверждения
- **Ctrl+Drag кластеров на карте (admin)** — перетаскивание кластерных маркеров для переназначения GPS всем фото в кластере
- **CLIP → Тег** — автоматическое присвоение тегов по CLIP-запросу (admin UI): "документ", "скриншот", "мем"

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
psql -U dev -d smart_photo_index -f sql/init_db.sql
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
PHOTOS_HOST_PATH=D:/PHOTO

# Модель (SigLIP по умолчанию)
CLIP_MODEL=SigLIP       # или ViT-B/32, ViT-B/16, ViT-L/14
CLIP_DEVICE=cuda         # или cpu

# Telegram бот (опционально)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_USERS=123456789,987654321

# Gemini AI помощник (опционально)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash    # или gemini-2.0-flash
```

### 3. Запуск

```bash
docker-compose build
docker-compose up -d db         # PostgreSQL + pgvector
docker-compose up -d api        # API + Web UI + индексация (GPU)
# Опционально: публичный туннель для доступа к карте
docker-compose up -d cloudflared # cloudflared (trycloudflare)
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
| **Основные** | | |
| `/health` | GET | Статус сервисов (БД, CLIP, лица) |
| `/stats` | GET | Статистика индексации (`failed_count` включён) |
| `/models` | GET | Список доступных CLIP моделей |
| **CLIP поиск** | | |
| `/search/text` | POST | Поиск по тексту (multi_model, tag_ids, exclude_tag_ids, include_hidden) |
| `/search/image` | POST | Поиск похожих изображений |
| **Фото** | | |
| `/photo/{id}` | GET | Информация о фото |
| `/photo/{id}/faces` | GET | Все лица на фото с bbox |
| `/image/{id}/thumb` | GET | Миниатюра 400px |
| `/image/{id}/full` | GET | Полное изображение (max 2000px) |
| `/photos/delete` | POST | Удалить файлы в корзину |
| **Лента** | | |
| `/timeline/photos` | GET | Хронологическая лента (sort: date DESC, limit/offset) |
| **Теги** | | |
| `/tags` | GET | Список всех тегов |
| `/tags` | POST | Создать тег (системные — admin only) |
| `/tags/{id}` | DELETE | Удалить тег (системные — admin only) |
| `/photo/{id}/tags` | GET/POST/DELETE | Теги фото (get/add/remove) |
| `/photos/tags/bulk` | POST | Массовое добавление/удаление тегов |
| **Индексация** | | |
| `/reindex` | POST | Запустить CLIP индексацию (опц. `model`) |
| `/reindex/status` | GET | Прогресс CLIP индексации |
| `/faces/reindex` | POST | Запустить индексацию лиц (`skip_indexed`, `batch_size`) |
| `/faces/reindex/status` | GET | Прогресс индексации лиц |
| `/faces/stats` | GET | Статистика по лицам и персонам |
| **Поиск по лицам** | | |
| `/search/face` | POST | Поиск по загруженному лицу |
| `/search/face/by_id/{id}` | POST | Найти похожие лица по face_id |
| `/search/person-clip` | POST | Комбинированный: персона + CLIP запрос |
| **Персоны** | | |
| `/persons` | GET | Список персон (опц. `search`, `limit`, `offset`) |
| `/persons` | POST | Создать персону |
| `/persons/{id}` | GET | Детали персоны |
| `/persons/{id}` | PUT | Обновить персону |
| `/persons/{id}` | DELETE | Удалить персону |
| `/persons/{id}/merge/{target}` | POST | Объединить две персоны |
| `/persons/{id}/photos` | GET | Фото с этой персоной |
| `/persons/{id}/auto-assign` | POST | Авто-привязка похожих лиц |
| **Привязка лиц** | | |
| `/faces/{id}/assign` | POST | Привязать лицо к персоне (или создать новую) |
| `/faces/{id}/assign` | DELETE | Отвязать лицо от персоны |
| **Дубликаты** | | |
| `/duplicates` | POST | Найти дубликаты |
| `/duplicates` | DELETE | Найти и удалить дубликаты |
| **AI помощник** | | |
| `/ai/clip-prompt` | POST | Оптимизация запроса для CLIP через Gemini |
| `/ai/assistant` | POST | AI помощник для карты (NL → фильтры) |
| `/ai/search-assistant` | POST | AI помощник для поиска (NL → фильтры + RRF) |
| **Admin** | | |
| `/admin/gpu/stats` | GET | GPU: VRAM used/free/util/temp + загруженные модели |
| `/admin/models/status` | GET | Статус всех 4 CLIP моделей (loaded/unloaded, память) |
| `/admin/models/warm` | POST | Загрузить модель в GPU память `{"model": "SigLIP"}` |
| `/admin/models/unload` | POST | Выгрузить модель из GPU памяти `{"model": "SigLIP"}` |
| `/admin/failed-files` | GET | Список файлов с ошибкой индексации (limit опц.) |
| `/admin/failed-files/reset` | POST | Сбросить флаг index_failed (всё или по image_ids) |
| `/admin/index-all` | POST | Очередь: CLIP + лица + pHash |
| `/admin/index-all/status` | GET | Статус очереди + прогресс текущей задачи |
| `/admin/cache/stats` | GET | Статистика кэша миниатюр |
| `/admin/cache/warm` | POST | Прогреть кэш (тяжёлые форматы) |
| `/admin/clip-tag-assign` | POST | Найти фото по CLIP и присвоить тег (admin) |
| `/admin/shutdown-flag` | POST/GET | Флаг выключения ПК после индексации |
| **Геокодирование** | | |
| `/geo/geocode` | POST | Текстовый адрес → GPS координаты (Nominatim + Gemini) |
| `/geo/assign` | POST | Привязать GPS к фото |

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

# === Индексация лиц ===
# Запустить индексацию лиц (пропустить уже обработанные)
curl -X POST "http://localhost:8000/faces/reindex?skip_indexed=true&batch_size=8"

# Проверить прогресс индексации лиц
curl http://localhost:8000/faces/reindex/status

# === Работа с персонами ===
# Создать персону
curl -X POST http://localhost:8000/persons \
  -H "Content-Type: application/json" \
  -d '{"name": "Таня", "description": "Моя сестра"}'

# Получить список персон
curl http://localhost:8000/persons

# === Поиск ===
# Найти фото Тани в горах (комбинированный поиск)
curl -X POST http://localhost:8000/search/person-clip \
  -H "Content-Type: application/json" \
  -d '{"person_id": 1, "query": "в горах", "top_k": 20}'

# Получить лица на конкретном фото
curl http://localhost:8000/photo/123/faces
```

## Web UI

Доступен по адресу http://localhost:8000/

### Функции интерфейса

- **Поиск** — текстовое описание на любом языке
- **Автоперевод EN** — чекбокс для включения/отключения автоперевода на английский
- **Фильтры по формату** — JPG, HEIC, PNG, NEF
- **Фильтр по тегам** — 3-состояния: включение (✓ зелёный), исключение (✗ красный), выкл; создание новых тегов inline
- **Порог сходства** — слайдер 0–50%
- **Количество результатов** — 10/20/50/100
- **Режим выбора** — кнопка "Select" включает мультивыбор
- **Удаление** — выбранные файлы перемещаются в корзину (TRASH_DIR)
- **Lightbox с лицами:**
  - Клик по фото открывает полноразмерный просмотр
  - Кнопки ↺↻ — неразрушающий поворот (90°/180°/270°), хранится в EXIF metadata
  - Кнопка "👤" включает/отключает отображение обнаруженных лиц
  - Клик по bbox лица открывает popup с информацией (возраст, пол, персона)
  - Можно привязать лицо к существующей персоне или создать новую
  - Зелёные рамки — назначенные лица, жёлтые пунктирные — неназначенные
- **Теги на фото** — цветные плашки на миниатюрах и в lightbox; добавление/удаление тегов в lightbox; массовое присвоение через режим выбора
- **AI помощник** — кнопка ✨ в тулбаре открывает чат с Gemini
  - Опишите что ищете на естественном языке: "закат на пляже", "дети в парке"
  - AI устанавливает фильтры и запускает мульти-модельный RRF поиск
- **Хронологическая лента** (`/timeline.html`) — Google Photos-style просмотр всего архива:
  - Justified grid layout: фото в строках одинаковой высоты, ширина заполняет экран
  - Группировка по дням с русскими заголовками ("Сегодня · 25 февраля 2026")
  - Infinite scroll, адаптивная высота строк (120–240px в зависимости от экрана)
  - Полный лайтбокс: GPS, альбомы, лица, поворот, клавиши ←→Esc, свайп на телефоне
  - Ролевой доступ к лицам: обычные пользователи видят рамки и имена, менять привязку не могут

### Горячие клавиши

- `Enter` — выполнить поиск
- `Escape` — закрыть popup/lightbox / выйти из режима выбора

## Архитектура

```
smart_photo_indexing/
├── main.py                 # Entry point (indexer daemon)
├── config/settings.py      # Pydantic settings (.env)
├── services/
│   ├── clip_embedder.py    # SigLIP/CLIP через HuggingFace transformers
│   ├── face_embedder.py    # InsightFace buffalo_l (детекция + эмбеддинги 512-dim)
│   ├── face_indexer.py     # Индексация лиц (batch GPU, pgvector HNSW)
│   ├── person_service.py   # Управление персонами, привязка лиц
│   ├── image_processor.py  # HEIC/JPG/PNG/RAW загрузка + EXIF
│   ├── indexer.py          # Оркестратор индексации (batch GPU)
│   ├── file_monitor.py     # Сканирование файловой системы
│   └── duplicate_finder.py # Поиск дубликатов по косинусному сходству
├── api/
│   ├── main.py             # FastAPI: CLIP search, face detection, person management
│   └── static/
│       ├── index.html      # Web UI с детекцией лиц и person management
│       └── map.html        # Карта фотографий (GPS геолокация)
├── bot/
│   └── telegram_bot.py     # Telegram бот для поиска
├── db/database.py          # SQLAlchemy + pgvector
├── models/data_models.py   # ORM модели (PhotoIndex, Face, Person)
├── scripts/                # Утилиты индексации и БД
├── sql/                    # Схема БД и миграции
├── docker-compose.yml      # 4 сервиса: db, indexer, api, bot
├── Dockerfile              # PyTorch 2.6 + CUDA 12.4 + InsightFace
└── sql/init_db.sql         # Схема БД + HNSW индексы (CLIP + faces)
```

## Docker сервисы

| Сервис | Контейнер | Описание |
|--------|-----------|----------|
| `db` | smart_photo_db | PostgreSQL 15 + pgvector |
| `api` | smart_photo_api | FastAPI + Web UI на порту 8000 (CLIP индексация выполняется внутри `api`) |
| `cloudflared` | smart_photo_tunnel | optional: quick tunnel (trycloudflare) — публичный доступ к карте |
| `bot` | smart_photo_bot | Telegram бот |

## Troubleshooting

### GPU не доступен

```bash
docker run --rm --gpus all pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.is_available())"
```

### Windows TDR timeout

Если GPU "зависает" при индексации, уменьшите `BATCH_SIZE_CLIP` в `docker-compose.yml`.
По умолчанию `16` — безопасно при динамическом управлении моделями (перед индексацией другие модели выгружаются автоматически). Для очень слабых GPU попробуйте `8` или `4`.

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
psql -U dev -d smart_photo_index -f sql/init_db.sql
```

## Реализовано

- ✅ **Распознавание лиц** — InsightFace buffalo_l, pgvector HNSW индексы
- ✅ **Управление персонами** — создание, редактирование, объединение
- ✅ **Привязка лиц** — ручная и автоматическая по сходству эмбеддингов
- ✅ **Комбинированный поиск** — "Таня в горах" (персона + CLIP)
- ✅ **Web UI для лиц** — отображение bbox, привязка персон в lightbox
- ✅ **Поддержка GPS** — карта фотографий (map.html)
- ✅ **pHash (perceptual hash)** — 256‑битный pHash, поиск и удаление точных/near-duplicates + host-скрипт `compute_phash.py`
- ✅ **Альбомы (Albums)** — CRUD API + UI (альбомы, просмотр, добавление фото)
- ✅ **Кэш миниатюр на диске + прогрев** — `THUMB_CACHE_DIR`, `admin/cache` endpoints, cache warm/clear
- ✅ **Admin UI** — Index All очередь, управление CLIP/Faces/pHash, cache warm
- ✅ **Mobile UI improvements** — drawer system, selection bar, instant filters, translucent panels
- ✅ **AI помощник (Gemini)** — умный поиск на естественном языке (chat-based, карта + поиск)
- ✅ **Мульти-модельный RRF** — Reciprocal Rank Fusion по всем CLIP моделям для лучшего качества
- ✅ **Миграции БД** — оптимизация индексации (флаг faces_indexed)
- ✅ **index_failed flag** — битые/нечитаемые файлы помечаются в БД, пропускаются при повторной индексации; Failed Files UI в Admin
- ✅ **Динамическое управление моделями** — lazy-load при старте (только одна модель); ручная загрузка/выгрузка через Admin UI; auto-unload перед индексацией освобождает VRAM
- ✅ **GPU Stats panel** — VRAM bar с цветовой индикацией, использование памяти по каждой модели, температура GPU
- ✅ **Поворот фотографий** — неразрушающий поворот в lightbox (кнопки ↺↻); хранится в `exif_data["UserRotation"]`; bbox лиц пересчитываются математически; rotation-aware кэш миниатюр (3-tier: memory/disk/browser)
- ✅ **Улучшенный UX выбора** — панель выбора (В альбом / Найти похожие / Удалить) вынесена в единую нижнюю панель для всех устройств; на PC/планшете — иконка+текст, на телефоне — только иконки; анимация поиска (skeleton-карточки + спиннер на кнопке)
- ✅ **Хронологическая лента (timeline.html)** — Google Photos-style feed: justified grid, группировка по дням, infinite scroll, полный лайтбокс с лицами/альбомами/поворотом, ролевой доступ (лица read-only для не-admin); Telegram команда `/feed`
- ✅ **Теги и фильтрация** — пользовательские/системные теги с 3-состояниями фильтра (включить/исключить); скрытие фото с системными тегами; массовое назначение (bulk)
- ✅ **Привязка GPS** — геокодирование адресов (Nominatim + Gemini AI); поддержка координат, DMS, Google Maps URL; массовое назначение из режима выбора; live-обновление в lightbox
- ✅ **CLIP → Тег (admin)** — поиск по CLIP-запросу и массовое присвоение тега; превью → подтверждение; multi-model RRF
- ✅ **Ctrl+Click GPS копирование** — Ctrl+Click на 🌐 копирует координаты в буфер; Ctrl+Hover меняет иконку на 📋; на всех страницах
- ✅ **Ctrl+Drag кластеров (admin)** — перетаскивание кластеров на карте для GPS-переназначения всех фото в кластере

## Планы развития

- [ ] Индексация видео (ключевые кадры)
- [ ] Кластеризация по событиям/датам
- [ ] Обучение CLIP на именах персон ("Таня" → эмбеддинги лиц Тани)

## License

MIT
