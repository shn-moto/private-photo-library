-- Инициализация БД для smart_photo_indexing
-- Выполнить перед первым запуском приложения

-- Создать расширение pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Таблица индекса фотографий
CREATE TABLE IF NOT EXISTS photo_index (
    image_id SERIAL PRIMARY KEY,
    file_path VARCHAR(1024) UNIQUE NOT NULL,
    file_name VARCHAR(256) NOT NULL,
    file_size INTEGER,
    file_format VARCHAR(10),
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    modified_at TIMESTAMP DEFAULT NOW(),
    photo_date TIMESTAMP,
    -- Геолокация
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    -- Колонки для эмбеддингов разных моделей
    clip_embedding_vit_b32 vector(512),
    clip_embedding_vit_b16 vector(512),
    clip_embedding_vit_l14 vector(768),
    clip_embedding_siglip vector(1152),
    exif_data JSONB
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_photo_index_file_path ON photo_index(file_path);
CREATE INDEX IF NOT EXISTS idx_photo_index_file_format ON photo_index(file_format);

-- Индекс для геопоиска (фильтрация по bounding box)
CREATE INDEX IF NOT EXISTS idx_photo_index_geo
    ON photo_index (latitude, longitude)
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Индекс для поиска по дате съемки
CREATE INDEX IF NOT EXISTS idx_photo_index_photo_date
    ON photo_index (photo_date)
    WHERE photo_date IS NOT NULL;

-- HNSW индексы для каждой модели
CREATE INDEX IF NOT EXISTS idx_clip_siglip_hnsw
    ON photo_index USING hnsw (clip_embedding_siglip vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_clip_vit_b32_hnsw
    ON photo_index USING hnsw (clip_embedding_vit_b32 vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_clip_vit_b16_hnsw
    ON photo_index USING hnsw (clip_embedding_vit_b16 vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_clip_vit_l14_hnsw
    ON photo_index USING hnsw (clip_embedding_vit_l14 vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Примеры запросов для поиска:

-- Поиск похожих изображений по CLIP embedding (топ 10):
-- SELECT image_id, file_path, 1 - (clip_embedding <=> '[0.1, 0.2, ...]'::vector) as similarity
-- FROM photo_index
-- WHERE indexed = 1 AND clip_embedding IS NOT NULL
-- ORDER BY clip_embedding <=> '[0.1, 0.2, ...]'::vector
-- LIMIT 10;

