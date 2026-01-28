-- Инициализация БД для smart_photo_indexing
-- Выполнить перед первым запуском приложения

-- Создать расширение pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Таблица индекса фотографий
CREATE TABLE IF NOT EXISTS photo_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id VARCHAR(256) UNIQUE NOT NULL,
    file_path VARCHAR(1024) UNIQUE NOT NULL,
    file_name VARCHAR(256) NOT NULL,
    file_size INTEGER,
    file_format VARCHAR(10),
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    modified_at TIMESTAMP DEFAULT NOW(),
    photo_date TIMESTAMP,
    clip_embedding vector(1152),  -- SigLIP so400m embedding
    exif_data JSONB,
    indexed INTEGER DEFAULT 0,
    indexed_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Таблица лиц
CREATE TABLE IF NOT EXISTS faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    face_id VARCHAR(256) UNIQUE NOT NULL,
    photo_id VARCHAR(256) NOT NULL,
    x1 FLOAT,
    y1 FLOAT,
    x2 FLOAT,
    y2 FLOAT,
    confidence FLOAT,
    age INTEGER,
    gender VARCHAR(1),
    emotion VARCHAR(20),
    ethnicity VARCHAR(50),
    landmarks JSON,
    face_embedding vector(512),  -- ArcFace embedding
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Таблица логов индексации
CREATE TABLE IF NOT EXISTS indexing_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP DEFAULT NOW(),
    operation VARCHAR(50),
    status VARCHAR(20),
    file_path VARCHAR(1024),
    error_message VARCHAR(1024),
    processing_time FLOAT,
    details JSONB DEFAULT '{}'
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_photo_index_image_id ON photo_index(image_id);
CREATE INDEX IF NOT EXISTS idx_photo_index_file_path ON photo_index(file_path);
CREATE INDEX IF NOT EXISTS idx_photo_index_indexed ON photo_index(indexed);
CREATE INDEX IF NOT EXISTS idx_photo_index_file_format ON photo_index(file_format);

CREATE INDEX IF NOT EXISTS idx_faces_face_id ON faces(face_id);
CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_age ON faces(age);
CREATE INDEX IF NOT EXISTS idx_faces_gender ON faces(gender);

CREATE INDEX IF NOT EXISTS idx_indexing_logs_timestamp ON indexing_logs(timestamp);

-- HNSW индексы для быстрого векторного поиска (cosine similarity)
-- Эти индексы значительно ускоряют поиск по эмбиддингам
CREATE INDEX IF NOT EXISTS idx_photo_clip_embedding_hnsw
    ON photo_index
    USING hnsw (clip_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_face_embedding_hnsw
    ON faces
    USING hnsw (face_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Примеры запросов для поиска:

-- Поиск похожих изображений по CLIP embedding (топ 10):
-- SELECT image_id, file_path, 1 - (clip_embedding <=> '[0.1, 0.2, ...]'::vector) as similarity
-- FROM photo_index
-- WHERE indexed = 1 AND clip_embedding IS NOT NULL
-- ORDER BY clip_embedding <=> '[0.1, 0.2, ...]'::vector
-- LIMIT 10;

-- Поиск похожих лиц по face embedding:
-- SELECT f.face_id, f.photo_id, p.file_path, 1 - (f.face_embedding <=> '[...]'::vector) as similarity
-- FROM faces f
-- JOIN photo_index p ON f.photo_id = p.image_id
-- WHERE f.face_embedding IS NOT NULL
-- ORDER BY f.face_embedding <=> '[...]'::vector
-- LIMIT 10;
