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
    exif_data JSONB,
    -- Флаг индексации лиц (для оптимизации skip_indexed)
    faces_indexed INTEGER NOT NULL DEFAULT 0
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_photo_index_file_path ON photo_index(file_path);
CREATE INDEX IF NOT EXISTS idx_photo_index_file_format ON photo_index(file_format);
CREATE INDEX IF NOT EXISTS idx_photo_index_faces_indexed ON photo_index(faces_indexed);

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

-- ============================================
-- Таблица персон (люди на фотографиях)
-- ============================================
CREATE TABLE IF NOT EXISTS person (
    person_id SERIAL PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    description TEXT,
    cover_face_id INTEGER,  -- Лучшее лицо для аватара (FK добавляется позже)
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_person_name ON person(name);
CREATE INDEX IF NOT EXISTS idx_person_name_lower ON person(LOWER(name));

-- ============================================
-- Таблица лиц на фотографиях
-- ============================================
CREATE TABLE IF NOT EXISTS faces (
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

-- Индексы для связей
CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);

-- Индекс для неназначенных лиц
CREATE INDEX IF NOT EXISTS idx_faces_unassigned ON faces(image_id) WHERE person_id IS NULL;

-- HNSW индекс для быстрого поиска похожих лиц
CREATE INDEX IF NOT EXISTS idx_faces_embedding_hnsw
    ON faces USING hnsw (face_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Добавить FK для person.cover_face_id
ALTER TABLE person
    DROP CONSTRAINT IF EXISTS fk_person_cover_face;

ALTER TABLE person
    ADD CONSTRAINT fk_person_cover_face
    FOREIGN KEY (cover_face_id) REFERENCES faces(face_id) ON DELETE SET NULL;

-- ============================================
-- View: персоны со статистикой
-- ============================================
CREATE OR REPLACE VIEW person_with_stats AS
SELECT
    p.person_id,
    p.name,
    p.description,
    p.cover_face_id,
    p.created_at,
    p.updated_at,
    COUNT(f.face_id) AS face_count,
    COUNT(DISTINCT f.image_id) AS photo_count
FROM person p
LEFT JOIN faces f ON f.person_id = p.person_id
GROUP BY p.person_id, p.name, p.description, p.cover_face_id, p.created_at, p.updated_at;

-- ============================================
-- Примеры запросов для поиска:

-- Поиск похожих изображений по CLIP embedding (топ 10):
-- SELECT image_id, file_path, 1 - (clip_embedding <=> '[0.1, 0.2, ...]'::vector) as similarity
-- FROM photo_index
-- WHERE indexed = 1 AND clip_embedding IS NOT NULL
-- ORDER BY clip_embedding <=> '[0.1, 0.2, ...]'::vector
-- LIMIT 10;

