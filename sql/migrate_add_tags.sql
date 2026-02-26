-- Migration: Add tags system with hidden photos support
-- Date: 2026-02-26
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_tags.sql

-- 1. Таблица тегов
CREATE TABLE IF NOT EXISTS tag (
    tag_id SERIAL PRIMARY KEY,
    name VARCHAR(64) NOT NULL UNIQUE,
    is_system BOOLEAN NOT NULL DEFAULT FALSE,
    color VARCHAR(7) NOT NULL DEFAULT '#6b7280',
    created_at TIMESTAMP DEFAULT NOW()
);

-- 2. Связь фото ↔ теги (many-to-many)
CREATE TABLE IF NOT EXISTS photo_tag (
    image_id INTEGER NOT NULL REFERENCES photo_index(image_id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tag(tag_id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (image_id, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_photo_tag_tag_id ON photo_tag(tag_id);
CREATE INDEX IF NOT EXISTS idx_photo_tag_image_id ON photo_tag(image_id);

-- 3. Денормализованный флаг скрытости на photo_index
--    Устанавливается в TRUE при добавлении системного тега, FALSE при удалении последнего
ALTER TABLE photo_index ADD COLUMN IF NOT EXISTS is_hidden BOOLEAN NOT NULL DEFAULT FALSE;

-- Частичный индекс — не замедляет основные запросы (большинство фото is_hidden=FALSE)
CREATE INDEX IF NOT EXISTS idx_photo_index_is_hidden ON photo_index(is_hidden) WHERE is_hidden = TRUE;

-- 4. Предустановленные служебные теги
INSERT INTO tag (name, is_system, color) VALUES
    ('private',  TRUE, '#8b5cf6'),
    ('trash',    TRUE, '#ef4444'),
    ('document', TRUE, '#f59e0b')
ON CONFLICT (name) DO NOTHING;
