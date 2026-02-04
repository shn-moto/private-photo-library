-- Удаление устаревших колонок из photo_index
-- Эти колонки больше не используются в коде

BEGIN;

-- Удаляем индекс для indexed
DROP INDEX IF EXISTS idx_photo_index_indexed;

-- Удаляем legacy колонки
ALTER TABLE photo_index DROP COLUMN IF EXISTS clip_embedding;
ALTER TABLE photo_index DROP COLUMN IF EXISTS clip_model;
ALTER TABLE photo_index DROP COLUMN IF EXISTS indexed;
ALTER TABLE photo_index DROP COLUMN IF EXISTS indexed_at;
ALTER TABLE photo_index DROP COLUMN IF EXISTS meta_data;

COMMIT;

-- Проверка: должны остаться только эти колонки с эмбеддингами:
-- clip_embedding_vit_b32
-- clip_embedding_vit_b16
-- clip_embedding_vit_l14
-- clip_embedding_siglip
