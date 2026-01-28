-- Миграция с CLIP ViT-B/32 (512 dim) на SigLIP so400m (1152 dim)
-- Запустить ПЕРЕД переиндексацией!

-- 1. Удалить старые HNSW индексы (не совместимы с новой размерностью)
DROP INDEX IF EXISTS idx_photo_clip_embedding_hnsw;

-- 2. Обнулить все старые эмбеддинги (512 dim != 1152 dim)
UPDATE photo_index SET clip_embedding = NULL, indexed = 0;

-- 3. Изменить размерность колонки
ALTER TABLE photo_index ALTER COLUMN clip_embedding TYPE vector(1152);

-- 4. Пересоздать HNSW индекс для новой размерности
CREATE INDEX idx_photo_clip_embedding_hnsw
    ON photo_index
    USING hnsw (clip_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Готово! Теперь запустите переиндексацию:
-- POST http://localhost:8000/reindex
