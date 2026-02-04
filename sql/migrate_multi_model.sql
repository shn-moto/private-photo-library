-- Миграция на мульти-модельную поддержку CLIP эмбеддингов
-- Каждая модель хранится в отдельной колонке с правильной размерностью

BEGIN;

-- 1. Новые колонки под каждую модель
ALTER TABLE photo_index
  ADD COLUMN IF NOT EXISTS clip_embedding_vit_b32 vector(512),
  ADD COLUMN IF NOT EXISTS clip_embedding_vit_b16 vector(512),
  ADD COLUMN IF NOT EXISTS clip_embedding_vit_l14 vector(768),
  ADD COLUMN IF NOT EXISTS clip_embedding_siglip vector(1152),
  ADD COLUMN IF NOT EXISTS clip_model VARCHAR(20);

-- 2. Миграция существующих данных
-- indexed=1 → SigLIP (текущая модель)
UPDATE photo_index
SET clip_embedding_siglip = clip_embedding,
    clip_model = 'SigLIP'
WHERE indexed = 1 AND clip_embedding IS NOT NULL;

-- indexed=0 → старая модель ViT-B/32 (нет эмбеддинга, только метка)
UPDATE photo_index
SET clip_model = 'ViT-B/32'
WHERE indexed = 0;

COMMIT;

-- 3. HNSW индексы (CONCURRENTLY нельзя внутри транзакции)
DROP INDEX IF EXISTS idx_photo_clip_embedding_hnsw;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_clip_siglip_hnsw
  ON photo_index USING hnsw (clip_embedding_siglip vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_clip_vit_b32_hnsw
  ON photo_index USING hnsw (clip_embedding_vit_b32 vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_clip_vit_b16_hnsw
  ON photo_index USING hnsw (clip_embedding_vit_b16 vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_clip_vit_l14_hnsw
  ON photo_index USING hnsw (clip_embedding_vit_l14 vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
