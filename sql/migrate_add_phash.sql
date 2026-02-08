-- Migration: add/upgrade phash column for perceptual hash duplicate detection
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_phash.sql

-- Add column if not exists, expand to VARCHAR(64) for 256-bit hashes
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'photo_index' AND column_name = 'phash') THEN
        ALTER TABLE photo_index ADD COLUMN phash VARCHAR(64);
    ELSE
        ALTER TABLE photo_index ALTER COLUMN phash TYPE VARCHAR(64);
    END IF;
END $$;

-- Reset all existing 64-bit hashes so they get recomputed as 256-bit
UPDATE photo_index SET phash = NULL WHERE phash IS NOT NULL;

-- Index for exact match lookups (distance=0)
CREATE INDEX IF NOT EXISTS idx_photo_index_phash
ON photo_index(phash) WHERE phash IS NOT NULL;
