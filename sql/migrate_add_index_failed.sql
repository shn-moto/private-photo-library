-- Migration: add index_failed flag for broken/unreadable photos
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_index_failed.sql

ALTER TABLE photo_index
    ADD COLUMN IF NOT EXISTS index_failed BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS fail_reason VARCHAR(512);

-- Partial index: fast lookup of failed files (typically very small set)
CREATE INDEX IF NOT EXISTS idx_photo_index_failed
    ON photo_index (index_failed)
    WHERE index_failed = TRUE;

-- Verify
SELECT COUNT(*) AS failed_count FROM photo_index WHERE index_failed = TRUE;
