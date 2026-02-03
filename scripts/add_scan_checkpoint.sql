-- Migration: Add scan_checkpoint table for NTFS USN Journal tracking
-- Run: psql -U dev -d smart_photo_index -f scripts/add_scan_checkpoint.sql

CREATE TABLE IF NOT EXISTS scan_checkpoint (
    id SERIAL PRIMARY KEY,
    drive_letter VARCHAR(10) NOT NULL UNIQUE,
    last_usn BIGINT NOT NULL DEFAULT 0,
    last_scan_time TIMESTAMP DEFAULT NOW(),
    files_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_scan_checkpoint_drive ON scan_checkpoint (drive_letter);

-- Show table info
\d scan_checkpoint
