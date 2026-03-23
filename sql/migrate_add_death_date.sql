-- Migration: add death_date to person table
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_death_date.sql

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'person' AND column_name = 'death_date'
    ) THEN
        ALTER TABLE person ADD COLUMN death_date DATE;
        RAISE NOTICE 'Added death_date column to person table';
    ELSE
        RAISE NOTICE 'death_date column already exists';
    END IF;
END $$;
