-- Migration: add approximate date flags to person table
-- Run: docker exec smart_photo_db psql -U dev -d smart_photo_index -f /path/migrate_add_date_approx.sql

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='person' AND column_name='birth_date_approx') THEN
        ALTER TABLE person ADD COLUMN birth_date_approx BOOLEAN NOT NULL DEFAULT FALSE;
        RAISE NOTICE 'Added birth_date_approx column';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='person' AND column_name='death_date_approx') THEN
        ALTER TABLE person ADD COLUMN death_date_approx BOOLEAN NOT NULL DEFAULT FALSE;
        RAISE NOTICE 'Added death_date_approx column';
    END IF;
END $$;

-- Mark all existing dates stored as YYYY-01-01 as approximate
UPDATE person SET birth_date_approx = TRUE
WHERE birth_date IS NOT NULL
  AND EXTRACT(MONTH FROM birth_date) = 1
  AND EXTRACT(DAY FROM birth_date) = 1
  AND birth_date < '2000-01-01';

UPDATE person SET death_date_approx = TRUE
WHERE death_date IS NOT NULL
  AND EXTRACT(MONTH FROM death_date) = 1
  AND EXTRACT(DAY FROM death_date) = 1
  AND death_date < '2000-01-01';
