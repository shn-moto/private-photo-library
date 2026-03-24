-- Migration: Add person_id to app_user (user-person association)
-- Links app_user to person for AI family context awareness

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'app_user' AND column_name = 'person_id'
    ) THEN
        ALTER TABLE app_user ADD COLUMN person_id INTEGER REFERENCES person(person_id) ON DELETE SET NULL;
        RAISE NOTICE 'Added person_id column to app_user';
    ELSE
        RAISE NOTICE 'person_id column already exists in app_user';
    END IF;
END $$;
