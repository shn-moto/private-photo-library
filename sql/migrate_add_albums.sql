-- Migration: add users and albums tables
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_albums.sql

-- 1. Create app_user table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'app_user') THEN
        CREATE TABLE app_user (
            user_id SERIAL PRIMARY KEY,
            telegram_id BIGINT UNIQUE,
            username VARCHAR(128),
            display_name VARCHAR(256) NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW(),
            last_seen_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX idx_app_user_telegram_id ON app_user(telegram_id) WHERE telegram_id IS NOT NULL;
    END IF;
END $$;

-- Seed admin user (user_id=1)
INSERT INTO app_user (user_id, telegram_id, username, display_name, is_admin)
VALUES (1, NULL, 'admin', 'Admin', TRUE)
ON CONFLICT (user_id) DO NOTHING;
SELECT setval('app_user_user_id_seq', GREATEST(1, (SELECT MAX(user_id) FROM app_user)));

-- 2. Create album table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'album') THEN
        CREATE TABLE album (
            album_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
            title VARCHAR(512) NOT NULL,
            description TEXT,
            cover_image_id INTEGER REFERENCES photo_index(image_id) ON DELETE SET NULL,
            is_public BOOLEAN NOT NULL DEFAULT FALSE,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX idx_album_user_id ON album(user_id);
        CREATE INDEX idx_album_public ON album(is_public) WHERE is_public = TRUE;
    END IF;
END $$;

-- 3. Create album_photo junction table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'album_photo') THEN
        CREATE TABLE album_photo (
            album_id INTEGER NOT NULL REFERENCES album(album_id) ON DELETE CASCADE,
            image_id INTEGER NOT NULL REFERENCES photo_index(image_id) ON DELETE CASCADE,
            sort_order INTEGER NOT NULL DEFAULT 0,
            added_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (album_id, image_id)
        );
        CREATE INDEX idx_album_photo_image_id ON album_photo(image_id);
        CREATE INDEX idx_album_photo_album_id ON album_photo(album_id);
    END IF;
END $$;
