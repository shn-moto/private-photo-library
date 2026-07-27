-- Migration: Google Photos album export
-- Per-user OAuth token storage on app_user + album/photo export mapping.
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_google_export.sql

-- 1. Google account link lives on app_user (one Google account per app user).
--    Access tokens are NOT stored — they live 1h and are kept in process memory.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'app_user' AND column_name = 'google_refresh_token'
    ) THEN
        ALTER TABLE app_user ADD COLUMN google_refresh_token TEXT;
        ALTER TABLE app_user ADD COLUMN google_email VARCHAR(256);
        ALTER TABLE app_user ADD COLUMN google_connected_at TIMESTAMP;
        RAISE NOTICE 'Added google_* columns to app_user';
    ELSE
        RAISE NOTICE 'google_* columns already exist in app_user';
    END IF;
END $$;

-- 2. Our album -> Google album, keyed per user:
--    two users may export the same album, each into their own Google library.
CREATE TABLE IF NOT EXISTS google_album_export (
    album_id         INTEGER NOT NULL REFERENCES album(album_id) ON DELETE CASCADE,
    user_id          INTEGER NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
    google_album_id  VARCHAR(256) NOT NULL,
    google_album_url TEXT,
    created_at       TIMESTAMP DEFAULT NOW(),
    last_export_at   TIMESTAMP,
    PRIMARY KEY (album_id, user_id)
);

-- 3. Per-photo export state — makes re-export idempotent: a repeated run only
--    uploads what is missing instead of creating duplicates in Google Photos.
CREATE TABLE IF NOT EXISTS google_export_item (
    album_id        INTEGER NOT NULL REFERENCES album(album_id) ON DELETE CASCADE,
    user_id         INTEGER NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
    image_id        INTEGER NOT NULL REFERENCES photo_index(image_id) ON DELETE CASCADE,
    google_media_id VARCHAR(256),
    status          VARCHAR(16) NOT NULL DEFAULT 'ok',  -- ok | failed
    error           TEXT,
    exported_at     TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (album_id, user_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_google_export_item_status
    ON google_export_item (album_id, user_id, status);

-- 4. RBAC: assignable section/function so non-admin users can export to their own account
INSERT INTO api_section (section_code, section_name, description, is_public, is_admin_only, sort_order)
VALUES ('google', 'Google Фото', 'Экспорт альбомов в Google Фото', FALSE, FALSE, 26)
ON CONFLICT (section_code) DO NOTHING;

INSERT INTO api_function (function_code, section_code, function_name, description, sort_order)
VALUES ('google.export', 'google', 'Экспорт в Google Фото',
        'Привязка Google аккаунта и экспорт альбомов (POST /albums/{id}/export/google)', 1)
ON CONFLICT (function_code) DO NOTHING;

-- Grant to all existing users (same as other assignable functions)
INSERT INTO user_function_permission (user_id, function_code)
SELECT au.user_id, 'google.export'
FROM app_user au
WHERE NOT EXISTS (
    SELECT 1 FROM user_function_permission ufp
    WHERE ufp.user_id = au.user_id AND ufp.function_code = 'google.export'
);
