-- Migration: Add user_session table for Telegram-based authentication
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_auth.sql

CREATE TABLE IF NOT EXISTS user_session (
    token          VARCHAR(64) PRIMARY KEY,
    user_id        INTEGER NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
    created_at     TIMESTAMP DEFAULT NOW(),
    last_active_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_session_user ON user_session(user_id);
CREATE INDEX IF NOT EXISTS idx_user_session_last_active ON user_session(last_active_at);
