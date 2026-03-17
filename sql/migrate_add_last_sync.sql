-- Add last_sync_at to app_user for WiFi sync tracking
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS last_sync_at TIMESTAMP;
