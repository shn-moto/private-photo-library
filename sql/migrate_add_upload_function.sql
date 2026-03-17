-- Add photos.upload function for WiFi photo sync
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_upload_function.sql

INSERT INTO api_function (function_code, section_code, function_name, description, sort_order)
VALUES ('photos.upload', 'photos', 'Загрузка фото', 'Загрузка фото через WiFi sync (POST /upload/photo)', 4)
ON CONFLICT (function_code) DO NOTHING;

-- Grant to all existing non-admin users (same as other assignable functions)
INSERT INTO user_function_permission (user_id, function_code)
SELECT au.user_id, 'photos.upload'
FROM app_user au
WHERE NOT EXISTS (
    SELECT 1 FROM user_function_permission ufp
    WHERE ufp.user_id = au.user_id AND ufp.function_code = 'photos.upload'
);
