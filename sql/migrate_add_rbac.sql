-- Migration: Add RBAC permission system (section-based permissions)
-- Run: psql -U dev -d smart_photo_index -f sql/migrate_add_rbac.sql

-- ============================================
-- API sections (permission groups)
-- ============================================
CREATE TABLE IF NOT EXISTS api_section (
    section_code VARCHAR(32) PRIMARY KEY,
    section_name VARCHAR(128) NOT NULL,
    description TEXT,
    is_public BOOLEAN NOT NULL DEFAULT FALSE,      -- Always allowed (auth, health, images)
    is_admin_only BOOLEAN NOT NULL DEFAULT FALSE,   -- Only admin can access
    sort_order INTEGER NOT NULL DEFAULT 0
);

-- ============================================
-- User permissions (many-to-many: user × section)
-- ============================================
CREATE TABLE IF NOT EXISTS user_permission (
    user_id INTEGER NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
    section_code VARCHAR(32) NOT NULL REFERENCES api_section(section_code) ON DELETE CASCADE,
    PRIMARY KEY (user_id, section_code)
);

CREATE INDEX IF NOT EXISTS idx_user_permission_user ON user_permission(user_id);

-- ============================================
-- Seed API sections
-- ============================================
INSERT INTO api_section (section_code, section_name, description, is_public, is_admin_only, sort_order) VALUES
-- Public (always accessible, no permission needed)
('auth',            'Аутентификация',           'Вход, выход, проверка сессии',                     TRUE,  FALSE, 1),
('health',          'Здоровье и статистика',    'Статус сервиса, модели, статистика',               TRUE,  FALSE, 2),
('images',          'Отдача изображений',       'Миниатюры, полноразмер, оригиналы',                TRUE,  FALSE, 3),

-- Assignable (can be granted to regular users)
('search',          'Поиск',                    'Текстовый поиск, поиск по изображению',            FALSE, FALSE, 10),
('photos',          'Фотографии',               'Детали фото, поворот, удаление',                   FALSE, FALSE, 11),
('timeline',        'Лента',                    'Хронологическая лента фотографий',                 FALSE, FALSE, 12),
('tags',            'Теги',                     'Управление тегами фотографий',                     FALSE, FALSE, 13),
('albums',          'Альбомы',                  'Фотоальбомы',                                      FALSE, FALSE, 14),
('map',             'Карта',                    'Карта фотографий, кластеры',                       FALSE, FALSE, 15),
('faces',           'Лица',                     'Просмотр и индексация лиц на фотографиях',         FALSE, FALSE, 16),
('face_search',     'Поиск по лицу',            'Поиск похожих лиц',                               FALSE, FALSE, 17),
('persons',         'Персоны',                  'Управление персонами',                             FALSE, FALSE, 18),
('ai',              'AI Ассистент',             'AI помощник для поиска',                           FALSE, FALSE, 19),
('books',           'Библиотека',               'Книги',                                            FALSE, FALSE, 20),

-- Admin-only (only admin can access)
('geo',             'Геолокация',               'Привязка GPS координат',                           FALSE, TRUE,  30),
('indexing',        'Индексация',               'CLIP индексация, сканирование, cleanup',           FALSE, TRUE,  31),
('duplicates_clip', 'Дубликаты CLIP',           'Поиск и удаление дубликатов по CLIP',              FALSE, TRUE,  32),
('duplicates_phash','Дубликаты pHash',          'Поиск и удаление дубликатов по pHash',             FALSE, TRUE,  33),
('face_assign',     'Назначение лиц',           'Привязка лиц к персонам',                         FALSE, TRUE,  34),
('admin_queue',     'Очередь индексации',       'Управление очередью индексации, stop, shutdown',   FALSE, TRUE,  40),
('admin_gpu',       'GPU и модели',             'Управление GPU и CLIP моделями',                   FALSE, TRUE,  41),
('admin_failed',    'Битые файлы',              'Управление битыми файлами',                        FALSE, TRUE,  42),
('admin_clip_tag',  'CLIP → Тег',              'Автоматическое тегирование по CLIP',               FALSE, TRUE,  43),
('admin_cache',     'Кэш',                     'Управление кэшем миниатюр',                        FALSE, TRUE,  44),
('admin_users',     'Управление пользователями','Список пользователей, права доступа',              FALSE, TRUE,  45)
ON CONFLICT (section_code) DO NOTHING;

-- ============================================
-- Grant default permissions to existing non-admin users
-- (all assignable sections: search, photos, timeline, tags, albums, map, faces, face_search, persons, ai, books)
-- ============================================
INSERT INTO user_permission (user_id, section_code)
SELECT u.user_id, s.section_code
FROM app_user u
CROSS JOIN api_section s
WHERE u.is_admin = FALSE
  AND s.is_public = FALSE
  AND s.is_admin_only = FALSE
ON CONFLICT DO NOTHING;
