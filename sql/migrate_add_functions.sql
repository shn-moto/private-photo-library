-- Migration: Add function-level permissions (granular control within sections)
-- Run: docker exec -i smart_photo_db psql -U dev -d smart_photo_index -f /tmp/migrate_add_functions.sql

-- ============================================
-- API functions (individual actions within sections)
-- ============================================
CREATE TABLE IF NOT EXISTS api_function (
    function_code VARCHAR(64) PRIMARY KEY,
    section_code VARCHAR(32) NOT NULL REFERENCES api_section(section_code) ON DELETE CASCADE,
    function_name VARCHAR(128) NOT NULL,
    description TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0
);

-- ============================================
-- User function permissions (user × function)
-- ============================================
CREATE TABLE IF NOT EXISTS user_function_permission (
    user_id INTEGER NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
    function_code VARCHAR(64) NOT NULL REFERENCES api_function(function_code) ON DELETE CASCADE,
    PRIMARY KEY (user_id, function_code)
);

CREATE INDEX IF NOT EXISTS idx_user_func_perm_user ON user_function_permission(user_id);

-- ============================================
-- Seed API functions
-- ============================================
INSERT INTO api_function (function_code, section_code, function_name, description, sort_order) VALUES
-- Public (always allowed)
('auth',              'auth',         'Аутентификация',              'Вход, выход, проверка сессии',                      1),
('health',            'health',       'Здоровье и статистика',       'Статус сервиса, модели, статистика',                1),
('images',            'images',       'Отдача изображений',          'Миниатюры, полноразмер, оригиналы',                 1),

-- Search
('search.text',       'search',       'Текстовый поиск',            'Поиск по текстовому описанию (CLIP)',               1),
('search.image',      'search',       'Поиск по изображению',       'Загрузить фото и найти похожие',                    2),

-- Photos
('photos.view',       'photos',       'Просмотр деталей',           'Информация о фотографии (GET /photo/{id})',         1),
('photos.rotate',     'photos',       'Поворот',                    'Поворот фотографий (POST /photo/{id}/rotate)',      2),
('photos.delete',     'photos',       'Удаление',                   'Удаление файлов в корзину (POST /photos/delete)',   3),

-- Timeline
('timeline.view',     'timeline',     'Лента',                      'Хронологическая лента фотографий',                  1),

-- Tags
('tags.view',         'tags',         'Просмотр тегов',             'Просмотр тегов (GET /tags, GET /photo/{id}/tags)',  1),
('tags.manage',       'tags',         'Управление тегами',          'Создание, удаление, назначение тегов на фото',      2),

-- Albums
('albums.view',       'albums',       'Просмотр альбомов',          'Список альбомов, фото в альбоме',                   1),
('albums.manage',     'albums',       'Управление альбомами',       'Создание, редактирование, удаление альбомов',       2),

-- Map
('map.view',          'map',          'Карта и кластеры',           'Просмотр карты с кластерами фото',                  1),
('map.photos',        'map',          'Просмотр фото в кластере',   'Открытие кластера и просмотр фотографий',           2),
('map.search',        'map',          'Поиск на карте',             'Текстовый поиск в географической области',           3),

-- Faces
('faces.view',        'faces',        'Просмотр лиц',              'Просмотр лиц на фото, миниатюры лиц',              1),
('faces.reindex',     'faces',        'Индексация лиц',            'Запуск индексации/реиндексации лиц',                 2),

-- Face search
('face_search.search','face_search',  'Поиск по лицу',             'Поиск похожих лиц по фото или ID',                  1),

-- Persons
('persons.view',      'persons',      'Просмотр персон',           'Список, детали, фото персон',                       1),
('persons.manage',    'persons',      'Управление персонами',      'Создание, удаление, объединение персон',             2),

-- AI
('ai.assistant',      'ai',           'AI ассистент',              'Умный поиск через Gemini, оптимизация запросов',     1),

-- Books
('books.view',        'books',        'Чтение книг',               'Доступ к библиотеке книг',                          1),

-- Admin-only (one function per section)
('geo',               'geo',              'Геолокация',               'Привязка GPS координат, геокодирование',         1),
('indexing',          'indexing',         'Индексация',               'CLIP индексация, cleanup, сканирование',          1),
('duplicates_clip',   'duplicates_clip',  'Дубликаты CLIP',           'Поиск/удаление дубликатов по CLIP',              1),
('duplicates_phash',  'duplicates_phash', 'Дубликаты pHash',          'Поиск/удаление дубликатов по pHash',             1),
('face_assign',       'face_assign',      'Назначение лиц',           'Привязка лиц к персонам',                        1),
('admin_queue',       'admin_queue',      'Очередь индексации',       'Управление очередью, stop, shutdown',             1),
('admin_gpu',         'admin_gpu',        'GPU и модели',             'Мониторинг GPU, загрузка/выгрузка моделей',       1),
('admin_failed',      'admin_failed',     'Битые файлы',              'Просмотр и сброс битых файлов',                  1),
('admin_clip_tag',    'admin_clip_tag',   'CLIP → Тег',              'Автотегирование по CLIP запросу',                  1),
('admin_cache',       'admin_cache',      'Кэш',                     'Управление кэшем миниатюр',                       1),
('admin_users',       'admin_users',      'Управление пользователями','Пользователи и права доступа',                    1)
ON CONFLICT (function_code) DO NOTHING;

-- ============================================
-- Migrate existing user_permission → user_function_permission
-- For each user×section, grant ALL functions in that section
-- ============================================
INSERT INTO user_function_permission (user_id, function_code)
SELECT up.user_id, af.function_code
FROM user_permission up
JOIN api_function af ON af.section_code = up.section_code
ON CONFLICT DO NOTHING;
