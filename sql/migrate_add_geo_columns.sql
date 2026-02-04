-- Миграция: добавление колонок для геолокации
-- Выполнить: psql -U dev -d smart_photo_index -f scripts/migrate_add_geo_columns.sql

-- Добавить колонки для координат
ALTER TABLE photo_index ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;
ALTER TABLE photo_index ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;

-- Индекс для поиска по координатам (B-tree composite)
-- Позволяет быстро фильтровать по bounding box
CREATE INDEX IF NOT EXISTS idx_photo_index_geo
    ON photo_index (latitude, longitude)
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Индекс для поиска по дате съемки
CREATE INDEX IF NOT EXISTS idx_photo_index_photo_date
    ON photo_index (photo_date)
    WHERE photo_date IS NOT NULL;

-- Проверить результат
SELECT
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'photo_index'
    AND column_name IN ('latitude', 'longitude', 'photo_date')
ORDER BY column_name;

-- Статистика по записям с координатами (будет 0 до запуска скрипта заполнения)
SELECT
    COUNT(*) as total_photos,
    COUNT(latitude) as with_gps,
    COUNT(photo_date) as with_date
FROM photo_index;
