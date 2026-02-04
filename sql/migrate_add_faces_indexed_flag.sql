-- Миграция: добавление флага faces_indexed для оптимизации face indexing
-- Дата: 2026-02-01
-- Описание: Добавляем колонку faces_indexed в photo_index для быстрой проверки 
--           индексированных фото вместо медленного JOIN с faces

-- Добавить колонку faces_indexed
ALTER TABLE photo_index 
ADD COLUMN IF NOT EXISTS faces_indexed INTEGER NOT NULL DEFAULT 0;

-- Комментарий
COMMENT ON COLUMN photo_index.faces_indexed IS 
'Флаг индексации лиц: 0 = не индексировалось, 1 = индексировано (есть или нет лиц)';

-- Создать индекс для быстрой фильтрации
CREATE INDEX IF NOT EXISTS idx_photo_index_faces_indexed 
ON photo_index(faces_indexed);

-- Установить флаг для уже проиндексированных фото (у которых есть записи в faces)
UPDATE photo_index 
SET faces_indexed = 1 
WHERE image_id IN (
    SELECT DISTINCT image_id FROM faces
);

-- Статистика после миграции
DO $$
DECLARE
    total_photos INTEGER;
    indexed_photos INTEGER;
    with_faces INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_photos FROM photo_index;
    SELECT COUNT(*) INTO indexed_photos FROM photo_index WHERE faces_indexed = 1;
    SELECT COUNT(DISTINCT image_id) INTO with_faces FROM faces;
    
    RAISE NOTICE 'Migration completed:';
    RAISE NOTICE '  Total photos: %', total_photos;
    RAISE NOTICE '  Photos with faces_indexed=1: %', indexed_photos;
    RAISE NOTICE '  Photos with actual faces: %', with_faces;
END $$;
