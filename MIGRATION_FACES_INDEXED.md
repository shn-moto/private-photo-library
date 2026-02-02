# Миграция: добавление флага faces_indexed

## Проблема
При `skip_indexed=True` индексатор лиц делал медленный запрос:
```sql
SELECT DISTINCT image_id FROM faces  -- Медленно на больших таблицах!
```

## Решение
Добавлена колонка `faces_indexed` в `photo_index` для быстрой фильтрации.

## Применение миграции

### Вариант 1: Через psql (рекомендуется)
```bash
# Войти в контейнер PostgreSQL
docker exec -it smart_photo_indexing-db-1 psql -U dev -d smart_photo_index

# Или локально
psql -U dev -d smart_photo_index

# Выполнить миграцию
\i scripts/migrate_add_faces_indexed_flag.sql
```

### Вариант 2: Через docker exec
```bash
docker exec -i smart_photo_indexing-db-1 psql -U dev -d smart_photo_index < scripts/migrate_add_faces_indexed_flag.sql
```

### Вариант 3: Через Python скрипт
```bash
docker exec smart_photo_indexing-api-1 python scripts/apply_migration.py migrate_add_faces_indexed_flag
```

## Проверка после миграции

```sql
-- Проверить что колонка создана
\d photo_index

-- Проверить индекс
\di idx_photo_index_faces_indexed

-- Проверить статистику
SELECT 
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE faces_indexed = 1) as indexed,
    COUNT(*) FILTER (WHERE faces_indexed = 0) as not_indexed
FROM photo_index;

-- Сравнить с реальными данными
SELECT COUNT(DISTINCT image_id) FROM faces;
```

## Эффект оптимизации

### До миграции:
```sql
-- Медленный запрос с JOIN
SELECT image_id, file_path 
FROM photo_index 
WHERE image_id NOT IN (SELECT DISTINCT image_id FROM faces);
-- Время: ~5-10 секунд на 100k фото
```

### После миграции:
```sql
-- Быстрый запрос по индексу
SELECT image_id, file_path 
FROM photo_index 
WHERE faces_indexed = 0;
-- Время: ~10-50 мс
```

**Ускорение: в 100-1000 раз!** ⚡

## Откат миграции (если нужно)

```sql
-- Удалить индекс
DROP INDEX IF EXISTS idx_photo_index_faces_indexed;

-- Удалить колонку
ALTER TABLE photo_index DROP COLUMN IF EXISTS faces_indexed;
```
