# Разделение папок для удаления и дубликатов

## Проблема
Все три функции использовали одну переменную `TRASH_DIR`:
- Удаление дубликатов (DELETE /duplicates)
- Обычное удаление (POST /photos/delete) 
- Новое перемещение в trash (POST /photos/move-to-trash)

## Решение
Разделены на две отдельные папки:

### 1. TRASH_DIR - для обычного удаления
- **Назначение**: Файлы удаленные вручную или по текстовому запросу
- **Путь в контейнере**: `/.trash`
- **Путь на хосте**: `${PHOTOS_HOST_PATH}/../.trash`
- **Используется в**:
  - POST /photos/delete
  - POST /photos/move-to-trash

### 2. DUPLICATES_DIR - для дубликатов
- **Назначение**: Автоматически найденные дубликаты
- **Путь в контейнере**: `/.photo_duplicates`
- **Путь на хосте**: `${PHOTOS_HOST_PATH}/../.photo_duplicates`
- **Используется в**:
  - DELETE /duplicates
  - services/duplicate_finder.py

**Важно**: Обе папки находятся ВНЕ `/photos` чтобы предотвратить повторную индексацию файлов в корзине.

## Измененные файлы

### config/settings.py
```python
TRASH_DIR: str = "/.trash"
DUPLICATES_DIR: str = "/.photo_duplicates"
```

### .env
```env
# Корзина для удалённых файлов вручную (путь внутри контейнера)
TRASH_DIR=/.trash

# Папка для дубликатов (отдельно от обычной корзины)
DUPLICATES_DIR=/.photo_duplicates
```

### docker-compose.yml
Добавлены volume mappings в оба сервиса (indexer и api):
```yaml
volumes:
  - ${PHOTOS_HOST_PATH}:/photos
  - ${PHOTOS_HOST_PATH}/../.trash:/.trash  # Корзина ВНЕ photos
  - ${PHOTOS_HOST_PATH}/../.photo_duplicates:/.photo_duplicates  # Дубликаты
```

И environment variables:
```yaml
environment:
  TRASH_DIR: ${TRASH_DIR:-/.trash}
  DUPLICATES_DIR: ${DUPLICATES_DIR:-/.photo_duplicates}
```

### services/duplicate_finder.py
```python
def _move_to_trash(file_path: str):
    duplicates_dir = settings.DUPLICATES_DIR  # Изменено с TRASH_DIR
    # ...
```

## Структура на хосте
```
H:/PHOTO/                    # Основные фото
H:/.trash/                   # Корзина (удаленные вручную) - ВНЕ photos!
H:/.photo_duplicates/        # Дубликаты (автоматически) - ВНЕ photos!
```

## Преимущества
✅ Разделение по назначению - дубликаты отдельно от обычного мусора
✅ Легче восстанавливать файлы - понятно откуда что
✅ Можно настроить разные политики хранения
✅ Меньше путаницы при ручной очистке
✅ **Файлы в корзине НЕ индексируются** - папки вне /photos
