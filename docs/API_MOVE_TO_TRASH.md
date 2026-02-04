# Эндпоинт перемещения фото в корзину

## POST /photos/move-to-trash

Эндпоинт для поиска фотографий по текстовому описанию и перемещения их в корзину.

### Особенности

1. **Поиск по текстовому запросу** - использует CLIP эмбеддинги для семантического поиска
2. **Поддержка моделей** - можно указать конкретную модель CLIP (ViT-B/32, ViT-B/16, ViT-L/14, SigLIP)
3. **Сохранение структуры папок** - файлы перемещаются в `/trash` с сохранением исходной структуры директорий относительно `/photos`
4. **Индексы НЕ удаляются** - записи в базе данных остаются нетронутыми
5. **Автоматический перевод** - запросы автоматически переводятся на английский язык для лучшего качества поиска

### Пример запроса

```json
{
  "query": "мусор на улице",
  "model": "SigLIP",
  "top_k": 50,
  "similarity_threshold": 0.1,
  "translate": true
}
```

### Параметры запроса

- `query` (string, обязательный) - текстовое описание для поиска фотографий
- `model` (string, опциональный) - модель CLIP для поиска. По умолчанию используется модель из настроек
- `top_k` (integer, опциональный, по умолчанию: 100) - максимальное количество файлов для перемещения
- `similarity_threshold` (float, опциональный, по умолчанию: 0.1) - порог схожести (0.0 - 1.0)
- `translate` (boolean, опциональный, по умолчанию: true) - автоматически переводить запрос на английский

### Пример ответа

```json
{
  "moved": 15,
  "found": 20,
  "errors": [
    "123: файл не существует - /photos/2024/photo.jpg",
    "456: ошибка перемещения - Permission denied"
  ],
  "query": "мусор на улице",
  "translated_query": "trash on the street"
}
```

### Параметры ответа

- `moved` (integer) - количество успешно перемещенных файлов
- `found` (integer) - общее количество найденных файлов
- `errors` (array) - список ошибок при перемещении (если есть)
- `query` (string) - исходный запрос
- `translated_query` (string, опциональный) - переведенный запрос (если перевод применялся)

### Структура папок

Файлы перемещаются с сохранением относительной структуры:
- Исходный файл: `/photos/2024/01/vacation/IMG_001.jpg`
- Новое расположение: `/.trash/2024/01/vacation/IMG_001.jpg`

**Важно**: 
- `TRASH_DIR` (`/.trash`) - для обычного удаления, находится ВНЕ `/photos` чтобы не индексироваться
- `DUPLICATES_DIR` (`/.photo_duplicates`) - для дубликатов, также вне `/photos`

Это позволяет разделить автоматически найденные дубликаты от файлов, удаленных вручную, и предотвращает повторную индексацию файлов в корзине.

### Маппинг папок в Docker

Убедитесь что папка `/.trash` замаплена в `docker-compose.yml`:

```yaml
volumes:
  - ${PHOTOS_HOST_PATH}:/photos:rw
  - ${PHOTOS_HOST_PATH}/../.trash:/.trash:rw  # Корзина ВНЕ photos
  - ${PHOTOS_HOST_PATH}/../.photo_duplicates:/.photo_duplicates:rw  # Дубликаты
```

И добавлены environment variables:
```yaml
environment:
  TRASH_DIR: ${TRASH_DIR:-/.trash}
  DUPLICATES_DIR: ${DUPLICATES_DIR:-/.photo_duplicates}
```

### cURL пример

```bash
curl -X POST "http://localhost:8000/photos/move-to-trash" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "размытые фото",
    "model": "SigLIP",
    "top_k": 100,
    "similarity_threshold": 0.15
  }'
```

### Примеры использования

#### Удаление размытых фотографий
```json
{
  "query": "blurry photos",
  "top_k": 200,
  "similarity_threshold": 0.2
}
```

#### Удаление скриншотов
```json
{
  "query": "screenshots",
  "model": "SigLIP",
  "top_k": 500
}
```

#### Удаление случайных кадров
```json
{
  "query": "accidental photo floor ceiling",
  "top_k": 100
}
```
