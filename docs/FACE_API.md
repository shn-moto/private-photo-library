# Face Detection & Person Management API

## Overview

API для распознавания лиц, управления персонами и привязки лиц к известным людям.

## Key Features

- **Face Detection** - обнаружение лиц на фотографиях с использованием InsightFace
- **Face Embeddings** - 512-мерные векторы для сравнения лиц
- **Person Management** - создание, редактирование, объединение персон
- **Auto-Assignment** - автоматическая привязка лиц к известным персонам
- **Cover Face Selection** - автоматический выбор лучшего лица для аватара персоны

## API Endpoints

### Face Detection

#### GET /photo/{image_id}/faces
Получить все лица на фотографии.

**Response:**
```json
{
  "image_id": 123,
  "count": 2,
  "faces": [
    {
      "face_id": 456,
      "bbox": [100, 50, 200, 180],
      "det_score": 0.95,
      "age": 25,
      "gender": 1,
      "person_id": 1,
      "person_name": "Иван",
      "person_description": "Брат"
    }
  ],
  "original_width": 4000,
  "original_height": 3000
}
```

#### POST /photo/{image_id}/faces/auto-assign
Автоматически привязать неназначенные лица на фото к известным персонам.

**Query Parameters:**
- `threshold` (float, 0.3-0.95, default: 0.6) - минимальное сходство для авто-привязки

**Response:**
```json
{
  "image_id": 123,
  "assigned": 2,
  "total_faces": 3,
  "faces": [...],
  "original_width": 4000,
  "original_height": 3000
}
```

**Логика:**
1. Для каждого неназначенного лица ищет похожие среди уже привязанных
2. Использует ВСЕ лица персоны для сравнения (не только cover_face)
3. Если найдено совпадение выше порога - привязывает лицо
4. Обновляет `cover_face_id` если новое лицо лучшего качества

### Face Assignment

#### POST /faces/{face_id}/assign
Привязать лицо к персоне.

**Request Body:**
```json
{
  "person_id": 1
}
```
или для создания новой персоны:
```json
{
  "new_person_name": "Имя персоны"
}
```

**Response:**
```json
{
  "status": "assigned",
  "face_id": 456,
  "person_id": 1,
  "person_name": "Иван",
  "created_new_person": false
}
```

**Важно:**
- При переназначении лица от одной персоны к другой старая привязка автоматически удаляется
- Если лицо было `cover_face_id` у старой персоны - выбирается новый cover с лучшим `det_score`
- При назначении обновляется `cover_face_id` новой персоны если это лицо лучшего качества

#### DELETE /faces/{face_id}/assign
Отвязать лицо от персоны.

### Person Management

#### GET /persons
Список персон с количеством лиц и фотографий.

#### POST /persons
Создать персону.

**Request Body:**
```json
{
  "name": "Имя",
  "description": "Описание (опционально)",
  "initial_face_id": 123
}
```

**Примечание:** Если `initial_face_id` уже привязан к другой персоне - он будет переназначен.

#### PUT /persons/{person_id}
Обновить персону.

#### DELETE /persons/{person_id}
Удалить персону (лица становятся неназначенными).

#### POST /persons/{person_id}/merge/{target_person_id}
Объединить персоны (все лица source переносятся в target, source удаляется).

#### POST /persons/{person_id}/auto-assign
Автоматически привязать похожие лица к персоне.

**Query Parameters:**
- `threshold` (float, 0.4-0.9, default: 0.6)

### Maintenance

#### POST /persons/maintenance/recalculate-covers
Пересчитать `cover_face_id` для всех персон на основе лучшего `det_score`.

**Response:**
```json
{
  "status": "ok",
  "updated": 15,
  "total": 42
}
```

**Когда использовать:**
- После миграции данных
- Для фикса cover_face_id, установленных на лица с низким качеством
- Для очистки orphan cover_face_id (указывающих на несуществующие лица)

## Cover Face Selection Logic

`cover_face_id` - лицо, используемое как аватар персоны. Выбирается автоматически:

1. **При создании персоны** - initial_face становится cover
2. **При назначении лица** - если новое лицо имеет лучший `det_score` чем текущий cover
3. **При авто-привязке** - аналогично, проверяется качество каждого назначенного лица
4. **При переназначении** - у старой персоны выбирается новый cover из оставшихся лиц

**Критерий качества:** `det_score` (confidence детектора лиц, 0-1)

## Face Search

#### POST /search/face
Поиск по загруженному изображению лица.

#### POST /search/face/by_id/{face_id}
Поиск по ID известного лица.

#### POST /search/person-clip
Комбинированный поиск: персона + текстовый запрос.

**Request:**
```json
{
  "person_id": 1,
  "query": "в горах",
  "top_k": 20,
  "translate": true
}
```

## Database Schema

```sql
-- Таблица лиц
CREATE TABLE faces (
    face_id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES photo_index(image_id),
    bbox REAL[4],                    -- [x1, y1, x2, y2]
    det_score REAL,                  -- Confidence детектора
    face_embedding vector(512),      -- ArcFace embedding
    age INTEGER,
    gender INTEGER,                  -- 0=F, 1=M
    person_id INTEGER REFERENCES person(person_id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Таблица персон
CREATE TABLE person (
    person_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    cover_face_id INTEGER,           -- Лицо для аватара (лучшее по det_score)
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Индексы для поиска
CREATE INDEX idx_faces_embedding_hnsw ON faces
    USING hnsw (face_embedding vector_cosine_ops);
CREATE INDEX idx_faces_person ON faces(person_id);
CREATE INDEX idx_faces_image ON faces(image_id);
```

## UI Integration

### Lightbox Behavior

1. **При открытии фото** - лица НЕ показываются по умолчанию
2. **В статус-баре** - отображается количество лиц: `Лица: 2/3` (назначено/всего)
3. **При клике на иконку лица:**
   - Вызывается `/photo/{id}/faces/auto-assign`
   - Автоматически привязываются известные лица
   - Отображаются рамки: зеленые (назначены), желтые (не назначены)

### Face Popup

При клике на рамку лица показывается popup:
- Confidence (det_score)
- Имя персоны (если назначено)
- Описание персоны
- Выпадающий список персон
- Поле для создания новой персоны
- Кнопки "Сохранить" / "Отмена"
