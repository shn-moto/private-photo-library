# Очистка и Переиндексирование

## Проблема

В индексе могут оставаться записи о файлах, которые были удалены с диска. Это происходит потому, что процесс индексирования:
- Пропускает файлы, которые уже в индексе
- **Не проверяет**, существуют ли эти файлы на диске
- Не удаляет orphaned записи

## Решение

### 1. Скрипт очистки orphaned записей

Файл: `scripts/cleanup_orphaned.py`

#### Операция 1: Проверить orphaned файлы (безопасно)
```bash
python scripts/cleanup_orphaned.py 1
```
Показывает какие файлы потеряны, БД не изменяется.

#### Операция 2: Удалить orphaned записи (реально удаляет)
```bash
python scripts/cleanup_orphaned.py 2
```
**⚠️ ВНИМАНИЕ**: Действительно удалит orphaned записи из БД!

#### Операция 3: Переиндексировать существующие файлы
```bash
python scripts/cleanup_orphaned.py 3
# или с указанием директории:
python scripts/cleanup_orphaned.py 3 /path/to/images
```
Переиндексирует все поддерживаемые файлы в директории.

#### Операция 4: Полная очистка + переиндексирование
```bash
python scripts/cleanup_orphaned.py 4
# или:
python scripts/cleanup_orphaned.py 4 /path/to/images
```
1. Удаляет orphaned записи
2. Переиндексирует все существующие файлы

### 2. Встроенная функция в IndexingService

Метод `cleanup_missing_files()` добавлен в `services/indexer.py`:

```python
from services.indexer import IndexingService

indexer = IndexingService()

# Только проверить
stats = indexer.cleanup_missing_files(check_only=True)
print(f"Missing files: {stats['missing']}")

# Реально удалить
stats = indexer.cleanup_missing_files(check_only=False)
print(f"Deleted records: {stats['deleted']}")
```

## Статистика

Каждая операция выдает статистику:

```
СТАТИСТИКА:
  Всего записей: 5432
  Потеряны файлы: 127
  Удалено записей о фото: 127
  Удалено записей о лицах: 342
  Ошибок при удалении: 0
```

## Автоматическая очистка (опционально)

Для периодической очистки можно добавить в cron:

```bash
# Каждый день в 2 ночи
0 2 * * * cd /path/to/project && python scripts/cleanup_orphaned.py 2
```

## Безопасность

- Всегда сначала запустите **операцию 1** (dry-run) для проверки
- Операция 2 безопасна - удаляются только orphaned записи
- Cascade удаление - удаляются также связанные записи о лицах
- Создаются логи всех операций

## Логирование

Логи записываются в `logs/` директорию. Проверьте файлы для деталей:

```bash
tail -f logs/indexing.log
```
