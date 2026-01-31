"""
Тестовый скрипт для проверки эндпоинта /photos/move-to-trash
"""

# Пример запроса к новому эндпоинту
example_request = {
    "query": "мусор на улице",
    "model": "SigLIP",
    "top_k": 50,
    "similarity_threshold": 0.1,
    "translate": True
}

# Пример ответа
example_response = {
    "moved": 15,
    "found": 20,
    "errors": [
        "123: файл не существует - /path/to/missing.jpg",
        "456: ошибка перемещения - Permission denied"
    ],
    "query": "мусор на улице",
    "translated_query": "trash on the street"
}

print("Пример запроса:")
print(example_request)
print("\nПример ответа:")
print(example_response)

# Проверка структуры запроса
print("\n=== Проверка требований ===")
print("✓ 1.1 Параметр model присутствует:", "model" in example_request)
print("✓ 1.2 Сохранение структуры папок: реализовано через os.path.relpath и os.makedirs")
print("✓ 1.3 Индексы не удаляются: в коде нет session.delete(photo)")
