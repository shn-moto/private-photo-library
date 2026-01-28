#!/usr/bin/env python3
"""Быстрый тест подключения к БД"""

import sys
from pathlib import Path

# Добавить родительскую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

print("1. Testing imports...", end="", flush=True)

try:
    from config.settings import settings
    print(" OK")
except Exception as e:
    print(f" FAILED: {e}")
    sys.exit(1)

print("2. Testing DB connection...", end="", flush=True)
try:
    from db.database import DatabaseManager
    db = DatabaseManager(settings.DATABASE_URL)
    print(" OK")
except Exception as e:
    print(f" FAILED: {e}")
    sys.exit(1)

print("3. Testing PhotoIndex model...", end="", flush=True)
try:
    from models.data_models import PhotoIndex
    session = db.get_session()
    count = session.query(PhotoIndex).count()
    session.close()
    print(f" OK (found {count} records)")
except Exception as e:
    print(f" FAILED: {e}")
    sys.exit(1)

print("\n✓ All checks passed!")
