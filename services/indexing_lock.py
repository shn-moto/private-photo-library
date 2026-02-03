"""Механизм блокировки для предотвращения одновременной индексации"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
import atexit

logger = logging.getLogger(__name__)

# Windows-specific imports
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl


class IndexingLock:
    """Файловая блокировка для предотвращения одновременной индексации
    
    Использует эксклюзивную блокировку файла для предотвращения
    запуска нескольких процессов индексации одновременно.
    Работает на Windows (msvcrt) и Unix (fcntl).
    """
    
    def __init__(self, lock_name: str = "indexing"):
        """
        Args:
            lock_name: Имя блокировки (для разных типов индексации)
        """
        self.lock_name = lock_name
        self.lock_file: Optional[Path] = None
        self.file_handle = None
        self._acquired = False
        
        # Директория для lock-файлов
        lock_dir = Path(__file__).parent.parent / "logs"
        lock_dir.mkdir(exist_ok=True)
        self.lock_file = lock_dir / f"{lock_name}.lock"
        
        # Зарегистрировать автоочистку
        atexit.register(self.release)
    
    def _lock_file_windows(self, handle) -> bool:
        """Блокировка файла на Windows"""
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except (IOError, OSError):
            return False
    
    def _unlock_file_windows(self, handle):
        """Разблокировка файла на Windows"""
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except (IOError, OSError) as e:
            logger.error(f"Ошибка разблокировки на Windows: {e}")
    
    def _lock_file_unix(self, handle) -> bool:
        """Блокировка файла на Unix"""
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (IOError, OSError):
            return False
    
    def _unlock_file_unix(self, handle):
        """Разблокировка файла на Unix"""
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (IOError, OSError) as e:
            logger.error(f"Ошибка разблокировки на Unix: {e}")
    
    def acquire(self, timeout: float = 0.0) -> bool:
        """Попытаться захватить блокировку
        
        Args:
            timeout: Время ожидания в секундах (0 = не ждать)
            
        Returns:
            True если блокировка захвачена, False если занята
        """
        if self._acquired:
            logger.warning(f"Lock {self.lock_name} уже захвачен этим процессом")
            return True
        
        start_time = time.time()
        
        while True:
            try:
                # Открыть файл для блокировки
                self.file_handle = open(self.lock_file, 'w')
                
                # Попытаться захватить эксклюзивную блокировку (платформозависимо)
                if sys.platform == 'win32':
                    success = self._lock_file_windows(self.file_handle)
                else:
                    success = self._lock_file_unix(self.file_handle)
                
                if not success:
                    raise IOError("Не удалось захватить блокировку")
                
                # Записать информацию о процессе
                self.file_handle.write(f"PID: {os.getpid()}\n")
                self.file_handle.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.file_handle.flush()
                
                self._acquired = True
                logger.info(f"Lock {self.lock_name} успешно захвачен")
                return True
                
            except (IOError, OSError) as e:
                # Блокировка занята
                if self.file_handle:
                    self.file_handle.close()
                    self.file_handle = None
                
                elapsed = time.time() - start_time
                if timeout == 0 or elapsed >= timeout:
                    logger.info(f"Lock {self.lock_name} занят другим процессом")
                    return False
                
                # Подождать немного и попробовать снова
                time.sleep(0.5)
    
    def release(self):
        """Освободить блокировку"""
        if not self._acquired:
            return
        
        try:
            if self.file_handle:
                # Освободить блокировку (платформозависимо)
                if sys.platform == 'win32':
                    self._unlock_file_windows(self.file_handle)
                else:
                    self._unlock_file_unix(self.file_handle)
                
                self.file_handle.close()
                self.file_handle = None
            
            # Удалить lock-файл
            if self.lock_file and self.lock_file.exists():
                self.lock_file.unlink()
            
            self._acquired = False
            logger.info(f"Lock {self.lock_name} освобожден")
            
        except Exception as e:
            logger.error(f"Ошибка освобождения lock {self.lock_name}: {e}")
    
    def is_locked(self) -> bool:
        """Проверить, захвачена ли блокировка (любым процессом)"""
        if self._acquired:
            return True
        
        if not self.lock_file.exists():
            return False
        
        try:
            # Попробовать открыть файл и захватить блокировку
            with open(self.lock_file, 'a') as f:
                if sys.platform == 'win32':
                    success = self._lock_file_windows(f)
                    if success:
                        self._unlock_file_windows(f)
                    return not success
                else:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return False
        except (IOError, OSError):
            # Файл заблокирован
            return True
    
    def __enter__(self):
        """Context manager: войти"""
        if not self.acquire():
            raise RuntimeError(f"Не удалось захватить блокировку {self.lock_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: выйти"""
        self.release()
        return False
    
    def __del__(self):
        """Деструктор: освободить блокировку"""
        self.release()

    
    def __enter__(self):
        """Context manager: войти"""
        if not self.acquire():
            raise RuntimeError(f"Не удалось захватить блокировку {self.lock_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: выйти"""
        self.release()
        return False
    
    def __del__(self):
        """Деструктор: освободить блокировку"""
        self.release()
