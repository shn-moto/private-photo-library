"""Обработка изображений (HEIC, JPG, PNG и т.д.)"""

import os
from pathlib import Path
from typing import Tuple, Optional
import logging
from PIL import Image
import pillow_heif
import cv2
import numpy as np

logger = logging.getLogger(__name__)
pillow_heif.register_heif_opener()

# RAW support (NEF, CR2, ARW)
try:
    import rawpy
    HAS_RAWPY = True
    logger.info("RAW support enabled (NEF/CR2/ARW)")
except ImportError:
    HAS_RAWPY = False
    logger.warning("rawpy не установлен, RAW файлы (NEF/CR2/ARW) не будут поддерживаться")

RAW_EXTENSIONS = {".nef", ".cr2", ".arw", ".dng", ".raf", ".orf", ".rw2"}


class ImageProcessor:
    """Процессор изображений"""

    SUPPORTED_FORMATS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".nef", ".cr2", ".arw", ".dng", ".raf", ".orf", ".rw2"}
    
    def __init__(self, max_size: Tuple[int, int] = (1024, 1024)):
        self.max_size = max_size
    
    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """Проверить, поддерживается ли формат файла"""
        ext = Path(file_path).suffix.lower()
        return ext in ImageProcessor.SUPPORTED_FORMATS
    
    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        Загрузить изображение и вернуть как numpy array (BGR)

        Args:
            file_path: Путь к файлу изображения

        Returns:
            numpy array в формате BGR или None если ошибка
        """
        try:
            ext = Path(file_path).suffix.lower()

            # RAW форматы (NEF, CR2, ARW и др.)
            if ext in RAW_EXTENSIONS:
                if not HAS_RAWPY:
                    logger.warning(f"rawpy не установлен, пропуск RAW файла: {file_path}")
                    return None
                with rawpy.imread(file_path) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # HEIC/HEIF
            if ext in ('.heic', '.heif'):
                img = Image.open(file_path).convert('RGB')
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Остальные форматы (JPG, PNG, BMP, WebP)
            img = cv2.imread(file_path)
            if img is None:
                # Fallback на Pillow
                img = Image.open(file_path).convert('RGB')
                return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {file_path}: {e}")
            return None
    
    def load_image_rgb(self, file_path: str) -> Optional[np.ndarray]:
        """Загрузить изображение в RGB формате"""
        img_bgr = self.load_image(file_path)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return None
    
    def resize_image(self, image: np.ndarray, max_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Изменить размер изображения, сохраняя aspect ratio
        
        Args:
            image: numpy array изображения
            max_size: максимальный размер (width, height)
            
        Returns:
            Ресайзированное изображение
        """
        if max_size is None:
            max_size = self.max_size
        
        h, w = image.shape[:2]
        max_w, max_h = max_size
        
        # Если размер уже меньше максимального
        if w <= max_w and h <= max_h:
            return image
        
        # Вычислить масштаб
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Нормализовать изображение для CLIP модели
        Преобразование в float32 и нормализация по стандартным значениям ImageNet
        """
        img = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Если RGB то прямая нормализация
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = (img - mean) / std
        
        return img
    
    def get_image_dimensions(self, file_path: str) -> Optional[Tuple[int, int]]:
        """
        Получить размеры изображения (width, height) без полной загрузки
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Кортеж (width, height) или None
        """
        try:
            with Image.open(file_path) as img:
                return img.size
        except Exception as e:
            logger.error(f"Ошибка получения размеров {file_path}: {e}")
            return None
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Получить информацию о файле изображения
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Словарь с информацией
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            dimensions = self.get_image_dimensions(file_path)
            
            return {
                "file_name": path.name,
                "file_size": stat.st_size,
                "file_format": path.suffix.lower()[1:],  # без точки
                "width": dimensions[0] if dimensions else None,
                "height": dimensions[1] if dimensions else None,
                "created_at": path.stat().st_ctime,
                "modified_at": path.stat().st_mtime,
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о файле {file_path}: {e}")
            return {}
    
    def extract_exif(self, file_path: str) -> dict:
        """
        Извлечь EXIF данные из изображения
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Словарь с EXIF данными
        """
        try:
            from PIL import Image as PILImage
            from PIL.ExifTags import TAGS
            
            image = PILImage.open(file_path)
            exif_data = {}
            
            if hasattr(image, '_getexif') and image._getexif() is not None:
                for tag_id, value in image._getexif().items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    # Очистить невалидные символы ДО конвертации в строку
                    if isinstance(value, bytes):
                        try:
                            # Попытаться декодировать и очистить
                            value = value.decode('utf-8', errors='ignore').strip('\x00').strip()
                        except:
                            value = str(value)
                    else:
                        value = str(value)
                    
                    # Финальная очистка от нулевых символов и контроля
                    clean_value = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')[:256]
                    if clean_value.strip():  # Сохранить только непустые значения
                        exif_data[tag_name] = clean_value
            
            return exif_data
        except Exception as e:
            logger.warning(f"Ошибка извлечения EXIF из {file_path}: {e}")
            return {}
