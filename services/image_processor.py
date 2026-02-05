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
            ext = Path(file_path).suffix.lower()

            # RAW files need rawpy - PIL only reads embedded thumbnail
            if ext in RAW_EXTENSIONS:
                if not HAS_RAWPY:
                    logger.warning(f"rawpy not available for RAW file: {file_path}")
                    return None
                with rawpy.imread(file_path) as raw:
                    # raw.sizes.width/height are sensor dimensions BEFORE rotation
                    # raw.sizes.flip indicates rotation applied by postprocess():
                    #   0=none, 3=180°, 5=90°CCW, 6=90°CW
                    # For 90° rotations (flip 5 or 6), swap width/height
                    w, h = raw.sizes.width, raw.sizes.height
                    if raw.sizes.flip in (5, 6):
                        return (h, w)  # Swap for 90° rotations
                    return (w, h)

            # Standard formats via PIL
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
    
    def extract_exif(self, file_path: str) -> Optional[dict]:
        """
        Извлечь EXIF данные из изображения (поддержка всех форматов)

        Args:
            file_path: Путь к файлу

        Returns:
            Словарь с EXIF данными или None если данных нет
        """
        ext = Path(file_path).suffix.lower()

        # Сначала пробуем exifread (работает с большинством форматов включая RAW)
        try:
            exif_data = self._extract_exif_exifread(file_path)
            if exif_data:
                return exif_data
        except Exception as e:
            logger.debug(f"exifread не смог прочитать {file_path}: {e}")

        # Для HEIC используем pillow-heif
        if ext in ('.heic', '.heif'):
            try:
                exif_data = self._extract_exif_heif(file_path)
                if exif_data:
                    return exif_data
            except Exception as e:
                logger.debug(f"pillow-heif не смог прочитать EXIF из {file_path}: {e}")

        # Fallback на Pillow для JPG/PNG
        try:
            exif_data = self._extract_exif_pillow(file_path)
            if exif_data:
                return exif_data
        except Exception as e:
            logger.debug(f"Pillow не смог прочитать EXIF из {file_path}: {e}")

        return None

    def _extract_exif_exifread(self, file_path: str) -> dict:
        """Извлечь EXIF через exifread (универсальный метод)"""
        try:
            import exifread
        except ImportError:
            logger.warning("exifread не установлен, некоторые форматы могут не поддерживаться")
            return {}

        exif_data = {}
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            for tag_name, value in tags.items():
                # Пропустить служебные теги
                if tag_name.startswith('Thumbnail') or tag_name.startswith('EXIF MakerNote'):
                    continue

                # Очистить имя тега (убрать префиксы типа "EXIF ", "Image ")
                clean_name = tag_name
                for prefix in ['EXIF ', 'Image ', 'GPS ', 'Thumbnail ']:
                    if clean_name.startswith(prefix):
                        clean_name = clean_name[len(prefix):]
                        break

                # Конвертировать значение в строку
                str_value = str(value).strip()
                if str_value and len(str_value) <= 256:
                    exif_data[clean_name] = str_value

            return exif_data
        except Exception as e:
            logger.debug(f"Ошибка exifread для {file_path}: {e}")
            return {}

    def _extract_exif_heif(self, file_path: str) -> dict:
        """Извлечь EXIF из HEIC/HEIF через pillow-heif"""
        exif_data = {}
        try:
            import pillow_heif
            from PIL.ExifTags import TAGS

            heif_file = pillow_heif.open_heif(file_path)

            # Попробовать получить EXIF
            if hasattr(heif_file, 'info') and 'exif' in heif_file.info:
                from PIL import Image as PILImage
                import io

                # Pillow может декодировать EXIF данные из HEIF
                img = PILImage.open(file_path)
                if hasattr(img, '_getexif') and img._getexif():
                    for tag_id, value in img._getexif().items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        str_value = self._clean_exif_value(value)
                        if str_value:
                            exif_data[tag_name] = str_value

            return exif_data
        except Exception as e:
            logger.debug(f"Ошибка извлечения EXIF из HEIF {file_path}: {e}")
            return {}

    def _extract_exif_pillow(self, file_path: str) -> dict:
        """Извлечь EXIF через Pillow (fallback для JPG/PNG)"""
        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS

        exif_data = {}
        try:
            image = PILImage.open(file_path)

            if hasattr(image, '_getexif') and image._getexif() is not None:
                for tag_id, value in image._getexif().items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    str_value = self._clean_exif_value(value)
                    if str_value:
                        exif_data[tag_name] = str_value

            return exif_data
        except Exception as e:
            logger.debug(f"Ошибка Pillow EXIF для {file_path}: {e}")
            return {}

    def _clean_exif_value(self, value) -> str:
        """Очистить EXIF значение и конвертировать в строку"""
        if isinstance(value, bytes):
            try:
                value = value.decode('utf-8', errors='ignore').strip('\x00').strip()
            except:
                value = str(value)
        else:
            value = str(value)

        # Убрать непечатаемые символы
        clean_value = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')[:256]
        return clean_value.strip() if clean_value.strip() else None

    def extract_gps_coordinates(self, file_path: str) -> tuple:
        """
        Извлечь GPS координаты из изображения

        Args:
            file_path: Путь к файлу

        Returns:
            Кортеж (latitude, longitude) или (None, None) если нет GPS данных
        """
        ext = Path(file_path).suffix.lower()

        # Используем exifread для GPS (наиболее надежный способ)
        try:
            import exifread

            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            # Проверить наличие GPS данных
            gps_lat = tags.get('GPS GPSLatitude')
            gps_lat_ref = tags.get('GPS GPSLatitudeRef')
            gps_lon = tags.get('GPS GPSLongitude')
            gps_lon_ref = tags.get('GPS GPSLongitudeRef')

            if gps_lat and gps_lon:
                lat = self._convert_gps_to_decimal(gps_lat.values, gps_lat_ref)
                lon = self._convert_gps_to_decimal(gps_lon.values, gps_lon_ref)
                if lat is not None and lon is not None:
                    return (lat, lon)
        except ImportError:
            logger.debug("exifread не установлен для GPS")
        except Exception as e:
            logger.debug(f"Ошибка извлечения GPS через exifread: {e}")

        # Fallback: Pillow для JPEG
        try:
            from PIL import Image as PILImage
            from PIL.ExifTags import TAGS, GPSTAGS

            img = PILImage.open(file_path)
            exif = img._getexif()

            if exif:
                gps_info = {}
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'GPSInfo':
                        for gps_tag_id, gps_value in value.items():
                            gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                            gps_info[gps_tag] = gps_value

                if gps_info:
                    lat = self._get_decimal_from_dms(
                        gps_info.get('GPSLatitude'),
                        gps_info.get('GPSLatitudeRef')
                    )
                    lon = self._get_decimal_from_dms(
                        gps_info.get('GPSLongitude'),
                        gps_info.get('GPSLongitudeRef')
                    )
                    if lat is not None and lon is not None:
                        return (lat, lon)
        except Exception as e:
            logger.debug(f"Ошибка извлечения GPS через Pillow: {e}")

        return (None, None)

    def _convert_gps_to_decimal(self, dms_values, ref) -> float:
        """Конвертировать GPS координаты из DMS в decimal (для exifread)"""
        try:
            if not dms_values or len(dms_values) < 3:
                return None

            # exifread возвращает Ratio объекты
            degrees = float(dms_values[0].num) / float(dms_values[0].den)
            minutes = float(dms_values[1].num) / float(dms_values[1].den)
            seconds = float(dms_values[2].num) / float(dms_values[2].den)

            decimal = degrees + minutes / 60 + seconds / 3600

            # Проверить направление (S/W = отрицательное)
            if ref and str(ref) in ['S', 'W']:
                decimal = -decimal

            return round(decimal, 6)
        except Exception as e:
            logger.debug(f"Ошибка конвертации GPS DMS: {e}")
            return None

    def _get_decimal_from_dms(self, dms, ref) -> float:
        """Конвертировать GPS координаты из DMS в decimal (для Pillow)"""
        try:
            if not dms:
                return None

            degrees = float(dms[0])
            minutes = float(dms[1])
            seconds = float(dms[2])

            decimal = degrees + minutes / 60 + seconds / 3600

            if ref and ref in ['S', 'W']:
                decimal = -decimal

            return round(decimal, 6)
        except Exception as e:
            logger.debug(f"Ошибка конвертации GPS: {e}")
            return None

    def extract_photo_date(self, file_path: str) -> 'datetime':
        """
        Извлечь дату съемки из EXIF

        Args:
            file_path: Путь к файлу

        Returns:
            datetime или None если дата не найдена
        """
        from datetime import datetime

        # Приоритет тегов даты
        date_tags = [
            'DateTimeOriginal',
            'DateTimeDigitized',
            'DateTime',
            'CreateDate',
            'ModifyDate'
        ]

        exif_data = self.extract_exif(file_path)

        for tag in date_tags:
            if tag in exif_data:
                try:
                    date_str = exif_data[tag]
                    # Формат EXIF: "YYYY:MM:DD HH:MM:SS"
                    dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    return dt
                except ValueError:
                    # Попробовать альтернативные форматы
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y:%m:%d"]:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except ValueError:
                            continue

        return None

    def get_full_metadata(self, file_path: str) -> dict:
        """
        Получить полные метаданные файла включая EXIF, GPS и дату

        Args:
            file_path: Путь к файлу

        Returns:
            Словарь с полными метаданными
        """
        # Базовая информация о файле
        info = self.get_file_info(file_path)

        # EXIF данные
        info['exif_data'] = self.extract_exif(file_path)

        # GPS координаты
        lat, lon = self.extract_gps_coordinates(file_path)
        info['latitude'] = lat
        info['longitude'] = lon

        # Дата съемки
        info['photo_date'] = self.extract_photo_date(file_path)

        return info
