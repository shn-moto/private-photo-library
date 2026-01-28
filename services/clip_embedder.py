"""CLIP эмбеддинги используя HuggingFace transformers"""

import logging
from typing import List, Union, Optional
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def get_video_extension(file_path: str) -> Optional[str]:
    """Проверить тип файла по magic bytes и вернуть правильное расширение для видео"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)
        # QuickTime/MOV/MP4: начинается с ftyp
        if len(header) >= 8 and header[4:8] == b'ftyp':
            ftyp = header[8:12].lower()
            # QuickTime
            if ftyp in (b'qt  ', b'mqt '):
                return '.mov'
            # MP4 варианты
            if ftyp in (b'mp4 ', b'mp41', b'mp42', b'isom', b'avc1', b'm4v '):
                return '.mp4'
            # MOV
            if ftyp == b'mov ':
                return '.mov'
        return None
    except Exception:
        return None


def is_video_file(file_path: str) -> bool:
    """Проверить, является ли файл видео"""
    return get_video_extension(file_path) is not None


def fix_video_extension(file_path: str) -> Optional[str]:
    """Переименовать видео файл с неправильным расширением. Возвращает новый путь или None"""
    import os

    video_ext = get_video_extension(file_path)
    if video_ext is None:
        return None

    current_ext = os.path.splitext(file_path)[1].lower()

    # Если расширение уже правильное
    if current_ext == video_ext:
        return None

    # Новое имя файла
    base = os.path.splitext(file_path)[0]
    new_path = base + video_ext

    # Если файл с таким именем уже существует, добавляем суффикс
    counter = 1
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{video_ext}"
        counter += 1

    try:
        os.rename(file_path, new_path)
        logger.info(f"Переименован: {os.path.basename(file_path)} -> {os.path.basename(new_path)}")
        return new_path
    except OSError as e:
        logger.warning(f"Не удалось переименовать {file_path}: {e}")
        return None

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    logger.info("HEIF/HEIC support enabled")
except ImportError:
    logger.warning("pillow_heif не установлен, HEIC файлы не будут поддерживаться")

# RAW support (NEF, CR2, ARW)
RAW_EXTENSIONS = {".nef", ".cr2", ".arw", ".dng", ".raf", ".orf", ".rw2"}
try:
    import rawpy
    HAS_RAWPY = True
    logger.info("RAW support enabled (NEF/CR2/ARW)")
except ImportError:
    HAS_RAWPY = False
    logger.warning("rawpy не установлен, RAW файлы (NEF/CR2/ARW) не будут поддерживаться")

# Transformers
try:
    from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoImageProcessor, SiglipProcessor, SiglipTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers не установлен. Установите: pip install transformers")


def log_gpu_info():
    """Вывести информацию о GPU"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        logger.warning("CUDA недоступна, используется CPU")
        return False


class CLIPEmbedder:
    """Генератор эмбеддингов на основе CLIP (HuggingFace)"""

    MODEL_MAPPING = {
        "ViT-B/32": "openai/clip-vit-base-patch32",
        "ViT-B/16": "openai/clip-vit-base-patch16",
        "ViT-L/14": "openai/clip-vit-large-patch14",
        "SigLIP": "google/siglip-so400m-patch14-384",
    }

    # Модели, которые используют AutoModel/AutoProcessor вместо CLIPModel/CLIPProcessor
    SIGLIP_MODELS = {"google/siglip-so400m-patch14-384"}

    def __init__(self, model_name: str = "SigLIP", device: str = "cuda"):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers не установлен")

        self.model_name = model_name

        # Проверка GPU
        has_gpu = log_gpu_info()
        self.device = device if has_gpu else "cpu"

        # HuggingFace model name
        hf_model_name = self.MODEL_MAPPING.get(model_name, model_name)

        logger.info(f"Загрузка модели {hf_model_name}...")

        self._is_siglip = hf_model_name in self.SIGLIP_MODELS

        if self._is_siglip:
            self.model = AutoModel.from_pretrained(hf_model_name).to(self.device)
            image_processor = AutoImageProcessor.from_pretrained(hf_model_name)
            tokenizer = SiglipTokenizer.from_pretrained(hf_model_name)
            self.processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
        else:
            if not hf_model_name.startswith("openai/"):
                hf_model_name = f"openai/{model_name}"
            self.model = CLIPModel.from_pretrained(hf_model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(hf_model_name, use_fast=True)

        self.model.eval()

        # Определяем размерность эмбеддинга
        if hasattr(self.model.config, 'projection_dim'):
            self.embedding_dim = self.model.config.projection_dim
        elif hasattr(self.model.config, 'text_config'):
            self.embedding_dim = self.model.config.text_config.hidden_size
        else:
            self.embedding_dim = 1152

        # GPU memory info
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"Модель загружена, GPU memory: {allocated:.2f} GB")

        logger.info(f"CLIP {model_name} готов (device={self.device}, dim={self.embedding_dim})")

    @torch.no_grad()
    def embed_image(self, image: Union[str, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """Получить эмбеддинг одного изображения"""
        try:
            if isinstance(image, str):
                # Пропустить видео файлы
                if is_video_file(image):
                    return None

                # RAW форматы (NEF, CR2, ARW и др.)
                import os
                ext = os.path.splitext(image)[1].lower()
                if ext in RAW_EXTENSIONS:
                    if not HAS_RAWPY:
                        logger.warning(f"rawpy не установлен, пропуск RAW файла: {image}")
                        return None
                    with rawpy.imread(image) as raw:
                        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False)
                    image = Image.fromarray(rgb)
                else:
                    image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8')).convert("RGB")
            elif isinstance(image, Image.Image):
                image = image.convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            outputs = self.model.get_image_features(**inputs)
            # transformers 5.0+ возвращает объект, извлекаем тензор
            if hasattr(outputs, 'pooler_output'):
                image_features = outputs.pooler_output
            else:
                image_features = outputs
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            logger.error(f"Ошибка эмбеддинга: {e}")
            return None

    def _load_single_image(self, img: Union[str, Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """Загрузить одно изображение (для параллельной загрузки)"""
        try:
            if isinstance(img, str):
                # Пропустить видео файлы (Live Photos и т.п.)
                if is_video_file(img):
                    return None

                # RAW форматы (NEF, CR2, ARW и др.)
                import os
                ext = os.path.splitext(img)[1].lower()
                if ext in RAW_EXTENSIONS:
                    if not HAS_RAWPY:
                        logger.warning(f"rawpy не установлен, пропуск RAW файла: {img}")
                        return None
                    with rawpy.imread(img) as raw:
                        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False)
                    return Image.fromarray(rgb)

                return Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                return Image.fromarray(img.astype('uint8')).convert("RGB")
            else:
                return img.convert("RGB")
        except Exception as e:
            logger.warning(f"Ошибка загрузки изображения {img}: {e}")
            return None

    @torch.no_grad()
    def embed_images_batch(self, images: List[Union[str, Image.Image, np.ndarray]],
                          batch_size: int = 32) -> List[Optional[np.ndarray]]:
        """Получить эмбеддинги для батча изображений (GPU optimized)"""
        from concurrent.futures import ThreadPoolExecutor

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Параллельная загрузка изображений (8 потоков)
            with ThreadPoolExecutor(max_workers=8) as executor:
                loaded_images = list(executor.map(self._load_single_image, batch))

            pil_images = []
            valid_indices = []

            for idx, pil_img in enumerate(loaded_images):
                if pil_img is not None:
                    pil_images.append(pil_img)
                    valid_indices.append(idx)

            batch_results = [None] * len(batch)

            if pil_images:
                # GPU batch processing
                inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)

                outputs = self.model.get_image_features(**inputs)
                # transformers 5.0+ возвращает объект, извлекаем тензор
                if hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                else:
                    image_features = outputs
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                embeddings = image_features.cpu().numpy().astype(np.float32)

                for result_idx, original_idx in enumerate(valid_indices):
                    batch_results[original_idx] = embeddings[result_idx]

            all_embeddings.extend(batch_results)

        return all_embeddings

    @torch.no_grad()
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Получить эмбеддинг текста"""
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            # transformers 5.0+ возвращает объект, извлекаем тензор
            if hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            else:
                text_features = outputs
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"Ошибка эмбеддинга текста: {e}")
            return None

    @torch.no_grad()
    def embed_texts_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Получить эмбеддинги для списка текстов"""
        try:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            # transformers 5.0+ возвращает объект, извлекаем тензор
            if hasattr(outputs, 'pooler_output'):
                text_features = outputs.pooler_output
            else:
                text_features = outputs
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return list(text_features.cpu().numpy().astype(np.float32))
        except Exception as e:
            logger.error(f"Ошибка батча текстовых эмбеддингов: {e}")
            return [None] * len(texts)

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Косинусное сходство"""
        return float(np.dot(embedding1, embedding2) /
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8))

    def search_by_text(self, text: str, image_embeddings: np.ndarray) -> np.ndarray:
        """Поиск по тексту"""
        text_embedding = self.embed_text(text)
        if text_embedding is None:
            return np.zeros(len(image_embeddings))
        return image_embeddings @ text_embedding

    def search_by_image(self, query_image: Union[str, Image.Image, np.ndarray],
                        image_embeddings: np.ndarray) -> np.ndarray:
        """Поиск по изображению"""
        query_embedding = self.embed_image(query_image)
        if query_embedding is None:
            return np.zeros(len(image_embeddings))
        return image_embeddings @ query_embedding
