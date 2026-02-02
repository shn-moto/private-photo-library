"""Face detection and embedding service using InsightFace"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from loguru import logger

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# InsightFace
try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False
    logger.warning("InsightFace not available. Face detection will be disabled.")


@dataclass
class FaceResult:
    """Result from face detection"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    det_score: float  # Detection confidence (0.0 - 1.0)
    landmarks: Optional[np.ndarray]  # 5-point or 106-point landmarks
    embedding: np.ndarray  # 512-dim face embedding
    age: Optional[int]  # Estimated age
    gender: Optional[int]  # 0=female, 1=male


class FaceEmbedder:
    """Face detection and embedding using InsightFace buffalo_l model"""

    MODEL_NAME = "buffalo_l"
    EMBEDDING_DIM = 512
    MIN_DET_SCORE = 0.65  # Minimum detection confidence
    DET_SIZE = (640, 640)  # Detection input size

    def __init__(self, device: str = "cuda", min_det_score: float = None):
        """
        Initialize InsightFace model.

        Args:
            device: "cuda" or "cpu"
            min_det_score: Minimum detection score (default 0.65)
        """
        if not HAS_INSIGHTFACE:
            raise RuntimeError("InsightFace is not installed. Run: pip install insightface onnxruntime-gpu")

        self.device = device
        self.min_det_score = min_det_score or self.MIN_DET_SCORE

        # Setup execution providers based on device
        if device == "cuda":
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            self.providers = ["CPUExecutionProvider"]
            ctx_id = -1

        logger.info(f"Initializing InsightFace model '{self.MODEL_NAME}' on {device}")

        # Initialize FaceAnalysis
        self.app = FaceAnalysis(name=self.MODEL_NAME, providers=self.providers)
        self.app.prepare(ctx_id=ctx_id, det_size=self.DET_SIZE)

        logger.info(f"InsightFace model loaded. Detection size: {self.DET_SIZE}")

    def _load_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """
        Load image and convert to RGB numpy array.

        Args:
            image: File path, PIL Image, or numpy array

        Returns:
            RGB numpy array or None if loading fails
        """
        try:
            if isinstance(image, np.ndarray):
                # Already numpy array
                if len(image.shape) == 3 and image.shape[2] == 3:
                    return image
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    # RGBA -> RGB
                    return image[:, :, :3]
                return None

            if isinstance(image, (str, Path)):
                # Load from file
                img = Image.open(image).convert("RGB")
                # Apply EXIF orientation - CRITICAL for correct bbox coordinates!
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            elif isinstance(image, Image.Image):
                img = image.convert("RGB")
                # Apply EXIF orientation
                from PIL import ImageOps
                img = ImageOps.exif_transpose(img)
            else:
                logger.warning(f"Unsupported image type: {type(image)}")
                return None

            return np.array(img)

        except Exception as e:
            logger.warning(f"Failed to load image: {e}")
            return None

    def _process_single_image(self, img_np: Optional[np.ndarray]) -> List[FaceResult]:
        """
        Runs face detection on a single, pre-loaded numpy image.
        Helper for parallel execution.
        """
        if img_np is None:
            return []

        try:
            faces = self.app.get(img_np)
            if not faces:
                return []

            results = []
            for face in faces:
                if face.bbox is None or face.embedding is None:
                    continue

                det_score = float(face.det_score) if face.det_score is not None else 0.0
                if det_score < self.min_det_score:
                    continue

                age = int(face.age) if hasattr(face, "age") and face.age is not None else None
                gender = int(face.gender) if hasattr(face, "gender") and face.gender is not None else None
                landmarks = face.landmark.tolist() if face.landmark is not None else None

                results.append(FaceResult(
                    bbox=tuple(face.bbox.tolist()),
                    det_score=det_score,
                    landmarks=landmarks,
                    embedding=face.embedding.astype(np.float32),
                    age=age,
                    gender=gender
                ))
            return results
        except Exception as e:
            logger.warning(f"Face detection failed for one image: {e}")
            return []

    def detect_faces(self, image: Union[str, Path, Image.Image, np.ndarray]) -> List[FaceResult]:
        """
        Detect all faces in a single image.

        Args:
            image: File path, PIL Image, or numpy array

        Returns:
            List of FaceResult objects
        """
        img_np = self._load_image(image)
        return self._process_single_image(img_np)

    def detect_faces_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 8,
        num_workers: int = 4
    ) -> List[List[FaceResult]]:
        """
        Batch face detection for multiple images with parallel loading and inference.
        This uses a thread pool to run multiple inference calls in parallel to
        better utilize the GPU, since the underlying model does not support
        true batching through the FaceAnalysis API.

        Args:
            images: List of file paths, PIL Images, or numpy arrays.
            batch_size: Not used. Kept for API compatibility.
            num_workers: Number of threads for both image loading and parallel inference.
                         A value of 2-4 is recommended for a single GPU.

        Returns:
            List of face result lists, in the same order as the input images.
        """
        start_time = time.time()
        total = len(images)

        if not images:
            return []

        # Clamp workers to a reasonable number for GPU tasks to avoid OOM.
        # Too many workers can also cause thread contention.
        gpu_workers = min(num_workers, 8) if self.device == "cuda" else num_workers

        with ThreadPoolExecutor(max_workers=gpu_workers) as executor:
            # Stage 1: Load images from paths in parallel
            # The map function preserves the order of the input list.
            logger.info(f"Loading {total} images with {gpu_workers} workers...")
            loaded_images = list(executor.map(self._load_image, images))

            # Stage 2: Process loaded images in parallel
            logger.info(f"Detecting faces in {total} images with {gpu_workers} workers...")
            results = list(executor.map(self._process_single_image, loaded_images))

        successful = sum(1 for r in results if r)
        total_faces = sum(len(r) for r in results)
        elapsed = time.time() - start_time
        speed = total / elapsed if elapsed > 0 else 0

        logger.info(
            f"Batch face detection complete: {total} images in {elapsed:.1f}s "
            f"({speed:.1f} img/s). Found {total_faces} faces in {successful} images "
            f"(processed with {gpu_workers} workers)."
        )

        return results

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Get embedding for a pre-cropped face image.
        Note: This runs full detection on the crop, so the crop should contain a clear face.

        Args:
            face_crop: RGB numpy array of cropped face

        Returns:
            512-dim embedding or None if no face detected
        """
        faces = self.detect_faces(face_crop)
        if not faces:
            return None
        # Return embedding of the most confident face
        best_face = max(faces, key=lambda f: f.det_score)
        return best_face.embedding

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two face embeddings.

        Args:
            embedding1: First 512-dim embedding
            embedding2: Second 512-dim embedding

        Returns:
            Cosine similarity (-1 to 1, higher is more similar)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    @staticmethod
    def l2_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate L2 (Euclidean) distance between two face embeddings.

        Args:
            embedding1: First 512-dim embedding
            embedding2: Second 512-dim embedding

        Returns:
            L2 distance (lower is more similar)
        """
        return float(np.linalg.norm(embedding1 - embedding2))
