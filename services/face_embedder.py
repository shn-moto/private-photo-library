"""Face detection and embedding service using InsightFace"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from loguru import logger

# RAW file extensions that need special handling
RAW_EXTENSIONS = {".nef", ".cr2", ".arw", ".dng", ".raf", ".orf", ".rw2"}

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
    # Lowered from 0.65 → 0.45: captures tilted/angled/partial faces that score 0.45-0.65
    # Tradeoff: occasional false positive on blurry faces or faces on posters
    MIN_DET_SCORE = 0.45
    DET_SIZE = (640, 640)  # Detection input size
    # InsightFace internal threshold — must be low enough to get all candidates for filtering
    # Set to 0.25 to allow very low detection scores to be returned and filtered by min_det_score
    DET_THRESH = 0.25

    def __init__(self, device: str = "cuda", min_det_score: float = None):
        """
        Initialize InsightFace model.

        Args:
            device: "cuda" or "cpu"
            min_det_score: Minimum detection score (default 0.45)
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
        # det_thresh: internal InsightFace threshold (must be ≤ MIN_DET_SCORE)
        self.app = FaceAnalysis(name=self.MODEL_NAME, providers=self.providers)
        self.app.prepare(ctx_id=ctx_id, det_size=self.DET_SIZE, det_thresh=self.DET_THRESH)

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
                file_path = str(image)
                ext = Path(file_path).suffix.lower()

                # RAW files need special handling via rawpy
                if ext in RAW_EXTENSIONS:
                    return self._load_raw_image(file_path)

                # Standard formats via PIL
                img = Image.open(file_path).convert("RGB")
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

    def _load_raw_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load RAW image (NEF, CR2, ARW, etc.) using rawpy.

        Uses same processing parameters as ImageProcessor for consistency
        between CLIP embeddings and face embeddings.

        Args:
            file_path: Path to RAW file

        Returns:
            RGB numpy array or None if loading fails
        """
        try:
            import rawpy
        except ImportError:
            logger.warning(f"rawpy not installed, cannot load RAW file: {file_path}")
            return None

        try:
            with rawpy.imread(file_path) as raw:
                # Use same parameters as ImageProcessor for consistency
                # rawpy.postprocess() automatically applies rotation based on raw.sizes.flip
                # DO NOT apply additional EXIF rotation - it would double-rotate!
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8  # 8-bit output for consistency with JPEG/HEIC
                )

            return rgb
        except Exception as e:
            logger.warning(f"Failed to load RAW image {file_path}: {e}")
            return None

    def _process_single_image(self, img_np: Optional[np.ndarray], min_det_score: float = None, det_size: tuple = None) -> List[FaceResult]:
        """
        Runs face detection on a single, pre-loaded numpy image.
        Helper for parallel execution.
        """
        if img_np is None:
            return []

        # Use provided threshold or default
        threshold = min_det_score if min_det_score is not None else self.min_det_score

        try:
            # Temporarily lower the internal detection threshold if needed.
            # Must update det_model.det_thresh (not just app.det_thresh) —
            # InsightFace's ONNX detection model reads its own attribute, not FaceAnalysis.det_thresh.
            original_det_thresh = getattr(self.app, 'det_thresh', 0.4)
            det_model = getattr(self.app, 'det_model', None)
            original_model_thresh = getattr(det_model, 'det_thresh', original_det_thresh) if det_model else original_det_thresh
            if threshold < original_det_thresh:
                logger.debug(f"Lowering det_thresh from {original_det_thresh} to {threshold}")
                self.app.det_thresh = threshold
                if det_model is not None:
                    det_model.det_thresh = threshold

            # Temporarily change detection resolution if requested.
            # Must update det_model.input_size — app.det_size is passed as `metric` (not input_size)
            # to det_model.detect(); actual image resize uses det_model.input_size.
            original_det_size = getattr(det_model, 'input_size', self.DET_SIZE) if det_model else self.DET_SIZE
            if det_size and det_size != original_det_size and det_model is not None:
                logger.debug(f"Changing det_model.input_size from {original_det_size} to {det_size}")
                det_model.input_size = det_size

            faces = self.app.get(img_np)

            # Restore original thresholds and det_size
            if threshold < original_det_thresh:
                self.app.det_thresh = original_det_thresh
                if det_model is not None:
                    det_model.det_thresh = original_model_thresh
            if det_size and det_size != original_det_size and det_model is not None:
                det_model.input_size = original_det_size
            
            logger.debug(f"InsightFace found {len(faces) if faces else 0} raw faces, filtering with threshold={threshold}")
            
            if not faces:
                return []

            results = []
            for face in faces:
                if face.bbox is None or face.embedding is None:
                    continue

                det_score = float(face.det_score) if face.det_score is not None else 0.0
                logger.debug(f"Face det_score={det_score:.3f}, threshold={threshold}")
                if det_score < threshold:
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

    def detect_faces(self, image: Union[str, Path, Image.Image, np.ndarray], min_det_score: float = None, det_size: tuple = None) -> List[FaceResult]:
        """
        Detect all faces in a single image.

        Args:
            image: File path, PIL Image, or numpy array
            min_det_score: Minimum detection score threshold (default: use self.min_det_score)
            det_size: Detection input resolution, e.g. (1280, 1280) for better detection of small faces

        Returns:
            List of FaceResult objects
        """
        img_np = self._load_image(image)
        return self._process_single_image(img_np, min_det_score=min_det_score, det_size=det_size)

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
