"""Face indexing service for detecting and storing faces in database"""

import os
from typing import List, Dict, Set, Optional, Callable
from pathlib import Path
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import func, text
from loguru import logger

from models.data_models import PhotoIndex, Face, Person, FACE_EMBEDDING_DIM
from services.face_embedder import FaceEmbedder, FaceResult


class FaceRepository:
    """Repository for face database operations"""

    def __init__(self, session_factory: Callable[[], Session]):
        """
        Initialize repository.

        Args:
            session_factory: Callable that returns a new SQLAlchemy session
        """
        self.session_factory = session_factory

    def add_face(self, session: Session, face_data: dict) -> int:
        """
        Add a face to the database.

        Args:
            session: SQLAlchemy session
            face_data: Dict with face attributes

        Returns:
            face_id of the created face
        """
        face = Face(
            image_id=face_data["image_id"],
            bbox_x1=face_data["bbox"][0],
            bbox_y1=face_data["bbox"][1],
            bbox_x2=face_data["bbox"][2],
            bbox_y2=face_data["bbox"][3],
            det_score=face_data["det_score"],
            landmarks=face_data.get("landmarks"),
            age=face_data.get("age"),
            gender=face_data.get("gender"),
            face_embedding=face_data["embedding"],
            person_id=face_data.get("person_id")
        )
        session.add(face)
        session.flush()
        return face.face_id

    def get_faces_by_image(self, session: Session, image_id: int) -> List[Face]:
        """Get all faces for an image."""
        return session.query(Face).filter(Face.image_id == image_id).all()

    def get_face_by_id(self, session: Session, face_id: int) -> Optional[Face]:
        """Get face by ID."""
        return session.query(Face).filter(Face.face_id == face_id).first()

    def get_faces_by_person(self, session: Session, person_id: int) -> List[Face]:
        """Get all faces for a person."""
        return session.query(Face).filter(Face.person_id == person_id).all()

    def update_face_person(self, session: Session, face_id: int, person_id: Optional[int]) -> bool:
        """
        Assign or unassign a face to a person.

        Args:
            session: SQLAlchemy session
            face_id: Face ID
            person_id: Person ID or None to unassign

        Returns:
            True if face was found and updated
        """
        face = session.query(Face).filter(Face.face_id == face_id).first()
        if not face:
            return False
        face.person_id = person_id
        return True

    def search_similar_faces(
        self,
        session: Session,
        embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Search for similar faces using pgvector cosine distance.

        Args:
            session: SQLAlchemy session
            embedding: 512-dim face embedding
            top_k: Maximum number of results
            threshold: Minimum similarity (0-1)

        Returns:
            List of dicts with face_id, image_id, similarity, person_id, person_name
        """
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
        cosine_threshold = 1.0 - threshold

        query = text("""
            SELECT
                f.face_id,
                f.image_id,
                p.file_path,
                1 - (f.face_embedding <=> :embedding::vector) as similarity,
                f.person_id,
                per.name as person_name,
                f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
                f.age, f.gender
            FROM faces f
            JOIN photo_index p ON p.image_id = f.image_id
            LEFT JOIN person per ON per.person_id = f.person_id
            WHERE (f.face_embedding <=> :embedding::vector) <= :threshold
            ORDER BY f.face_embedding <=> :embedding::vector
            LIMIT :top_k
        """)

        results = session.execute(query, {
            "embedding": embedding_str,
            "threshold": cosine_threshold,
            "top_k": top_k
        }).fetchall()

        return [
            {
                "face_id": row[0],
                "image_id": row[1],
                "file_path": row[2],
                "similarity": float(row[3]),
                "person_id": row[4],
                "person_name": row[5],
                "bbox": [row[6], row[7], row[8], row[9]],
                "age": row[10],
                "gender": row[11]
            }
            for row in results
        ]

    def get_unassigned_faces(self, session: Session, limit: int = 100, offset: int = 0) -> List[Face]:
        """Get faces without person assignment."""
        return (
            session.query(Face)
            .filter(Face.person_id == None)
            .order_by(Face.face_id)
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_indexed_image_ids(self, session: Session) -> Set[int]:
        """Get set of image_ids that already have faces indexed."""
        result = session.query(Face.image_id).distinct().all()
        return {row[0] for row in result}

    def delete_faces_by_image(self, session: Session, image_id: int) -> int:
        """Delete all faces for an image. Returns count of deleted faces."""
        count = session.query(Face).filter(Face.image_id == image_id).delete()
        return count

    def get_face_count(self, session: Session) -> int:
        """Get total face count."""
        return session.query(func.count(Face.face_id)).scalar() or 0

    def get_unassigned_count(self, session: Session) -> int:
        """Get count of unassigned faces."""
        return session.query(func.count(Face.face_id)).filter(Face.person_id == None).scalar() or 0


class FaceIndexingService:
    """Service for indexing faces in photos"""

    def __init__(self, session_factory: Callable[[], Session], device: str = "cuda"):
        """
        Initialize face indexing service.

        Args:
            session_factory: Callable that returns a new SQLAlchemy session
            device: "cuda" or "cpu"
        """
        self.session_factory = session_factory
        self.device = device
        self.face_embedder = None
        self.face_repository = FaceRepository(session_factory)

        # Indexing state
        self._state = {
            "running": False,
            "started_at": None,
            "finished_at": None,
            "total": 0,
            "processed": 0,
            "with_faces": 0,
            "faces_found": 0,
            "failed": 0,
            "current_batch": 0,
            "total_batches": 0,
            "speed_imgs_per_sec": 0.0,
            "eta_seconds": 0,
            "error": None
        }

    def _ensure_embedder(self):
        """Lazy initialization of FaceEmbedder."""
        if self.face_embedder is None:
            self.face_embedder = FaceEmbedder(device=self.device)

    def get_indexed_image_ids(self) -> Set[int]:
        """Get set of image_ids that already have faces indexed."""
        session = self.session_factory()
        try:
            return self.face_repository.get_indexed_image_ids(session)
        finally:
            session.close()

    @staticmethod
    def _normalize_embedding(embedding):
        """Normalize embedding to unit length for cosine similarity."""
        import numpy as np
        emb = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.tolist()

    def index_image(self, image_id: int, file_path: str) -> List[int]:
        """
        Index all faces in a single image.

        Args:
            image_id: Database image ID
            file_path: Path to image file

        Returns:
            List of face_ids for newly created faces
        """
        self._ensure_embedder()

        # Detect faces
        faces = self.face_embedder.detect_faces(file_path)

        session = self.session_factory()
        try:
            face_ids = []
            for face_result in faces:
                # Normalize embedding for consistent cosine similarity
                normalized_embedding = self._normalize_embedding(face_result.embedding)
                face_data = {
                    "image_id": image_id,
                    "bbox": face_result.bbox,
                    "det_score": face_result.det_score,
                    "landmarks": face_result.landmarks,
                    "age": face_result.age,
                    "gender": face_result.gender,
                    "embedding": normalized_embedding
                }
                face_id = self.face_repository.add_face(session, face_data)
                face_ids.append(face_id)

            # Установить флаг faces_indexed в photo_index
            session.execute(
                text("UPDATE photo_index SET faces_indexed = 1 WHERE image_id = :image_id"),
                {"image_id": image_id}
            )

            session.commit()
            return face_ids

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to index faces for image {image_id}: {e}")
            raise
        finally:
            session.close()

    def index_batch(self, photos: List[Dict], batch_size: int = 8) -> Dict[str, int]:
        """
        Index faces in a batch of photos.

        Args:
            photos: List of dicts with {image_id, file_path}
            batch_size: Controls both DB transaction size and GPU parallel processing workers.

        Returns:
            {processed: int, with_faces: int, total_faces: int, failed: int}
        """
        self._ensure_embedder()

        stats = {"processed": 0, "with_faces": 0, "total_faces": 0, "failed": 0}

        # Process in batches
        for i in range(0, len(photos), batch_size):
            batch = photos[i:i + batch_size]
            file_paths = [p["file_path"] for p in batch]

            # Detect faces in batch, using batch_size to control GPU parallelism
            batch_results = self.face_embedder.detect_faces_batch(
                file_paths, num_workers=batch_size
            )

            # Save to database
            session = self.session_factory()
            try:
                image_ids_to_update = []  # Список ID для обновления флага
                
                for photo, faces in zip(batch, batch_results):
                    stats["processed"] += 1
                    self._state["processed"] = stats["processed"]
                    
                    image_ids_to_update.append(photo["image_id"])  # Добавляем в список

                    if not faces:
                        continue

                    stats["with_faces"] += 1

                    for face_result in faces:
                        # Normalize embedding for consistent cosine similarity
                        normalized_embedding = self._normalize_embedding(face_result.embedding)
                        face_data = {
                            "image_id": photo["image_id"],
                            "bbox": face_result.bbox,
                            "det_score": face_result.det_score,
                            "landmarks": face_result.landmarks,
                            "age": face_result.age,
                            "gender": face_result.gender,
                            "embedding": normalized_embedding
                        }
                        try:
                            self.face_repository.add_face(session, face_data)
                            stats["total_faces"] += 1
                            self._state["faces_found"] = stats["total_faces"]
                        except Exception as e:
                            logger.warning(f"Failed to save face for image {photo['image_id']}: {e}")
                            stats["failed"] += 1

                # Установить флаг faces_indexed для всех обработанных фото
                if image_ids_to_update:
                    session.execute(
                        text("UPDATE photo_index SET faces_indexed = 1 WHERE image_id = ANY(:ids)"),
                        {"ids": image_ids_to_update}
                    )

                session.commit()

            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save batch: {e}")
                stats["failed"] += len(batch)
            finally:
                session.close()

        return stats

    def reindex_all(self, skip_indexed: bool = True, batch_size: int = 8) -> Dict[str, int]:
        """
        Reindex all photos in the database.

        Args:
            skip_indexed: If True, skip photos that already have faces indexed
            batch_size: Number of images to process in parallel and commit in one transaction.
                        Higher values increase GPU utilization but use more memory.
                        A value of 8-16 is a good start for an 8GB GPU.

        Returns:
            {total: int, processed: int, with_faces: int, total_faces: int, failed: int}
        """
        if self._state["running"]:
            raise RuntimeError("Face indexing is already running")

        import time
        start_time = time.time()

        self._state = {
            "running": True,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "total": 0,
            "processed": 0,
            "with_faces": 0,
            "faces_found": 0,
            "failed": 0,
            "current_batch": 0,
            "total_batches": 0,
            "speed_imgs_per_sec": 0.0,
            "eta_seconds": 0,
            "error": None
        }

        try:
            session = self.session_factory()
            try:
                # Get all photos - используем флаг faces_indexed вместо JOIN
                if skip_indexed:
                    # Быстрый запрос по индексу - нет JOIN!
                    photos_query = session.query(
                        PhotoIndex.image_id, PhotoIndex.file_path
                    ).filter(
                        PhotoIndex.faces_indexed == 0
                    )
                    logger.info("Пропуск уже проиндексированных фото (faces_indexed=1)")
                else:
                    photos_query = session.query(PhotoIndex.image_id, PhotoIndex.file_path)

                photos = [
                    {"image_id": row[0], "file_path": row[1]}
                    for row in photos_query.all()
                ]

            finally:
                session.close()

            self._state["total"] = len(photos)
            logger.info(f"Starting face indexing for {len(photos)} photos")

            if not photos:
                return {"total": 0, "processed": 0, "with_faces": 0, "total_faces": 0, "failed": 0}

            # Filter to existing files only
            existing_photos = [
                p for p in photos
                if os.path.exists(p["file_path"])
            ]

            if len(existing_photos) < len(photos):
                logger.warning(
                    f"Skipping {len(photos) - len(existing_photos)} photos with missing files"
                )

            # Calculate total batches
            total_batches = (len(existing_photos) + batch_size - 1) // batch_size
            self._state["total_batches"] = total_batches

            # Index in batches with progress tracking
            stats = {"total": len(photos), "processed": 0, "with_faces": 0, "total_faces": 0, "failed": 0}
            
            for batch_idx in range(0, len(existing_photos), batch_size):
                batch = existing_photos[batch_idx:batch_idx + batch_size]
                self._state["current_batch"] = (batch_idx // batch_size) + 1
                
                # Process batch
                batch_stats = self.index_batch(batch, batch_size=batch_size)
                
                # Update cumulative stats
                stats["processed"] += batch_stats["processed"]
                stats["with_faces"] += batch_stats["with_faces"]
                stats["total_faces"] += batch_stats["total_faces"]
                stats["failed"] += batch_stats["failed"]
                
                # Update state
                self._state["processed"] = stats["processed"]
                self._state["with_faces"] = stats["with_faces"]
                self._state["faces_found"] = stats["total_faces"]
                self._state["failed"] = stats["failed"]
                
                # Calculate speed and ETA
                elapsed = time.time() - start_time
                if elapsed > 0:
                    self._state["speed_imgs_per_sec"] = round(stats["processed"] / elapsed, 2)
                    remaining = len(existing_photos) - stats["processed"]
                    if self._state["speed_imgs_per_sec"] > 0:
                        self._state["eta_seconds"] = int(remaining / self._state["speed_imgs_per_sec"])

            return stats

        except Exception as e:
            self._state["error"] = str(e)
            logger.error(f"Face indexing failed: {e}")
            raise
        finally:
            self._state["running"] = False
            self._state["finished_at"] = datetime.now().isoformat()

    def get_indexing_status(self) -> Dict:
        """
        Get face indexing status.

        Returns:
            {running, total, processed, faces_found, percentage, started_at, finished_at, error}
        """
        status = dict(self._state)

        # Add percentage
        if status["total"] > 0:
            status["percentage"] = round(status["processed"] / status["total"] * 100, 1)
        else:
            status["percentage"] = 0
        
        # Format ETA
        if status["eta_seconds"] > 0:
            eta_mins = status["eta_seconds"] // 60
            eta_secs = status["eta_seconds"] % 60
            status["eta_formatted"] = f"{eta_mins}m {eta_secs}s"
        else:
            status["eta_formatted"] = "N/A"

        # Add database stats
        session = self.session_factory()
        try:
            status["total_faces_in_db"] = self.face_repository.get_face_count(session)
            status["unassigned_faces"] = self.face_repository.get_unassigned_count(session)
        finally:
            session.close()

        return status

    def get_faces_for_photo(self, image_id: int) -> List[Dict]:
        """
        Get all faces for a photo.

        Args:
            image_id: Photo image ID

        Returns:
            List of face info dicts
        """
        session = self.session_factory()
        try:
            faces = self.face_repository.get_faces_by_image(session, image_id)
            result = []
            for face in faces:
                # Get person name if assigned
                person_name = None
                if face.person_id:
                    person = session.query(Person).filter(Person.person_id == face.person_id).first()
                    if person:
                        person_name = person.name

                result.append({
                    "face_id": face.face_id,
                    "bbox": [face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2],
                    "det_score": face.det_score,
                    "age": face.age,
                    "gender": face.gender,
                    "person_id": face.person_id,
                    "person_name": person_name
                })
            return result
        finally:
            session.close()

    def search_by_face(
        self,
        embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Search for photos with similar faces.

        Args:
            embedding: 512-dim face embedding
            top_k: Maximum number of results
            threshold: Minimum similarity (0-1)

        Returns:
            List of search results
        """
        session = self.session_factory()
        try:
            return self.face_repository.search_similar_faces(
                session, embedding, top_k, threshold
            )
        finally:
            session.close()
