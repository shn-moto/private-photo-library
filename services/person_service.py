"""Person management service for face grouping and search"""

from typing import List, Dict, Optional, Callable
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import func, text
from loguru import logger

from models.data_models import Person, Face, PhotoIndex


class PersonRepository:
    """Repository for person database operations"""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def create_person(
        self,
        session: Session,
        name: str,
        description: Optional[str] = None,
        cover_face_id: Optional[int] = None
    ) -> int:
        """Create a new person. Returns person_id."""
        person = Person(
            name=name,
            description=description,
            cover_face_id=cover_face_id
        )
        session.add(person)
        session.flush()
        return person.person_id

    def get_person(self, session: Session, person_id: int) -> Optional[Person]:
        """Get person by ID."""
        return session.query(Person).filter(Person.person_id == person_id).first()

    def update_person(
        self,
        session: Session,
        person_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        cover_face_id: Optional[int] = None
    ) -> bool:
        """Update person. Returns True if person was found."""
        person = session.query(Person).filter(Person.person_id == person_id).first()
        if not person:
            return False

        if name is not None:
            person.name = name
        if description is not None:
            person.description = description
        if cover_face_id is not None:
            person.cover_face_id = cover_face_id

        person.updated_at = datetime.now()
        return True

    def delete_person(self, session: Session, person_id: int) -> bool:
        """
        Delete person. Faces are automatically unassigned (person_id = NULL).
        Returns True if person was found and deleted.
        """
        person = session.query(Person).filter(Person.person_id == person_id).first()
        if not person:
            return False

        session.delete(person)
        return True

    def list_persons(
        self,
        session: Session,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        List all persons with face counts.

        Args:
            session: SQLAlchemy session
            search: Optional name search (case-insensitive partial match)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of person dicts with face_count and photo_count
        """
        query = session.query(
            Person.person_id,
            Person.name,
            Person.description,
            Person.cover_face_id,
            Person.created_at,
            Person.updated_at,
            func.count(Face.face_id).label("face_count"),
            func.count(func.distinct(Face.image_id)).label("photo_count")
        ).outerjoin(Face, Face.person_id == Person.person_id).group_by(
            Person.person_id,
            Person.name,
            Person.description,
            Person.cover_face_id,
            Person.created_at,
            Person.updated_at
        )

        if search:
            query = query.filter(func.lower(Person.name).contains(search.lower()))

        query = query.order_by(Person.name).offset(offset).limit(limit)
        results = query.all()

        return [
            {
                "person_id": row[0],
                "name": row[1],
                "description": row[2],
                "cover_face_id": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "updated_at": row[5].isoformat() if row[5] else None,
                "face_count": row[6],
                "photo_count": row[7]
            }
            for row in results
        ]

    def search_by_name(self, session: Session, name: str, limit: int = 10) -> List[Dict]:
        """Search persons by name (case-insensitive partial match)."""
        return self.list_persons(session, search=name, limit=limit)

    def get_person_count(self, session: Session) -> int:
        """Get total person count."""
        return session.query(func.count(Person.person_id)).scalar() or 0


class PersonService:
    """Service for managing persons and face assignments"""

    def __init__(self, session_factory: Callable[[], Session]):
        """
        Initialize person service.

        Args:
            session_factory: Callable that returns a new SQLAlchemy session
        """
        self.session_factory = session_factory
        self.repository = PersonRepository(session_factory)

    def create_person(
        self,
        name: str,
        description: Optional[str] = None,
        initial_face_id: Optional[int] = None
    ) -> int:
        """
        Create a new person.

        Args:
            name: Person's name
            description: Optional description
            initial_face_id: Optional face to assign immediately

        Returns:
            person_id
        """
        session = self.session_factory()
        try:
            person_id = self.repository.create_person(session, name, description)

            # Assign initial face if provided
            if initial_face_id:
                face = session.query(Face).filter(Face.face_id == initial_face_id).first()
                if face:
                    face.person_id = person_id
                    # Set as cover face
                    person = session.query(Person).filter(Person.person_id == person_id).first()
                    if person:
                        person.cover_face_id = initial_face_id

            session.commit()
            logger.info(f"Created person '{name}' with ID {person_id}")
            return person_id

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create person: {e}")
            raise
        finally:
            session.close()

    def get_person(self, person_id: int) -> Optional[Dict]:
        """
        Get person details including face count and sample faces.

        Args:
            person_id: Person ID

        Returns:
            Person dict with stats or None if not found
        """
        session = self.session_factory()
        try:
            person = self.repository.get_person(session, person_id)
            if not person:
                return None

            # Get face stats
            face_count = session.query(func.count(Face.face_id)).filter(
                Face.person_id == person_id
            ).scalar() or 0

            photo_count = session.query(func.count(func.distinct(Face.image_id))).filter(
                Face.person_id == person_id
            ).scalar() or 0

            # Get sample faces (up to 5)
            sample_faces = session.query(Face).filter(
                Face.person_id == person_id
            ).order_by(Face.det_score.desc()).limit(5).all()

            return {
                "person_id": person.person_id,
                "name": person.name,
                "description": person.description,
                "cover_face_id": person.cover_face_id,
                "created_at": person.created_at.isoformat() if person.created_at else None,
                "updated_at": person.updated_at.isoformat() if person.updated_at else None,
                "face_count": face_count,
                "photo_count": photo_count,
                "sample_faces": [
                    {
                        "face_id": f.face_id,
                        "image_id": f.image_id,
                        "det_score": f.det_score
                    }
                    for f in sample_faces
                ]
            }

        finally:
            session.close()

    def list_persons(
        self,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List all persons with optional name search."""
        session = self.session_factory()
        try:
            return self.repository.list_persons(session, search, limit, offset)
        finally:
            session.close()

    def update_person(
        self,
        person_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        cover_face_id: Optional[int] = None
    ) -> bool:
        """Update person details. Returns True if found and updated."""
        session = self.session_factory()
        try:
            success = self.repository.update_person(
                session, person_id, name, description, cover_face_id
            )
            if success:
                session.commit()
                logger.info(f"Updated person {person_id}")
            return success
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update person: {e}")
            raise
        finally:
            session.close()

    def delete_person(self, person_id: int) -> bool:
        """
        Delete a person. Faces become unassigned.

        Args:
            person_id: Person ID

        Returns:
            True if person was found and deleted
        """
        session = self.session_factory()
        try:
            success = self.repository.delete_person(session, person_id)
            if success:
                session.commit()
                logger.info(f"Deleted person {person_id}")
            return success
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete person: {e}")
            raise
        finally:
            session.close()

    def merge_persons(self, source_person_id: int, target_person_id: int) -> bool:
        """
        Merge source person into target (move all faces, delete source).

        Args:
            source_person_id: Person to merge from (will be deleted)
            target_person_id: Person to merge into

        Returns:
            True if merge was successful
        """
        session = self.session_factory()
        try:
            # Check both persons exist
            source = session.query(Person).filter(Person.person_id == source_person_id).first()
            target = session.query(Person).filter(Person.person_id == target_person_id).first()

            if not source or not target:
                return False

            # Move all faces from source to target
            session.query(Face).filter(
                Face.person_id == source_person_id
            ).update({"person_id": target_person_id})

            # Delete source person
            session.delete(source)
            session.commit()

            logger.info(f"Merged person {source_person_id} into {target_person_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to merge persons: {e}")
            raise
        finally:
            session.close()

    def assign_face_to_person(self, face_id: int, person_id: int) -> bool:
        """Assign a face to a person. Returns True if successful."""
        session = self.session_factory()
        try:
            face = session.query(Face).filter(Face.face_id == face_id).first()
            person = session.query(Person).filter(Person.person_id == person_id).first()

            if not face or not person:
                return False

            face.person_id = person_id
            session.commit()

            logger.info(f"Assigned face {face_id} to person {person_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to assign face: {e}")
            raise
        finally:
            session.close()

    def unassign_face(self, face_id: int) -> bool:
        """Remove face from person (set person_id=NULL). Returns True if successful."""
        session = self.session_factory()
        try:
            face = session.query(Face).filter(Face.face_id == face_id).first()
            if not face:
                return False

            old_person_id = face.person_id
            face.person_id = None
            session.commit()

            logger.info(f"Unassigned face {face_id} from person {old_person_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to unassign face: {e}")
            raise
        finally:
            session.close()

    def auto_assign_faces(self, person_id: int, threshold: float = 0.6) -> Dict[str, int]:
        """
        Automatically assign unassigned faces to a person based on similarity.

        Args:
            person_id: Person to assign faces to
            threshold: Minimum similarity for auto-assignment

        Returns:
            {assigned: int, candidates: int}
        """
        session = self.session_factory()
        try:
            # Get person's existing faces to use as reference
            person_faces = session.query(Face).filter(Face.person_id == person_id).all()
            if not person_faces:
                return {"assigned": 0, "candidates": 0}

            # Calculate average embedding for person
            import numpy as np
            embeddings = [np.array(f.face_embedding) for f in person_faces]
            avg_embedding = np.mean(embeddings, axis=0)
            embedding_str = "[" + ",".join(map(str, avg_embedding.tolist())) + "]"

            cosine_threshold = 1.0 - threshold

            # Find similar unassigned faces
            query = text("""
                SELECT face_id, 1 - (face_embedding <=> :embedding::vector) as similarity
                FROM faces
                WHERE person_id IS NULL
                AND (face_embedding <=> :embedding::vector) <= :threshold
                ORDER BY face_embedding <=> :embedding::vector
            """)

            results = session.execute(query, {
                "embedding": embedding_str,
                "threshold": cosine_threshold
            }).fetchall()

            assigned = 0
            for row in results:
                face = session.query(Face).filter(Face.face_id == row[0]).first()
                if face:
                    face.person_id = person_id
                    assigned += 1

            session.commit()

            logger.info(
                f"Auto-assigned {assigned} faces to person {person_id} "
                f"(threshold={threshold}, candidates={len(results)})"
            )

            return {"assigned": assigned, "candidates": len(results)}

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to auto-assign faces: {e}")
            raise
        finally:
            session.close()

    def get_photos_by_person(
        self,
        person_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get all photos containing this person.

        Args:
            person_id: Person ID
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of photo dicts with face info
        """
        session = self.session_factory()
        try:
            query = session.query(
                PhotoIndex.image_id,
                PhotoIndex.file_path,
                PhotoIndex.file_format,
                PhotoIndex.photo_date,
                func.count(Face.face_id).label("face_count")
            ).join(Face, Face.image_id == PhotoIndex.image_id).filter(
                Face.person_id == person_id
            ).group_by(
                PhotoIndex.image_id,
                PhotoIndex.file_path,
                PhotoIndex.file_format,
                PhotoIndex.photo_date
            ).order_by(PhotoIndex.photo_date.desc().nullslast()).offset(offset).limit(limit)

            results = query.all()

            return [
                {
                    "image_id": row[0],
                    "file_path": row[1],
                    "file_format": row[2],
                    "photo_date": row[3].isoformat() if row[3] else None,
                    "face_count": row[4]
                }
                for row in results
            ]

        finally:
            session.close()

    def get_photo_count_by_person(self, person_id: int) -> int:
        """Get count of photos containing this person."""
        session = self.session_factory()
        try:
            return session.query(func.count(func.distinct(Face.image_id))).filter(
                Face.person_id == person_id
            ).scalar() or 0
        finally:
            session.close()

    def search_by_name(self, name: str, limit: int = 10) -> List[Dict]:
        """Search persons by name (case-insensitive partial match)."""
        return self.list_persons(search=name, limit=limit)
