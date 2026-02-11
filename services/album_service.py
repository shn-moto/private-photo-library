"""Album management service for photo collections"""

from typing import List, Dict, Optional, Callable, Tuple
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import func, text
from loguru import logger

from models.data_models import Album, AlbumPhoto, AppUser, PhotoIndex


class AlbumRepository:
    """Repository for album database operations"""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def create_album(
        self,
        session: Session,
        user_id: int,
        title: str,
        description: Optional[str] = None,
        is_public: bool = False
    ) -> int:
        """Create a new album. Returns album_id."""
        album = Album(
            user_id=user_id,
            title=title,
            description=description,
            is_public=is_public
        )
        session.add(album)
        session.flush()
        return album.album_id

    def get_album(self, session: Session, album_id: int) -> Optional[Album]:
        """Get album by ID."""
        return session.query(Album).filter(Album.album_id == album_id).first()

    def update_album(
        self,
        session: Session,
        album_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        cover_image_id: Optional[int] = None,
        is_public: Optional[bool] = None
    ) -> bool:
        """Update album. Returns True if album was found."""
        album = session.query(Album).filter(Album.album_id == album_id).first()
        if not album:
            return False

        if title is not None:
            album.title = title
        if description is not None:
            album.description = description
        if cover_image_id is not None:
            album.cover_image_id = cover_image_id
        if is_public is not None:
            album.is_public = is_public

        album.updated_at = datetime.now()
        return True

    def delete_album(self, session: Session, album_id: int) -> bool:
        """Delete album. Album photos are cascade deleted. Returns True if found."""
        album = session.query(Album).filter(Album.album_id == album_id).first()
        if not album:
            return False
        session.delete(album)
        return True

    def list_albums(
        self,
        session: Session,
        user_id: Optional[int] = None,
        include_public: bool = True,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List albums with photo counts."""
        query = session.query(
            Album.album_id,
            Album.title,
            Album.description,
            Album.cover_image_id,
            Album.is_public,
            Album.user_id,
            Album.created_at,
            Album.updated_at,
            func.count(AlbumPhoto.image_id).label("photo_count")
        ).outerjoin(AlbumPhoto, AlbumPhoto.album_id == Album.album_id).group_by(
            Album.album_id,
            Album.title,
            Album.description,
            Album.cover_image_id,
            Album.is_public,
            Album.user_id,
            Album.created_at,
            Album.updated_at
        )

        # Filter by user
        if user_id is not None:
            if include_public:
                query = query.filter(
                    (Album.user_id == user_id) | (Album.is_public == True)
                )
            else:
                query = query.filter(Album.user_id == user_id)

        if search:
            query = query.filter(func.lower(Album.title).contains(search.lower()))

        query = query.order_by(Album.updated_at.desc()).offset(offset).limit(limit)
        results = query.all()

        return [
            {
                "album_id": row[0],
                "title": row[1],
                "description": row[2],
                "cover_image_id": row[3],
                "is_public": row[4],
                "user_id": row[5],
                "created_at": row[6].isoformat() if row[6] else None,
                "updated_at": row[7].isoformat() if row[7] else None,
                "photo_count": row[8]
            }
            for row in results
        ]

    def add_photos(
        self,
        session: Session,
        album_id: int,
        image_ids: List[int]
    ) -> Dict[str, int]:
        """Add photos to album. Returns {added, already_in}."""
        # Get existing photo IDs in album
        existing = set(
            row[0] for row in
            session.query(AlbumPhoto.image_id).filter(
                AlbumPhoto.album_id == album_id
            ).all()
        )

        # Get max sort_order
        max_order = session.query(func.max(AlbumPhoto.sort_order)).filter(
            AlbumPhoto.album_id == album_id
        ).scalar() or 0

        added = 0
        already_in = 0
        for image_id in image_ids:
            if image_id in existing:
                already_in += 1
                continue
            max_order += 1
            ap = AlbumPhoto(
                album_id=album_id,
                image_id=image_id,
                sort_order=max_order
            )
            session.add(ap)
            added += 1

        # Auto-set cover if album has no cover
        if added > 0:
            album = session.query(Album).filter(Album.album_id == album_id).first()
            if album and album.cover_image_id is None:
                # Pick the first added photo as cover
                album.cover_image_id = image_ids[0]
                album.updated_at = datetime.now()

        return {"added": added, "already_in": already_in}

    def remove_photos(
        self,
        session: Session,
        album_id: int,
        image_ids: List[int]
    ) -> int:
        """Remove photos from album. Returns removed count."""
        removed = session.query(AlbumPhoto).filter(
            AlbumPhoto.album_id == album_id,
            AlbumPhoto.image_id.in_(image_ids)
        ).delete(synchronize_session='fetch')

        # If cover photo was removed, auto-pick new cover
        album = session.query(Album).filter(Album.album_id == album_id).first()
        if album and album.cover_image_id in image_ids:
            # Pick most recent remaining photo
            newest = session.query(AlbumPhoto.image_id).filter(
                AlbumPhoto.album_id == album_id
            ).order_by(AlbumPhoto.added_at.desc()).first()
            album.cover_image_id = newest[0] if newest else None
            album.updated_at = datetime.now()

        return removed

    def get_album_photos(
        self,
        session: Session,
        album_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict], int]:
        """Get photos in album with pagination. Returns (photos, total)."""
        total = session.query(func.count(AlbumPhoto.image_id)).filter(
            AlbumPhoto.album_id == album_id
        ).scalar() or 0

        query = session.query(
            PhotoIndex.image_id,
            PhotoIndex.file_path,
            PhotoIndex.file_format,
            PhotoIndex.photo_date,
            PhotoIndex.latitude,
            PhotoIndex.longitude,
            AlbumPhoto.sort_order,
            AlbumPhoto.added_at
        ).join(AlbumPhoto, AlbumPhoto.image_id == PhotoIndex.image_id).filter(
            AlbumPhoto.album_id == album_id
        ).order_by(AlbumPhoto.sort_order).offset(offset).limit(limit)

        photos = [
            {
                "image_id": row[0],
                "file_path": row[1],
                "file_format": row[2],
                "photo_date": row[3].isoformat() if row[3] else None,
                "latitude": row[4],
                "longitude": row[5],
                "sort_order": row[6],
                "added_at": row[7].isoformat() if row[7] else None
            }
            for row in query.all()
        ]

        return photos, total

    def get_photo_albums(
        self,
        session: Session,
        image_id: int
    ) -> List[Dict]:
        """Get all albums containing this photo."""
        query = session.query(
            Album.album_id,
            Album.title,
            Album.cover_image_id,
            Album.is_public
        ).join(AlbumPhoto, AlbumPhoto.album_id == Album.album_id).filter(
            AlbumPhoto.image_id == image_id
        ).order_by(Album.title)

        return [
            {
                "album_id": row[0],
                "title": row[1],
                "cover_image_id": row[2],
                "is_public": row[3]
            }
            for row in query.all()
        ]

    def set_cover(self, session: Session, album_id: int, image_id: int) -> bool:
        """Set album cover photo. Returns True if album found and photo is in album."""
        album = session.query(Album).filter(Album.album_id == album_id).first()
        if not album:
            return False

        # Verify photo is in album
        in_album = session.query(AlbumPhoto).filter(
            AlbumPhoto.album_id == album_id,
            AlbumPhoto.image_id == image_id
        ).first()
        if not in_album:
            return False

        album.cover_image_id = image_id
        album.updated_at = datetime.now()
        return True


class AlbumService:
    """Service for managing photo albums"""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory
        self.repository = AlbumRepository(session_factory)

    def create_album(
        self,
        user_id: int = 1,
        title: str = "New Album",
        description: Optional[str] = None,
        is_public: bool = False
    ) -> int:
        """Create a new album. Returns album_id."""
        session = self.session_factory()
        try:
            album_id = self.repository.create_album(
                session, user_id, title, description, is_public
            )
            session.commit()
            logger.info(f"Created album '{title}' (id={album_id}) for user {user_id}")
            return album_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create album: {e}")
            raise
        finally:
            session.close()

    def get_album(self, album_id: int) -> Optional[Dict]:
        """Get album details with photo count."""
        session = self.session_factory()
        try:
            album = self.repository.get_album(session, album_id)
            if not album:
                return None

            photo_count = session.query(func.count(AlbumPhoto.image_id)).filter(
                AlbumPhoto.album_id == album_id
            ).scalar() or 0

            return {
                "album_id": album.album_id,
                "title": album.title,
                "description": album.description,
                "cover_image_id": album.cover_image_id,
                "is_public": album.is_public,
                "user_id": album.user_id,
                "sort_order": album.sort_order,
                "created_at": album.created_at.isoformat() if album.created_at else None,
                "updated_at": album.updated_at.isoformat() if album.updated_at else None,
                "photo_count": photo_count
            }
        finally:
            session.close()

    def update_album(
        self,
        album_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        cover_image_id: Optional[int] = None,
        is_public: Optional[bool] = None
    ) -> bool:
        """Update album details. Returns True if found and updated."""
        session = self.session_factory()
        try:
            success = self.repository.update_album(
                session, album_id, title, description, cover_image_id, is_public
            )
            if success:
                session.commit()
                logger.info(f"Updated album {album_id}")
            return success
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update album: {e}")
            raise
        finally:
            session.close()

    def delete_album(self, album_id: int) -> bool:
        """Delete album. Returns True if found and deleted."""
        session = self.session_factory()
        try:
            success = self.repository.delete_album(session, album_id)
            if success:
                session.commit()
                logger.info(f"Deleted album {album_id}")
            return success
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete album: {e}")
            raise
        finally:
            session.close()

    def list_albums(
        self,
        user_id: Optional[int] = 1,
        include_public: bool = True,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List albums with optional search."""
        session = self.session_factory()
        try:
            return self.repository.list_albums(
                session, user_id, include_public, search, limit, offset
            )
        finally:
            session.close()

    def add_photos(self, album_id: int, image_ids: List[int]) -> Dict[str, int]:
        """Add photos to album. Returns {added, already_in}."""
        session = self.session_factory()
        try:
            result = self.repository.add_photos(session, album_id, image_ids)
            session.commit()
            logger.info(
                f"Album {album_id}: added {result['added']}, "
                f"already_in {result['already_in']}"
            )
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add photos to album: {e}")
            raise
        finally:
            session.close()

    def remove_photos(self, album_id: int, image_ids: List[int]) -> int:
        """Remove photos from album. Returns removed count."""
        session = self.session_factory()
        try:
            removed = self.repository.remove_photos(session, album_id, image_ids)
            session.commit()
            logger.info(f"Album {album_id}: removed {removed} photos")
            return removed
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to remove photos from album: {e}")
            raise
        finally:
            session.close()

    def get_album_photos(
        self,
        album_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict], int]:
        """Get photos in album with pagination. Returns (photos, total)."""
        session = self.session_factory()
        try:
            return self.repository.get_album_photos(session, album_id, limit, offset)
        finally:
            session.close()

    def get_photo_albums(self, image_id: int) -> List[Dict]:
        """Get all albums containing this photo."""
        session = self.session_factory()
        try:
            return self.repository.get_photo_albums(session, image_id)
        finally:
            session.close()

    def set_cover(self, album_id: int, image_id: int) -> bool:
        """Set album cover photo. Returns True if successful."""
        session = self.session_factory()
        try:
            success = self.repository.set_cover(session, album_id, image_id)
            if success:
                session.commit()
                logger.info(f"Set cover for album {album_id}: image {image_id}")
            return success
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to set cover: {e}")
            raise
        finally:
            session.close()
