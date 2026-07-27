"""Google Photos export service.

Uploads photos from a local album into the *user's own* Google Photos library
and collects them into a Google album.

Scope: `photoslibrary.appendonly` — write-only. Google removed every
library-read scope in March 2025, so an app can only touch data it created
itself; that is exactly what export needs (and why the created album cannot
be shared or read back through the API).

Tokens are per user: the refresh token lives in `app_user.google_refresh_token`,
access tokens (1h) are kept in process memory only.
"""

import base64
import io
import json
import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import httpx
from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session

from config.settings import settings

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
PHOTOS_API = "https://photoslibrary.googleapis.com/v1"

# appendonly = upload + create albums. openid/email only to show which account is linked.
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/photoslibrary.appendonly",
    "openid",
    "email",
]

# Google caps batchCreate at 50 media items per call
BATCH_CREATE_LIMIT = 50

# Parallel byte uploads. Google asks for sequential batchCreate, but the upload
# step is plain HTTP POSTs and benefits from a little concurrency.
UPLOAD_WORKERS = 4

# RAW is converted to JPEG: a NEF is ~25 MB and would eat the user's Google
# quota for no visible gain. Conversion drops the original EXIF, so capture date
# and GPS are written back from our DB (see _build_exif_bytes).
RAW_FORMATS = {"nef", "cr2", "arw", "dng", "raf", "orf", "rw2"}

# Everything here is uploaded byte-for-byte: EXIF (capture date, GPS) stays
# intact, and Google Photos dates items from EXIF. A re-encode without it would
# file every photo under the upload date instead.
NATIVE_UPLOAD_FORMATS = {
    "jpg", "jpeg", "png", "heic", "heif", "webp", "gif", "bmp", "tiff",
}


class GoogleAuthRequired(Exception):
    """Raised when the user has no usable Google token.

    Either they never linked an account, or the refresh token died (Google
    expires refresh tokens after 7 days while the OAuth app is in "Testing").
    Callers should answer with an auth URL so the UI can re-run consent.
    """


class GooglePhotosService:
    """Per-user export of local albums into Google Photos."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory
        # user_id -> (access_token, expires_at_epoch)
        self._token_cache: Dict[int, tuple] = {}
        self._token_lock = threading.Lock()
        # user_id -> progress dict (several users may export at the same time,
        # so this cannot be a single module-level state like the indexers use)
        self._states: Dict[int, dict] = {}
        self._stop_flags: Dict[int, bool] = {}

    # ------------------------------------------------------------------ config

    @property
    def is_configured(self) -> bool:
        return bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET)

    def _require_config(self):
        if not self.is_configured:
            raise RuntimeError(
                "Google OAuth is not configured: set GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET"
            )

    # ------------------------------------------------------------------- OAuth

    def build_auth_url(self, state: str) -> str:
        """Consent URL for the authorization-code flow.

        access_type=offline + prompt=consent are both required: without them a
        repeat authorization returns only an access token and no refresh token.
        """
        self._require_config()
        from urllib.parse import urlencode

        params = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "redirect_uri": settings.GOOGLE_OAUTH_REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(OAUTH_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
            "state": state,
        }
        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    def exchange_code(self, code: str) -> dict:
        """Exchange the authorization code for tokens. Returns {refresh_token, email}."""
        self._require_config()
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "redirect_uri": settings.GOOGLE_OAUTH_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
        if resp.status_code != 200:
            logger.error(f"Google code exchange failed: {resp.status_code} {resp.text[:300]}")
            raise RuntimeError(f"Google token exchange failed: {resp.text[:200]}")

        data = resp.json()
        refresh_token = data.get("refresh_token")
        if not refresh_token:
            # Happens when the user already granted access and prompt=consent was lost
            raise RuntimeError(
                "Google did not return a refresh token. Revoke access in the Google "
                "account settings and retry, or make sure prompt=consent is sent."
            )

        return {
            "refresh_token": refresh_token,
            "access_token": data.get("access_token"),
            "expires_in": data.get("expires_in", 3600),
            "email": self._email_from_id_token(data.get("id_token")),
        }

    @staticmethod
    def _email_from_id_token(id_token: Optional[str]) -> Optional[str]:
        """Read the email claim out of the id_token.

        No signature check needed: the token came straight from Google over TLS
        in a server-to-server call, and it is only used for display.
        """
        if not id_token:
            return None
        try:
            payload = id_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)  # restore base64 padding
            return json.loads(base64.urlsafe_b64decode(payload)).get("email")
        except Exception as e:
            logger.warning(f"Could not decode id_token: {e}")
            return None

    # ------------------------------------------------------------ token storage

    def save_user_token(self, user_id: int, refresh_token: str, email: Optional[str]):
        session = self.session_factory()
        try:
            session.execute(
                text(
                    "UPDATE app_user SET google_refresh_token = :tok, google_email = :em, "
                    "google_connected_at = NOW() WHERE user_id = :uid"
                ),
                {"tok": refresh_token, "em": email, "uid": user_id},
            )
            session.commit()
            logger.info(f"Google account linked for user_id={user_id} ({email})")
        finally:
            session.close()

    def disconnect(self, user_id: int):
        with self._token_lock:
            self._token_cache.pop(user_id, None)
        session = self.session_factory()
        try:
            session.execute(
                text(
                    "UPDATE app_user SET google_refresh_token = NULL, google_email = NULL, "
                    "google_connected_at = NULL WHERE user_id = :uid"
                ),
                {"uid": user_id},
            )
            session.commit()
            logger.info(f"Google account unlinked for user_id={user_id}")
        finally:
            session.close()

    def get_status(self, user_id: int) -> dict:
        session = self.session_factory()
        try:
            row = session.execute(
                text(
                    "SELECT google_email, google_connected_at, "
                    "       (google_refresh_token IS NOT NULL) AS connected "
                    "FROM app_user WHERE user_id = :uid"
                ),
                {"uid": user_id},
            ).fetchone()
            if not row:
                return {"connected": False, "email": None, "connected_at": None}
            return {
                "connected": bool(row.connected),
                "email": row.google_email,
                "connected_at": row.google_connected_at.isoformat() if row.google_connected_at else None,
            }
        finally:
            session.close()

    def _get_refresh_token(self, user_id: int) -> str:
        session = self.session_factory()
        try:
            row = session.execute(
                text("SELECT google_refresh_token FROM app_user WHERE user_id = :uid"),
                {"uid": user_id},
            ).fetchone()
        finally:
            session.close()
        if not row or not row[0]:
            raise GoogleAuthRequired("No Google account linked")
        return row[0]

    def _get_access_token(self, user_id: int) -> str:
        """Return a valid access token, refreshing it when needed."""
        self._require_config()

        with self._token_lock:
            cached = self._token_cache.get(user_id)
            if cached and cached[1] > time.time() + 60:  # 60s safety margin
                return cached[0]

        refresh_token = self._get_refresh_token(user_id)

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "refresh_token": refresh_token,
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "grant_type": "refresh_token",
                },
            )

        if resp.status_code != 200:
            err = ""
            try:
                err = resp.json().get("error", "")
            except Exception:
                pass
            # invalid_grant = refresh token revoked or expired (7-day limit while
            # the OAuth app is in "Testing"). Drop the dead token so the UI can
            # send the user through consent again instead of failing forever.
            if err == "invalid_grant":
                logger.warning(f"Google refresh token dead for user_id={user_id}, clearing it")
                self.disconnect(user_id)
                raise GoogleAuthRequired("Google token expired, re-authorization required")
            logger.error(f"Google token refresh failed: {resp.status_code} {resp.text[:300]}")
            raise RuntimeError(f"Google token refresh failed: {resp.text[:200]}")

        data = resp.json()
        token = data["access_token"]
        expires_at = time.time() + int(data.get("expires_in", 3600))
        with self._token_lock:
            self._token_cache[user_id] = (token, expires_at)
        return token

    # ------------------------------------------------------------- Photos calls

    def _create_google_album(self, user_id: int, title: str) -> dict:
        token = self._get_access_token(user_id)
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{PHOTOS_API}/albums",
                headers={"Authorization": f"Bearer {token}"},
                json={"album": {"title": title}},
            )
        if resp.status_code != 200:
            logger.error(f"Album create failed: {resp.status_code} {resp.text[:300]}")
            raise RuntimeError(f"Could not create Google album: {resp.text[:200]}")
        data = resp.json()
        logger.info(f"Created Google album '{title}' id={data.get('id')}")
        return data

    def _upload_bytes(self, token: str, data: bytes, file_name: str, mime: str) -> Optional[str]:
        """Upload raw bytes, return an upload token (not a media item yet)."""
        try:
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(
                    f"{PHOTOS_API}/uploads",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-type": "application/octet-stream",
                        "X-Goog-Upload-Content-Type": mime,
                        "X-Goog-Upload-Protocol": "raw",
                        "X-Goog-Upload-File-Name": file_name,
                    },
                    content=data,
                )
            if resp.status_code != 200:
                logger.warning(f"Upload failed for {file_name}: {resp.status_code} {resp.text[:200]}")
                return None
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"Upload error for {file_name}: {e}")
            return None

    def _batch_create(self, user_id: int, google_album_id: str, items: List[dict]) -> List[dict]:
        """Turn upload tokens into media items inside the album (max 50 per call)."""
        token = self._get_access_token(user_id)
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{PHOTOS_API}/mediaItems:batchCreate",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "albumId": google_album_id,
                    "newMediaItems": [
                        {
                            "description": it.get("description", ""),
                            "simpleMediaItem": {
                                "uploadToken": it["upload_token"],
                                "fileName": it["file_name"],
                            },
                        }
                        for it in items
                    ],
                },
            )
        if resp.status_code != 200:
            logger.error(f"batchCreate failed: {resp.status_code} {resp.text[:300]}")
            raise RuntimeError(f"Google batchCreate failed: {resp.text[:200]}")
        return resp.json().get("newMediaItemResults", [])

    # ------------------------------------------------------------------- export

    @staticmethod
    def _valid_coords(photo: dict) -> Optional[tuple]:
        """Usable (lat, lon) or None.

        Zero is "no GPS" here, not a real position: unset coordinates land in
        the DB as 0 rather than NULL, and the rest of the project already treats
        them that way (see the /geo/stats filter). Injecting them would tag the
        photo with 0°,0° in the Gulf of Guinea.
        """
        latitude, longitude = photo.get("latitude"), photo.get("longitude")
        if latitude is None or longitude is None:
            return None
        if latitude == 0 or longitude == 0:
            return None
        return latitude, longitude

    @staticmethod
    def _to_dms(value: float):
        """Decimal degrees -> EXIF (degrees, minutes, seconds) rationals."""
        from PIL.TiffImagePlugin import IFDRational

        value = abs(float(value))
        degrees = int(value)
        minutes = int((value - degrees) * 60)
        seconds = (value - degrees - minutes / 60) * 3600
        return (
            IFDRational(degrees, 1),
            IFDRational(minutes, 1),
            IFDRational(round(seconds * 10000), 10000),
        )

    @classmethod
    def _build_exif_bytes(cls, photo: dict) -> Optional[bytes]:
        """Rebuild the EXIF a converted RAW needs.

        Sources, in order of reliability:
      * capture date — `exif_data.DateTimeOriginal` (already stored in exact
        EXIF form), falling back to the parsed `photo_index.photo_date`;
      * GPS — the `latitude`/`longitude` columns. They are a strict superset of
        `exif_data.GPSLatitude`: 93.4k photos have the columns filled vs 48.3k
        with GPS inside the blob, no photo has the blob without the columns,
        and 44k got their coordinates assigned manually (map drag / geo picker)
        where the blob was never updated. Reading the blob would drop those.
      * camera — `Make`/`Model`/lens strings, copied verbatim.

        The rest of `exif_data` holds exifread's human-readable text
        ("Flash did not fire...", "Horizontal (normal)"), which cannot be
        turned back into binary EXIF values reliably, so it is left out.

        Without DateTimeOriginal Google Photos files the item under its upload
        date, which would scatter converted RAWs across the user's timeline.
        """
        from PIL import Image

        try:
            exif_src = photo.get("exif") or {}
            exif = Image.Exif()
            has_data = False

            stamp = exif_src.get("DateTimeOriginal") or exif_src.get("DateTime")
            if not stamp and photo.get("photo_date"):
                photo_date = photo["photo_date"]
                if isinstance(photo_date, str):
                    photo_date = datetime.fromisoformat(photo_date)
                stamp = photo_date.strftime("%Y:%m:%d %H:%M:%S")
            if stamp:
                exif[0x0132] = stamp  # IFD0 DateTime
                sub = exif.get_ifd(0x8769)  # Exif SubIFD
                sub[0x9003] = stamp  # DateTimeOriginal
                sub[0x9004] = stamp  # DateTimeDigitized
                has_data = True

            for tag, key in ((0x010F, "Make"), (0x0110, "Model"), (0x0131, "Software")):
                value = exif_src.get(key)
                if value:
                    exif[tag] = str(value)
                    has_data = True

            for tag, key in ((0xA433, "LensMake"), (0xA434, "LensModel")):
                value = exif_src.get(key)
                if value:
                    exif.get_ifd(0x8769)[tag] = str(value)
                    has_data = True

            coords = cls._valid_coords(photo)
            if coords:
                latitude, longitude = coords
                gps = exif.get_ifd(0x8825)
                gps[1] = "N" if latitude >= 0 else "S"
                gps[2] = cls._to_dms(latitude)
                gps[3] = "E" if longitude >= 0 else "W"
                gps[4] = cls._to_dms(longitude)
                has_data = True

            return exif.tobytes() if has_data else None
        except Exception as e:
            logger.warning(f"Could not build EXIF: {e}")
            return None

    @staticmethod
    def _to_dms_piexif(value: float):
        """Decimal degrees -> piexif rational triple."""
        value = abs(float(value))
        degrees = int(value)
        minutes = int((value - degrees) * 60)
        seconds = (value - degrees - minutes / 60) * 3600
        return ((degrees, 1), (minutes, 1), (round(seconds * 10000), 10000))

    @classmethod
    def _inject_missing_exif(cls, jpeg_bytes: bytes, photo: dict) -> bytes:
        """Fill in EXIF the DB knows but the file does not (JPEG only).

        GPS assigned through the map / geo picker is written to the
        latitude/longitude columns only — `/geo/assign` never rewrites the file.
        Uploading such a JPEG untouched would land it in Google Photos without a
        location, so the coordinates are injected here.

        piexif rewrites only the APP1 segment, so pixel data is untouched and
        there is no re-encoding loss. Only *missing* values are filled — an
        existing EXIF value from the camera always wins.
        """
        try:
            import piexif

            exif_dict = piexif.load(jpeg_bytes)
            changed = False

            gps = exif_dict.get("GPS") or {}
            coords = cls._valid_coords(photo)
            if coords and piexif.GPSIFD.GPSLatitude not in gps:
                latitude, longitude = coords
                gps[piexif.GPSIFD.GPSLatitudeRef] = "N" if latitude >= 0 else "S"
                gps[piexif.GPSIFD.GPSLatitude] = cls._to_dms_piexif(latitude)
                gps[piexif.GPSIFD.GPSLongitudeRef] = "E" if longitude >= 0 else "W"
                gps[piexif.GPSIFD.GPSLongitude] = cls._to_dms_piexif(longitude)
                exif_dict["GPS"] = gps
                changed = True

            exif_ifd = exif_dict.get("Exif") or {}
            if piexif.ExifIFD.DateTimeOriginal not in exif_ifd and photo.get("photo_date"):
                photo_date = photo["photo_date"]
                if isinstance(photo_date, str):
                    photo_date = datetime.fromisoformat(photo_date)
                stamp = photo_date.strftime("%Y:%m:%d %H:%M:%S")
                exif_ifd[piexif.ExifIFD.DateTimeOriginal] = stamp
                exif_ifd[piexif.ExifIFD.DateTimeDigitized] = stamp
                exif_dict["Exif"] = exif_ifd
                changed = True

            if not changed:
                return jpeg_bytes

            # Thumbnails occasionally fail to re-dump; they are not worth a failure
            exif_dict.pop("thumbnail", None)
            exif_dict["1st"] = {}

            # piexif.insert() writes to its third argument when handed raw bytes;
            # without it, it raises instead of returning the new image.
            out = io.BytesIO()
            piexif.insert(piexif.dump(exif_dict), jpeg_bytes, out)
            return out.getvalue()
        except Exception as e:
            # Never fail an upload over metadata — ship the original bytes
            logger.warning(f"Could not inject EXIF into {photo.get('file_path')}: {e}")
            return jpeg_bytes

    @classmethod
    def _prepare_upload(cls, photo: dict) -> Optional[tuple]:
        """Return (bytes, mime, file_name) ready for upload."""
        file_path = photo["file_path"]
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File missing, skipping: {file_path}")
            return None

        fmt = (photo.get("file_format") or path.suffix.lstrip(".")).lower()

        # Non-RAW: ship the file as-is — its own EXIF is authoritative and our
        # rotation stays an overlay, not baked in. JPEG additionally gets any
        # GPS/date the DB has but the file lacks (see _inject_missing_exif).
        if fmt not in RAW_FORMATS:
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if fmt in ("heic", "heif"):
                mime = "image/heic"
            data = path.read_bytes()
            if fmt in ("jpg", "jpeg"):
                data = cls._inject_missing_exif(data, photo)
            return data, mime, path.name

        # RAW: decode, apply our rotation, re-encode as JPEG, restore EXIF
        try:
            import rawpy
            from PIL import Image as PILImage

            # postprocess() already applies the camera orientation via
            # raw.sizes.flip — adding EXIF rotation on top would double-rotate.
            # Same call as services/face_embedder.py uses for RAW.
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
            img = PILImage.fromarray(rgb)

            rotation = photo.get("rotation") or 0
            if rotation:
                transpose = {
                    90: PILImage.Transpose.ROTATE_270,
                    180: PILImage.Transpose.ROTATE_180,
                    270: PILImage.Transpose.ROTATE_90,
                }.get(rotation % 360)
                if transpose is not None:
                    img = img.transpose(transpose)

            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            exif_bytes = cls._build_exif_bytes(photo)

            buf = io.BytesIO()
            if exif_bytes:
                img.save(buf, format="JPEG", quality=95, exif=exif_bytes)
            else:
                img.save(buf, format="JPEG", quality=95)

            return buf.getvalue(), "image/jpeg", path.stem + ".jpg"
        except Exception as e:
            logger.warning(f"Could not convert RAW {file_path}: {e}")
            return None

    def get_state(self, user_id: int) -> dict:
        return self._states.get(user_id) or {"running": False}

    def begin_state(self, user_id: int, album_id: int, total: int):
        """Mark the export as running *before* the request returns.

        FastAPI runs background tasks only after the response is sent, so a
        client that starts polling immediately would otherwise see
        `running: False` and conclude the export had already finished.
        """
        self._states[user_id] = {
            "running": True,
            "album_id": album_id,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "total": total,
            "processed": 0,
            "uploaded": 0,
            "skipped": 0,
            "failed": 0,
            "google_album_url": None,
            "error": None,
        }

    def request_stop(self, user_id: int):
        self._stop_flags[user_id] = True
        logger.info(f"Google export stop requested for user_id={user_id}")

    def get_album_mapping(self, user_id: int, album_id: int) -> Optional[dict]:
        session = self.session_factory()
        try:
            row = session.execute(
                text(
                    "SELECT google_album_id, google_album_url, last_export_at "
                    "FROM google_album_export WHERE album_id = :aid AND user_id = :uid"
                ),
                {"aid": album_id, "uid": user_id},
            ).fetchone()
            if not row:
                return None
            return {
                "google_album_id": row.google_album_id,
                "google_album_url": row.google_album_url,
                "last_export_at": row.last_export_at.isoformat() if row.last_export_at else None,
            }
        finally:
            session.close()

    def _ensure_google_album(self, user_id: int, album_id: int, title: str) -> dict:
        """Reuse the previously created Google album, or make a new one."""
        existing = self.get_album_mapping(user_id, album_id)
        if existing:
            return existing

        created = self._create_google_album(user_id, title)
        session = self.session_factory()
        try:
            session.execute(
                text(
                    "INSERT INTO google_album_export (album_id, user_id, google_album_id, google_album_url) "
                    "VALUES (:aid, :uid, :gid, :url) "
                    "ON CONFLICT (album_id, user_id) DO UPDATE "
                    "SET google_album_id = EXCLUDED.google_album_id, google_album_url = EXCLUDED.google_album_url"
                ),
                {
                    "aid": album_id,
                    "uid": user_id,
                    "gid": created.get("id"),
                    "url": created.get("productUrl"),
                },
            )
            session.commit()
        finally:
            session.close()
        return {
            "google_album_id": created.get("id"),
            "google_album_url": created.get("productUrl"),
            "last_export_at": None,
        }

    def _load_album_photos(self, album_id: int) -> List[dict]:
        """Everything the export needs, straight from photo_index.

        Single source of truth: file location, capture date and GPS (used to
        rebuild EXIF for converted RAW) and our non-destructive rotation.
        """
        session = self.session_factory()
        try:
            rows = session.execute(
                text(
                    "SELECT p.image_id, p.file_path, p.file_format, p.photo_date, "
                    "       p.latitude, p.longitude, p.exif_data "
                    "FROM album_photo ap "
                    "JOIN photo_index p ON p.image_id = ap.image_id "
                    "WHERE ap.album_id = :aid "
                    "ORDER BY ap.sort_order, p.photo_date NULLS LAST, p.image_id"
                ),
                {"aid": album_id},
            ).fetchall()
        finally:
            session.close()

        photos = []
        for r in rows:
            exif = r.exif_data if isinstance(r.exif_data, dict) else {}
            photos.append({
                "image_id": r.image_id,
                "file_path": r.file_path,
                "file_format": r.file_format,
                "photo_date": r.photo_date,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "rotation": exif.get("UserRotation", 0),
                "exif": exif,  # source for rebuilding EXIF on RAW conversion
            })
        return photos

    def count_album_photos(self, album_id: int) -> int:
        session = self.session_factory()
        try:
            return session.execute(
                text("SELECT COUNT(*) FROM album_photo WHERE album_id = :aid"),
                {"aid": album_id},
            ).scalar() or 0
        finally:
            session.close()

    def _already_exported(self, album_id: int, user_id: int) -> set:
        session = self.session_factory()
        try:
            rows = session.execute(
                text(
                    "SELECT image_id FROM google_export_item "
                    "WHERE album_id = :aid AND user_id = :uid AND status = 'ok'"
                ),
                {"aid": album_id, "uid": user_id},
            ).fetchall()
            return {r[0] for r in rows}
        finally:
            session.close()

    def _record_items(self, album_id: int, user_id: int, records: List[dict]):
        if not records:
            return
        session = self.session_factory()
        try:
            for rec in records:
                session.execute(
                    text(
                        "INSERT INTO google_export_item "
                        "  (album_id, user_id, image_id, google_media_id, status, error, exported_at) "
                        "VALUES (:aid, :uid, :iid, :gid, :st, :err, NOW()) "
                        "ON CONFLICT (album_id, user_id, image_id) DO UPDATE "
                        "SET google_media_id = EXCLUDED.google_media_id, status = EXCLUDED.status, "
                        "    error = EXCLUDED.error, exported_at = NOW()"
                    ),
                    {
                        "aid": album_id,
                        "uid": user_id,
                        "iid": rec["image_id"],
                        "gid": rec.get("google_media_id"),
                        "st": rec.get("status", "ok"),
                        "err": rec.get("error"),
                    },
                )
            session.commit()
        finally:
            session.close()

    def export_album(
        self,
        user_id: int,
        album_id: int,
        album_title: str,
        on_progress: Optional[Callable] = None,
    ) -> dict:
        """Export an album into the user's Google Photos.

        Photo data is read from photo_index directly. Already-exported photos
        are skipped, so a repeated call resumes instead of duplicating.
        """
        self._stop_flags[user_id] = False
        photos = self._load_album_photos(album_id)
        # begin_state() may have seeded this already; keep its started_at
        started_at = (self._states.get(user_id) or {}).get("started_at") or datetime.now().isoformat()
        state = {
            "running": True,
            "album_id": album_id,
            "started_at": started_at,
            "finished_at": None,
            "total": len(photos),
            "processed": 0,
            "uploaded": 0,
            "skipped": 0,
            "failed": 0,
            "google_album_url": None,
            "error": None,
        }
        self._states[user_id] = state

        try:
            mapping = self._ensure_google_album(user_id, album_id, album_title)
            state["google_album_url"] = mapping.get("google_album_url")
            google_album_id = mapping["google_album_id"]

            done = self._already_exported(album_id, user_id)
            pending = [p for p in photos if p["image_id"] not in done]
            state["skipped"] = len(photos) - len(pending)
            state["processed"] = state["skipped"]

            for start in range(0, len(pending), BATCH_CREATE_LIMIT):
                if self._stop_flags.get(user_id):
                    logger.info(f"Google export stopped by request (user_id={user_id})")
                    break

                chunk = pending[start:start + BATCH_CREATE_LIMIT]
                token = self._get_access_token(user_id)

                # Step 1: prepare + upload bytes in parallel -> upload tokens
                def _prepare(photo):
                    prepared = self._prepare_upload(photo)
                    if not prepared:
                        return photo, None, None
                    data, mime, file_name = prepared
                    up_token = self._upload_bytes(token, data, file_name, mime)
                    return photo, up_token, file_name

                with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
                    uploaded = list(pool.map(_prepare, chunk))

                items, failures = [], []
                for photo, up_token, file_name in uploaded:
                    if up_token:
                        items.append({
                            "image_id": photo["image_id"],
                            "upload_token": up_token,
                            "file_name": file_name,
                        })
                    else:
                        failures.append({
                            "image_id": photo["image_id"],
                            "status": "failed",
                            "error": "prepare/upload failed",
                        })

                # Step 2: create media items inside the album (sequential per Google's guidance)
                records = list(failures)
                if items:
                    results = self._batch_create(user_id, google_album_id, items)
                    for item, result in zip(items, results):
                        status = result.get("status", {})
                        media = result.get("mediaItem") or {}
                        ok = media.get("id") is not None
                        records.append({
                            "image_id": item["image_id"],
                            "google_media_id": media.get("id"),
                            "status": "ok" if ok else "failed",
                            "error": None if ok else status.get("message", "batchCreate failed"),
                        })

                self._record_items(album_id, user_id, records)

                state["uploaded"] += sum(1 for r in records if r["status"] == "ok")
                state["failed"] += sum(1 for r in records if r["status"] == "failed")
                state["processed"] += len(chunk)
                if on_progress:
                    on_progress(state)

            session = self.session_factory()
            try:
                session.execute(
                    text(
                        "UPDATE google_album_export SET last_export_at = NOW() "
                        "WHERE album_id = :aid AND user_id = :uid"
                    ),
                    {"aid": album_id, "uid": user_id},
                )
                session.commit()
            finally:
                session.close()

            logger.info(
                f"Google export done: album={album_id} user={user_id} "
                f"uploaded={state['uploaded']} skipped={state['skipped']} failed={state['failed']}"
            )

        except GoogleAuthRequired as e:
            state["error"] = "auth_required"
            logger.warning(f"Google export needs re-auth (user_id={user_id}): {e}")
        except Exception as e:
            state["error"] = str(e)
            logger.error(f"Google export failed (user_id={user_id}): {e}", exc_info=True)
        finally:
            state["running"] = False
            state["finished_at"] = datetime.now().isoformat()
            self._stop_flags.pop(user_id, None)

        return state
