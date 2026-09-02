"""Document photo preparation: crop to spec, replace background, optional gentle correction.

Pipeline: detect face on the ORIGINAL → segment person → composite on the spec
background → crop to the spec geometry → resize to the output size.

Two things learned the hard way and encoded here:

* Measurements are taken on the original and carried through the transform.
  InsightFace does NOT detect a face on the finished tight crop (the face fills
  the frame), so the output cannot be re-validated by detecting it again.
* Head height comes from the detector's bbox, calibrated against a real accepted
  output. Deriving it from "eyes sit mid-head" anthropometry overshoots by ~20%.

Correction is deliberately conservative: illumination-field flattening plus a
mild white balance, strength chosen from a measured unevenness ratio. No CLAHE,
no skin smoothing — the specs forbid retouching, and a fixed aggressive setting
produced visibly unnatural over-contrast.
"""

import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from config.document_photo_specs import get_spec

# u2net segmentation model. Not bundled: fetched once into a mounted volume.
U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
U2NET_PATH = Path("/models/u2net.onnx")

# Expensive, parameter-independent work (detection + segmentation) is cached so
# that a dialogue turn like "усиль контраст" recomputes only the cheap part.
_CACHE_TTL = 1800  # seconds


class DocumentPhotoError(Exception):
    """Something makes this photo unusable (no face, unreadable file, ...)."""


class DocumentPhotoService:
    def __init__(self, face_app=None):
        # Reuses the API's already-loaded InsightFace app when given one: a
        # second buffalo_l would cost another ~300 MB for a once-in-a-while call.
        self._sess = None
        self._face_app = face_app
        self._lock = threading.Lock()
        self._cache: Dict[str, dict] = {}

    # ------------------------------------------------------------------ models

    def _ensure_model(self):
        """Download u2net once into the mounted volume."""
        if U2NET_PATH.exists():
            return
        U2NET_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading u2net to {U2NET_PATH} ...")
        import urllib.request

        tmp = U2NET_PATH.with_suffix(".part")
        urllib.request.urlretrieve(U2NET_URL, tmp)
        tmp.rename(U2NET_PATH)
        logger.info("u2net downloaded")

    def _seg_session(self):
        if self._sess is None:
            with self._lock:
                if self._sess is None:
                    self._ensure_model()
                    import onnxruntime as ort

                    self._sess = ort.InferenceSession(
                        str(U2NET_PATH), providers=["CPUExecutionProvider"]
                    )
        return self._sess

    def _faces(self):
        """Falls back to its own FaceAnalysis when the API has not passed one."""
        if self._face_app is None:
            with self._lock:
                if self._face_app is None:
                    from insightface.app import FaceAnalysis

                    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                    app.prepare(ctx_id=-1, det_size=(640, 640))
                    self._face_app = app
        return self._face_app

    # ------------------------------------------------------------------- steps

    @staticmethod
    def _load(file_path: str) -> np.ndarray:
        """RGB array from any supported format."""
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        ext = Path(file_path).suffix.lower()
        if ext in {".nef", ".cr2", ".arw", ".dng", ".raf", ".orf", ".rw2"}:
            import rawpy

            with rawpy.imread(file_path) as raw:
                return raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
        with Image.open(file_path) as im:
            from PIL import ImageOps

            return np.array(ImageOps.exif_transpose(im).convert("RGB"))

    def _segment(self, rgb: np.ndarray) -> np.ndarray:
        """Person mask in 0..1, at the source resolution."""
        sess = self._seg_session()
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = ((x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]))
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        mask = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        # Pull near-certain values to exactly 0/1 so the background composites to
        # a pure colour rather than 254-ish, and the edge halo is reduced.
        return np.clip((mask - 0.02) / 0.96, 0, 1)

    def _analyze(self, file_path: str) -> dict:
        """Detection + segmentation. Cached: independent of user adjustments."""
        key = hashlib.sha1(file_path.encode()).hexdigest()
        hit = self._cache.get(key)
        if hit and time.time() - hit["at"] < _CACHE_TTL:
            return hit

        rgb = self._load(file_path)
        faces = self._faces().get(rgb[:, :, ::-1])  # InsightFace expects BGR
        if not faces:
            raise DocumentPhotoError(
                "На фото не найдено лицо. Для документа нужен портрет анфас."
            )
        face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        kps = face.kps
        eye_l, eye_r, nose = kps[0], kps[1], kps[2]
        eye = (eye_l + eye_r) / 2.0

        entry = {
            "at": time.time(),
            "rgb": rgb,
            "mask": self._segment(rgb),
            "bbox": (x1, y1, x2, y2),
            "eye": eye,
            "nose": nose,
            "face_count": len(faces),
            # Positive = head tilted clockwise in the image
            "tilt_deg": float(np.degrees(np.arctan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0]))),
            "det_score": float(face.det_score),
        }
        self._cache[key] = entry
        return entry

    # -------------------------------------------------------------- correction

    @staticmethod
    def _correct(rgb: np.ndarray, bbox, brightness: float, contrast: float,
                 white_balance: float, light: float) -> np.ndarray:
        """Adjustments allowed on a document photo.

        `light` flattens the illumination field: the low-frequency luminance is
        divided out, which evens a side-lit face while leaving facial relief —
        and therefore identity — untouched. Everything here is global tone work;
        nothing alters features.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        out = rgb.astype(np.float32)

        if white_balance > 0:
            face = out[y1:y2, x1:x2]
            means = face.reshape(-1, 3).mean(axis=0)
            gray = means.mean()
            out = np.clip(out * ((1 - white_balance) + white_balance * gray / np.maximum(means, 1)), 0, 255)

        if brightness:
            out = np.clip(out * (1.0 + brightness), 0, 255)

        if light > 0 or contrast:
            lab = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2LAB)
            L = lab[:, :, 0].astype(np.float32)
            if light > 0:
                sigma = max((x2 - x1) / 1.5, 25)
                illum = cv2.GaussianBlur(L, (0, 0), sigma)
                flat = L / np.maximum(illum, 1.0) * illum.mean()
                L = L * (1 - light) + flat * light
            if contrast:
                L = np.clip((L - 128.0) * (1.0 + contrast) + 128.0, 0, 255)
            lab[:, :, 0] = np.clip(L, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)

        return out

    def measure_unevenness(self, entry: dict) -> float:
        """How uneven the light across the face is — drives the default `light`."""
        x1, y1, x2, y2 = [int(v) for v in entry["bbox"]]
        lab = cv2.cvtColor(entry["rgb"], cv2.COLOR_RGB2LAB)
        L = lab[y1:y2, x1:x2, 0].astype(np.float32)
        field = cv2.GaussianBlur(L, (0, 0), max((x2 - x1) / 2.0, 15))
        return float((field.max() - field.min()) / max(field.mean(), 1.0))

    # ------------------------------------------------------------------ render

    def render(self, file_path: str, profile: str, *, brightness: float = 0.0,
               contrast: float = 0.0, white_balance: float = 0.0,
               light: Optional[float] = None, fix_head_tilt: bool = False) -> dict:
        """Produce the finished photo plus a measurement report.

        `light=None` picks a strength from the measured unevenness, capped — a
        fixed high value flattens natural shading and looks wrong.
        """
        spec = get_spec(profile)
        if not spec:
            raise DocumentPhotoError(f"Неизвестный профиль: {profile}")

        e = self._analyze(file_path)
        rgb, mask = e["rgb"], e["mask"]
        x1, y1, x2, y2 = e["bbox"]
        eye, nose = e["eye"], e["nose"]

        auto_light = light is None
        if auto_light:
            light = float(np.clip(self.measure_unevenness(e) * 1.2, 0.0, 0.45))

        if any((brightness, contrast, white_balance, light)):
            rgb = self._correct(rgb, e["bbox"], brightness, contrast, white_balance, light)
        rgb = rgb.astype(np.float32)

        if fix_head_tilt and abs(e["tilt_deg"]) > 0.5:
            m = cv2.getRotationMatrix2D(tuple(eye.astype(float)), e["tilt_deg"], 1.0)
            rgb = cv2.warpAffine(rgb, m, (rgb.shape[1], rgb.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            mask = cv2.warpAffine(mask, m, (mask.shape[1], mask.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderValue=0)

        bg = np.array(spec["background_rgb"], dtype=np.float32)
        a = mask[..., None]
        composed = rgb * a + bg * (1 - a)

        # Geometry: head height sets the scale, the eye line sets the vertical
        # position, and the nose (not the bbox centre) sets the horizontal one —
        # on a slightly turned head the bbox centre drifts off the face axis.
        head = y2 - y1
        frame_h = head / spec["head_ratio"]
        frame_w = frame_h * spec["out_w"] / spec["out_h"]
        top = eye[1] - frame_h * (1.0 - spec["eye_from_bottom"])
        cx = 0.7 * nose[0] + 0.3 * eye[0]
        left = cx - frame_w / 2.0

        canvas = np.full((int(round(frame_h)), int(round(frame_w)), 3), bg, np.float32)
        L, T = int(round(left)), int(round(top))
        H, W = composed.shape[:2]
        sx1, sy1 = max(0, L), max(0, T)
        sx2, sy2 = min(W, L + canvas.shape[1]), min(H, T + canvas.shape[0])
        if sx2 > sx1 and sy2 > sy1:
            canvas[sy1 - T:sy2 - T, sx1 - L:sx2 - L] = composed[sy1:sy2, sx1:sx2]
        out_of_frame = L < 0 or T < 0 or L + canvas.shape[1] > W or T + canvas.shape[0] > H

        out = cv2.resize(canvas.astype(np.uint8), (spec["out_w"], spec["out_h"]),
                         interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", out[:, :, ::-1],
                               [cv2.IMWRITE_JPEG_QUALITY, spec["quality"]])
        if not ok:
            raise DocumentPhotoError("Не удалось закодировать результат")
        jpeg = buf.tobytes()

        k = spec["out_h"] / frame_h
        head_pct = head * k / spec["out_h"]
        border = np.vstack([out[:6, :].reshape(-1, 3), out[-6:, :].reshape(-1, 3),
                            out[:, :6].reshape(-1, 3), out[:, -6:].reshape(-1, 3)])
        bg_uniform = float((np.abs(border.astype(int) - np.array(spec["background_rgb"])).sum(1) < 25).mean())

        return {
            "jpeg": jpeg,
            "spec": profile,
            "params": {
                "brightness": brightness, "contrast": contrast,
                "white_balance": white_balance, "light": light,
                "light_auto": auto_light, "fix_head_tilt": fix_head_tilt,
            },
            "report": {
                "faces_found": e["face_count"],
                "head_ratio": round(head_pct, 3),
                "head_mm": round(head_pct * spec["print_mm"][1], 1),
                "head_in_spec": spec["head_ratio_min"] <= head_pct <= spec["head_ratio_max"],
                "tilt_deg": round(e["tilt_deg"], 1),
                "background_uniform": round(bg_uniform, 3),
                "out_of_frame": out_of_frame,
                "bytes": len(jpeg),
                "size_ok": len(jpeg) <= spec["max_bytes"],
                "output": f"{spec['out_w']}x{spec['out_h']}",
            },
        }

    @staticmethod
    def warnings_from(report: dict, spec_key: str) -> list:
        """Problems the geometry can see. Expression, glasses and gaze are for
        the vision model — this only reports what is measurable."""
        spec = get_spec(spec_key)
        w = []
        if report["faces_found"] > 1:
            w.append(f"на фото {report['faces_found']} лица — для документа нужен один человек")
        if not report["head_in_spec"]:
            w.append(
                f"голова занимает {report['head_ratio']*100:.0f}% кадра "
                f"(норма {spec['head_ratio_min']*100:.0f}–{spec['head_ratio_max']*100:.0f}%)"
            )
        if abs(report["tilt_deg"]) > 3:
            w.append(f"голова наклонена на {report['tilt_deg']:+.1f}° — можно выровнять")
        if report["out_of_frame"]:
            w.append("кадр выходит за границы исходника — поля добиты фоном, проверьте край")
        if report["background_uniform"] < 0.6:
            w.append("фон получился неоднородным — возможен ореол вокруг головы")
        if not report["size_ok"]:
            w.append("файл больше допустимого размера")
        return w
