"""Document photo preparation: crop to spec, replace background, optional gentle correction.

Pipeline: detect face on the ORIGINAL → segment person → crop to the spec
geometry → resize to the output size → optionally composite on the spec
background.

Background replacement is OPT-IN (`remove_background`). The segmentation is
always run, because the crown of the head is read off the mask, but applying it
to the pixels costs thin strands of hair — so on an already-light, even
background it is a downgrade, not an improvement.

Two things learned the hard way and encoded here:

* The frame is anchored on the crown and the chin — the two lines the MOS editor
  draws and the two the regulation constrains. The crown comes from the
  segmentation mask (hair included), the chin from the detector's bbox. Anchoring
  on the eye line instead put the whole head ~2 mm above both guides, and using
  the bbox for the crown leaves the hair out and oversizes the head by ~10%.
* Measurements are taken on the original and carried through the transform, so
  the geometry never depends on re-detecting a face in the finished crop.

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

    @staticmethod
    def _crown_y(mask: np.ndarray, bbox, nose) -> float:
        """Topmost point of the head, hair included.

        Restricted to a band around the face so a raised hand or a hat brim
        elsewhere in the frame cannot pass for the crown, and a row must carry a
        real run of pixels so stray hairs do not either.
        """
        x1, _y1, x2, y2 = bbox
        half = (x2 - x1) * 0.7
        lo, hi = int(max(0, nose[0] - half)), int(min(mask.shape[1], nose[0] + half))
        band = mask[: int(y2), lo:hi] > 0.5
        rows = np.where(band.sum(axis=1) >= 15)[0]
        if len(rows) == 0:
            return float(_y1)  # segmentation gave nothing usable — fall back to the bbox
        return float(rows[0])

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

        mask = self._segment(rgb)
        entry = {
            "at": time.time(),
            "rgb": rgb,
            "mask": mask,
            "crown_y": self._crown_y(mask, (x1, y1, x2, y2), nose),
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
            # Estimated from the brightest neutral pixels OUTSIDE the face. Pulling
            # the face itself towards grey is what greys out the skin: skin is not
            # neutral, and forcing it to be removes exactly the warmth that makes a
            # portrait look alive.
            ref = out.copy()
            ref[y1:y2, x1:x2] = 0
            lum = ref.sum(axis=2)
            thr = np.percentile(lum[lum > 0], 99) if (lum > 0).any() else 0
            patch = ref[lum >= thr] if thr > 0 else np.empty((0, 3))
            if len(patch) >= 50:
                means = patch.mean(axis=0)
                gray = means.mean()
                gain = (1 - white_balance) + white_balance * gray / np.maximum(means, 1)
                out = np.clip(out * gain, 0, 255)

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

    @staticmethod
    def measure_unevenness(rgb: np.ndarray, bbox) -> float:
        """Spread of the low-frequency luminance across the face, relative to its
        mean. Drives the default `light` and reports what is left after it."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
        L = lab[y1:y2, x1:x2, 0].astype(np.float32)
        field = cv2.GaussianBlur(L, (0, 0), max((x2 - x1) / 2.0, 15))
        return float((field.max() - field.min()) / max(field.mean(), 1.0))

    # ------------------------------------------------------------------ render

    def render(self, file_path: str, profile: str, *, brightness: float = 0.0,
               contrast: float = 0.0, white_balance: float = 0.0,
               light: Optional[float] = None, fix_head_tilt: bool = False,
               remove_background: bool = False) -> dict:
        """Produce the finished photo plus a measurement report.

        Order matters for speed: crop and downscale FIRST, correct after. The
        illumination blur uses a sigma proportional to the face, so running it on
        the full-resolution original means a ~1000px-sigma convolution over 12 MP
        — that alone took ~115 s. On the 684x883 crop the whole render is ~1 s,
        and the field being estimated is low-frequency anyway, so nothing is lost.

        `light=None` picks a strength from the measured unevenness, capped — a
        fixed high value flattens natural shading and looks wrong.

        `remove_background` is off by default: the mask is needed for the crown
        either way, but compositing on it shaves thin hair strands, so it is
        applied only when asked for.
        """
        spec = get_spec(profile)
        if not spec:
            raise DocumentPhotoError(f"Неизвестный профиль: {profile}")

        e = self._analyze(file_path)
        rgb, mask = e["rgb"], e["mask"]
        x1, y1, x2, y2 = e["bbox"]
        eye, nose, crown = e["eye"], e["nose"], e["crown_y"]
        H, W = rgb.shape[:2]
        bg = np.array(spec["background_rgb"], dtype=np.float32)

        if fix_head_tilt and abs(e["tilt_deg"]) > 0.5:
            m = cv2.getRotationMatrix2D(tuple(eye.astype(float)), e["tilt_deg"], 1.0)
            rgb = cv2.warpAffine(rgb, m, (W, H), flags=cv2.INTER_LINEAR,
                                 borderValue=tuple(int(v) for v in bg))
            mask = cv2.warpAffine(mask, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            # The landmarks must follow the pixels, or the crop is centred on
            # where the face used to be.
            def _tp(pt):
                return m @ np.array([pt[0], pt[1], 1.0])
            nose = _tp(nose)
            corners = np.array([_tp(pt) for pt in
                                ((x1, y1), (x2, y1), (x1, y2), (x2, y2))])
            x1, y1 = corners.min(axis=0)
            x2, y2 = corners.max(axis=0)
            crown = self._crown_y(mask, (x1, y1, x2, y2), nose)

        # Geometry: crown-to-chin sets the scale and the crown sets the vertical
        # position, which is what pins both of the editor's guides at once. The
        # nose (not the bbox centre) sets the horizontal one — on a slightly
        # turned head the bbox centre drifts off the face axis.
        head = y2 - crown
        frame_h = head / spec["head_ratio"]
        frame_w = frame_h * spec["out_w"] / spec["out_h"]
        top = crown - frame_h * spec["crown_from_top"]
        cx = 0.7 * nose[0] + 0.3 * eye[0]
        left = cx - frame_w / 2.0

        fh, fw = int(round(frame_h)), int(round(frame_w))
        crop = np.full((fh, fw, 3), bg, np.float32)
        crop_mask = np.zeros((fh, fw), np.float32)
        L, T = int(round(left)), int(round(top))
        sx1, sy1 = max(0, L), max(0, T)
        sx2, sy2 = min(W, L + fw), min(H, T + fh)
        if sx2 > sx1 and sy2 > sy1:
            crop[sy1 - T:sy2 - T, sx1 - L:sx2 - L] = rgb[sy1:sy2, sx1:sx2]
            crop_mask[sy1 - T:sy2 - T, sx1 - L:sx2 - L] = mask[sy1:sy2, sx1:sx2]
        out_of_frame = L < 0 or T < 0 or L + fw > W or T + fh > H

        ow, oh = spec["out_w"], spec["out_h"]
        small = cv2.resize(crop.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(crop_mask, (ow, oh), interpolation=cv2.INTER_AREA)
        k = oh / frame_h
        sbox = (max(0, (x1 - left) * k), max(0, (y1 - top) * k),
                min(ow, (x2 - left) * k), min(oh, (y2 - top) * k))

        uneven_before = self.measure_unevenness(small, sbox)
        auto_light = light is None
        if auto_light:
            light = float(np.clip(uneven_before * 1.5, 0.0, 0.6))

        corrected = self._correct(small, sbox, brightness, contrast, white_balance, light) \
            if any((brightness, contrast, white_balance, light)) else small.astype(np.float32)

        if remove_background:
            a = small_mask[..., None]
            corrected = corrected * a + bg * (1 - a)
        out = np.clip(corrected, 0, 255).astype(np.uint8)

        ok, buf = cv2.imencode(".jpg", out[:, :, ::-1],
                               [cv2.IMWRITE_JPEG_QUALITY, spec["quality"]])
        if not ok:
            raise DocumentPhotoError("Не удалось закодировать результат")
        jpeg = buf.tobytes()

        head_pct = float(head / frame_h)
        top_margin_mm = float((crown - top) / frame_h * spec["print_mm"][1])
        # float()/bool() are not cosmetic: these derive from numpy scalars, and a
        # np.float32/np.bool_ in the report makes the whole API response
        # unserialisable.
        chin_from_bottom_mm = float((top + frame_h - y2) / frame_h * spec["print_mm"][1])
        eye_from_bottom = float((top + frame_h - eye[1]) / frame_h)
        # Only where background is actually expected: the shoulders legitimately
        # reach the bottom corners, so sampling the full border understates it.
        border = np.vstack([out[:8, :].reshape(-1, 3),
                            out[:int(oh * 0.5), :8].reshape(-1, 3),
                            out[:int(oh * 0.5), -8:].reshape(-1, 3)])
        bg_uniform = float((np.abs(border.astype(int) - np.array(spec["background_rgb"])).sum(1) < 25).mean())

        return {
            "jpeg": jpeg,
            "spec": profile,
            "params": {
                "brightness": brightness, "contrast": contrast,
                "white_balance": white_balance, "light": light,
                "light_auto": auto_light, "fix_head_tilt": fix_head_tilt,
                "remove_background": remove_background,
            },
            "report": {
                "faces_found": e["face_count"],
                "head_ratio": round(head_pct, 3),
                "head_mm": round(head_pct * spec["print_mm"][1], 1),
                "head_in_spec": bool(spec["head_ratio_min"] <= head_pct <= spec["head_ratio_max"]),
                "top_margin_mm": round(top_margin_mm, 1),
                "chin_from_bottom_mm": round(chin_from_bottom_mm, 1),
                "eye_from_bottom": round(eye_from_bottom, 3),
                "light_unevenness": round(self.measure_unevenness(out, sbox), 3),
                "light_unevenness_before": round(uneven_before, 3),
                "tilt_deg": round(float(e["tilt_deg"]), 1),
                "background_removed": bool(remove_background),
                "background_uniform": round(bg_uniform, 3),
                "out_of_frame": out_of_frame,
                "bytes": len(jpeg),
                "size_ok": len(jpeg) <= spec["max_bytes"],
                "output": f"{ow}x{oh}",
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
        if report.get("top_margin_mm", 99) < 2.0:
            w.append(f"над головой всего {report['top_margin_mm']} мм — макушка почти у края")
        if report.get("light_unevenness", 0) > 0.28:
            w.append("свет на лице всё ещё заметно неравномерный — можно поднять выравнивание")
        if abs(report["tilt_deg"]) > 3:
            w.append(f"голова наклонена на {report['tilt_deg']:+.1f}° — можно выровнять")
        if report["out_of_frame"]:
            w.append("кадр выходит за границы исходника — поля добиты фоном, проверьте край")
        if report["background_uniform"] < 0.6:
            if report.get("background_removed"):
                w.append("фон заменён, но у края остаётся ореол — проверьте контур волос")
            else:
                w.append(
                    "фон на фото не белый или неоднородный — для документа нужен светлый "
                    "ровный фон; могу заменить его на белый, если попросишь (учти: тонкие "
                    "пряди волос при этом могут обрезаться)"
                )
        if not report["size_ok"]:
            w.append("файл больше допустимого размера")
        return w
