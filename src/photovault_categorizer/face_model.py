"""
Face detection and embedding via InsightFace buffalo_l.

Each detected face is returned as a FaceDetection with:
  - bbox (x, y, w, h) in pixels of the input image
  - det_score: detection confidence (0–1)
  - embedding: L2-normalised float32 ArcFace vector [512]

Faces smaller than face_min_px in either dimension, or below face_det_thresh
confidence, are filtered out before returning.

Device selection: CUDA (CUDAExecutionProvider) if available, otherwise CPU
(CPUExecutionProvider).  Uses ONNX runtime — no PyTorch dependency here.

INSIGHTFACE_HOME must point to a directory containing the pre-downloaded
buffalo_l pack.  The runtime never fetches models from the network.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import FACE_DET_SIZE, FACE_DET_THRESH, FACE_MIN_PX, FACE_MODEL_ID

log = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Single detected face within an image."""

    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    det_score: float
    embedding: np.ndarray  # float32 [512], L2-normalised


def _providers() -> list[str]:
    """Return ONNX execution providers in preference order."""
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            log.info("Face model: using CUDAExecutionProvider")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except ImportError:
        pass
    log.info("Face model: using CPUExecutionProvider")
    return ["CPUExecutionProvider"]


def load_face_app(
    det_thresh: float = FACE_DET_THRESH,
    det_size: int = FACE_DET_SIZE,
) -> object:
    """Load the InsightFace FaceAnalysis app (buffalo_l).

    Args:
        det_thresh: minimum detection confidence; faces below this are discarded by InsightFace.
        det_size:   input resolution fed to the detector (square, pixels).

    Returns:
        A prepared insightface.app.FaceAnalysis instance ready for inference.

    Raises:
        ImportError: if insightface or onnxruntime is not installed.
        RuntimeError: if the model pack cannot be found under INSIGHTFACE_HOME.
    """
    import insightface
    from insightface.app import FaceAnalysis  # type: ignore[import]

    log.info(
        "Loading InsightFace model pack '%s' (det_thresh=%.2f, det_size=%d)",
        FACE_MODEL_ID,
        det_thresh,
        det_size,
    )

    app = FaceAnalysis(
        name=FACE_MODEL_ID,
        providers=_providers(),
    )
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(det_size, det_size))

    log.info(
        "InsightFace app ready — models: %s",
        [m.__class__.__name__ for m in app.models.values()],
    )
    return app


def detect_and_embed(
    app: object,
    image_path: Path | str,
    face_min_px: int = FACE_MIN_PX,
) -> list[FaceDetection]:
    """Detect faces in *image_path* and return their bounding boxes + ArcFace embeddings.

    Args:
        app:         FaceAnalysis instance returned by load_face_app().
        image_path:  absolute path to the image file (medium.jpg).
        face_min_px: faces smaller than this in either dimension are silently discarded.

    Returns:
        List of FaceDetection objects, one per accepted face.  Returns an empty list if
        the image cannot be decoded or no faces pass the size/confidence filters.
    """
    import cv2  # InsightFace requires OpenCV for image loading

    path = Path(image_path)
    img = cv2.imread(str(path))
    if img is None:
        log.warning("Could not decode image at %s — skipping face detection", path)
        return []

    try:
        raw_faces = app.get(img)
    except Exception:
        log.error("InsightFace inference failed for %s", path, exc_info=True)
        return []

    detections: list[FaceDetection] = []
    for face in raw_faces:
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        w = x2 - x1
        h = y2 - y1

        if w < face_min_px or h < face_min_px:
            log.debug(
                "Discarding small face at (%d,%d,%d,%d) in %s", x1, y1, w, h, path.name
            )
            continue

        if face.embedding is None:
            log.debug("Face at (%d,%d) has no embedding — skipping", x1, y1)
            continue

        emb = face.embedding.astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm

        detections.append(
            FaceDetection(
                bbox_x=max(0, x1),
                bbox_y=max(0, y1),
                bbox_w=w,
                bbox_h=h,
                det_score=float(face.det_score),
                embedding=emb,
            )
        )

    if detections:
        log.debug("Detected %d face(s) in %s", len(detections), path.name)
    return detections
