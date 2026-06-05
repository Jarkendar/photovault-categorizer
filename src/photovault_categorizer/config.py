"""
Configuration loaded from environment variables.

Accepts DB_URL in JDBC format (jdbc:postgresql://host:port/db) or plain psycopg3 DSN
(postgresql://host:port/db) — whichever the PhotoVault server .env already uses.
"""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

# ── CLIP visual pipeline (Phase 1) ───────────────────────────────────────────

# Identifiers kept in sync with DB column `embedding_model` and the vector store filename.
MODEL_ID = "mobileclip-s2-datacompdr"
MODEL_NAME = "MobileCLIP-S2"
MODEL_PRETRAINED = "datacompdr"

# ── Face pipeline (Phase 2) ───────────────────────────────────────────────────

# InsightFace model pack.  Kept in sync with DB column `face_detection_model`
# and the face vector store filename.
FACE_MODEL_ID = "buffalo_l"

# Detection confidence threshold: detections below this score are discarded.
# Tune on real photos — too low = false positives (background objects detected as faces);
# too high = missed profiles/small faces.
# Value chosen after calibration: see categorizer/README.md, section "Face detection knobs".
FACE_DET_THRESH: float = 0.5

# Minimum face bounding-box dimension (pixels, in medium.jpg coordinates).
# Faces smaller than this in either width or height are too blurry to reliably embed.
# Tune together with FACE_DET_THRESH — see README.
# Default 40 px is a conservative lower bound; raise to 60–80 if cluster quality is poor.
FACE_MIN_PX: int = 40

# Input size (square) fed to the face detector (pixels).  512 is the InsightFace default
# and works well on medium-res photos (~1200 px longest side).
FACE_DET_SIZE: int = 512

# Identity matching threshold (cosine similarity, Phase 2 Iter 3).
# A detected face is matched to a known person when similarity >= this value.
# Start at 0.50 and calibrate on photos of the same person at different ages/lighting.
# Too low = wrong person matched; too high = known person not recognised.
FACE_MATCH_THRESHOLD: float = 0.50


@dataclass(frozen=True)
class Config:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    storage_root: str
    vector_store_dir: str
    prompts_path: str
    tag_threshold: float
    category_top_k: int
    category_min_score: float
    # Phase 2 — face pipeline
    face_store_dir: str
    face_det_thresh: float
    face_min_px: int
    face_match_threshold: float

    @classmethod
    def from_env(cls) -> "Config":
        raw_url = os.environ.get("DB_URL", "jdbc:postgresql://localhost:5432/photovault")
        # Strip 'jdbc:' prefix that the Ktor server uses.
        if raw_url.startswith("jdbc:"):
            raw_url = raw_url[5:]
        parsed = urlparse(raw_url)

        return cls(
            db_host=parsed.hostname or "localhost",
            db_port=parsed.port or 5432,
            db_name=(parsed.path or "/photovault").lstrip("/"),
            db_user=os.environ.get("DB_USER", "photovault"),
            db_password=os.environ.get("DB_PASSWORD", ""),
            storage_root=os.environ.get("PHOTO_STORAGE_ROOT", "./data/photos"),
            vector_store_dir=os.environ.get("VECTOR_STORE_DIR", "./data/vectors"),
            prompts_path=os.environ.get("PROMPTS_PATH", "prompts.yaml"),
            tag_threshold=float(os.environ.get("TAG_THRESHOLD", "0.25")),
            category_top_k=int(os.environ.get("CATEGORY_TOP_K", "1")),
            category_min_score=float(os.environ.get("CATEGORY_MIN_SCORE", "0.0")),
            face_store_dir=os.environ.get("FACE_STORE_DIR", "./data/vectors"),
            face_det_thresh=float(os.environ.get("FACE_DET_THRESH", str(FACE_DET_THRESH))),
            face_min_px=int(os.environ.get("FACE_MIN_PX", str(FACE_MIN_PX))),
            face_match_threshold=float(
                os.environ.get("FACE_MATCH_THRESHOLD", str(FACE_MATCH_THRESHOLD))
            ),
        )
