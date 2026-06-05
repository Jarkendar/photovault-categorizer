"""
One-shot full-library face detection backfill.

Use this on the PC (RTX 2070 Super) after deploying Phase 2, or after bumping the
face model to a new version (all photos will reappear in the backlog because
face_detection_model changes).

Algorithm:
  1. Acquire exclusive file lock; exit 0 if already locked.
  2. Fetch the face detection backlog:
       photos WHERE faces_detected_at IS NULL
                 OR face_detection_model != current FACE_MODEL_ID
  3. Load InsightFace buffalo_l (CUDA if available, else CPU).
  4. Load the face vector store (faces-buffalo_l.npz).
  5. For each photo (in chunks of CHUNK_SIZE):
       a. Detect and embed all faces from medium.jpg.
       b. In one transaction: delete stale face rows + insert new face rows
          + update photos.faces_detected_at / face_detection_model.
       c. Upsert face vectors into the store; save the store every chunk.
  6. Prune orphan face vectors (faces whose DB row was deleted).
  7. Log a summary.

This script only detects — it does NOT cluster, does NOT label persons, and does
NOT write photo_tags/photo_categories.  Run cluster_faces.py and then label clusters
via the admin API before identity matching begins.

Resumable: re-running skips photos already processed with the current FACE_MODEL_ID.

Usage:
    python -m photovault_categorizer.cli.detect_faces

All configuration via environment variables — see .env.example.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import FACE_MODEL_ID, Config
from ..db import (
    fetch_all_face_ids,
    fetch_face_detection_backlog,
    get_connection,
    insert_faces,
    mark_faces_detected,
)
from ..embed import resolve_path
from ..face_model import FaceDetection, detect_and_embed, load_face_app
from ..face_store import FaceStore
from ..lock import with_lock

log = logging.getLogger(__name__)

CHUNK_SIZE = 64  # Smaller than CLIP bulk_embed — face detection is heavier per image.


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with with_lock():
        _run()


def _run() -> None:
    config = Config.from_env()

    with get_connection(config) as conn:
        backlog = fetch_face_detection_backlog(conn)

    if not backlog:
        log.info("No photos require face detection — nothing to do")
        return

    log.info("Face detection backlog: %d photo(s)", len(backlog))

    app = load_face_app(
        det_thresh=config.face_det_thresh,
    )

    store_path = Path(config.face_store_dir) / f"faces-{FACE_MODEL_ID}.npz"
    store = FaceStore(store_path)
    store.load()
    log.info("Face store: %d existing vectors at %s", len(store), store_path)

    run_id = datetime.now(timezone.utc).strftime("detect-%Y%m%dT%H%M%SZ") + f"/{FACE_MODEL_ID}"

    total_photos = 0
    total_faces = 0
    total_failed = 0

    chunks = [backlog[i : i + CHUNK_SIZE] for i in range(0, len(backlog), CHUNK_SIZE)]
    log.info("Processing %d chunk(s) of up to %d photos each", len(chunks), CHUNK_SIZE)

    for chunk_idx, chunk in enumerate(chunks, start=1):
        log.info("Chunk %d/%d — %d photo(s)", chunk_idx, len(chunks), len(chunk))

        processed_ids: list[str] = []
        now = datetime.now(timezone.utc)

        with get_connection(config) as conn:
            for photo_id, medium_path in chunk:
                try:
                    abs_path = resolve_path(config.storage_root, medium_path)
                except ValueError as exc:
                    log.error("Skipping photo %s: %s", photo_id, exc)
                    total_failed += 1
                    continue

                if not abs_path.exists():
                    log.warning(
                        "medium.jpg not found for photo %s at %s — skipping",
                        photo_id,
                        abs_path,
                    )
                    total_failed += 1
                    continue

                detections: list[FaceDetection] = detect_and_embed(
                    app,
                    abs_path,
                    face_min_px=config.face_min_px,
                )

                try:
                    with conn.transaction():
                        face_ids = insert_faces(
                            conn, photo_id, detections, run_id, FACE_MODEL_ID, now
                        )
                        mark_faces_detected(conn, [photo_id], FACE_MODEL_ID, now)
                except Exception:
                    log.error(
                        "DB write failed for photo %s — skipping", photo_id, exc_info=True
                    )
                    total_failed += 1
                    continue

                # Upsert embeddings into the in-memory store.
                for face_id, det in zip(face_ids, detections):
                    store.upsert(face_id, det.embedding)

                total_faces += len(detections)
                processed_ids.append(photo_id)

        total_photos += len(processed_ids)
        store.save()
        log.info(
            "Chunk %d done: %d photo(s) processed, %d face(s) detected",
            chunk_idx,
            len(processed_ids),
            total_faces,
        )

    # Prune vectors whose DB row has since been deleted.
    with get_connection(config) as conn:
        all_face_ids = set(fetch_all_face_ids(conn))
    pruned = store.prune(all_face_ids)
    store.save()
    if pruned:
        log.info("Pruned %d orphan face vector(s) from store", pruned)

    log.info(
        "detect-faces: %d photos processed, %d faces detected, %d photos failed",
        total_photos,
        total_faces,
        total_failed,
    )


if __name__ == "__main__":
    main()
