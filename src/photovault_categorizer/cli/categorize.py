"""
Nightly categorization run — the main Pi entrypoint.

Algorithm:
  1. Acquire an exclusive file lock; exit 0 immediately if already locked
     (prevents two simultaneous runs from corrupting the vector store).
  2. Fetch the delta queue: photos with processing_status='pending_categorization'.
  3. Self-heal pass: photos in 'ready' status whose vector is missing from the
     store (e.g. store was reset, or the photo was deleted and re-uploaded).
     These are appended to the work list and processed identically to new photos.
  4. Load MobileCLIP-S2, the vector store, and CLIP text prototypes for every
     auto-enabled+rolled-out tag/category that has a prompts.yaml entry.
  5. Build identity prototypes from labelled face clusters (Phase 2 Iter 3):
     for each labelled cluster, average all face vectors → L2-normed prototype per person.
  6. For each photo in the combined queue:
       a. Embed medium.jpg → upsert CLIP vector into store.
       b. Score against all tag/category CLIP prototypes → write source='auto' rows.
       c. Detect faces → insert face rows + upsert face vectors.
       d. Match each new face against identity prototypes; if score >= threshold,
          write source='auto' assignment for the person's tag/category.
       e. In one transaction: all auto-assignment rows + flip processing_status to 'ready'
          + update embedded_at/embedding_model + update faces_detected_at/face_detection_model.
  7. Prune orphan vectors (deleted photos) from the CLIP store and save.
  8. Prune orphan face vectors (deleted faces) from the face store and save.
  9. Log a one-line summary: photos, tags, categories, faces detected, faces matched.

Usage:
    python -m photovault_categorizer.cli.categorize

All configuration via environment variables — see .env.example.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import FACE_MODEL_ID, MODEL_ID, Config
from ..db import (
    fetch_all_face_ids,
    fetch_all_photo_ids,
    fetch_auto_enabled_labels,
    fetch_delta_queue,
    fetch_labelled_cluster_faces,
    fetch_ready_with_medium,
    get_connection,
    insert_faces,
    mark_faces_detected,
)
from ..embed import embed_photos, resolve_path
from ..face_match import build_identity_prototypes, match_faces
from ..face_model import detect_and_embed, load_face_app
from ..face_store import FaceStore
from ..lock import with_lock
from ..model import load_model
from ..prompts import build_prototypes, load_prompts
from ..scoring import score_photo
from ..store import VectorStore
from ..write import assign_auto

log = logging.getLogger(__name__)


def missing_from_store(
    rows: list[tuple[str, str]],
    store: VectorStore,
) -> list[tuple[str, str]]:
    """Return only those (photo_id, medium_path) pairs whose vector is absent from *store*."""
    return [(pid, path) for pid, path in rows if pid not in store]


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
        delta = fetch_delta_queue(conn)
        ready_rows = fetch_ready_with_medium(conn)
        labels = fetch_auto_enabled_labels(conn)
        labelled_face_info = fetch_labelled_cluster_faces(conn)

    # Load CLIP vector store before self-heal check (we need it to detect missing vectors).
    store_path = Path(config.vector_store_dir) / f"{MODEL_ID}.npz"
    store = VectorStore(store_path)
    store.load()
    log.info("CLIP vector store: %d existing vectors at %s", len(store), store_path)

    # Self-heal: ready photos whose CLIP vector disappeared from the store.
    heal_rows = missing_from_store(ready_rows, store)
    if heal_rows:
        log.info("Self-heal: %d ready photo(s) missing from CLIP store — will re-embed", len(heal_rows))

    work_queue = delta + heal_rows

    if not work_queue:
        log.info("No photos pending categorization and no self-heal needed — nothing to do")
        return

    log.info(
        "Work queue: %d photo(s) (%d delta, %d self-heal) | auto-enabled labels: %d",
        len(work_queue),
        len(delta),
        len(heal_rows),
        len(labels),
    )

    # Load CLIP model (CUDA on PC/RTX, CPU on Pi — same code path)
    model, preprocess, tokenizer, device = load_model()

    # Build CLIP text prototypes once per run (cheap: only text encoding, no image I/O)
    prompts_map = load_prompts(config.prompts_path)
    tag_labels = [(i, n, k) for i, n, k in labels if k == "tag"]
    cat_labels = [(i, n, k) for i, n, k in labels if k == "category"]
    tag_protos = build_prototypes(tag_labels, prompts_map, model, tokenizer, device)
    cat_protos = build_prototypes(cat_labels, prompts_map, model, tokenizer, device)
    log.info(
        "Prototypes built: %d tag(s), %d category/ies", len(tag_protos), len(cat_protos)
    )

    # Load InsightFace app for face detection (Phase 2 Iter 1).
    face_app = load_face_app(det_thresh=config.face_det_thresh)
    face_store_path = Path(config.face_store_dir) / f"faces-{FACE_MODEL_ID}.npz"
    face_store = FaceStore(face_store_path)
    face_store.load()
    log.info("Face store: %d existing vectors at %s", len(face_store), face_store_path)

    # Build per-person identity prototypes from labelled clusters (Phase 2 Iter 3).
    identity_prototypes = build_identity_prototypes(labelled_face_info, face_store)
    log.info(
        "Identity prototypes: %d known person(s), threshold=%.2f",
        len(identity_prototypes),
        config.face_match_threshold,
    )

    photos_done = 0
    tags_inserted = 0
    cats_inserted = 0
    denied_skipped = 0
    faces_detected = 0
    faces_matched = 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"/{MODEL_ID}"

    with get_connection(config) as conn:
        for photo_id, medium_path in work_queue:
            try:
                tags_inserted, cats_inserted, denied_skipped, new_faces, new_matched, ok = _process_photo(
                    conn=conn,
                    photo_id=photo_id,
                    medium_path=medium_path,
                    model=model,
                    preprocess=preprocess,
                    device=device,
                    storage_root=config.storage_root,
                    store=store,
                    face_app=face_app,
                    face_store=face_store,
                    tag_protos=tag_protos,
                    cat_protos=cat_protos,
                    identity_prototypes=identity_prototypes,
                    config=config,
                    run_id=run_id,
                    counters=(tags_inserted, cats_inserted, denied_skipped),
                )
                if ok:
                    photos_done += 1
                    faces_detected += new_faces
                    faces_matched += new_matched
            except Exception:
                log.exception("Unexpected error processing photo %s", photo_id)

    # Prune orphan CLIP vectors (photos deleted since last run) and persist.
    with get_connection(config) as conn:
        all_photo_ids = set(fetch_all_photo_ids(conn))
    pruned = store.prune(all_photo_ids)
    store.save()
    if pruned:
        log.info("Pruned %d orphan CLIP vector(s) from store", pruned)

    # Prune orphan face vectors (face rows deleted via photo cascade or re-detection).
    with get_connection(config) as conn:
        all_face_ids = set(fetch_all_face_ids(conn))
    face_pruned = face_store.prune(all_face_ids)
    face_store.save()
    if face_pruned:
        log.info("Pruned %d orphan face vector(s) from face store", face_pruned)

    log.info(
        "photos processed: %d, tags inserted: %d, categories inserted: %d,"
        " denied skipped: %d, faces detected: %d, faces matched: %d,"
        " new unlabeled faces: %d",
        photos_done,
        tags_inserted,
        cats_inserted,
        denied_skipped,
        faces_detected,
        faces_matched,
        faces_detected - faces_matched,
    )


def _process_photo(
    conn,
    photo_id: str,
    medium_path: str,
    model,
    preprocess,
    device: str,
    storage_root: str,
    store: VectorStore,
    face_app,
    face_store: FaceStore,
    tag_protos: dict,
    cat_protos: dict,
    identity_prototypes: dict,
    config: Config,
    run_id: str,
    counters: tuple[int, int, int],
) -> tuple[int, int, int, int, int, bool]:
    """Embed, score, detect faces, match identities and persist one photo.

    Returns (tags_ins, cats_ins, denied_skip, faces_detected, faces_matched, success).
    """
    tags_inserted, cats_inserted, denied_skipped = counters

    # ── CLIP embedding ────────────────────────────────────────────────────────
    embedded = embed_photos(
        [(photo_id, medium_path)], model, preprocess, device, storage_root, store
    )
    if photo_id not in embedded:
        log.warning("Skipping photo %s — CLIP embedding failed", photo_id)
        return tags_inserted, cats_inserted, denied_skipped, 0, 0, False

    photo_vec = store.get_vec(photo_id)
    tag_hits, cat_hits = score_photo(
        photo_vec,
        tag_protos,
        cat_protos,
        config.tag_threshold,
        config.category_top_k,
        config.category_min_score,
    )

    # ── Face detection ────────────────────────────────────────────────────────
    face_detections = []
    try:
        abs_path = resolve_path(storage_root, medium_path)
        face_detections = detect_and_embed(
            face_app, abs_path, face_min_px=config.face_min_px
        )
    except Exception:
        log.warning("Face detection failed for photo %s — continuing without faces", photo_id, exc_info=True)

    # ── Atomic DB write ───────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    face_run_id = run_id.replace(MODEL_ID, FACE_MODEL_ID)
    with conn.transaction():
        for label_id, sc in tag_hits:
            if assign_auto(conn, photo_id, label_id, "tag", sc, run_id):
                tags_inserted += 1
            else:
                denied_skipped += 1

        for label_id, sc in cat_hits:
            if assign_auto(conn, photo_id, label_id, "category", sc, run_id):
                cats_inserted += 1
            else:
                denied_skipped += 1

        face_ids = insert_faces(conn, photo_id, face_detections, face_run_id, FACE_MODEL_ID, now)

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE photos"
                " SET processing_status = 'ready',"
                "     embedded_at = %s,"
                "     embedding_model = %s,"
                "     faces_detected_at = %s,"
                "     face_detection_model = %s"
                " WHERE id = %s",
                (now, MODEL_ID, now, FACE_MODEL_ID, photo_id),
            )

    # Upsert face vectors into the in-memory store (save happens at end of run).
    for face_id, det in zip(face_ids, face_detections):
        face_store.upsert(face_id, det.embedding)

    # ── Identity matching (Phase 2 Iter 3) ───────────────────────────────────
    # Vectors are freshly upserted above, so they're available for matching.
    identity_run_id = face_run_id + "/identity"
    face_id_matches = match_faces(
        face_ids, face_store, identity_prototypes, config.face_match_threshold
    )
    if face_id_matches:
        with conn.transaction():
            for _face_id, label_id, kind, score in face_id_matches:
                assign_auto(conn, photo_id, label_id, kind, score, identity_run_id)

    return tags_inserted, cats_inserted, denied_skipped, len(face_ids), len(face_id_matches), True


if __name__ == "__main__":
    main()
