"""
Nightly categorization run — the main Pi entrypoint.

Algorithm:
  1. Acquire an exclusive file lock; exit 0 immediately if already locked
     (prevents two simultaneous runs from corrupting the vector store).
  2. Fetch the delta queue: photos with processing_status='pending_categorization'.
  3. Load MobileCLIP-S2, the vector store, and CLIP text prototypes for every
     auto-enabled+rolled-out tag/category that has a prompts.yaml entry.
  4. For each photo in the delta:
       a. Embed medium.jpg → upsert vector into store.
       b. Score against all tag/category prototypes.
       c. In one transaction: insert/update auto-assignment rows + flip
          processing_status to 'ready' + update embedded_at/embedding_model.
  5. Prune orphan vectors (deleted photos) from the store and save.
  6. Log a one-line summary.

Usage:
    python -m photovault_categorizer.cli.categorize

All configuration via environment variables — see .env.example.
"""

import fcntl
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import MODEL_ID, Config
from ..db import (
    fetch_all_photo_ids,
    fetch_auto_enabled_labels,
    fetch_delta_queue,
    get_connection,
)
from ..embed import embed_photos
from ..model import load_model
from ..prompts import build_prototypes, load_prompts
from ..scoring import score_photo
from ..store import VectorStore
from ..write import assign_auto

log = logging.getLogger(__name__)

LOCK_PATH = "/tmp/photovault-categorize.lock"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    lock_file = open(LOCK_PATH, "w")  # noqa: WPS515
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.info("Another categorize run is active (lock held at %s) — exiting 0", LOCK_PATH)
        lock_file.close()
        sys.exit(0)

    try:
        _run()
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def _run() -> None:
    config = Config.from_env()

    with get_connection(config) as conn:
        delta = fetch_delta_queue(conn)
        labels = fetch_auto_enabled_labels(conn)

    if not delta:
        log.info("No photos pending categorization — nothing to do")
        return

    log.info(
        "Delta queue: %d photo(s) | auto-enabled labels: %d",
        len(delta),
        len(labels),
    )

    # Load model (CUDA on PC/RTX, CPU on Pi — same code path)
    model, preprocess, tokenizer, device = load_model()

    # Load vector store (creates a fresh one if the .npz does not exist yet)
    store_path = Path(config.vector_store_dir) / f"{MODEL_ID}.npz"
    store = VectorStore(store_path)
    store.load()
    log.info("Vector store: %d existing vectors at %s", len(store), store_path)

    # Build text prototypes once per run (cheap: only text encoding, no image I/O)
    prompts_map = load_prompts(config.prompts_path)
    tag_labels = [(i, n, k) for i, n, k in labels if k == "tag"]
    cat_labels = [(i, n, k) for i, n, k in labels if k == "category"]
    tag_protos = build_prototypes(tag_labels, prompts_map, model, tokenizer, device)
    cat_protos = build_prototypes(cat_labels, prompts_map, model, tokenizer, device)
    log.info(
        "Prototypes built: %d tag(s), %d category/ies", len(tag_protos), len(cat_protos)
    )

    photos_done = 0
    tags_inserted = 0
    cats_inserted = 0
    denied_skipped = 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"/{MODEL_ID}"

    with get_connection(config) as conn:
        for photo_id, medium_path in delta:
            try:
                tags_inserted, cats_inserted, denied_skipped, ok = _process_photo(
                    conn=conn,
                    photo_id=photo_id,
                    medium_path=medium_path,
                    model=model,
                    preprocess=preprocess,
                    device=device,
                    storage_root=config.storage_root,
                    store=store,
                    tag_protos=tag_protos,
                    cat_protos=cat_protos,
                    config=config,
                    run_id=run_id,
                    counters=(tags_inserted, cats_inserted, denied_skipped),
                )
                if ok:
                    photos_done += 1
            except Exception:
                log.exception("Unexpected error processing photo %s", photo_id)

    # Prune orphan vectors (photos deleted since last run) and persist
    with get_connection(config) as conn:
        all_ids = set(fetch_all_photo_ids(conn))
    pruned = store.prune(all_ids)
    store.save()
    if pruned:
        log.info("Pruned %d orphan vector(s) from store", pruned)

    log.info(
        "photos processed: %d, tags inserted: %d, categories inserted: %d, denied skipped: %d",
        photos_done,
        tags_inserted,
        cats_inserted,
        denied_skipped,
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
    tag_protos: dict,
    cat_protos: dict,
    config: Config,
    run_id: str,
    counters: tuple[int, int, int],
) -> tuple[int, int, int, bool]:
    """Embed, score, and persist one photo.  Returns (tags_ins, cats_ins, denied_skip, success)."""
    tags_inserted, cats_inserted, denied_skipped = counters

    embedded = embed_photos(
        [(photo_id, medium_path)], model, preprocess, device, storage_root, store
    )
    if photo_id not in embedded:
        log.warning("Skipping photo %s — embedding failed", photo_id)
        return tags_inserted, cats_inserted, denied_skipped, False

    photo_vec = store.get_vec(photo_id)
    tag_hits, cat_hits = score_photo(
        photo_vec,
        tag_protos,
        cat_protos,
        config.tag_threshold,
        config.category_top_k,
        config.category_min_score,
    )

    now = datetime.now(timezone.utc)
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

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE photos"
                " SET processing_status = 'ready',"
                "     embedded_at = %s,"
                "     embedding_model = %s"
                " WHERE id = %s",
                (now, MODEL_ID, photo_id),
            )

    return tags_inserted, cats_inserted, denied_skipped, True


if __name__ == "__main__":
    main()
