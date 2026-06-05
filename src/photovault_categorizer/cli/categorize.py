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
  5. For each photo in the combined queue:
       a. Embed medium.jpg → upsert vector into store.
       b. Score against all tag/category prototypes.
       c. In one transaction: insert/update auto-assignment rows + flip
          processing_status to 'ready' + update embedded_at/embedding_model.
  6. Prune orphan vectors (deleted photos) from the store and save.
  7. Log a one-line summary.

Usage:
    python -m photovault_categorizer.cli.categorize

All configuration via environment variables — see .env.example.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import MODEL_ID, Config
from ..db import (
    fetch_all_photo_ids,
    fetch_auto_enabled_labels,
    fetch_delta_queue,
    fetch_ready_with_medium,
    get_connection,
)
from ..embed import embed_photos
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

    # Load vector store before self-heal check (we need it to detect missing vectors).
    store_path = Path(config.vector_store_dir) / f"{MODEL_ID}.npz"
    store = VectorStore(store_path)
    store.load()
    log.info("Vector store: %d existing vectors at %s", len(store), store_path)

    # Self-heal: ready photos whose vector disappeared from the store.
    heal_rows = missing_from_store(ready_rows, store)
    if heal_rows:
        log.info("Self-heal: %d ready photo(s) missing from store — will re-embed", len(heal_rows))

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

    # Load model (CUDA on PC/RTX, CPU on Pi — same code path)
    model, preprocess, tokenizer, device = load_model()

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
        for photo_id, medium_path in work_queue:
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
