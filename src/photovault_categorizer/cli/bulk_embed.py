"""
One-shot full-library embedding backfill.

Use this when you have a large existing photo library that was imported before
the categorizer was deployed, or after bumping the embedding model to a new
version (all photos will show up in the backlog because embedding_model changes).

This script only embeds — it does NOT score, does NOT write junction rows, and
does NOT change processing_status.  Run the nightly categorize.py separately to
score and assign labels after embedding.

Usage:
    python -m photovault_categorizer.cli.bulk_embed

All configuration via environment variables — see .env.example.

Resumable: the run is checkpointed every CHUNK_SIZE photos.  If interrupted,
re-running will skip photos that are already embedded with the current MODEL_ID.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import MODEL_ID, Config
from ..db import fetch_embed_backlog, get_connection, mark_embedded
from ..embed import embed_photos
from ..lock import with_lock
from ..model import load_model
from ..store import VectorStore

log = logging.getLogger(__name__)

CHUNK_SIZE = 256


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
        backlog = fetch_embed_backlog(conn)

    if not backlog:
        log.info("No photos require embedding — nothing to do")
        return

    log.info("Embedding backlog: %d photo(s)", len(backlog))

    model, preprocess, _tokenizer, device = load_model()

    store_path = Path(config.vector_store_dir) / f"{MODEL_ID}.npz"
    store = VectorStore(store_path)
    store.load()
    log.info("Vector store: %d existing vectors at %s", len(store), store_path)

    total_embedded = 0
    total_failed = 0

    chunks = [backlog[i : i + CHUNK_SIZE] for i in range(0, len(backlog), CHUNK_SIZE)]
    log.info("Processing %d chunk(s) of up to %d photos each", len(chunks), CHUNK_SIZE)

    for chunk_idx, chunk in enumerate(chunks, start=1):
        log.info("Chunk %d/%d — %d photo(s)", chunk_idx, len(chunks), len(chunk))

        embedded_ids = embed_photos(chunk, model, preprocess, device, config.storage_root, store)

        now = datetime.now(timezone.utc)
        with get_connection(config) as conn:
            mark_embedded(conn, embedded_ids, MODEL_ID, now)

        store.save()

        chunk_failed = len(chunk) - len(embedded_ids)
        total_embedded += len(embedded_ids)
        total_failed += chunk_failed

        if chunk_failed:
            log.warning(
                "Chunk %d: %d photo(s) failed to embed (file missing or decode error)",
                chunk_idx,
                chunk_failed,
            )

    log.info("bulk embed: %d photos embedded, %d failed", total_embedded, total_failed)


if __name__ == "__main__":
    main()
