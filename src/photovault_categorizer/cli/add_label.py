"""
New-label backfill — score all cached image vectors against a single label.

Use this after creating a new tag or category with auto_enabled=true and adding
its mapping entry to prompts.yaml.  The script:

  1. Validates the label exists, has auto_enabled=true, and has a prompts.yaml entry.
  2. Flips rolled_out=false on the label (marks it as in-progress).
  3. Loads MobileCLIP-S2 text encoder + the full vector store.
  4. Scores every cached photo vector against the label prototype.
  5. Writes source='auto' junction rows for hits (respects manual/denied tombstones).
  6. Flips rolled_out=true (backfill complete).

No image re-embedding is performed — only photos already in the vector store are
scored.  Run bulk_embed.py first if the store is missing photos you care about.

Usage:
    python -m photovault_categorizer.cli.add_label <tag-or-category-id>

    <tag-or-category-id> must start with 'tag-' or 'cat-'.

All other configuration via environment variables — see .env.example.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import MODEL_ID, Config
from ..db import fetch_label, get_connection, set_rolled_out
from ..lock import with_lock
from ..model import load_model
from ..prompts import build_prototypes, load_prompts
from ..store import VectorStore
from ..write import assign_auto

log = logging.getLogger(__name__)


# ── Pure helpers (unit-testable without DB or model) ─────────────────────────


def label_kind_from_id(label_id: str) -> str | None:
    """Derive 'tag' or 'category' from the id prefix, or None if unrecognised."""
    if label_id.startswith("tag-"):
        return "tag"
    if label_id.startswith("cat-"):
        return "category"
    return None


def select_hits(
    ids: list[str],
    matrix: np.ndarray,
    prototype: np.ndarray,
    threshold: float,
) -> list[tuple[str, float]]:
    """Return (id, score) pairs where cosine similarity >= threshold.

    Both *matrix* rows and *prototype* must be L2-normalised so that cosine
    similarity equals the dot product.

    Args:
        ids:       photo ids aligned with *matrix* rows.
        matrix:    float32 array [N, D] — all cached photo vectors.
        prototype: float32 array [D]    — L2-normalised label prototype.
        threshold: minimum cosine similarity to include.

    Returns:
        List of (photo_id, score) sorted by score descending.
    """
    scores: np.ndarray = matrix @ prototype  # [N]
    hits = [
        (photo_id, float(sc))
        for photo_id, sc in zip(ids, scores)
        if float(sc) >= threshold
    ]
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) != 2:
        print(
            "Usage: python -m photovault_categorizer.cli.add_label <tag-or-category-id>",
            file=sys.stderr,
        )
        sys.exit(2)

    label_id = sys.argv[1]

    with with_lock():
        _run(label_id)


def _run(label_id: str) -> None:
    kind = label_kind_from_id(label_id)
    if kind is None:
        log.error(
            "Unrecognised id prefix for '%s'. Expected 'tag-...' or 'cat-...'.",
            label_id,
        )
        sys.exit(2)

    config = Config.from_env()

    # ── Validate the label ──────────────────────────────────────────────────
    with get_connection(config) as conn:
        label_row = fetch_label(conn, label_id, kind)

    if label_row is None:
        log.error("%s '%s' not found in the database.", kind.capitalize(), label_id)
        sys.exit(1)

    _, label_name, auto_enabled, _ = label_row

    if not auto_enabled:
        log.error(
            "%s '%s' (%s) has auto_enabled=false. "
            "Enable it via PATCH /v1/%ss/%s before running this script.",
            kind.capitalize(),
            label_name,
            label_id,
            kind,
            label_id,
        )
        sys.exit(1)

    # ── Validate scoring threshold ──────────────────────────────────────────
    if kind == "category" and config.category_min_score <= 0.0:
        log.error(
            "CATEGORY_MIN_SCORE is %.4f (≤ 0). "
            "A zero threshold would assign this category to every photo. "
            "Set a positive value (e.g. CATEGORY_MIN_SCORE=0.20) and retry.",
            config.category_min_score,
        )
        sys.exit(1)

    threshold = config.tag_threshold if kind == "tag" else config.category_min_score

    # ── Load prompts + build prototype ─────────────────────────────────────
    prompts_map = load_prompts(config.prompts_path)

    model, _preprocess, tokenizer, device = load_model()

    prototypes = build_prototypes(
        [(label_id, label_name, kind)],
        prompts_map,
        model,
        tokenizer,
        device,
    )

    if label_id not in prototypes:
        log.error(
            "No prompts.yaml entry for label '%s' (id=%s). "
            "Add an entry and retry — rolled_out has NOT been changed.",
            label_name,
            label_id,
        )
        sys.exit(1)

    prototype = prototypes[label_id]

    # ── Load vector store ───────────────────────────────────────────────────
    store_path = Path(config.vector_store_dir) / f"{MODEL_ID}.npz"
    store = VectorStore(store_path)
    store.load()

    if len(store) == 0:
        log.warning(
            "Vector store is empty — no photos have been embedded yet. "
            "Run bulk_embed.py first, then retry."
        )
        sys.exit(1)

    log.info(
        "Scoring %d cached vector(s) for label '%s' (%s, threshold=%.4f)",
        len(store),
        label_name,
        label_id,
        threshold,
    )

    # ── Score ───────────────────────────────────────────────────────────────
    ids, matrix = store.matrix()
    hits = select_hits(ids, matrix, prototype, threshold)

    log.info("Hits above threshold: %d / %d photos", len(hits), len(ids))

    # ── Mark label as in-progress, write rows, mark complete ────────────────
    run_id = datetime.now(timezone.utc).strftime("addlabel-%Y%m%dT%H%M%SZ") + f"/{MODEL_ID}"

    assigned = 0
    skipped = 0

    with get_connection(config) as conn:
        set_rolled_out(conn, label_id, kind, False)

        for photo_id, score in hits:
            if assign_auto(conn, photo_id, label_id, kind, score, run_id):
                assigned += 1
            else:
                skipped += 1

        set_rolled_out(conn, label_id, kind, True)

    log.info(
        "add-label %s: scored %d, assigned %d, skipped %d (manual/denied)",
        label_id,
        len(ids),
        assigned,
        skipped,
    )


if __name__ == "__main__":
    main()
