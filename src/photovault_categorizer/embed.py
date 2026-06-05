"""
Image embedding: reads medium.jpg from disk, encodes it, and upserts into the store.

Path resolution mirrors PhotoAssetStorage in the Ktor server:
  abs_path = PHOTO_STORAGE_ROOT / medium_path (relative stored in DB)
A path-traversal guard rejects any relative path that escapes the root.
"""

import logging
from pathlib import Path

import torch

from .model import encode_image
from .store import VectorStore

log = logging.getLogger(__name__)


def resolve_path(storage_root: str, relative_path: str) -> Path:
    """Resolve *relative_path* under *storage_root* with a path-traversal guard."""
    root = Path(storage_root).resolve()
    resolved = (root / relative_path).resolve()
    if not str(resolved).startswith(str(root)):
        raise ValueError(
            f"Path traversal attempt: '{relative_path}' escapes storage root '{storage_root}'"
        )
    return resolved


def embed_photos(
    photo_rows: list[tuple[str, str]],
    model: torch.nn.Module,
    preprocess,
    device: str,
    storage_root: str,
    store: VectorStore,
) -> list[str]:
    """Encode each photo in *photo_rows* and upsert its vector into *store*.

    Args:
        photo_rows:   list of (photo_id, medium_path) — medium_path is relative to storage_root.
        model:        open_clip model.
        preprocess:   inference transform returned by load_model().
        device:       'cuda' or 'cpu'.
        storage_root: absolute path to the photo storage directory.
        store:        VectorStore instance — updated in-place; caller is responsible for save().

    Returns:
        List of photo_ids that were successfully embedded.
        Photos whose file is missing or fails to decode are logged and omitted.
    """
    valid_rows: list[tuple[str, Path]] = []
    for photo_id, medium_path in photo_rows:
        try:
            abs_path = resolve_path(storage_root, medium_path)
        except ValueError as exc:
            log.error("Skipping photo %s: %s", photo_id, exc)
            continue

        if not abs_path.exists():
            log.warning(
                "medium.jpg not found for photo %s at %s — skipping", photo_id, abs_path
            )
            continue

        valid_rows.append((photo_id, abs_path))

    if not valid_rows:
        return []

    ids = [r[0] for r in valid_rows]
    paths = [r[1] for r in valid_rows]

    embedded: list[str] = []
    try:
        vecs = encode_image(model, preprocess, paths, device)  # [N, D]
        for photo_id, vec in zip(ids, vecs):
            store.upsert(photo_id, vec)
            embedded.append(photo_id)
    except Exception:
        # Batch failed — try photos one-by-one so a single bad file doesn't block the rest.
        log.warning("Batch encode failed — retrying photos individually", exc_info=True)
        for photo_id, path in valid_rows:
            try:
                vecs = encode_image(model, preprocess, [path], device)
                store.upsert(photo_id, vecs[0])
                embedded.append(photo_id)
            except Exception:
                log.error("Failed to embed photo %s at %s", photo_id, path, exc_info=True)

    return embedded
