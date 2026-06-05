"""
File-backed vector store for face embeddings — mirrors store.py for CLIP vectors.

Layout on disk:
  face_ids:  object array of face-id strings  [N]   (e.g. "face-<uuid>")
  vectors:   float32 array                    [N, 512]

One file per model version:
  <face_store_dir>/faces-buffalo_l.npz

The store is designed for in-memory use: load() once, mutate via upsert(),
query via get_vec() / matrix(), and flush via save().

Keyed by face-id (not photo-id) because one photo has many faces.
Use matrix() to get all vectors for batch cosine scoring during clustering
and identity matching.
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_DIM = 512


class FaceStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._id_to_idx: dict[str, int] = {}
        self._dirty: bool = False

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load vectors from disk.  No-op (fresh store) if the file does not exist."""
        if not self.path.exists():
            log.info("No existing face store at %s — starting fresh", self.path)
            return

        data = np.load(self.path, allow_pickle=True)
        self._ids = list(data["face_ids"])
        vecs_arr: np.ndarray = data["vectors"]
        self._vecs = [vecs_arr[i] for i in range(len(self._ids))]
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}
        log.info("Loaded %d face vectors from %s", len(self._ids), self.path)

    def save(self) -> None:
        """Flush to disk.  Skips the write if nothing changed and the file already exists."""
        if not self._dirty and self.path.exists():
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        vecs_arr = (
            np.stack(self._vecs).astype(np.float32)
            if self._vecs
            else np.empty((0, _DEFAULT_DIM), dtype=np.float32)
        )
        np.savez(
            self.path,
            face_ids=np.array(self._ids, dtype=object),
            vectors=vecs_arr,
        )
        self._dirty = False
        log.info("Saved %d face vectors to %s", len(self._ids), self.path)

    # ── Mutations ────────────────────────────────────────────────────────────

    def upsert(self, face_id: str, vec: np.ndarray) -> None:
        """Insert or replace the vector for *face_id*."""
        if face_id in self._id_to_idx:
            self._vecs[self._id_to_idx[face_id]] = vec.astype(np.float32)
        else:
            idx = len(self._ids)
            self._ids.append(face_id)
            self._vecs.append(vec.astype(np.float32))
            self._id_to_idx[face_id] = idx
        self._dirty = True

    def prune(self, valid_ids: set[str]) -> int:
        """Remove vectors whose face_id is not in *valid_ids*.

        Returns the number of vectors removed.
        """
        before = len(self._ids)
        kept = [
            (id_, vec)
            for id_, vec in zip(self._ids, self._vecs)
            if id_ in valid_ids
        ]
        removed = before - len(kept)
        if removed:
            self._ids = [p[0] for p in kept]
            self._vecs = [p[1] for p in kept]
            self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}
            self._dirty = True
        return removed

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_vec(self, face_id: str) -> np.ndarray | None:
        """Return the vector for *face_id*, or None if not present."""
        idx = self._id_to_idx.get(face_id)
        return self._vecs[idx] if idx is not None else None

    def matrix(self) -> tuple[list[str], np.ndarray]:
        """Return (face_ids, vectors) as a dense float32 matrix [N, 512].

        Used for batch cosine scoring during clustering and identity matching.
        """
        if not self._ids:
            return [], np.empty((0, _DEFAULT_DIM), dtype=np.float32)
        return list(self._ids), np.stack(self._vecs).astype(np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, face_id: str) -> bool:
        return face_id in self._id_to_idx
