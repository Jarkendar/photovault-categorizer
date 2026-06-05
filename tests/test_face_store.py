"""Tests for FaceStore — mirrors test_store.py for the face-embedding store.

The key structural difference from VectorStore:
  - keyed by face-id strings ("face-<uuid>"), not photo-id strings
  - npz arrays are named "face_ids" and "vectors" (not "ids")

All tests are pure in-memory + tempfile — no DB, no model required.
"""

import tempfile
from pathlib import Path

import numpy as np

from photovault_categorizer.face_store import FaceStore


def _unit_vec(hot_idx: int = 0, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[hot_idx] = 1.0
    return v


def test_upsert_and_get_vec():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "faces.npz")
        v = _unit_vec(0)
        store.upsert("face-1", v)
        np.testing.assert_array_almost_equal(store.get_vec("face-1"), v)


def test_upsert_replaces_existing():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "faces.npz")
        store.upsert("face-1", _unit_vec(0))
        v2 = _unit_vec(1)
        store.upsert("face-1", v2)

        np.testing.assert_array_almost_equal(store.get_vec("face-1"), v2)
        assert len(store) == 1  # no duplicate


def test_get_vec_returns_none_for_unknown_id():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "faces.npz")
        assert store.get_vec("face-unknown") is None


def test_contains():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "faces.npz")
        store.upsert("face-x", _unit_vec())
        assert "face-x" in store
        assert "face-y" not in store


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "faces.npz"
        v = _unit_vec(3)

        store = FaceStore(path)
        store.upsert("face-abc", v)
        store.save()

        store2 = FaceStore(path)
        store2.load()
        assert len(store2) == 1
        np.testing.assert_array_almost_equal(store2.get_vec("face-abc"), v)


def test_load_fresh_store_when_file_missing():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "nonexistent.npz")
        store.load()  # must not raise
        assert len(store) == 0


def test_prune_removes_orphans():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "faces.npz")
        for i in range(4):
            store.upsert(f"face-{i}", _unit_vec(i))

        removed = store.prune({"face-0", "face-2"})

        assert removed == 2
        assert len(store) == 2
        assert store.get_vec("face-1") is None
        assert store.get_vec("face-3") is None
        assert store.get_vec("face-0") is not None
        assert store.get_vec("face-2") is not None


def test_prune_no_change_when_all_valid():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "faces.npz")
        store.upsert("face-a", _unit_vec())
        removed = store.prune({"face-a"})
        assert removed == 0
        assert len(store) == 1


def test_matrix_shape_and_order():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "faces.npz")
        for i in range(3):
            store.upsert(f"face-{i}", _unit_vec(i))

        ids, mat = store.matrix()
        assert len(ids) == 3
        assert mat.shape == (3, 512)
        assert mat.dtype == np.float32


def test_matrix_empty_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "faces.npz")
        ids, mat = store.matrix()
        assert ids == []
        assert mat.shape == (0, 512)


def test_multiple_faces_same_photo_all_stored():
    """Multiple face-ids from the same photo should all coexist in the store."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "faces.npz")
        # Simulate two faces detected in photo-1
        store.upsert("face-a1", _unit_vec(0))
        store.upsert("face-a2", _unit_vec(1))
        # And one in photo-2
        store.upsert("face-b1", _unit_vec(2))

        assert len(store) == 3

        # Prune face-a2 (e.g. photo re-detected and face set changed)
        removed = store.prune({"face-a1", "face-b1"})
        assert removed == 1
        assert store.get_vec("face-a2") is None
        assert store.get_vec("face-a1") is not None
