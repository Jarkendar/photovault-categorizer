"""Tests for VectorStore — upsert, get_vec, save/load round-trip, prune, matrix."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from photovault_categorizer.store import VectorStore


def _unit_vec(hot_idx: int = 0, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[hot_idx] = 1.0
    return v


def test_upsert_and_get_vec():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "store.npz")
        v = _unit_vec(0)
        store.upsert("photo-1", v)
        np.testing.assert_array_almost_equal(store.get_vec("photo-1"), v)


def test_upsert_replaces_existing():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "store.npz")
        store.upsert("photo-1", _unit_vec(0))
        v2 = _unit_vec(1)
        store.upsert("photo-1", v2)

        np.testing.assert_array_almost_equal(store.get_vec("photo-1"), v2)
        assert len(store) == 1  # no duplicate


def test_get_vec_returns_none_for_unknown_id():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "store.npz")
        assert store.get_vec("photo-unknown") is None


def test_contains():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "store.npz")
        store.upsert("photo-x", _unit_vec())
        assert "photo-x" in store
        assert "photo-y" not in store


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "store.npz"
        v = _unit_vec(3)

        store = VectorStore(path)
        store.upsert("photo-abc", v)
        store.save()

        store2 = VectorStore(path)
        store2.load()
        assert len(store2) == 1
        np.testing.assert_array_almost_equal(store2.get_vec("photo-abc"), v)


def test_load_fresh_store_when_file_missing():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "nonexistent.npz")
        store.load()  # must not raise
        assert len(store) == 0


def test_prune_removes_orphans():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "store.npz")
        for i in range(4):
            store.upsert(f"photo-{i}", _unit_vec(i))

        removed = store.prune({"photo-0", "photo-2"})

        assert removed == 2
        assert len(store) == 2
        assert store.get_vec("photo-1") is None
        assert store.get_vec("photo-3") is None
        assert store.get_vec("photo-0") is not None
        assert store.get_vec("photo-2") is not None


def test_prune_no_change_when_all_valid():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "store.npz")
        store.upsert("photo-a", _unit_vec())
        removed = store.prune({"photo-a"})
        assert removed == 0
        assert len(store) == 1


def test_matrix_shape_and_order():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "store.npz")
        for i in range(3):
            store.upsert(f"photo-{i}", _unit_vec(i))

        ids, mat = store.matrix()
        assert len(ids) == 3
        assert mat.shape == (3, 512)
        assert mat.dtype == np.float32


def test_matrix_empty_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "store.npz")
        ids, mat = store.matrix()
        assert ids == []
        assert mat.shape == (0, 512)
