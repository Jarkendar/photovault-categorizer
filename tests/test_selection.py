"""
Unit tests for pure selection helpers — no DB, no model, no disk I/O.

Covers:
  - missing_from_store()  (categorize.py)
  - select_hits()         (add_label.py)
  - label_kind_from_id()  (add_label.py)
"""

import numpy as np
import pytest

from photovault_categorizer.cli.add_label import label_kind_from_id, select_hits
from photovault_categorizer.cli.categorize import missing_from_store
from photovault_categorizer.store import VectorStore


# ── Helpers ───────────────────────────────────────────────────────────────────


def _unit_vec(hot_idx: int, dim: int = 4) -> np.ndarray:
    """Return a unit vector with a 1 at *hot_idx*."""
    v = np.zeros(dim, dtype=np.float32)
    v[hot_idx] = 1.0
    return v


def _make_store(ids_vecs: list[tuple[str, int]], dim: int = 4) -> VectorStore:
    """Return an in-memory VectorStore with the given (id, hot_idx) pairs."""
    store = VectorStore("/dev/null")  # path never written in this test
    for pid, hot in ids_vecs:
        store.upsert(pid, _unit_vec(hot, dim))
    return store


# ── missing_from_store ────────────────────────────────────────────────────────


def test_missing_from_store_returns_absent_rows():
    store = _make_store([("photo-1", 0), ("photo-2", 1)])
    rows = [("photo-1", "p1.jpg"), ("photo-3", "p3.jpg"), ("photo-4", "p4.jpg")]
    result = missing_from_store(rows, store)
    assert result == [("photo-3", "p3.jpg"), ("photo-4", "p4.jpg")]


def test_missing_from_store_empty_when_all_present():
    store = _make_store([("photo-1", 0), ("photo-2", 1)])
    rows = [("photo-1", "p1.jpg"), ("photo-2", "p2.jpg")]
    assert missing_from_store(rows, store) == []


def test_missing_from_store_all_absent():
    store = _make_store([])
    rows = [("photo-x", "x.jpg"), ("photo-y", "y.jpg")]
    assert missing_from_store(rows, store) == rows


def test_missing_from_store_empty_input():
    store = _make_store([("photo-1", 0)])
    assert missing_from_store([], store) == []


# ── label_kind_from_id ────────────────────────────────────────────────────────


@pytest.mark.parametrize("label_id,expected", [
    ("tag-abc123", "tag"),
    ("tag-", "tag"),
    ("cat-xyz789", "category"),
    ("cat-", "category"),
    ("photo-123", None),
    ("", None),
    ("TAG-abc", None),   # case-sensitive
])
def test_label_kind_from_id(label_id, expected):
    assert label_kind_from_id(label_id) == expected


# ── select_hits ───────────────────────────────────────────────────────────────


def _build_matrix_and_ids(id_hot_pairs: list[tuple[str, int]], dim: int = 4):
    """Return (ids, matrix) aligned with id_hot_pairs."""
    ids = [p[0] for p in id_hot_pairs]
    matrix = np.stack([_unit_vec(hot, dim) for _, hot in id_hot_pairs])
    return ids, matrix


def test_select_hits_exact_threshold():
    # prototype points at hot_idx=0; photo-1 cosine=1.0, photo-2 cosine=0.0
    ids, matrix = _build_matrix_and_ids([("photo-1", 0), ("photo-2", 1)])
    proto = _unit_vec(0)
    hits = select_hits(ids, matrix, proto, threshold=1.0)
    assert len(hits) == 1
    assert hits[0][0] == "photo-1"
    assert abs(hits[0][1] - 1.0) < 1e-5


def test_select_hits_below_threshold_excluded():
    ids, matrix = _build_matrix_and_ids([("photo-1", 0), ("photo-2", 1)])
    proto = _unit_vec(0)
    hits = select_hits(ids, matrix, proto, threshold=0.5)
    # photo-2 has cosine=0.0 < 0.5 → excluded
    assert all(pid == "photo-1" for pid, _ in hits)


def test_select_hits_sorted_descending():
    # photo-A: cosine=1.0, photo-B: cosine=0.5 (via a 45-degree vector), photo-C: cosine=0.0
    dim = 2
    ids = ["photo-A", "photo-B", "photo-C"]
    v45 = np.array([1.0, 1.0], dtype=np.float32)
    v45 = v45 / np.linalg.norm(v45)
    matrix = np.stack([
        np.array([1.0, 0.0], dtype=np.float32),  # cosine 1.0
        v45,                                       # cosine ≈ 0.707
        np.array([0.0, 1.0], dtype=np.float32),  # cosine 0.0
    ])
    proto = np.array([1.0, 0.0], dtype=np.float32)

    hits = select_hits(ids, matrix, proto, threshold=0.0)
    assert len(hits) == 3
    scores = [sc for _, sc in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0][0] == "photo-A"


def test_select_hits_empty_matrix():
    ids: list[str] = []
    matrix = np.empty((0, 4), dtype=np.float32)
    proto = _unit_vec(0)
    assert select_hits(ids, matrix, proto, threshold=0.0) == []


def test_select_hits_no_hits_above_threshold():
    ids, matrix = _build_matrix_and_ids([("photo-1", 1), ("photo-2", 2)])
    proto = _unit_vec(0)  # orthogonal to all rows → cosine=0
    hits = select_hits(ids, matrix, proto, threshold=0.5)
    assert hits == []
