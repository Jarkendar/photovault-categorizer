"""
Unit tests for face_match — build_identity_prototypes and match_faces.

All tests are pure in-memory: no DB, no InsightFace model, no filesystem I/O.
The FaceStore is populated directly via upsert() so tests run in milliseconds.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from photovault_categorizer.face_match import build_identity_prototypes, match_faces
from photovault_categorizer.face_store import FaceStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _unit(hot_idx: int, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[hot_idx] = 1.0
    return v


def _store_with(*items: tuple[str, np.ndarray]) -> FaceStore:
    """Create an in-memory FaceStore without touching disk."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FaceStore(Path(tmp) / "test.npz")
        for face_id, vec in items:
            store.upsert(face_id, vec)
    return store


# ── build_identity_prototypes ─────────────────────────────────────────────────

def test_empty_labelled_info_returns_empty():
    store = _store_with()
    result = build_identity_prototypes([], store)
    assert result == {}


def test_single_face_prototype_is_l2_normed():
    vec = _unit(0)
    store = _store_with(("face-a", vec))
    protos = build_identity_prototypes([("face-a", "tag-grandma", "tag")], store)
    assert "tag-grandma" in protos
    proto_vec, kind = protos["tag-grandma"]
    assert kind == "tag"
    assert abs(float(np.linalg.norm(proto_vec)) - 1.0) < 1e-5


def test_two_faces_same_person_averaged():
    # Two orthogonal unit vectors → mean = (1,1,0,...)/sqrt(2)
    v1 = _unit(0)
    v2 = _unit(1)
    store = _store_with(("face-1", v1), ("face-2", v2))
    info = [("face-1", "tag-alice", "tag"), ("face-2", "tag-alice", "tag")]
    protos = build_identity_prototypes(info, store)
    proto_vec, _ = protos["tag-alice"]
    # prototype must be unit-normed
    assert abs(float(np.linalg.norm(proto_vec)) - 1.0) < 1e-5
    # and lie along the (1,1,0,...) diagonal
    assert proto_vec[0] > 0.6
    assert proto_vec[1] > 0.6


def test_two_distinct_persons_separate_prototypes():
    v1 = _unit(0)   # person A
    v2 = _unit(100) # person B
    store = _store_with(("face-a", v1), ("face-b", v2))
    info = [("face-a", "tag-alice", "tag"), ("face-b", "tag-bob", "tag")]
    protos = build_identity_prototypes(info, store)
    assert "tag-alice" in protos
    assert "tag-bob" in protos
    # prototypes should be nearly orthogonal
    dot = float(protos["tag-alice"][0] @ protos["tag-bob"][0])
    assert abs(dot) < 1e-4


def test_missing_face_in_store_skipped_gracefully():
    store = _store_with()  # empty store
    info = [("face-missing", "tag-alice", "tag")]
    protos = build_identity_prototypes(info, store)
    assert protos == {}


def test_kind_category_preserved():
    store = _store_with(("face-1", _unit(5)))
    info = [("face-1", "cat-vacation", "category")]
    protos = build_identity_prototypes(info, store)
    _, kind = protos["cat-vacation"]
    assert kind == "category"


# ── match_faces ───────────────────────────────────────────────────────────────

def test_no_prototypes_returns_empty():
    store = _store_with(("face-x", _unit(0)))
    result = match_faces(["face-x"], store, {}, threshold=0.5)
    assert result == []


def test_no_face_ids_returns_empty():
    store = _store_with()
    protos = {"tag-alice": (_unit(0), "tag")}
    result = match_faces([], store, protos, threshold=0.5)
    assert result == []


def test_face_near_prototype_matches():
    vec = _unit(0)
    store = _store_with(("face-new", vec))
    protos = {"tag-alice": (vec, "tag")}  # identical → cosine sim = 1.0
    result = match_faces(["face-new"], store, protos, threshold=0.5)
    assert len(result) == 1
    face_id, label_id, kind, score = result[0]
    assert face_id == "face-new"
    assert label_id == "tag-alice"
    assert kind == "tag"
    assert score > 0.99


def test_face_far_from_prototype_no_match():
    # face near axis 0, prototype near axis 100 → cosine sim ≈ 0
    store = _store_with(("face-stranger", _unit(0)))
    protos = {"tag-alice": (_unit(100), "tag")}
    result = match_faces(["face-stranger"], store, protos, threshold=0.5)
    assert result == []


def test_threshold_boundary():
    v = _unit(0)
    store = _store_with(("face-x", v))
    protos = {"tag-alice": (v, "tag")}
    # exactly at threshold → should match
    assert len(match_faces(["face-x"], store, protos, threshold=1.0)) == 1
    # just above perfect similarity → impossible, but a threshold of 1.01 should not match
    assert len(match_faces(["face-x"], store, protos, threshold=1.01)) == 0


def test_picks_closest_prototype():
    face_vec = _unit(0)  # near axis 0
    proto_a = _unit(0)   # distance 0 (same person)
    proto_b = _unit(1)   # orthogonal (different person)
    store = _store_with(("face-x", face_vec))
    protos = {"tag-alice": (proto_a, "tag"), "tag-bob": (proto_b, "tag")}
    result = match_faces(["face-x"], store, protos, threshold=0.5)
    assert len(result) == 1
    assert result[0][1] == "tag-alice"


def test_missing_face_vector_skipped():
    store = _store_with()  # no vectors loaded
    protos = {"tag-alice": (_unit(0), "tag")}
    result = match_faces(["face-ghost"], store, protos, threshold=0.5)
    assert result == []


def test_denied_precedence_is_write_responsibility():
    # match_faces is pure computation — it always returns matches above threshold.
    # The caller (categorize.py) passes the result to write.assign_auto() which
    # enforces the denied/manual precedence via ON CONFLICT WHERE source='auto'.
    # This test documents that match_faces does NOT filter denied faces itself.
    vec = _unit(0)
    store = _store_with(("face-denied", vec))
    protos = {"tag-alice": (vec, "tag")}
    result = match_faces(["face-denied"], store, protos, threshold=0.5)
    assert len(result) == 1  # match returned — write layer handles precedence
