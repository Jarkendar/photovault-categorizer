"""
Unit tests for the clustering logic used by cluster_faces.py.

Tests the DBSCAN pass in isolation — no DB, no face store I/O, no InsightFace.

Strategy: synthetic ArcFace-like vectors (L2-normalised, 512-d) arranged in tight
groups.  Within a group vectors are nearly identical (cosine distance ~0); between
groups they are orthogonal (cosine distance = 1.0).  DBSCAN_EPS=0.4 cleanly
separates them.
"""

import numpy as np
import pytest

# The clustering logic lives inside _run(), so we test it directly via the same
# algorithm path rather than going through the full CLI.

DBSCAN_EPS = 0.4
DBSCAN_MIN_SAMPLES = 2


def _l2(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of v."""
    return v / np.linalg.norm(v)


def _cluster(vecs: list[np.ndarray]) -> list[int]:
    """Run DBSCAN on a list of 512-d vectors and return integer labels."""
    from sklearn.cluster import DBSCAN

    if not vecs:
        return []
    matrix = np.stack(vecs).astype(np.float64)
    labels: np.ndarray = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="cosine",
        n_jobs=1,
    ).fit_predict(matrix)
    return labels.tolist()


def _make_group(center_idx: int, count: int, dim: int = 512, noise_scale: float = 0.01) -> list[np.ndarray]:
    """Return `count` nearly-identical L2-normed vectors near a one-hot direction."""
    base = np.zeros(dim, dtype=np.float32)
    base[center_idx] = 1.0
    vecs = []
    rng = np.random.default_rng(seed=center_idx)
    for _ in range(count):
        v = base + rng.normal(0, noise_scale, dim).astype(np.float32)
        vecs.append(_l2(v))
    return vecs


# ── Tests ────────────────────────────────────────────────────────────────────

def test_empty_input():
    assert _cluster([]) == []


def test_single_face_is_noise():
    vecs = _make_group(0, 1)
    labels = _cluster(vecs)
    assert labels == [-1]


def test_two_faces_same_person_form_cluster():
    vecs = _make_group(0, 2)
    labels = _cluster(vecs)
    assert labels[0] == labels[1]
    assert labels[0] != -1


def test_two_distinct_groups_get_different_labels():
    group_a = _make_group(0, 3)    # near axis 0
    group_b = _make_group(100, 3)  # near axis 100 — orthogonal to axis 0
    vecs = group_a + group_b
    labels = _cluster(vecs)

    label_a = set(labels[:3])
    label_b = set(labels[3:])
    assert len(label_a) == 1, "all faces in group A should share one label"
    assert len(label_b) == 1, "all faces in group B should share one label"
    assert label_a != label_b, "the two groups should have different labels"
    assert -1 not in label_a
    assert -1 not in label_b


def test_noise_outlier_labelled_minus_one():
    group = _make_group(0, 3)

    # Outlier: unit vector along axis 200 (fully orthogonal to group, distance = 1.0)
    outlier = np.zeros(512, dtype=np.float32)
    outlier[200] = 1.0

    vecs = group + [outlier]
    labels = _cluster(vecs)

    assert labels[-1] == -1, "isolated outlier must be noise"
    cluster_label = labels[0]
    assert all(l == cluster_label for l in labels[:3]), "group members must share a label"


def test_three_groups_each_labelled():
    all_vecs = _make_group(0, 2) + _make_group(100, 2) + _make_group(200, 2)
    labels = _cluster(all_vecs)

    assert -1 not in labels
    # Three distinct labels expected.
    assert len(set(labels)) == 3


def test_min_samples_two_prevents_singletons():
    """A single face in its own neighborhood must not form a cluster."""
    solo = [_l2(np.array([1.0] + [0.0] * 511, dtype=np.float32))]
    # Two different singletons — neither should cluster with the other
    other = [_l2(np.array([0.0] * 255 + [1.0] + [0.0] * 256, dtype=np.float32))]
    labels = _cluster(solo + other)
    assert labels == [-1, -1]
