"""Tests for score_photo — tag threshold, category top-k/min-score edge cases."""

import numpy as np
import pytest

from photovault_categorizer.scoring import score_photo


def _unit(idx: int, dim: int = 512) -> np.ndarray:
    """One-hot unit vector at position *idx*."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


# ── Tag threshold ────────────────────────────────────────────────────────────


def test_tag_above_threshold_is_returned():
    tag_hits, _ = score_photo(
        _unit(0), {"tag-sea": _unit(0)}, {}, tag_threshold=0.25, category_top_k=1, category_min_score=0.0
    )
    assert len(tag_hits) == 1
    assert tag_hits[0][0] == "tag-sea"
    assert tag_hits[0][1] == pytest.approx(1.0)


def test_tag_below_threshold_is_excluded():
    tag_hits, _ = score_photo(
        _unit(0), {"tag-sea": _unit(1)}, {}, tag_threshold=0.25, category_top_k=1, category_min_score=0.0
    )
    assert tag_hits == []


def test_tag_exactly_at_threshold_is_included():
    # Build a photo vec and a proto whose dot product equals threshold exactly.
    threshold = 0.25
    photo_vec = np.zeros(512, dtype=np.float32)
    photo_vec[0] = threshold
    photo_vec[1] = (1.0 - threshold**2) ** 0.5  # normalise
    proto = _unit(0)
    tag_hits, _ = score_photo(
        photo_vec, {"tag-x": proto}, {}, tag_threshold=threshold, category_top_k=1, category_min_score=0.0
    )
    assert len(tag_hits) == 1


def test_multiple_tags_can_be_assigned():
    protos = {"tag-a": _unit(0), "tag-b": _unit(1), "tag-c": _unit(2)}
    # photo vec is equal weight across dims 0 and 1 only
    photo_vec = np.zeros(512, dtype=np.float32)
    photo_vec[0] = 1.0 / 2**0.5
    photo_vec[1] = 1.0 / 2**0.5
    tag_hits, _ = score_photo(photo_vec, protos, {}, 0.5, 1, 0.0)
    hit_ids = {h[0] for h in tag_hits}
    assert "tag-a" in hit_ids
    assert "tag-b" in hit_ids
    assert "tag-c" not in hit_ids


# ── Category top-k / min-score ────────────────────────────────────────────────


def test_category_top_1_returns_best():
    cat_protos = {
        "cat-nature": _unit(0),  # sim = 1.0 (best)
        "cat-city": _unit(1),    # sim = 0.0
    }
    _, cat_hits = score_photo(_unit(0), {}, cat_protos, 0.25, 1, 0.0)
    assert len(cat_hits) == 1
    assert cat_hits[0][0] == "cat-nature"


def test_category_top_k_limits_results():
    cat_protos = {f"cat-{i}": _unit(i) for i in range(5)}
    photo_vec = np.zeros(512, dtype=np.float32)
    photo_vec[:3] = 1.0 / 3**0.5  # non-zero similarity for first 3
    _, cat_hits = score_photo(photo_vec, {}, cat_protos, 0.25, 2, 0.0)
    assert len(cat_hits) <= 2


def test_category_min_score_filters_low_scores():
    cat_protos = {"cat-maybe": _unit(1)}  # photo is _unit(0) → sim = 0.0
    _, cat_hits = score_photo(_unit(0), {}, cat_protos, 0.25, 1, 0.1)
    assert cat_hits == []


# ── Empty inputs ─────────────────────────────────────────────────────────────


def test_empty_prototypes_return_empty_hits():
    tag_hits, cat_hits = score_photo(_unit(0), {}, {}, 0.25, 1, 0.0)
    assert tag_hits == []
    assert cat_hits == []
