"""
Unit tests for event_clustering — all pure in-memory, no DB, no filesystem.
"""

from datetime import datetime, timedelta, timezone

import pytest

from photovault_categorizer.event_clustering import (
    PhotoRecord,
    cluster_by_time,
    cluster_centroid,
    haversine_km,
    passes_heuristic,
    pick_category_color,
    suggest_category_name,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_BASE = datetime(2026, 5, 10, 10, 0, tzinfo=timezone.utc)


def _photo(
    hours_offset: float,
    lat: float | None = 54.0,
    lng: float | None = 18.0,
    place: str | None = "Sopot",
    photo_id: str = "photo-x",
) -> PhotoRecord:
    return PhotoRecord(
        photo_id=photo_id,
        captured_at=_BASE + timedelta(hours=hours_offset),
        lat=lat,
        lng=lng,
        place_name=place,
    )


# ── cluster_by_time ───────────────────────────────────────────────────────────

def test_empty_input_returns_empty():
    assert cluster_by_time([], gap_hours=6.0) == []


def test_single_photo_is_one_cluster():
    result = cluster_by_time([_photo(0)], gap_hours=6.0)
    assert len(result) == 1
    assert len(result[0]) == 1


def test_two_close_photos_stay_together():
    photos = [_photo(0), _photo(2)]
    result = cluster_by_time(photos, gap_hours=6.0)
    assert len(result) == 1
    assert len(result[0]) == 2


def test_photos_split_on_gap():
    photos = [_photo(0), _photo(1), _photo(10), _photo(11)]
    result = cluster_by_time(photos, gap_hours=6.0)
    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 2


def test_photos_sorted_by_time():
    photos = [_photo(5), _photo(0), _photo(3)]  # unsorted
    result = cluster_by_time(photos, gap_hours=6.0)
    assert len(result) == 1
    times = [p.captured_at for p in result[0]]
    assert times == sorted(times)


def test_gap_exactly_at_boundary_stays_together():
    # gap_hours=6 means a gap of exactly 6h does NOT split (must be strictly >)
    photos = [_photo(0), _photo(6)]
    result = cluster_by_time(photos, gap_hours=6.0)
    assert len(result) == 1


def test_gap_just_over_boundary_splits():
    photos = [_photo(0), _photo(6.01)]
    result = cluster_by_time(photos, gap_hours=6.0)
    assert len(result) == 2


# ── haversine_km ──────────────────────────────────────────────────────────────

def test_haversine_same_point_is_zero():
    assert haversine_km(54.0, 18.0, 54.0, 18.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance():
    # Gdańsk → Warsaw is roughly 300 km
    dist = haversine_km(54.35, 18.65, 52.23, 21.01)
    assert 280 < dist < 340


def test_haversine_symmetric():
    d1 = haversine_km(54.0, 18.0, 52.0, 21.0)
    d2 = haversine_km(52.0, 21.0, 54.0, 18.0)
    assert d1 == pytest.approx(d2, rel=1e-6)


# ── cluster_centroid ──────────────────────────────────────────────────────────

def test_centroid_no_gps_returns_none():
    photos = [_photo(0, lat=None, lng=None), _photo(1, lat=None, lng=None)]
    assert cluster_centroid(photos) is None


def test_centroid_single_photo():
    photos = [_photo(0, lat=54.0, lng=18.0)]
    lat, lng = cluster_centroid(photos)
    assert lat == pytest.approx(54.0)
    assert lng == pytest.approx(18.0)


def test_centroid_averages_correctly():
    photos = [_photo(0, lat=54.0, lng=18.0), _photo(1, lat=56.0, lng=20.0)]
    lat, lng = cluster_centroid(photos)
    assert lat == pytest.approx(55.0)
    assert lng == pytest.approx(19.0)


def test_centroid_ignores_photos_without_gps():
    photos = [
        _photo(0, lat=54.0, lng=18.0),
        _photo(1, lat=None, lng=None),  # no GPS
    ]
    lat, lng = cluster_centroid(photos)
    assert lat == pytest.approx(54.0)
    assert lng == pytest.approx(18.0)


# ── passes_heuristic ──────────────────────────────────────────────────────────

HOME = (54.35, 18.65)  # Gdańsk


def _trip_cluster(n: int = 15, days: int = 3) -> list[PhotoRecord]:
    """Build a cluster far from home (~Warsaw coords), spanning `days` calendar days."""
    return [
        _photo(
            hours_offset=i * (24 * days / n),
            lat=52.23,
            lng=21.01,
            place="Warsaw",
            photo_id=f"photo-{i}",
        )
        for i in range(n)
    ]


def test_passes_basic_trip():
    cluster = _trip_cluster(n=15, days=3)
    assert passes_heuristic(cluster, *HOME, home_radius_km=25, min_photos=10, min_days=2)


def test_fails_too_few_photos():
    cluster = _trip_cluster(n=5, days=3)
    assert not passes_heuristic(cluster, *HOME, home_radius_km=25, min_photos=10, min_days=2)


def test_fails_single_day():
    # All photos within the same calendar day
    cluster = [
        _photo(hours_offset=i * 0.5, lat=52.23, lng=21.01, photo_id=f"p{i}")
        for i in range(15)
    ]
    assert not passes_heuristic(cluster, *HOME, home_radius_km=25, min_photos=10, min_days=2)


def test_fails_too_close_to_home():
    # Photos near home (Gdańsk area)
    cluster = [
        _photo(hours_offset=i * 8, lat=54.4, lng=18.7, photo_id=f"p{i}")
        for i in range(15)
    ]
    # Force them to span 3 days
    from photovault_categorizer.event_clustering import PhotoRecord
    cluster = [
        PhotoRecord(
            photo_id=f"p{i}",
            captured_at=_BASE + timedelta(days=i % 3, hours=i),
            lat=54.4,
            lng=18.7,
            place_name="Gdańsk",
        )
        for i in range(15)
    ]
    assert not passes_heuristic(cluster, *HOME, home_radius_km=25, min_photos=10, min_days=2)


def test_fails_no_gps():
    cluster = [
        PhotoRecord(
            photo_id=f"p{i}",
            captured_at=_BASE + timedelta(days=i % 3, hours=i),
            lat=None,
            lng=None,
            place_name=None,
        )
        for i in range(15)
    ]
    assert not passes_heuristic(cluster, *HOME, home_radius_km=25, min_photos=10, min_days=2)


# ── suggest_category_name ────────────────────────────────────────────────────

def test_name_with_place():
    cluster = [_photo(i, place="Sopot") for i in range(5)]
    assert suggest_category_name(cluster) == "Trip · Sopot · 2026-05"


def test_name_picks_most_common_place():
    cluster = (
        [_photo(i, place="Sopot", photo_id=f"p{i}") for i in range(3)]
        + [_photo(i + 3, place="Gdańsk", photo_id=f"q{i}") for i in range(2)]
    )
    assert suggest_category_name(cluster) == "Trip · Sopot · 2026-05"


def test_name_without_place_falls_back_to_month():
    cluster = [_photo(i, place=None, photo_id=f"p{i}") for i in range(5)]
    assert suggest_category_name(cluster) == "Trip · 2026-05"


def test_name_uses_earliest_month():
    cluster = [
        _photo(0, place="Sopot", photo_id="a"),                        # May 2026
        _photo(24 * 35, place="Sopot", photo_id="b"),                  # June 2026
    ]
    name = suggest_category_name(cluster)
    assert "2026-05" in name


# ── pick_category_color ───────────────────────────────────────────────────────

def test_color_is_deterministic():
    assert pick_category_color("Trip · Sopot · 2026-05") == pick_category_color("Trip · Sopot · 2026-05")


def test_different_names_may_differ():
    c1 = pick_category_color("Trip · Sopot · 2026-05")
    c2 = pick_category_color("Trip · Warsaw · 2026-08")
    # Not guaranteed to differ (palette has 8 slots) but a sanity check that the
    # function returns a valid hex color regardless.
    assert c1.startswith("#") and len(c1) == 7
    assert c2.startswith("#") and len(c2) == 7
