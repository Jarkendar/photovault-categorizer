"""
Pure event-clustering logic — no DB, no filesystem I/O.

All functions operate on PhotoRecord objects and return plain Python values
so they can be unit-tested without any external dependencies.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import atan2, cos, radians, sin, sqrt
from typing import Optional

_TRIP_COLORS = [
    "#1E88E5",  # blue
    "#43A047",  # green
    "#FB8C00",  # orange
    "#8E24AA",  # purple
    "#E53935",  # red
    "#00ACC1",  # cyan
    "#F4511E",  # deep orange
    "#6D4C41",  # brown
]


@dataclass
class PhotoRecord:
    photo_id: str
    captured_at: datetime
    lat: Optional[float]
    lng: Optional[float]
    place_name: Optional[str]


def cluster_by_time(
    photos: list[PhotoRecord],
    gap_hours: float,
) -> list[list[PhotoRecord]]:
    """Split photos into sessions; a gap > gap_hours between consecutive shots starts a new one."""
    if not photos:
        return []
    sorted_photos = sorted(photos, key=lambda p: p.captured_at)
    clusters: list[list[PhotoRecord]] = [[sorted_photos[0]]]
    for photo in sorted_photos[1:]:
        gap = (photo.captured_at - clusters[-1][-1].captured_at).total_seconds() / 3600
        if gap > gap_hours:
            clusters.append([])
        clusters[-1].append(photo)
    return clusters


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometres between two lat/lng points."""
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return r * 2 * atan2(sqrt(a), sqrt(1.0 - a))


def cluster_centroid(
    cluster: list[PhotoRecord],
) -> tuple[float, float] | None:
    """Mean lat/lng of photos in the cluster that have GPS data. None if none have GPS."""
    gps = [(p.lat, p.lng) for p in cluster if p.lat is not None and p.lng is not None]
    if not gps:
        return None
    return (
        sum(lat for lat, _ in gps) / len(gps),
        sum(lng for _, lng in gps) / len(gps),
    )


def passes_heuristic(
    cluster: list[PhotoRecord],
    home_lat: float,
    home_lng: float,
    home_radius_km: float,
    min_photos: int,
    min_days: int,
) -> bool:
    """Return True if the cluster looks like a trip away from home.

    All three conditions must hold:
    - at least min_photos photos in the cluster
    - spans at least min_days distinct calendar days
    - centroid of GPS-tagged photos is more than home_radius_km from home
      (clusters with no GPS data are skipped — we cannot verify location)
    """
    if len(cluster) < min_photos:
        return False

    dates = {p.captured_at.date() for p in cluster}
    if (max(dates) - min(dates)).days < min_days - 1:
        return False

    centroid = cluster_centroid(cluster)
    if centroid is None:
        return False
    lat, lng = centroid
    if haversine_km(lat, lng, home_lat, home_lng) < home_radius_km:
        return False

    return True


def suggest_category_name(cluster: list[PhotoRecord]) -> str:
    """Generate a human-readable category name like 'Trip · Sopot · 2026-05'."""
    month_str = min(p.captured_at for p in cluster).strftime("%Y-%m")
    place_names = [p.place_name for p in cluster if p.place_name]
    if place_names:
        most_common = Counter(place_names).most_common(1)[0][0]
        return f"Trip · {most_common} · {month_str}"
    return f"Trip · {month_str}"


def pick_category_color(name: str) -> str:
    """Deterministic color for a trip category, stable across re-runs."""
    idx = int(hashlib.md5(name.encode()).hexdigest(), 16) % len(_TRIP_COLORS)
    return _TRIP_COLORS[idx]
