"""
Detect vacation / trip events in the photo library.

Algorithm:
  1. Fetch all photos with captured_at set (GPS optional but needed for heuristic).
  2. Sort by captured_at and split into temporal sessions (gap > EVENT_GAP_HOURS).
  3. For each session apply the trip heuristic:
       - at least EVENT_MIN_PHOTOS photos
       - spans at least EVENT_MIN_DAYS distinct calendar days
       - centroid of GPS-tagged photos is > HOME_RADIUS_KM from home
  4. For each passing session:
       a. Create a category row in the DB (idempotent by name) with auto_enabled=true
          and rolled_out=true (backfill is done inline).
       b. Write source='auto' photo_categories rows for every photo in the session
          via assign_auto (manual/denied precedence preserved).
  5. Log a one-line summary.

Usage:
    python -m photovault_categorizer.cli.detect_events

All configuration via environment variables — see .env.example.
HOME_LAT and HOME_LNG are required; the script exits with an error if missing.
"""

import logging
import sys
from datetime import datetime, timezone

from ..config import Config
from ..db import create_event_category, fetch_photos_with_location, get_connection
from ..event_clustering import (
    cluster_by_time,
    passes_heuristic,
    pick_category_color,
    suggest_category_name,
)
from ..lock import with_lock
from ..write import assign_auto

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    with with_lock():
        _run()


def _run() -> None:
    config = Config.from_env()

    if config.home_lat is None or config.home_lng is None:
        log.error("HOME_LAT and HOME_LNG must be set in the environment — aborting")
        sys.exit(1)

    with get_connection(config) as conn:
        photos = fetch_photos_with_location(conn)

    log.info("Fetched %d photos with captured_at set", len(photos))
    if not photos:
        log.info("No timestamped photos found — nothing to do")
        return

    clusters = cluster_by_time(photos, gap_hours=config.event_gap_hours)
    log.info(
        "Temporal clustering: %d session(s) from %d photo(s) (gap threshold: %.1f h)",
        len(clusters),
        len(photos),
        config.event_gap_hours,
    )

    trip_clusters = [
        c for c in clusters
        if passes_heuristic(
            c,
            home_lat=config.home_lat,
            home_lng=config.home_lng,
            home_radius_km=config.home_radius_km,
            min_photos=config.event_min_photos,
            min_days=config.event_min_days,
        )
    ]
    log.info(
        "%d trip(s) pass heuristic (home_radius=%.0f km, min_photos=%d, min_days=%d)",
        len(trip_clusters),
        config.home_radius_km,
        config.event_min_photos,
        config.event_min_days,
    )

    categories_created = 0
    photos_assigned = 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "/events"

    with get_connection(config) as conn:
        for cluster in trip_clusters:
            name = suggest_category_name(cluster)
            color = pick_category_color(name)
            category_id, created = create_event_category(conn, name, color)
            if created:
                categories_created += 1
                log.info("Created category %r (%s, %s)", name, category_id, color)
            else:
                log.info("Category %r already exists (%s)", name, category_id)

            assigned_in_cluster = 0
            with conn.transaction():
                for photo in cluster:
                    if assign_auto(conn, photo.photo_id, category_id, "category", 1.0, run_id):
                        assigned_in_cluster += 1
            photos_assigned += assigned_in_cluster
            log.info(
                "  %s: %d photo(s) assigned (%d in cluster)",
                name,
                assigned_in_cluster,
                len(cluster),
            )

    log.info(
        "Done: %d trip(s) found, %d new category/ies created, %d photo(s) assigned",
        len(trip_clusters),
        categories_created,
        photos_assigned,
    )


if __name__ == "__main__":
    main()
