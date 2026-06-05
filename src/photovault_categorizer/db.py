"""
Database access layer — thin wrappers around psycopg3.

All connections use autocommit=True.  Atomic multi-statement work is wrapped
by the caller in `with conn.transaction():` blocks.
"""

import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

import psycopg

from .config import Config, FACE_MODEL_ID, MODEL_ID

# Map kind string to the DB table name.
_KIND_TABLE = {"tag": "tags", "category": "categories"}


@contextmanager
def get_connection(config: Config) -> Generator[psycopg.Connection, None, None]:
    """Yields a psycopg3 connection and closes it on exit."""
    conn = psycopg.connect(
        host=config.db_host,
        port=config.db_port,
        dbname=config.db_name,
        user=config.db_user,
        password=config.db_password,
        autocommit=True,
    )
    try:
        yield conn
    finally:
        conn.close()


def fetch_delta_queue(conn: psycopg.Connection) -> list[tuple[str, str]]:
    """Returns (photo_id, medium_path) rows pending categorization.

    Rows where medium_path is NULL are excluded — the asset was not written yet.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, medium_path FROM photos"
            " WHERE processing_status = 'pending_categorization'"
            "   AND medium_path IS NOT NULL"
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


def fetch_auto_enabled_labels(conn: psycopg.Connection) -> list[tuple[str, str, str]]:
    """Returns (id, name, kind) for all auto-enabled and fully rolled-out labels.

    kind is 'tag' or 'category'.
    """
    results: list[tuple[str, str, str]] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name FROM tags WHERE auto_enabled = true AND rolled_out = true"
        )
        for row in cur.fetchall():
            results.append((row[0], row[1], "tag"))

        cur.execute(
            "SELECT id, name FROM categories WHERE auto_enabled = true AND rolled_out = true"
        )
        for row in cur.fetchall():
            results.append((row[0], row[1], "category"))

    return results


def fetch_all_photo_ids(conn: psycopg.Connection) -> list[str]:
    """Returns all photo IDs currently in the database (used for vector store pruning)."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM photos")
        return [row[0] for row in cur.fetchall()]


# ── Iteration 2 helpers ───────────────────────────────────────────────────────


def fetch_embed_backlog(conn: psycopg.Connection) -> list[tuple[str, str]]:
    """Returns (photo_id, medium_path) for photos that need (re-)embedding.

    A photo is in the backlog when:
      - embedded_at IS NULL, or
      - embedding_model differs from the current MODEL_ID (model bump).

    Photos with medium_path IS NULL are excluded — the asset is not written yet.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, medium_path FROM photos"
            " WHERE medium_path IS NOT NULL"
            "   AND (embedded_at IS NULL OR embedding_model IS DISTINCT FROM %s)",
            (MODEL_ID,),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


def mark_embedded(
    conn: psycopg.Connection,
    photo_ids: list[str],
    model_id: str,
    now: datetime,
) -> None:
    """Batch-update embedded_at and embedding_model for a list of photo IDs."""
    if not photo_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE photos SET embedded_at = %s, embedding_model = %s"
            " WHERE id = ANY(%s)",
            (now, model_id, photo_ids),
        )


def fetch_ready_with_medium(conn: psycopg.Connection) -> list[tuple[str, str]]:
    """Returns (photo_id, medium_path) for all photos that are in 'ready' status.

    Used by the self-heal pass to find photos whose vector was lost from the store.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, medium_path FROM photos"
            " WHERE processing_status = 'ready'"
            "   AND medium_path IS NOT NULL"
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


def fetch_label(
    conn: psycopg.Connection,
    label_id: str,
    kind: str,
) -> tuple[str, str, bool, bool] | None:
    """Fetch one tag or category row by ID.

    Returns (id, name, auto_enabled, rolled_out), or None if not found.
    kind must be 'tag' or 'category'.
    """
    table = _KIND_TABLE[kind]
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, name, auto_enabled, rolled_out FROM {table} WHERE id = %s",  # noqa: S608
            (label_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return (row[0], row[1], bool(row[2]), bool(row[3]))


def set_rolled_out(
    conn: psycopg.Connection,
    label_id: str,
    kind: str,
    value: bool,
) -> None:
    """Flip the rolled_out flag on a tag or category row."""
    table = _KIND_TABLE[kind]
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET rolled_out = %s WHERE id = %s",  # noqa: S608
            (value, label_id),
        )


# ── Phase 2 — face detection helpers ─────────────────────────────────────────


def fetch_face_detection_backlog(conn: psycopg.Connection) -> list[tuple[str, str]]:
    """Returns (photo_id, medium_path) for photos that need (re-)face-detection.

    A photo is in the backlog when:
      - faces_detected_at IS NULL (never processed), or
      - face_detection_model differs from the current FACE_MODEL_ID (model bump).

    Photos with medium_path IS NULL are excluded — the asset is not written yet.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, medium_path FROM photos"
            " WHERE medium_path IS NOT NULL"
            "   AND (faces_detected_at IS NULL"
            "        OR face_detection_model IS DISTINCT FROM %s)",
            (FACE_MODEL_ID,),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]


def insert_faces(
    conn: psycopg.Connection,
    photo_id: str,
    detections: list,
    run_id: str,
    model_id: str = FACE_MODEL_ID,
    now: datetime | None = None,
) -> list[str]:
    """Insert face rows for one photo and return the generated face ids.

    Args:
        conn:       psycopg3 connection (autocommit=True; caller wraps in transaction).
        photo_id:   photo-<uuid> string.
        detections: list of FaceDetection (from face_model.detect_and_embed).
        run_id:     embedding run identifier for auditability.
        model_id:   face model id string (default = current FACE_MODEL_ID).
        now:        timestamp to use for detected_at (defaults to utcnow).

    Returns:
        List of generated face-id strings, aligned with *detections*.

    Notes:
        - Existing face rows for this photo (from a previous detection run or model bump)
          are deleted first so the set stays consistent with the current detection pass.
        - This must be called inside a transaction so the delete+insert is atomic.
    """
    from datetime import timezone

    if now is None:
        now = datetime.now(timezone.utc)

    face_ids: list[str] = []

    with conn.cursor() as cur:
        # Remove stale rows from previous detection runs for this photo.
        cur.execute("DELETE FROM faces WHERE photo_id = %s", (photo_id,))

        for det in detections:
            face_id = f"face-{uuid.uuid4()}"
            cur.execute(
                "INSERT INTO faces"
                "  (id, photo_id, bbox_x, bbox_y, bbox_w, bbox_h,"
                "   det_score, face_model, embedding_run, detected_at)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    face_id,
                    photo_id,
                    det.bbox_x,
                    det.bbox_y,
                    det.bbox_w,
                    det.bbox_h,
                    det.det_score,
                    model_id,
                    run_id,
                    now,
                ),
            )
            face_ids.append(face_id)

    return face_ids


def mark_faces_detected(
    conn: psycopg.Connection,
    photo_ids: list[str],
    model_id: str,
    now: datetime,
) -> None:
    """Batch-update faces_detected_at and face_detection_model for a list of photo IDs."""
    if not photo_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE photos SET faces_detected_at = %s, face_detection_model = %s"
            " WHERE id = ANY(%s)",
            (now, model_id, photo_ids),
        )


def fetch_all_face_ids(conn: psycopg.Connection) -> list[str]:
    """Returns all face IDs currently in the database (used for face vector store pruning)."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM faces")
        return [row[0] for row in cur.fetchall()]


def fetch_labelled_cluster_faces(conn: psycopg.Connection) -> list[tuple[str, str, str]]:
    """Returns (face_id, label_id, kind) for all faces that belong to labelled clusters.

    A cluster is labelled when it has a tag_id or category_id set via the admin API.
    Results are used by face_match.build_identity_prototypes() to build per-person
    prototype vectors for identity matching.

    kind is 'tag' or 'category'.
    """
    results: list[tuple[str, str, str]] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT f.id, fc.tag_id"
            " FROM faces f"
            " JOIN face_clusters fc ON f.cluster_id = fc.id"
            " WHERE fc.tag_id IS NOT NULL"
        )
        for row in cur.fetchall():
            results.append((row[0], row[1], "tag"))

        cur.execute(
            "SELECT f.id, fc.category_id"
            " FROM faces f"
            " JOIN face_clusters fc ON f.cluster_id = fc.id"
            " WHERE fc.category_id IS NOT NULL"
        )
        for row in cur.fetchall():
            results.append((row[0], row[1], "category"))

    return results


def fetch_unclustered_faces(conn: psycopg.Connection) -> list[tuple[str, float]]:
    """Returns (face_id, det_score) for all faces not yet assigned to a cluster.

    Sorted by det_score descending so the first result is always the highest-confidence
    detection (useful when picking a cluster representative).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, det_score FROM faces WHERE cluster_id IS NULL ORDER BY det_score DESC"
        )
        return [(row[0], float(row[1])) for row in cur.fetchall()]


def insert_face_cluster(
    conn: psycopg.Connection,
    cluster_id: str,
    face_count: int,
    representative_face_id: str,
    now: datetime,
) -> None:
    """Insert a new face_clusters row."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO face_clusters (id, face_count, representative_face_id, created_at)"
            " VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (id) DO UPDATE"
            "   SET face_count = EXCLUDED.face_count,"
            "       representative_face_id = EXCLUDED.representative_face_id",
            (cluster_id, face_count, representative_face_id, now),
        )


def update_faces_cluster_id(
    conn: psycopg.Connection,
    face_ids: list[str],
    cluster_id: str,
) -> None:
    """Assign a cluster to a batch of face rows."""
    if not face_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE faces SET cluster_id = %s WHERE id = ANY(%s)",
            (cluster_id, face_ids),
        )
