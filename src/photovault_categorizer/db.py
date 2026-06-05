"""
Database access layer — thin wrappers around psycopg3.

All connections use autocommit=True.  Atomic multi-statement work is wrapped
by the caller in `with conn.transaction():` blocks.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Generator

import psycopg

from .config import Config, MODEL_ID

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
