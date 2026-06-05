"""
Database access layer — thin wrappers around psycopg3.

All connections use autocommit=True.  Atomic multi-statement work is wrapped
by the caller in `with conn.transaction():` blocks.
"""

from contextlib import contextmanager
from typing import Generator

import psycopg

from .config import Config


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
