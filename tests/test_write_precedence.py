"""
Integration tests for assign_auto() source-precedence rules.

Requires a live PostgreSQL instance matching the DB_URL environment variable
(the same database used by the PhotoVault server).  Tests are automatically
skipped if the connection cannot be established.

The fixture inserts isolated test rows and cleans them up after each test,
so it is safe to run against the development database.
"""

import pytest
import psycopg

from photovault_categorizer.config import Config
from photovault_categorizer.db import get_connection
from photovault_categorizer.write import assign_auto

# ── Shared IDs (unique enough to avoid collisions with real data) ────────────

_PHOTO_ID = "photo-test-write-prec-zz9"
_TAG_ID = "tag-test-write-prec-zz9"
_USER_ID = "user-admin"  # seeded by DatabaseInit.kt
_RUN_A = "run-20240101T000000Z/mobileclip-s2-datacompdr"
_RUN_B = "run-20240102T000000Z/mobileclip-s2-datacompdr"


@pytest.fixture(scope="module")
def conn():
    """Module-scoped connection — skips the whole module if DB is unreachable."""
    config = Config.from_env()
    try:
        with get_connection(config) as c:
            # Smoke-check that the schema exists
            with c.cursor() as cur:
                cur.execute("SELECT 1 FROM photos LIMIT 0")
            yield c
    except Exception as exc:
        pytest.skip(f"Database unavailable — skipping integration tests ({exc})")


@pytest.fixture(autouse=True)
def test_rows(conn):
    """Insert a minimal photo + tag for each test, clean up after."""
    with conn.cursor() as cur:
        # Ensure the admin user exists (idempotent)
        cur.execute(
            "INSERT INTO users (id, username, display_name, password_hash, created_at)"
            " VALUES (%s, 'admin', 'Admin', 'x', NOW()) ON CONFLICT (id) DO NOTHING",
            (_USER_ID,),
        )
        # Photo
        cur.execute(
            "INSERT INTO photos"
            " (id, name, size_bytes, mime_type, width, height, uploaded_at, uploaded_by, processing_status)"
            " VALUES (%s, 'prec.jpg', 1, 'image/jpeg', 1, 1, NOW(), %s, 'pending_categorization')"
            " ON CONFLICT (id) DO NOTHING",
            (_PHOTO_ID, _USER_ID),
        )
        # Tag
        cur.execute(
            "INSERT INTO tags (id, name, auto_enabled, rolled_out)"
            " VALUES (%s, '#test-prec-zz9', true, true) ON CONFLICT (id) DO NOTHING",
            (_TAG_ID,),
        )
        # Clear any leftover junction rows from previous runs
        cur.execute(
            "DELETE FROM photo_tags WHERE photo_id = %s AND tag_id = %s",
            (_PHOTO_ID, _TAG_ID),
        )

    yield

    # Cleanup
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM photo_tags WHERE photo_id = %s AND tag_id = %s",
            (_PHOTO_ID, _TAG_ID),
        )


def _get_row(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT source, score, embedding_run FROM photo_tags"
            " WHERE photo_id = %s AND tag_id = %s",
            (_PHOTO_ID, _TAG_ID),
        )
        return cur.fetchone()


# ── Tests ────────────────────────────────────────────────────────────────────


def test_auto_insert_creates_row(conn):
    written = assign_auto(conn, _PHOTO_ID, _TAG_ID, "tag", 0.8, _RUN_A)
    assert written is True
    row = _get_row(conn)
    assert row is not None
    assert row[0] == "auto"
    assert abs(row[1] - 0.8) < 1e-6
    assert row[2] == _RUN_A


def test_auto_rescore_updates_existing_auto(conn):
    assign_auto(conn, _PHOTO_ID, _TAG_ID, "tag", 0.8, _RUN_A)
    written = assign_auto(conn, _PHOTO_ID, _TAG_ID, "tag", 0.95, _RUN_B)
    assert written is True
    row = _get_row(conn)
    assert abs(row[1] - 0.95) < 1e-6
    assert row[2] == _RUN_B


def test_manual_row_is_not_overwritten(conn):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO photo_tags (photo_id, tag_id, source) VALUES (%s, %s, 'manual')",
            (_PHOTO_ID, _TAG_ID),
        )
    written = assign_auto(conn, _PHOTO_ID, _TAG_ID, "tag", 0.9, _RUN_A)
    assert written is False
    row = _get_row(conn)
    assert row[0] == "manual"  # source unchanged


def test_denied_row_is_not_overwritten(conn):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO photo_tags (photo_id, tag_id, source) VALUES (%s, %s, 'denied')",
            (_PHOTO_ID, _TAG_ID),
        )
    written = assign_auto(conn, _PHOTO_ID, _TAG_ID, "tag", 0.9, _RUN_A)
    assert written is False
    row = _get_row(conn)
    assert row[0] == "denied"  # tombstone preserved
