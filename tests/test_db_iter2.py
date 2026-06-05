"""
Integration tests for the Iteration 2 DB helpers.

Requires a live PostgreSQL instance matching the DB_URL environment variable.
Tests are automatically skipped if the connection cannot be established.

Each test inserts isolated rows identified by unique sentinel IDs and cleans
them up after the test, so it is safe to run against the development database.
"""

import pytest
from datetime import datetime, timezone

from photovault_categorizer.config import Config, MODEL_ID
from photovault_categorizer.db import (
    fetch_embed_backlog,
    fetch_label,
    fetch_ready_with_medium,
    get_connection,
    mark_embedded,
    set_rolled_out,
)

# ── Sentinel IDs ─────────────────────────────────────────────────────────────

_USER_ID = "user-admin"
_PHOTO_A = "photo-test-iter2-a-zz9"  # embedded_at IS NULL → in backlog
_PHOTO_B = "photo-test-iter2-b-zz9"  # embedded with current MODEL_ID → not in backlog
_PHOTO_C = "photo-test-iter2-c-zz9"  # ready + has medium_path → for self-heal
_TAG_ID = "tag-test-iter2-zz9"
_CAT_ID = "cat-test-iter2-zz9"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def conn():
    """Module-scoped connection — skips entire module if DB is unreachable."""
    config = Config.from_env()
    try:
        with get_connection(config) as c:
            with c.cursor() as cur:
                cur.execute("SELECT 1 FROM photos LIMIT 0")
            yield c
    except Exception as exc:
        pytest.skip(f"Database unavailable — skipping integration tests ({exc})")


@pytest.fixture(autouse=True)
def seed_rows(conn):
    """Insert minimal test rows, yield for the test, then clean up."""
    with conn.cursor() as cur:
        # Ensure admin user exists (idempotent)
        cur.execute(
            "INSERT INTO users (id, username, display_name, password_hash, created_at)"
            " VALUES (%s, 'admin', 'Admin', 'x', NOW()) ON CONFLICT (id) DO NOTHING",
            (_USER_ID,),
        )

        def _insert_photo(pid, status, embedded_model=None):
            cur.execute(
                "INSERT INTO photos"
                " (id, name, size_bytes, mime_type, width, height,"
                "  uploaded_at, uploaded_by, processing_status,"
                "  medium_path, embedded_at, embedding_model)"
                " VALUES (%s, %s, 1, 'image/jpeg', 1, 1, NOW(), %s, %s, %s, %s, %s)"
                " ON CONFLICT (id) DO NOTHING",
                (
                    pid,
                    f"{pid}.jpg",
                    _USER_ID,
                    status,
                    f"photos/{pid}/medium.jpg",
                    datetime.now(timezone.utc) if embedded_model else None,
                    embedded_model,
                ),
            )

        _insert_photo(_PHOTO_A, "pending_categorization", embedded_model=None)
        _insert_photo(_PHOTO_B, "ready", embedded_model=MODEL_ID)
        _insert_photo(_PHOTO_C, "ready", embedded_model=MODEL_ID)

        # Tag
        cur.execute(
            "INSERT INTO tags (id, name, auto_enabled, rolled_out)"
            " VALUES (%s, '#test-iter2-zz9', true, true) ON CONFLICT (id) DO NOTHING",
            (_TAG_ID,),
        )
        # Category
        cur.execute(
            "INSERT INTO categories (id, name, color_hex, auto_enabled, rolled_out)"
            " VALUES (%s, 'Test Iter2', '#aabbcc', true, true) ON CONFLICT (id) DO NOTHING",
            (_CAT_ID,),
        )

    yield

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM photos WHERE id = ANY(%s)",
            ([_PHOTO_A, _PHOTO_B, _PHOTO_C],),
        )
        cur.execute("DELETE FROM tags WHERE id = %s", (_TAG_ID,))
        cur.execute("DELETE FROM categories WHERE id = %s", (_CAT_ID,))


# ── fetch_embed_backlog ───────────────────────────────────────────────────────


def test_fetch_embed_backlog_includes_unembedded_photo(conn):
    rows = fetch_embed_backlog(conn)
    ids = [r[0] for r in rows]
    assert _PHOTO_A in ids, "Photo with embedded_at=NULL must be in the backlog"


def test_fetch_embed_backlog_excludes_already_embedded(conn):
    rows = fetch_embed_backlog(conn)
    ids = [r[0] for r in rows]
    assert _PHOTO_B not in ids, "Photo already embedded with current MODEL_ID must be excluded"


def test_fetch_embed_backlog_row_has_medium_path(conn):
    rows = fetch_embed_backlog(conn)
    row_map = {r[0]: r[1] for r in rows}
    assert row_map.get(_PHOTO_A) == f"photos/{_PHOTO_A}/medium.jpg"


# ── mark_embedded ─────────────────────────────────────────────────────────────


def test_mark_embedded_updates_columns(conn):
    now = datetime.now(timezone.utc)
    mark_embedded(conn, [_PHOTO_A], MODEL_ID, now)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedded_at, embedding_model FROM photos WHERE id = %s", (_PHOTO_A,)
        )
        row = cur.fetchone()

    assert row is not None
    assert row[1] == MODEL_ID


def test_mark_embedded_noop_for_empty_list(conn):
    # Should not raise
    mark_embedded(conn, [], MODEL_ID, datetime.now(timezone.utc))


# ── fetch_ready_with_medium ───────────────────────────────────────────────────


def test_fetch_ready_with_medium_includes_ready_photos(conn):
    rows = fetch_ready_with_medium(conn)
    ids = [r[0] for r in rows]
    assert _PHOTO_C in ids


def test_fetch_ready_with_medium_excludes_pending(conn):
    rows = fetch_ready_with_medium(conn)
    ids = [r[0] for r in rows]
    assert _PHOTO_A not in ids


# ── fetch_label ───────────────────────────────────────────────────────────────


def test_fetch_label_tag_found(conn):
    result = fetch_label(conn, _TAG_ID, "tag")
    assert result is not None
    label_id, name, auto_enabled, rolled_out = result
    assert label_id == _TAG_ID
    assert auto_enabled is True
    assert rolled_out is True


def test_fetch_label_category_found(conn):
    result = fetch_label(conn, _CAT_ID, "category")
    assert result is not None
    assert result[0] == _CAT_ID


def test_fetch_label_returns_none_for_unknown_id(conn):
    assert fetch_label(conn, "tag-does-not-exist-zzz", "tag") is None


# ── set_rolled_out ────────────────────────────────────────────────────────────


def test_set_rolled_out_flips_to_false(conn):
    set_rolled_out(conn, _TAG_ID, "tag", False)
    row = fetch_label(conn, _TAG_ID, "tag")
    assert row is not None
    assert row[3] is False  # rolled_out


def test_set_rolled_out_flips_back_to_true(conn):
    set_rolled_out(conn, _TAG_ID, "tag", False)
    set_rolled_out(conn, _TAG_ID, "tag", True)
    row = fetch_label(conn, _TAG_ID, "tag")
    assert row is not None
    assert row[3] is True


def test_set_rolled_out_works_for_category(conn):
    set_rolled_out(conn, _CAT_ID, "category", False)
    row = fetch_label(conn, _CAT_ID, "category")
    assert row is not None
    assert row[3] is False
    # Restore
    set_rolled_out(conn, _CAT_ID, "category", True)
