"""
Database write helpers for auto-assignment rows.

Source precedence is enforced via a single SQL statement using
INSERT ... ON CONFLICT DO UPDATE ... WHERE source = 'auto':

  - No existing row               → INSERT (source='auto')
  - Existing row, source='auto'   → UPDATE score and embedding_run
  - Existing row, source='manual' → WHERE condition false → no-op
  - Existing row, source='denied' → WHERE condition false → no-op

When the no-op path is taken, cur.rowcount == 0 so the caller can count
denied/manual skips for the summary log.
"""

import logging

import psycopg

log = logging.getLogger(__name__)

_TAG_SQL = """
    INSERT INTO photo_tags (photo_id, tag_id, score, source, embedding_run)
    VALUES (%(photo_id)s, %(label_id)s, %(score)s, 'auto', %(run_id)s)
    ON CONFLICT (photo_id, tag_id) DO UPDATE
        SET score          = EXCLUDED.score,
            embedding_run  = EXCLUDED.embedding_run
        WHERE photo_tags.source = 'auto'
"""

_CAT_SQL = """
    INSERT INTO photo_categories (photo_id, category_id, score, source, embedding_run)
    VALUES (%(photo_id)s, %(label_id)s, %(score)s, 'auto', %(run_id)s)
    ON CONFLICT (photo_id, category_id) DO UPDATE
        SET score          = EXCLUDED.score,
            embedding_run  = EXCLUDED.embedding_run
        WHERE photo_categories.source = 'auto'
"""


def assign_auto(
    conn: psycopg.Connection,
    photo_id: str,
    label_id: str,
    label_kind: str,
    score: float,
    run_id: str,
) -> bool:
    """Upsert an auto-assignment row, respecting source precedence.

    Args:
        conn:       psycopg3 connection (autocommit=True; caller wraps in transaction).
        photo_id:   photo-<uuid> string.
        label_id:   tag-<uuid> or cat-<uuid> string.
        label_kind: 'tag' or 'category'.
        score:      cosine similarity score from the classifier.
        run_id:     embedding run identifier for auditability.

    Returns:
        True  — row was inserted or updated (new auto assignment or re-score).
        False — row was blocked (existing source='manual' or source='denied').
    """
    sql = _TAG_SQL if label_kind == "tag" else _CAT_SQL
    params = {"photo_id": photo_id, "label_id": label_id, "score": score, "run_id": run_id}

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount > 0
