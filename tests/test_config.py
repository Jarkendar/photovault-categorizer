"""Tests for Config.from_env() — JDBC URL parsing and defaults."""

import os
import pytest
from photovault_categorizer.config import Config


def _set_env(**kwargs):
    """Override env vars for the duration of a test."""
    for k, v in kwargs.items():
        os.environ[k] = v


def test_jdbc_url_parsed_correctly():
    _set_env(DB_URL="jdbc:postgresql://myhost:5433/mydb", DB_USER="user", DB_PASSWORD="pass")
    c = Config.from_env()
    assert c.db_host == "myhost"
    assert c.db_port == 5433
    assert c.db_name == "mydb"
    assert c.db_user == "user"
    assert c.db_password == "pass"


def test_plain_postgresql_url_accepted():
    _set_env(DB_URL="postgresql://otherhost:5432/otherdb")
    c = Config.from_env()
    assert c.db_host == "otherhost"
    assert c.db_name == "otherdb"


def test_default_port_when_omitted():
    _set_env(DB_URL="jdbc:postgresql://localhost/photovault")
    c = Config.from_env()
    assert c.db_port == 5432


def test_scoring_defaults():
    c = Config.from_env()
    assert c.tag_threshold == pytest.approx(0.25)
    assert c.category_top_k == 1
    assert c.category_min_score == pytest.approx(0.0)


def test_scoring_overrides():
    _set_env(TAG_THRESHOLD="0.35", CATEGORY_TOP_K="2", CATEGORY_MIN_SCORE="0.1")
    c = Config.from_env()
    assert c.tag_threshold == pytest.approx(0.35)
    assert c.category_top_k == 2
    assert c.category_min_score == pytest.approx(0.1)
