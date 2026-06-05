"""
Shared pytest configuration.

Sets default environment variables so unit tests work without a real .env file.
Integration tests (test_write_precedence.py) require a live Postgres instance
matching DB_URL — they are skipped automatically if the connection fails.
"""

import os

# Unit tests run without a DB; integration tests pick these up if present.
os.environ.setdefault("DB_URL", "jdbc:postgresql://localhost:5432/photovault")
os.environ.setdefault("DB_USER", "photovault")
os.environ.setdefault("DB_PASSWORD", "change-me")
os.environ.setdefault("PHOTO_STORAGE_ROOT", "/tmp/test-photos")
os.environ.setdefault("VECTOR_STORE_DIR", "/tmp/test-vectors")
os.environ.setdefault("PROMPTS_PATH", "prompts.yaml")
