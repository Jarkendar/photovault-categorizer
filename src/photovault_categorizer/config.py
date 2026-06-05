"""
Configuration loaded from environment variables.

Accepts DB_URL in JDBC format (jdbc:postgresql://host:port/db) or plain psycopg3 DSN
(postgresql://host:port/db) — whichever the PhotoVault server .env already uses.
"""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

# Identifiers kept in sync with DB column `embedding_model` and the vector store filename.
MODEL_ID = "mobileclip-s2-datacompdr"
MODEL_NAME = "MobileCLIP-S2"
MODEL_PRETRAINED = "datacompdr"


@dataclass(frozen=True)
class Config:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    storage_root: str
    vector_store_dir: str
    prompts_path: str
    tag_threshold: float
    category_top_k: int
    category_min_score: float

    @classmethod
    def from_env(cls) -> "Config":
        raw_url = os.environ.get("DB_URL", "jdbc:postgresql://localhost:5432/photovault")
        # Strip 'jdbc:' prefix that the Ktor server uses.
        if raw_url.startswith("jdbc:"):
            raw_url = raw_url[5:]
        parsed = urlparse(raw_url)

        return cls(
            db_host=parsed.hostname or "localhost",
            db_port=parsed.port or 5432,
            db_name=(parsed.path or "/photovault").lstrip("/"),
            db_user=os.environ.get("DB_USER", "photovault"),
            db_password=os.environ.get("DB_PASSWORD", ""),
            storage_root=os.environ.get("PHOTO_STORAGE_ROOT", "./data/photos"),
            vector_store_dir=os.environ.get("VECTOR_STORE_DIR", "./data/vectors"),
            prompts_path=os.environ.get("PROMPTS_PATH", "prompts.yaml"),
            tag_threshold=float(os.environ.get("TAG_THRESHOLD", "0.25")),
            category_top_k=int(os.environ.get("CATEGORY_TOP_K", "1")),
            category_min_score=float(os.environ.get("CATEGORY_MIN_SCORE", "0.0")),
        )
