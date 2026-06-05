"""
On-demand face clustering using DBSCAN.

Run this on the PC after detect_faces.py has processed the library.

Algorithm:
  1. Acquire exclusive file lock; exit 0 if already locked.
  2. Load the face vector store (faces-buffalo_l.npz).
  3. Fetch all faces with cluster_id IS NULL from the database.
     Previously-labelled clusters are frozen — this script never touches them.
  4. Retrieve their vectors from the store.  Faces missing a vector are skipped
     (should not happen after detect_faces, but handled defensively).
  5. Run DBSCAN with cosine metric (eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).
  6. For each cluster (DBSCAN label >= 0):
       a. Pick the representative face = the face with the highest det_score.
       b. INSERT a face_clusters row (id=fcluster-<uuid>, face_count, representative_face_id).
       c. UPDATE faces.cluster_id for all faces in the cluster.
  7. Noise points (DBSCAN label -1) are left with cluster_id = NULL for a future run.
  8. Log a summary: N clusters created, M faces assigned, K noise points.

Tuning:
  DBSCAN_EPS=0.4   — cosine distance threshold.  ArcFace R50 embeddings are L2-normalised,
                      so cosine distance = 1 - dot(a, b).  Two faces of the same person
                      typically score 0.2–0.3 distance; different persons 0.6–0.8.
                      Lower eps = stricter clusters (fewer false positives, more fragments);
                      raise if family members are being split across many small clusters.
  DBSCAN_MIN_SAMPLES=2 — minimum faces to form a cluster.  A single face = noise.

Usage:
    python -m photovault_categorizer.cli.cluster_faces

All configuration via environment variables — see .env.example.
"""

import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import FACE_MODEL_ID, Config
from ..db import (
    fetch_unclustered_faces,
    get_connection,
    insert_face_cluster,
    update_faces_cluster_id,
)
from ..face_store import FaceStore
from ..lock import with_lock

log = logging.getLogger(__name__)

# DBSCAN hyper-parameters — tune on real data; see module docstring for guidance.
DBSCAN_EPS: float = 0.4
DBSCAN_MIN_SAMPLES: int = 2


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with with_lock():
        _run()


def _run() -> None:
    from sklearn.cluster import DBSCAN

    config = Config.from_env()

    store_path = Path(config.face_store_dir) / f"faces-{FACE_MODEL_ID}.npz"
    store = FaceStore(store_path)
    store.load()
    log.info("Face store: %d vectors at %s", len(store), store_path)

    if len(store) == 0:
        log.info("Face store is empty — run detect_faces.py first")
        return

    with get_connection(config) as conn:
        unclustered = fetch_unclustered_faces(conn)

    if not unclustered:
        log.info("No unclustered faces found — nothing to do")
        return

    log.info("Unclustered faces: %d", len(unclustered))

    # Build (face_id, det_score, vector) triples, skipping faces missing a store entry.
    face_ids_ordered: list[str] = []
    det_scores: dict[str, float] = {}
    vecs: list[np.ndarray] = []

    missing_vecs = 0
    for face_id, det_score in unclustered:
        vec = store.get_vec(face_id)
        if vec is None:
            missing_vecs += 1
            continue
        face_ids_ordered.append(face_id)
        det_scores[face_id] = det_score
        vecs.append(vec)

    if missing_vecs:
        log.warning(
            "%d face(s) skipped — no vector in store (re-run detect_faces.py to fix)",
            missing_vecs,
        )

    if not face_ids_ordered:
        log.info("No vectors available for clustering")
        return

    matrix = np.stack(vecs).astype(np.float64)  # DBSCAN works in float64

    log.info(
        "Running DBSCAN on %d face vectors (eps=%.2f, min_samples=%d)",
        len(face_ids_ordered),
        DBSCAN_EPS,
        DBSCAN_MIN_SAMPLES,
    )

    labels: np.ndarray = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="cosine",
        n_jobs=-1,
    ).fit_predict(matrix)

    # Group faces by cluster label.
    label_to_face_ids: dict[int, list[str]] = {}
    for face_id, label in zip(face_ids_ordered, labels.tolist()):
        if label == -1:
            continue  # noise — leave cluster_id NULL
        label_to_face_ids.setdefault(label, []).append(face_id)

    n_clusters = len(label_to_face_ids)
    n_noise = int((labels == -1).sum())
    log.info("DBSCAN: %d cluster(s), %d noise point(s)", n_clusters, n_noise)

    if n_clusters == 0:
        log.info("No clusters formed — all faces are noise; try lowering DBSCAN_EPS")
        return

    now = datetime.now(timezone.utc)

    with get_connection(config) as conn:
        for face_id_list in label_to_face_ids.values():
            cluster_id = f"fcluster-{uuid.uuid4()}"

            # Representative = face with highest det_score in this cluster.
            representative = max(face_id_list, key=lambda fid: det_scores[fid])

            with conn.transaction():
                insert_face_cluster(
                    conn,
                    cluster_id=cluster_id,
                    face_count=len(face_id_list),
                    representative_face_id=representative,
                    now=now,
                )
                update_faces_cluster_id(conn, face_id_list, cluster_id)

    log.info(
        "cluster-faces: %d cluster(s) created, %d face(s) assigned, %d noise point(s) skipped",
        n_clusters,
        sum(len(v) for v in label_to_face_ids.values()),
        n_noise,
    )


if __name__ == "__main__":
    main()
