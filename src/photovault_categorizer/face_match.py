"""
Identity matching for face vectors — Phase 2 Iter 3.

Mirrors the role that prompts.build_prototypes + scoring.score_photo plays for CLIP,
but for face embeddings:

  1. build_identity_prototypes() — takes labelled face info + the face store,
     groups face vectors by person label (one person may span multiple clusters),
     computes the mean L2-normed prototype per person.

  2. match_faces() — for a batch of newly-detected face IDs, scores each face
     against all prototypes via dot product (vectors are L2-normed → cosine similarity),
     returns matches above the threshold.

Both functions are pure computations over numpy arrays — no DB access, no model loading —
so they are fully unit-testable without mocking.
"""

import logging
from collections import defaultdict

import numpy as np

from .face_store import FaceStore

log = logging.getLogger(__name__)


def build_identity_prototypes(
    labelled_face_info: list[tuple[str, str, str]],
    face_store: FaceStore,
) -> dict[str, tuple[np.ndarray, str]]:
    """Compute per-person prototype vectors from labelled cluster face embeddings.

    One person may have faces in multiple clusters all mapped to the same tag or
    category (e.g. same person photographed at different ages).  All their vectors
    are averaged together so the prototype is more robust than any single face.

    Args:
        labelled_face_info: list of (face_id, label_id, kind) for every face that
                            belongs to a labelled cluster.  kind is 'tag' or 'category'.
        face_store:         loaded FaceStore (call load() before passing in).

    Returns:
        dict mapping label_id → (prototype_vector float32 [512], kind).
        Only labels that have at least one vector in the store are included.
    """
    label_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
    label_kinds: dict[str, str] = {}
    missing = 0

    for face_id, label_id, kind in labelled_face_info:
        vec = face_store.get_vec(face_id)
        if vec is None:
            missing += 1
            continue
        label_vecs[label_id].append(vec.astype(np.float32))
        label_kinds[label_id] = kind

    if missing:
        log.warning(
            "%d labelled face(s) missing from face store — re-run detect_faces.py to fix",
            missing,
        )

    prototypes: dict[str, tuple[np.ndarray, str]] = {}
    for label_id, vecs in label_vecs.items():
        mean_vec: np.ndarray = np.mean(np.stack(vecs), axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm
        prototypes[label_id] = (mean_vec.astype(np.float32), label_kinds[label_id])

    log.info(
        "Identity prototypes built: %d person(s) from %d labelled face(s)",
        len(prototypes),
        len(labelled_face_info) - missing,
    )
    return prototypes


def match_faces(
    face_ids: list[str],
    face_store: FaceStore,
    identity_prototypes: dict[str, tuple[np.ndarray, str]],
    threshold: float,
) -> list[tuple[str, str, str, float]]:
    """Match a batch of face vectors against known identity prototypes.

    Uses dot product as similarity metric — face vectors are L2-normed by
    detect_and_embed(), so dot(a, b) == cosine_similarity(a, b).

    Args:
        face_ids:            IDs of newly-detected faces to classify.
        face_store:          loaded FaceStore (must already contain vectors for face_ids).
        identity_prototypes: output of build_identity_prototypes().
        threshold:           minimum cosine similarity accepted as a match.

    Returns:
        List of (face_id, label_id, kind, score) for faces that matched.
        Faces with no prototype above the threshold are omitted.
        When multiple prototypes exceed the threshold, only the best is returned.
    """
    if not face_ids or not identity_prototypes:
        return []

    label_ids = list(identity_prototypes.keys())
    # [P, 512] — one row per prototype
    proto_matrix = np.stack(
        [identity_prototypes[lid][0] for lid in label_ids]
    ).astype(np.float32)

    matches: list[tuple[str, str, str, float]] = []

    for face_id in face_ids:
        vec = face_store.get_vec(face_id)
        if vec is None:
            continue
        # dot product with all prototypes at once → [P] cosine similarities
        sims: np.ndarray = proto_matrix @ vec.astype(np.float32)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= threshold:
            matched_label = label_ids[best_idx]
            kind = identity_prototypes[matched_label][1]
            matches.append((face_id, matched_label, kind, best_score))

    return matches
