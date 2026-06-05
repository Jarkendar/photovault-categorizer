"""
Zero-shot scoring: cosine similarity between a photo vector and label prototypes.

Because both photo and prototype vectors are L2-normalised, cosine similarity
equals the dot product — so scoring is a single np.dot call per label.

Tags    — multi-label: any tag whose similarity >= TAG_THRESHOLD is assigned.
Categories — ranked: the top CATEGORY_TOP_K categories above CATEGORY_MIN_SCORE are assigned.
"""

import numpy as np


def score_photo(
    photo_vec: np.ndarray,
    tag_prototypes: dict[str, np.ndarray],
    category_prototypes: dict[str, np.ndarray],
    tag_threshold: float,
    category_top_k: int,
    category_min_score: float,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Compute cosine similarities and apply selection rules.

    Args:
        photo_vec:           L2-normalised float32 vector for the photo.
        tag_prototypes:      {tag_id: normalised_vec}
        category_prototypes: {category_id: normalised_vec}
        tag_threshold:       minimum cosine similarity for a tag hit.
        category_top_k:      maximum number of category hits to return.
        category_min_score:  minimum cosine similarity for any category hit.

    Returns:
        tag_hits: [(tag_id, score)] where score >= tag_threshold
        cat_hits: [(cat_id, score)] top-k categories with score >= category_min_score
    """
    tag_hits: list[tuple[str, float]] = []
    for label_id, proto in tag_prototypes.items():
        score = float(np.dot(photo_vec, proto))
        if score >= tag_threshold:
            tag_hits.append((label_id, score))

    cat_scores = [
        (label_id, float(np.dot(photo_vec, proto)))
        for label_id, proto in category_prototypes.items()
    ]
    cat_scores.sort(key=lambda x: x[1], reverse=True)
    cat_hits = [
        (label_id, score)
        for label_id, score in cat_scores[:category_top_k]
        if score >= category_min_score
    ]

    return tag_hits, cat_hits
