"""
PL->EN prompt mapping and CLIP text prototype builder.

Flow:
  1. load_prompts(path) reads prompts.yaml → {label_name: [eng_term, ...]}
  2. build_prototypes(labels, map, model, tokenizer, device) →
       {label_id: normalised_float32_vec}

Each label's prototype is the L2-normalised mean of its per-term CLIP text vectors.
Averaging and re-normalising is standard practice for multi-synonym prototypes — it
places the prototype near the centroid of the cluster of related text embeddings.

Labels with auto_enabled=true but no mapping entry are skipped with a WARNING,
serving as a reminder to add the entry to prompts.yaml.
"""

import logging
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = "a photo of {term}"


def load_prompts(path: str) -> dict[str, list[str]]:
    """Load the YAML mapping file.  Returns an empty dict if the file is missing."""
    p = Path(path)
    if not p.exists():
        log.warning(
            "prompts.yaml not found at '%s' — all auto-enabled labels will be skipped", path
        )
        return {}
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def build_prototypes(
    labels: list[tuple[str, str, str]],
    prompts_map: dict[str, list[str]],
    model,
    tokenizer,
    device: str,
) -> dict[str, np.ndarray]:
    """Build one normalised float32 prototype vector per label.

    Args:
        labels:      list of (id, name, kind) — only name is used for the lookup.
        prompts_map: {label_name: [eng_term, ...]} from load_prompts().
        model:       open_clip model (must support encode_text).
        tokenizer:   open_clip tokenizer for the same model.
        device:      'cuda' or 'cpu'.

    Returns:
        {label_id: vec} — labels without a mapping entry are absent from the dict.
    """
    import torch

    from .model import encode_text  # local import to avoid circular at module level

    prototypes: dict[str, np.ndarray] = {}

    for label_id, label_name, _ in labels:
        terms = prompts_map.get(label_name)
        if not terms:
            log.warning(
                "No prompt mapping for label '%s' (id=%s) — "
                "add an entry to prompts.yaml to enable ML assignment",
                label_name,
                label_id,
            )
            continue

        texts = [PROMPT_TEMPLATE.format(term=t) for t in terms]
        vecs = encode_text(model, tokenizer, texts, device)  # [K, D]

        mean_vec = vecs.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec = mean_vec / norm

        prototypes[label_id] = mean_vec.astype(np.float32)

    return prototypes
