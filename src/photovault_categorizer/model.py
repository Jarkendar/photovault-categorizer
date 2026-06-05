"""
Model loading and encoding helpers for MobileCLIP-S2.

Both encode_image and encode_text return L2-normalised float32 numpy arrays,
so cosine similarity reduces to a plain dot product.

Device selection: CUDA if available, otherwise CPU.  This means the same code
runs on the RTX 2070 Super (fast tests) and on the Pi 5 (nightly automat).
"""

import logging
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from .config import MODEL_NAME, MODEL_PRETRAINED

log = logging.getLogger(__name__)


def load_model(
    device: str | None = None,
) -> tuple:
    """Load MobileCLIP-S2 and return (model, preprocess, tokenizer, device).

    The returned *preprocess* is the inference transform (center-crop + normalise,
    no random augmentations).  Its parameters are logged at INFO level for auditability.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # create_model_and_transforms returns (model, preprocess_train, preprocess_val).
    # We use preprocess_val (index 2) — deterministic, no augmentation.
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=MODEL_PRETRAINED
    )
    model = model.to(device).eval().float()  # ensure fp32 on both CPU and GPU
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    log.info("Loaded model %s/%s on device=%s", MODEL_NAME, MODEL_PRETRAINED, device)
    log.info("Inference preprocess: %s", preprocess)

    return model, preprocess, tokenizer, device


def encode_image(
    model: torch.nn.Module,
    preprocess,
    image_paths: list[Path | str],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode images from disk paths.

    Returns a float32 array of shape [N, D] with L2-normalised row vectors.
    Images are processed in batches to bound peak memory usage.
    """
    results: list[np.ndarray] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        ).to(device)

        with torch.no_grad():
            feats = model.encode_image(tensors.float())
            feats = feats / feats.norm(dim=-1, keepdim=True)

        results.append(feats.cpu().float().numpy())

    return np.concatenate(results, axis=0) if results else np.empty((0, 512), dtype=np.float32)


def encode_text(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    device: str,
) -> np.ndarray:
    """Encode text strings.

    Returns a float32 array of shape [N, D] with L2-normalised row vectors.
    """
    tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu().float().numpy()
