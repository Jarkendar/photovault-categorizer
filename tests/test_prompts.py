"""Tests for load_prompts() and build_prototypes()."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from photovault_categorizer.prompts import PROMPT_TEMPLATE, build_prototypes, load_prompts


# ── Fake model / tokenizer stubs ────────────────────────────────────────────


class _FakeTokenizer:
    """Returns zero integer tensors — shape matches open_clip tokenizer output."""

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return torch.zeros(len(texts), 77, dtype=torch.long)


class _FakeModel:
    """Returns random unit vectors for encode_text (seed is fixed for reproducibility)."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        N = tokens.shape[0]
        vecs = torch.randn(N, 512, generator=self._rng)
        return vecs / vecs.norm(dim=-1, keepdim=True)


# ── load_prompts ─────────────────────────────────────────────────────────────


def test_load_prompts_parses_yaml(tmp_path):
    (tmp_path / "p.yaml").write_text(
        '"#morze":\n  - sea\n  - ocean\n"#rower":\n  - bicycle\n',
        encoding="utf-8",
    )
    result = load_prompts(str(tmp_path / "p.yaml"))
    assert result["#morze"] == ["sea", "ocean"]
    assert result["#rower"] == ["bicycle"]


def test_load_prompts_returns_empty_dict_when_file_missing(tmp_path):
    result = load_prompts(str(tmp_path / "nonexistent.yaml"))
    assert result == {}


def test_load_prompts_handles_empty_file(tmp_path):
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    result = load_prompts(str(tmp_path / "empty.yaml"))
    assert result == {}


# ── build_prototypes ─────────────────────────────────────────────────────────


def test_build_prototypes_skips_missing_mapping(capsys):
    """Label without a prompts_map entry is absent from the result dict."""
    labels = [("tag-1", "#unknown", "tag")]
    result = build_prototypes(labels, {}, _FakeModel(), _FakeTokenizer(), "cpu")
    assert result == {}


def test_build_prototypes_returns_normalised_vector():
    labels = [("tag-1", "#morze", "tag")]
    prompts_map = {"#morze": ["sea", "ocean"]}
    result = build_prototypes(labels, prompts_map, _FakeModel(), _FakeTokenizer(), "cpu")

    assert "tag-1" in result
    vec = result["tag-1"]
    assert vec.dtype == np.float32
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


def test_build_prototypes_multiple_terms_averaged():
    """Prototype is the L2-normalised mean of per-term CLIP text vectors."""
    # Two known unit vectors at orthogonal dimensions
    v1 = np.zeros(512, dtype=np.float32); v1[0] = 1.0
    v2 = np.zeros(512, dtype=np.float32); v2[1] = 1.0

    class _FixedModel:
        """Returns a unit vector at dimension i for row i in the batch."""
        def encode_text(self, tokens):
            N = tokens.shape[0]
            vecs = np.zeros((N, 512), dtype=np.float32)
            for i in range(N):
                vecs[i, i] = 1.0  # row 0 → v1, row 1 → v2
            return torch.from_numpy(vecs)

    labels = [("tag-x", "term", "tag")]
    result = build_prototypes(
        labels, {"term": ["t1", "t2"]}, _FixedModel(), _FakeTokenizer(), "cpu"
    )

    # Expected: mean(v1, v2) / ||mean(v1, v2)||
    expected = (v1 + v2) / 2.0
    expected = expected / np.linalg.norm(expected)

    np.testing.assert_array_almost_equal(result["tag-x"], expected, decimal=5)


def test_build_prototypes_prompt_template():
    """The PROMPT_TEMPLATE constant is applied to each term."""
    assert PROMPT_TEMPLATE == "a photo of {term}"
    # Verify template expansion produces expected strings
    terms = ["sea", "ocean"]
    expected = ["a photo of sea", "a photo of ocean"]
    assert [PROMPT_TEMPLATE.format(term=t) for t in terms] == expected
