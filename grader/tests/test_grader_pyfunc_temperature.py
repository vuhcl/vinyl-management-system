"""Softmax temperature helper used by MLflow VinylGraderModel inference."""

from __future__ import annotations

import torch

from grader.src.models.grader_pyfunc import softmax_with_temperature


def test_temperature_one_matches_plain_softmax():
    logits = torch.tensor([[2.0, 1.0, 0.0], [-1.0, 3.0, 0.5]])
    got = softmax_with_temperature(logits, 1.0)
    want = torch.softmax(logits, dim=-1)
    assert torch.allclose(got, want)


def test_temperature_above_one_reduces_peak_preserves_argmax():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    tight = softmax_with_temperature(logits, 1.0)
    loose = softmax_with_temperature(logits, 4.0)
    assert float(tight.max()) > float(loose.max())
    assert int(tight.argmax()) == int(loose.argmax())


def test_invalid_temperature_falls_back_to_one():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    base = torch.softmax(logits, dim=-1)
    assert torch.allclose(softmax_with_temperature(logits, 0.0), base)
    assert torch.allclose(softmax_with_temperature(logits, -3.0), base)
