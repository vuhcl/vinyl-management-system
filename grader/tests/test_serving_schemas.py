"""Tests for grader.serving Pydantic schemas."""

import pytest
from pydantic import ValidationError

from grader.serving.schemas import MAX_TEXT_LEN, PredictItem, PredictRequest


def test_predict_request_text_only_ok():
    r = PredictRequest(text="NM vinyl")
    assert r.text == "NM vinyl"
    assert r.items is None


def test_predict_request_items_only_ok():
    r = PredictRequest(items=[PredictItem(text="a"), PredictItem(text="b")])
    assert r.text is None
    assert len(r.items) == 2


def test_predict_request_rejects_both_text_and_items():
    with pytest.raises(ValidationError):
        PredictRequest(text="x", items=[PredictItem(text="y")])


def test_predict_request_rejects_neither():
    with pytest.raises(ValidationError):
        PredictRequest()


def test_predict_request_rejects_empty_text_without_items():
    with pytest.raises(ValidationError):
        PredictRequest(text="   ")


def test_predict_request_rejects_empty_items_list():
    with pytest.raises(ValidationError):
        PredictRequest(items=[])


def test_predict_item_respects_max_text_len():
    with pytest.raises(ValidationError):
        PredictItem(text="x" * (MAX_TEXT_LEN + 1))
