#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for incremental extraction safety limits.
"""

import pytest

from text_to_graph_service import TextToGraphService


def _service_without_init() -> TextToGraphService:
    """Create a lightweight service instance without loading heavy models."""
    return TextToGraphService.__new__(TextToGraphService)


def test_incremental_rejects_overlap_greater_or_equal_chunk_size():
    service = _service_without_init()
    with pytest.raises(ValueError, match="overlap باید از chunk_size کمتر باشد"):
        service.extract_incremental(
            text="A" * 1000,
            chunk_size=200,
            overlap=200,
            base_method="spacy",
        )


def test_incremental_rejects_excessive_chunk_count_estimate():
    service = _service_without_init()
    with pytest.raises(ValueError, match="تعداد chunkهای تخمینی"):
        service.extract_incremental(
            text="A" * 5000,
            chunk_size=100,
            overlap=50,
            base_method="spacy",
            max_chunks=10,
        )


def test_incremental_caps_entities_and_relationships():
    service = _service_without_init()
    state = {"idx": 0}

    def fake_extract(_text, **_kwargs):
        idx = state["idx"]
        state["idx"] += 1
        return {
            "entities": [
                {"name": f"entity_{idx}_a", "attributes": {}},
                {"name": f"entity_{idx}_b", "attributes": {}},
            ],
            "relationships": [
                {
                    "source": f"entity_{idx}_a",
                    "target": f"entity_{idx}_b",
                    "relation": "linked_to",
                    "attributes": {},
                }
            ],
        }

    service.extract = fake_extract  # type: ignore[method-assign]

    result = service.extract_incremental(
        text="A" * 700,
        chunk_size=200,
        overlap=50,
        base_method="spacy",
        max_entities=3,
        max_relationships=2,
    )

    assert len(result["entities"]) == 3
    assert len(result["relationships"]) <= 2
    assert result["stats"]["dropped_entities"] > 0
