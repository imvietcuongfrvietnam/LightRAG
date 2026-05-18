"""Tests for entity extraction via frame-semantic-transformer pipeline."""

from unittest.mock import AsyncMock, patch

import pytest

# Fake LLM filter response: keeps all entities, normalizes names (no-op normalization)
_FILTER_KEEP_ALL = (
    '{"entities": ['
    '{"original": "The seller", "normalized": "The seller"}, '
    '{"original": "the car", "normalized": "car"}'
    ']}'
)
_FILTER_KEEP_KEYWORDS = (
    '{"high_level_keywords": ["Commerce_sell"], '
    '"low_level_keywords": ["The seller", "car"]}'
)


def _make_global_config() -> dict:
    """Build a minimal global_config dict for extract_entities."""
    return {
        "llm_model_func": AsyncMock(return_value=_FILTER_KEEP_ALL),
        "entity_extract_max_gleaning": 0,
        "addon_params": {},
        "llm_model_max_async": 2,
    }


def _make_chunks(contents: list[str] | None = None) -> dict[str, dict]:
    contents = contents or ["Test content about selling cars."]
    return {
        f"chunk-{i:03d}": {
            "tokens": len(c),
            "content": c,
            "full_doc_id": "doc-001",
            "chunk_order_index": i,
        }
        for i, c in enumerate(contents)
    }


# Fake frame-extraction results used across tests
_FAKE_NODES = {
    "The seller": [
        {
            "entity_name": "The seller",
            "entity_type": "seller",
            "description": '"The seller" participates as Seller in the "Commerce_sell" semantic frame.',
            "source_id": "chunk-000",
            "file_path": "unknown_source",
            "timestamp": 1000000,
        }
    ],
    "the car": [
        {
            "entity_name": "the car",
            "entity_type": "goods",
            "description": '"the car" participates as Goods in the "Commerce_sell" semantic frame.',
            "source_id": "chunk-000",
            "file_path": "unknown_source",
            "timestamp": 1000000,
        }
    ],
}
_FAKE_EDGES = {
    ("The seller", "the car"): [
        {
            "src_id": "The seller",
            "tgt_id": "the car",
            "weight": 1.0,
            "keywords": "Commerce_sell",
            "description": "[Commerce_sell] ...",
            "source_id": "chunk-000",
            "file_path": "unknown_source",
            "timestamp": 1000000,
        }
    ]
}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_calls_frame_extractor():
    """extract_entities() must use frame extractor for primary extraction.

    In full mode the LLM is still called ONCE for post-filtering noise,
    but primary entity/relation extraction must come from frame_extractor.
    """
    from lightrag.operate import extract_entities

    global_config = _make_global_config()

    with patch(
        "lightrag.operate.extract_entities_from_frames",
        new_callable=AsyncMock,
        return_value=(_FAKE_NODES, _FAKE_EDGES),
    ) as mock_frame:
        results = await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    # Frame extractor must be called once per chunk
    assert mock_frame.await_count == 1
    # LLM called exactly once for the post-filter step (not for primary extraction)
    assert global_config["llm_model_func"].await_count == 1
    # extract_entities returns a list of (maybe_nodes, maybe_edges) tuples
    assert len(results) == 1
    nodes, edges = results[0]
    # LLM normalizes "the car" → "car"; "The seller" stays as-is
    assert "The seller" in nodes
    assert "car" in nodes           # normalized from "the car"
    assert ("The seller", "car") in edges  # edge keys updated to normalized names


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_processes_multiple_chunks():
    """All chunks must be processed independently via frame extraction."""
    from lightrag.operate import extract_entities

    # When frame extraction returns empty, filter gets keep=[] and falls back to originals
    global_config = _make_global_config()
    global_config["llm_model_func"] = AsyncMock(return_value='{"keep": []}')
    chunks = _make_chunks(["Sentence one.", "Sentence two.", "Sentence three."])

    with patch(
        "lightrag.operate.extract_entities_from_frames",
        new_callable=AsyncMock,
        return_value=({}, {}),
    ) as mock_frame:
        results = await extract_entities(
            chunks=chunks,
            global_config=global_config,
        )

    assert mock_frame.await_count == 3
    assert len(results) == 3


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_returns_empty_on_frame_failure():
    """Graceful degradation: frame extractor failure should not crash the pipeline."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config()

    with patch(
        "lightrag.operate.extract_entities_from_frames",
        new_callable=AsyncMock,
        return_value=({}, {}),
    ):
        results = await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    assert results is not None
    assert len(results) == 1
    nodes, edges = results[0]
    assert nodes == {}
    assert edges == {}
