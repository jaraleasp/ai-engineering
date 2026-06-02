"""Validation tests for the embedding pipeline schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.embedding_pipeline.schemas import (
    Budget,
    Chunk,
    EmbeddedChunk,
    IngestRequest,
    IngestResponse,
)


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data"


def _sample_budget() -> dict[str, object]:
    payload = json.loads((FIXTURES_DIR / "budgets_sample.json").read_text(encoding="utf-8"))
    return payload[0]


def sample_budget() -> dict[str, object]:
    return _sample_budget()


def _embedded_chunk_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "chunk_id": "BUD-2024-014:0",
        "text": "OAuth 2.0 authentication backend with rate limiting.",
        "metadata": {
            "budget_id": "BUD-2024-014",
            "component_id": "AUTH-001",
            "year": 2024,
            "has_dependencies": False,
        },
        "token_count": 18,
        "embedding": [0.12, -0.44, 0.91],
    }
    payload.update(overrides)
    return payload


def test_budget_accepts_sample_json_shape() -> None:
    budget = Budget(**_sample_budget())
    assert budget.budget_id == "BUD-2024-014"
    assert budget.client_metadata.country == "ES"
    assert budget.components[0].complexity == "high"


def test_ingest_request_wraps_sample_budget_list() -> None:
    request = IngestRequest(budgets=[_sample_budget()])
    assert len(request.budgets) == 1
    assert request.budgets[0].components[0].tech_stack == [
        "ruby_on_rails",
        "postgresql",
        "redis",
    ]


def test_budget_rejects_unknown_component_complexity() -> None:
    bad_budget = _sample_budget()
    bad_budget["components"][0]["complexity"] = "extreme"
    with pytest.raises(ValidationError) as exc_info:
        Budget(**bad_budget)
    assert any(err["loc"] == ("components", 0, "complexity") for err in exc_info.value.errors())


def test_chunk_rejects_negative_token_count() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Chunk(
            chunk_id="BUD-2024-014:0",
            text="Short but valid chunk body.",
            metadata={"budget_id": "BUD-2024-014"},
            token_count=-1,
        )
    assert any(err["loc"] == ("token_count",) for err in exc_info.value.errors())


def test_embedded_chunk_requires_non_empty_embedding() -> None:
    with pytest.raises(ValidationError) as exc_info:
        EmbeddedChunk(**_embedded_chunk_payload(embedding=[]))
    assert any(err["loc"] == ("embedding",) for err in exc_info.value.errors())


def test_ingest_response_rejects_negative_stats() -> None:
    with pytest.raises(ValidationError) as exc_info:
        IngestResponse(
            chunks=[_embedded_chunk_payload()],
            stats={
                "total_budgets": 1,
                "total_chunks": 1,
                "total_tokens": -10,
                "estimated_cost_usd": 0.01,
            },
        )
    assert any(err["loc"] == ("stats", "total_tokens") for err in exc_info.value.errors())


def test_ingest_response_accepts_embedded_chunks() -> None:
    response = IngestResponse(
        chunks=[_embedded_chunk_payload()],
        stats={
            "total_budgets": 1,
            "total_chunks": 1,
            "total_tokens": 18,
            "estimated_cost_usd": 0.0004,
        },
    )
    assert response.stats.total_chunks == 1
    assert response.chunks[0].embedding[0] == pytest.approx(0.12)