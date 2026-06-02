"""Tests for the structural JSON chunker."""

from __future__ import annotations

import tiktoken

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.schemas import Budget
from tests.test_embedding_pipeline_schemas import sample_budget


def test_chunker_emits_one_chunk_per_component() -> None:
    chunker = JSONStructuralChunker()

    chunks = chunker.chunk([Budget(**sample_budget())])

    assert len(chunks) == 1


def test_chunker_builds_traceable_id_text_metadata_and_tokens() -> None:
    budget = Budget(**sample_budget())
    chunker = JSONStructuralChunker()

    chunk = chunker.chunk([budget])[0]
    expected_text = (
        "[Project: Mobile banking API with OAuth 2.0 authentication and PSD2 compliance]\n"
        "[Client sector: finance | Year: 2024 | Main tech: ruby_on_rails]\n\n"
        "Component: OAuth 2.0 authentication backend\n"
        "Description: Implementation of OAuth 2.0 flows (authorization code, refresh token) "
        "with JWT-based session management, multi-tenant token isolation, and rate limiting "
        "per client.\n"
        "Tech stack: ruby_on_rails, postgresql, redis\n"
        "Complexity: high\n"
        "Estimated hours: 120"
    )
    expected_metadata = {
        "budget_id": "BUD-2024-014",
        "component_id": "AUTH-001",
        "client_sector": "finance",
        "main_technology": "ruby_on_rails",
        "year": 2024,
        "complexity": "high",
        "estimated_hours": 120,
    }
    expected_token_count = len(
        tiktoken.encoding_for_model("text-embedding-3-small").encode(expected_text)
    )

    assert chunk.chunk_id == "BUD-2024-014::AUTH-001"
    assert chunk.text == expected_text
    assert chunk.metadata == expected_metadata
    assert chunk.token_count == expected_token_count


def test_chunker_keeps_parent_context_for_each_component() -> None:
    payload = sample_budget()
    payload["components"].append(
        {
            "component_id": "OBS-002",
            "name": "Observability dashboard",
            "description": "Metrics, traces, and alerting for the banking API.",
            "tech_stack": ["grafana", "prometheus"],
            "estimated_hours": 40,
            "complexity": "medium",
            "dependencies": ["AUTH-001"],
        }
    )
    chunker = JSONStructuralChunker()

    chunks = chunker.chunk([Budget(**payload)])

    assert len(chunks) == 2
    assert "[Client sector: finance | Year: 2024 | Main tech: ruby_on_rails]" in chunks[0].text
    assert "[Client sector: finance | Year: 2024 | Main tech: ruby_on_rails]" in chunks[1].text
    assert chunks[1].chunk_id == "BUD-2024-014::OBS-002"