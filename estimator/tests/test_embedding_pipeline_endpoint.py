"""HTTP-level tests for POST /embeddings/ingest."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.embedding_pipeline.router import get_chunker, get_embedder
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk
from app.main import app
from tests.test_embedding_pipeline_schemas import sample_budget


class _FakeChunker:
    def chunk(self, budgets):
        return [
            Chunk(
                chunk_id="BUD-2024-014::AUTH-001",
                text="chunk text",
                metadata={
                    "budget_id": "BUD-2024-014",
                    "component_id": "AUTH-001",
                    "client_sector": "finance",
                    "main_technology": "ruby_on_rails",
                    "year": 2024,
                    "complexity": "high",
                    "estimated_hours": 120,
                },
                token_count=18,
            )
        ]


class _FakeEmbedder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.last_estimated_cost_usd = 0.00036

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if self.fail:
            raise RuntimeError("OpenAI unavailable")
        return [
            EmbeddedChunk(
                chunk_id=chunks[0].chunk_id,
                text=chunks[0].text,
                metadata=chunks[0].metadata,
                token_count=chunks[0].token_count,
                embedding=[0.1, -0.2, 0.3],
            )
        ]


def _client_with_overrides(*, fail_embedder: bool = False) -> TestClient:
    app.dependency_overrides[get_chunker] = lambda: _FakeChunker()
    app.dependency_overrides[get_embedder] = lambda: _FakeEmbedder(fail=fail_embedder)
    return TestClient(app)


def test_post_embeddings_ingest_returns_200_with_chunks_and_stats() -> None:
    client = _client_with_overrides()
    payload = {"budgets": [sample_budget()]}

    response = client.post("/embeddings/ingest", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stats"]["total_budgets"] == 1
    assert body["stats"]["total_chunks"] == 1
    assert body["stats"]["total_tokens"] == 18
    assert body["stats"]["estimated_cost_usd"] == 0.00036
    assert body["chunks"][0]["chunk_id"] == "BUD-2024-014::AUTH-001"

    app.dependency_overrides.clear()


def test_post_embeddings_ingest_returns_422_on_invalid_body() -> None:
    client = _client_with_overrides()

    response = client.post("/embeddings/ingest", json={"wrong": []})

    assert response.status_code == 422
    app.dependency_overrides.clear()


def test_post_embeddings_ingest_returns_500_on_embedder_error() -> None:
    client = _client_with_overrides(fail_embedder=True)
    payload = {"budgets": [sample_budget()]}

    response = client.post("/embeddings/ingest", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal embedding pipeline error"
    app.dependency_overrides.clear()


def test_openapi_includes_embeddings_ingest_route() -> None:
    client = _client_with_overrides()

    schema = client.get("/openapi.json").json()

    assert "/embeddings/ingest" in schema["paths"]
    assert "post" in schema["paths"]["/embeddings/ingest"]
    app.dependency_overrides.clear()