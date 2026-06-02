"""Unit tests for OpenAIEmbedder (no network calls)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import Chunk


def _chunk(i: int, token_count: int = 10) -> Chunk:
    return Chunk(
        chunk_id=f"BUD-2024-014::CMP-{i:03d}",
        text=f"chunk-{i}",
        metadata={
            "budget_id": "BUD-2024-014",
            "component_id": f"CMP-{i:03d}",
            "client_sector": "finance",
            "main_technology": "ruby_on_rails",
            "year": 2024,
            "complexity": "medium",
            "estimated_hours": 40,
        },
        token_count=token_count,
    )


def test_embed_one_calls_openai_and_returns_vector() -> None:
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=MagicMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.1, -0.2, 0.3])]
                )
            )
        )
    )
    embedder = OpenAIEmbedder(client=client)

    vector = embedder.embed_one("hello")

    assert vector == [0.1, -0.2, 0.3]
    assert client.embeddings.create.call_count == 1
    assert client.embeddings.create.call_args.kwargs["input"] == ["hello"]


def test_embed_many_batches_inputs_and_keeps_order() -> None:
    call_sizes: list[int] = []

    def fake_create(*, model: str, input: list[str]):
        call_sizes.append(len(input))
        data = [SimpleNamespace(embedding=[float(text.split("-")[1])]) for text in input]
        return SimpleNamespace(data=data)

    client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    embedder = OpenAIEmbedder(client=client, batch_size=100)
    chunks = [_chunk(i, token_count=5) for i in range(205)]

    embedded = embedder.embed_many(chunks)

    assert len(embedded) == 205
    assert call_sizes == [100, 100, 5]
    assert embedded[0].embedding == [0.0]
    assert embedded[204].embedding == [204.0]


def test_embed_many_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRateLimitError(Exception):
        pass

    attempts = {"count": 0}

    def flaky_create(*, model: str, input: list[str]):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeRateLimitError("slow down")
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.42])])

    client = SimpleNamespace(embeddings=SimpleNamespace(create=flaky_create))
    embedder = OpenAIEmbedder(client=client)
    sleep_calls: list[int] = []

    monkeypatch.setattr("app.embedding_pipeline.embedder.RateLimitError", FakeRateLimitError)
    monkeypatch.setattr("app.embedding_pipeline.embedder.time.sleep", lambda s: sleep_calls.append(s))

    embedded = embedder.embed_many([_chunk(1, token_count=20)])

    assert len(embedded) == 1
    assert embedded[0].embedding == [0.42]
    assert sleep_calls == [1, 2]


def test_embed_many_tracks_estimated_cost_from_input_tokens() -> None:
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.11]), SimpleNamespace(embedding=[0.22])]
            )
        )
    )
    embedder = OpenAIEmbedder(client=client, batch_size=100)

    embedder.embed_many([_chunk(1, token_count=500), _chunk(2, token_count=1500)])

    assert embedder.last_estimated_cost_usd == pytest.approx(0.00004)