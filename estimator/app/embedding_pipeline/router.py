"""POST /embeddings/ingest — structural chunking + OpenAI embeddings."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


def get_chunker() -> JSONStructuralChunker:
	return JSONStructuralChunker()


def get_embedder() -> OpenAIEmbedder:
	return OpenAIEmbedder()


@router.post("/ingest", response_model=IngestResponse, status_code=200)
def ingest_embeddings(
	request: IngestRequest,
	chunker: JSONStructuralChunker = Depends(get_chunker),
	embedder: OpenAIEmbedder = Depends(get_embedder),
) -> IngestResponse:
	"""Chunk incoming budgets, embed them, and return chunks + aggregate stats."""
	try:
		chunks = chunker.chunk(request.budgets)
		embedded_chunks = embedder.embed_many(chunks)
		return IngestResponse(
			chunks=embedded_chunks,
			stats=IngestStats(
				total_budgets=len(request.budgets),
				total_chunks=len(embedded_chunks),
				total_tokens=sum(chunk.token_count for chunk in embedded_chunks),
				estimated_cost_usd=embedder.last_estimated_cost_usd,
			),
		)
	except Exception as exc:  # noqa: BLE001
		log.error(
			"embeddings_ingest_failed",
			error_type=type(exc).__name__,
			error=str(exc)[:400],
		)
		raise HTTPException(
			status_code=500,
			detail="Internal embedding pipeline error",
		) from exc
