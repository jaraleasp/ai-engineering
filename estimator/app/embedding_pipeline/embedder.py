"""OpenAI embedding adapter for the structural chunking pipeline."""
from __future__ import annotations

import time

import structlog
from openai import OpenAI, RateLimitError

from app.config import Settings, get_settings
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger()

class OpenAIEmbedder:
	"""Batch embedder with simple retry and cost accounting."""

	def __init__(
		self,
		*,
		client: OpenAI | None = None,
		model_name: str = Settings.model_fields["EMBEDDING_MODEL"].default,
		batch_size: int = Settings.model_fields["EMBEDDING_BATCH_SIZE"].default,
		max_retries: int = Settings.model_fields["EMBEDDING_MAX_RETRIES"].default,
		cost_per_1m_tokens_usd: float = Settings.model_fields[
			"EMBEDDING_COST_PER_1M_TOKENS_USD"
		].default,
		rate_limit_backoff_seconds: tuple[int, ...] = Settings.model_fields[
			"EMBEDDING_RATE_LIMIT_BACKOFF_SECONDS"
		].default,
	) -> None:
		if client is None:
			settings = get_settings()
			if not settings.OPENAI_API_KEY:
				raise ValueError("OPENAI_API_KEY is required for OpenAIEmbedder")
			client = OpenAI(api_key=settings.OPENAI_API_KEY)

		if batch_size <= 0:
			raise ValueError("batch_size must be > 0")
		if max_retries < 0:
			raise ValueError("max_retries must be >= 0")
		if not rate_limit_backoff_seconds:
			raise ValueError("rate_limit_backoff_seconds must contain at least one value")
		if any(wait <= 0 for wait in rate_limit_backoff_seconds):
			raise ValueError("rate_limit_backoff_seconds values must be > 0")
		if cost_per_1m_tokens_usd < 0:
			raise ValueError("cost_per_1m_tokens_usd must be >= 0")

		self._client = client
		self._model_name = model_name
		self._batch_size = batch_size
		self._max_retries = max_retries
		self._cost_per_1m_tokens_usd = cost_per_1m_tokens_usd
		self._rate_limit_backoff_seconds = rate_limit_backoff_seconds
		self.last_estimated_cost_usd: float = 0.0

	def embed_one(self, text: str) -> list[float]:
		"""Embed a single text payload."""
		vectors = self._create_embeddings_with_retry([text])
		return vectors[0]

	def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
		"""Embed chunks in batched calls to OpenAI embeddings API."""
		if not chunks:
			self.last_estimated_cost_usd = 0.0
			return []

		embedded: list[EmbeddedChunk] = []
		total_tokens = 0

		for offset in range(0, len(chunks), self._batch_size):
			batch = chunks[offset : offset + self._batch_size]
			texts = [chunk.text for chunk in batch]
			batch_tokens = sum(chunk.token_count for chunk in batch)
			total_tokens += batch_tokens

			t0 = time.perf_counter()
			vectors = self._create_embeddings_with_retry(texts)
			latency_ms = int((time.perf_counter() - t0) * 1000)

			if len(vectors) != len(batch):
				raise RuntimeError(
					"OpenAI embeddings response size mismatch: "
					f"expected {len(batch)}, got {len(vectors)}"
				)

			log.info(
				"embedding_batch_processed",
				model=self._model_name,
				chunks_count=len(batch),
				total_tokens=batch_tokens,
				latency_ms=latency_ms,
			)

			for chunk, embedding in zip(batch, vectors, strict=True):
				embedded.append(
					EmbeddedChunk(
						chunk_id=chunk.chunk_id,
						text=chunk.text,
						metadata=chunk.metadata,
						token_count=chunk.token_count,
						embedding=embedding,
					)
				)

		self.last_estimated_cost_usd = _estimate_embedding_cost_usd(
			total_tokens,
			cost_per_1m_tokens_usd=self._cost_per_1m_tokens_usd,
		)
		return embedded

	def _create_embeddings_with_retry(self, inputs: list[str]) -> list[list[float]]:
		attempt = 0
		while True:
			try:
				response = self._client.embeddings.create(model=self._model_name, input=inputs)
				return [list(item.embedding) for item in response.data]
			except RateLimitError as exc:
				if attempt >= self._max_retries:
					raise
				wait_seconds = self._rate_limit_backoff_seconds[
					min(attempt, len(self._rate_limit_backoff_seconds) - 1)
				]
				log.warning(
					"embedding_rate_limited_retrying",
					model=self._model_name,
					attempt=attempt + 1,
					max_retries=self._max_retries,
					wait_seconds=wait_seconds,
					error=str(exc)[:200],
				)
				time.sleep(wait_seconds)
				attempt += 1


def _estimate_embedding_cost_usd(
	total_input_tokens: int,
	*,
	cost_per_1m_tokens_usd: float,
) -> float:
	return round((total_input_tokens * cost_per_1m_tokens_usd) / 1_000_000, 6)
