"""Pydantic v2 schemas for the embedding ingestion pipeline.

The contract mirrors the raw budget JSON at the boundary and then introduces
the chunk-oriented shapes the embedder consumes and returns.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FilterableMetadata = dict[str, str | int | float | bool | None]


class ClientMetadata(BaseModel):
	"""Client fields carried by each budget record."""

	model_config = ConfigDict(extra="forbid")

	name: str = Field(min_length=1, max_length=120)
	sector: str = Field(min_length=1, max_length=80)
	country: str = Field(min_length=2, max_length=2)


class BudgetComponent(BaseModel):
	"""One component inside a budget."""

	model_config = ConfigDict(extra="forbid")

	component_id: str = Field(min_length=1, max_length=64)
	name: str = Field(min_length=1, max_length=200)
	description: str = Field(min_length=1, max_length=4000)
	tech_stack: list[str] = Field(min_length=1)
	estimated_hours: int = Field(ge=0)
	complexity: Literal["low", "medium", "high"]
	dependencies: list[str] = Field(default_factory=list)


class Budget(BaseModel):
	"""One complete budget payload from the ingestion endpoint."""

	model_config = ConfigDict(extra="forbid")

	budget_id: str = Field(min_length=1, max_length=64)
	client_metadata: ClientMetadata
	project_summary: str = Field(min_length=1, max_length=4000)
	main_technology: str = Field(min_length=1, max_length=80)
	year: int = Field(ge=2000, le=2100)
	total_estimated_hours: int = Field(ge=0)
	components: list[BudgetComponent] = Field(min_length=1)


class Chunk(BaseModel):
	"""Chunk ready to be sent to the embedding model."""

	model_config = ConfigDict(extra="forbid")

	chunk_id: str = Field(min_length=1, max_length=128)
	text: str = Field(min_length=1, max_length=20000)
	metadata: FilterableMetadata = Field(default_factory=dict)
	token_count: int = Field(ge=0)


class EmbeddedChunk(Chunk):
	"""Chunk plus the generated embedding vector."""

	embedding: list[float] = Field(min_length=1)


class IngestRequest(BaseModel):
	"""Input payload accepted by the embedding ingestion endpoint."""

	model_config = ConfigDict(extra="forbid")

	budgets: list[Budget] = Field(min_length=1)


class IngestStats(BaseModel):
	"""Aggregate counters returned after embedding ingestion."""

	model_config = ConfigDict(extra="forbid")

	total_budgets: int = Field(ge=0)
	total_chunks: int = Field(ge=0)
	total_tokens: int = Field(ge=0)
	estimated_cost_usd: float = Field(ge=0)


class IngestResponse(BaseModel):
	"""Output payload returned by the embedding ingestion endpoint."""

	model_config = ConfigDict(extra="forbid")

	chunks: list[EmbeddedChunk] = Field(default_factory=list)
	stats: IngestStats



