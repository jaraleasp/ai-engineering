"""Structural chunking for budget JSON payloads.

Each budget component becomes one chunk. The chunk text keeps a compact header
with the parent budget context so embeddings preserve what project and sector
the component belongs to.
"""
from __future__ import annotations

import tiktoken

from app.config import Settings
from app.embedding_pipeline.schemas import Budget, Chunk


class JSONStructuralChunker:
	"""Split budgets into one chunk per component."""

	def __init__(
		self,
		model_name: str = Settings.model_fields["EMBEDDING_MODEL"].default,
	) -> None:
		self._encoding = tiktoken.encoding_for_model(model_name)

	def chunk(self, budgets: list[Budget]) -> list[Chunk]:
		chunks: list[Chunk] = []
		for budget in budgets:
			for component in budget.components:
				text = self._render_text(budget, component)
				chunks.append(
					Chunk(
						chunk_id=f"{budget.budget_id}::{component.component_id}",
						text=text,
						metadata={
							"budget_id": budget.budget_id,
							"component_id": component.component_id,
							"client_sector": budget.client_metadata.sector,
							"main_technology": budget.main_technology,
							"year": budget.year,
							"complexity": component.complexity,
							"estimated_hours": component.estimated_hours,
						},
						token_count=len(self._encoding.encode(text)),
					)
				)
		return chunks

	@staticmethod
	def _render_text(budget: Budget, component) -> str:
		return (
			f"[Project: {budget.project_summary}]\n"
			f"[Client sector: {budget.client_metadata.sector} | "
			f"Year: {budget.year} | Main tech: {budget.main_technology}]\n\n"
			f"Component: {component.name}\n"
			f"Description: {component.description}\n"
			f"Tech stack: {', '.join(component.tech_stack)}\n"
			f"Complexity: {component.complexity}\n"
			f"Estimated hours: {component.estimated_hours}"
		)
