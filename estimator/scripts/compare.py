"""CLI to compare two texts using OpenAI embeddings cosine similarity."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Support direct execution via `python scripts/compare.py`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from app.embedding_pipeline.embedder import OpenAIEmbedder


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
	"""Compute cosine similarity with the Python standard library only."""
	if len(vec_a) != len(vec_b):
		raise ValueError("vectors must have the same length")
	if not vec_a:
		raise ValueError("vectors must not be empty")

	dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
	norm_a = math.sqrt(sum(a * a for a in vec_a))
	norm_b = math.sqrt(sum(b * b for b in vec_b))

	if norm_a == 0.0 or norm_b == 0.0:
		raise ValueError("cosine similarity is undefined for zero vectors")

	return dot_product / (norm_a * norm_b)


def run_compare(text_a: str, text_b: str) -> float:
	"""Embed both texts and return cosine similarity."""
	embedder = OpenAIEmbedder()
	vector_a = embedder.embed_one(text_a)
	vector_b = embedder.embed_one(text_b)
	return cosine_similarity(vector_a, vector_b)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Compare two texts using OpenAI embeddings cosine similarity.",
	)
	parser.add_argument("--text-a", required=True, help="First text to compare")
	parser.add_argument("--text-b", required=True, help="Second text to compare")
	return parser


def main() -> None:
	args = build_parser().parse_args()
	similarity = run_compare(args.text_a, args.text_b)

	print(f"Text A: {args.text_a}")
	print(f"Text B: {args.text_b}")
	print(f"Cosine similarity: {similarity:.4f}")


if __name__ == "__main__":
	main()
