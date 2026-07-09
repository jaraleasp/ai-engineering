#!/usr/bin/env python3
"""Run the Session 12 estimation agent over a transcript and print its trace.

Usage (host, with OPENAI_API_KEY in .env)::

    # Real pipeline (needs the DB up and the budget corpus ingested):
    uv run python scripts/run_agent_s12.py exercises/session-12/sample_transcript_complex.txt

    # Offline stub (no DB needed — canned historical items):
    uv run python scripts/run_agent_s12.py \
        exercises/session-12/sample_transcript_complex.txt --stub --model gpt-5-mini

``--stub`` swaps the real hybrid+rerank retrieval for the standalone
``reference_retrieval.search_budgets_stub`` — same item shape, so the agent code
does not change. ``--model`` overrides ``AGENT_MODEL`` for a cheaper/faster run.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dependencies import get_estimation_agent  # noqa: E402
from app.generation.agentic.agent_loop import render_trace  # noqa: E402


def _build_stub_retriever():
    """Wrap the offline stub (sync, standalone) as an async BudgetRetriever."""
    stub_dir = ROOT / "exercises" / "session-12"
    if str(stub_dir) not in sys.path:
        sys.path.insert(0, str(stub_dir))
    from reference_retrieval import search_budgets_stub  # noqa: E402

    async def retriever(query: str, filters: dict | None) -> list[dict]:
        return search_budgets_stub(query, filters)

    return retriever


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Session 12 estimation agent.")
    parser.add_argument("transcript", type=Path, help="Path to the transcript .txt file.")
    parser.add_argument("--model", default=None, help="Override AGENT_MODEL (e.g. gpt-5-mini).")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use the offline canned retriever instead of the real pipeline (no DB).",
    )
    args = parser.parse_args()

    transcript = args.transcript.read_text(encoding="utf-8")
    retriever = _build_stub_retriever() if args.stub else None
    agent = get_estimation_agent(retriever=retriever, model=args.model)

    print(f"Running agent (model={agent._model}, stub={args.stub}) ...\n", file=sys.stderr)
    result = await agent.run(transcript)
    print(render_trace(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
