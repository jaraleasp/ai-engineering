"""Tool implementations for the Session 12 estimation agent.

Two tools back the schemas in :mod:`agent_schemas`:

* ``calculate_estimate`` — a pure, deterministic cost roll-up (NO LLM).
* ``search_budgets`` — a thin wrapper over an INJECTED retrieval backend.

Layering note (see ``ARCHITECTURE.md`` §3): ``agentic`` may not import a
``generation`` sibling, so this module never imports ``generation/rag``. The real
retrieval pipeline (embed → hybrid search → rerank → map) is wired in
``dependencies.py`` (the composition root) and passed in as a ``BudgetRetriever``
callable. That same seam lets the run script swap in the offline stub with
``--stub`` — the agent code does not change.
"""

from __future__ import annotations

import statistics
from typing import Any, Awaitable, Callable

# A retrieval backend: given a component query and optional filters, return
# historical budget items in the canonical shape (id, content_preview, sector,
# budget_id, estimated_hours, distance). Injected from the composition root.
BudgetRetriever = Callable[[str, dict[str, Any] | None], Awaitable[list[dict[str, Any]]]]

# Flat contingency buffer added to every component's central estimate. Kept
# transparent — no hidden multipliers (the value is defensible in the live session).
CONTINGENCY_FACTOR = 0.15


def calculate_estimate(args: dict[str, Any]) -> dict[str, Any]:
    """Cost each component from its historical reference amounts, then total.

    Deterministic and LLM-free. The central estimate is the MEDIAN of the
    references (robust to a single outlier budget), plus a flat contingency
    buffer. A component with no references is costed at 0 and flagged
    ``unbudgeted`` — never invented — so the agent can notice and search again.
    """
    components = args["components"]
    breakdown: list[dict[str, Any]] = []
    total = 0.0

    for component in components:
        name = component["name"]
        refs = component.get("reference_amounts", []) or []

        if refs:
            central = statistics.median(refs)
            hours = round(central * (1 + CONTINGENCY_FACTOR), 1)
            unbudgeted = False
        else:
            hours = 0.0
            unbudgeted = True

        total += hours
        breakdown.append(
            {
                "name": name,
                "reference_count": len(refs),
                "estimated_hours": hours,
                "unbudgeted": unbudgeted,
            }
        )

    total = round(total, 1)
    return {
        "components": breakdown,
        "total_hours": total,
        "summary": f"total={total}h across {len(breakdown)} components",
    }


async def search_budgets(
    args: dict[str, Any], *, retriever: BudgetRetriever
) -> list[dict[str, Any]]:
    """Retrieve historical budgets for one component via the injected backend.

    The tool itself is intentionally thin: the actual pipeline (embedding, hybrid
    search, reranking, mapping) lives behind ``retriever``. ``filters`` may be
    ``None`` (the model passes ``null`` when it wants the whole corpus).
    """
    query = args["query"]
    filters = args.get("filters")
    return await retriever(query, filters)


def format_search_observation(query: str, items: list[dict[str, Any]]) -> str:
    """One-line trace observation summarising a ``search_budgets`` result."""
    if not items:
        return f"no historical items for {query!r}"
    hours = [item.get("estimated_hours") for item in items]
    return f"{len(items)} historical items for {query!r}; hours={hours}"


def format_calculate_observation(result: dict[str, Any]) -> str:
    """One-line trace observation summarising a ``calculate_estimate`` result."""
    return result["summary"]


async def execute_tool(
    name: str, args: dict[str, Any], *, retriever: BudgetRetriever
) -> tuple[Any, str]:
    """Dispatch a tool call by name.

    Returns ``(result, observation)`` where ``result`` is the structured payload
    fed back to the model as a ``function_call_output`` and ``observation`` is the
    short human-readable line captured in the trace.
    """
    if name == "search_budgets":
        items = await search_budgets(args, retriever=retriever)
        return items, format_search_observation(args["query"], items)
    if name == "calculate_estimate":
        result = calculate_estimate(args)
        return result, format_calculate_observation(result)
    raise ValueError(f"unknown tool: {name!r}")
