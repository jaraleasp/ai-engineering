"""Tool schemas for the Session 12 estimation agent (OpenAI Responses API).

These are the ONLY thing the model reads to decide when and how to call each
tool — it never sees the Python implementations. So the ``description`` fields
carry real weight: they are the tool's contract as far as the model is concerned.

Responses API note: the schema is FLAT here —
``{"type": "function", "name": ..., "description": ..., "parameters": {...}}`` —
unlike Chat Completions, which nests everything under a ``"function"`` key.

``strict: true`` note: strict mode requires that EVERY key in ``properties`` is
listed in ``required`` and that ``additionalProperties`` is false. An "optional"
parameter is therefore expressed as a NULLABLE type (``["object", "null"]``),
not by leaving it out of ``required`` — the model passes ``null`` when it has
nothing to add.
"""

from __future__ import annotations

# Recover historical budgets for ONE component/requirement at a time. The
# description steers the model to search per-component (many focused calls)
# rather than one broad query — the behaviour the exercise grades.
SEARCH_BUDGETS_SCHEMA = {
    "type": "function",
    "name": "search_budgets",
    "description": (
        "Search past project budgets for historical references relevant to ONE "
        "component or requirement. Call it once per distinct component you have "
        "identified (e.g. a backend, an ERP integration, a mobile app), with a "
        "focused query describing that single component. Returns a list of "
        "historical items, each with its recorded engineer-hours and metadata. "
        "If a search returns nothing, broaden the query or relax the filters and "
        "search again."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["query", "filters"],
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of the single component to price, "
                    "rich enough for semantic retrieval (what it does, its main "
                    "features and technology). One component per call."
                ),
            },
            "filters": {
                "type": ["object", "null"],
                "description": (
                    "Optional structural filters to narrow the search. Pass null "
                    "when you want to search the whole corpus."
                ),
                "additionalProperties": False,
                "required": ["sectors", "component_type"],
                "properties": {
                    "sectors": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": (
                            "Restrict to these client sectors (e.g. ['logistics']). "
                            "Null = any sector. Relax this to null if a filtered "
                            "search returns nothing."
                        ),
                    },
                    "component_type": {
                        "type": ["string", "null"],
                        "description": (
                            "Coarse component category for your own bookkeeping "
                            "(e.g. 'backend/API', 'ERP integration', 'mobile app', "
                            "'analytics/dashboard'). Null when not useful."
                        ),
                    },
                },
            },
        },
    },
}

# Deterministic cost roll-up. The model gathers reference amounts from the
# search results and hands them here; the tool does the arithmetic (no LLM).
CALCULATE_ESTIMATE_SCHEMA = {
    "type": "function",
    "name": "calculate_estimate",
    "description": (
        "Compute the effort estimate (partial or total) from a set of components "
        "and their historical reference amounts. Call this ONCE you have gathered "
        "reference hours for every component. Each component carries its name and "
        "the list of historical hours you found for it (pass an empty list if you "
        "found none — it will be flagged as unbudgeted, not invented). Returns a "
        "per-component breakdown with a central estimate plus a contingency buffer, "
        "and the grand total in engineer-hours."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "required": ["components"],
        "properties": {
            "components": {
                "type": "array",
                "description": "The components to price, one object per component.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "reference_amounts"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Human-readable component name.",
                        },
                        "reference_amounts": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": (
                                "Historical engineer-hours found for this component. "
                                "Empty list if no reference was found."
                            ),
                        },
                    },
                },
            },
        },
    },
}

# The list passed to ``client.responses.create(..., tools=AGENT_TOOLS)``.
AGENT_TOOLS = [SEARCH_BUDGETS_SCHEMA, CALCULATE_ESTIMATE_SCHEMA]
