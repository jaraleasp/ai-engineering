"""Tests for the Session 12 agent tools: deterministic calc + injected search."""

from __future__ import annotations

import pytest

from app.generation.agentic.agent_tools import (
    calculate_estimate,
    execute_tool,
    format_search_observation,
    search_budgets,
    validate_estimate,
)


def test_calculate_estimate_uses_median_plus_contingency():
    result = calculate_estimate(
        {"components": [{"name": "Backend", "reference_amounts": [940, 1150]}]}
    )
    component = result["components"][0]
    # median(940, 1150) = 1045 -> * 1.15 = 1201.75 -> round(1) = 1201.8
    assert component["estimated_hours"] == 1201.8
    assert component["unbudgeted"] is False
    assert component["reference_count"] == 2
    assert result["total_hours"] == 1201.8


def test_calculate_estimate_flags_empty_references_as_unbudgeted():
    result = calculate_estimate({"components": [{"name": "X", "reference_amounts": []}]})
    component = result["components"][0]
    # No references -> costed at 0 and flagged, never invented.
    assert component["estimated_hours"] == 0.0
    assert component["unbudgeted"] is True
    assert result["total_hours"] == 0.0


def test_calculate_estimate_totals_across_components():
    result = calculate_estimate(
        {
            "components": [
                {"name": "A", "reference_amounts": [100]},  # 115.0
                {"name": "B", "reference_amounts": [200]},  # 230.0
            ]
        }
    )
    assert result["total_hours"] == 345.0
    assert "across 2 components" in result["summary"]


async def test_search_budgets_delegates_to_injected_retriever():
    captured: dict = {}

    async def retriever(query, filters):
        captured["query"] = query
        captured["filters"] = filters
        return [{"id": 1, "estimated_hours": 420.0}]

    items = await search_budgets(
        {"query": "auth backend", "filters": {"sectors": ["finance"]}}, retriever=retriever
    )
    assert items == [{"id": 1, "estimated_hours": 420.0}]
    assert captured["query"] == "auth backend"
    assert captured["filters"] == {"sectors": ["finance"]}


def test_format_search_observation_handles_empty_and_nonempty():
    assert "no historical items" in format_search_observation("q", [])
    observation = format_search_observation(
        "q", [{"estimated_hours": 940.0}, {"estimated_hours": 1150.0}]
    )
    assert "2 historical items" in observation
    assert "940.0" in observation and "1150.0" in observation


async def test_execute_tool_dispatches_search_and_calculate():
    async def retriever(query, filters):
        return [{"id": 1, "estimated_hours": 500.0}]

    result, observation = await execute_tool(
        "search_budgets", {"query": "x", "filters": None}, retriever=retriever
    )
    assert result == [{"id": 1, "estimated_hours": 500.0}]
    assert "1 historical items" in observation

    result, observation = await execute_tool(
        "calculate_estimate",
        {"components": [{"name": "A", "reference_amounts": [100]}]},
        retriever=retriever,
    )
    assert result["total_hours"] == 115.0
    assert observation == result["summary"]


async def test_execute_tool_rejects_unknown_tool():
    async def retriever(query, filters):
        return []

    with pytest.raises(ValueError):
        await execute_tool("does_not_exist", {}, retriever=retriever)


def test_validate_estimate_passes_a_clean_estimate():
    result = validate_estimate(
        {
            "components": [{"name": "A", "estimated_hours": 115.0, "reference_amounts": [100, 120]}],
            "total_hours": 115.0,
        }
    )
    assert result["ok"] is True
    assert result["issues"] == []
    assert result["summary"] == "estimate passed all guardrails"


def test_validate_estimate_flags_unbudgeted_and_out_of_range():
    result = validate_estimate(
        {
            "components": [
                {"name": "A", "estimated_hours": 100.0, "reference_amounts": []},  # unbudgeted
                {"name": "B", "estimated_hours": 5000.0, "reference_amounts": [100, 120]},  # too high
            ],
            "total_hours": 5100.0,
        }
    )
    assert result["ok"] is False
    assert any("unbudgeted" in issue for issue in result["issues"])
    assert any("outside the plausible range" in issue for issue in result["issues"])


def test_validate_estimate_flags_total_mismatch():
    result = validate_estimate(
        {
            "components": [{"name": "A", "estimated_hours": 115.0, "reference_amounts": [100, 120]}],
            "total_hours": 999.0,
        }
    )
    assert result["ok"] is False
    assert any("does not match the sum" in issue for issue in result["issues"])


async def test_execute_tool_dispatches_validate_estimate():
    async def retriever(query, filters):
        return []

    result, observation = await execute_tool(
        "validate_estimate",
        {
            "components": [{"name": "A", "estimated_hours": 115.0, "reference_amounts": [100, 120]}],
            "total_hours": 115.0,
        },
        retriever=retriever,
    )
    assert result["ok"] is True
    assert observation == result["summary"]
