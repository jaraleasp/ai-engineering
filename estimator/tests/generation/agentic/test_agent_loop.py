"""Tests for the hand-rolled agent loop, driven by a fake Responses client.

No network: a scripted fake stands in for ``client.responses`` so we can assert
the loop mechanics — call_id propagation, previous_response_id chaining, trace
capture and the max-iterations safeguard — deterministically and for free.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.generation.agentic.agent_loop import EstimationAgent, render_trace


def _function_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=arguments)


def _reasoning(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=text)])


def _message() -> SimpleNamespace:
    return SimpleNamespace(type="message")


def _response(rid: str, output: list, output_text: str = "") -> SimpleNamespace:
    return SimpleNamespace(id=rid, output=output, output_text=output_text)


class _FakeResponses:
    """Records every ``create`` kwargs and returns the next scripted response."""

    def __init__(self, scripted: list) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class _FakeClient:
    def __init__(self, scripted: list) -> None:
        self.responses = _FakeResponses(scripted)


async def _retriever(query, filters):
    return [{"id": 1, "estimated_hours": 940.0}, {"id": 2, "estimated_hours": 1150.0}]


async def test_loop_runs_tools_then_stops_and_captures_trace():
    scripted = [
        _response(
            "r1",
            [
                _reasoning("planning the decomposition"),
                _function_call("call_a", "search_budgets", '{"query": "backend", "filters": null}'),
            ],
        ),
        _response(
            "r2",
            [
                _function_call(
                    "call_b",
                    "calculate_estimate",
                    '{"components": [{"name": "Backend", "reference_amounts": [940, 1150]}]}',
                )
            ],
        ),
        _response("r3", [_message()], output_text="Final estimate ready."),
    ]
    agent = EstimationAgent(
        client=_FakeClient(scripted), retriever=_retriever, model="fake", max_iterations=5
    )
    result = await agent.run("transcript ...")

    assert result.stopped == "completed"
    assert result.iterations == 2
    assert result.summary == "Final estimate ready."
    assert [step.tool for step in result.trace] == ["search_budgets", "calculate_estimate"]
    assert result.trace[0].reasoning == "planning the decomposition"
    # The estimate is the last calculate_estimate result: median(940,1150)*1.15 = 1201.8.
    assert result.estimate["total_hours"] == 1201.8
    # The trace renders as the STEP format the exercise asks for.
    rendered = render_trace(result)
    assert "STEP 1" in rendered and "search_budgets" in rendered


async def test_loop_echoes_call_id_and_chains_previous_response_id():
    scripted = [
        _response("r1", [_function_call("call_xyz", "search_budgets", '{"query": "q", "filters": null}')]),
        _response("r2", [_message()], output_text="done"),
    ]
    client = _FakeClient(scripted)
    agent = EstimationAgent(client=client, retriever=_retriever, model="fake")
    await agent.run("t")

    # The second API call must chain from r1 and feed the tool output back with
    # the SAME call_id (the two most common loop bugs).
    second_call = client.responses.calls[1]
    assert second_call["previous_response_id"] == "r1"
    tool_outputs = second_call["input"]
    assert tool_outputs[0]["type"] == "function_call_output"
    assert tool_outputs[0]["call_id"] == "call_xyz"


class _EndlessClient:
    """A client that ALWAYS asks for another tool call — would loop forever
    without the max-iterations guard."""

    def __init__(self) -> None:
        self.responses = self
        self._i = 0

    def create(self, **kwargs):
        self._i += 1
        return _response(
            f"r{self._i}",
            [_function_call(f"c{self._i}", "search_budgets", '{"query": "q", "filters": null}')],
        )


async def test_loop_stops_at_max_iterations():
    agent = EstimationAgent(
        client=_EndlessClient(), retriever=_retriever, model="fake", max_iterations=3
    )
    result = await agent.run("t")
    assert result.stopped == "max_iterations"
    assert result.iterations == 3
