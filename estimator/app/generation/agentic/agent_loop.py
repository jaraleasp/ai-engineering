"""Hand-rolled agentic loop for transcript estimation (Session 12).

No framework. An agent is just a loop that lets an LLM DECIDE, runs the tools it
asks for, and stops when it is done — this module is exactly that, driven against
the OpenAI Responses API so every decision is visible and captured in a trace.

The loop (``EstimationAgent.run``):

1. Call ``responses.create`` with the system prompt, the transcript as input, and
   the tool schemas.
2. Scan ``response.output`` for ``function_call`` items. For each, run the tool and
   build a ``function_call_output`` carrying the SAME ``call_id``.
3. Chain the next turn with ``previous_response_id`` and the tool outputs.
4. Repeat while the model keeps calling tools; stop when it returns a final message
   (natural exit) or when ``max_iterations`` is hit (safeguard against a runaway).
5. One final ``responses.parse`` turns the accumulated context into a validated
   ``AgentEstimate`` (the structured deliverable).

We drive the chaining ourselves instead of delegating to the API's built-in
agentic behaviour — that is the only way to capture each reasoning → action →
observation step, which is half the exercise.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.generation.agentic.agent_schemas import AGENT_TOOLS, AgentEstimate
from app.generation.agentic.agent_tools import BudgetRetriever, execute_tool

log = structlog.get_logger()

DEFAULT_MAX_ITERATIONS = 8

# Role + method. The method is deliberately procedural (decompose → search per
# component → calculate → consolidate) but NOT prescriptive about HOW MANY searches
# or in which order — that is the agent's call, which is what makes it an agent.
SYSTEM_PROMPT = """\
You are a software project estimation agent. You are given the transcript of a \
discovery meeting and must produce a structured effort estimate grounded in \
historical budgets.

Method:
1. Read the transcript and DECOMPOSE the project into its distinct components \
(for example: a business backend, an ERP/SAP integration, a mobile app, an \
analytics dashboard). A real project usually has several unrelated components.
2. For EACH component, call `search_budgets` with a focused query describing that \
single component to retrieve historical references. Use one call per component. \
If a search returns nothing, broaden the query or relax the filters and try again.
3. Collect the historical engineer-hours from the results. Judge relevance: drop \
items that clearly belong to a different kind of work, even if they matched.
4. Once you have references for every component, call `calculate_estimate` with \
the components and the reference amounts you gathered. Pass an empty list of \
references for a component you could not find — do not invent numbers.
5. Call `validate_estimate` as the LAST tool step and fix anything it flags \
(e.g. search again for an unbudgeted component) before finishing.
6. When the estimate passes validation, stop calling tools. You will then be \
asked to return the final structured estimate.

You have exactly three tools: `search_budgets`, `calculate_estimate` and \
`validate_estimate`. Ground your numbers in what `search_budgets` returns; record \
anything the transcript did not specify as an assumption."""


# Sent as the final user turn (after the tool loop) to elicit the structured
# estimate via ``responses.parse``. Kept separate from the loop instructions.
FINAL_INSTRUCTION = (
    "Return the final structured estimate now, consolidating the components you "
    "costed. Set total_hours to the sum of the components, cite the historical "
    "source ids you relied on per component, list your assumptions, and choose a "
    "confidence level reflecting how well the historical budgets matched the work."
)


@dataclass
class TraceStep:
    """One captured decision: the reasoning, the tool acted, and what came back."""

    index: int
    reasoning: str
    tool: str
    arguments: dict[str, Any]
    observation: str


@dataclass
class AgentResult:
    """The agent's final output: the structured estimate plus its full trace."""

    estimate: AgentEstimate | None
    trace: list[TraceStep] = field(default_factory=list)
    iterations: int = 0
    stopped: str = "completed"  # "completed" | "max_iterations" | "no_final_estimate"


def _reasoning_summary(output: list[Any]) -> str:
    """Concatenate the reasoning-summary text emitted this turn (may be empty)."""
    parts: list[str] = []
    for item in output:
        if getattr(item, "type", None) == "reasoning":
            for summary in getattr(item, "summary", None) or []:
                text = getattr(summary, "text", None)
                if text:
                    parts.append(text)
    return "\n".join(parts).strip()


def _function_calls(output: list[Any]) -> list[Any]:
    """The function_call items in an output, in order (parallel calls included)."""
    return [item for item in output if getattr(item, "type", None) == "function_call"]


class EstimationAgent:
    """Drives the manual reason → act → observe loop over the Responses API.

    ``client`` is anything exposing ``responses.create`` (injected so tests can
    pass a fake). ``retriever`` is the injected budget-retrieval backend — the
    real S9/S10 pipeline or the offline stub; the loop does not care which.
    """

    def __init__(
        self,
        *,
        client: Any,
        retriever: BudgetRetriever,
        model: str,
        reasoning_effort: str = "medium",
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self._client = client
        self._retriever = retriever
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations

    async def _create(self, **kwargs: Any) -> Any:
        """Call the (blocking) Responses ``create`` off the event loop."""
        return await asyncio.to_thread(self._client.responses.create, **kwargs)

    async def _parse(self, **kwargs: Any) -> Any:
        """Call the (blocking) Responses ``parse`` off the event loop."""
        return await asyncio.to_thread(self._client.responses.parse, **kwargs)

    async def run(self, transcript: str) -> AgentResult:
        """Run the agent to completion and return the estimate plus the trace."""
        response = await self._create(
            model=self._model,
            instructions=self._system_prompt,
            input=transcript,
            tools=AGENT_TOOLS,
            reasoning={"effort": self._reasoning_effort, "summary": "auto"},
            store=True,
        )

        trace: list[TraceStep] = []
        turns = 0
        stopped = "completed"

        while True:
            calls = _function_calls(response.output)
            if not calls:
                break  # natural exit: the model returned a final message
            if turns >= self._max_iterations:
                stopped = "max_iterations"
                log.warning("agent_max_iterations", max_iterations=self._max_iterations)
                break

            turns += 1
            reasoning = _reasoning_summary(response.output)
            tool_outputs: list[dict[str, Any]] = []

            for position, call in enumerate(calls):
                args = _parse_arguments(call.arguments)
                result, observation = await self._run_one_call(call, args)
                # First call of the turn owns the turn's reasoning; the rest are
                # parallel calls issued in the same turn.
                step_reasoning = reasoning if position == 0 else "(parallel tool call in same turn)"
                trace.append(
                    TraceStep(
                        index=len(trace) + 1,
                        reasoning=step_reasoning,
                        tool=call.name,
                        arguments=args if isinstance(args, dict) else {},
                        observation=observation,
                    )
                )
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )

            response = await self._create(
                model=self._model,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=AGENT_TOOLS,
                reasoning={"effort": self._reasoning_effort, "summary": "auto"},
                store=True,
            )

        # One terminal parse turns the accumulated context into the structured
        # deliverable — skipped if we bailed out at the iteration ceiling.
        estimate: AgentEstimate | None = None
        if stopped != "max_iterations":
            estimate, stopped = await self._finalize(response.id, stopped)

        log.info(
            "agent_done",
            iterations=turns,
            stopped=stopped,
            trace_steps=len(trace),
            total_hours=(estimate.total_hours if estimate else None),
        )
        return AgentResult(
            estimate=estimate,
            trace=trace,
            iterations=turns,
            stopped=stopped,
        )

    async def _finalize(
        self, previous_response_id: str, stopped: str
    ) -> tuple[AgentEstimate | None, str]:
        """Elicit the final structured estimate via ``responses.parse``.

        Returns ``(estimate, stopped_reason)``. A parse failure is a stop reason
        (``no_final_estimate``), not a crash — the trace is still returned.
        """
        try:
            parsed = await self._parse(
                model=self._model,
                previous_response_id=previous_response_id,
                input=[{"role": "user", "content": FINAL_INSTRUCTION}],
                text_format=AgentEstimate,
                store=True,
            )
            return parsed.output_parsed, stopped
        except Exception as exc:  # noqa: BLE001 — a failed final parse is a stop reason.
            log.error("agent_final_parse_failed", error=str(exc)[:300])
            return None, "no_final_estimate"

    async def _run_one_call(self, call: Any, args: Any) -> tuple[Any, str]:
        """Execute a single function_call, returning (result_for_model, observation).

        A malformed-arguments or tool failure is fed back to the model as the tool
        output so it can recover, rather than crashing the loop.
        """
        if not isinstance(args, dict):
            msg = f"could not parse arguments for {call.name}"
            log.error("agent_bad_arguments", tool=call.name)
            return {"error": msg}, f"tool {call.name} failed: {msg}"
        try:
            return await execute_tool(call.name, args, retriever=self._retriever)
        except Exception as exc:  # noqa: BLE001 — surface tool errors to the model.
            log.error("agent_tool_failed", tool=call.name, error=str(exc)[:200])
            return {"error": str(exc)}, f"tool {call.name} failed: {exc}"


def _parse_arguments(raw: Any) -> dict[str, Any] | None:
    """Parse a function_call's ``arguments`` JSON string; None if malformed."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


_RULE = "=" * 78


def render_trace(result: AgentResult) -> str:
    """Render an :class:`AgentResult` as the console trace the exercise asks for.

    One block per step (reasoning / action / observation), then the final
    structured estimate. Kept pure so it is trivially testable and reusable by the
    run script.
    """
    lines: list[str] = [_RULE, "AGENT TRACE", _RULE]
    for step in result.trace:
        action = f"{step.tool}({json.dumps(step.arguments, ensure_ascii=False)})"
        lines += [
            f"STEP {step.index}",
            f"  reasoning:   {step.reasoning or '(none emitted)'}",
            f"  action:      {action}",
            f"  observation: {step.observation}",
            "",
        ]

    lines += [
        _RULE,
        f"FINAL ESTIMATE  (iterations={result.iterations}, stopped={result.stopped})",
        _RULE,
    ]
    estimate = result.estimate
    if estimate:
        for component in estimate.components:
            cites = (
                f"  [sources: {', '.join(map(str, component.cited_source_ids))}]"
                if component.cited_source_ids
                else ""
            )
            lines.append(f"  - {component.name}: {component.estimated_hours}h{cites}")
            if component.rationale:
                lines.append(f"      {component.rationale}")
        lines.append(f"\n  TOTAL: {estimate.total_hours}h    confidence: {estimate.confidence}")
        if estimate.assumptions:
            lines.append("  assumptions:")
            lines += [f"    · {assumption}" for assumption in estimate.assumptions]
    else:
        lines.append("  (the agent produced no final structured estimate)")

    return "\n".join(lines)
