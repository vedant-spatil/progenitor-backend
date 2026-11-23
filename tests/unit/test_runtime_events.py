"""Phase 8 step 1 (ADR 0008): an agent step runs without Django and its streamed events rebuild the same records."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from asgiref.sync import sync_to_async
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.audited import AuditedLLM as PlainAuditedLLM
from agents.audited import AuditedReader as PlainAuditedReader
from agents.events import (
    AuditRecord,
    DecisionEvent,
    ErrorEvent,
    EventChannel,
    ResultEvent,
    parse_event,
    serialize_result,
)
from agents.graph import Deps, build_graph, initial_state
from agents.runner import paused_question
from apps.audit.models import AuditEvent
from apps.audit.services import AuditedLLM, AuditedReader
from apps.patients.models import Patient
from apps.runs.ledger import RunLedger
from apps.runs.models import Decision, Run
from apps.runs.stream import RuntimeStepError, apply_events
from tests.fakes.llm import ScriptedLLM
from tests.unit.agents.test_graph import CRITERIA, VERDICT, FakeReader, FakeTools, dec

A, B = "NCT00000000", "NCT00000001"


def script() -> dict[str, list[dict[str, Any]]]:
    return {
        "supervisor": [
            dec("fetch_patient_data"),
            dec("ask_clinician", question="Any insulin in the last 3 months?"),
            dec("search_trials", conditions="type 2 diabetes"),
            dec("parse_criteria", trial_ids=[A, B]),
            dec("judge_trials", trial_ids=[A, B]),
            dec("rank"),
            dec("finish", summary="done"),
        ],
        "parse_criteria": [CRITERIA, CRITERIA],
        "judge_trial": [VERDICT, VERDICT],
        "rank_trials": [
            {"ranked": [{"nct_id": B, "rank": 1, "explanation": "x"}, {"nct_id": A, "rank": 2, "explanation": "y"}]}
        ],
    }


def cfg(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}, "recursion_limit": 100}


# ---------------- serialization ----------------


async def test_serialized_result_is_json_and_keeps_the_question() -> None:
    llm = ScriptedLLM({"supervisor": [dec("fetch_patient_data"), dec("ask_clinician", question="Pregnant?")]})
    deps = Deps(llm=llm, call_tool=FakeTools(), reader=FakeReader(), ledger=RunLedgerStub())
    out = await build_graph(deps, checkpointer=InMemorySaver()).ainvoke(initial_state("r1", "p1"), cfg("s1"))
    safe = serialize_result(out)
    assert json.loads(json.dumps(safe)) == safe
    assert paused_question(out) == paused_question(safe) == {"run_id": "r1", "question": "Pregnant?"}
    assert paused_question(serialize_result({"status": "finished"})) is None


def test_events_round_trip_as_json() -> None:
    events = [
        AuditRecord(kind="llm_call", run_id="r", node="supervisor", payload={"tokens_in": 3}),
        ResultEvent(state={"status": "finished"}),
        ErrorEvent(error="LlmError"),
    ]
    for e in events:
        assert parse_event(e.model_dump_json()) == e
        assert parse_event(e.model_dump()) == e
    with pytest.raises(ValueError, match="type"):
        parse_event('{"type": "bogus"}')


async def test_channel_accepts_events_from_worker_threads_in_order() -> None:
    ch = EventChannel()
    ch.emit("llm_call", run_id="r", node="a")
    await asyncio.to_thread(ch.emit, "phi_read", run_id="r", node="b")
    ch.emit("llm_call", run_id="r", node="c")
    await asyncio.sleep(0)
    ch.close()
    assert [e.node async for e in ch.events() if isinstance(e, AuditRecord)] == ["a", "b", "c"]


# ---------------- same records on both paths ----------------


class RunLedgerStub:
    def record(self, entry: Any) -> None:
        pass


def _new_run(fhir_id: str) -> str:
    return str(Run.objects.create(patient=Patient.objects.create(source="synthea", fhir_id=fhir_id)).id)


def _records(run_id: str) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    decisions = [
        (d.step, d.attempt, d.action, d.rationale, d.state_refs, d.guard_outcome, d.tokens_in, d.tokens_out)
        for d in Decision.objects.filter(run_id=run_id).order_by("step", "attempt")
    ]
    audits = [(e.kind, e.node, e.payload) for e in AuditEvent.objects.filter(run_id=run_id).order_by("id")]
    return decisions, audits


async def _celery_path(run_id: str) -> None:
    """In-process, as apps.runs.services does today: Django sinks write as the graph runs."""
    llm = AuditedLLM(ScriptedLLM(script()), run_id)
    reader = AuditedReader(FakeReader(), run_id)
    # parallel=1: sqlite in unit tests cannot take concurrent writes from the judge fan-out threads.
    deps = Deps(llm=llm, call_tool=FakeTools(), reader=reader, ledger=RunLedger(), parallel=1)
    graph = build_graph(deps, checkpointer=InMemorySaver())
    await graph.ainvoke(initial_state(run_id, "p1"), cfg(run_id))
    await graph.ainvoke(Command(resume={"answer": "No"}), cfg(run_id))


async def _runtime_step(graph: Any, ch: EventChannel, payload: Any, run_id: str) -> list[Any]:
    """What the AgentCore entrypoint does: run the step, stream its events, end with the result."""
    out = await graph.ainvoke(payload, cfg(run_id))
    ch.put(ResultEvent(state=serialize_result(out)))
    ch.close()
    return [e async for e in ch.events()]


async def _runtime_path(run_id: str) -> list[dict[str, Any]]:
    """No Django sinks: events only, then applied by the caller. Returns each step's final state."""
    saver = InMemorySaver()
    llm = ScriptedLLM(script())  # one LLM for both steps, as one script spans the pause
    states = []
    steps: list[Any] = [initial_state(run_id, "p1"), Command(resume={"answer": "No"})]
    for payload in steps:
        ch = EventChannel()
        deps = Deps(
            llm=PlainAuditedLLM(llm, run_id, ch.emit),
            call_tool=FakeTools(),
            reader=PlainAuditedReader(FakeReader(), run_id, ch.emit),
            ledger=ch,
            parallel=1,
        )
        events = await _runtime_step(build_graph(deps, checkpointer=saver), ch, payload, run_id)
        assert all(json.loads(e.model_dump_json()) == e.model_dump(mode="json") for e in events)
        states.append(await sync_to_async(apply_events)(events))
    return states


@pytest.mark.django_db(transaction=True)
async def test_runtime_events_rebuild_the_same_records(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.runs import events as run_events

    monkeypatch.setattr(run_events, "publish", lambda *a, **k: None)  # no Redis in unit tests
    celery_run = await sync_to_async(_new_run)("p-celery")
    runtime_run = await sync_to_async(_new_run)("p-runtime")
    await _celery_path(celery_run)
    first, final = await _runtime_path(runtime_run)

    want_dec, want_audit = await sync_to_async(_records)(celery_run)
    got_dec, got_audit = await sync_to_async(_records)(runtime_run)
    assert [d[2] for d in got_dec] == [
        "fetch_patient_data",
        "ask_clinician",
        "search_trials",
        "parse_criteria",
        "judge_trials",
        "rank",
        "finish",
    ]
    assert got_dec == want_dec
    assert sorted(map(json.dumps, got_audit)) == sorted(map(json.dumps, want_audit))
    assert {k for k, _, _ in got_audit} >= {"llm_call", "phi_read", "decision"}
    assert paused_question(first) == {"run_id": runtime_run, "question": "Any insulin in the last 3 months?"}
    assert final["status"] == "finished"
    assert [r["nct_id"] for r in final["shortlist"]] == [B, A]


def test_stream_without_result_or_with_error_fails() -> None:
    with pytest.raises(RuntimeStepError, match="NoResult"):
        apply_events([])
    with pytest.raises(RuntimeStepError, match="LlmError"):
        apply_events([ErrorEvent(error="LlmError")])


@pytest.mark.django_db
def test_decision_event_writes_row_audit_and_live_event(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.ledger import LedgerEntry
    from apps.runs import events as run_events

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(run_events, "publish", lambda run_id, event, data: sent.append((run_id, event)))
    run_id = _new_run("p-one")
    entry = LedgerEntry(run_id, 1, "rank", {}, "why", ["Observation/o1"], 0.9, "accepted")
    ch_event = DecisionEvent(entry=json.loads(json.dumps(entry.__dict__)))
    state = apply_events([ch_event, ResultEvent(state={"status": "finished"})])
    assert state == {"status": "finished"}
    assert Decision.objects.get(run_id=run_id).action == "rank"
    assert AuditEvent.objects.filter(run_id=run_id, kind="decision").count() == 1
    assert sent == [(run_id, "decision")]
