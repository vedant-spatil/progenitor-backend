"""Phase 8 step 2 (ADR 0008): one run step streamed as events, and the AgentCore entrypoint around it."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agents.audited import AuditedLLM, AuditedReader
from agents.events import EventChannel, parse_event
from agents.graph import Deps
from agents.runner import paused_question
from agents.runtime_step import StepRequest, run_step
from agents.tools import connections_from_env
from tests.fakes.llm import ScriptedLLM
from tests.unit.agents.test_graph import FakeReader, FakeTools, dec

SECRET = "Patient says SECRET-RECORD-TEXT"


def scripted(replies: dict[str, list[dict[str, Any]]]) -> Any:
    llm = ScriptedLLM(replies)  # shared across steps: one script spans the pause

    @asynccontextmanager
    async def make(req: StepRequest, ch: EventChannel) -> AsyncIterator[Deps]:
        yield Deps(
            llm=AuditedLLM(llm, req.run_id, ch.emit),
            call_tool=FakeTools(),
            reader=AuditedReader(FakeReader(), req.run_id, ch.emit),
            ledger=ch,
        )

    return make


async def collect(gen: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e async for e in gen]


async def test_start_pause_resume_finish() -> None:
    make = scripted(
        {
            "supervisor": [
                dec("fetch_patient_data"),
                dec("ask_clinician", question="Any insulin?"),
                dec("finish", summary="done"),
            ]
        }
    )
    saver = InMemorySaver()
    first = await collect(run_step(StepRequest(run_id="r1", patient_fhir_id="p1"), make, saver))
    assert [e["type"] for e in first][-1] == "result"
    assert [e["entry"]["action"] for e in first if e["type"] == "decision"] == ["fetch_patient_data", "ask_clinician"]
    assert {e["kind"] for e in first if e["type"] == "audit"} == {"llm_call", "phi_read"}
    assert paused_question(first[-1]["state"]) == {"run_id": "r1", "question": "Any insulin?"}
    for e in first:  # every event survives the JSON hop and parses back
        parse_event(json.dumps(e))

    second = await collect(
        run_step(StepRequest(run_id="r1", patient_fhir_id="p1", answer={"answer": "No"}), make, saver)
    )
    final = second[-1]
    assert final["type"] == "result"
    assert final["state"]["status"] == "finished"
    assert [e["entry"]["action"] for e in second if e["type"] == "decision"] == ["finish"]


async def test_failure_ends_with_class_name_only() -> None:
    @asynccontextmanager
    async def broken(req: StepRequest, ch: EventChannel) -> AsyncIterator[Deps]:
        raise RuntimeError(SECRET)
        yield  # pragma: no cover

    events = await collect(run_step(StepRequest(run_id="r2", patient_fhir_id="p1"), broken, InMemorySaver()))
    assert events == [{"type": "error", "error": "RuntimeError"}]
    assert SECRET not in json.dumps(events)


async def test_config_extra_reaches_the_checkpointer() -> None:
    seen: list[dict[str, Any]] = []

    class SpySaver(InMemorySaver):
        async def aget_tuple(self, config: Any) -> Any:
            seen.append(dict(config["configurable"]))
            return await super().aget_tuple(config)

    make = scripted({"supervisor": [dec("finish", summary="done")]})
    await collect(run_step(StepRequest(run_id="r3", patient_fhir_id="p1"), make, SpySaver(), {"actor_id": "svc"}))
    assert seen
    assert all(c.get("actor_id") == "svc" and c["thread_id"] == "r3" for c in seen)


def test_step_request_rejects_bad_payloads() -> None:
    with pytest.raises(ValueError, match="run_id"):
        StepRequest.model_validate({"patient_fhir_id": "p1"})
    with pytest.raises(ValueError, match="run_id"):
        StepRequest.model_validate({"run_id": "", "patient_fhir_id": "p1"})


def test_mcp_connections_carry_the_service_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_SERVICE_TOKEN", raising=False)
    assert all("headers" not in c for c in connections_from_env().values())
    monkeypatch.setenv("AGENT_SERVICE_TOKEN", "tok-123")
    monkeypatch.setenv("FHIR_MCP_URL", "https://cdn.example/internal/mcp/fhir/mcp")
    conns = connections_from_env()
    assert dict(conns["fhir"])["url"] == "https://cdn.example/internal/mcp/fhir/mcp"
    assert all(c.get("headers") == {"Authorization": "Bearer tok-123"} for c in conns.values())


# ---------------- the AgentCore entrypoint (needs the agentcore dependency group) ----------------


async def test_entrypoint_streams_events_and_rejects_bad_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("bedrock_agentcore")
    pytest.importorskip("langgraph_checkpoint_aws")
    from agents import agentcore_app

    saver = InMemorySaver()
    monkeypatch.setattr(agentcore_app, "_saver", lambda: saver)
    monkeypatch.setattr(agentcore_app, "_deps", scripted({"supervisor": [dec("finish", summary="ok")]}))
    events = await collect(agentcore_app.invoke({"run_id": "r4", "patient_fhir_id": "p1"}))
    assert events[-1]["type"] == "result"
    assert events[-1]["state"]["status"] == "finished"
    assert await collect(agentcore_app.invoke({"run_id": "r5"})) == [{"type": "error", "error": "BadRequest"}]


def test_entrypoint_saver_uses_memory_id_and_region(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("bedrock_agentcore")
    pytest.importorskip("langgraph_checkpoint_aws")
    from agents import agentcore_app

    monkeypatch.setenv("AGENTCORE_MEMORY_ID", "mem-abc")
    monkeypatch.setenv("AWS_REGION", "ap-south-1")
    saver = agentcore_app._saver()
    assert saver.memory_id == "mem-abc"
