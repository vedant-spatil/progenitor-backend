"""Gate 5: audited LLM, FHIR reader, and run ledger write audit events (ids and counts only)."""

from __future__ import annotations

from typing import Any

import pytest

from agents.ledger import LedgerEntry
from agents.schemas import Ranking
from apps.audit.models import AppendOnlyError, AuditEvent, AuditKind
from apps.audit.services import AuditedLLM, AuditedReader
from apps.patients.models import Patient
from apps.runs import ledger as run_ledger
from apps.runs.models import Decision, Run
from tests.fakes.llm import ScriptedLLM
from tests.unit.agents.test_graph import FakeReader

pytestmark = pytest.mark.django_db(transaction=True)  # wrappers write from worker threads


@pytest.fixture
def run() -> Run:
    return Run.objects.create(patient=Patient.objects.create(source="synthea", fhir_id="p-audit"))


def test_llm_calls_are_audited_without_prompt_text(run: Run) -> None:
    inner = ScriptedLLM({"rank_trials": [{"ranked": []}]})
    inner.usage.tokens_in = 10
    llm = AuditedLLM(inner, str(run.id))
    llm.complete(task="rank_trials", system="SECRET SYSTEM", user="SECRET RECORD", schema=Ranking)
    with pytest.raises(AssertionError):  # not scripted -> the call fails, still audited
        llm.complete(task="rank_trials", system="s", user="u", schema=Ranking)
    events = list(AuditEvent.objects.filter(run=run, kind=AuditKind.LLM_CALL))
    assert [(e.node, e.payload["status"]) for e in events] == [("rank_trials", "ok"), ("rank_trials", "error")]
    assert "SECRET" not in str([e.payload for e in events])
    assert llm.usage is inner.usage
    assert llm.model == "scripted"


async def test_fhir_reads_are_audited(run: Run) -> None:
    reader = AuditedReader(FakeReader(), str(run.id))
    await reader.read("Patient", "p1")
    matches, _ = await reader.search("Observation", {"patient": "p1", "category": "laboratory"})
    rows: list[AuditEvent] = [e async for e in AuditEvent.objects.filter(run=run, kind=AuditKind.PHI_READ)]
    assert [r.node for r in rows] == ["fhir_read", "fhir_search"]
    assert rows[0].payload == {"resource": "Patient/p1"}
    assert rows[1].payload == {
        "resource_type": "Observation",
        "params": ["category", "patient"],
        "matches": len(matches),
        "included": 0,
    }


def test_run_ledger_writes_decision_audit_and_event(run: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr("apps.runs.events.publish", lambda *a: published.append(a))
    entry = LedgerEntry(
        run_id=str(run.id),
        step=3,
        action="rank",
        args={},
        rationale="r",
        state_refs=["NCT1"],
        confidence=0.9,
        guard_outcome="accepted",
    )
    run_ledger.RunLedger().record(entry)
    assert Decision.objects.filter(run=run, step=3, action="rank").exists()
    event = AuditEvent.objects.get(run=run, kind=AuditKind.DECISION)
    assert event.payload == {"step": 3, "attempt": 0, "action": "rank", "guard_outcome": "accepted"}
    assert published[0][1] == "decision"
    assert published[0][2]["action"] == "rank"


def test_audit_rows_are_append_only(run: Run) -> None:
    event = AuditEvent.objects.create(kind=AuditKind.APPROVAL, run=run)
    with pytest.raises(AppendOnlyError):
        AuditEvent.objects.filter(pk=event.pk).update(kind="x")
    with pytest.raises(AppendOnlyError):
        event.delete()
