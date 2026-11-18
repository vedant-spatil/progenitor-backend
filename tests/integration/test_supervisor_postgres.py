"""Gate 4: runs are resumable from the Postgres checkpointer; the ledger writes runs.Decision rows."""

import os
import uuid
from typing import Any

import pytest

from agents.graph import Deps
from agents.ledger import LedgerEntry, MemoryLedger
from agents.runner import paused_question, postgres_checkpointer, resume_run, start_run
from tests.fakes.llm import ScriptedLLM
from tests.unit.agents.test_graph import FakeReader, FakeTools, dec

pytestmark = pytest.mark.integration

URL = os.environ.get("GATE_DATABASE_URL", "postgresql://progenitor:progenitor@localhost:5432/progenitor")


async def test_paused_run_resumes_from_postgres_after_restart() -> None:
    run_id = f"test-{uuid.uuid4()}"
    script = {
        "supervisor": [
            dec("fetch_patient_data"),
            dec("ask_clinician", question="Any insulin use in the last 3 months?"),
            dec("finish", summary="done"),
        ]
    }
    llm = ScriptedLLM(script)

    # Process 1: runs until the interrupt, then everything is torn down (connection closed).
    async with postgres_checkpointer(URL) as saver:
        deps = Deps(llm=llm, call_tool=FakeTools(), reader=FakeReader(), ledger=MemoryLedger())
        out = await start_run(deps, saver, run_id, "p1")
    assert paused_question(out) == {"run_id": run_id, "question": "Any insulin use in the last 3 months?"}

    # Process 2: new connection, new graph object, same thread id -> continues from the checkpoint.
    async with postgres_checkpointer(URL) as saver:
        ledger = MemoryLedger()
        deps = Deps(llm=llm, call_tool=FakeTools(), reader=FakeReader(), ledger=ledger)
        out = await resume_run(deps, saver, run_id, resume={"answer": "No", "actor_user_id": 1})
    assert out["status"] == "finished"
    assert out["profile"]["sex"] == "female"  # loaded before the restart, restored from the checkpoint
    assert out["clinician_qa"][0]["answer"] == {"answer": "No", "actor_user_id": 1}
    assert [e.action for e in ledger.entries] == ["finish"]  # only the post-resume decision ran again


@pytest.mark.django_db
def test_django_ledger_writes_decision_rows() -> None:
    from apps.patients.models import Patient
    from apps.runs.ledger import DjangoLedger
    from apps.runs.models import Decision, Run

    run = Run.objects.create(patient=Patient.objects.create(source="synthea", fhir_id="p-ledger"))
    ledger = DjangoLedger()
    base: dict[str, Any] = {
        "run_id": str(run.id),
        "step": 0,
        "args": {},
        "rationale": "r",
        "state_refs": [],
        "confidence": 1.0,
    }
    ledger.record(
        LedgerEntry(action="draft_referrals", guard_outcome="rejected", guard_detail="needs approval", **base)
    )
    ledger.record(LedgerEntry(action="finish", guard_outcome="accepted", attempt=1, **base))
    rows = list(Decision.objects.filter(run=run).values_list("step", "attempt", "action", "guard_outcome"))
    assert rows == [(0, 0, "draft_referrals", "rejected"), (0, 1, "finish", "accepted")]
