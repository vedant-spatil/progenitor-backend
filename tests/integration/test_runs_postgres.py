"""Gate 5: runs through the service layer with the Postgres checkpointer and Redis events (scripted LLM).

- full flow: start -> ask_clinician pause -> answer -> shortlist -> review (approve one, reject one) -> referral
- resume: the worker dies mid-run; on restart the run is re-queued and completes from its checkpoint
- SSE: events published by the worker reach the stream, which ends when the run pauses
"""

from __future__ import annotations

import asyncio
import gc
import os
from typing import Any

import pytest
from asgiref.sync import sync_to_async
from django.contrib.auth.models import Group, User
from django.db import connection

from agents.graph import Deps
from agents.runner import postgres_checkpointer
from apps.audit.models import AuditEvent, AuditKind
from apps.audit.services import AuditedLLM, AuditedReader
from apps.patients.models import Patient
from apps.reviews.models import ReferralDraft
from apps.runs import events, services
from apps.runs.ledger import RunLedger
from apps.runs.models import Decision, Run, RunStatus
from tests.fakes.llm import ScriptedLLM
from tests.unit.agents.test_graph import CRITERIA, VERDICT, FakeReader, FakeTools, dec

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("flushable")]

URL = os.environ.get("GATE_DATABASE_URL", "postgresql://progenitor:progenitor@localhost:5432/progenitor")
A, B = "NCT00000000", "NCT00000001"


class WorkerKilled(BaseException):
    """Stands in for SIGKILL: not an Exception, so nothing in the run code catches it."""


class KillOnCall(ScriptedLLM):
    def __init__(self, replies: dict[str, list[dict[str, Any]]], kill_at_supervisor_call: int) -> None:
        super().__init__(replies)
        self.kill_at = kill_at_supervisor_call
        self.supervisor_calls = 0

    def complete(self, *, task: str, system: str, user: str, schema: Any) -> Any:
        if task == "supervisor":
            self.supervisor_calls += 1
            if self.supervisor_calls == self.kill_at:
                raise WorkerKilled
        return super().complete(task=task, system=system, user=user, schema=schema)


@pytest.fixture(scope="module")
def _restore_audit_trigger(django_db_setup: Any, django_db_blocker: Any) -> Any:
    yield
    with django_db_blocker.unblock(), connection.cursor() as cur:
        cur.execute("ALTER TABLE audit_auditevent ENABLE TRIGGER audit_no_truncate")
        # Worker-thread connections outlive their threads; end them so the test database can be dropped.
        cur.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = current_database() AND pid <> pg_backend_pid()"
        )


@pytest.fixture
def flushable(_restore_audit_trigger: Any, transactional_db: Any) -> Any:
    """These tests need committed rows (the run code writes from worker threads), so the test database is
    flushed after each one. Flush TRUNCATEs every table, which the append-only trigger blocks; turn that one
    trigger off in the test database until this module ends (test_audit_postgres checks it separately)."""
    yield
    gc.collect()  # drop DB connections held by finished worker threads before the flush
    with connection.cursor() as cur:
        cur.execute("ALTER TABLE audit_auditevent DISABLE TRIGGER audit_no_truncate")


def deps_for(run_id: str, llm: ScriptedLLM) -> Deps:
    return Deps(
        llm=AuditedLLM(llm, run_id),
        call_tool=FakeTools(2),
        reader=AuditedReader(FakeReader(), run_id),
        ledger=RunLedger(),
    )


@pytest.fixture
def enqueued(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(services, "_enqueue", lambda run, **kw: calls.append(kw))
    return calls


def _make_run(name: str) -> tuple[Run, User]:
    clinician = User.objects.create_user(f"clin-{name}")
    clinician.groups.add(Group.objects.get_or_create(name="clinician")[0])  # flush removes migration rows
    run = Run.objects.create(patient=Patient.objects.create(source="synthea", fhir_id=f"p-{name}"))
    return run, clinician


async def test_full_flow_pause_answer_review_draft(enqueued: list[dict[str, Any]]) -> None:
    run, clinician = await sync_to_async(_make_run)("flow")
    rid = str(run.id)
    llm = ScriptedLLM(
        {
            "supervisor": [
                dec("fetch_patient_data"),
                dec("ask_clinician", question="Any insulin use in the last 3 months?"),
                dec("search_trials", conditions="type 2 diabetes"),
                dec("parse_criteria", trial_ids=[A, B]),
                dec("judge_trials", trial_ids=[A, B]),
                dec("rank"),
                dec("finish", summary="ranked two trials"),
                dec("draft_referrals", trial_ids=[A]),
                dec("finish", summary="drafted"),
            ],
            "parse_criteria": [CRITERIA, CRITERIA],
            "judge_trial": [VERDICT, VERDICT],
            "rank_trials": [
                {"ranked": [{"nct_id": A, "rank": 1, "explanation": "x"}, {"nct_id": B, "rank": 2, "explanation": "y"}]}
            ],
            "draft_referral": [{"nct_id": A, "body": "Referral note for A"}],
        }
    )
    deps = deps_for(rid, llm)
    async with postgres_checkpointer(URL) as saver:
        assert await services.execute(rid, deps=deps, saver=saver) == RunStatus.WAITING_CLINICIAN
        await sync_to_async(services.answer)(rid, clinician, "No insulin")
        assert await services.execute(rid, answer=enqueued[-1]["answer"], deps=deps, saver=saver) == (
            RunStatus.AWAITING_REVIEW
        )
        items = [{"nct_id": A, "decision": "approve"}, {"nct_id": B, "decision": "reject"}]
        await sync_to_async(services.review)(rid, clinician, items)
        assert enqueued[-1] == {"approved": [A]}
        assert await services.execute(rid, approved=[A], deps=deps, saver=saver) == RunStatus.COMPLETED

    def check() -> None:
        r = Run.objects.get(pk=rid)
        assert r.summary == "drafted"
        assert r.finished_at is not None
        drafts = list(ReferralDraft.objects.filter(shortlist_item__run=r).select_related("shortlist_item__trial"))
        assert [(d.shortlist_item.trial.nct_id, d.review.reviewer_id) for d in drafts] == [(A, clinician.id)]
        decisions = Decision.objects.filter(run=r).count()
        kinds = AuditEvent.objects.filter(run=r)
        assert kinds.filter(kind=AuditKind.DECISION).count() == decisions == 9
        assert kinds.filter(kind=AuditKind.LLM_CALL).count() == len(llm.calls)
        assert kinds.filter(kind=AuditKind.PHI_READ).count() > 0
        assert kinds.filter(kind=AuditKind.ANSWER, actor=clinician).count() == 1
        assert kinds.filter(kind=AuditKind.APPROVAL, actor=clinician).count() == 2
        assert not kinds.filter(kind=AuditKind.APPROVAL, actor__isnull=True).exists()

    await sync_to_async(check)()


async def test_run_resumes_from_checkpoint_after_worker_dies(monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = await sync_to_async(_make_run)("kill")
    rid = str(run.id)
    first = KillOnCall(
        {"supervisor": [dec("fetch_patient_data"), dec("search_trials", conditions="type 2 diabetes")]},
        kill_at_supervisor_call=3,
    )
    async with postgres_checkpointer(URL) as saver:
        with pytest.raises(WorkerKilled):
            await services.execute(rid, deps=deps_for(rid, first), saver=saver)
    assert (await Run.objects.aget(pk=rid)).status == RunStatus.RUNNING  # orphaned by the dead worker

    requeued: list[str] = []
    monkeypatch.setattr("apps.runs.tasks.run_graph.delay", lambda run_id, **_: requeued.append(run_id))
    assert rid in await sync_to_async(services.recover_orphaned_runs)()
    assert rid in requeued

    second = ScriptedLLM(
        {
            "supervisor": [
                dec("parse_criteria", trial_ids=[A]),
                dec("judge_trials", trial_ids=[A]),
                dec("rank"),
                dec("finish", summary="done"),
            ],
            "parse_criteria": [CRITERIA],
            "judge_trial": [VERDICT],
            "rank_trials": [{"ranked": [{"nct_id": A, "rank": 1, "explanation": "x"}]}],
        }
    )
    async with postgres_checkpointer(URL) as saver:  # new connection: a restarted worker
        assert await services.execute(rid, deps=deps_for(rid, second), saver=saver) == RunStatus.AWAITING_REVIEW

    def actions() -> list[str]:
        return list(Decision.objects.filter(run_id=rid).order_by("step").values_list("action", flat=True))

    # Steps before the kill ran once; the new worker continued at step 3.
    assert await sync_to_async(actions)() == [
        "fetch_patient_data",
        "search_trials",
        "parse_criteria",
        "judge_trials",
        "rank",
        "finish",
    ]


async def test_sse_stream_relays_worker_events_until_pause() -> None:
    rid = "sse-test-run"
    gen = events.stream(rid, {"status": "running", "steps_used": 0}, heartbeat_s=0.2)
    assert (await anext(gen)).startswith("event: snapshot")

    async def worker() -> None:
        await asyncio.sleep(0.5)  # after the stream subscribed
        await asyncio.to_thread(events.publish, rid, "decision", {"step": 0, "action": "fetch_patient_data"})
        await asyncio.to_thread(events.publish, rid, "status", {"status": "waiting_clinician"})

    task = asyncio.create_task(worker())
    rest = [c async for c in gen]
    await task
    data = [c for c in rest if not c.startswith(":")]
    assert data == [
        'event: decision\ndata: {"step": 0, "action": "fetch_patient_data"}\n\n',
        'event: status\ndata: {"status": "waiting_clinician"}\n\n',
    ]
