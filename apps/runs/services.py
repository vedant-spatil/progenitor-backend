"""Run lifecycle between the API, the Celery worker, and the supervisor graph.

    POST /api/runs            -> queued -> (worker) running -> waiting_clinician | awaiting_review | completed | failed
    POST /api/runs/{id}/answer  waiting_clinician -> queued -> (worker) resumes the paused graph
    POST /api/runs/{id}/review  awaiting_review -> queued (some approved: supervisor may draft referrals)
                                                -> completed (all rejected: no referrals)

The LangGraph checkpoint (thread id = run id) is the source of truth for agent state; Django rows
are the persisted, reviewable view of it. A run left `running` by a dead worker is re-queued when
a worker starts (`recover_orphaned_runs`) and continues from its last checkpoint.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from django.db import transaction
from django.utils import timezone

from agents.graph import Deps
from agents.live_deps import live_deps
from agents.runner import advance, paused_question, postgres_checkpointer
from apps.audit.models import AuditKind
from apps.audit.services import audit
from apps.patients.models import Patient
from apps.reviews.models import ReferralDraft, Review, ReviewDecision
from apps.runs.events import publish
from apps.runs.ledger import RunLedger
from apps.runs.models import Run, RunStatus, ShortlistItem, ShortlistStatus, Verdict
from apps.trials.models import Criterion, Trial

log = logging.getLogger(__name__)


class RunStateError(Exception):
    """The run is not in a state that allows this action (HTTP 409)."""


# ---------------- API-side actions (sync, inside a request) ----------------


def _enqueue(run: Run, **kwargs: Any) -> None:
    from apps.runs.tasks import run_graph

    transaction.on_commit(lambda: run_graph.delay(str(run.id), **kwargs))


def start(patient: Patient, user: Any, config: dict[str, Any] | None = None) -> Run:
    with transaction.atomic():
        run = Run.objects.create(patient=patient, created_by=user, config=config or {})
        audit(AuditKind.RUN_CREATED, run_id=str(run.id), actor_id=user.id, node="api", patient=patient.pk)
        _enqueue(run)
    return run


def answer(run_id: Any, user: Any, text: str) -> Run:
    with transaction.atomic():
        run = Run.objects.select_for_update().get(pk=run_id)
        if run.status != RunStatus.WAITING_CLINICIAN:
            raise RunStateError(f"run is {run.status}, not waiting for a clinician answer")
        audit(AuditKind.ANSWER, run_id=str(run.id), actor_id=user.id, node="api", answer_chars=len(text))
        run.status, run.pending_question = RunStatus.QUEUED, ""
        run.save(update_fields=["status", "pending_question"])
        _enqueue(run, answer={"answer": text, "actor_user_id": user.id})
    publish(str(run.id), "status", {"status": run.status})
    return run


def review(run_id: Any, user: Any, items: list[dict[str, Any]]) -> Run:
    """Record one review per shortlisted trial. approve/edit approve it; reject rejects it."""
    with transaction.atomic():
        run = Run.objects.select_for_update().get(pk=run_id)
        if run.status != RunStatus.AWAITING_REVIEW:
            raise RunStateError(f"run is {run.status}, not awaiting review")
        by_nct = {i.trial.nct_id: i for i in run.shortlist.select_related("trial")}
        unknown = sorted({it["nct_id"] for it in items} - set(by_nct))
        if unknown:
            raise ValueError(f"not on this run's shortlist: {', '.join(unknown)}")
        approved: list[str] = []
        for it in items:
            item = by_nct[it["nct_id"]]
            decision = it["decision"]
            Review.objects.create(
                run=run,
                shortlist_item=item,
                reviewer=user,
                decision=decision,
                comment=it.get("comment", ""),
                edits=it.get("edits") or {},
            )
            ok = decision in (ReviewDecision.APPROVE, ReviewDecision.EDIT)
            item.status = ShortlistStatus.APPROVED if ok else ShortlistStatus.REJECTED
            item.save(update_fields=["status", "updated_at"])
            audit(
                AuditKind.APPROVAL,
                run_id=str(run.id),
                actor_id=user.id,
                node="api",
                nct_id=item.trial.nct_id,
                decision=decision,
            )
            if ok:
                approved.append(item.trial.nct_id)
        if approved:
            run.status = RunStatus.QUEUED
            _enqueue(run, approved=approved)
        else:
            run.status, run.finished_at = RunStatus.COMPLETED, timezone.now()
        run.save(update_fields=["status", "finished_at"])
    publish(str(run.id), "status", {"status": run.status})
    return run


# ---------------- worker side ----------------


@asynccontextmanager
async def default_deps(run: Run) -> AsyncIterator[Deps]:
    """Live dependencies for the in-process path: audit records and ledger entries become Django rows."""
    async with live_deps(str(run.id), run.patient.fhir_id, audit, RunLedger()) as deps:
        yield deps


def _set_status(run_id: str, status: str, **fields: Any) -> None:
    Run.objects.filter(pk=run_id).update(status=status, **fields)
    publish(run_id, "status", {"status": status, **{k: v for k, v in fields.items() if k == "error"}})


async def execute(
    run_id: str,
    *,
    answer: Any = None,
    approved: list[str] | None = None,
    deps: Deps | None = None,
    saver: Any = None,
) -> str:
    """Advance one run as far as it goes, then persist the outcome. Returns the new status."""
    run = await asyncio.to_thread(Run.objects.select_related("patient").get, pk=run_id)
    if run.status in (RunStatus.COMPLETED, RunStatus.CANCELLED):
        return run.status
    started = run.started_at or timezone.now()
    await asyncio.to_thread(_set_status, run_id, RunStatus.RUNNING, started_at=started)
    try:
        async with _maybe(saver, postgres_checkpointer) as cp, _maybe(deps, lambda: default_deps(run)) as d:
            before = (d.llm.usage.tokens_in, d.llm.usage.tokens_out)
            out = await advance(
                d,
                cp,
                run_id,
                run.patient.fhir_id,
                as_of=run.config.get("as_of"),
                answer=answer,
                approved_trial_ids=approved,
            )
            used = (d.llm.usage.tokens_in - before[0], d.llm.usage.tokens_out - before[1])
            model = d.llm.model
    except Exception as exc:
        # Error class only at ERROR level: messages from tools or the LLM may quote record text.
        log.error("run_failed run=%s error=%s", run_id, type(exc).__name__)
        log.debug("run_failed detail", exc_info=True)
        await asyncio.to_thread(
            _set_status, run_id, RunStatus.FAILED, error=type(exc).__name__[:255], finished_at=timezone.now()
        )
        return RunStatus.FAILED
    return await asyncio.to_thread(persist, run_id, out, used, model)


@asynccontextmanager
async def _maybe(given: Any, factory: Any) -> AsyncIterator[Any]:
    if given is not None:
        yield given
        return
    async with factory() as made:
        yield made


# ---------------- persistence (graph state -> Django rows) ----------------


def persist(run_id: str, out: dict[str, Any], used: tuple[int, int] = (0, 0), model: str = "") -> str:
    with transaction.atomic():
        run = Run.objects.select_for_update().get(pk=run_id)
        run.steps_used = int(out.get("steps", run.steps_used))
        run.tokens_in += used[0]
        run.tokens_out += used[1]
        question = paused_question(out)
        if question:
            run.status, run.pending_question = RunStatus.WAITING_CLINICIAN, str(question["question"])
        else:
            run.summary = str(out.get("summary") or "")[:4000]
            reason = out.get("finish_reason") or ""
            run.error = "" if reason in ("", "supervisor") else str(reason)[:255]
            if out.get("approved_trial_ids"):
                _save_referrals(run, out, model)
                run.status, run.finished_at = RunStatus.COMPLETED, timezone.now()
            elif _save_shortlist(run, out, model):
                run.status = RunStatus.AWAITING_REVIEW
            else:
                run.status, run.finished_at = RunStatus.COMPLETED, timezone.now()
        run.save()
    publish(run_id, "status", {"status": run.status, "steps_used": run.steps_used})
    return run.status


def _upsert_trial(nct: str, t: dict[str, Any]) -> Trial:
    s = t.get("summary") or {}
    trial, _ = Trial.objects.update_or_create(
        nct_id=nct,
        defaults={
            "title": s.get("title", ""),
            "overall_status": s.get("status", ""),
            "phases": s.get("phases") or [],
            "conditions": s.get("conditions") or [],
            "minimum_age": s.get("minimum_age") or "",
            "maximum_age": s.get("maximum_age") or "",
            "sex": s.get("sex") or "",
            "eligibility_text": (t.get("eligibility") or {}).get("criteria_text", ""),
            "fetched_at": timezone.now(),
        },
    )
    return trial


def _save_shortlist(run: Run, out: dict[str, Any], model: str) -> int:
    if run.shortlist.exists():
        return run.shortlist.count()
    trials = out.get("trials") or {}
    saved = 0
    for row in out.get("shortlist") or []:
        nct = row["nct_id"]
        if nct not in trials:
            continue
        trial = _upsert_trial(nct, trials[nct])
        ShortlistItem.objects.create(
            run=run,
            trial=trial,
            rank=int(row["rank"]),
            explanation=row.get("explanation", ""),
            counts=row.get("counts") or {},
        )
        _save_verdicts(run, trial, trials[nct], model)
        saved += 1
    return saved


def _save_verdicts(run: Run, trial: Trial, t: dict[str, Any], model: str) -> None:
    by_index: dict[int, Criterion] = {}
    for c in t.get("criteria") or []:
        by_index[int(c["index"])], _ = Criterion.objects.update_or_create(
            trial=trial,
            position=int(c["index"]),
            defaults={
                "kind": c["kind"],
                "source_text": c["source_text"],
                "entity": (c.get("entity") or "")[:255],
                "entity_codes": [code for e in c.get("entities") or [] for code in e.get("codes") or []],
                "operator": c.get("operator") or "",
                "value": (c.get("value") or "")[:128],
                "unit": (c.get("unit") or "")[:32],
                "temporal_window": (c.get("temporal_window") or "")[:64],
                "parser_model": model,
            },
        )
    for v in t.get("verdicts") or []:
        criterion = by_index.get(int(v["criterion_index"]))
        if criterion is None:
            continue
        Verdict.objects.update_or_create(
            run=run,
            criterion=criterion,
            defaults={
                "trial": trial,
                "status": v["status"],
                "evidence_refs": v.get("evidence_refs") or [],
                "reason": v.get("reason", ""),
                "model": model,
            },
        )


def _save_referrals(run: Run, out: dict[str, Any], model: str) -> None:
    approved = set(out.get("approved_trial_ids") or [])
    for note in out.get("referrals") or []:
        nct = note.get("nct_id")
        if nct not in approved:
            continue  # defence in depth: the graph guard already blocks drafts for unapproved trials
        item = run.shortlist.filter(trial__nct_id=nct, status=ShortlistStatus.APPROVED).first()
        if item is None or item.referrals.exists():
            continue
        approval = (
            item.reviews.filter(decision__in=[ReviewDecision.APPROVE, ReviewDecision.EDIT])
            .order_by("-created_at")
            .first()
        )
        if approval is None:
            continue
        ReferralDraft.objects.create(shortlist_item=item, review=approval, body=note.get("body", ""), model=model)


def recover_orphaned_runs() -> list[str]:
    """Re-queue runs left `running` by a worker that died. Called when a worker starts.

    Assumes one worker deployment (the compose `worker` service): at its start no run can be running.
    """
    from apps.runs.tasks import run_graph

    ids = [str(pk) for pk in Run.objects.filter(status=RunStatus.RUNNING).values_list("pk", flat=True)]
    for run_id in ids:
        log.info("run_requeued run=%s", run_id)
        run_graph.delay(run_id)
    return ids
