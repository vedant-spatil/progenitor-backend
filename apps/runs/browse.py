"""Read-only app views (PLAN.md Phase 7c, ADR 0007): overview, patient detail, trials, referral inbox, audit, evals.

The frontend mirrors these shapes in lib/api/schemas.ts (Zod); change both together.
Every view is GET only. Run data needs coordinator or clinician; the audit log needs admin and never returns
event payloads.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db.models import Count, Q, QuerySet
from django.shortcuts import get_object_or_404
from django.utils.dateparse import parse_datetime
from pydantic import ValidationError
from rest_framework import serializers
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from agents.schemas import PatientProfile
from apps.accounts.roles import CLINICIAN, COORDINATOR, CanReadRuns, IsAdmin, has_role
from apps.audit.models import AuditEvent, AuditKind
from apps.audit.services import audit
from apps.patients.models import Patient
from apps.reviews.models import ReferralDraft, ReferralStatus
from apps.runs.models import Decision, Run, RunStatus, ShortlistItem
from apps.runs.serializers import RunListSerializer
from apps.trials.models import Trial

PAGE = 50
TRIAL_PAGE = 25
QUEUE_LIMIT = 20
RECENT_DECISIONS = 10


def _int(value: str | None, default: int, lo: int = 1, hi: int = 10_000) -> int:
    try:
        return max(lo, min(hi, int(value or default)))
    except ValueError:
        return default


def _before(qs: QuerySet[Any], request: Request, field: str = "created_at") -> QuerySet[Any]:
    """Cursor paging: rows strictly older than ?before=<ISO time>. A bad cursor is ignored."""
    raw = request.query_params.get("before")
    when = parse_datetime(raw) if raw else None
    return qs.filter(**{f"{field}__lt": when}) if when else qs


def _page(rows: list[Any], field: str = "created_at") -> tuple[list[Any], str | None]:
    """rows holds PAGE + 1 items at most; the extra one only says that there is a next page."""
    if len(rows) <= PAGE:
        return rows, None
    rows = rows[:PAGE]
    return rows, getattr(rows[-1], field).isoformat()


# ---------------- runs: filters on the list ----------------


def filter_runs(request: Request) -> tuple[list[Run], str | None]:
    """GET /api/runs filters. With no parameters: the 50 newest runs, as in 7b."""
    qs = Run.objects.select_related("patient").order_by("-created_at")
    p = request.query_params
    statuses = [s for s in p.get("status", "").split(",") if s in RunStatus.values]
    if statuses:
        qs = qs.filter(status__in=statuses)
    if p.get("patient"):
        qs = qs.filter(patient_id=_int(p.get("patient"), 0, lo=0, hi=2**31))
    if p.get("q"):
        qs = qs.filter(patient__label__icontains=p["q"][:64])
    if p.get("mine") in ("1", "true"):
        qs = qs.filter(created_by_id=request.user.pk)
    qs = _before(qs, request)
    return _page(list(qs[: PAGE + 1]))


# ---------------- overview ----------------


class OverviewView(APIView):
    """Counts by status, the signed-in user's queue (by role), and the latest decisions (action only)."""

    permission_classes = [CanReadRuns]

    def get(self, request: Request) -> Response:
        counts = {s: 0 for s in RunStatus.values}
        for row in Run.objects.values("status").annotate(n=Count("id")):
            counts[row["status"]] = row["n"]

        queue: list[dict[str, Any]] = []

        def add(qs: QuerySet[Run], reason: str, since: str) -> None:
            for run in qs.select_related("patient").order_by(since)[:QUEUE_LIMIT]:
                queue.append(
                    {
                        "run_id": str(run.id),
                        "patient_label": run.patient.label,
                        "status": run.status,
                        "reason": reason,
                        "since": (run.finished_at or run.started_at or run.created_at).isoformat(),
                    }
                )

        if has_role(request.user, CLINICIAN):
            add(Run.objects.filter(status=RunStatus.WAITING_CLINICIAN), "answer_question", "created_at")
            add(Run.objects.filter(status=RunStatus.AWAITING_REVIEW), "review_shortlist", "created_at")
        if has_role(request.user, COORDINATOR):
            add(Run.objects.filter(status=RunStatus.FAILED), "failed", "-created_at")
            drafts = ReferralDraft.objects.filter(status=ReferralStatus.DRAFT).values("shortlist_item__run_id")
            add(Run.objects.filter(status=RunStatus.COMPLETED, id__in=drafts), "referral_drafts", "-created_at")

        recent = (
            Decision.objects.select_related("run__patient")
            .order_by("-created_at", "-id")
            .only("run__id", "run__patient__label", "step", "action", "guard_outcome", "created_at")[:RECENT_DECISIONS]
        )
        return Response(
            {
                "counts": counts,
                "queue": queue,
                "recent": [
                    {
                        "run_id": str(d.run.id),
                        "patient_label": d.run.patient.label,
                        "step": d.step,
                        "action": d.action,
                        "guard_outcome": d.guard_outcome,
                        "created_at": d.created_at.isoformat(),
                    }
                    for d in recent
                ],
            }
        )


# ---------------- patient detail ----------------


class PatientDetailView(APIView):
    """One patient: cached Profiler output (validated), run history, trials shortlisted for them."""

    permission_classes = [CanReadRuns]

    def get(self, request: Request, patient_id: int) -> Response:
        patient = get_object_or_404(Patient, pk=patient_id)
        profile: dict[str, Any] | None = None
        if patient.profile:
            try:
                profile = PatientProfile.model_validate(patient.profile).model_dump(mode="json")
            except ValidationError:
                profile = None  # an old cache shape: show none rather than raw JSON
            # The profile restates record facts: reading it is a PHI read.
            audit(AuditKind.PHI_READ, actor_id=request.user.id, node="api_patient", patient_id=patient.pk)
        runs = Run.objects.filter(patient=patient).select_related("patient").order_by("-created_at")[:PAGE]
        shortlisted = (
            ShortlistItem.objects.filter(run__patient=patient)
            .select_related("trial", "run")
            .order_by("-run__created_at", "rank")[:PAGE]
        )
        return Response(
            {
                "id": patient.pk,
                "label": patient.label,
                "source": patient.source,
                "profile_cached_at": patient.profile_cached_at.isoformat() if patient.profile_cached_at else None,
                "profile": profile,
                "runs": RunListSerializer(runs, many=True).data,
                "shortlisted": [
                    {
                        "run_id": str(s.run_id),
                        "nct_id": s.trial.nct_id,
                        "title": s.trial.title,
                        "rank": s.rank,
                        "counts": s.counts,
                        "status": s.status,
                        "created_at": s.run.created_at.isoformat(),
                    }
                    for s in shortlisted
                ],
            }
        )


# ---------------- trials ----------------


class TrialRowSerializer(serializers.ModelSerializer[Trial]):
    criteria_count = serializers.IntegerField(read_only=True)
    shortlisted_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Trial
        fields = [
            "nct_id",
            "title",
            "overall_status",
            "phases",
            "conditions",
            "minimum_age",
            "maximum_age",
            "sex",
            "criteria_count",
            "shortlisted_count",
        ]


class TrialListView(APIView):
    """Registry copy of trials. ?q= text (NCT id, title, condition), ?status=, ?phase=, ?shortlisted=1, ?page=."""

    permission_classes = [CanReadRuns]

    def get(self, request: Request) -> Response:
        p = request.query_params
        qs = Trial.objects.annotate(
            criteria_count=Count("criteria", distinct=True), shortlisted_count=Count("shortlist_items", distinct=True)
        )
        if p.get("q"):
            q = p["q"][:100]
            # conditions is a JSON list: matching its text form keeps this portable (sqlite and Postgres).
            qs = qs.filter(Q(nct_id__icontains=q) | Q(title__icontains=q) | Q(conditions__icontains=q))
        if p.get("status"):
            qs = qs.filter(overall_status=p["status"][:40])
        if p.get("phase"):
            qs = qs.filter(phases__icontains=p["phase"][:20])
        if p.get("shortlisted") in ("1", "true"):
            qs = qs.filter(shortlisted_count__gt=0)
        total = qs.count()
        page = _int(p.get("page"), 1)
        rows = qs.order_by("-shortlisted_count", "nct_id")[(page - 1) * TRIAL_PAGE : page * TRIAL_PAGE]
        statuses = sorted(s for s in Trial.objects.values_list("overall_status", flat=True).distinct() if s)
        return Response(
            {
                "items": TrialRowSerializer(rows, many=True).data,
                "total": total,
                "page": page,
                "page_size": TRIAL_PAGE,
                "statuses": statuses,
            }
        )


class TrialDetailView(APIView):
    """One trial: criteria in order, plus each run that shortlisted it (counts and review status, no reasons)."""

    permission_classes = [CanReadRuns]

    def get(self, request: Request, nct_id: str) -> Response:
        trial = get_object_or_404(Trial, nct_id=nct_id.upper())
        items = (ShortlistItem.objects.filter(trial=trial).select_related("run__patient").order_by("-run__created_at"))[
            :PAGE
        ]
        return Response(
            {
                "nct_id": trial.nct_id,
                "title": trial.title,
                "overall_status": trial.overall_status,
                "phases": trial.phases,
                "conditions": trial.conditions,
                "minimum_age": trial.minimum_age,
                "maximum_age": trial.maximum_age,
                "sex": trial.sex,
                "healthy_volunteers": trial.healthy_volunteers,
                "source_last_updated": trial.source_last_updated.isoformat() if trial.source_last_updated else None,
                "fetched_at": trial.fetched_at.isoformat(),
                "criteria": [
                    {"position": c.position, "kind": c.kind, "source_text": c.source_text}
                    for c in trial.criteria.order_by("position")
                ],
                "matches": [
                    {
                        "run_id": str(s.run_id),
                        "patient_id": s.run.patient_id,
                        "patient_label": s.run.patient.label,
                        "rank": s.rank,
                        "counts": s.counts,
                        "status": s.status,
                        "created_at": s.run.created_at.isoformat(),
                    }
                    for s in items
                ],
            }
        )


# ---------------- referral inbox ----------------


class ReferralInboxView(APIView):
    """GET /api/referrals?status=draft|edited|discarded: referral drafts across runs, newest first."""

    permission_classes = [CanReadRuns]

    def get(self, request: Request) -> Response:
        qs = ReferralDraft.objects.select_related(
            "shortlist_item__trial", "shortlist_item__run__patient", "review", "edited_by"
        ).order_by("-created_at", "-id")
        wanted = request.query_params.get("status", "")
        if wanted in ReferralStatus.values:
            qs = qs.filter(status=wanted)
        rows, nxt = _page(list(_before(qs, request)[: PAGE + 1]))
        counts = {s: 0 for s in ReferralStatus.values}
        for row in ReferralDraft.objects.values("status").annotate(n=Count("id")):
            counts[row["status"]] = row["n"]
        # Draft bodies can restate record facts: listing them is a PHI read.
        audit(AuditKind.PHI_READ, actor_id=request.user.id, node="api_referral_inbox", count=len(rows))
        return Response(
            {
                "counts": counts,
                "next": nxt,
                "items": [
                    {
                        "id": r.pk,
                        "nct_id": r.shortlist_item.trial.nct_id,
                        "title": r.shortlist_item.trial.title,
                        "run_id": str(r.shortlist_item.run_id),
                        "patient_label": r.shortlist_item.run.patient.label,
                        "body": r.body,
                        "status": r.status,
                        "approved_by": r.review.reviewer_id,
                        "edited_by": r.edited_by.get_username() if r.edited_by else None,
                        "created_at": r.created_at.isoformat(),
                        "updated_at": r.updated_at.isoformat(),
                    }
                    for r in rows
                ],
            }
        )


# ---------------- audit log (admin) ----------------


class AuditLogView(APIView):
    """Admin only. Metadata of audit events: time, kind, actor, node, run id. Never the payload."""

    permission_classes = [IsAdmin]

    def get(self, request: Request) -> Response:
        p = request.query_params
        qs = AuditEvent.objects.select_related("actor").order_by("-created_at", "-id")
        if p.get("kind") in AuditKind.values:
            qs = qs.filter(kind=p["kind"])
        if p.get("actor"):
            qs = qs.filter(actor__username=p["actor"][:150])
        since = parse_datetime(p.get("since", "")) if p.get("since") else None
        if since:
            qs = qs.filter(created_at__gte=since)
        rows, nxt = _page(list(_before(qs, request)[: PAGE + 1]))
        labels = dict(AuditKind.choices)
        return Response(
            {
                "kinds": [{"value": v, "label": label} for v, label in AuditKind.choices],
                "next": nxt,
                "items": [
                    {
                        "id": e.pk,
                        "created_at": e.created_at.isoformat(),
                        "kind": e.kind,
                        "kind_label": labels.get(e.kind, e.kind),
                        "actor": e.actor.get_username() if e.actor else None,
                        "node": e.node,
                        "run_id": str(e.run_id) if e.run_id else None,
                    }
                    for e in rows
                ],
            }
        )


# ---------------- eval dashboard ----------------

REPORT_NAME = re.compile(r"^phase6-(\d{8}-\d{4})\.json$")
CONFIG_KEYS = (
    "model",
    "patients",
    "scored",
    "per_cohort",
    "pool_size",
    "seed",
    "snapshot",
    "replay",
    "spent_calls",
    "spent_tokens_in",
    "spent_tokens_out",
    "spent_usd",
)


def load_reports(folder: Path) -> list[dict[str, Any]]:
    """Phase 6 JSON reports, newest first. Keeps config, table, fails; drops the per-patient section."""
    out: list[dict[str, Any]] = []
    if not folder.is_dir():
        return out
    for path in folder.iterdir():
        m = REPORT_NAME.match(path.name)
        if not m:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("table"), dict):
            continue
        config = data.get("config") or {}
        out.append(
            {
                "id": m.group(1),
                "created_at": datetime.strptime(m.group(1), "%Y%m%d-%H%M").strftime("%Y-%m-%dT%H:%M:00Z"),
                "config": {k: config.get(k) for k in CONFIG_KEYS},
                "table": data["table"],
                "fails": [str(f) for f in data.get("fails") or []],
            }
        )
    return sorted(out, key=lambda r: r["id"], reverse=True)


def load_thresholds(path: Path) -> dict[str, Any] | None:
    try:
        phase6 = json.loads(path.read_text()).get("phase6")
    except (OSError, ValueError):
        return None
    if not isinstance(phase6, dict):
        return None
    keys = ("supervisor_minus_bm25_ndcg10", "supervisor_minus_fixed_ndcg10", "citation_validity", "baseline", "adr")
    return {k: phase6.get(k) for k in keys}


class EvalReportsView(APIView):
    permission_classes = [CanReadRuns]

    def get(self, request: Request) -> Response:
        return Response(
            {
                "thresholds": load_thresholds(Path(settings.EVAL_THRESHOLDS_FILE)),
                "reports": load_reports(Path(settings.EVAL_REPORTS_DIR)),
            }
        )
