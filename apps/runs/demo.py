"""Demo cases (PLAN.md Phase 9, ADR 0009): preset synthetic patients a visitor can run in one click.

GET /api/demo lists the cases (coordinator or clinician). POST /api/demo/<patient_id>/start (coordinator) starts a
demo run, or returns the case's open one. POST /api/demo/reset (coordinator) cancels open demo runs; their decisions
and audit events stay (AuditEvent is append-only). Demo runs are ordinary runs marked `config["demo"] = True`.
The frontend mirrors these shapes in lib/api/schemas.ts (DemoCases, DemoStart, DemoReset).
"""

from __future__ import annotations

from typing import Any

from django.conf import settings
from django.db.models import QuerySet
from django.http import Http404
from pydantic import ValidationError
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from agents.schemas import PatientProfile
from apps.accounts.roles import CanReadRuns, IsCoordinator
from apps.audit.models import AuditKind
from apps.audit.services import audit
from apps.patients.models import Patient, PatientSource
from apps.runs import services
from apps.runs.models import Run

DEFAULT_COUNT = 3
MAX_CONDITIONS = 3


def demo_patients() -> list[Patient]:
    """The demo set, in DEMO_PATIENTS order. Synthea patients only: the card is for anyone trying the app."""
    synthea = Patient.objects.filter(source=PatientSource.SYNTHEA)
    ids: list[str] = list(settings.DEMO_PATIENTS)
    if not ids:
        return list(synthea.order_by("label", "id")[:DEFAULT_COUNT])
    found = {p.fhir_id: p for p in synthea.filter(fhir_id__in=ids)}
    return [found[i] for i in ids if i in found]


def demo_runs() -> QuerySet[Run]:
    return Run.objects.filter(config__demo=True)


def _profile(patient: Patient) -> PatientProfile | None:
    if not patient.profile:
        return None
    try:
        return PatientProfile.model_validate(patient.profile)
    except ValidationError:
        return None  # an old cache shape: show no facts rather than raw JSON


def _conditions(profile: PatientProfile | None) -> list[str]:
    """First few distinct condition names, active ones first (Synthea repeats conditions across encounters)."""
    if profile is None:
        return []
    ordered = sorted(profile.conditions, key=lambda c: c.clinical_status != "active")
    return list(dict.fromkeys(c.display for c in ordered if c.display))[:MAX_CONDITIONS]


def _case(patient: Patient, latest: Run | None) -> dict[str, Any]:
    profile = _profile(patient)
    return {
        "patient_id": patient.pk,
        "label": patient.label or f"Patient {patient.pk}",
        "age_years": profile.age_years if profile else None,
        "sex": profile.sex if profile else "",
        "conditions": _conditions(profile),
        "latest_run": (
            {"id": str(latest.id), "status": latest.status, "created_at": latest.created_at.isoformat()}
            if latest
            else None
        ),
    }


class DemoListView(APIView):
    permission_classes = [CanReadRuns]

    def get(self, request: Request) -> Response:
        patients = demo_patients()
        latest: dict[int, Run] = {}
        for run in demo_runs().filter(patient__in=patients).order_by("-created_at"):
            latest.setdefault(run.patient_id, run)
        if any(p.profile for p in patients):
            # Condition names restate record facts: reading cached profiles is a PHI read (synthetic here).
            audit(
                AuditKind.PHI_READ,
                actor_id=request.user.id,
                node="api_demo",
                patient_ids=[p.pk for p in patients if p.profile],
            )
        return Response({"items": [_case(p, latest.get(p.pk)) for p in patients]})


class DemoStartView(APIView):
    permission_classes = [IsCoordinator]

    def post(self, request: Request, patient_id: int) -> Response:
        patient = next((p for p in demo_patients() if p.pk == patient_id), None)
        if patient is None:
            raise Http404("not a demo case")
        open_run = (
            demo_runs().filter(patient=patient, status__in=services.OPEN_STATUSES).order_by("-created_at").first()
        )
        if open_run is not None:  # one click twice must not start two runs
            return Response({"id": str(open_run.id), "status": open_run.status, "created": False})
        run = services.start(patient, request.user, {"demo": True})
        return Response({"id": str(run.id), "status": run.status, "created": True}, status=status.HTTP_201_CREATED)


class DemoResetView(APIView):
    permission_classes = [IsCoordinator]

    def post(self, request: Request) -> Response:
        ids = list(demo_runs().filter(status__in=services.OPEN_STATUSES).values_list("pk", flat=True))
        cancelled = sum(services.cancel(pk, request.user, reason="demo_reset") for pk in ids)
        return Response({"cancelled": cancelled})
