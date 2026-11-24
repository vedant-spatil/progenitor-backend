"""Gate 9 (PLAN.md Phase 9, ADR 0009): demo cases: roles, the demo set, one-click start, reset, append-only audit."""

from __future__ import annotations

from typing import Any

import pytest
from django.contrib.auth.models import User
from django.test import override_settings

from apps.audit.models import AuditEvent, AuditKind
from apps.audit.services import audit
from apps.patients.models import Patient
from apps.runs import services
from apps.runs.models import Run, RunStatus
from tests.unit.test_runs_api import client, enqueued, graph_out, users

pytestmark = pytest.mark.django_db
_ = (enqueued, users)  # fixtures shared with the 7b API tests

PROFILE = {
    "patient_ref": "Patient/s-1",
    "age_years": 61,
    "sex": "female",
    "conditions": [
        {"resource_id": "Condition/c0", "display": "Hypertension", "clinical_status": "resolved"},
        {"resource_id": "Condition/c1", "display": "Type 2 diabetes", "clinical_status": "active"},
        {"resource_id": "Condition/c2", "display": "Type 2 diabetes", "clinical_status": "active"},
    ],
}


@pytest.fixture
def cases() -> list[Patient]:
    made = [Patient.objects.create(source="synthea", fhir_id=f"s-{i}", label=f"Synthea 00{i}") for i in (3, 1, 2, 4)]
    made[1].profile = PROFILE
    made[1].save()
    Patient.objects.create(source="mimic", fhir_id="m-1", label="A MIMIC patient")
    return made


def _ids(body: dict[str, Any]) -> list[str]:
    return [i["label"] for i in body["items"]]


# ---------------- roles ----------------


@pytest.mark.parametrize(
    ("role", "code"), [("coordinator", 200), ("clinician", 200), ("admin", 403), ("none", 403), (None, 401)]
)
def test_list_roles(role: str | None, code: int, users: dict[str, User], cases: list[Patient]) -> None:
    assert client(users[role] if role else None).get("/api/demo").status_code == code


@pytest.mark.parametrize(
    ("role", "code"), [("coordinator", 201), ("clinician", 403), ("admin", 403), ("none", 403), (None, 401)]
)
def test_start_roles(
    role: str | None, code: int, users: dict[str, User], cases: list[Patient], enqueued: list[Any]
) -> None:
    resp = client(users[role] if role else None).post(f"/api/demo/{cases[1].pk}/start")
    assert resp.status_code == code


@pytest.mark.parametrize(
    ("role", "code"), [("coordinator", 200), ("clinician", 403), ("admin", 403), ("none", 403), (None, 401)]
)
def test_reset_roles(role: str | None, code: int, users: dict[str, User], enqueued: list[Any]) -> None:
    assert client(users[role] if role else None).post("/api/demo/reset").status_code == code


# ---------------- the demo set ----------------


def test_default_set_is_first_three_synthea_by_label(users: dict[str, User], cases: list[Patient]) -> None:
    body = client(users["clinician"]).get("/api/demo").json()
    assert _ids(body) == ["Synthea 001", "Synthea 002", "Synthea 003"]
    first = body["items"][0]
    assert first == {
        "patient_id": cases[1].pk,
        "label": "Synthea 001",
        "age_years": 61,
        "sex": "female",
        "conditions": ["Type 2 diabetes", "Hypertension"],  # active first, repeats dropped
        "latest_run": None,
    }
    assert "fhir_id" not in first
    assert "profile" not in first
    # Reading the cached profile is a PHI read; ids only in the payload.
    event = AuditEvent.objects.get(kind=AuditKind.PHI_READ, node="api_demo")
    assert event.payload == {"patient_ids": [cases[1].pk]}


@override_settings(DEMO_PATIENTS=["s-4", "m-1", "missing", "s-2"])
def test_configured_set_keeps_order_and_only_synthea(users: dict[str, User], cases: list[Patient]) -> None:
    assert _ids(client(users["coordinator"]).get("/api/demo").json()) == ["Synthea 004", "Synthea 002"]


@override_settings(DEMO_PATIENTS=["s-4"])
def test_start_refuses_patients_outside_the_set(
    users: dict[str, User], cases: list[Patient], enqueued: list[Any]
) -> None:
    resp = client(users["coordinator"]).post(f"/api/demo/{cases[1].pk}/start")
    assert resp.status_code == 404
    assert not Run.objects.exists()


# ---------------- start ----------------


def test_start_creates_a_demo_run_then_returns_the_open_one(
    users: dict[str, User], cases: list[Patient], enqueued: list[Any]
) -> None:
    coord = client(users["coordinator"])
    first = coord.post(f"/api/demo/{cases[1].pk}/start")
    assert first.status_code == 201
    assert first.json()["created"] is True
    run = Run.objects.get(pk=first.json()["id"])
    assert run.config == {"demo": True}
    assert run.created_by == users["coordinator"]
    assert enqueued == [{"run": str(run.id)}]
    assert AuditEvent.objects.filter(kind=AuditKind.RUN_CREATED, run=run).count() == 1

    again = coord.post(f"/api/demo/{cases[1].pk}/start")  # double click
    assert again.status_code == 200
    assert again.json() == {"id": str(run.id), "status": "queued", "created": False}
    assert Run.objects.count() == 1

    latest = client(users["clinician"]).get("/api/demo").json()["items"][0]["latest_run"]
    assert latest["id"] == str(run.id)
    assert latest["status"] == "queued"


def test_finished_demo_run_allows_a_new_one(users: dict[str, User], cases: list[Patient], enqueued: list[Any]) -> None:
    Run.objects.create(patient=cases[1], status=RunStatus.COMPLETED, config={"demo": True})
    resp = client(users["coordinator"]).post(f"/api/demo/{cases[1].pk}/start")
    assert resp.status_code == 201


def test_ordinary_runs_are_not_demo_runs(users: dict[str, User], cases: list[Patient], enqueued: list[Any]) -> None:
    Run.objects.create(patient=cases[1], status=RunStatus.RUNNING)
    assert client(users["clinician"]).get("/api/demo").json()["items"][0]["latest_run"] is None
    assert client(users["coordinator"]).post(f"/api/demo/{cases[1].pk}/start").status_code == 201


# ---------------- reset ----------------


def test_reset_cancels_only_open_demo_runs_and_keeps_the_audit_trail(
    users: dict[str, User], cases: list[Patient], enqueued: list[Any]
) -> None:
    demo = {"demo": True}
    open_runs = [
        Run.objects.create(
            patient=cases[0], status=s, config=demo, pending_question="Q?" if s == "waiting_clinician" else ""
        )
        for s in (RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.WAITING_CLINICIAN, RunStatus.AWAITING_REVIEW)
    ]
    done = Run.objects.create(patient=cases[0], status=RunStatus.COMPLETED, config=demo)
    ordinary = Run.objects.create(patient=cases[0], status=RunStatus.RUNNING)
    audit(AuditKind.RUN_CREATED, run_id=str(open_runs[0].id), node="api")
    before = AuditEvent.objects.count()

    resp = client(users["coordinator"]).post("/api/demo/reset")
    assert resp.json() == {"cancelled": 4}
    for r in open_runs:
        r.refresh_from_db()
        assert r.status == RunStatus.CANCELLED
        assert r.finished_at is not None
        assert r.pending_question == ""
    done.refresh_from_db()
    ordinary.refresh_from_db()
    assert done.status == RunStatus.COMPLETED
    assert ordinary.status == RunStatus.RUNNING

    cancels = AuditEvent.objects.filter(kind=AuditKind.RUN_CANCELLED)
    assert cancels.count() == 4
    assert all(e.payload == {"reason": "demo_reset"} and e.actor == users["coordinator"] for e in cancels)
    assert AuditEvent.objects.count() == before + 4  # nothing deleted

    assert client(users["coordinator"]).post("/api/demo/reset").json() == {"cancelled": 0}


# ---------------- a cancelled run stays cancelled ----------------


def test_worker_finishing_after_cancel_does_not_revive_the_run(
    users: dict[str, User], cases: list[Patient], enqueued: list[Any]
) -> None:
    run = Run.objects.create(patient=cases[0], status=RunStatus.RUNNING, config={"demo": True})
    assert services.cancel(run.id, users["coordinator"], reason="demo_reset") is True
    assert services.persist(str(run.id), graph_out(), used=(10, 5)) == RunStatus.CANCELLED
    services._set_status(str(run.id), RunStatus.FAILED, error="LlmError")
    run.refresh_from_db()
    assert run.status == RunStatus.CANCELLED
    assert run.error == ""
    assert run.tokens_in == 10  # usage is still counted
    assert not run.shortlist.exists()
    assert services.cancel(run.id, users["coordinator"], reason="demo_reset") is False
