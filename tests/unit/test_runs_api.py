"""Gate 5: run API per endpoint and role, approval flow, audit (sqlite; the graph is not run here)."""

from __future__ import annotations

from typing import Any

import pytest
from django.contrib.auth.models import Group, User
from django.test import RequestFactory
from rest_framework.test import APIClient

from apps.audit.models import AuditEvent, AuditKind
from apps.patients.models import Patient
from apps.reviews.models import ReferralDraft, Review
from apps.runs import services, views
from apps.runs.models import Run, RunStatus, ShortlistItem, Verdict

pytestmark = pytest.mark.django_db

A, B = "NCT00000001", "NCT00000002"


def _user(name: str, *roles: str) -> User:
    user = User.objects.create_user(name)  # unusable password: tests authenticate directly
    for r in roles:
        user.groups.add(Group.objects.get(name=r))
    return user


@pytest.fixture
def users() -> dict[str, User]:
    return {
        "coordinator": _user("coord", "coordinator"),
        "clinician": _user("clin", "clinician"),
        "admin": _user("adm", "admin"),
        "none": _user("norole"),
    }


@pytest.fixture
def enqueued(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(services, "_enqueue", lambda run, **kw: calls.append({"run": str(run.id), **kw}))
    monkeypatch.setattr(services, "publish", lambda *a, **k: None)
    return calls


@pytest.fixture
def patient() -> Patient:
    return Patient.objects.create(source="synthea", fhir_id="p-api", label="Synthea 001")


def client(user: User | None) -> APIClient:
    c = APIClient()
    if user is not None:
        c.force_authenticate(user)
    return c


def graph_out(**extra: Any) -> dict[str, Any]:
    crit = {"index": 0, "kind": "inclusion", "source_text": "HbA1c 7-10%", "entity": "HbA1c"}
    verdict = {"criterion_index": 0, "status": "met", "evidence_refs": ["Observation/o1"], "reason": "8.2%"}
    trials = {
        n: {"summary": {"nct_id": n, "title": f"Trial {n}"}, "criteria": [crit], "verdicts": [verdict]} for n in (A, B)
    }
    return {
        "status": "finished",
        "steps": 6,
        "summary": "ranked",
        "finish_reason": "supervisor",
        "trials": trials,
        "shortlist": [
            {"nct_id": A, "rank": 1, "explanation": "best", "counts": {"met": 1}},
            {"nct_id": B, "rank": 2, "explanation": "next", "counts": {"met": 1}},
        ],
        "approved_trial_ids": [],
        "referrals": [],
        **extra,
    }


def awaiting_review(patient: Patient) -> Run:
    run = Run.objects.create(patient=patient, status=RunStatus.RUNNING)
    assert services.persist(str(run.id), graph_out()) == RunStatus.AWAITING_REVIEW
    run.refresh_from_db()
    return run


# ---------------- roles per endpoint ----------------


def test_roles_exist() -> None:
    assert set(Group.objects.values_list("name", flat=True)) >= {"coordinator", "clinician", "admin"}


@pytest.mark.parametrize(
    ("role", "code"), [("coordinator", 201), ("clinician", 403), ("admin", 403), ("none", 403), (None, 401)]
)
def test_create_run(
    role: str | None, code: int, users: dict[str, User], patient: Patient, enqueued: list[dict[str, Any]]
) -> None:
    resp = client(users[role] if role else None).post("/api/runs", {"patient_id": patient.pk}, format="json")
    assert resp.status_code == code
    if code == 201:
        run = Run.objects.get(pk=resp.json()["id"])
        assert run.status == RunStatus.QUEUED
        assert run.created_by == users["coordinator"]
        assert enqueued == [{"run": str(run.id)}]
        assert AuditEvent.objects.filter(run=run, kind=AuditKind.RUN_CREATED, actor=users["coordinator"]).exists()
    else:
        assert not Run.objects.exists()
        assert not enqueued


def test_create_run_unknown_patient(users: dict[str, User], enqueued: list[dict[str, Any]]) -> None:
    resp = client(users["coordinator"]).post("/api/runs", {"patient_id": 999}, format="json")
    assert resp.status_code == 400


@pytest.mark.parametrize("path", ["", "/shortlist", "/referrals"])
@pytest.mark.parametrize(
    ("role", "code"), [("coordinator", 200), ("clinician", 200), ("admin", 403), ("none", 403), (None, 401)]
)
def test_read_endpoints(
    path: str, role: str | None, code: int, users: dict[str, User], patient: Patient, enqueued: list[Any]
) -> None:
    run = awaiting_review(patient)
    resp = client(users[role] if role else None).get(f"/api/runs/{run.id}{path}")
    assert resp.status_code == code


def test_run_detail_shape(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = awaiting_review(patient)
    body = client(users["clinician"]).get(f"/api/runs/{run.id}").json()
    assert body["status"] == "awaiting_review"
    assert body["patient_label"] == "Synthea 001"
    assert body["steps_used"] == 6


def test_shortlist_lists_verdicts_and_audits_the_read(
    users: dict[str, User], patient: Patient, enqueued: list[Any]
) -> None:
    run = awaiting_review(patient)
    body = client(users["clinician"]).get(f"/api/runs/{run.id}/shortlist").json()
    assert [i["nct_id"] for i in body["items"]] == [A, B]
    assert body["items"][0]["verdicts"][0] == {
        "criterion_index": 0,
        "kind": "inclusion",
        "criterion_text": "HbA1c 7-10%",
        "status": "met",
        "evidence_refs": ["Observation/o1"],
        "reason": "8.2%",
    }
    assert AuditEvent.objects.filter(run=run, kind=AuditKind.PHI_READ, actor=users["clinician"]).count() == 1


def test_missing_run_is_404(users: dict[str, User]) -> None:
    resp = client(users["clinician"]).get("/api/runs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


# ---------------- answer ----------------


@pytest.mark.parametrize(("role", "code"), [("clinician", 202), ("coordinator", 403), ("admin", 403)])
def test_answer_roles(role: str, code: int, users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = Run.objects.create(patient=patient, status=RunStatus.WAITING_CLINICIAN, pending_question="Insulin?")
    resp = client(users[role]).post(f"/api/runs/{run.id}/answer", {"answer": "No insulin"}, format="json")
    assert resp.status_code == code
    run.refresh_from_db()
    if code == 202:
        assert run.status == RunStatus.QUEUED
        assert run.pending_question == ""
        assert enqueued == [
            {"run": str(run.id), "answer": {"answer": "No insulin", "actor_user_id": users["clinician"].id}}
        ]
        event = AuditEvent.objects.get(run=run, kind=AuditKind.ANSWER)
        assert event.actor == users["clinician"]
        assert event.payload == {"answer_chars": 10}  # length only, never the text
    else:
        assert run.status == RunStatus.WAITING_CLINICIAN
        assert not enqueued


def test_answer_when_not_waiting_is_409(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = awaiting_review(patient)
    resp = client(users["clinician"]).post(f"/api/runs/{run.id}/answer", {"answer": "x"}, format="json")
    assert resp.status_code == 409


# ---------------- review / approval flow ----------------


def test_review_roles(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = awaiting_review(patient)
    payload = {"items": [{"nct_id": A, "decision": "approve"}]}
    for role in ("coordinator", "admin"):
        assert client(users[role]).post(f"/api/runs/{run.id}/review", payload, format="json").status_code == 403
    assert client(users["clinician"]).post(f"/api/runs/{run.id}/review", payload, format="json").status_code == 202
    assert client(users["clinician"]).post(f"/api/runs/{run.id}/review", payload, format="json").status_code == 409


def test_review_validation(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = awaiting_review(patient)
    c = client(users["clinician"])
    url = f"/api/runs/{run.id}/review"
    assert c.post(url, {"items": []}, format="json").status_code == 400
    assert c.post(url, {"items": [{"nct_id": "NCT09999999", "decision": "approve"}]}, format="json").status_code == 400
    assert c.post(url, {"items": [{"nct_id": A, "decision": "edit"}]}, format="json").status_code == 400  # no edits
    dup = {"items": [{"nct_id": A, "decision": "approve"}, {"nct_id": A, "decision": "reject"}]}
    assert c.post(url, dup, format="json").status_code == 400
    assert not Review.objects.exists()


def test_approve_then_drafts_only_for_approved(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = awaiting_review(patient)
    clin = users["clinician"]
    payload = {
        "items": [
            {"nct_id": A, "decision": "edit", "edits": {"note": "mention eGFR"}},
            {"nct_id": B, "decision": "reject", "comment": "travel distance"},
        ]
    }
    assert client(clin).post(f"/api/runs/{run.id}/review", payload, format="json").status_code == 202
    run.refresh_from_db()
    assert run.status == RunStatus.QUEUED
    assert enqueued == [{"run": str(run.id), "approved": [A]}]
    statuses = dict(ShortlistItem.objects.filter(run=run).values_list("trial__nct_id", "status"))
    assert statuses == {A: "approved", B: "rejected"}

    # The worker resumes the graph; the supervisor drafts. A draft for B must never be stored.
    out = graph_out(
        approved_trial_ids=[A],
        referrals=[{"nct_id": A, "body": "Referral for A"}, {"nct_id": B, "body": "should be ignored"}],
    )
    assert services.persist(str(run.id), out, model="scripted") == RunStatus.COMPLETED
    drafts = list(ReferralDraft.objects.filter(shortlist_item__run=run))
    assert [(d.shortlist_item.trial.nct_id, d.body, d.review.reviewer_id) for d in drafts] == [
        (A, "Referral for A", clin.id)
    ]
    body = client(users["coordinator"]).get(f"/api/runs/{run.id}/referrals").json()
    assert body["items"][0]["nct_id"] == A
    assert body["items"][0]["approved_by"] == clin.id


def test_reject_all_creates_no_referrals(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = awaiting_review(patient)
    payload = {"items": [{"nct_id": A, "decision": "reject"}, {"nct_id": B, "decision": "reject"}]}
    assert client(users["clinician"]).post(f"/api/runs/{run.id}/review", payload, format="json").status_code == 202
    run.refresh_from_db()
    assert run.status == RunStatus.COMPLETED
    assert run.finished_at is not None
    assert enqueued == []  # nothing resumes the graph, so nothing can draft
    assert not ReferralDraft.objects.exists()


def test_every_approval_has_an_actor(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = awaiting_review(patient)
    payload = {"items": [{"nct_id": A, "decision": "approve"}, {"nct_id": B, "decision": "reject"}]}
    client(users["clinician"]).post(f"/api/runs/{run.id}/review", payload, format="json")
    approvals = AuditEvent.objects.filter(kind=AuditKind.APPROVAL, run=run)
    assert approvals.count() == 2
    assert all(e.actor_id == users["clinician"].id for e in approvals)
    assert {e.payload["nct_id"]: e.payload["decision"] for e in approvals} == {A: "approve", B: "reject"}


# ---------------- persistence ----------------


def test_persist_paused_and_empty_runs(patient: Patient, enqueued: list[Any]) -> None:
    paused = Run.objects.create(patient=patient, status=RunStatus.RUNNING)
    q = type("I", (), {"value": {"run_id": str(paused.id), "question": "Insulin in the last 3 months?"}})()
    assert services.persist(str(paused.id), {"steps": 2, "__interrupt__": [q]}) == RunStatus.WAITING_CLINICIAN
    paused.refresh_from_db()
    assert paused.pending_question == "Insulin in the last 3 months?"

    empty = Run.objects.create(patient=patient, status=RunStatus.RUNNING)
    out = {"status": "finished", "steps": 3, "shortlist": [], "finish_reason": "forced_finish"}
    assert services.persist(str(empty.id), out) == RunStatus.COMPLETED
    empty.refresh_from_db()
    assert empty.error == "forced_finish"


def test_persist_shortlist_is_idempotent(patient: Patient, enqueued: list[Any]) -> None:
    run = awaiting_review(patient)
    services.persist(str(run.id), graph_out())
    assert ShortlistItem.objects.filter(run=run).count() == 2
    assert Verdict.objects.filter(run=run).count() == 2


# ---------------- SSE auth ----------------


def test_sse_snapshot_auth(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = awaiting_review(patient)
    rf = RequestFactory()
    assert views._snapshot(rf.get("/"), str(run.id))[0] == 401
    from rest_framework_simplejwt.tokens import AccessToken

    def req(user: User) -> Any:
        return rf.get("/", HTTP_AUTHORIZATION=f"Bearer {AccessToken.for_user(user)}")

    assert views._snapshot(req(users["admin"]), str(run.id))[0] == 403
    code, body = views._snapshot(req(users["clinician"]), str(run.id))
    assert code == 200
    assert body["status"] == "awaiting_review"
    assert views._snapshot(req(users["clinician"]), "00000000-0000-0000-0000-000000000000")[0] == 404


async def test_sse_stream_ends_after_snapshot_for_paused_runs() -> None:
    from apps.runs.events import stream

    chunks = [c async for c in stream("r1", {"status": "awaiting_review", "steps_used": 6})]
    assert chunks == ['event: snapshot\ndata: {"status": "awaiting_review", "steps_used": 6}\n\n']
