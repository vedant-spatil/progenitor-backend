"""Gate 7c: read-only app endpoints (ADR 0007): roles, filters, paging, and what each response must not contain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from django.contrib.auth.models import User
from django.test import override_settings

from apps.audit.models import AuditEvent, AuditKind
from apps.patients.models import Patient
from apps.reviews.models import ReferralDraft, ReferralStatus
from apps.runs import browse
from apps.runs.models import Decision, Run, RunStatus
from tests.unit.test_runs_api import A, B, _approved_draft, awaiting_review, client, enqueued, patient, users

pytestmark = pytest.mark.django_db
_ = (enqueued, patient, users)  # fixtures shared with the 7b API tests

READERS_ONLY = [("coordinator", 200), ("clinician", 200), ("admin", 403), ("none", 403), (None, 401)]


def _get(users: dict[str, User], role: str | None, url: str) -> Any:
    return client(users[role] if role else None).get(url)


# ---------------- roles per endpoint ----------------


@pytest.mark.parametrize("url", ["/api/overview", "/api/trials", "/api/referrals", "/api/evals", "/api/runs"])
@pytest.mark.parametrize(("role", "code"), READERS_ONLY)
def test_reader_endpoints_roles(url: str, role: str | None, code: int, users: dict[str, User]) -> None:
    assert _get(users, role, url).status_code == code


@pytest.mark.parametrize(("role", "code"), READERS_ONLY)
def test_patient_and_trial_detail_roles(
    role: str | None, code: int, users: dict[str, User], patient: Patient, enqueued: list[Any]
) -> None:
    awaiting_review(patient)
    assert _get(users, role, f"/api/patients/{patient.pk}").status_code == code
    assert _get(users, role, f"/api/trials/{A}").status_code == code


@pytest.mark.parametrize(
    ("role", "code"), [("admin", 200), ("coordinator", 403), ("clinician", 403), ("none", 403), (None, 401)]
)
def test_audit_log_admin_only(role: str | None, code: int, users: dict[str, User]) -> None:
    assert _get(users, role, "/api/audit").status_code == code


# ---------------- patients ----------------


def test_patient_list_run_stats(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    Run.objects.create(patient=patient, status=RunStatus.FAILED)
    Run.objects.create(patient=patient, status=RunStatus.COMPLETED)
    other = Patient.objects.create(source="synthea", fhir_id="p-2", label="Synthea 002")
    items = {i["id"]: i for i in _get(users, "clinician", "/api/patients").json()["items"]}
    assert items[patient.pk]["run_count"] == 2
    assert items[patient.pk]["last_run_status"] == "completed"
    assert items[other.pk]["run_count"] == 0
    assert items[other.pk]["last_run_status"] is None
    assert "fhir_id" not in items[patient.pk]
    assert "profile" not in items[patient.pk]


def test_patient_detail_profile_runs_and_audit(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    patient.profile = {
        "patient_ref": "Patient/p-api",
        "age_years": 54,
        "sex": "female",
        "conditions": [{"resource_id": "Condition/c1", "display": "Type 2 diabetes", "clinical_status": "active"}],
    }
    patient.save()
    run = awaiting_review(patient)
    body = _get(users, "coordinator", f"/api/patients/{patient.pk}").json()
    assert body["profile"]["conditions"][0]["resource_id"] == "Condition/c1"
    assert body["profile"]["labs"] == []  # validated through PatientProfile: defaults filled in
    assert [r["id"] for r in body["runs"]] == [str(run.id)]
    assert {s["nct_id"] for s in body["shortlisted"]} == {A, B}
    assert "fhir_id" not in body
    ev = AuditEvent.objects.get(kind=AuditKind.PHI_READ, node="api_patient")
    assert ev.payload == {"patient_id": patient.pk}


def test_patient_detail_bad_cached_profile_is_hidden(users: dict[str, User], patient: Patient) -> None:
    patient.profile = {"unexpected": "shape"}
    patient.save()
    assert _get(users, "clinician", f"/api/patients/{patient.pk}").json()["profile"] is None
    assert _get(users, "clinician", "/api/patients/999999").status_code == 404


# ---------------- runs list filters ----------------


def test_run_filters_and_cursor(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    other = Patient.objects.create(source="synthea", fhir_id="p-3", label="Synthea 300")
    mine = Run.objects.create(patient=patient, status=RunStatus.FAILED, created_by=users["coordinator"])
    Run.objects.create(patient=other, status=RunStatus.COMPLETED)
    c = client(users["coordinator"])

    def ids(url: str) -> set[str]:
        return {i["id"] for i in c.get(url).json()["items"]}

    assert ids("/api/runs?status=failed") == {str(mine.id)}
    assert len(ids("/api/runs?status=failed,completed")) == 2
    assert len(ids("/api/runs?status=bogus")) == 2  # unknown status values are ignored
    assert ids(f"/api/runs?patient={patient.pk}") == {str(mine.id)}
    assert ids("/api/runs?q=300") != {str(mine.id)}
    assert len(ids("/api/runs?q=300")) == 1
    assert ids("/api/runs?mine=1") == {str(mine.id)}
    assert c.get("/api/runs").json()["next"] is None


def test_run_list_pages_by_cursor(
    users: dict[str, User], patient: Patient, enqueued: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(browse, "PAGE", 2)
    for _ in range(3):
        Run.objects.create(patient=patient, status=RunStatus.COMPLETED)
    c = client(users["clinician"])
    first = c.get("/api/runs").json()
    assert len(first["items"]) == 2
    assert first["next"]
    second = c.get("/api/runs", {"before": first["next"]}).json()
    assert len(second["items"]) == 1
    assert second["next"] is None
    assert not {i["id"] for i in first["items"]} & {i["id"] for i in second["items"]}


# ---------------- overview ----------------


def test_overview_counts_and_queue_by_role(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    review = awaiting_review(patient)
    question = Run.objects.create(patient=patient, status=RunStatus.WAITING_CLINICIAN, pending_question="Insulin?")
    failed = Run.objects.create(patient=patient, status=RunStatus.FAILED)
    Decision.objects.create(run=review, step=1, action="fetch_record", rationale="secret rationale")

    clin = _get(users, "clinician", "/api/overview").json()
    assert clin["counts"]["awaiting_review"] == 1
    assert clin["counts"]["failed"] == 1
    assert {(q["run_id"], q["reason"]) for q in clin["queue"]} == {
        (str(question.id), "answer_question"),
        (str(review.id), "review_shortlist"),
    }
    assert clin["recent"][0]["action"] == "fetch_record"
    assert "secret rationale" not in json.dumps(clin)
    assert "Insulin?" not in json.dumps(clin)

    coord = _get(users, "coordinator", "/api/overview").json()
    assert [(q["run_id"], q["reason"]) for q in coord["queue"]] == [(str(failed.id), "failed")]


def test_overview_coordinator_sees_new_referral_drafts(
    users: dict[str, User], patient: Patient, enqueued: list[Any]
) -> None:
    draft = _approved_draft(users, patient)
    reasons = [q["reason"] for q in _get(users, "coordinator", "/api/overview").json()["queue"]]
    assert reasons == ["referral_drafts"]
    draft.status = ReferralStatus.EDITED
    draft.save()
    assert _get(users, "coordinator", "/api/overview").json()["queue"] == []


# ---------------- trials ----------------


def test_trial_list_filters_and_counts(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    awaiting_review(patient)
    c = client(users["clinician"])
    body = c.get("/api/trials").json()
    assert body["total"] == 2
    assert body["page"] == 1
    row = {r["nct_id"]: r for r in body["items"]}[A]
    assert row["criteria_count"] == 1
    assert row["shortlisted_count"] == 1
    assert [r["nct_id"] for r in c.get("/api/trials", {"q": B.lower()}).json()["items"]] == [B]
    assert c.get("/api/trials", {"shortlisted": "1"}).json()["total"] == 2
    assert c.get("/api/trials", {"page": "abc"}).json()["page"] == 1


def test_trial_detail_criteria_and_matches(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = awaiting_review(patient)
    body = _get(users, "coordinator", f"/api/trials/{A.lower()}").json()
    assert body["nct_id"] == A
    assert body["criteria"] == [{"position": 0, "kind": "inclusion", "source_text": "HbA1c 7-10%"}]
    assert [(m["run_id"], m["patient_label"], m["status"]) for m in body["matches"]] == [
        (str(run.id), "Synthea 001", "draft")
    ]
    assert "reason" not in json.dumps(body["matches"])  # counts and status only, no verdict reasons
    assert _get(users, "coordinator", "/api/trials/NCT99999999").status_code == 404


# ---------------- referral inbox ----------------


def test_referral_inbox_filters_and_audits(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    draft = _approved_draft(users, patient)
    c = client(users["coordinator"])
    body = c.get("/api/referrals").json()
    assert body["counts"] == {"draft": 1, "edited": 0, "discarded": 0}
    item = body["items"][0]
    assert (item["id"], item["nct_id"], item["patient_label"], item["status"]) == (draft.pk, A, "Synthea 001", "draft")
    assert c.get("/api/referrals?status=edited").json()["items"] == []
    assert len(c.get("/api/referrals?status=bogus").json()["items"]) == 1
    ev = AuditEvent.objects.filter(kind=AuditKind.PHI_READ, node="api_referral_inbox").first()
    assert ev is not None
    assert ev.payload == {"count": 1}
    # Edits stay clinician-only, through the 7b PATCH endpoint.
    assert c.patch(f"/api/referrals/{draft.pk}", {"body": "x"}, format="json").status_code == 403
    assert ReferralDraft.objects.get(pk=draft.pk).body == "Draft"


# ---------------- audit log ----------------


def test_audit_log_has_no_payload(users: dict[str, User], patient: Patient, enqueued: list[Any]) -> None:
    run = Run.objects.create(patient=patient, status=RunStatus.RUNNING)
    AuditEvent.objects.create(kind=AuditKind.ANSWER, run=run, actor=users["clinician"], node="api", payload={"x": 1})
    AuditEvent.objects.create(kind=AuditKind.PHI_READ, node="fhir", payload={"resource": "Patient/p"})
    c = client(users["admin"])
    body = c.get("/api/audit").json()
    assert {k["value"] for k in body["kinds"]} >= {"phi_read", "answer"}
    assert set(body["items"][0]) == {"id", "created_at", "kind", "kind_label", "actor", "node", "run_id"}
    assert "Patient/p" not in json.dumps(body)
    answers = c.get("/api/audit", {"kind": "answer"}).json()["items"]
    assert [(e["actor"], e["run_id"]) for e in answers] == [("clin", str(run.id))]
    assert [e["kind"] for e in c.get("/api/audit", {"actor": "clin"}).json()["items"]] == ["answer"]
    assert c.get("/api/audit", {"since": "2999-01-01T00:00:00Z"}).json()["items"] == []


# ---------------- evals ----------------


def _report(folder: Path, stamp: str, **extra: Any) -> None:
    data = {
        "config": {"model": "m", "patients": 3, "scored": 3, "replay": False, "price": {"source": "x"}},
        "table": {"sigir": {"bm25": {"ndcg@10": 0.7}, "supervisor": {"ndcg@10": 0.85}}},
        "fails": [],
        "patients": [{"topic_id": "sigir-1", "text": "PATIENT TEXT"}],
        **extra,
    }
    (folder / f"phase6-{stamp}.json").write_text(json.dumps(data))


def test_evals_reports_newest_first_without_patients(users: dict[str, User], tmp_path: Path) -> None:
    _report(tmp_path, "20261119-0707")
    _report(tmp_path, "20261120-0633", fails=["sigir: below"])
    (tmp_path / "phase6-broken.json").write_text("{")
    (tmp_path / "phase3-20261117-1914.json").write_text("{}")
    (tmp_path / "phase6-20261121-0000.json").write_text("not json")
    thresholds = tmp_path / "t.json"
    thresholds.write_text(json.dumps({"phase6": {"supervisor_minus_bm25_ndcg10": 0.05, "citation_validity": 1.0}}))
    with override_settings(EVAL_REPORTS_DIR=tmp_path, EVAL_THRESHOLDS_FILE=thresholds):
        body = _get(users, "clinician", "/api/evals").json()
    assert [r["id"] for r in body["reports"]] == ["20261120-0633", "20261119-0707"]
    assert body["reports"][0]["created_at"] == "2026-11-20T06:33:00Z"
    assert body["reports"][0]["fails"] == ["sigir: below"]
    assert body["reports"][0]["table"]["sigir"]["supervisor"]["ndcg@10"] == 0.85
    assert body["thresholds"]["supervisor_minus_bm25_ndcg10"] == 0.05
    text = json.dumps(body)
    assert "PATIENT TEXT" not in text
    assert 'patients": [' not in text
    assert "price" not in text


def test_evals_missing_folder(users: dict[str, User], tmp_path: Path) -> None:
    with override_settings(EVAL_REPORTS_DIR=tmp_path / "nope", EVAL_THRESHOLDS_FILE=tmp_path / "nope.json"):
        assert _get(users, "coordinator", "/api/evals").json() == {"thresholds": None, "reports": []}
