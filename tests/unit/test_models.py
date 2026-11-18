from datetime import UTC, datetime

import pytest
from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction

from apps.audit.models import AppendOnlyError, AuditEvent, AuditKind
from apps.patients.models import Patient, PatientSource
from apps.reviews.models import ReferralDraft, Review, ReviewDecision
from apps.runs.models import Decision, DecisionAction, Run, ShortlistItem, Verdict, VerdictStatus
from apps.trials.models import Criterion, CriterionKind, Trial

pytestmark = pytest.mark.django_db


def _trial(nct: str = "NCT00000001") -> Trial:
    return Trial.objects.create(nct_id=nct, title="t", fetched_at=datetime(2026, 1, 1, tzinfo=UTC))


def _run() -> Run:
    patient = Patient.objects.create(source=PatientSource.SYNTHEA, fhir_id="p1", label="Synthea 001")
    return Run.objects.create(patient=patient)


class TestAuditAppendOnly:
    def test_create_allowed(self) -> None:
        event = AuditEvent.objects.create(kind=AuditKind.DATA_LOAD, payload={"created": 3})
        assert event.pk is not None

    def test_instance_update_blocked(self) -> None:
        event = AuditEvent.objects.create(kind=AuditKind.DATA_LOAD)
        event.kind = AuditKind.DECISION
        with pytest.raises(AppendOnlyError):
            event.save()

    def test_instance_delete_blocked(self) -> None:
        event = AuditEvent.objects.create(kind=AuditKind.DATA_LOAD)
        with pytest.raises(AppendOnlyError):
            event.delete()

    def test_queryset_update_blocked(self) -> None:
        event = AuditEvent.objects.create(kind=AuditKind.DATA_LOAD)
        with pytest.raises(AppendOnlyError):
            AuditEvent.objects.filter(pk=event.pk).update(kind="x")

    def test_queryset_delete_blocked(self) -> None:
        AuditEvent.objects.create(kind=AuditKind.DATA_LOAD)
        with pytest.raises(AppendOnlyError):
            AuditEvent.objects.all().delete()


def test_patient_unique_per_source() -> None:
    Patient.objects.create(source=PatientSource.MIMIC, fhir_id="abc")
    Patient.objects.create(source=PatientSource.SYNTHEA, fhir_id="abc")
    with pytest.raises(IntegrityError), transaction.atomic():
        Patient.objects.create(source=PatientSource.MIMIC, fhir_id="abc")


def test_decision_step_unique_per_run() -> None:
    run = _run()
    Decision.objects.create(run=run, step=1, action=DecisionAction.SEARCH_TRIALS)
    with pytest.raises(IntegrityError), transaction.atomic():
        Decision.objects.create(run=run, step=1, action=DecisionAction.RANK)


def test_shortlist_rank_and_trial_unique_per_run() -> None:
    run = _run()
    a, b = _trial("NCT00000001"), _trial("NCT00000002")
    ShortlistItem.objects.create(run=run, trial=a, rank=1)
    with pytest.raises(IntegrityError), transaction.atomic():
        ShortlistItem.objects.create(run=run, trial=b, rank=1)
    with pytest.raises(IntegrityError), transaction.atomic():
        ShortlistItem.objects.create(run=run, trial=a, rank=2)


def test_full_chain_run_to_referral() -> None:
    user = get_user_model().objects.create_user(username="clin", password="a-long-test-password")
    run = _run()
    trial = _trial()
    crit = Criterion.objects.create(trial=trial, position=1, kind=CriterionKind.INCLUSION, source_text="HbA1c > 7.5%")
    Verdict.objects.create(
        run=run, trial=trial, criterion=crit, status=VerdictStatus.MET, evidence_refs=["Observation/1"]
    )
    item = ShortlistItem.objects.create(run=run, trial=trial, rank=1)
    review = Review.objects.create(run=run, shortlist_item=item, reviewer=user, decision=ReviewDecision.APPROVE)
    draft = ReferralDraft.objects.create(shortlist_item=item, review=review, body="draft")
    assert draft.status == "draft"
    assert item.status == "draft"


def test_decision_actions_match_plan_catalog() -> None:
    expected = {
        "fetch_patient_data",
        "search_trials",
        "refine_search",
        "parse_criteria",
        "judge_trials",
        "resolve_unknowns",
        "drop_trials",
        "gather_evidence",
        "rank",
        "ask_clinician",
        "draft_referrals",
        "finish",
    }
    assert set(DecisionAction.values) == expected
