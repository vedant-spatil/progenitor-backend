import uuid

from django.conf import settings
from django.db import models


class RunStatus(models.TextChoices):
    QUEUED = "queued", "Queued"
    RUNNING = "running", "Running"
    WAITING_CLINICIAN = "waiting_clinician", "Waiting for clinician answer"
    AWAITING_REVIEW = "awaiting_review", "Awaiting clinician review"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"
    CANCELLED = "cancelled", "Cancelled"


class Run(models.Model):
    """One matching run for one patient."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey("patients.Patient", on_delete=models.PROTECT, related_name="runs")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.PROTECT, related_name="runs")
    status = models.CharField(max_length=24, choices=RunStatus.choices, default=RunStatus.QUEUED, db_index=True)
    config = models.JSONField(default=dict, blank=True, help_text="Budgets, provider, model, search defaults.")
    summary = models.TextField(blank=True)
    pending_question = models.TextField(blank=True, help_text="Open ask_clinician question while waiting.")
    error = models.CharField(max_length=255, blank=True, help_text="Error class and short code only.")
    steps_used = models.PositiveIntegerField(default=0)
    tokens_in = models.PositiveBigIntegerField(default=0)
    tokens_out = models.PositiveBigIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"run {self.id} ({self.status})"


class DecisionAction(models.TextChoices):
    """The supervisor's action catalog (PLAN.md section 3.1)."""

    FETCH_PATIENT_DATA = "fetch_patient_data"
    SEARCH_TRIALS = "search_trials"
    REFINE_SEARCH = "refine_search"
    PARSE_CRITERIA = "parse_criteria"
    JUDGE_TRIALS = "judge_trials"
    RESOLVE_UNKNOWNS = "resolve_unknowns"
    DROP_TRIALS = "drop_trials"
    GATHER_EVIDENCE = "gather_evidence"
    RANK = "rank"
    ASK_CLINICIAN = "ask_clinician"
    DRAFT_REFERRALS = "draft_referrals"
    FINISH = "finish"


class GuardOutcome(models.TextChoices):
    ACCEPTED = "accepted", "Accepted"
    REASKED = "reasked", "Re-asked after schema error"
    REJECTED = "rejected", "Rejected by a guard"
    FORCED_FINISH = "forced_finish", "Forced finish (budget)"
    LOOP_STOPPED = "loop_stopped", "Stopped by loop guard"
    ORCHESTRATION_ERROR = "orchestration_error", "Orchestration error"


class Decision(models.Model):
    """Decision ledger: one row per supervisor decision (PLAN.md section 3.4)."""

    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="decisions")
    step = models.PositiveIntegerField()
    # 0 = first decision at this step; 1 = the re-ask after a guard rejection.
    attempt = models.PositiveSmallIntegerField(default=0)
    action = models.CharField(max_length=32, choices=DecisionAction.choices, blank=True)
    args = models.JSONField(default=dict, blank=True)
    rationale = models.TextField(blank=True)
    state_refs = models.JSONField(default=list, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    model = models.CharField(max_length=128, blank=True)
    tokens_in = models.PositiveIntegerField(default=0)
    tokens_out = models.PositiveIntegerField(default=0)
    latency_ms = models.PositiveIntegerField(default=0)
    guard_outcome = models.CharField(max_length=24, choices=GuardOutcome.choices, default=GuardOutcome.ACCEPTED)
    guard_detail = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["run", "step", "attempt"], name="decision_unique_run_step_attempt"),
        ]
        ordering = ["run", "step", "attempt"]

    def __str__(self) -> str:
        return f"{self.run_id} step {self.step}: {self.action}"


class VerdictStatus(models.TextChoices):
    MET = "met", "Met"
    NOT_MET = "not_met", "Not met"
    UNKNOWN = "unknown", "Unknown"
    NOT_APPLICABLE = "not_applicable", "Not applicable"


class Verdict(models.Model):
    """Eligibility Judge output for one criterion in one run.

    A met/not_met verdict must cite evidence; the citation guard (Phase 3)
    downgrades it to unknown otherwise.
    """

    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="verdicts")
    trial = models.ForeignKey("trials.Trial", on_delete=models.PROTECT, related_name="verdicts")
    criterion = models.ForeignKey("trials.Criterion", on_delete=models.PROTECT, related_name="verdicts")
    status = models.CharField(max_length=16, choices=VerdictStatus.choices)
    evidence_refs = models.JSONField(default=list, blank=True, help_text="FHIR resource ids, e.g. 'Observation/123'.")
    reason = models.TextField(blank=True)
    downgraded = models.BooleanField(default=False, help_text="Set when the citation guard forced 'unknown'.")
    model = models.CharField(max_length=128, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["run", "criterion"], name="verdict_unique_run_criterion"),
        ]
        ordering = ["run", "trial", "criterion"]

    def __str__(self) -> str:
        return f"{self.criterion_id}: {self.status}"


class ShortlistStatus(models.TextChoices):
    DRAFT = "draft", "Draft"
    APPROVED = "approved", "Approved"
    REJECTED = "rejected", "Rejected"


class ShortlistItem(models.Model):
    """One ranked trial in a run's shortlist. Stays draft until a clinician reviews it."""

    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name="shortlist")
    trial = models.ForeignKey("trials.Trial", on_delete=models.PROTECT, related_name="shortlist_items")
    rank = models.PositiveIntegerField()
    explanation = models.TextField(blank=True)
    counts = models.JSONField(
        default=dict, blank=True, help_text="Display only: {'met': n, 'not_met': n, 'unknown': n}."
    )
    status = models.CharField(max_length=16, choices=ShortlistStatus.choices, default=ShortlistStatus.DRAFT)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["run", "trial"], name="shortlist_unique_run_trial"),
            models.UniqueConstraint(fields=["run", "rank"], name="shortlist_unique_run_rank"),
        ]
        ordering = ["run", "rank"]

    def __str__(self) -> str:
        return f"{self.run_id} #{self.rank} {self.trial_id}"


class LlmCache(models.Model):
    """Cached structured LLM replies keyed by sha256(provider, model, prompt, schema). Reruns cost no tokens."""

    key = models.CharField(max_length=64, primary_key=True)
    task = models.CharField(max_length=64, blank=True)
    response = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.task} {self.key[:12]}"
