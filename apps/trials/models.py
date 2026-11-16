from django.db import models


class TrialSource(models.TextChoices):
    CTGOV_API = "ctgov_api", "ClinicalTrials.gov API v2"
    AACT = "aact", "AACT snapshot"
    EVAL = "eval", "Eval corpus (TrialGPT)"


class Trial(models.Model):
    """ClinicalTrials.gov cache. `raw` keeps the full study JSON; the other
    columns are the fields the agents filter and display on."""

    nct_id = models.CharField(max_length=16, unique=True)
    source = models.CharField(max_length=16, choices=TrialSource.choices, default=TrialSource.CTGOV_API)
    title = models.TextField(blank=True)
    overall_status = models.CharField(max_length=40, blank=True, db_index=True)
    phases = models.JSONField(default=list, blank=True)
    conditions = models.JSONField(default=list, blank=True)
    eligibility_text = models.TextField(blank=True)
    minimum_age = models.CharField(max_length=32, blank=True)
    maximum_age = models.CharField(max_length=32, blank=True)
    sex = models.CharField(max_length=16, blank=True)
    healthy_volunteers = models.BooleanField(null=True)
    raw = models.JSONField(default=dict, blank=True)
    source_last_updated = models.DateField(
        null=True, blank=True, help_text="Registry 'last update posted' date."
    )
    fetched_at = models.DateTimeField(help_text="When this copy was fetched from the source.")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["nct_id"]

    def __str__(self) -> str:
        return self.nct_id


class CriterionKind(models.TextChoices):
    INCLUSION = "inclusion", "Inclusion"
    EXCLUSION = "exclusion", "Exclusion"


class Criterion(models.Model):
    """One parsed eligibility rule (Criteria Parser output, Phase 3)."""

    trial = models.ForeignKey(Trial, on_delete=models.CASCADE, related_name="criteria")
    position = models.PositiveIntegerField(help_text="Order within the trial's criteria text.")
    kind = models.CharField(max_length=16, choices=CriterionKind.choices)
    source_text = models.TextField()
    entity = models.CharField(max_length=255, blank=True)
    entity_codes = models.JSONField(
        default=list, blank=True, help_text="e.g. [{'system': 'loinc', 'code': '4548-4'}]"
    )
    operator = models.CharField(max_length=16, blank=True)
    value = models.CharField(max_length=128, blank=True)
    unit = models.CharField(max_length=32, blank=True)
    temporal_window = models.CharField(max_length=64, blank=True)
    parser_model = models.CharField(max_length=128, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["trial", "position"], name="criterion_unique_trial_position"),
        ]
        ordering = ["trial", "position"]

    def __str__(self) -> str:
        return f"{self.trial_id}#{self.position} {self.kind}"
