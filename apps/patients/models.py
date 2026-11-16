from django.db import models


class PatientSource(models.TextChoices):
    SYNTHEA = "synthea", "Synthea (synthetic)"
    MIMIC = "mimic", "MIMIC-IV FHIR demo (de-identified)"
    HAPI_PUBLIC = "hapi_public", "HAPI public server (test data)"
    SMART = "smart", "SMART sandbox (test data)"
    TEXT = "text", "Free-text eval topic (TREC / SIGIR)"


class Patient(models.Model):
    """A pointer to a patient record held in a FHIR server.

    Holds no identifiers beyond the FHIR resource id: no names, birth dates, or
    addresses. Clinical data stays in FHIR; this row is the app-side handle.
    """

    source = models.CharField(max_length=16, choices=PatientSource.choices)
    fhir_id = models.CharField(max_length=128)
    fhir_base_url = models.URLField(max_length=255, blank=True)
    label = models.CharField(
        max_length=64, blank=True, help_text="Synthetic display label, e.g. 'Synthea 017'. Never a real name."
    )
    profile = models.JSONField(null=True, blank=True, help_text="Cached PatientProfile (Phase 3).")
    profile_cached_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["source", "fhir_id"], name="patient_unique_source_fhir_id"),
        ]
        ordering = ["source", "label", "id"]

    def __str__(self) -> str:
        return self.label or f"{self.source}:{self.fhir_id}"
