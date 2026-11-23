from typing import Any, NoReturn

from django.conf import settings
from django.db import models


class AppendOnlyError(Exception):
    """Raised when code tries to change or remove an audit event."""


def _blocked(op: str) -> NoReturn:
    raise AppendOnlyError(f"audit_auditevent is append-only: {op} is not allowed")


class AuditEventQuerySet(models.QuerySet["AuditEvent"]):
    # Postgres also enforces this with a trigger (migration 0002); these guards
    # give a clear error earlier and cover non-Postgres test databases.
    def update(self, **kwargs: Any) -> int:
        _blocked("UPDATE")

    def delete(self) -> tuple[int, dict[str, int]]:
        _blocked("DELETE")


class AuditKind(models.TextChoices):
    PHI_READ = "phi_read", "Patient record read"
    LLM_CALL = "llm_call", "LLM call"
    DECISION = "decision", "Supervisor decision"
    APPROVAL = "approval", "Clinician approval or rejection"
    ANSWER = "answer", "Clinician answer"
    EGRESS_BLOCKED = "egress_blocked", "Outbound call blocked by PHI filter"
    DATA_LOAD = "data_load", "Data loader run"
    RUN_CREATED = "run_created", "Run started by a user"
    REFERRAL_EDITED = "referral_edited", "Referral draft edited or discarded by a clinician"


class AuditEvent(models.Model):
    """Append-only audit log. `payload` holds ids and counts only, never record text."""

    run = models.ForeignKey("runs.Run", null=True, blank=True, on_delete=models.PROTECT, related_name="audit_events")
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.PROTECT, related_name="audit_events"
    )
    node = models.CharField(max_length=64, blank=True)
    kind = models.CharField(max_length=32, choices=AuditKind.choices)
    payload = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    objects = AuditEventQuerySet.as_manager()

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self) -> str:
        return f"{self.created_at:%Y-%m-%d %H:%M:%S} {self.kind}"

    def save(self, *args: Any, **kwargs: Any) -> None:
        if self.pk is not None and not self._state.adding:
            _blocked("UPDATE")
        super().save(*args, **kwargs)

    def delete(self, *args: Any, **kwargs: Any) -> tuple[int, dict[str, int]]:
        _blocked("DELETE")
