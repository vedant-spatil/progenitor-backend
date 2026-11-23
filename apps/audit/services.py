"""Audit writers and the Django-backed audit wrappers (the wrappers themselves are in agents.audited).

Every PHI read, LLM call, supervisor decision, clinician answer, and approval becomes one append-only
AuditEvent. Payloads hold ids, types, and counts only, never record text, prompts, or answers.
"""

from __future__ import annotations

from typing import Any

from agents.audited import AuditedLLM as _AuditedLLM
from agents.audited import AuditedReader as _AuditedReader
from agents.fhir_reader import FhirReader
from agents.ledger import LedgerEntry
from agents.llm import StructuredLLM
from apps.audit.models import AuditEvent, AuditKind


def audit(
    kind: str, *, run_id: str | None = None, actor_id: int | None = None, node: str = "", **payload: Any
) -> AuditEvent:
    return AuditEvent.objects.create(kind=kind, run_id=run_id, actor_id=actor_id, node=node, payload=payload)


class AuditedLLM(_AuditedLLM):
    """Celery path: each LLM call becomes an AuditEvent row."""

    def __init__(self, inner: StructuredLLM, run_id: str) -> None:
        super().__init__(inner, run_id, audit)


class AuditedReader(_AuditedReader):
    """Celery path: each FHIR read or search becomes an AuditEvent row."""

    def __init__(self, inner: FhirReader, run_id: str) -> None:
        super().__init__(inner, run_id, audit)


def audit_decision(entry: LedgerEntry) -> None:
    audit(
        AuditKind.DECISION,
        run_id=entry.run_id,
        node="supervisor",
        step=entry.step,
        attempt=entry.attempt,
        action=entry.action,
        guard_outcome=entry.guard_outcome,
    )
