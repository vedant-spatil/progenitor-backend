"""Audit writers and the wrappers that make agent runs audited.

Every PHI read, LLM call, supervisor decision, clinician answer, and approval becomes one append-only
AuditEvent. Payloads hold ids, types, and counts only, never record text, prompts, or answers.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

from pydantic import BaseModel

from agents.fhir_reader import FhirReader, Resource
from agents.ledger import LedgerEntry
from agents.llm import StructuredLLM, Usage
from apps.audit.models import AuditEvent, AuditKind

T = TypeVar("T", bound=BaseModel)


def audit(
    kind: str, *, run_id: str | None = None, actor_id: int | None = None, node: str = "", **payload: Any
) -> AuditEvent:
    return AuditEvent.objects.create(kind=kind, run_id=run_id, actor_id=actor_id, node=node, payload=payload)


class AuditedLLM:
    """StructuredLLM that writes one llm_call event per call (task, model, token counts, outcome)."""

    def __init__(self, inner: StructuredLLM, run_id: str) -> None:
        self.inner = inner
        self.run_id = run_id
        self.model: str = inner.model
        self.usage: Usage = inner.usage  # same object: token counts stay shared with the inner client

    def complete(self, *, task: str, system: str, user: str, schema: type[T]) -> T:
        before_in, before_out = self.usage.tokens_in, self.usage.tokens_out
        status = "error"
        try:
            result = self.inner.complete(task=task, system=system, user=user, schema=schema)
            status = "ok"
            return result
        finally:
            # Called from worker threads (asyncio.to_thread), so the sync ORM is safe here.
            audit(
                AuditKind.LLM_CALL,
                run_id=self.run_id,
                node=task,
                model=self.model,
                status=status,
                tokens_in=self.usage.tokens_in - before_in,
                tokens_out=self.usage.tokens_out - before_out,
            )


class AuditedReader:
    """FhirReader that writes one phi_read event per FHIR read or search (resource type, ids, counts)."""

    def __init__(self, inner: FhirReader, run_id: str) -> None:
        self.inner = inner
        self.run_id = run_id

    async def read(self, resource_type: str, resource_id: str) -> Resource:
        body = await self.inner.read(resource_type, resource_id)
        await asyncio.to_thread(
            audit,
            AuditKind.PHI_READ,
            run_id=self.run_id,
            node="fhir_read",
            resource=f"{resource_type}/{resource_id}",
        )
        return body

    async def search(self, resource_type: str, params: dict[str, str]) -> tuple[list[Resource], list[Resource]]:
        matches, included = await self.inner.search(resource_type, params)
        await asyncio.to_thread(
            audit,
            AuditKind.PHI_READ,
            run_id=self.run_id,
            node="fhir_search",
            resource_type=resource_type,
            # Search keys only: values can be patient ids or codes; the patient id is on the run already.
            params=sorted(params),
            matches=len(matches),
            included=len(included),
        )
        return matches, included


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
