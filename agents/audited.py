"""Audit wrappers for the LLM and the FHIR reader, independent of Django.

Each call produces one audit record through `emit`. On the Celery path `emit` writes an AuditEvent row
(apps.audit.services); on the AgentCore path it puts an event on the run's stream (agents.events), and the
Celery caller writes the row. Payloads hold ids, types, and counts only, never record text, prompts, or answers.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from agents.fhir_reader import FhirReader, Resource
from agents.llm import StructuredLLM, Usage

T = TypeVar("T", bound=BaseModel)

# Values of apps.audit.models.AuditKind (agents/ must not import Django apps).
LLM_CALL = "llm_call"
PHI_READ = "phi_read"


class AuditEmit(Protocol):
    def __call__(self, kind: str, *, run_id: str | None = ..., node: str = ..., **payload: Any) -> Any: ...


class AuditedLLM:
    """StructuredLLM that emits one llm_call record per call (task, model, token counts, outcome)."""

    def __init__(self, inner: StructuredLLM, run_id: str, emit: AuditEmit) -> None:
        self.inner = inner
        self.run_id = run_id
        self.emit = emit
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
            # Called from worker threads (asyncio.to_thread): emit must be thread-safe.
            self.emit(
                LLM_CALL,
                run_id=self.run_id,
                node=task,
                model=self.model,
                status=status,
                tokens_in=self.usage.tokens_in - before_in,
                tokens_out=self.usage.tokens_out - before_out,
            )


class AuditedReader:
    """FhirReader that emits one phi_read record per FHIR read or search (resource type, ids, counts)."""

    def __init__(self, inner: FhirReader, run_id: str, emit: AuditEmit) -> None:
        self.inner = inner
        self.run_id = run_id
        self.emit = emit

    async def read(self, resource_type: str, resource_id: str) -> Resource:
        body = await self.inner.read(resource_type, resource_id)
        await asyncio.to_thread(
            self.emit, PHI_READ, run_id=self.run_id, node="fhir_read", resource=f"{resource_type}/{resource_id}"
        )
        return body

    async def search(self, resource_type: str, params: dict[str, str]) -> tuple[list[Resource], list[Resource]]:
        matches, included = await self.inner.search(resource_type, params)
        await asyncio.to_thread(
            self.emit,
            PHI_READ,
            run_id=self.run_id,
            node="fhir_search",
            resource_type=resource_type,
            # Search keys only: values can be patient ids or codes; the patient id is on the run already.
            params=sorted(params),
            matches=len(matches),
            included=len(included),
        )
        return matches, included
