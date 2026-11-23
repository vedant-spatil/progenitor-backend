"""Apply a runtime step's events to Django (AgentCore path, ADR 0008).

The runtime cannot reach Postgres or Redis. It streams agents.events.RunEvent items back; this module writes
the same rows and live events the in-process Celery path writes, so both paths leave identical records.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable
from typing import Any

from asgiref.sync import sync_to_async

from agents.events import AuditRecord, DecisionEvent, ErrorEvent, ResultEvent, RunEvent
from apps.audit.services import audit
from apps.runs.ledger import RunLedger


class RuntimeStepError(RuntimeError):
    """The runtime reported an error. Holds the error class name only."""


def apply_event(event: RunEvent, ledger: RunLedger | None = None) -> dict[str, Any] | None:
    """Write one event. Returns the final state for a result event, else None. Raises on an error event."""
    if isinstance(event, DecisionEvent):
        (ledger or RunLedger()).record(event.ledger_entry())
    elif isinstance(event, AuditRecord):
        audit(event.kind, run_id=event.run_id, node=event.node, **event.payload)
    elif isinstance(event, ResultEvent):
        return event.state
    elif isinstance(event, ErrorEvent):
        raise RuntimeStepError(event.error)
    return None


def apply_events(events: Iterable[RunEvent]) -> dict[str, Any]:
    """Write every event in order; returns the step's final state. A stream without a result is an error."""
    ledger = RunLedger()
    result: dict[str, Any] | None = None
    for event in events:
        out = apply_event(event, ledger)
        if out is not None:
            result = out
    if result is None:
        raise RuntimeStepError("NoResult")
    return result


async def apply_events_async(events: AsyncIterable[RunEvent]) -> dict[str, Any]:
    ledger = RunLedger()
    result: dict[str, Any] | None = None
    async for event in events:
        out = await sync_to_async(apply_event)(event, ledger)
        if out is not None:
            result = out
    if result is None:
        raise RuntimeStepError("NoResult")
    return result
