"""Run events: what an agent step emits when it runs away from Django (AgentCore Runtime, ADR 0008).

The runtime streams these to the Celery caller, which writes the rows Django used to write directly:
- `decision`: one ledger entry -> runs.Decision row, decision audit event, live SSE event
- `audit`: one audit record (LLM call or PHI read; ids and counts only) -> AuditEvent row
- `result`: the final graph state of this step (JSON-safe) -> services.persist
- `error`: the error class name only (messages may quote record text) -> run marked failed

Events cross a process boundary as JSON, so every payload must be JSON-safe.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from agents.ledger import LedgerEntry


class DecisionEvent(BaseModel):
    type: Literal["decision"] = "decision"
    entry: dict[str, Any]

    def ledger_entry(self) -> LedgerEntry:
        return LedgerEntry(**self.entry)


class AuditRecord(BaseModel):
    type: Literal["audit"] = "audit"
    kind: str
    run_id: str | None = None
    node: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class ResultEvent(BaseModel):
    type: Literal["result"] = "result"
    state: dict[str, Any]


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: str


RunEvent = Annotated[DecisionEvent | AuditRecord | ResultEvent | ErrorEvent, Field(discriminator="type")]
RUN_EVENT: TypeAdapter[RunEvent] = TypeAdapter(RunEvent)


def parse_event(raw: str | bytes | dict[str, Any]) -> RunEvent:
    if isinstance(raw, dict):
        return RUN_EVENT.validate_python(raw)
    return RUN_EVENT.validate_json(raw)


def serialize_result(out: dict[str, Any]) -> dict[str, Any]:
    """Graph output -> JSON-safe dict. Interrupt objects become {"value": ...} so paused_question still works."""
    state = {k: v for k, v in out.items() if k != "__interrupt__"}
    safe: dict[str, Any] = json.loads(json.dumps(state, default=str))
    interrupts = out.get("__interrupt__") or []
    if interrupts:
        safe["__interrupt__"] = [
            {"value": json.loads(json.dumps(getattr(i, "value", i), default=str))} for i in interrupts
        ]
    return safe


class EventChannel:
    """Collects a step's events in order. Safe to write from worker threads (the graph audits from to_thread).

    The runtime entrypoint drains it with `events()` while the step runs, then calls `close()`.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        self._owner = threading.get_ident()

    def put(self, event: RunEvent) -> None:
        if threading.get_ident() == self._owner:
            self._queue.put_nowait(event)
        else:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def close(self) -> None:
        self.put(None)  # type: ignore[arg-type]

    async def events(self) -> Any:
        while (event := await self._queue.get()) is not None:
            yield event

    # ---- sinks the graph uses ----

    def record(self, entry: LedgerEntry) -> None:
        """LedgerSink: one decision event per ledger entry."""
        self.put(DecisionEvent(entry=json.loads(json.dumps(asdict(entry), default=str))))

    def emit(self, kind: str, *, run_id: str | None = None, node: str = "", **payload: Any) -> None:
        """AuditEmit: one audit event (agents.audited)."""
        self.put(AuditRecord(kind=kind, run_id=run_id, node=node, payload=json.loads(json.dumps(payload, default=str))))
