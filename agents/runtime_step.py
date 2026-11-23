"""One step of a run, streamed as events (AgentCore Runtime path, ADR 0008).

A step is what one Celery task does today: start the run, resume it with a clinician answer, continue after a
crash, or hand approvals back to the supervisor (agents.runner.advance). Here the graph writes nothing itself:
ledger entries and audit records go on an EventChannel, and the step ends with a `result` or an `error` event.
Kept free of AgentCore and Django imports so it runs in unit tests with the scripted LLM.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from pydantic import BaseModel, Field

from agents.events import ErrorEvent, EventChannel, ResultEvent, serialize_result
from agents.graph import Deps
from agents.runner import advance

log = logging.getLogger(__name__)


class StepRequest(BaseModel):
    """Invocation payload from the Celery caller. Validated before anything runs."""

    run_id: str = Field(min_length=1, max_length=64)
    patient_fhir_id: str = Field(min_length=1, max_length=128)
    as_of: str | None = None
    answer: Any = None
    approved: list[str] | None = None


# Builds the graph's dependencies for one step, wired to the step's channel.
DepsFactory = Callable[[StepRequest, EventChannel], AbstractAsyncContextManager[Deps]]


async def run_step(
    req: StepRequest, make_deps: DepsFactory, saver: Any, config_extra: dict[str, Any] | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Yield the step's events as JSON-safe dicts while it runs; the last one is `result` or `error`."""
    ch = EventChannel()

    async def work() -> None:
        try:
            async with make_deps(req, ch) as deps:
                out = await advance(
                    deps,
                    saver,
                    req.run_id,
                    req.patient_fhir_id,
                    as_of=req.as_of,
                    answer=req.answer,
                    approved_trial_ids=req.approved,
                    config_extra=config_extra,
                )
            ch.put(ResultEvent(state=serialize_result(out)))
        except Exception as exc:  # noqa: BLE001 - any failure must end the stream with an error event
            # Class name only: messages from tools or the LLM may quote record text.
            log.error("runtime_step_failed run=%s error=%s", req.run_id, type(exc).__name__)
            ch.put(ErrorEvent(error=type(exc).__name__[:255]))
        finally:
            ch.close()

    task = asyncio.create_task(work())
    try:
        async for event in ch.events():
            yield event.model_dump(mode="json")
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
