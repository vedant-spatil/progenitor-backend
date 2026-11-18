"""Run the supervisor graph with a Postgres checkpointer (runs survive restarts and can resume).

Each run uses its run id as the LangGraph thread id. A paused run (ask_clinician) is resumed
with `Command(resume=...)` on the same thread. A rerun after a crash continues from the last
checkpoint; to start over, use a new thread id (e.g. f"{run_id}-{attempt}").
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from agents.graph import Deps, build_graph, initial_state

RECURSION_LIMIT = 200  # above the 40-step budget plus fan-out, so a routing bug raises instead of looping


def checkpoint_url() -> str:
    url = os.environ.get("CHECKPOINT_DATABASE_URL") or os.environ.get("DATABASE_URL", "")
    if not url.startswith("postgres"):
        raise RuntimeError("checkpointer needs a Postgres CHECKPOINT_DATABASE_URL or DATABASE_URL")
    return url.replace("postgresql+psycopg://", "postgresql://")


@asynccontextmanager
async def postgres_checkpointer(url: str | None = None) -> AsyncIterator[AsyncPostgresSaver]:
    async with AsyncPostgresSaver.from_conn_string(url or checkpoint_url()) as saver:
        await saver.setup()  # idempotent: creates checkpoint tables on first use
        yield saver


def run_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


async def start_run(deps: Deps, saver: Any, run_id: str, patient_id: str, as_of: Any = None) -> dict[str, Any]:
    graph = build_graph(deps, checkpointer=saver)
    out: dict[str, Any] = await graph.ainvoke(initial_state(run_id, patient_id, as_of), run_config(run_id))
    return out


async def resume_run(deps: Deps, saver: Any, run_id: str, resume: Any = None) -> dict[str, Any]:
    """Resume a paused run with a clinician answer, or continue a crashed run (resume=None)."""
    graph = build_graph(deps, checkpointer=saver)
    payload: Any = Command(resume=resume) if resume is not None else None
    out: dict[str, Any] = await graph.ainvoke(payload, run_config(run_id))
    return out


async def advance(
    deps: Deps,
    saver: Any,
    run_id: str,
    patient_id: str,
    *,
    as_of: Any = None,
    answer: Any = None,
    approved_trial_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Move a run forward from whatever its checkpoint says. Safe to call again after a crash.

    - no checkpoint: start the run.
    - paused at ask_clinician and an answer given: resume with it.
    - unfinished (worker died mid-run, or a redelivered task): continue from the last checkpoint.
    - finished and approvals given: record the approved trials and hand control back to the supervisor,
      which decides whether to draft referrals (the approval guard still checks every draft).
    - otherwise: return the final state unchanged.
    """
    graph = build_graph(deps, checkpointer=saver)
    config = run_config(run_id)
    snap = await graph.aget_state(config)
    if not snap.values:
        out: dict[str, Any] = await graph.ainvoke(initial_state(run_id, patient_id, as_of), config)
        return out
    waiting = any(task.interrupts for task in snap.tasks)
    if waiting:
        if answer is None:
            return {**snap.values, "__interrupt__": [i for t in snap.tasks for i in t.interrupts]}
        out = await graph.ainvoke(Command(resume=answer), config)
        return out
    if snap.next:
        out = await graph.ainvoke(None, config)
        return out
    if approved_trial_ids and not snap.values.get("approved_trial_ids"):
        update = {"approved_trial_ids": list(approved_trial_ids), "status": "running"}
        await graph.aupdate_state(config, update, as_node="ranker")  # ranker -> supervisor edge
        out = await graph.ainvoke(None, config)
        return out
    return dict(snap.values)


def paused_question(out: dict[str, Any]) -> dict[str, Any] | None:
    """The pending ask_clinician payload if the run is waiting for a clinician, else None."""
    interrupts = out.get("__interrupt__") or []
    return dict(interrupts[0].value) if interrupts else None
