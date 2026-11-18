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


def paused_question(out: dict[str, Any]) -> dict[str, Any] | None:
    """The pending ask_clinician payload if the run is waiting for a clinician, else None."""
    interrupts = out.get("__interrupt__") or []
    return dict(interrupts[0].value) if interrupts else None
