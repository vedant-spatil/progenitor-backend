"""Celery tasks: run the supervisor graph for one run."""

from __future__ import annotations

import asyncio
from typing import Any

from celery import shared_task
from celery.signals import worker_ready


@shared_task(name="runs.run_graph", ignore_result=True)
def run_graph(run_id: str, answer: Any = None, approved: list[str] | None = None) -> str:
    from apps.runs.services import execute

    return asyncio.run(execute(run_id, answer=answer, approved=approved))


@worker_ready.connect
def _recover(**_: Any) -> None:
    from apps.runs.services import recover_orphaned_runs

    recover_orphaned_runs()
