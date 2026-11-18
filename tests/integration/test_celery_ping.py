"""Gate 0: a task goes through Redis to the running worker and its result comes back."""

import pytest

pytestmark = pytest.mark.integration


def test_celery_ping_round_trip() -> None:
    from progenitor import celery_app

    replies = celery_app.control.ping(timeout=5.0)
    assert replies, "no Celery worker answered the control ping"

    result = celery_app.send_task("progenitor.ping")
    assert result.get(timeout=20) == "pong"


def test_worker_registers_run_task() -> None:
    """Gate 5: the compose worker serves the run task (apps.runs.tasks via autodiscovery)."""
    from progenitor import celery_app

    registered = celery_app.control.inspect(timeout=5.0).registered() or {}
    assert any("runs.run_graph" in tasks for tasks in registered.values()), registered
