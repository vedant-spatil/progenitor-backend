"""Gate 0: a task goes through Redis to the running worker and its result comes back."""

import pytest

pytestmark = pytest.mark.integration


def test_celery_ping_round_trip() -> None:
    from progenitor import celery_app

    replies = celery_app.control.ping(timeout=5.0)
    assert replies, "no Celery worker answered the control ping"

    result = celery_app.send_task("progenitor.ping")
    assert result.get(timeout=20) == "pong"
