"""Run events over Redis pub/sub: the Celery worker publishes, the SSE endpoint streams.

Events carry ids, statuses, and counts. They never carry record text, prompts, or rationale text
beyond what GET /api/runs/{id} already returns to the same roles.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import redis
import redis.asyncio as aioredis
from django.conf import settings

log = logging.getLogger(__name__)

TERMINAL = frozenset({"awaiting_review", "waiting_clinician", "completed", "failed", "cancelled"})


def channel(run_id: str) -> str:
    return f"progenitor:run:{run_id}:events"


def publish(run_id: str, event: str, data: dict[str, Any]) -> None:
    """Best effort: a Redis outage must not fail the run (the DB stays the source of truth)."""
    try:
        client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        client.publish(channel(run_id), json.dumps({"event": event, "data": data}, default=str))
    except redis.RedisError as exc:
        log.warning("run_event_publish_failed event=%s error=%s", event, type(exc).__name__)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream(
    run_id: str, snapshot: dict[str, Any], *, heartbeat_s: float = 15.0, url: str | None = None
) -> AsyncIterator[str]:
    """SSE body: a snapshot first, then live events until the run stops or pauses."""
    yield sse("snapshot", snapshot)
    if snapshot.get("status") in TERMINAL:
        return
    client = aioredis.Redis.from_url(url or settings.REDIS_URL)
    pubsub = client.pubsub()
    await pubsub.subscribe(channel(run_id))
    try:
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=heartbeat_s)
            if msg is None:
                yield ": ping\n\n"
                continue
            body = json.loads(msg["data"])
            yield sse(body["event"], body["data"])
            if body["event"] == "status" and body["data"].get("status") in TERMINAL:
                return
    finally:
        await pubsub.unsubscribe(channel(run_id))
        await pubsub.aclose()  # type: ignore[no-untyped-call]
        await client.aclose()
