"""Gate 0: every compose service answers its health check."""

import json
import os
import socket
import urllib.request
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.integration


def _get(url: str, timeout: float = 10.0, accept: str = "application/json, text/html") -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"Accept": accept})  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
        return resp.status, resp.read()


def test_postgres_accepts_connections() -> None:
    import psycopg

    dsn = os.environ.get("GATE_DATABASE_URL", "postgresql://progenitor:progenitor@localhost:5432/progenitor")
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        row = conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None, "pgvector extension is not installed"


def test_redis_ping() -> None:
    import redis

    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    assert redis.Redis.from_url(url, socket_connect_timeout=5).ping()


def test_hapi_fhir_metadata(stack_urls: dict[str, str]) -> None:
    # HAPI serves an HTML browser view when Accept includes text/html.
    status, body = _get(f"{stack_urls['hapi']}/metadata", timeout=30, accept="application/fhir+json")
    assert status == 200
    capability = json.loads(body)
    assert capability["resourceType"] == "CapabilityStatement"
    assert capability["fhirVersion"].startswith("4.")


@pytest.mark.skipif(
    os.environ.get("TRACING_MODE", "langfuse") != "langfuse",
    reason="Langfuse runs only with TRACING_MODE=langfuse (the `tracing` compose profile)",
)
def test_langfuse_ui(stack_urls: dict[str, str]) -> None:
    status, _ = _get(f"{stack_urls['langfuse']}/api/public/health")
    assert status == 200
    status, _ = _get(stack_urls["langfuse"])
    assert status == 200


def test_backend_health(stack_urls: dict[str, str]) -> None:
    status, body = _get(f"{stack_urls['backend']}/api/health")
    assert status == 200
    assert json.loads(body) == {"status": "ok", "checks": {"database": True, "redis": True}}


def test_frontend_health_page(stack_urls: dict[str, str]) -> None:
    status, body = _get(f"{stack_urls['frontend']}/health")
    assert status == 200
    assert b"Progenitor" in body


def test_frontend_reachable_port(stack_urls: dict[str, str]) -> None:
    parsed = urlparse(stack_urls["frontend"])
    with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=5):
        pass
