import os

import pytest


def _url(name: str, default: str) -> str:
    return os.environ.get(name, default).rstrip("/")


@pytest.fixture(scope="session")
def stack_urls() -> dict[str, str]:
    """Host-side URLs of the compose services (see docker-compose.yml port mappings)."""
    return {
        "backend": _url(
            "GATE_BACKEND_URL", f"http://localhost:{os.environ.get('BACKEND_HOST_PORT', '8000')}"
        ),
        "frontend": _url(
            "GATE_FRONTEND_URL", f"http://localhost:{os.environ.get('FRONTEND_HOST_PORT', '3000')}"
        ),
        "hapi": _url("GATE_HAPI_URL", f"http://localhost:{os.environ.get('HAPI_HOST_PORT', '8080')}/fhir"),
        "langfuse": _url("GATE_LANGFUSE_URL", "http://localhost:3001"),
    }
