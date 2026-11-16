"""Gate 2: tool loading rules: FHIR read-only, PHI interceptor on outbound servers only."""

from typing import Any

import pytest
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

from agents.tools import EGRESS_SERVERS, EgressInterceptor, connections_from_env, is_allowed
from mcp_servers.common.phi import KnownIdentifiers, PhiBlockedError

KNOWN = KnownIdentifiers.from_fhir_patient(
    {"id": "pat-9", "name": [{"family": "Okafor", "given": ["Adaeze"]}], "birthDate": "1961-07-19"}
)


@pytest.mark.parametrize(
    ("server", "tool", "allowed"),
    [
        ("fhir", "search", True),
        ("fhir", "read", True),
        ("fhir", "get_capabilities", True),
        ("fhir", "create", False),
        ("fhir", "update", False),
        ("fhir", "delete", False),
        ("fhir", "get_user", False),
        ("clinicaltrials", "search_trials", True),
    ],
)
def test_fhir_is_read_only(server: str, tool: str, allowed: bool) -> None:
    assert is_allowed(server, tool) is allowed


def test_connections_cover_all_servers() -> None:
    conns = connections_from_env()
    assert set(conns) == {"clinicaltrials", "terminology", "pubmed", "fhir"}
    assert all(c["transport"] == "streamable_http" for c in conns.values())


class Handler:
    def __init__(self) -> None:
        self.calls: list[MCPToolCallRequest] = []

    async def __call__(self, request: MCPToolCallRequest) -> Any:
        self.calls.append(request)
        return "ok"


@pytest.mark.parametrize("server", sorted(EGRESS_SERVERS))
async def test_interceptor_blocks_patient_name_on_outbound(server: str) -> None:
    handler = Handler()
    req = MCPToolCallRequest(name="search", args={"query": "breast cancer okafor"}, server_name=server)
    with pytest.raises(PhiBlockedError):
        await EgressInterceptor(KNOWN)(req, handler)
    assert handler.calls == []


async def test_interceptor_passes_clean_outbound() -> None:
    handler = Handler()
    req = MCPToolCallRequest(
        name="search_trials", args={"conditions": "breast cancer"}, server_name="clinicaltrials"
    )
    assert await EgressInterceptor(KNOWN)(req, handler) == "ok"


async def test_interceptor_does_not_filter_internal_fhir() -> None:
    """FHIR is our own server; reads legitimately use patient ids."""
    handler = Handler()
    req = MCPToolCallRequest(name="read", args={"type": "Patient", "id": "pat-9"}, server_name="fhir")
    assert await EgressInterceptor(KNOWN)(req, handler) == "ok"
