"""Load MCP tools into LangGraph.

- FHIR (wso2 fhir-mcp-server) is read-only here: only get_capabilities, search,
  and read are exposed; create/update/delete never reach an agent.
- Calls to servers that leave the system (clinicaltrials, terminology, pubmed)
  pass an egress interceptor that checks every string argument against the
  patient's known identifiers before the call is sent. The servers also run the
  generic PHI patterns themselves (defence in depth).
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_mcp_adapters.sessions import Connection

from mcp_servers.common.phi import KnownIdentifiers, check_outbound

FHIR_SERVER = "fhir"
FHIR_READ_ONLY_TOOLS = frozenset({"get_capabilities", "search", "read"})
EGRESS_SERVERS = frozenset({"clinicaltrials", "terminology", "pubmed"})

Handler = Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]


def connections_from_env() -> dict[str, Connection]:
    """MCP servers from env. With AGENT_SERVICE_TOKEN set (AgentCore path, ADR 0008), every request carries it,
    because the servers are then reached through CloudFront `/internal/mcp/*`."""

    def url(var: str, default: str) -> str:
        return os.environ.get(var, default)

    conns: dict[str, Connection] = {
        "clinicaltrials": {
            "transport": "streamable_http",
            "url": url("CT_MCP_URL", "http://localhost:8101/mcp"),
        },
        "terminology": {
            "transport": "streamable_http",
            "url": url("TERM_MCP_URL", "http://localhost:8102/mcp"),
        },
        "pubmed": {"transport": "streamable_http", "url": url("PUBMED_MCP_URL", "http://localhost:8103/mcp")},
        FHIR_SERVER: {
            "transport": "streamable_http",
            "url": url("FHIR_MCP_URL", "http://localhost:8104/mcp"),
        },
    }
    token = os.environ.get("AGENT_SERVICE_TOKEN", "")
    if token:
        for conn in conns.values():
            conn["headers"] = {"Authorization": f"Bearer {token}"}  # type: ignore[typeddict-unknown-key]
    return conns


@dataclass
class EgressInterceptor:
    """Blocks outbound tool calls whose arguments contain this patient's identifiers."""

    known: KnownIdentifiers | None = None

    async def __call__(self, request: MCPToolCallRequest, handler: Handler) -> MCPToolCallResult:
        if request.server_name in EGRESS_SERVERS:
            check_outbound(request.args, destination=request.server_name, known=self.known)
        return await handler(request)


def is_allowed(server: str, tool_name: str) -> bool:
    return server != FHIR_SERVER or tool_name in FHIR_READ_ONLY_TOOLS


async def load_tools(
    known: KnownIdentifiers | None = None, connections: dict[str, Connection] | None = None
) -> list[BaseTool]:
    """All MCP tools for one run, named '<server>_<tool>', FHIR restricted to reads."""
    conns = connections or connections_from_env()
    client = MultiServerMCPClient(conns, tool_interceptors=[EgressInterceptor(known)], tool_name_prefix=True)
    tools: list[BaseTool] = []
    for server in conns:
        tools += [
            t
            for t in await client.get_tools(server_name=server)
            if is_allowed(server, t.name.removeprefix(f"{server}_"))
        ]
    return tools
