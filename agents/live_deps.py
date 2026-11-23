"""Live graph dependencies (MCP tools, configured LLM), shared by the Celery and AgentCore paths.

Django-free: the caller supplies where audit records and ledger entries go (Django rows on the Celery path,
the event stream on the AgentCore path).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from agents.audited import AuditedLLM, AuditedReader, AuditEmit
from agents.fhir_reader import McpFhirReader
from agents.graph import Deps
from agents.guards import Limits
from agents.ledger import LedgerSink
from agents.llm import get_llm
from agents.tools import connections_from_env, load_tools
from agents.tracing import tracer_from_env
from agents.workers.common import tool_caller
from mcp_servers.common.phi import KnownIdentifiers


@asynccontextmanager
async def live_deps(run_id: str, patient_fhir_id: str, emit: AuditEmit, ledger: LedgerSink) -> AsyncIterator[Deps]:
    """MCP tools (FHIR read-only, egress filter keyed to this patient) and the configured LLM."""
    conns = connections_from_env()
    fhir_tools = {t.name: t for t in await load_tools(connections={"fhir": conns["fhir"]})}
    reader = McpFhirReader(fhir_tools["fhir_read"], fhir_tools["fhir_search"])
    patient = await reader.read("Patient", patient_fhir_id)
    tools = {t.name: t for t in await load_tools(KnownIdentifiers.from_fhir_patient(patient), conns)}
    yield Deps(
        llm=AuditedLLM(get_llm(cache=None), run_id, emit),
        call_tool=tool_caller(tools),
        reader=AuditedReader(McpFhirReader(tools["fhir_read"], tools["fhir_search"]), run_id, emit),
        ledger=ledger,
        limits=Limits.from_env(),
        tracer=tracer_from_env(),
    )
