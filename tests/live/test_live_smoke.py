"""Live smoke: every outbound MCP tool against the real API. Nightly only, never in the gate.

uv run pytest tests/live -m live
"""

from typing import Any

import pytest
from fastmcp import Client, FastMCP

from mcp_servers.clinicaltrials.server import build_server as ct_server
from mcp_servers.pubmed.server import build_server as pubmed_server
from mcp_servers.terminology.server import build_server as term_server

pytestmark = pytest.mark.live

CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("ct", "search_trials", {"conditions": "type 2 diabetes", "page_size": 3}),
    ("ct", "get_trial", {"nct_id": "NCT04280705"}),
    ("ct", "get_eligibility", {"nct_id": "NCT04280705"}),
    ("term", "normalize_drug", {"name": "metformin"}),
    ("term", "drug_classes", {"rxcui": "274783"}),
    ("term", "lookup_loinc", {"name": "hba1c"}),
    ("term", "lookup_icd10", {"text": "asthma"}),
    ("pubmed", "search_pubmed", {"query": "remdesivir covid-19", "max_results": 2}),
    ("pubmed", "get_abstracts", {"pmids": ["32445440"]}),
]


@pytest.fixture
def servers() -> dict[str, FastMCP]:
    # Per test: each server's HTTP client binds to the event loop of its first use.
    return {"ct": ct_server(), "term": term_server(), "pubmed": pubmed_server()}


@pytest.mark.parametrize(("server", "tool", "args"), CASES, ids=[c[1] for c in CASES])
async def test_live_tool(servers: dict[str, FastMCP], server: str, tool: str, args: dict[str, Any]) -> None:
    async with Client(servers[server]) as c:
        result = await c.call_tool(tool, args)
    assert not result.is_error
    assert result.structured_content
