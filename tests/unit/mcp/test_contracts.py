"""Gate 2: contract tests per MCP tool, replayed from VCR cassettes (no live calls in CI).

Re-record against the live APIs with:
    uv run pytest tests/unit/mcp/test_contracts.py --record-mode=rewrite
"""

from typing import Any

import pytest
from fastmcp import Client, FastMCP

from mcp_servers.clinicaltrials import server as ct
from mcp_servers.common.http import OutboundClient
from mcp_servers.pubmed import server as pubmed
from mcp_servers.terminology import server as term

pytestmark = pytest.mark.vcr


def _client(destination: str, base_url: str) -> OutboundClient:
    return OutboundClient(destination, base_url, rate_per_min=0, cache=None, backoff=0)


@pytest.fixture
def ct_server() -> FastMCP:
    return ct.build_server(client=_client(ct.DESTINATION, ct.BASE_URL))


@pytest.fixture
def term_server() -> FastMCP:
    return term.build_server(
        rxnav=_client("rxnav.nlm.nih.gov", term.RXNAV_URL),
        tables=_client("clinicaltables.nlm.nih.gov", term.TABLES_URL),
    )


@pytest.fixture
def pubmed_server() -> FastMCP:
    return pubmed.build_server(client=_client(pubmed.DESTINATION, pubmed.EUTILS_URL))


async def call(server: FastMCP, tool: str, args: dict[str, Any]) -> Any:
    async with Client(server) as c:
        return (await c.call_tool(tool, args)).structured_content


async def test_tools_listed_and_read_only(
    ct_server: FastMCP, term_server: FastMCP, pubmed_server: FastMCP
) -> None:
    expected = {
        "clinicaltrials": {"search_trials", "get_trial", "get_eligibility"},
        "terminology": {"normalize_drug", "drug_classes", "lookup_loinc", "lookup_icd10"},
        "pubmed": {"search_pubmed", "get_abstracts"},
    }
    for server in (ct_server, term_server, pubmed_server):
        async with Client(server) as c:
            tools = await c.list_tools()
        assert {t.name for t in tools} == expected[server.name]
        assert all(t.annotations and t.annotations.readOnlyHint for t in tools)


class TestClinicalTrials:
    async def test_search_trials(self, ct_server: FastMCP) -> None:
        out = await call(
            ct_server,
            "search_trials",
            {"conditions": "type 2 diabetes", "phases": ["PHASE3"], "page_size": 5},
        )
        assert out["source"] == "ctgov"
        assert 1 <= len(out["studies"]) <= 5
        assert out["total"] >= len(out["studies"])
        for s in out["studies"]:
            assert s["nct_id"].startswith("NCT")
            assert s["status"] == "RECRUITING"
            assert "PHASE3" in s["phases"]
        assert out["next_page_token"]

    async def test_search_next_page(self, ct_server: FastMCP) -> None:
        first = await call(ct_server, "search_trials", {"conditions": "asthma", "page_size": 3})
        second = await call(
            ct_server,
            "search_trials",
            {"conditions": "asthma", "page_size": 3, "page_token": first["next_page_token"]},
        )
        assert {s["nct_id"] for s in first["studies"]}.isdisjoint({s["nct_id"] for s in second["studies"]})

    async def test_get_trial_has_locations(self, ct_server: FastMCP) -> None:
        out = await call(ct_server, "get_trial", {"nct_id": "NCT04280705"})
        assert out["nct_id"] == "NCT04280705"
        assert out["status"] == "COMPLETED"
        assert out["locations"]
        assert {"facility", "city", "country"} <= set(out["locations"][0])

    async def test_get_eligibility(self, ct_server: FastMCP) -> None:
        out = await call(ct_server, "get_eligibility", {"nct_id": "nct04280705"})
        assert out["criteria_text"].startswith("Inclusion Criteria")
        assert out["minimum_age"] == "18 Years"
        assert out["sex"] == "ALL"


class TestTerminology:
    async def test_normalize_drug(self, term_server: FastMCP) -> None:
        out = await call(term_server, "normalize_drug", {"name": "metformin"})
        assert out["candidates"][0]["rxcui"] == "6809"
        assert out["candidates"][0]["name"].lower() == "metformin"

    async def test_drug_classes_insulin(self, term_server: FastMCP) -> None:
        # 274783 = insulin glargine; ATC A10A = insulins and analogues ("no insulin" criteria).
        # The bare ingredient "insulin" (5856) has no ATC class.
        out = await call(term_server, "drug_classes", {"rxcui": "274783"})
        assert any(c["class_id"].startswith("A10A") for c in out["classes"])

    async def test_lookup_loinc(self, term_server: FastMCP) -> None:
        out = await call(term_server, "lookup_loinc", {"name": "HbA1c"})
        assert out["preferred"]["code"] == "4548-4"
        assert out["matches"]

    async def test_lookup_icd10(self, term_server: FastMCP) -> None:
        out = await call(term_server, "lookup_icd10", {"text": "type 2 diabetes"})
        assert any(m["code"].startswith("E11") for m in out["matches"])


class TestPubmed:
    async def test_search_pubmed(self, pubmed_server: FastMCP) -> None:
        out = await call(
            pubmed_server, "search_pubmed", {"query": "metformin type 2 diabetes", "max_results": 3}
        )
        assert out["count"] > 1000
        assert len(out["articles"]) == 3
        assert all(a["pmid"].isdigit() and a["title"] for a in out["articles"])

    async def test_get_abstracts(self, pubmed_server: FastMCP) -> None:
        # PMID 32445440: ACTT-1 remdesivir final report (NEJM 2020)
        out = await call(pubmed_server, "get_abstracts", {"pmids": ["32445440"]})
        art = out["articles"][0]
        assert art["pmid"] == "32445440"
        assert "Remdesivir" in art["title"]
        assert art["abstract"]
        assert "BACKGROUND" in art["abstract"]
