"""Gate 2: MCP servers running in compose, loaded into LangChain via langchain-mcp-adapters."""

import json
import os
from typing import Any

import httpx
import pytest

from agents.tools import load_tools
from mcp_servers.common.phi import KnownIdentifiers, PhiBlockedError

pytestmark = pytest.mark.integration

EXPECTED_TOOLS = {
    "clinicaltrials_search_trials",
    "clinicaltrials_get_trial",
    "clinicaltrials_get_eligibility",
    "terminology_normalize_drug",
    "terminology_drug_classes",
    "terminology_lookup_loinc",
    "terminology_lookup_icd10",
    "pubmed_search_pubmed",
    "pubmed_get_abstracts",
    "fhir_get_capabilities",
    "fhir_search",
    "fhir_read",
}
HBA1C = "http://loinc.org|4548-4"


def fhir_base() -> str:
    return os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir").rstrip("/")


@pytest.fixture(scope="module")
def seeded_patient() -> dict[str, Any]:
    """A Synthea patient with at least one HbA1c observation, read straight from HAPI."""
    with httpx.Client(
        base_url=fhir_base(), timeout=60, headers={"Accept": "application/fhir+json"}
    ) as client:
        obs = client.get(
            "/Observation",
            params={"code": HBA1C, "_tag": "https://progenitor.dev/fhir/source|synthea", "_count": "1"},
        ).json()
        assert obs.get("entry"), "no seeded HbA1c observations; run `make data`"
        pid = obs["entry"][0]["resource"]["subject"]["reference"].split("/")[1]
        patient: dict[str, Any] = client.get(f"/Patient/{pid}").json()
    return patient


@pytest.fixture
async def tools() -> dict[str, Any]:
    return {t.name: t for t in await load_tools()}


async def test_all_servers_load_with_fhir_read_only(tools: dict[str, Any]) -> None:
    assert set(tools) == EXPECTED_TOOLS
    assert not {"fhir_create", "fhir_update", "fhir_delete", "fhir_get_user"} & set(tools)


def _bundle(result: Any) -> dict[str, Any]:
    text = result[0]["text"] if isinstance(result, list) else result
    bundle: dict[str, Any] = json.loads(text)
    return bundle


async def test_fhir_mcp_returns_observation_by_loinc(
    tools: dict[str, Any], seeded_patient: dict[str, Any]
) -> None:
    pid = seeded_patient["id"]
    out = await tools["fhir_search"].ainvoke(
        {"type": "Observation", "searchParam": {"patient": pid, "code": HBA1C, "_count": "50"}}
    )
    bundle = _bundle(out)
    assert bundle["resourceType"] == "Bundle"
    entries = [e["resource"] for e in bundle.get("entry", [])]
    assert entries
    for obs in entries:
        assert obs["resourceType"] == "Observation"
        assert obs["subject"]["reference"] == f"Patient/{pid}"
        assert any(c["code"] == "4548-4" for c in obs["code"]["coding"])


async def test_fhir_mcp_read_patient(tools: dict[str, Any], seeded_patient: dict[str, Any]) -> None:
    out = await tools["fhir_read"].ainvoke({"type": "Patient", "id": seeded_patient["id"]})
    assert _bundle(out)["id"] == seeded_patient["id"]


async def test_egress_interceptor_blocks_patient_identifiers(seeded_patient: dict[str, Any]) -> None:
    known = KnownIdentifiers.from_fhir_patient(seeded_patient)
    tools = {t.name: t for t in await load_tools(known=known)}
    family = seeded_patient["name"][0]["family"]
    with pytest.raises(PhiBlockedError):
        await tools["clinicaltrials_search_trials"].ainvoke({"conditions": f"diabetes {family}"})
    with pytest.raises(PhiBlockedError):
        await tools["pubmed_search_pubmed"].ainvoke({"query": f"metformin {seeded_patient['birthDate']}"})


async def test_outbound_server_blocks_generic_phi_itself() -> None:
    """Defence in depth: the server rejects identifiers even when called without the interceptor."""
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    url = os.environ.get("TERM_MCP_URL", "http://localhost:8102/mcp")
    async with Client(url) as client:
        with pytest.raises(ToolError, match="blocked"):
            await client.call_tool("lookup_icd10", {"text": "asthma SSN 123-45-6789"})
