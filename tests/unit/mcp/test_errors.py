"""Gate 2: error handling (timeout, 429, malformed), PHI blocking before any request, AACT fallback."""

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from mcp_servers.clinicaltrials import server as ct
from mcp_servers.common.http import OutboundClient, UpstreamError, UpstreamErrorKind
from mcp_servers.pubmed import server as pubmed
from mcp_servers.terminology import server as term

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)


def client_with(handler: Handler, destination: str = "test", max_attempts: int = 3) -> tuple[OutboundClient, Recorder]:
    rec = Recorder(handler)
    http = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    return (
        OutboundClient(
            destination,
            "https://example.test",
            rate_per_min=0,
            http=http,
            max_attempts=max_attempts,
            backoff=0,
        ),
        rec,
    )


def raise_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


class TestOutboundClient:
    async def test_timeout_retries_then_raises(self) -> None:
        client, rec = client_with(raise_timeout)
        with pytest.raises(UpstreamError) as err:
            await client.get("/x")
        assert err.value.kind == UpstreamErrorKind.TIMEOUT
        assert len(rec.requests) == 3

    async def test_429_retries_then_succeeds(self) -> None:
        responses = iter([httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(200, json={"ok": 1})])
        client, rec = client_with(lambda r: next(responses))
        assert await client.get("/x") == {"ok": 1}
        assert len(rec.requests) == 2

    async def test_429_exhausted_is_rate_limited(self) -> None:
        client, _ = client_with(lambda r: httpx.Response(429))
        with pytest.raises(UpstreamError) as err:
            await client.get("/x")
        assert err.value.kind == UpstreamErrorKind.RATE_LIMITED

    async def test_malformed_json(self) -> None:
        client, rec = client_with(lambda r: httpx.Response(200, text="<html>oops</html>"))
        with pytest.raises(UpstreamError) as err:
            await client.get("/x")
        assert err.value.kind == UpstreamErrorKind.MALFORMED
        assert len(rec.requests) == 1  # no retry on a bad body

    async def test_400_not_retried(self) -> None:
        client, rec = client_with(lambda r: httpx.Response(400, json={"error": "bad"}))
        with pytest.raises(UpstreamError) as err:
            await client.get("/x")
        assert err.value.kind == UpstreamErrorKind.HTTP_ERROR
        assert len(rec.requests) == 1

    async def test_phi_blocked_before_any_request(self) -> None:
        client, rec = client_with(lambda r: httpx.Response(200, json={}))
        from mcp_servers.common.phi import PhiBlockedError

        with pytest.raises(PhiBlockedError):
            await client.get("/x", {"q": "asthma, SSN 123-45-6789"}, free_text=("q",))
        assert rec.requests == []

    async def test_structured_params_skip_free_text_check(self) -> None:
        client, rec = client_with(lambda r: httpx.Response(200, json={"ok": 1}))
        await client.get("/x", {"rxcui": "123456789", "q": "asthma"}, free_text=("q",))
        assert len(rec.requests) == 1

    async def test_cache_hit_skips_request(self) -> None:
        class MemCache:
            def __init__(self) -> None:
                self.store: dict[str, str] = {}

            async def get(self, key: str) -> str | None:
                return self.store.get(key)

            async def set(self, key: str, value: str, ex: int | None = None) -> None:
                self.store[key] = value

        rec = Recorder(lambda r: httpx.Response(200, json={"n": 1}))
        client = OutboundClient(
            "t",
            "https://example.test",
            rate_per_min=0,
            cache=MemCache(),
            http=httpx.AsyncClient(transport=httpx.MockTransport(rec)),
        )
        assert await client.get("/x", {"a": "1"}) == {"n": 1}
        assert await client.get("/x", {"a": "1"}) == {"n": 1}
        assert len(rec.requests) == 1


async def tool_error(server: FastMCP, tool: str, args: dict[str, Any]) -> str:
    async with Client(server) as c:
        with pytest.raises(ToolError) as err:
            await c.call_tool(tool, args)
    return str(err.value)


class TestToolErrors:
    async def test_ct_timeout_becomes_tool_error(self) -> None:
        client, _ = client_with(raise_timeout, destination=ct.DESTINATION)
        msg = await tool_error(ct.build_server(client=client), "get_eligibility", {"nct_id": "NCT04280705"})
        assert "timeout" in msg

    async def test_ct_malformed_search(self) -> None:
        client, _ = client_with(lambda r: httpx.Response(200, json={"unexpected": True}), ct.DESTINATION)
        msg = await tool_error(ct.build_server(client=client), "search_trials", {"conditions": "asthma"})
        assert "malformed" in msg

    async def test_ct_bad_nct_id_rejected_without_request(self) -> None:
        client, rec = client_with(lambda r: httpx.Response(200, json={}), ct.DESTINATION)
        msg = await tool_error(ct.build_server(client=client), "get_trial", {"nct_id": "Jane Doe"})
        assert "invalid nct_id" in msg
        assert rec.requests == []

    async def test_ct_phi_in_query_blocked(self) -> None:
        client, rec = client_with(lambda r: httpx.Response(200, json={}), ct.DESTINATION)
        msg = await tool_error(
            ct.build_server(client=client),
            "search_trials",
            {"conditions": "diabetes", "terms": "MRN: 00412345"},
        )
        assert "blocked" in msg
        assert "mrn" in msg
        assert "00412345" not in msg
        assert rec.requests == []

    async def test_ct_not_found(self) -> None:
        client, _ = client_with(lambda r: httpx.Response(404), ct.DESTINATION)
        msg = await tool_error(ct.build_server(client=client), "get_trial", {"nct_id": "NCT99999999"})
        assert "not found" in msg

    async def test_term_rate_limited(self) -> None:
        client, _ = client_with(lambda r: httpx.Response(429), "rxnav.nlm.nih.gov")
        server = term.build_server(rxnav=client, tables=client)
        assert "rate_limited" in await tool_error(server, "normalize_drug", {"name": "metformin"})

    async def test_term_malformed_tables(self) -> None:
        client, _ = client_with(lambda r: httpx.Response(200, json={"not": "a list"}), "clinicaltables")
        server = term.build_server(rxnav=client, tables=client)
        assert "malformed" in await tool_error(server, "lookup_icd10", {"text": "asthma"})

    async def test_pubmed_malformed_xml(self) -> None:
        client, _ = client_with(lambda r: httpx.Response(200, text="<PubmedArticleSet><broken"), pubmed.DESTINATION)
        msg = await tool_error(pubmed.build_server(client=client), "get_abstracts", {"pmids": ["32445440"]})
        assert "malformed" in msg

    async def test_pubmed_rejects_non_numeric_pmid(self) -> None:
        client, rec = client_with(lambda r: httpx.Response(200, text=""), pubmed.DESTINATION)
        msg = await tool_error(pubmed.build_server(client=client), "get_abstracts", {"pmids": ["abc"]})
        assert "invalid pmid" in msg
        assert rec.requests == []


class FakeAact:
    async def study(self, nct_id: str) -> dict[str, Any] | None:
        return {
            "protocolSection": {
                "identificationModule": {"nctId": nct_id, "briefTitle": "From AACT"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria: adults", "sex": "ALL"},
            }
        }

    async def search(self, conditions: str, status: list[str], limit: int) -> list[dict[str, Any]]:
        return [{"nct_id": "NCT00000001", "title": f"AACT {conditions}", "status": status[0], "phases": []}]


class TestAactFallback:
    async def test_eligibility_falls_back_on_503(self) -> None:
        client, _ = client_with(lambda r: httpx.Response(503), ct.DESTINATION)
        async with Client(ct.build_server(client=client, aact=FakeAact())) as c:
            out = (await c.call_tool("get_eligibility", {"nct_id": "NCT04280705"})).structured_content
        assert out["criteria_text"] == "Inclusion Criteria: adults"

    async def test_search_falls_back_on_timeout(self) -> None:
        client, _ = client_with(raise_timeout, ct.DESTINATION)
        async with Client(ct.build_server(client=client, aact=FakeAact())) as c:
            out = (await c.call_tool("search_trials", {"conditions": "asthma"})).structured_content
        assert out["source"] == "aact"
        assert out["studies"][0]["nct_id"] == "NCT00000001"

    async def test_no_fallback_for_phi_block(self) -> None:
        client, _ = client_with(lambda r: httpx.Response(503), ct.DESTINATION)
        msg = await tool_error(
            ct.build_server(client=client, aact=FakeAact()),
            "search_trials",
            {"conditions": "asthma, DOB 1950"},
        )
        assert "blocked" in msg
